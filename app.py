# app.py
# Pocket Option Signals - single-file deploy (Flask 3.1+ safe)
# Flask + Deriv streaming + quotas/cooldowns + tallies + Telegram + APScheduler + Postgres/SQLite
# (c) 2025 Chris — built by ChatGPT

import os
import sys
import json
import time
import math
import ssl
import signal
import queue
import random
import logging
import threading
from datetime import datetime, timedelta, timezone, date
from collections import deque, defaultdict

import requests
from flask import Flask, jsonify, Response, render_template_string, request as flask_request

# Optional deps present on your Render app (per your logs / prior setup)
try:
    import psycopg2
    import psycopg2.extras
except Exception:  # SQLite fallback only
    psycopg2 = None

import sqlite3

try:
    import websockets
    import asyncio
except Exception:
    websockets = None
    asyncio = None

try:
    from apscheduler.schedulers.background import BackgroundScheduler
    from apscheduler.triggers.cron import CronTrigger
except Exception as e:
    raise RuntimeError("APScheduler is required (pip install apscheduler)") from e


# ===========
# CONFIG
# ===========
APP_NAME = "Pocket Option Signals"
DEFAULT_SYMBOLS = os.getenv("SYMBOLS", "frxAUDCAD,frxEURUSD,frxGBPUSD").replace(" ", "").split(",")
DERIV_APP_ID = os.getenv("DERIV_APP_ID", "99185")  # from your logs
DERIV_API_TOKEN = os.getenv("DERIV_API_TOKEN")  # optional (public market data usually works without)
CANDLE_GRANULARITY = int(os.getenv("CANDLE_GRANULARITY", "60"))  # 60s candles
CANDLE_HISTORY = int(os.getenv("CANDLE_HISTORY", "120"))  # how many candles keep per symbol (in-memory)

# Strategy
SMA_FAST = int(os.getenv("SMA_FAST", "9"))
SMA_SLOW = int(os.getenv("SMA_SLOW", "21"))
PULLBACK_MIN = float(os.getenv("PULLBACK_MIN", "0.00010"))  # filter tiny chop (pair-dependent)
COOLDOWN_SECONDS = int(os.getenv("COOLDOWN_SECONDS", "120"))  # per-symbol cooldown
DAILY_SIGNAL_LIMIT = int(os.getenv("DAILY_SIGNAL_LIMIT", "100"))
PER_SYMBOL_DAILY_LIMIT = int(os.getenv("PER_SYMBOL_DAILY_LIMIT", "30"))

# Telegram
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_IDS = [x for x in os.getenv("TELEGRAM_CHAT_ID", "").replace(" ", "").split(",") if x]  # allow multiple

# DB
DATABASE_URL = os.getenv("DATABASE_URL", "").strip()
DB_TIMEOUT = 10

# Server
PORT = int(os.getenv("PORT", "8000"))
ENV = os.getenv("ENV", "production")

# Logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

# Time
UTC = timezone.utc


# ===========
# UTIL
# ===========
def to_dt(ts):
    """Accept epoch seconds or ISO; return aware UTC datetime."""
    if ts is None:
        return None
    try:
        return datetime.fromtimestamp(float(ts), tz=UTC)
    except Exception:
        try:
            return datetime.fromisoformat(str(ts).replace("Z", "+00:00")).astimezone(UTC)
        except Exception:
            return None


def now_utc():
    return datetime.now(tz=UTC)


def today_utc():
    return now_utc().date()


def fmt_dt(dt):
    if not dt:
        return "—"
    return dt.astimezone(UTC).strftime("%Y-%m-%d %H:%M:%S UTC")


def safe_float(x, default=None):
    try:
        return float(x)
    except Exception:
        return default


# ===========
# DB LAYER
# ===========
class DB:
    def __init__(self, url: str):
        self.url = url
        self.is_pg = url.startswith("postgres://") or url.startswith("postgresql://")
        self._lock = threading.Lock()
        self._conn = None
        self._ensure_conn()
        self._init_schema()

    def _ensure_conn(self):
        with self._lock:
            if self._conn:
                return
            if self.is_pg:
                if psycopg2 is None:
                    raise RuntimeError("psycopg2 not available but DATABASE_URL is Postgres")
                self._conn = psycopg2.connect(self.url, connect_timeout=DB_TIMEOUT, sslmode="require")
                self._conn.autocommit = True
            else:
                path = self.url if self.url else "signals.db"
                self._conn = sqlite3.connect(path, check_same_thread=False, timeout=DB_TIMEOUT)
                self._conn.row_factory = sqlite3.Row

    def execute(self, sql, params=None, fetch=False, many=False):
        params = params or ()
        self._ensure_conn()
        with self._lock:
            cur = self._conn.cursor()
            try:
                if many and isinstance(params, list):
                    cur.executemany(sql, params)
                else:
                    cur.execute(sql, params)
                if fetch:
                    rows = cur.fetchall()
                    return [dict(zip([c[0] for c in cur.description], r)) for r in rows] if rows else []
                else:
                    self._conn.commit()
                    return None
            finally:
                cur.close()

    def _init_schema(self):
        if self.is_pg:
            self.execute("""
            CREATE TABLE IF NOT EXISTS signals (
                id BIGSERIAL PRIMARY KEY,
                symbol TEXT NOT NULL,
                direction TEXT NOT NULL,
                price DOUBLE PRECISION,
                ts TIMESTAMPTZ NOT NULL,
                strategy TEXT NOT NULL
            );""")
            self.execute("""
            CREATE TABLE IF NOT EXISTS tallies (
                dt DATE NOT NULL,
                kind TEXT NOT NULL,
                count INTEGER NOT NULL DEFAULT 0,
                PRIMARY KEY (dt, kind)
            );""")
            self.execute("""
            CREATE TABLE IF NOT EXISTS quotas (
                dt DATE NOT NULL,
                symbol TEXT NOT NULL,
                sent INTEGER NOT NULL DEFAULT 0,
                PRIMARY KEY (dt, symbol)
            );""")
        else:
            self.execute("""
            CREATE TABLE IF NOT EXISTS signals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                direction TEXT NOT NULL,
                price REAL,
                ts TEXT NOT NULL,
                strategy TEXT NOT NULL
            );""")
            self.execute("""
            CREATE TABLE IF NOT EXISTS tallies (
                dt TEXT NOT NULL,
                kind TEXT NOT NULL,
                count INTEGER NOT NULL DEFAULT 0,
                PRIMARY KEY (dt, kind)
            );""")
            self.execute("""
            CREATE TABLE IF NOT EXISTS quotas (
                dt TEXT NOT NULL,
                symbol TEXT NOT NULL,
                sent INTEGER NOT NULL DEFAULT 0,
                PRIMARY KEY (dt, symbol)
            );""")

    # convenience ops
    def inc_tally(self, kind: str, amount=1):
        dt = today_utc().isoformat()
        self.execute("INSERT OR IGNORE INTO tallies(dt, kind, count) VALUES (?, ?, 0);" if not self.is_pg else
                     "INSERT INTO tallies(dt, kind, count) VALUES (%s, %s, 0) ON CONFLICT (dt, kind) DO NOTHING;",
                     (dt, kind))
        self.execute("UPDATE tallies SET count = count + ? WHERE dt = ? AND kind = ?;" if not self.is_pg else
                     "UPDATE tallies SET count = count + %s WHERE dt = %s AND kind = %s;",
                     (amount, dt, kind))

    def get_tally(self, kind: str):
        dt = today_utc().isoformat()
        rows = self.execute("SELECT count FROM tallies WHERE dt = ? AND kind = ?;" if not self.is_pg else
                            "SELECT count FROM tallies WHERE dt = %s AND kind = %s;",
                            (dt, kind), fetch=True)
        return rows[0]["count"] if rows else 0

    def reset_tallies_kind(self, kind_prefix: str):
        # tallies are date-keyed; natural rollover each day
        pass

    def save_signal(self, symbol, direction, price, ts: datetime, strategy: str):
        if self.is_pg:
            self.execute("INSERT INTO signals(symbol, direction, price, ts, strategy) VALUES (%s,%s,%s,%s,%s);",
                         (symbol, direction, price, ts, strategy))
        else:
            self.execute("INSERT INTO signals(symbol, direction, price, ts, strategy) VALUES (?,?,?,?,?);",
                         (symbol, direction, price, ts.isoformat(), strategy))

    def inc_quota(self, symbol, amount=1):
        dt = today_utc().isoformat()
        self.execute("INSERT OR IGNORE INTO quotas(dt, symbol, sent) VALUES (?, ?, 0);" if not self.is_pg else
                     "INSERT INTO quotas(dt, symbol, sent) VALUES (%s, %s, 0) ON CONFLICT (dt, symbol) DO NOTHING;",
                     (dt, symbol))
        self.execute("UPDATE quotas SET sent = sent + ? WHERE dt = ? AND symbol = ?;" if not self.is_pg else
                     "UPDATE quotas SET sent = sent + %s WHERE dt = %s AND symbol = %s;",
                     (amount, dt, symbol))

    def get_quota_symbol(self, symbol):
        dt = today_utc().isoformat()
        rows = self.execute("SELECT sent FROM quotas WHERE dt = ? AND symbol = ?;" if not self.is_pg else
                            "SELECT sent FROM quotas WHERE dt = %s AND symbol = %s;",
                            (dt, symbol), fetch=True)
        return rows[0]["sent"] if rows else 0

    def get_quota_total(self):
        dt = today_utc().isoformat()
        rows = self.execute("SELECT SUM(sent) AS total FROM quotas WHERE dt = ?;" if not self.is_pg else
                            "SELECT SUM(sent) AS total FROM quotas WHERE dt = %s;",
                            (dt,), fetch=True)
        val = rows[0]["total"] if rows and rows[0]["total"] is not None else 0
        return int(val)


# ===========
# APP & LOGGER
# ===========
app = Flask(__name__)
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    stream=sys.stdout,
)
logger = app.logger
logger.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))
logger.info(f"{APP_NAME} starting…")


# ===========
# GLOBAL STATE
# ===========
DBH = DB(DATABASE_URL)
STATE_LOCK = threading.Lock()
CANDLES = defaultdict(lambda: deque(maxlen=CANDLE_HISTORY))  # symbol -> deque[(dt, close)]
ROLLING = {}  # symbol -> {'fast': deque, 'slow': deque, 'sum_fast': float, 'sum_slow': float, 'prev_fast': float, 'prev_slow': float}
LAST_EPOCH = defaultdict(lambda: None)  # symbol -> last candle epoch processed
LAST_SIGNAL_AT = defaultdict(lambda: None)  # symbol -> dt
WS_THREAD = None
SCHED = BackgroundScheduler(timezone=UTC)
STOP_EVENT = threading.Event()

# pre-fill rolling buffers structures
def _ensure_symbol_struct(symbol):
    if symbol not in ROLLING:
        ROLLING[symbol] = {
            "fast": deque(maxlen=SMA_FAST),
            "slow": deque(maxlen=SMA_SLOW),
            "sum_fast": 0.0,
            "sum_slow": 0.0,
            "prev_fast": None,
            "prev_slow": None,
        }


# ===========
# TELEGRAM
# ===========
def tg_send(text: str):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_IDS:
        logger.warning("Telegram not configured; skipping send")
        return
    for chat_id in TELEGRAM_CHAT_IDS:
        try:
            url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
            payload = {
                "chat_id": chat_id,
                "text": text,
                "parse_mode": "HTML",
                "disable_web_page_preview": True,
            }
            r = requests.post(url, json=payload, timeout=10)
            if r.status_code != 200:
                logger.error(f"Telegram send failed: {r.status_code} {r.text}")
        except Exception as e:
            logger.exception(f"Telegram send exception: {e}")


# ===========
# STRATEGY
# ===========
def _update_sma(symbol: str, close_price: float):
    _ensure_symbol_struct(symbol)
    st = ROLLING[symbol]

    # fast
    if len(st["fast"]) == st["fast"].maxlen:
        st["sum_fast"] -= st["fast"][0]
    st["fast"].append(close_price)
    st["sum_fast"] += close_price
    fast = st["sum_fast"] / len(st["fast"])

    # slow
    if len(st["slow"]) == st["slow"].maxlen:
        st["sum_slow"] -= st["slow"][0]
    st["slow"].append(close_price)
    st["sum_slow"] += close_price
    slow = st["sum_slow"] / len(st["slow"])

    prev_fast, prev_slow = st["prev_fast"], st["prev_slow"]
    st["prev_fast"], st["prev_slow"] = fast, slow
    return prev_fast, prev_slow, fast, slow


def _cooldown_ok(symbol: str):
    last = LAST_SIGNAL_AT.get(symbol)
    if not last:
        return True
    return (now_utc() - last).total_seconds() >= COOLDOWN_SECONDS


def _quotas_ok(symbol: str):
    total = DBH.get_quota_total()
    if total >= DAILY_SIGNAL_LIMIT:
        return False, f"Global daily limit reached ({total}/{DAILY_SIGNAL_LIMIT})"
    per = DBH.get_quota_symbol(symbol)
    if per >= PER_SYMBOL_DAILY_LIMIT:
        return False, f"{symbol} daily limit reached ({per}/{PER_SYMBOL_DAILY_LIMIT})"
    return True, ""


def _pullback_ok(symbol: str, close_price: float, slow: float):
    # basic micro-pullback: demand price is at least PULLBACK_MIN away from slow SMA to avoid chop
    if slow is None:
        return False
    return abs(close_price - slow) >= PULLBACK_MIN


def maybe_emit_signal(symbol: str, close_price: float, candle_dt: datetime, fast: float, slow: float, prev_fast: float, prev_slow: float):
    # Need enough history
    if len(ROLLING[symbol]["slow"]) < SMA_SLOW:
        return

    # Cross detection
    crossed_up = prev_fast is not None and prev_slow is not None and prev_fast <= prev_slow and fast > slow
    crossed_dn = prev_fast is not None and prev_slow is not None and prev_fast >= prev_slow and fast < slow

    if not (crossed_up or crossed_dn):
        return

    if not _cooldown_ok(symbol):
        return

    ok, reason = _quotas_ok(symbol)
    if not ok:
        logger.info(f"Quota blocked for {symbol}: {reason}")
        return

    # pullback filter
    if not _pullback_ok(symbol, close_price, slow):
        return

    direction = "CALL" if crossed_up else "PUT"
    strategy = f"SMA{SMA_FAST}/{SMA_SLOW}+pullback"
    text = (
        f"⚡ <b>Pocket Option Signal</b>\n"
        f"Symbol: <b>{symbol}</b>\n"
        f"Direction: <b>{direction}</b>\n"
        f"Price: <code>{close_price:.5f}</code>\n"
        f"Time: {fmt_dt(candle_dt)}\n"
        f"Granularity: {CANDLE_GRANULARITY}s\n"
        f"Strategy: {strategy}\n"
        f"AppID: {DERIV_APP_ID}\n"
        f"Cooldown: {COOLDOWN_SECONDS}s"
    )

    # Persist + send
    try:
        DBH.save_signal(symbol, direction, close_price, candle_dt, strategy)
        DBH.inc_quota(symbol, 1)
        DBH.inc_tally("signals_sent", 1)
        tg_send(text)
        LAST_SIGNAL_AT[symbol] = now_utc()
        logger.info(f"Signal sent: {symbol} {direction} @ {close_price:.5f}")
    except Exception as e:
        logger.exception(f"Failed to persist/send signal: {e}")


# ===========
# DERIV WS CONSUMER
# ===========
async def deriv_stream(symbols):
    if websockets is None:
        logger.error("websockets lib not available; streaming disabled")
        return

    uri = f"wss://ws.derivws.com/websockets/v3?app_id={DERIV_APP_ID}"

    backoff = 1
    while not STOP_EVENT.is_set():
        try:
            ssl_ctx = ssl.create_default_context()
            async with websockets.connect(uri, ssl=ssl_ctx, ping_interval=20, ping_timeout=20, max_queue=2048) as ws:
                logger.info(f"Connected to Deriv WS: {uri}")

                # optional: authorize if DERIV_API_TOKEN is provided
                if DERIV_API_TOKEN:
                    await ws.send(json.dumps({"authorize": DERIV_API_TOKEN}))
                    auth = json.loads(await ws.recv())
                    if "error" in auth:
                        logger.error(f"Deriv authorize error: {auth['error']}")
                    else:
                        logger.info("Deriv authorized")

                # subscribe to candle streams
                for s in symbols:
                    sub = {
                        "ticks_history": s,
                        "style": "candles",
                        "granularity": CANDLE_GRANULARITY,
                        "count": max(SMA_SLOW * 2, 60),
                        "adjust_start_time": 1,
                        "subscribe": 1,
                    }
                    await ws.send(json.dumps(sub))
                    await asyncio.sleep(0.05)  # gentle pacing

                backoff = 1  # reset backoff after successful connect

                while not STOP_EVENT.is_set():
                    raw = await ws.recv()
                    msg = json.loads(raw)
                    await handle_deriv_message(msg)
        except Exception as e:
            logger.error(f"Deriv WS connection error: {e}")
            # exponential backoff up to 60s
            time.sleep(backoff)
            backoff = min(backoff * 2, 60)


async def handle_deriv_message(msg: dict):
    mtype = msg.get("msg_type")
    if "error" in msg:
        logger.error(f"Deriv error: {msg['error']}")
        return

    if mtype == "ohlc":
        o = msg.get("ohlc") or {}
        symbol = o.get("symbol")
        epoch = o.get("epoch")  # candle open epoch (seconds)
        close = safe_float(o.get("close"))
        if not symbol or epoch is None or close is None:
            logger.warning(f"Incomplete ohlc frame: {msg}")
            return

        candle_dt = to_dt(epoch)
        prev_epoch = LAST_EPOCH.get(symbol)
        LAST_EPOCH[symbol] = epoch

        with STATE_LOCK:
            CANDLES[symbol].append((candle_dt, close))
            prev_fast, prev_slow, fast, slow = _update_sma(symbol, close)

        # trigger only when epoch changes (new candle started)
        if prev_epoch is not None and epoch == prev_epoch:
            return

        maybe_emit_signal(symbol, close, candle_dt, fast, slow, prev_fast, prev_slow)
        return

    if mtype == "history":
        his = msg.get("history") or {}
        symbol = his.get("symbol")
        prices = his.get("prices") or []
        times = his.get("times") or []
        if symbol and prices and times and len(prices) == len(times):
            with STATE_LOCK:
                CANDLES[symbol].clear()
                for p, t in zip(prices, times):
                    val = safe_float(p)
                    if val is None:
                        continue
                    CANDLES[symbol].append((to_dt(t), val))
                    _update_sma(symbol, val)
        return

    # ignore other types


def ws_thread_target(symbols):
    if asyncio is None:
        logger.error("asyncio/websockets not available; WS thread cannot start.")
        return
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(deriv_stream(symbols))


# ===========
# SCHEDULER JOBS
# ===========
def housekeeping():
    # ensure WS thread alive, otherwise restart
    global WS_THREAD
    if WS_THREAD is None or not WS_THREAD.is_alive():
        try:
            logger.warning("WS thread not alive — starting…")
            WS_THREAD = threading.Thread(target=ws_thread_target, args=(DEFAULT_SYMBOLS,), daemon=True)
            WS_THREAD.start()
        except Exception as e:
            logger.exception(f"Failed to start WS thread: {e}")

    total_today = DBH.get_quota_total()
    logger.info(f"housekeeping: symbols={len(DEFAULT_SYMBOLS)} signals_today={total_today}")


def reset_daily():
    logger.info("Daily reset checkpoint reached (UTC).")


def reset_weekly():
    logger.info("Weekly reset checkpoint reached (UTC).")


# ===========
# FLASK ROUTES
# ===========
@app.route("/", methods=["GET"])
def root():
    return jsonify({
        "name": APP_NAME,
        "status": "ok",
        "utc_time": fmt_dt(now_utc()),
        "symbols": DEFAULT_SYMBOLS,
        "signals_today": DBH.get_tally("signals_sent"),
        "daily_quota_used": DBH.get_quota_total(),
        "limits": {
            "daily_total": DAILY_SIGNAL_LIMIT,
            "per_symbol": PER_SYMBOL_DAILY_LIMIT,
            "cooldown_seconds": COOLDOWN_SECONDS
        },
        "granularity_sec": CANDLE_GRANULARITY,
        "app_id": DERIV_APP_ID
    })


@app.route("/metrics", methods=["GET"])
def metrics():
    total_today = DBH.get_quota_total()
    signals_tally = DBH.get_tally("signals_sent")
    lines = [
        f"# {APP_NAME} metrics",
        f"service_up 1",
        f"signals_sent_today {signals_tally}",
        f"quota_total_used_today {total_today}",
        f"granularity_seconds {CANDLE_GRANULARITY}",
        f"sma_fast {SMA_FAST}",
        f"sma_slow {SMA_SLOW}",
    ]
    return Response("\n".join(lines) + "\n", mimetype="text/plain")


DASHBOARD_HTML = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <title>{{ app_name }} Dashboard</title>
  <style>
    body { font-family: system-ui, sans-serif; background:#0f172a; color:#e2e8f0; padding:24px; }
    h1 { color:#38bdf8; }
    .card { background:#111827; border:1px solid #1f2937; border-radius:12px; padding:16px; margin:12px 0; }
    code { background:#0b1220; padding:2px 6px; border-radius:6px; }
    table { width:100%; border-collapse: collapse; margin-top: 12px;}
    th, td { border:1px solid #1f2937; padding:8px; text-align:center;}
    th{ background:#0b1220;}
  </style>
</head>
<body>
  <h1>{{ app_name }}</h1>
  <div class="card">
    <b>Status:</b> <span>OK</span><br/>
    <b>UTC:</b> <code>{{ utc_time }}</code><br/>
    <b>Symbols:</b> <code>{{ symbols|join(", ") }}</code><br/>
    <b>Signals Today:</b> <code>{{ signals_today }}</code><br/>
    <b>Quota Used:</b> <code>{{ quota_used }}/{{ daily_limit }}</code> (per symbol: {{ per_symbol_limit }})<br/>
    <b>Granularity:</b> <code>{{ granularity }}s</code><br/>
    <b>Cooldown:</b> <code>{{ cooldown }}s</code><br/>
  </div>

  <div class="card">
    <h3>Recent Candles (up to {{ hist }} per symbol)</h3>
    {% for sym, candles in candles_map.items() %}
      <h4>{{ sym }}</h4>
      <table>
        <tr><th>Time (UTC)</th><th>Close</th></tr>
        {% for dt, close in candles|list|reverse|slice(0,10) %}
            <tr><td>{{ dt }}</td><td>{{ "%.5f"|format(close) }}</td></tr>
        {% endfor %}
      </table>
    {% endfor %}
  </div>
</body>
</html>
"""

@app.route("/dashboard", methods=["GET"])
def dashboard():
    with STATE_LOCK:
        candles_map = {
            s: [(fmt_dt(dt), c) for (dt, c) in list(CANDLES[s])]
            for s in DEFAULT_SYMBOLS
        }
    return render_template_string(
        DASHBOARD_HTML,
        app_name=APP_NAME,
        utc_time=fmt_dt(now_utc()),
        symbols=DEFAULT_SYMBOLS,
        signals_today=DBH.get_tally("signals_sent"),
        quota_used=DBH.get_quota_total(),
        daily_limit=DAILY_SIGNAL_LIMIT,
        per_symbol_limit=PER_SYMBOL_DAILY_LIMIT,
        granularity=CANDLE_GRANULARITY,
        cooldown=COOLDOWN_SECONDS,
        hist=CANDLE_HISTORY,
        candles_map=candles_map
    )


@app.route("/send_test", methods=["POST", "GET"])
def send_test():
    msg = flask_request.args.get("m") or "Test signal from app.py ✅"
    tg_send(f"🔔 <b>Test</b>\n{msg}\nUTC: {fmt_dt(now_utc())}")
    return jsonify({"ok": True, "sent": msg})


# ===========
# STARTUP (Flask 3.1+ safe) —— runs once per worker at import
# ===========
INIT_LOCK = threading.Lock()
INIT_DONE = False

def start_bot_scheduler():
    SCHED.add_job(housekeeping, CronTrigger(second="*/30"))
    SCHED.add_job(reset_daily, CronTrigger(hour="0", minute="0"))
    SCHED.add_job(reset_weekly, CronTrigger(day_of_week="mon", hour="0", minute="5"))
    SCHED.start()
    logger.info("Scheduler started with housekeeping/daily/weekly jobs.")

def startup_event():
    global WS_THREAD, INIT_DONE
    with INIT_LOCK:
        if INIT_DONE:
            return
        start_bot_scheduler()
        if WS_THREAD is None or not WS_THREAD.is_alive():
            WS_THREAD = threading.Thread(target=ws_thread_target, args=(DEFAULT_SYMBOLS,), daemon=True)
            WS_THREAD.start()
            logger.info("Deriv WS thread started.")
        INIT_DONE = True
        logger.info("Startup init completed.")

# Trigger on module import so gunicorn workers initialize properly
startup_event()


def shutdown(signum=None, frame=None):
    logger.warning("Shutdown signal received; stopping services…")
    STOP_EVENT.set()
    try:
        SCHED.shutdown(wait=False)
    except Exception:
        pass
    time.sleep(0.5)
    sys.exit(0)


signal.signal(signal.SIGTERM, shutdown)
signal.signal(signal.SIGINT, shutdown)


# ===========
# MAIN
# ===========
if __name__ == "__main__":
    # Local dev: python app.py
    app.run(host="0.0.0.0", port=PORT, debug=(ENV != "production"))
