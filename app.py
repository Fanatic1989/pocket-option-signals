# app.py
# Pocket Option Signals — single-file deploy (Flask 3.1+ safe)
# Flask + Deriv streaming + TREND & CHOP strategies + quotas/cooldowns + tallies + Telegram + APScheduler + Postgres/SQLite
# (c) 2025 Chris — built by ChatGPT

import os
import sys
import json
import time
import ssl
import signal
import logging
import threading
from math import sqrt
from datetime import datetime, timezone
from collections import deque, defaultdict

import requests
from flask import Flask, jsonify, Response, render_template_string, request as flask_request

# Optional Postgres
try:
    import psycopg2
except Exception:
    psycopg2 = None

import sqlite3

# Async WS
try:
    import websockets
    import asyncio
except Exception:
    websockets = None
    asyncio = None

# Scheduler
try:
    from apscheduler.schedulers.background import BackgroundScheduler
    from apscheduler.triggers.cron import CronTrigger
except Exception as e:
    raise RuntimeError("APScheduler is required (pip install APScheduler)") from e


# ===========
# CONFIG (env tunables)
# ===========
APP_NAME = "Pocket Option Signals"

DEFAULT_SYMBOLS = os.getenv("SYMBOLS", "frxAUDCAD,frxEURUSD,frxGBPUSD").replace(" ", "").split(",")
DERIV_APP_ID = os.getenv("DERIV_APP_ID", "99185")
DERIV_API_TOKEN = os.getenv("DERIV_API_TOKEN")  # optional

# Candle stream
CANDLE_GRANULARITY = int(os.getenv("CANDLE_GRANULARITY", "60"))  # seconds
CANDLE_HISTORY = int(os.getenv("CANDLE_HISTORY", "200"))         # stored per symbol in-memory

# SMA strategy
SMA_FAST = int(os.getenv("SMA_FAST", "9"))
SMA_SLOW = int(os.getenv("SMA_SLOW", "21"))
PULLBACK_MIN = float(os.getenv("PULLBACK_MIN", "0.00010"))
COOLDOWN_SECONDS = int(os.getenv("COOLDOWN_SECONDS", "120"))
DAILY_SIGNAL_LIMIT = int(os.getenv("DAILY_SIGNAL_LIMIT", "150"))
PER_SYMBOL_DAILY_LIMIT = int(os.getenv("PER_SYMBOL_DAILY_LIMIT", "50"))

# TREND/CHOP classifiers
ENABLE_TRENDING = os.getenv("ENABLE_TRENDING", "1") == "1"
ENABLE_CHOPPY = os.getenv("ENABLE_CHOPPY", "1") == "1"

ATR_WINDOW = int(os.getenv("ATR_WINDOW", "14"))
BB_WINDOW = int(os.getenv("BB_WINDOW", "20"))
BB_STD_MULT = float(os.getenv("BB_STD_MULT", "1.0"))          # band width for CHOP signals
SLOPE_ATR_MULT = float(os.getenv("SLOPE_ATR_MULT", "0.20"))   # trend slope threshold as % of ATR

# Telegram
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_IDS = [x for x in os.getenv("TELEGRAM_CHAT_ID", "").replace(" ", "").split(",") if x]

# DB
DATABASE_URL = os.getenv("DATABASE_URL", "").strip()
DB_TIMEOUT = 10

# Server
PORT = int(os.getenv("PORT", "8000"))
ENV = os.getenv("ENV", "production")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

UTC = timezone.utc


# ===========
# UTIL
# ===========
def to_dt(ts):
    if ts is None:
        return None
    try:
        return datetime.fromtimestamp(float(ts), tz=UTC)
    except Exception:
        return None

def now_utc():
    return datetime.now(tz=UTC)

def today_iso():
    return now_utc().date().isoformat()

def fmt_dt(dt):
    return dt.astimezone(UTC).strftime("%Y-%m-%d %H:%M:%S UTC") if dt else "—"

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
        self.is_pg = url.startswith(("postgres://", "postgresql://"))
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
                    cols = [c[0] for c in cur.description]
                    return [dict(zip(cols, r)) for r in rows] if rows else []
                self._conn.commit()
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
                );
            """)
            self.execute("""
                CREATE TABLE IF NOT EXISTS tallies (
                    dt DATE NOT NULL,
                    kind TEXT NOT NULL,
                    count INTEGER NOT NULL DEFAULT 0,
                    PRIMARY KEY (dt, kind)
                );
            """)
            self.execute("""
                CREATE TABLE IF NOT EXISTS quotas (
                    dt DATE NOT NULL,
                    symbol TEXT NOT NULL,
                    sent INTEGER NOT NULL DEFAULT 0,
                    PRIMARY KEY (dt, symbol)
                );
            """)
        else:
            self.execute("""
                CREATE TABLE IF NOT EXISTS signals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    direction TEXT NOT NULL,
                    price REAL,
                    ts TEXT NOT NULL,
                    strategy TEXT NOT NULL
                );
            """)
            self.execute("""
                CREATE TABLE IF NOT EXISTS tallies (
                    dt TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    count INTEGER NOT NULL DEFAULT 0,
                    PRIMARY KEY (dt, kind)
                );
            """)
            self.execute("""
                CREATE TABLE IF NOT EXISTS quotas (
                    dt TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    sent INTEGER NOT NULL DEFAULT 0,
                    PRIMARY KEY (dt, symbol)
                );
            """)

    # helpers
    def inc_tally(self, kind: str, amount=1):
        dt = today_iso()
        if self.is_pg:
            self.execute("INSERT INTO tallies(dt, kind, count) VALUES (%s,%s,0) ON CONFLICT (dt, kind) DO NOTHING;", (dt, kind))
            self.execute("UPDATE tallies SET count = count + %s WHERE dt = %s AND kind = %s;", (amount, dt, kind))
        else:
            self.execute("INSERT OR IGNORE INTO tallies(dt, kind, count) VALUES (?,?,0);", (dt, kind))
            self.execute("UPDATE tallies SET count = count + ? WHERE dt = ? AND kind = ?;", (amount, dt, kind))

    def get_tally(self, kind: str):
        dt = today_iso()
        rows = self.execute("SELECT count FROM tallies WHERE dt = ? AND kind = ?;" if not self.is_pg else
                            "SELECT count FROM tallies WHERE dt = %s AND kind = %s;", (dt, kind), fetch=True)
        return int(rows[0]["count"]) if rows else 0

    def save_signal(self, symbol, direction, price, ts: datetime, strategy: str):
        if self.is_pg:
            self.execute("INSERT INTO signals(symbol, direction, price, ts, strategy) VALUES (%s,%s,%s,%s,%s);",
                         (symbol, direction, price, ts, strategy))
        else:
            self.execute("INSERT INTO signals(symbol, direction, price, ts, strategy) VALUES (?,?,?,?,?);",
                         (symbol, direction, price, ts.isoformat(), strategy))

    def inc_quota(self, symbol, amount=1):
        dt = today_iso()
        if self.is_pg:
            self.execute("INSERT INTO quotas(dt, symbol, sent) VALUES (%s,%s,0) ON CONFLICT (dt, symbol) DO NOTHING;", (dt, symbol))
            self.execute("UPDATE quotas SET sent = sent + %s WHERE dt = %s AND symbol = %s;", (amount, dt, symbol))
        else:
            self.execute("INSERT OR IGNORE INTO quotas(dt, symbol, sent) VALUES (?,?,0);", (dt, symbol))
            self.execute("UPDATE quotas SET sent = sent + ? WHERE dt = ? AND symbol = ?;", (amount, dt, symbol))

    def get_quota_symbol(self, symbol):
        dt = today_iso()
        rows = self.execute("SELECT sent FROM quotas WHERE dt = ? AND symbol = ?;" if not self.is_pg else
                            "SELECT sent FROM quotas WHERE dt = %s AND symbol = %s;", (dt, symbol), fetch=True)
        return int(rows[0]["sent"]) if rows else 0

    def get_quota_total(self):
        dt = today_iso()
        rows = self.execute("SELECT SUM(sent) AS total FROM quotas WHERE dt = ?;" if not self.is_pg else
                            "SELECT SUM(sent) AS total FROM quotas WHERE dt = %s;", (dt,), fetch=True)
        return int(rows[0]["total"]) if rows and rows[0]["total"] is not None else 0


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
# STATE
# ===========
DBH = DB(DATABASE_URL)
STATE_LOCK = threading.Lock()
STOP_EVENT = threading.Event()
SCHED = BackgroundScheduler(timezone=UTC)
WS_THREAD = None

# Per-symbol rolling state
# Includes:
#  - SMA fast/slow (with rolling sums)
#  - Close stats for BB (sum/sumsq)
#  - ATR (rolling TR sum)
#  - prev_fast/prev_slow, prev_close
ROLLING = {}  # symbol -> dict of deques & sums
CANDLES = defaultdict(lambda: deque(maxlen=CANDLE_HISTORY))  # (dt, close) for dashboard
LAST_EPOCH = defaultdict(lambda: None)  # last ohlc epoch processed
LAST_SIGNAL_AT = defaultdict(lambda: None)

def ensure_symbol_state(sym: str):
    if sym in ROLLING:
        return
    ROLLING[sym] = {
        # SMA
        "fast": deque(maxlen=SMA_FAST), "sum_fast": 0.0,
        "slow": deque(maxlen=SMA_SLOW), "sum_slow": 0.0,
        "prev_fast": None, "prev_slow": None,
        # BB (closes)
        "closes": deque(maxlen=BB_WINDOW), "sum_close": 0.0, "sumsq_close": 0.0,
        # ATR
        "trs": deque(maxlen=ATR_WINDOW), "sum_tr": 0.0,
        "prev_close": None,
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
            r = requests.post(url, json={"chat_id": chat_id, "text": text, "parse_mode": "HTML", "disable_web_page_preview": True}, timeout=10)
            if r.status_code != 200:
                logger.error(f"Telegram send failed: {r.status_code} {r.text}")
        except Exception as e:
            logger.exception(f"Telegram send exception: {e}")


# ===========
# CALCS
# ===========
def update_sma(sym: str, close_price: float):
    st = ROLLING[sym]
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

def update_bb(sym: str, close_price: float):
    st = ROLLING[sym]
    if len(st["closes"]) == st["closes"].maxlen:
        oldest = st["closes"][0]
        st["sum_close"] -= oldest
        st["sumsq_close"] -= oldest * oldest
    st["closes"].append(close_price)
    st["sum_close"] += close_price
    st["sumsq_close"] += close_price * close_price

    n = len(st["closes"])
    if n == 0:
        return None, None
    mean = st["sum_close"] / n
    # pop variance
    var = max(st["sumsq_close"] / n - mean * mean, 0.0)
    std = sqrt(var)
    return mean, std

def update_atr(sym: str, high: float, low: float, close: float):
    st = ROLLING[sym]
    prev_close = st["prev_close"]
    # True Range
    if prev_close is None:
        tr = abs(high - low)
    else:
        tr = max(abs(high - low), abs(high - prev_close), abs(low - prev_close))
    if len(st["trs"]) == st["trs"].maxlen:
        st["sum_tr"] -= st["trs"][0]
    st["trs"].append(tr)
    st["sum_tr"] += tr
    st["prev_close"] = close
    n = len(st["trs"])
    atr = (st["sum_tr"] / n) if n > 0 else None
    return atr


# ===========
# QUOTAS / COOLDOWNS
# ===========
def cooldown_ok(sym: str):
    last = LAST_SIGNAL_AT.get(sym)
    return True if last is None else (now_utc() - last).total_seconds() >= COOLDOWN_SECONDS

def quotas_ok(sym: str):
    if DBH.get_quota_total() >= DAILY_SIGNAL_LIMIT:
        return False, f"Global daily limit reached"
    if DBH.get_quota_symbol(sym) >= PER_SYMBOL_DAILY_LIMIT:
        return False, f"{sym} per-symbol daily limit reached"
    return True, ""


# ===========
# STRATEGY EMITTERS
# ===========
def send_signal(sym: str, direction: str, price: float, when: datetime, strategy_name: str, extra_lines=None):
    extra = "\n".join(extra_lines or [])
    text = (
        f"⚡ <b>Pocket Option Signal</b>\n"
        f"Symbol: <b>{sym}</b>\n"
        f"Direction: <b>{direction}</b>\n"
        f"Price: <code>{price:.5f}</code>\n"
        f"Time: {fmt_dt(when)}\n"
        f"Granularity: {CANDLE_GRANULARITY}s\n"
        f"Strategy: {strategy_name}\n"
        f"{extra}"
        f"AppID: {DERIV_APP_ID}\n"
        f"Cooldown: {COOLDOWN_SECONDS}s"
    )
    DBH.save_signal(sym, direction, price, when, strategy_name)
    DBH.inc_quota(sym, 1)
    DBH.inc_tally("signals_sent", 1)
    tg_send(text)
    LAST_SIGNAL_AT[sym] = now_utc()
    logger.info(f"[{strategy_name}] {sym} {direction} @ {price:.5f}")

def maybe_emit_trend(sym: str, close_price: float, when: datetime, fast: float, slow: float, prev_fast: float, prev_slow: float, atr: float):
    # Need established SMAs & ATR
    if len(ROLLING[sym]["slow"]) < SMA_SLOW or atr is None:
        return
    slope = abs(slow - (prev_slow if prev_slow is not None else slow))
    slope_thresh = SLOPE_ATR_MULT * atr

    # Trend condition: slow slope significant vs ATR and price not too close to mean
    is_trending = slope >= slope_thresh and abs(close_price - slow) >= (0.25 * atr if atr > 0 else PULLBACK_MIN)
    if not (ENABLE_TRENDING and is_trending):
        return

    crossed_up = prev_fast is not None and prev_slow is not None and prev_fast <= prev_slow and fast > slow
    crossed_dn = prev_fast is not None and prev_slow is not None and prev_fast >= prev_slow and fast < slow
    if not (crossed_up or crossed_dn):
        return

    if not cooldown_ok(sym):
        return
    ok, reason = quotas_ok(sym)
    if not ok:
        logger.info(f"[TREND] quota blocked {sym}: {reason}")
        return

    direction = "CALL" if crossed_up else "PUT"
    send_signal(sym, direction, close_price, when, "TREND:SMA_CROSS+SLOPE_ATR",
                [f"ATR({ATR_WINDOW})≈{atr:.6f}", f"Slope≈{slope:.6f} (≥ {slope_thresh:.6f})"])

def maybe_emit_chop(sym: str, close_price: float, when: datetime, mean: float, std: float, slow: float, prev_slow: float, atr: float):
    if not ENABLE_CHOPPY:
        return
    if mean is None or std is None or prev_slow is None or atr is None:
        return
    # Choppy if slow slope small vs ATR
    slope = abs(slow - prev_slow)
    slope_thresh = SLOPE_ATR_MULT * atr
    is_choppy = slope < slope_thresh
    if not is_choppy:
        return

    # Bands around mean
    upper = mean + BB_STD_MULT * std
    lower = mean - BB_STD_MULT * std

    direction = None
    if close_price >= upper:
        direction = "PUT"   # fade top of range
    elif close_price <= lower:
        direction = "CALL"  # fade bottom of range
    else:
        return

    if not cooldown_ok(sym):
        return
    ok, reason = quotas_ok(sym)
    if not ok:
        logger.info(f"[CHOP] quota blocked {sym}: {reason}")
        return

    send_signal(sym, direction, close_price, when, "CHOP:MeanRevert@BB",
                [f"BB({BB_WINDOW},{BB_STD_MULT}) hit",
                 f"Mean≈{mean:.6f}, Std≈{std:.6f}",
                 f"Slope≈{slope:.6f} (< {slope_thresh:.6f})"])

def maybe_emit_base(sym: str, close_price: float, when: datetime, fast: float, slow: float, prev_fast: float, prev_slow: float):
    # Simple cross + pullback as conservative fallback when neither TREND nor CHOP fired
    if len(ROLLING[sym]["slow"]) < SMA_SLOW:
        return
    crossed_up = prev_fast is not None and prev_slow is not None and prev_fast <= prev_slow and fast > slow
    crossed_dn = prev_fast is not None and prev_slow is not None and prev_fast >= prev_slow and fast < slow
    if not (crossed_up or crossed_dn):
        return
    if abs(close_price - slow) < PULLBACK_MIN:
        return
    if not cooldown_ok(sym):
        return
    ok, reason = quotas_ok(sym)
    if not ok:
        logger.info(f"[BASE] quota blocked {sym}: {reason}")
        return
    direction = "CALL" if crossed_up else "PUT"
    send_signal(sym, direction, close_price, when, f"BASE:SMA{SMA_FAST}/{SMA_SLOW}+Pullback")


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

                # optional auth
                if DERIV_API_TOKEN:
                    await ws.send(json.dumps({"authorize": DERIV_API_TOKEN}))
                    auth = json.loads(await ws.recv())
                    if "error" in auth:
                        logger.error(f"Deriv authorize error: {auth['error']}")
                    else:
                        logger.info("Deriv authorized")

                # subscribe per symbol (FIX: requires "end":"latest")
                for s in symbols:
                    sub = {
                        "ticks_history": s,
                        "style": "candles",
                        "granularity": CANDLE_GRANULARITY,
                        "count": max(SMA_SLOW * 2, 60),
                        "adjust_start_time": 1,
                        "end": "latest",    # <<< REQUIRED
                        "subscribe": 1
                    }
                    await ws.send(json.dumps(sub))
                    await asyncio.sleep(0.05)

                backoff = 1  # reset after success

                while not STOP_EVENT.is_set():
                    raw = await ws.recv()
                    msg = json.loads(raw)
                    await handle_deriv_message(msg)

        except Exception as e:
            logger.error(f"Deriv WS connection error: {e}")
            time.sleep(backoff)
            backoff = min(backoff * 2, 60)


async def handle_deriv_message(msg: dict):
    if "error" in msg:
        logger.error(f"Deriv error: {msg['error']}")
        return
    mtype = msg.get("msg_type")

    if mtype == "ohlc":
        o = msg.get("ohlc") or {}
        sym = o.get("symbol")
        epoch = o.get("epoch")
        o_, h_, l_, c_ = safe_float(o.get("open")), safe_float(o.get("high")), safe_float(o.get("low")), safe_float(o.get("close"))
        if not sym or epoch is None or c_ is None or h_ is None or l_ is None:
            logger.warning(f"Incomplete ohlc: {msg}")
            return

        when = to_dt(epoch)
        prev_epoch = LAST_EPOCH.get(sym)
        LAST_EPOCH[sym] = epoch

        with STATE_LOCK:
            ensure_symbol_state(sym)
            # update rolling structures
            CANDLES[sym].append((when, c_))
            prev_fast, prev_slow, fast, slow = update_sma(sym, c_)
            mean, std = update_bb(sym, c_)
            atr = update_atr(sym, h_, l_, c_)

        # only act on new candle (epoch change)
        if prev_epoch is not None and epoch == prev_epoch:
            return

        # Strategy cascade: TREND → CHOP → BASE
        if ENABLE_TRENDING:
            maybe_emit_trend(sym, c_, when, fast, slow, prev_fast, prev_slow, atr)
        if ENABLE_CHOPPY:
            maybe_emit_chop(sym, c_, when, mean, std, slow, prev_slow, atr)
        # BASE fallback (will only fire if its conditions pass and CHOP/TREND didn't already send due to cooldown)
        maybe_emit_base(sym, c_, when, fast, slow, prev_fast, prev_slow)
        return

    if mtype == "history":
        his = msg.get("history") or {}
        sym = his.get("symbol")
        prices = his.get("prices") or []
        times = his.get("times") or []
        # Only closes are provided in 'history'; we'll still warm SMAs/BB but ATR waits for live ohlc
        if sym and prices and times and len(prices) == len(times):
            with STATE_LOCK:
                ensure_symbol_state(sym)
                CANDLES[sym].clear()
                for p, t in zip(prices, times):
                    close = safe_float(p)
                    if close is None:
                        continue
                    dt = to_dt(t)
                    CANDLES[sym].append((dt, close))
                    update_sma(sym, close)
                    update_bb(sym, close)
        return

    # ignore other types


def ws_thread_target(symbols):
    if asyncio is None:
        logger.error("asyncio/websockets not available; cannot start WS thread.")
        return
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(deriv_stream(symbols))


# ===========
# SCHEDULER JOBS
# ===========
def housekeeping():
    # ensure WS thread is up
    global WS_THREAD
    if WS_THREAD is None or not WS_THREAD.is_alive():
        try:
            logger.warning("WS thread not alive — starting…")
            WS_THREAD = threading.Thread(target=ws_thread_target, args=(DEFAULT_SYMBOLS,), daemon=True)
            WS_THREAD.start()
        except Exception as e:
            logger.exception(f"Failed to start WS thread: {e}")
    logger.info(f"housekeeping: symbols={len(DEFAULT_SYMBOLS)} signals_today={DBH.get_quota_total()}")

def reset_daily():
    logger.info("Daily reset (UTC).")

def reset_weekly():
    logger.info("Weekly reset (UTC).")


# ===========
# ROUTES
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
        "limits": {"daily_total": DAILY_SIGNAL_LIMIT, "per_symbol": PER_SYMBOL_DAILY_LIMIT, "cooldown_seconds": COOLDOWN_SECONDS},
        "granularity_sec": CANDLE_GRANULARITY,
        "strategies": {"trend": ENABLE_TRENDING, "chop": ENABLE_CHOPPY, "base": True},
        "app_id": DERIV_APP_ID
    })

@app.route("/metrics", methods=["GET"])
def metrics():
    lines = [
        f"# {APP_NAME} metrics",
        f"service_up 1",
        f"signals_sent_today {DBH.get_tally('signals_sent')}",
        f"quota_total_used_today {DBH.get_quota_total()}",
        f"granularity_seconds {CANDLE_GRANULARITY}",
        f"sma_fast {SMA_FAST}",
        f"sma_slow {SMA_SLOW}",
        f"trend_enabled {1 if ENABLE_TRENDING else 0}",
        f"chop_enabled {1 if ENABLE_CHOPPY else 0}",
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
    <b>UTC:</b> <code>{{ utc_time }}</code><br/>
    <b>Symbols:</b> <code>{{ symbols|join(", ") }}</code><br/>
    <b>Signals Today:</b> <code>{{ signals_today }}</code><br/>
    <b>Quota Used:</b> <code>{{ quota_used }}/{{ daily_limit }}</code> (per symbol: {{ per_symbol_limit }})<br/>
    <b>Granularity:</b> <code>{{ granularity }}s</code><br/>
    <b>Cooldown:</b> <code>{{ cooldown }}s</code><br/>
    <b>Strategies:</b> <code>TREND={{ trend }}, CHOP={{ chop }}, BASE=1</code>
  </div>

  <div class="card">
    <h3>Recent Candles (latest 10 shown per symbol)</h3>
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
        candles_map = {s: [(fmt_dt(dt), c) for (dt, c) in list(CANDLES[s])] for s in DEFAULT_SYMBOLS}
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
        trend=1 if ENABLE_TRENDING else 0,
        chop=1 if ENABLE_CHOPPY else 0,
        candles_map=candles_map
    )

@app.route("/send_test", methods=["POST", "GET"])
def send_test():
    msg = flask_request.args.get("m") or "Test signal ✅"
    tg_send(f"🔔 <b>Test</b>\n{msg}\nUTC: {fmt_dt(now_utc())}")
    return jsonify({"ok": True, "sent": msg})


# ===========
# STARTUP (Flask 3.1+ safe) — initialize once per worker on import
# ===========
INIT_LOCK = threading.Lock()
INIT_DONE = False

def start_scheduler():
    SCHED.add_job(housekeeping, CronTrigger(second="*/30"))
    SCHED.add_job(reset_daily, CronTrigger(hour="0", minute="0"))
    SCHED.add_job(reset_weekly, CronTrigger(day_of_week="mon", hour="0", minute="5"))
    SCHED.start()
    logger.info("Scheduler started.")

def startup_event():
    global WS_THREAD, INIT_DONE
    with INIT_LOCK:
        if INIT_DONE:
            return
        start_scheduler()
        if WS_THREAD is None or not WS_THREAD.is_alive():
            WS_THREAD = threading.Thread(target=ws_thread_target, args=(DEFAULT_SYMBOLS,), daemon=True)
            WS_THREAD.start()
            logger.info("Deriv WS thread started.")
        INIT_DONE = True
        logger.info("Startup init complete.")

startup_event()  # trigger on module import


def shutdown(signum=None, frame=None):
    logger.warning("Shutdown signal received; stopping…")
    STOP_EVENT.set()
    try:
        SCHED.shutdown(wait=False)
    except Exception:
        pass
    time.sleep(0.3)
    sys.exit(0)

signal.signal(signal.SIGTERM, shutdown)
signal.signal(signal.SIGINT, shutdown)


# ===========
# MAIN (local dev)
# ===========
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT, debug=(ENV != "production"))
