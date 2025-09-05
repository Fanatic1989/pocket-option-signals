# app.py
# Pocket Option Signals — clean Telegram text, tiered quotas, local (Trinidad) market-hours guard
# Flask 3.1+ safe; Deriv WS ("end":"latest"); TREND + CHOP + BASE; env: Render + .env + /etc/secrets

import os, sys, json, time, ssl, signal, logging, threading, html
from math import sqrt
from datetime import datetime, time as dtime, timezone
from collections import deque, defaultdict
from zoneinfo import ZoneInfo  # Python 3.9+

# ---------- ENV BOOTSTRAP ----------
def _bootstrap_env():
    try:
        from dotenv import load_dotenv
        load_dotenv(override=False)
        secdir = "/etc/secrets"
        if os.path.isdir(secdir):
            for fn in os.listdir(secdir):
                fp = os.path.join(secdir, fn)
                if not os.path.isfile(fp):
                    continue
                try: load_dotenv(fp, override=False)
                except Exception: pass
                try:
                    if fn not in os.environ:
                        with open(fp, "r", encoding="utf-8") as fh:
                            os.environ[fn] = fh.read().strip()
                except Exception: pass
    except Exception:
        pass
_bootstrap_env()

import requests
from flask import Flask, jsonify, Response, render_template_string, request as flask_request

# Optional Postgres/SQLite
try:
    import psycopg2
except Exception:
    psycopg2 = None
import sqlite3

# Async WS + scheduler
try:
    import websockets, asyncio
except Exception:
    websockets = None; asyncio = None

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

# ---------- CONFIG ----------
APP_NAME = "Pocket Option Signals"

DEFAULT_SYMBOLS = os.getenv("SYMBOLS", "frxAUDCAD,frxEURUSD,frxGBPUSD").replace(" ", "").split(",")
DERIV_APP_ID    = os.getenv("DERIV_APP_ID", "99185")
DERIV_API_TOKEN = os.getenv("DERIV_API_TOKEN")

CANDLE_GRANULARITY = int(os.getenv("CANDLE_GRANULARITY", "60"))  # seconds
CANDLE_HISTORY     = int(os.getenv("CANDLE_HISTORY", "200"))

SMA_FAST       = int(os.getenv("SMA_FAST", "9"))
SMA_SLOW       = int(os.getenv("SMA_SLOW", "21"))
PULLBACK_MIN   = float(os.getenv("PULLBACK_MIN", "0.00010"))
COOLDOWN_SECONDS = int(os.getenv("COOLDOWN_SECONDS", "120"))

# Global limits (kept; tiers enforce sends)
DAILY_SIGNAL_LIMIT       = int(os.getenv("DAILY_SIGNAL_LIMIT", "1000000"))
PER_SYMBOL_DAILY_LIMIT   = int(os.getenv("PER_SYMBOL_DAILY_LIMIT", "1000000"))

# Tier limits (per day)
FREE_DAILY_LIMIT  = int(os.getenv("FREE_DAILY_LIMIT",  "3"))
BASIC_DAILY_LIMIT = int(os.getenv("BASIC_DAILY_LIMIT", "6"))
PRO_DAILY_LIMIT   = int(os.getenv("PRO_DAILY_LIMIT",   "15"))
# VIP unlimited

# Strategies toggles
ENABLE_TRENDING = os.getenv("ENABLE_TRENDING", "1") == "1"
ENABLE_CHOPPY   = os.getenv("ENABLE_CHOPPY", "1") == "1"

ATR_WINDOW     = int(os.getenv("ATR_WINDOW", "14"))
BB_WINDOW      = int(os.getenv("BB_WINDOW", "20"))
BB_STD_MULT    = float(os.getenv("BB_STD_MULT", "1.0"))
SLOPE_ATR_MULT = float(os.getenv("SLOPE_ATR_MULT", "0.20"))

# Message timing annotations
CANDLE_MIN = int(os.getenv("CANDLE_MIN", "1"))
EXPIRY_MIN = int(os.getenv("EXPIRY_MIN", "5"))

# Local market-hours (defaults: Trinidad & Tobago, Mon–Fri, 08:00–17:00)
TRADING_TZ         = os.getenv("TRADING_TZ", "America/Port_of_Spain")  # UTC-4, no DST
LOCAL_TRADING_DAYS = [d.strip() for d in os.getenv("LOCAL_TRADING_DAYS", "Mon-Fri").split(",") if d.strip()]
LOCAL_START_LOCAL  = os.getenv("LOCAL_TRADING_START", "08:00")  # HH:MM in local time
LOCAL_END_LOCAL    = os.getenv("LOCAL_TRADING_END",   "17:00")  # HH:MM in local time

# Telegram
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")

def _parse_ids(val: str):
    if not val: return []
    val = val.split('#', 1)[0]
    return [p.strip() for p in val.split(",") if p.strip()]

TELEGRAM_FREE   = _parse_ids(os.getenv("TELEGRAM_CHAT_FREE",  ""))
TELEGRAM_BASIC  = _parse_ids(os.getenv("TELEGRAM_CHAT_BASIC", ""))
TELEGRAM_PRO    = _parse_ids(os.getenv("TELEGRAM_CHAT_PRO",   ""))
TELEGRAM_VIP    = _parse_ids(os.getenv("TELEGRAM_CHAT_VIP",   ""))
TELEGRAM_ALL    = []
for _grp in (TELEGRAM_FREE, TELEGRAM_BASIC, TELEGRAM_PRO, TELEGRAM_VIP):
    for _id in _grp:
        if _id not in TELEGRAM_ALL:
            TELEGRAM_ALL.append(_id)

# DB & server
DATABASE_URL = os.getenv("DATABASE_URL", "").strip()
DB_TIMEOUT   = 10
PORT         = int(os.getenv("PORT", "8000"))
ENV          = os.getenv("ENV", "production")
LOG_LEVEL    = os.getenv("LOG_LEVEL", "INFO").upper()
UTC = timezone.utc

# ---------- APP & LOG ----------
app = Flask(__name__)
logging.basicConfig(level=getattr(logging, LOG_LEVEL, logging.INFO),
                    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
                    stream=sys.stdout)
logger = app.logger
logger.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))
logger.info(f"{APP_NAME} starting…")

# ---------- UTIL ----------
def to_dt(ts):
    try: return datetime.fromtimestamp(float(ts), tz=UTC)
    except Exception: return None

def now_utc(): return datetime.now(tz=UTC)
def today_iso(): return now_utc().date().isoformat()
def fmt_dt(dt): return dt.astimezone(UTC).strftime("%Y-%m-%d %H:%M:%S UTC") if dt else "—"
def safe_float(x, d=None):
    try: return float(x)
    except Exception: return d

def weekday_name_idx(idx: int):
    return ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"][idx]

def parse_hhmm_local(s: str, tz: ZoneInfo):
    hh, mm = s.split(":")
    return dtime(int(hh), int(mm), tzinfo=tz)

def expand_day_tokens(tokens):
    # supports "Mon-Fri" and comma lists
    ref = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
    days = set()
    for token in tokens:
        if "-" in token:
            a, b = token.split("-", 1)
            ia, ib = ref.index(a), ref.index(b)
            if ia <= ib: rng = ref[ia:ib+1]
            else: rng = ref[ia:]+ref[:ib+1]
            days.update(rng)
        else:
            days.add(token)
    return days

def market_is_open(ts_utc: datetime):
    """
    Gate by Trinidad local time:
      - Only Mon–Fri (LOCAL_TRADING_DAYS default Mon-Fri)
      - Only between LOCAL_TRADING_START and LOCAL_TRADING_END in TRADING_TZ
    """
    tz = ZoneInfo(TRADING_TZ)
    local_dt = ts_utc.astimezone(tz)
    wd_name = weekday_name_idx(local_dt.weekday())
    allowed_days = expand_day_tokens(LOCAL_TRADING_DAYS)
    if wd_name not in allowed_days:
        return False
    start_t = parse_hhmm_local(LOCAL_START_LOCAL, tz)
    end_t   = parse_hhmm_local(LOCAL_END_LOCAL, tz)
    t = local_dt.timetz()
    return start_t <= t <= end_t

def pretty_symbol(sym: str):
    # frxGBPUSD -> GBP/USD
    if sym.startswith("frx") and len(sym) == 8:
        p = sym[3:]
        return f"{p[:3]}/{p[3:]}"
    if len(sym) >= 6 and sym[-3:].isalpha() and sym[:3].isalpha():
        return f"{sym[:3]}/{sym[3:]}"
    return sym

# ---------- DB ----------
class DB:
    def __init__(self, url: str):
        self.url = url
        self.is_pg = url.startswith(("postgres://","postgresql://"))
        self._lock = threading.Lock()
        self._conn = None
        self._ensure_conn()
        self._init_schema()
    def _ensure_conn(self):
        with self._lock:
            if self._conn: return
            if self.is_pg:
                if psycopg2 is None: raise RuntimeError("psycopg2 not available")
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
                if many and isinstance(params, list): cur.executemany(sql, params)
                else: cur.execute(sql, params)
                if fetch:
                    rows = cur.fetchall()
                    cols = [c[0] for c in cur.description]
                    return [dict(zip(cols, r)) for r in rows] if rows else []
                self._conn.commit()
            finally:
                cur.close()
    def _init_schema(self):
        if self.is_pg:
            self.execute("""CREATE TABLE IF NOT EXISTS signals(
                id BIGSERIAL PRIMARY KEY, symbol TEXT, direction TEXT, price DOUBLE PRECISION,
                ts TIMESTAMPTZ NOT NULL, strategy TEXT );""")
            self.execute("""CREATE TABLE IF NOT EXISTS tallies(
                dt DATE NOT NULL, kind TEXT NOT NULL, count INTEGER NOT NULL DEFAULT 0,
                PRIMARY KEY(dt,kind));""")
            self.execute("""CREATE TABLE IF NOT EXISTS quotas(
                dt DATE NOT NULL, symbol TEXT NOT NULL, sent INTEGER NOT NULL DEFAULT 0,
                PRIMARY KEY(dt, symbol));""")
        else:
            self.execute("""CREATE TABLE IF NOT EXISTS signals(
                id INTEGER PRIMARY KEY AUTOINCREMENT, symbol TEXT, direction TEXT, price REAL,
                ts TEXT NOT NULL, strategy TEXT );""")
            self.execute("""CREATE TABLE IF NOT EXISTS tallies(
                dt TEXT NOT NULL, kind TEXT NOT NULL, count INTEGER NOT NULL DEFAULT 0,
                PRIMARY KEY(dt,kind));""")
            self.execute("""CREATE TABLE IF NOT EXISTS quotas(
                dt TEXT NOT NULL, symbol TEXT NOT NULL, sent INTEGER NOT NULL DEFAULT 0,
                PRIMARY KEY(dt, symbol));""")
    # helpers
    def inc_tally(self, kind: str, amount=1):
        dt = today_iso()
        if self.is_pg:
            self.execute("INSERT INTO tallies(dt,kind,count) VALUES(%s,%s,0) ON CONFLICT DO NOTHING;", (dt, kind))
            self.execute("UPDATE tallies SET count = count + %s WHERE dt=%s AND kind=%s;", (amount, dt, kind))
        else:
            self.execute("INSERT OR IGNORE INTO tallies(dt,kind,count) VALUES(?,?,0);", (dt, kind))
            self.execute("UPDATE tallies SET count = count + ? WHERE dt=? AND kind=?;", (amount, dt, kind))
    def get_tally(self, kind: str):
        dt = today_iso()
        rows = self.execute(("SELECT count FROM tallies WHERE dt=? AND kind=?;") if not self.is_pg
                            else ("SELECT count FROM tallies WHERE dt=%s AND kind=%s;"), (dt, kind), fetch=True)
        return int(rows[0]["count"]) if rows else 0
    def save_signal(self, symbol, direction, price, ts: datetime, strategy: str):
        if self.is_pg:
            self.execute("INSERT INTO signals(symbol,direction,price,ts,strategy) VALUES(%s,%s,%s,%s,%s);",
                         (symbol, direction, price, ts, strategy))
        else:
            self.execute("INSERT INTO signals(symbol,direction,price,ts,strategy) VALUES(?,?,?,?,?);",
                         (symbol, direction, price, ts.isoformat(), strategy))
    def inc_quota(self, symbol, amount=1):
        dt = today_iso()
        if self.is_pg:
            self.execute("INSERT INTO quotas(dt,symbol,sent) VALUES(%s,%s,0) ON CONFLICT DO NOTHING;", (dt, symbol))
            self.execute("UPDATE quotas SET sent = sent + %s WHERE dt=%s AND symbol=%s;", (amount, dt, symbol))
        else:
            self.execute("INSERT OR IGNORE INTO quotas(dt,symbol,sent) VALUES(?,?,0);", (dt, symbol))
            self.execute("UPDATE quotas SET sent = sent + ? WHERE dt=? AND symbol=?;", (amount, dt, symbol))
    def get_quota_symbol(self, symbol):
        dt = today_iso()
        rows = self.execute(("SELECT sent FROM quotas WHERE dt=? AND symbol=?;") if not self.is_pg
                            else ("SELECT sent FROM quotas WHERE dt=%s AND symbol=%s;"), (dt, symbol), fetch=True)
        return int(rows[0]["sent"]) if rows else 0
    def get_quota_total(self):
        dt = today_iso()
        rows = self.execute(("SELECT SUM(sent) AS total FROM quotas WHERE dt=?;") if not self.is_pg
                            else ("SELECT SUM(sent) AS total FROM quotas WHERE dt=%s;"), (dt,), fetch=True)
        return int(rows[0]["total"]) if rows and rows[0]["total"] is not None else 0

DBH = DB(DATABASE_URL)

# ---------- STATE ----------
STATE_LOCK     = threading.Lock()
STOP_EVENT     = threading.Event()
SCHED          = BackgroundScheduler(timezone=UTC)
WS_THREAD      = None

ROLLING   = {}  # per-symbol windows
CANDLES   = defaultdict(lambda: deque(maxlen=CANDLE_HISTORY))  # (dt, close)
LAST_EPOCH = defaultdict(lambda: None)
LAST_SIGNAL_AT = defaultdict(lambda: None)

def ensure_symbol_state(sym):
    if sym in ROLLING: return
    ROLLING[sym] = {
        "fast": deque(maxlen=SMA_FAST), "sum_fast": 0.0,
        "slow": deque(maxlen=SMA_SLOW), "sum_slow": 0.0,
        "prev_fast": None, "prev_slow": None,
        "closes": deque(maxlen=BB_WINDOW), "sum_close": 0.0, "sumsq_close": 0.0,
        "trs": deque(maxlen=ATR_WINDOW), "sum_tr": 0.0, "prev_close": None
    }

# ---------- TELEGRAM (HTML-safe minimal message) ----------
def tg_send_to(chat_ids, text: str):
    if not TELEGRAM_BOT_TOKEN or not chat_ids: return
    for cid in chat_ids:
        try:
            r = requests.post(
                f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage",
                json={"chat_id": cid, "text": text, "parse_mode": "HTML", "disable_web_page_preview": True},
                timeout=10
            )
            if r.status_code != 200:
                logger.error(f"Telegram send failed ({cid}): {r.status_code} {r.text}")
        except Exception as e:
            logger.exception(f"Telegram send exception ({cid}): {e}")

# ---------- CALCS ----------
def update_sma(sym, close_price):
    st = ROLLING[sym]
    if len(st["fast"]) == st["fast"].maxlen: st["sum_fast"] -= st["fast"][0]
    st["fast"].append(close_price); st["sum_fast"] += close_price
    fast = st["sum_fast"] / len(st["fast"])
    if len(st["slow"]) == st["slow"].maxlen: st["sum_slow"] -= st["slow"][0]
    st["slow"].append(close_price); st["sum_slow"] += close_price
    slow = st["sum_slow"] / len(st["slow"])
    prev_fast, prev_slow = st["prev_fast"], st["prev_slow"]
    st["prev_fast"], st["prev_slow"] = fast, slow
    return prev_fast, prev_slow, fast, slow

def update_bb(sym, close_price):
    st = ROLLING[sym]
    if len(st["closes"]) == st["closes"].maxlen:
        old = st["closes"][0]; st["sum_close"] -= old; st["sumsq_close"] -= old*old
    st["closes"].append(close_price); st["sum_close"] += close_price; st["sumsq_close"] += close_price*close_price
    n = len(st["closes"]); 
    if n == 0: return None, None
    mean = st["sum_close"]/n
    var = max(st["sumsq_close"]/n - mean*mean, 0.0)
    return mean, sqrt(var)

def update_atr(sym, high, low, close):
    st = ROLLING[sym]; pc = st["prev_close"]
    tr = abs(high - low) if pc is None else max(abs(high-low), abs(high-pc), abs(low-pc))
    if len(st["trs"]) == st["trs"].maxlen: st["sum_tr"] -= st["trs"][0]
    st["trs"].append(tr); st["sum_tr"] += tr; st["prev_close"] = close
    n = len(st["trs"]); return (st["sum_tr"]/n) if n>0 else None

# ---------- QUOTAS / COOLDOWN ----------
def cooldown_ok(sym):
    last = LAST_SIGNAL_AT.get(sym)
    return True if last is None else (now_utc() - last).total_seconds() >= COOLDOWN_SECONDS

def quotas_ok(sym):
    if DBH.get_quota_symbol(sym) >= PER_SYMBOL_DAILY_LIMIT: return False, f"{sym} per-symbol daily limit reached"
    if DBH.get_quota_total() >= DAILY_SIGNAL_LIMIT: return False, "Global daily limit reached"
    return True, ""

def tier_remaining():
    return {
        "free":  max(FREE_DAILY_LIMIT  - DBH.get_tally("sent_free"),  0),
        "basic": max(BASIC_DAILY_LIMIT - DBH.get_tally("sent_basic"), 0),
        "pro":   max(PRO_DAILY_LIMIT   - DBH.get_tally("sent_pro"),   0),
        "vip":   10**9
    }

def choose_tier_targets():
    rem = tier_remaining()
    targets = {}
    if TELEGRAM_FREE   and rem["free"]  > 0: targets["free"]  = TELEGRAM_FREE
    if TELEGRAM_BASIC  and rem["basic"] > 0: targets["basic"] = TELEGRAM_BASIC
    if TELEGRAM_PRO    and rem["pro"]   > 0: targets["pro"]   = TELEGRAM_PRO
    if TELEGRAM_VIP:                        targets["vip"]   = TELEGRAM_VIP
    return targets, rem

def record_tier_send(count_by_tier):
    for tier, n in count_by_tier.items():
        if n > 0:
            DBH.inc_tally(f"sent_{tier}", 1)

# ---------- SIGNAL EMIT ----------
def minimal_message(sym_pretty: str, direction: str, when: datetime):
    line1 = f"⚡ <b>{html.escape(APP_NAME, quote=False)}</b>"
    line2 = f"{html.escape(sym_pretty, quote=False)} — <b>{html.escape(direction, quote=False)}</b>"
    line3 = f"Candle: {CANDLE_MIN}m | Expiry: {EXPIRY_MIN}m"
    line4 = f"Time: {html.escape(fmt_dt(when), quote=False)}"
    return "\n".join([line1, line2, line3, line4])

def send_signal(sym: str, direction: str, price: float, when: datetime, strategy_name: str):
    # Local (Trinidad) market-hours guard
    if not market_is_open(when):
        logger.info(f"[SKIP local market closed] {sym} {direction} @ {price:.5f}")
        return
    if not cooldown_ok(sym): return
    ok, reason = quotas_ok(sym)
    if not ok:
        logger.info(f"[quota block] {sym}: {reason}"); return

    sym_pretty = pretty_symbol(sym)
    text = minimal_message(sym_pretty, direction, when)

    targets, rem = choose_tier_targets()
    if not targets:
        logger.info(f"[tier caps reached] no tiers available for send ({sym_pretty})")
        return

    used = {}
    for tier, ids in targets.items():
        if not ids: continue
        tg_send_to(ids, text)
        used[tier] = 1

    DBH.save_signal(sym, direction, price, when, strategy_name)
    DBH.inc_quota(sym, 1)
    DBH.inc_tally("signals_sent", 1)
    record_tier_send(used)

    LAST_SIGNAL_AT[sym] = now_utc()
    logger.info(f"[{strategy_name}] {sym} {direction} @ {price:.5f} | sent tiers: {','.join(used.keys())}")

# ---------- STRATEGIES ----------
def maybe_emit_trend(sym, close_price, when, fast, slow, prev_fast, prev_slow, atr):
    if not ENABLE_TRENDING: return
    if len(ROLLING[sym]["slow"]) < SMA_SLOW or atr is None: return
    slope = abs(slow - (prev_slow if prev_slow is not None else slow))
    slope_thresh = SLOPE_ATR_MULT * atr
    is_trending = slope >= slope_thresh and abs(close_price - slow) >= (0.25 * atr if atr and atr > 0 else PULLBACK_MIN)
    if not is_trending: return
    crossed_up = prev_fast is not None and prev_slow is not None and prev_fast <= prev_slow and fast > slow
    crossed_dn = prev_fast is not None and prev_slow is not None and prev_fast >= prev_slow and fast < slow
    if not (crossed_up or crossed_dn): return
    direction = "CALL" if crossed_up else "PUT"
    send_signal(sym, direction, close_price, when, "TREND:SMA_CROSS+SLOPE_ATR")

def maybe_emit_chop(sym, close_price, when, mean, std, slow, prev_slow, atr):
    if not ENABLE_CHOPPY: return
    if mean is None or std is None or prev_slow is None or atr is None: return
    slope = abs(slow - prev_slow)
    slope_thresh = SLOPE_ATR_MULT * atr
    if slope >= slope_thresh: return
    upper, lower = mean + BB_STD_MULT * std, mean - BB_STD_MULT * std
    if close_price >= upper: direction = "PUT"
    elif close_price <= lower: direction = "CALL"
    else: return
    send_signal(sym, direction, close_price, when, "CHOP:MeanRevert@BB")

def maybe_emit_base(sym, close_price, when, fast, slow, prev_fast, prev_slow):
    if len(ROLLING[sym]["slow"]) < SMA_SLOW: return
    crossed_up = prev_fast is not None and prev_slow is not None and prev_fast <= prev_slow and fast > slow
    crossed_dn = prev_fast is not None and prev_slow is not None and prev_fast >= prev_slow and fast < slow
    if not (crossed_up or crossed_dn): return
    if abs(close_price - slow) < PULLBACK_MIN: return
    direction = "CALL" if crossed_up else "PUT"
    send_signal(sym, direction, close_price, when, f"BASE:SMA{SMA_FAST}/{SMA_SLOW}+Pullback")

# ---------- DERIV WS ----------
async def deriv_stream(symbols):
    if websockets is None:
        logger.error("websockets lib not available; streaming disabled"); return
    uri = f"wss://ws.derivws.com/websockets/v3?app_id={DERIV_APP_ID}"
    backoff = 1
    while not STOP_EVENT.is_set():
        try:
            ssl_ctx = ssl.create_default_context()
            async with websockets.connect(uri, ssl=ssl_ctx, ping_interval=20, ping_timeout=20, max_queue=2048) as ws:
                logger.info(f"Connected to Deriv WS: {uri}")

                if DERIV_API_TOKEN:
                    await ws.send(json.dumps({"authorize": DERIV_API_TOKEN}))
                    auth = json.loads(await ws.recv())
                    if "error" in auth: logger.error(f"Deriv authorize error: {auth['error']}")
                    else: logger.info("Deriv authorized")

                # Subscribe per symbol (requires end="latest")
                for s in symbols:
                    sub = {
                        "ticks_history": s,
                        "style": "candles",
                        "granularity": CANDLE_GRANULARITY,
                        "count": max(SMA_SLOW * 2, 60),
                        "adjust_start_time": 1,
                        "end": "latest",
                        "subscribe": 1
                    }
                    await ws.send(json.dumps(sub))
                    await asyncio.sleep(0.05)

                backoff = 1
                while not STOP_EVENT.is_set():
                    raw = await ws.recv()
                    msg = json.loads(raw)
                    await handle_deriv_message(msg)

        except Exception as e:
            logger.error(f"Deriv WS connection error: {e}")
            time.sleep(backoff); backoff = min(backoff * 2, 60)

async def handle_deriv_message(msg: dict):
    if "error" in msg:
        logger.error(f"Deriv error: {msg['error']}"); return
    t = msg.get("msg_type")
    if t == "ohlc":
        o = msg.get("ohlc") or {}
        sym = o.get("symbol"); epoch = o.get("epoch")
        h_, l_, c_ = safe_float(o.get("high")), safe_float(o.get("low")), safe_float(o.get("close"))
        if not sym or epoch is None or h_ is None or l_ is None or c_ is None:
            logger.warning(f"Incomplete ohlc: {msg}"); return
        when = to_dt(epoch)
        prev_epoch = LAST_EPOCH.get(sym); LAST_EPOCH[sym] = epoch

        with STATE_LOCK:
            ensure_symbol_state(sym)
            CANDLES[sym].append((when, c_))
            prev_fast, prev_slow, fast, slow = update_sma(sym, c_)
            mean, std = update_bb(sym, c_)
            atr = update_atr(sym, h_, l_, c_)

        if prev_epoch is not None and epoch == prev_epoch: return

        maybe_emit_trend(sym, c_, when, fast, slow, prev_fast, prev_slow, atr)
        maybe_emit_chop(sym, c_, when, mean, std, slow, prev_slow, atr)
        maybe_emit_base(sym, c_, when, fast, slow, prev_fast, prev_slow)
        return

    if t == "history":
        his = msg.get("history") or {}
        sym = his.get("symbol"); prices = his.get("prices") or []; times = his.get("times") or []
        if sym and prices and times and len(prices) == len(times):
            with STATE_LOCK:
                ensure_symbol_state(sym)
                CANDLES[sym].clear()
                for p, ts in zip(prices, times):
                    close = safe_float(p)
                    if close is None: continue
                    dt = to_dt(ts)
                    CANDLES[sym].append((dt, close))
                    update_sma(sym, close); update_bb(sym, close)
        return
    # ignore others

def ws_thread_target(symbols):
    if asyncio is None:
        logger.error("asyncio/websockets not available; cannot start WS thread."); return
    loop = asyncio.new_event_loop(); asyncio.set_event_loop(loop)
    loop.run_until_complete(deriv_stream(symbols))

# ---------- SCHED ----------
def housekeeping():
    global WS_THREAD
    if WS_THREAD is None or not WS_THREAD.is_alive():
        try:
            logger.warning("WS thread not alive — starting…")
            WS_THREAD = threading.Thread(target=ws_thread_target, args=(DEFAULT_SYMBOLS,), daemon=True)
            WS_THREAD.start()
        except Exception as e:
            logger.exception(f"Failed to start WS thread: {e}")
    tz = ZoneInfo(TRADING_TZ)
    local_now = now_utc().astimezone(tz).strftime("%Y-%m-%d %H:%M:%S %Z")
    logger.info(f"housekeeping: symbols={len(DEFAULT_SYMBOLS)} sent_today={DBH.get_tally('signals_sent')} "
                f"free={DBH.get_tally('sent_free')} basic={DBH.get_tally('sent_basic')} pro={DBH.get_tally('sent_pro')} vip={DBH.get_tally('sent_vip')} "
                f"| local_now={local_now} | market_open={market_is_open(now_utc())}")

def reset_daily():
    logger.info("Daily reset marker (UTC).")

def reset_weekly():
    logger.info("Weekly reset marker (UTC).")

SCHED = BackgroundScheduler(timezone=UTC)
SCHED.add_job(housekeeping,   CronTrigger(second="*/30"))
SCHED.add_job(reset_daily,    CronTrigger(hour="0", minute="0"))
SCHED.add_job(reset_weekly,   CronTrigger(day_of_week="mon", hour="0", minute="5"))
SCHED.start()
logger.info("Scheduler started.")

# ---------- ROUTES ----------
@app.route("/", methods=["GET"])
def root():
    tz = ZoneInfo(TRADING_TZ)
    local_now = now_utc().astimezone(tz).strftime("%Y-%m-%d %H:%M:%S %Z")
    return jsonify({
        "name": APP_NAME, "status": "ok",
        "utc_time": fmt_dt(now_utc()),
        "local_time": local_now, "trading_tz": TRADING_TZ,
        "market_open": market_is_open(now_utc()),
        "local_trading_window": {"days": LOCAL_TRADING_DAYS, "start": LOCAL_START_LOCAL, "end": LOCAL_END_LOCAL},
        "symbols": DEFAULT_SYMBOLS,
        "signals_today": DBH.get_tally("signals_sent"),
        "tier_usage": {
            "free": DBH.get_tally("sent_free"), "basic": DBH.get_tally("sent_basic"),
            "pro": DBH.get_tally("sent_pro"),   "vip": DBH.get_tally("sent_vip")
        },
        "limits": {
            "free": FREE_DAILY_LIMIT, "basic": BASIC_DAILY_LIMIT, "pro": PRO_DAILY_LIMIT, "vip": None,
            "cooldown_seconds": COOLDOWN_SECONDS
        },
        "granularity_sec": CANDLE_GRANULARITY,
        "msg_times": {"candle_min": CANDLE_MIN, "expiry_min": EXPIRY_MIN},
        "strategies": {"trend": ENABLE_TRENDING, "chop": ENABLE_CHOPPY, "base": True},
        "app_id": DERIV_APP_ID
    })

@app.route("/metrics", methods=["GET"])
def metrics():
    lines = [
        f"service_up 1",
        f"signals_sent_today {DBH.get_tally('signals_sent')}",
        f"tier_free_sent {DBH.get_tally('sent_free')}",
        f"tier_basic_sent {DBH.get_tally('sent_basic')}",
        f"tier_pro_sent {DBH.get_tally('sent_pro')}",
        f"tier_vip_sent {DBH.get_tally('sent_vip')}",
        f"granularity_seconds {CANDLE_GRANULARITY}",
        f"sma_fast {SMA_FAST}",
        f"sma_slow {SMA_SLOW}",
        f"trend_enabled {1 if ENABLE_TRENDING else 0}",
        f"chop_enabled {1 if ENABLE_CHOPPY else 0}",
        f"market_open {1 if market_is_open(now_utc()) else 0}",
    ]
    return Response("\n".join(lines) + "\n", mimetype="text/plain")

DASHBOARD_HTML = """
<!doctype html><html><head><meta charset="utf-8"/>
<title>{{ app_name }} Dashboard</title>
<style>
body{font-family:system-ui,sans-serif;background:#0f172a;color:#e2e8f0;padding:24px}
h1{color:#38bdf8}.card{background:#111827;border:1px solid #1f2937;border-radius:12px;padding:16px;margin:12px 0}
code{background:#0b1220;padding:2px 6px;border-radius:6px}table{width:100%;border-collapse:collapse;margin-top:12px}
th,td{border:1px solid #1f2937;padding:8px;text-align:center}th{background:#0b1220}
</style></head><body>
<h1>{{ app_name }}</h1>
<div class="card">
<b>UTC:</b> <code>{{ utc_time }}</code><br/>
<b>Local:</b> <code>{{ local_time }}</code> ({{ tz }})<br/>
<b>Market open:</b> <code>{{ market_open }}</code> ({{ window.days|join(",") }} {{ window.start }}→{{ window.end }} local)<br/>
<b>Symbols:</b> <code>{{ symbols|join(", ") }}</code><br/>
<b>Signals Today:</b> <code>{{ signals_today }}</code><br/>
<b>Tiers:</b> <code>free {{ free }}/{{ lim_free }}, basic {{ basic }}/{{ lim_basic }}, pro {{ pro }}/{{ lim_pro }}, vip ∞</code><br/>
<b>Granularity:</b> <code>{{ granularity }}s</code> · <b>Cooldown:</b> <code>{{ cooldown }}s</code> · <b>Msg:</b> <code>{{ candle }}m→{{ expiry }}m</code><br/>
<b>Strategies:</b> <code>TREND={{ trend }}, CHOP={{ chop }}, BASE=1</code>
</div>
<div class="card">
<h3>Recent Closes (latest 10)</h3>
{% for sym, candles in candles_map.items() %}
  <h4>{{ sym }}</h4>
  <table><tr><th>Time (UTC)</th><th>Close</th></tr>
  {% for dt, close in candles|list|reverse|slice(0,10) %}
    <tr><td>{{ dt }}</td><td>{{ "%.5f"|format(close) }}</td></tr>
  {% endfor %}</table>
{% endfor %}
</div></body></html>
"""

@app.route("/dashboard", methods=["GET"])
def dashboard():
    with STATE_LOCK:
        candles_map = { s: [(fmt_dt(dt), c) for (dt,c) in list(CANDLES[s])] for s in DEFAULT_SYMBOLS }
    tz = ZoneInfo(TRADING_TZ)
    return render_template_string(
        DASHBOARD_HTML,
        app_name=APP_NAME,
        utc_time=fmt_dt(now_utc()),
        local_time=now_utc().astimezone(tz).strftime("%Y-%m-%d %H:%M:%S %Z"),
        tz=TRADING_TZ,
        market_open=market_is_open(now_utc()),
        window={"days": LOCAL_TRADING_DAYS, "start": LOCAL_START_LOCAL, "end": LOCAL_END_LOCAL},
        symbols=DEFAULT_SYMBOLS,
        signals_today=DBH.get_tally("signals_sent"),
        free=DBH.get_tally("sent_free"), basic=DBH.get_tally("sent_basic"),
        pro=DBH.get_tally("sent_pro"),
        lim_free=FREE_DAILY_LIMIT, lim_basic=BASIC_DAILY_LIMIT, lim_pro=PRO_DAILY_LIMIT,
        granularity=CANDLE_GRANULARITY, cooldown=COOLDOWN_SECONDS,
        candle=CANDLE_MIN, expiry=EXPIRY_MIN,
        trend=1 if ENABLE_TRENDING else 0, chop=1 if ENABLE_CHOPPY else 0,
        candles_map=candles_map
    )

@app.route("/send_test", methods=["GET","POST"])
def send_test():
    sym = flask_request.args.get("sym", "GBP/USD")
    direction = flask_request.args.get("dir", "CALL").upper()
    msg = minimal_message(sym, direction, now_utc())
    tg_send_to(TELEGRAM_ALL, msg)
    return jsonify({"ok": True, "sent_to": len(TELEGRAM_ALL)})

# ---------- STARTUP / SHUTDOWN ----------
def shutdown(signum=None, frame=None):
    logger.warning("Shutdown signal received; stopping…")
    STOP_EVENT.set()
    try: SCHED.shutdown(wait=False)
    except Exception: pass
    time.sleep(0.3); sys.exit(0)

def ws_thread_target(symbols):
    if asyncio is None:
        logger.error("asyncio/websockets not available; cannot start WS thread."); return
    loop = asyncio.new_event_loop(); asyncio.set_event_loop(loop)
    loop.run_until_complete(deriv_stream(symbols))

# Kick off on import
WS_THREAD = threading.Thread(target=ws_thread_target, args=(DEFAULT_SYMBOLS,), daemon=True)
WS_THREAD.start()
signal.signal(signal.SIGTERM, shutdown)
signal.signal(signal.SIGINT, shutdown)

# ---------- MAIN ----------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT","8000")), debug=(ENV!="production"))
