# app.py
# Pocket Option Signals — live control panel + built-in backtesting simulator
# - Live features:
#   * Telegram minimal messages (safe HTML)
#   * Trinidad market-hours gate (Mon–Fri, 08:00–17:00 local)
#   * Tiers per LOCAL day (FREE/BASIC/PRO; VIP unlimited)
#   * TREND / CHOP / BASE strategies, cooldown, Deriv WS ("end":"latest")
#   * Settings persisted in DB; ENV are defaults
#   * Dashboard control panel (symbols, candle/expiry/cooldown, strategy params, tier caps, trading window)
# - Backtesting Simulator:
#   * On-dashboard form (symbol/date range/params)
#   * Runs TREND→CHOP→BASE logic on historical 1m candles from Deriv WS
#   * Summary + per-strategy stats + last 50 trades table
#   * CSV download of full trade list
#   * Admin token required for execution
#   * Range capped by BACKTEST_MAX_DAYS (default 31; editable in settings)

import os, sys, json, time, ssl, signal, logging, threading, html, io, csv
from math import sqrt
from datetime import datetime, date, timedelta, time as dtime, timezone
from collections import deque, defaultdict
from zoneinfo import ZoneInfo

# ---------- ENV BOOTSTRAP ----------
def _bootstrap_env():
    try:
        from dotenv import load_dotenv
        load_dotenv(override=False)
        secdir = "/etc/secrets"
        if os.path.isdir(secdir):
            for fn in os.listdir(secdir):
                fp = os.path.join(secdir, fn)
                if not os.path.isfile(fp): continue
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
from flask import Flask, jsonify, Response, render_template_string, request as flask_request, redirect, url_for, send_file, make_response

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

# ---------- DEFAULT CONFIG (ENV defaults; DB overrides at runtime) ----------
APP_NAME      = os.getenv("APP_NAME", "Pocket Option Signals")
ADMIN_TOKEN   = os.getenv("ADMIN_TOKEN", "")  # required to edit settings / run backtests

DEFAULT_SYMBOLS = os.getenv("SYMBOLS", "frxAUDCAD,frxEURUSD,frxGBPUSD").replace(" ","").split(",")

DERIV_APP_ID    = os.getenv("DERIV_APP_ID", "99185")
DERIV_API_TOKEN = os.getenv("DERIV_API_TOKEN", "")

CANDLE_GRANULARITY = int(os.getenv("CANDLE_GRANULARITY", "60"))
CANDLE_HISTORY     = int(os.getenv("CANDLE_HISTORY", "200"))

SMA_FAST       = int(os.getenv("SMA_FAST", "9"))
SMA_SLOW       = int(os.getenv("SMA_SLOW", "21"))
PULLBACK_MIN   = float(os.getenv("PULLBACK_MIN", "0.00010"))
COOLDOWN_SECONDS = int(os.getenv("COOLDOWN_SECONDS", "120"))

DAILY_SIGNAL_LIMIT       = int(os.getenv("DAILY_SIGNAL_LIMIT", "1000000"))
PER_SYMBOL_DAILY_LIMIT   = int(os.getenv("PER_SYMBOL_DAILY_LIMIT", "1000000"))

FREE_DAILY_LIMIT  = int(os.getenv("FREE_DAILY_LIMIT",  "3"))
BASIC_DAILY_LIMIT = int(os.getenv("BASIC_DAILY_LIMIT", "6"))
PRO_DAILY_LIMIT   = int(os.getenv("PRO_DAILY_LIMIT",   "15"))

ENABLE_TRENDING = os.getenv("ENABLE_TRENDING", "1") == "1"
ENABLE_CHOPPY   = os.getenv("ENABLE_CHOPPY", "1") == "1"

ATR_WINDOW     = int(os.getenv("ATR_WINDOW", "14"))
BB_WINDOW      = int(os.getenv("BB_WINDOW", "20"))
BB_STD_MULT    = float(os.getenv("BB_STD_MULT", "1.0"))
SLOPE_ATR_MULT = float(os.getenv("SLOPE_ATR_MULT", "0.20"))

CANDLE_MIN = int(os.getenv("CANDLE_MIN", "1"))
EXPIRY_MIN = int(os.getenv("EXPIRY_MIN", "5"))

TRADING_TZ         = os.getenv("TRADING_TZ", "America/Port_of_Spain")
LOCAL_TRADING_DAYS = [d.strip() for d in os.getenv("LOCAL_TRADING_DAYS", "Mon-Fri").split(",") if d.strip()]
LOCAL_START_LOCAL  = os.getenv("LOCAL_TRADING_START", "08:00")
LOCAL_END_LOCAL    = os.getenv("LOCAL_TRADING_END",   "17:00")

BACKTEST_MAX_DAYS  = int(os.getenv("BACKTEST_MAX_DAYS", "31"))  # safe cap for UI

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")

def _parse_ids(val: str):
    if not val: return []
    val = val.split('#', 1)[0]
    return [p.strip() for p in val.split(",") if p.strip()]

TELEGRAM_FREE   = _parse_ids(os.getenv("TELEGRAM_CHAT_FREE",  ""))
TELEGRAM_BASIC  = _parse_ids(os.getenv("TELEGRAM_CHAT_BASIC", ""))
TELEGRAM_PRO    = _parse_ids(os.getenv("TELEGRAM_CHAT_PRO",   ""))
TELEGRAM_VIP    = _parse_ids(os.getenv("TELEGRAM_CHAT_VIP",   ""))

DATABASE_URL = os.getenv("DATABASE_URL", "").strip()
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
def tz_obj():
    try:
        return ZoneInfo(get_setting("TRADING_TZ", TRADING_TZ))
    except Exception:
        return ZoneInfo(TRADING_TZ)

TT_TZ = tz_obj()

def to_dt(ts):
    try: return datetime.fromtimestamp(float(ts), tz=UTC)
    except Exception: return None

def now_utc(): return datetime.now(tz=UTC)
def now_local(): return now_utc().astimezone(TT_TZ)
def fmt_dt(dt): return dt.astimezone(UTC).strftime("%Y-%m-%d %H:%M:%S UTC") if dt else "—"

def weekday_name(i): return ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"][i]

def pretty_symbol(sym: str):
    if sym.startswith("frx") and len(sym) == 8:
        p = sym[3:]; return f"{p[:3]}/{p[3:]}"
    if len(sym) >= 6 and sym[-3:].isalpha() and sym[:3].isalpha():
        return f"{sym[:3]}/{sym[3:]}"
    return sym

def safe_float(x, d=None):
    try: return float(x)
    except Exception: return d

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
                self._conn = psycopg2.connect(self.url, connect_timeout=10, sslmode="require")
                self._conn.autocommit = True
            else:
                path = self.url if self.url else "signals.db"
                self._conn = sqlite3.connect(path, check_same_thread=False, timeout=10)
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
                dt TEXT NOT NULL, kind TEXT NOT NULL, count INTEGER NOT NULL DEFAULT 0,
                PRIMARY KEY(dt,kind));""")
            self.execute("""CREATE TABLE IF NOT EXISTS quotas(
                dt TEXT NOT NULL, symbol TEXT NOT NULL, sent INTEGER NOT NULL DEFAULT 0,
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
        # Settings table
        self.execute("""CREATE TABLE IF NOT EXISTS settings(
            key TEXT PRIMARY KEY, value TEXT
        );""")
        # Backtest cache (optional, store CSVs by key)
        self.execute("""CREATE TABLE IF NOT EXISTS backtest_cache(
            id TEXT PRIMARY KEY, created_ts TEXT NOT NULL, summary TEXT NOT NULL, csv BLOB
        );""")
    # Tallies/quotas
    def inc_tally(self, kind: str, amount=1, ts: datetime=None):
        dt = local_day_key(ts)
        if self.is_pg:
            self.execute("INSERT INTO tallies(dt,kind,count) VALUES(%s,%s,0) ON CONFLICT DO NOTHING;", (dt, kind))
            self.execute("UPDATE tallies SET count = count + %s WHERE dt=%s AND kind=%s;", (amount, dt, kind))
        else:
            self.execute("INSERT OR IGNORE INTO tallies(dt,kind,count) VALUES(?,?,0);", (dt, kind))
            self.execute("UPDATE tallies SET count = count + ? WHERE dt=? AND kind=?;", (amount, dt, kind))
    def get_tally(self, kind: str, ts: datetime=None):
        dt = local_day_key(ts)
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
    def inc_quota(self, symbol, amount=1, ts: datetime=None):
        dt = local_day_key(ts)
        if self.is_pg:
            self.execute("INSERT INTO quotas(dt,symbol,sent) VALUES(%s,%s,0) ON CONFLICT DO NOTHING;", (dt, symbol))
            self.execute("UPDATE quotas SET sent = sent + %s WHERE dt=%s AND symbol=%s;", (amount, dt, symbol))
        else:
            self.execute("INSERT OR IGNORE INTO quotas(dt,symbol,sent) VALUES(?,?,0);", (dt, symbol))
            self.execute("UPDATE quotas SET sent = sent + ? WHERE dt=? AND symbol=?;", (amount, dt, symbol))
    def get_quota_symbol(self, symbol, ts: datetime=None):
        dt = local_day_key(ts)
        rows = self.execute(("SELECT sent FROM quotas WHERE dt=? AND symbol=?;") if not self.is_pg
                            else ("SELECT sent FROM quotas WHERE dt=%s AND symbol=%s;"), (dt, symbol), fetch=True)
        return int(rows[0]["sent"]) if rows else 0
    def get_quota_total(self, ts: datetime=None):
        dt = local_day_key(ts)
        rows = self.execute(("SELECT SUM(sent) AS total FROM quotas WHERE dt=?;") if not self.is_pg
                            else ("SELECT SUM(sent) AS total FROM quotas WHERE dt=%s;"), (dt,), fetch=True)
        return int(rows[0]["total"]) if rows and rows[0]["total"] is not None else 0
    # Settings
    def set_setting(self, key: str, value: str):
        if self.is_pg:
            self.execute("INSERT INTO settings(key,value) VALUES(%s,%s) ON CONFLICT (key) DO UPDATE SET value=EXCLUDED.value;", (key, value))
        else:
            self.execute("INSERT INTO settings(key,value) VALUES(?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value;", (key, value))
    def get_setting(self, key: str):
        rows = self.execute(("SELECT value FROM settings WHERE key=?;") if not self.is_pg
                            else ("SELECT value FROM settings WHERE key=%s;"), (key,), fetch=True)
        return rows[0]["value"] if rows else None
    def get_all_settings(self):
        rows = self.execute("SELECT key, value FROM settings;", fetch=True)
        return {r["key"]: r["value"] for r in rows} if rows else {}
    # Backtest cache
    def put_backtest(self, key: str, summary: dict, csv_bytes: bytes):
        self.execute("INSERT OR REPLACE INTO backtest_cache(id,created_ts,summary,csv) VALUES(?,?,?,?);",
                     (key, datetime.utcnow().isoformat(), json.dumps(summary), sqlite3.Binary(csv_bytes)))
    def get_backtest(self, key: str):
        rows = self.execute("SELECT summary,csv FROM backtest_cache WHERE id=?;", (key,), fetch=True)
        if not rows: return None, None
        return json.loads(rows[0]["summary"]), rows[0]["csv"]

DBH = DB(DATABASE_URL)

# ---------- SETTINGS HELPERS ----------
def get_setting(key, default_val):
    v = DBH.get_setting(key)
    return v if v is not None else default_val

def get_bool(key, default_bool):
    v = DBH.get_setting(key)
    if v is None: return default_bool
    return str(v).strip() in ("1","true","True","yes","on")

def get_int(key, default_int):
    v = DBH.get_setting(key)
    if v is None: return default_int
    try: return int(v)
    except Exception: return default_int

def get_float(key, default_float):
    v = DBH.get_setting(key)
    if v is None: return default_float
    try: return float(v)
    except Exception: return default_float

def get_symbols():
    v = DBH.get_setting("SYMBOLS")
    if v:
        xs = [s.strip() for s in v.split(",") if s.strip()]
        return xs if xs else DEFAULT_SYMBOLS
    return DEFAULT_SYMBOLS

def local_day_key(ts_utc: datetime=None):
    tz = tz_obj()
    if ts_utc is None: ts_utc = now_utc()
    return ts_utc.astimezone(tz).date().isoformat()

def local_market_open(ts_utc: datetime):
    tz = tz_obj()
    ld = ts_utc.astimezone(tz)
    days = get_setting("LOCAL_TRADING_DAYS", ",".join(LOCAL_TRADING_DAYS)).split(",")
    def expand(tokens):
        ref = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
        S=set()
        for token in [t.strip() for t in tokens if t.strip()]:
            if "-" in token:
                a,b=token.split("-",1);ia,ib=ref.index(a),ref.index(b)
                rng = ref[ia:ib+1] if ia<=ib else ref[ia:]+ref[:ib+1]
                S.update(rng)
            else: S.add(token)
        return S
    if ["*"] != days and weekday_name(ld.weekday()) not in expand(days): return False
    start = get_setting("LOCAL_TRADING_START", LOCAL_START_LOCAL)
    end   = get_setting("LOCAL_TRADING_END",   LOCAL_END_LOCAL)
    try:
        start_t = dtime(int(start.split(":")[0]), int(start.split(":")[1]), tzinfo=tz)
        end_t   = dtime(int(end.split(":")[0]),   int(end.split(":")[1]),   tzinfo=tz)
    except Exception:
        start_t = dtime(8,0,tzinfo=tz); end_t=dtime(17,0,tzinfo=tz)
    t = ld.timetz()
    return start_t <= t <= end_t

# ---------- STATE ----------
STATE_LOCK     = threading.Lock()
STOP_EVENT     = threading.Event()
SCHED          = BackgroundScheduler(timezone=UTC)
WS_THREAD      = None

ROLLING   = {}  # per-symbol indicator state
CANDLES   = defaultdict(lambda: deque(maxlen=CANDLE_HISTORY))  # (dt, close)
LAST_EPOCH = defaultdict(lambda: None)
LAST_SIGNAL_AT = defaultdict(lambda: None)

def ensure_symbol_state(sym):
    if sym in ROLLING: return
    st = {
        "fast": deque(maxlen=get_int("SMA_FAST", SMA_FAST)), "sum_fast": 0.0,
        "slow": deque(maxlen=get_int("SMA_SLOW", SMA_SLOW)), "sum_slow": 0.0,
        "prev_fast": None, "prev_slow": None,
        "closes": deque(maxlen=get_int("BB_WINDOW", BB_WINDOW)), "sum_close": 0.0, "sumsq_close": 0.0,
        "trs": deque(maxlen=get_int("ATR_WINDOW", ATR_WINDOW)), "sum_tr": 0.0, "prev_close": None
    }
    ROLLING[sym] = st

# ---------- TELEGRAM ----------
def tg_send_to(chat_ids, text: str):
    token = get_setting("TELEGRAM_BOT_TOKEN", TELEGRAM_BOT_TOKEN)
    if not token or not chat_ids: return
    for cid in chat_ids:
        try:
            r = requests.post(
                f"https://api.telegram.org/bot{token}/sendMessage",
                json={"chat_id": cid, "text": text, "parse_mode": "HTML", "disable_web_page_preview": True},
                timeout=10
            )
            if r.status_code != 200:
                logger.error(f"Telegram send failed ({cid}): {r.status_code} {r.text}")
        except Exception as e:
            logger.exception(f"Telegram send exception ({cid}): {e}")

# ---------- INDICATORS ----------
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
    n = len(st["closes"])
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
    return True if last is None else (now_utc() - last).total_seconds() >= get_int("COOLDOWN_SECONDS", COOLDOWN_SECONDS)

def quotas_ok(sym):
    if DBH.get_quota_symbol(sym) >= get_int("PER_SYMBOL_DAILY_LIMIT", PER_SYMBOL_DAILY_LIMIT): return False, f"{sym} per-symbol daily limit reached"
    if DBH.get_quota_total() >= get_int("DAILY_SIGNAL_LIMIT", DAILY_SIGNAL_LIMIT): return False, "Global daily limit reached"
    return True, ""

def tier_remaining():
    return {
        "free":  max(get_int("FREE_DAILY_LIMIT", FREE_DAILY_LIMIT)   - DBH.get_tally("sent_free"),  0),
        "basic": max(get_int("BASIC_DAILY_LIMIT", BASIC_DAILY_LIMIT) - DBH.get_tally("sent_basic"), 0),
        "pro":   max(get_int("PRO_DAILY_LIMIT", PRO_DAILY_LIMIT)     - DBH.get_tally("sent_pro"),   0),
        "vip":   10**9
    }

def choose_tier_targets():
    ids_free  = TELEGRAM_FREE
    ids_basic = TELEGRAM_BASIC
    ids_pro   = TELEGRAM_PRO
    ids_vip   = TELEGRAM_VIP
    rem = tier_remaining()
    targets = {}
    if ids_free  and rem["free"]  > 0: targets["free"]  = ids_free
    if ids_basic and rem["basic"] > 0: targets["basic"] = ids_basic
    if ids_pro   and rem["pro"]   > 0: targets["pro"]   = ids_pro
    if ids_vip:                        targets["vip"]   = ids_vip
    return targets, rem

def record_tier_send(used_tiers: dict, ts: datetime):
    for tier in used_tiers.keys():
        DBH.inc_tally(f"sent_{tier}", 1, ts=ts)

# ---------- SIGNAL EMIT ----------
def minimal_message(sym_pretty: str, direction: str, when: datetime):
    line1 = f"⚡ <b>{html.escape(APP_NAME, quote=False)}</b>"
    line2 = f"{html.escape(sym_pretty, quote=False)} — <b>{html.escape(direction, quote=False)}</b>"
    line3 = f"Candle: {get_int('CANDLE_MIN', CANDLE_MIN)}m | Expiry: {get_int('EXPIRY_MIN', EXPIRY_MIN)}m"
    line4 = f"Time: {html.escape(fmt_dt(when), quote=False)}"
    return "\n".join([line1, line2, line3, line4])

def send_signal(sym: str, direction: str, price: float, when: datetime, strategy_name: str):
    if not local_market_open(when):
        logger.info(f"[SKIP local market closed] {sym} {direction} @ {price:.5f}"); return
    if not cooldown_ok(sym): return
    ok, reason = quotas_ok(sym)
    if not ok:
        logger.info(f"[quota block] {sym}: {reason}"); return

    sym_pretty = pretty_symbol(sym)
    text = minimal_message(sym_pretty, direction, when)

    targets, _rem = choose_tier_targets()
    if not targets:
        logger.info(f"[tier caps reached] no tiers available for send ({sym_pretty})")
        return

    used = {}
    for tier, ids in targets.items():
        if not ids: continue
        tg_send_to(ids, text)
        used[tier] = 1

    DBH.save_signal(sym, direction, price, when, strategy_name)
    DBH.inc_quota(sym, 1, ts=when)
    DBH.inc_tally("signals_sent", 1, ts=when)
    if strategy_name.startswith("TREND"): DBH.inc_tally("signals_trend", 1, ts=when)
    elif strategy_name.startswith("CHOP"): DBH.inc_tally("signals_chop", 1, ts=when)
    elif strategy_name.startswith("BASE"): DBH.inc_tally("signals_base", 1, ts=when)
    record_tier_send(used, ts=when)

    LAST_SIGNAL_AT[sym] = now_utc()
    logger.info(f"[{strategy_name}] {sym} {direction} @ {price:.5f} | sent tiers: {','.join(used.keys())}")

# ---------- STRATEGIES ----------
def maybe_emit_trend(sym, close_price, when, fast, slow, prev_fast, prev_slow, atr):
    if not get_bool("ENABLE_TRENDING", ENABLE_TRENDING): return
    if len(ROLLING[sym]["slow"]) < get_int("SMA_SLOW", SMA_SLOW) or atr is None: return
    slope = abs(slow - (prev_slow if prev_slow is not None else slow))
    slope_thresh = get_float("SLOPE_ATR_MULT", SLOPE_ATR_MULT) * atr
    ppmin = get_float("PULLBACK_MIN", PULLBACK_MIN)
    trending = slope >= slope_thresh and abs(close_price - slow) >= (0.25 * atr if atr and atr > 0 else ppmin)
    if not trending: return
    crossed_up = prev_fast is not None and prev_slow is not None and prev_fast <= prev_slow and fast > slow
    crossed_dn = prev_fast is not None and prev_slow is not None and prev_fast >= prev_slow and fast < slow
    if not (crossed_up or crossed_dn): return
    direction = "CALL" if crossed_up else "PUT"
    send_signal(sym, direction, close_price, when, "TREND:SMA_CROSS+SLOPE_ATR")

def maybe_emit_chop(sym, close_price, when, mean, std, slow, prev_slow, atr):
    if not get_bool("ENABLE_CHOPPY", ENABLE_CHOPPY): return
    if mean is None or std is None or prev_slow is None or atr is None: return
    slope = abs(slow - prev_slow)
    slope_thresh = get_float("SLOPE_ATR_MULT", SLOPE_ATR_MULT) * atr
    if slope >= slope_thresh: return
    upper, lower = mean + get_float("BB_STD_MULT", BB_STD_MULT) * std, mean - get_float("BB_STD_MULT", BB_STD_MULT) * std
    if close_price >= upper: direction = "PUT"
    elif close_price <= lower: direction = "CALL"
    else: return
    send_signal(sym, direction, close_price, when, "CHOP:MeanRevert@BB")

def maybe_emit_base(sym, close_price, when, fast, slow, prev_fast, prev_slow):
    if len(ROLLING[sym]["slow"]) < get_int("SMA_SLOW", SMA_SLOW): return
    crossed_up = prev_fast is not None and prev_slow is not None and prev_fast <= prev_slow and fast > slow
    crossed_dn = prev_fast is not None and prev_slow is not None and prev_fast >= prev_slow and fast < slow
    if not (crossed_up or crossed_dn): return
    if abs(close_price - slow) < get_float("PULLBACK_MIN", PULLBACK_MIN): return
    direction = "CALL" if crossed_up else "PUT"
    send_signal(sym, direction, close_price, when, f"BASE:SMA{get_int('SMA_FAST',SMA_FAST)}/{get_int('SMA_SLOW',SMA_SLOW)}+Pullback")

# ---------- DERIV WS (Live) ----------
async def deriv_stream(symbols):
    if websockets is None:
        logger.error("websockets lib not available; streaming disabled"); return
    uri = f"wss://ws.derivws.com/websockets/v3?app_id={get_setting('DERIV_APP_ID', DERIV_APP_ID)}"
    backoff = 1
    while not STOP_EVENT.is_set():
        try:
            ssl_ctx = ssl.create_default_context()
            async with websockets.connect(uri, ssl=ssl_ctx, ping_interval=20, ping_timeout=20, max_queue=4096) as ws:
                logger.info(f"Connected to Deriv WS: {uri}")

                token = get_setting("DERIV_API_TOKEN", DERIV_API_TOKEN)
                if token:
                    await ws.send(json.dumps({"authorize": token}))
                    auth = json.loads(await ws.recv())
                    if "error" in auth: logger.error(f"Deriv authorize error: {auth['error']}")
                    else: logger.info("Deriv authorized")

                gran = get_int("CANDLE_GRANULARITY", CANDLE_GRANULARITY)
                count_hist = max(get_int("SMA_SLOW", SMA_SLOW) * 2, 60)

                for s in symbols:
                    sub = {
                        "ticks_history": s,
                        "style": "candles",
                        "granularity": gran,
                        "count": count_hist,
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
            return
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

# ---------- WS THREAD MGMT ----------
WS_LOCK = threading.Lock()
WS_THREAD = None

def start_ws_thread(symbols):
    global WS_THREAD
    with WS_LOCK:
        if WS_THREAD and WS_THREAD.is_alive(): return
        if asyncio is None:
            logger.error("asyncio/websockets not available; cannot start WS thread."); return
        def _target():
            loop = asyncio.new_event_loop(); asyncio.set_event_loop(loop)
            loop.run_until_complete(deriv_stream(symbols))
        WS_THREAD = threading.Thread(target=_target, name="deriv-ws", daemon=True)
        WS_THREAD.start()
        logger.info(f"WS thread started for symbols: {symbols}")

def restart_ws_thread():
    logger.info("Requested WS restart — soft-refresh")
    if not (WS_THREAD and WS_THREAD.is_alive()):
        start_ws_thread(get_symbols())
    with STATE_LOCK:
        ROLLING.clear(); CANDLES.clear(); LAST_EPOCH.clear(); LAST_SIGNAL_AT.clear()
        for s in get_symbols(): ensure_symbol_state(s)
    logger.info("WS soft-refresh done")

# ---------- SCHED ----------
def housekeeping():
    if not (WS_THREAD and WS_THREAD.is_alive()):
        start_ws_thread(get_symbols())
    logger.info(
        f"housekeeping: symbols={len(get_symbols())} sent_today={DBH.get_tally('signals_sent')} "
        f"free={DBH.get_tally('sent_free')} basic={DBH.get_tally('sent_basic')} "
        f"pro={DBH.get_tally('sent_pro')} vip={DBH.get_tally('sent_vip')} "
        f"| local_now={now_local().strftime('%Y-%m-%d %H:%M:%S %Z')} "
        f"| market_open={local_market_open(now_utc())}"
    )

SCHED = BackgroundScheduler(timezone=UTC)
SCHED.add_job(housekeeping, CronTrigger(second="*/30"))
SCHED.start()
logger.info("Scheduler started.")

# ---------- ADMIN ----------
def admin_ok(req):
    need = ADMIN_TOKEN or DBH.get_setting("ADMIN_TOKEN")
    supplied = req.args.get("token") or req.form.get("token") or req.headers.get("X-Admin-Token","")
    if not need: return False
    return supplied == need

def current_settings_snapshot():
    return {
        "SYMBOLS": ",".join(get_symbols()),
        "LOCAL_TRADING_DAYS": get_setting("LOCAL_TRADING_DAYS", ",".join(LOCAL_TRADING_DAYS)),
        "LOCAL_TRADING_START": get_setting("LOCAL_TRADING_START", LOCAL_START_LOCAL),
        "LOCAL_TRADING_END": get_setting("LOCAL_TRADING_END", LOCAL_END_LOCAL),
        "CANDLE_GRANULARITY": str(get_int("CANDLE_GRANULARITY", CANDLE_GRANULARITY)),
        "COOLDOWN_SECONDS": str(get_int("COOLDOWN_SECONDS", COOLDOWN_SECONDS)),
        "CANDLE_MIN": str(get_int("CANDLE_MIN", CANDLE_MIN)),
        "EXPIRY_MIN": str(get_int("EXPIRY_MIN", EXPIRY_MIN)),
        "ENABLE_TRENDING": "1" if get_bool("ENABLE_TRENDING", ENABLE_TRENDING) else "0",
        "SMA_FAST": str(get_int("SMA_FAST", SMA_FAST)),
        "SMA_SLOW": str(get_int("SMA_SLOW", SMA_SLOW)),
        "ENABLE_CHOPPY": "1" if get_bool("ENABLE_CHOPPY", ENABLE_CHOPPY) else "0",
        "BB_WINDOW": str(get_int("BB_WINDOW", BB_WINDOW)),
        "BB_STD_MULT": str(get_float("BB_STD_MULT", BB_STD_MULT)),
        "ATR_WINDOW": str(get_int("ATR_WINDOW", ATR_WINDOW)),
        "SLOPE_ATR_MULT": str(get_float("SLOPE_ATR_MULT", SLOPE_ATR_MULT)),
        "PULLBACK_MIN": str(get_float("PULLBACK_MIN", PULLBACK_MIN)),
        "FREE_DAILY_LIMIT": str(get_int("FREE_DAILY_LIMIT", FREE_DAILY_LIMIT)),
        "BASIC_DAILY_LIMIT": str(get_int("BASIC_DAILY_LIMIT", BASIC_DAILY_LIMIT)),
        "PRO_DAILY_LIMIT": str(get_int("PRO_DAILY_LIMIT", PRO_DAILY_LIMIT)),
        "TRADING_TZ": get_setting("TRADING_TZ", TRADING_TZ),
        "BACKTEST_MAX_DAYS": str(get_int("BACKTEST_MAX_DAYS", BACKTEST_MAX_DAYS)),
    }

def save_settings_from_form(form):
    for k in [
        "SYMBOLS","LOCAL_TRADING_DAYS","LOCAL_TRADING_START","LOCAL_TRADING_END",
        "CANDLE_GRANULARITY","COOLDOWN_SECONDS","CANDLE_MIN","EXPIRY_MIN",
        "ENABLE_TRENDING","SMA_FAST","SMA_SLOW","ENABLE_CHOPPY","BB_WINDOW","BB_STD_MULT",
        "ATR_WINDOW","SLOPE_ATR_MULT","PULLBACK_MIN",
        "FREE_DAILY_LIMIT","BASIC_DAILY_LIMIT","PRO_DAILY_LIMIT","TRADING_TZ",
        "BACKTEST_MAX_DAYS"
    ]:
        if k in form:
            DBH.set_setting(k, str(form.get(k)).strip())

# ---------- BACKTEST (Core) ----------
def decide_for_backtest(close_price, high, low, state, params):
    # state: dict with sma/atr/bb deques and prevs; params: dict of ints/floats/bools
    # Update indicators
    # SMA
    if len(state["fast"]) == state["fast"].maxlen: state["sum_fast"] -= state["fast"][0]
    state["fast"].append(close_price); state["sum_fast"] += close_price
    fast = state["sum_fast"]/len(state["fast"])
    if len(state["slow"]) == state["slow"].maxlen: state["sum_slow"] -= state["slow"][0]
    state["slow"].append(close_price); state["sum_slow"] += close_price
    slow = state["sum_slow"]/len(state["slow"])
    prev_fast, prev_slow = state["prev_fast"], state["prev_slow"]
    state["prev_fast"], state["prev_slow"] = fast, slow
    # BB
    if len(state["closes"]) == state["closes"].maxlen:
        old=state["closes"][0]; state["sum_close"]-=old; state["sumsq_close"]-=old*old
    state["closes"].append(close_price); state["sum_close"]+=close_price; state["sumsq_close"]+=close_price*close_price
    n=len(state["closes"]); mean=None; std=None
    if n>0:
        mean=state["sum_close"]/n; var=max(state["sumsq_close"]/n - mean*mean, 0.0); std=var**0.5
    # ATR
    pc = state["prev_close"]
    tr = abs(high-low) if pc is None else max(abs(high-low), abs(high-pc), abs(low-pc))
    if len(state["trs"]) == state["trs"].maxlen: state["sum_tr"] -= state["trs"][0]
    state["trs"].append(tr); state["sum_tr"] += tr; state["prev_close"] = close_price
    atr = (state["sum_tr"]/len(state["trs"])) if len(state["trs"])>0 else None

    # TREND
    if params["ENABLE_TRENDING"] and len(state["slow"])>=params["SMA_SLOW"] and atr is not None:
        slope = abs(slow - (prev_slow if prev_slow is not None else slow))
        slope_thresh = params["SLOPE_ATR_MULT"] * atr
        trending = slope >= slope_thresh and abs(close_price - slow) >= (0.25*atr if atr and atr>0 else params["PULLBACK_MIN"])
        if trending:
            up = prev_fast is not None and prev_slow is not None and prev_fast <= prev_slow and fast > slow
            dn = prev_fast is not None and prev_slow is not None and prev_fast >= prev_slow and fast < slow
            if up:  return "CALL","TREND",fast,slow,mean,std,atr
            if dn:  return "PUT","TREND",fast,slow,mean,std,atr

    # CHOP
    if params["ENABLE_CHOPPY"] and mean is not None and std is not None and prev_slow is not None and atr is not None:
        slope = abs(slow - prev_slow)
        if slope < params["SLOPE_ATR_MULT"] * atr:
            upper = mean + params["BB_STD_MULT"]*std; lower = mean - params["BB_STD_MULT"]*std
            if close_price >= upper: return "PUT","CHOP",fast,slow,mean,std,atr
            if close_price <= lower: return "CALL","CHOP",fast,slow,mean,std,atr

    # BASE
    if len(state["slow"])>=params["SMA_SLOW"]:
        up = prev_fast is not None and prev_slow is not None and prev_fast <= prev_slow and fast > slow
        dn = prev_fast is not None and prev_slow is not None and prev_fast >= prev_slow and fast < slow
        if up and abs(close_price - slow) >= params["PULLBACK_MIN"]: return "CALL","BASE",fast,slow,mean,std,atr
        if dn and abs(close_price - slow) >= params["PULLBACK_MIN"]: return "PUT","BASE",fast,slow,mean,std,atr

    return None, None, fast, slow, mean, std, atr

async def fetch_candles_range(symbol: str, start_epoch: int, end_epoch: int, gran: int, app_id: str, token: str=""):
    uri = f"wss://ws.derivws.com/websockets/v3?app_id={app_id}"
    out = []
    ssl_ctx = ssl.create_default_context()
    async with websockets.connect(uri, ssl=ssl_ctx, ping_interval=20, ping_timeout=20, max_queue=1024) as ws:
        if token:
            await ws.send(json.dumps({"authorize": token}))
            auth = json.loads(await ws.recv())
            if "error" in auth: raise RuntimeError(f"Authorize error: {auth['error']}")

        CHUNK = 7*86400
        s = start_epoch
        while s < end_epoch:
            e = min(s + CHUNK, end_epoch)
            req = {"ticks_history": symbol, "style": "candles", "granularity": gran, "start": s, "end": e}
            await ws.send(json.dumps(req))
            resp = json.loads(await ws.recv())
            if "error" in resp: raise RuntimeError(str(resp["error"]))
            candles = []
            if "candles" in resp:
                candles = resp["candles"]
            elif "history" in resp and "prices" in resp["history"]:
                prices=resp["history"]["prices"]; times=resp["history"]["times"]
                candles = [{"open":p,"high":p,"low":p,"close":p,"epoch":t} for p,t in zip(prices,times)]
            out.extend(candles)
            s = e
            await asyncio.sleep(0.03)
    # dedup/sort
    seen=set(); cleaned=[]
    for c in out:
        ep=int(c["epoch"])
        if ep in seen: continue
        seen.add(ep); cleaned.append(c)
    cleaned.sort(key=lambda x: int(x["epoch"]))
    return cleaned

def eval_trade_result(direction, entry_close, expiry_close, tie_is_win=False):
    if direction=="CALL":
        if expiry_close>entry_close: return +1
        if expiry_close==entry_close: return (+1 if tie_is_win else 0)
        return -1
    else:
        if expiry_close<entry_close: return +1
        if expiry_close==entry_close: return (+1 if tie_is_win else 0)
        return -1

def run_backtest_sync(symbol: str, start_iso: str, end_iso: str, params: dict, csv_limit=None, tie_is_win=False):
    # dates are UTC
    start_dt = datetime.fromisoformat(start_iso)
    if start_dt.tzinfo is None: start_dt = start_dt.replace(tzinfo=UTC)
    end_dt = datetime.fromisoformat(end_iso)
    if end_dt.tzinfo is None: end_dt = end_dt.replace(hour=23, minute=59, second=59, tzinfo=UTC)
    # range cap
    max_days = get_int("BACKTEST_MAX_DAYS", BACKTEST_MAX_DAYS)
    if (end_dt - start_dt).days > max_days:
        raise ValueError(f"Range too large. Max {max_days} days.")

    start_epoch, end_epoch = int(start_dt.timestamp()), int(end_dt.timestamp())
    # fetch candles
    if asyncio is None or websockets is None:
        raise RuntimeError("websockets/asyncio not available on server")
    candles = asyncio.run(fetch_candles_range(
        symbol, start_epoch, end_epoch, params["CANDLE_GRANULARITY"],
        get_setting("DERIV_APP_ID", DERIV_APP_ID), get_setting("DERIV_API_TOKEN", DERIV_API_TOKEN)
    ))

    # backtest
    state = {
        "fast": deque(maxlen=params["SMA_FAST"]), "sum_fast":0.0,
        "slow": deque(maxlen=params["SMA_SLOW"]), "sum_slow":0.0,
        "prev_fast": None, "prev_slow": None,
        "closes": deque(maxlen=params["BB_WINDOW"]), "sum_close":0.0, "sumsq_close":0.0,
        "trs": deque(maxlen=params["ATR_WINDOW"]), "sum_tr":0.0, "prev_close": None
    }

    closes = [c for c in candles if c.get("close") is not None]
    trades = []
    wins=losses=ties=0
    per_strat = defaultdict(lambda: {"wins":0,"losses":0,"ties":0,"count":0})

    for i, c in enumerate(closes):
        when = to_dt(c["epoch"]); close = safe_float(c["close"])
        high = safe_float(c.get("high"), close); low = safe_float(c.get("low"), close)
        direction, strat, *_ = decide_for_backtest(close, high, low, state, params)
        if direction:
            expiry_idx = min(i + params["EXPIRY_MIN"], len(closes)-1)
            expiry_close = safe_float(closes[expiry_idx]["close"], close)
            r = eval_trade_result(direction, close, expiry_close, tie_is_win=tie_is_win)
            if r>0: wins+=1; per_strat[strat]["wins"]+=1
            elif r<0: losses+=1; per_strat[strat]["losses"]+=1
            else: ties+=1; per_strat[strat]["ties"]+=1
            per_strat[strat]["count"]+=1
            trades.append({
                "time_utc": when.strftime("%Y-%m-%d %H:%M:%S"),
                "symbol": symbol, "symbol_pretty": pretty_symbol(symbol),
                "strategy": strat, "direction": direction,
                "entry": f"{close:.6f}", "expiry_min": params["EXPIRY_MIN"],
                "expiry_close": f"{expiry_close:.6f}",
                "result": "WIN" if r>0 else ("LOSS" if r<0 else "TIE")
            })

    total = wins + losses + ties
    winrate = (wins / (wins + losses) * 100) if (wins+losses)>0 else 0.0
    strat_stats = {}
    for s in ("TREND","CHOP","BASE"):
        st = per_strat[s]; w,l = st["wins"], st["losses"]
        strat_stats[s] = {
            "count": st["count"], "wins": w, "losses": l, "ties": st["ties"],
            "winrate": (w/(w+l)*100) if (w+l)>0 else 0.0
        }

    # CSV (optional return full list; UI shows last 50)
    csv_buf = io.StringIO()
    writer = csv.DictWriter(csv_buf, fieldnames=["time_utc","symbol","symbol_pretty","strategy","direction","entry","expiry_min","expiry_close","result"])
    writer.writeheader(); writer.writerows(trades)
    csv_bytes = csv_buf.getvalue().encode("utf-8")

    summary = {
        "symbol": symbol, "symbol_pretty": pretty_symbol(symbol),
        "start": start_dt.strftime("%Y-%m-%d %H:%M"), "end": end_dt.strftime("%Y-%m-%d %H:%M"),
        "granularity": params["CANDLE_GRANULARITY"], "expiry": params["EXPIRY_MIN"],
        "signals": total, "wins": wins, "losses": losses, "ties": ties, "winrate": round(winrate,2),
        "per_strategy": strat_stats
    }
    return summary, trades, csv_bytes

# ---------- UI (Dashboard / Backtest / Metrics) ----------
DASHBOARD_HTML = """
<!doctype html><html><head><meta charset="utf-8"/>
<title>{{ app_name }} Dashboard</title>
<style>
body{font-family:system-ui,sans-serif;background:#0f172a;color:#e2e8f0;padding:24px;max-width:1100px;margin:auto}
h1{color:#38bdf8} .card{background:#111827;border:1px solid #1f2937;border-radius:12px;padding:16px;margin:12px 0}
code{background:#0b1220;padding:2px 6px;border-radius:6px}
table{width:100%;border-collapse:collapse;margin-top:12px}th,td{border:1px solid #1f2937;padding:8px;text-align:center}th{background:#0b1220}
label{display:block;margin-top:8px} input[type=text],input[type=number],textarea,select{width:100%;padding:8px;border-radius:8px;border:1px solid #1f2937;background:#0b1220;color:#e2e8f0}
button{background:#38bdf8;border:none;border-radius:10px;padding:10px 14px;color:#001018;cursor:pointer;font-weight:600;margin-top:10px}
button.secondary{background:#1f2937;color:#e2e8f0}
.row{display:grid;grid-template-columns:1fr 1fr;gap:12px}
small{color:#94a3b8}
</style></head><body>
<h1>{{ app_name }}</h1>

<div class="card">
<div><b>UTC:</b> <code>{{ utc_time }}</code> · <b>Local:</b> <code>{{ local_time }}</code> <small>({{ tz }})</small> · <b>Market open:</b> <code>{{ market_open }}</code></div>
<div><b>Window:</b> <code>{{ window.days }} {{ window.start }}→{{ window.end }} local</code> · <b>Today:</b> <code>{{ local_day }}</code></div>
</div>

<div class="card">
<h3>Today (Local)</h3>
<div><b>Total:</b> <code>{{ totals.signals }}</code> · <b>TREND:</b> <code>{{ totals.trend }}</code> · <b>CHOP:</b> <code>{{ totals.chop }}</code> · <b>BASE:</b> <code>{{ totals.base }}</code></div>
<div style="margin-top:8px"><b>FREE:</b> <code>{{ tiers.free }}/{{ limits.free }}</code> · <b>BASIC:</b> <code>{{ tiers.basic }}/{{ limits.basic }}</code> · <b>PRO:</b> <code>{{ tiers.pro }}/{{ limits.pro }}</code> · <b>VIP sent:</b> <code>{{ tiers.vip }}</code></div>
</div>

<div class="card">
<h3>Symbols & Recent Closes</h3>
<b>Tracking:</b> <code>{{ symbols|join(", ") }}</code> · <b>Granularity:</b> <code>{{ granularity }}s</code> · <b>Cooldown:</b> <code>{{ cooldown }}s</code> · <b>Msg:</b> <code>{{ candle }}m→{{ expiry }}m</code><br/><br/>
{% for sym, candles in candles_map.items() %}
  <h4>{{ sym }}</h4>
  <table><tr><th>Time (UTC)</th><th>Close</th></tr>
  {% for dt, close in candles %}
    <tr><td>{{ dt }}</td><td>{{ "%.5f"|format(close) }}</td></tr>
  {% endfor %}
  </table>
{% endfor %}
</div>

<div class="card">
<h3>Control Panel {% if not can_edit %}<small>(read-only: add ?token=YOUR_TOKEN)</small>{% endif %}</h3>
<form method="POST">
<input type="hidden" name="token" value="{{ token }}"/>
<div class="row">
  <div>
    <label>Symbols (comma separated)</label>
    <textarea name="SYMBOLS" rows="3" {{ 'disabled' if not can_edit else '' }}>{{ current.SYMBOLS }}</textarea>
  </div>
  <div>
    <label>Trading Days (Mon-Fri, Mon,Wed,Fri or *)</label>
    <input type="text" name="LOCAL_TRADING_DAYS" value="{{ current.LOCAL_TRADING_DAYS }}" {{ 'disabled' if not can_edit else '' }}/>
    <label>Local Start (HH:MM)</label>
    <input type="text" name="LOCAL_TRADING_START" value="{{ current.LOCAL_TRADING_START }}" {{ 'disabled' if not can_edit else '' }}/>
    <label>Local End (HH:MM)</label>
    <input type="text" name="LOCAL_TRADING_END" value="{{ current.LOCAL_TRADING_END }}" {{ 'disabled' if not can_edit else '' }}/>
  </div>
</div>

<div class="row">
  <div>
    <label>Candle Granularity (seconds)</label>
    <input type="number" name="CANDLE_GRANULARITY" value="{{ current.CANDLE_GRANULARITY }}" min="5" step="5" {{ 'disabled' if not can_edit else '' }}/>
    <label>Cooldown Seconds</label>
    <input type="number" name="COOLDOWN_SECONDS" value="{{ current.COOLDOWN_SECONDS }}" min="0" step="1" {{ 'disabled' if not can_edit else '' }}/>
  </div>
  <div>
    <label>Message Candle Label (minutes)</label>
    <input type="number" name="CANDLE_MIN" value="{{ current.CANDLE_MIN }}" min="1" step="1" {{ 'disabled' if not can_edit else '' }}/>
    <label>Expiry Minutes</label>
    <input type="number" name="EXPIRY_MIN" value="{{ current.EXPIRY_MIN }}" min="1" step="1" {{ 'disabled' if not can_edit else '' }}/>
  </div>
</div>

<div class="row">
  <div>
    <label>Enable TREND</label>
    <select name="ENABLE_TRENDING" {{ 'disabled' if not can_edit else '' }}>
      <option value="1" {{ 'selected' if current.ENABLE_TRENDING=='1' else '' }}>On</option>
      <option value="0" {{ 'selected' if current.ENABLE_TRENDING=='0' else '' }}>Off</option>
    </select>
    <label>SMA Fast / Slow</label>
    <div class="row">
      <input type="number" name="SMA_FAST" value="{{ current.SMA_FAST }}" min="2" step="1" {{ 'disabled' if not can_edit else '' }}/>
      <input type="number" name="SMA_SLOW" value="{{ current.SMA_SLOW }}" min="3" step="1" {{ 'disabled' if not can_edit else '' }}/>
    </div>
  </div>
  <div>
    <label>Enable CHOP</label>
    <select name="ENABLE_CHOPPY" {{ 'disabled' if not can_edit else '' }}>
      <option value="1" {{ 'selected' if current.ENABLE_CHOPPY=='1' else '' }}>On</option>
      <option value="0" {{ 'selected' if current.ENABLE_CHOPPY=='0' else '' }}>Off</option>
    </select>
    <label>BB Window / Std Mult</label>
    <div class="row">
      <input type="number" name="BB_WINDOW" value="{{ current.BB_WINDOW }}" min="5" step="1" {{ 'disabled' if not can_edit else '' }}/>
      <input type="number" name="BB_STD_MULT" value="{{ current.BB_STD_MULT }}" min="0.1" step="0.1" {{ 'disabled' if not can_edit else '' }}/>
    </div>
  </div>
</div>

<div class="row">
  <div>
    <label>ATR Window</label>
    <input type="number" name="ATR_WINDOW" value="{{ current.ATR_WINDOW }}" min="2" step="1" {{ 'disabled' if not can_edit else '' }}/>
    <label>Slope ATR Mult</label>
    <input type="number" name="SLOPE_ATR_MULT" value="{{ current.SLOPE_ATR_MULT }}" min="0.05" step="0.05" {{ 'disabled' if not can_edit else '' }}/>
  </div>
  <div>
    <label>Pullback Min (price units)</label>
    <input type="text" name="PULLBACK_MIN" value="{{ current.PULLBACK_MIN }}" {{ 'disabled' if not can_edit else '' }}/>
  </div>
</div>

<div class="row">
  <div>
    <label>Tier Limits (per LOCAL day)</label>
    <div class="row">
      <input type="number" name="FREE_DAILY_LIMIT" value="{{ current.FREE_DAILY_LIMIT }}" min="0" step="1" {{ 'disabled' if not can_edit else '' }}/>
      <input type="number" name="BASIC_DAILY_LIMIT" value="{{ current.BASIC_DAILY_LIMIT }}" min="0" step="1" {{ 'disabled' if not can_edit else '' }}/>
    </div>
    <div style="margin-top:6px" class="row">
      <input type="number" name="PRO_DAILY_LIMIT" value="{{ current.PRO_DAILY_LIMIT }}" min="0" step="1" {{ 'disabled' if not can_edit else '' }}/>
      <input type="text" value="VIP = unlimited" disabled/>
    </div>
  </div>
  <div>
    <label>Trading Timezone (IANA)</label>
    <input type="text" name="TRADING_TZ" value="{{ current.TRADING_TZ }}" {{ 'disabled' if not can_edit else '' }}/>
    <label>Backtest Max Days</label>
    <input type="number" name="BACKTEST_MAX_DAYS" value="{{ current.BACKTEST_MAX_DAYS }}" min="1" step="1" {{ 'disabled' if not can_edit else '' }}/>
  </div>
</div>

{% if can_edit %}
  <button type="submit">Save Settings</button>
  <a href="{{ url_for('apply_and_restart') }}?token={{ token }}"><button class="secondary" type="button">Apply & Restart Stream</button></a>
{% endif %}
</form>
</div>

<div class="card">
<h3>Backtesting Simulator {% if not can_edit %}<small>(read-only: add ?token=YOUR_TOKEN)</small>{% endif %}</h3>
<form method="POST" action="{{ url_for('run_backtest') }}">
<input type="hidden" name="token" value="{{ token }}"/>
<div class="row">
  <div>
    <label>Symbol</label>
    <select name="symbol" {{ 'disabled' if not can_edit else '' }}>
      {% for s in symbols %}
        <option value="{{ s }}">{{ s }}</option>
      {% endfor %}
    </select>
    <label>Start (UTC, YYYY-MM-DD)</label>
    <input type="text" name="start" placeholder="2025-08-01" {{ 'disabled' if not can_edit else '' }}/>
    <label>End (UTC, YYYY-MM-DD)</label>
    <input type="text" name="end" placeholder="2025-09-01" {{ 'disabled' if not can_edit else '' }}/>
  </div>
  <div>
    <label>Granularity (seconds)</label>
    <input type="number" name="gran" value="{{ granularity }}" min="5" step="5" {{ 'disabled' if not can_edit else '' }}/>
    <label>Expiry (minutes)</label>
    <input type="number" name="expiry" value="{{ expiry }}" min="1" step="1" {{ 'disabled' if not can_edit else '' }}/>
    <label>Count tie as win?</label>
    <select name="tiewin" {{ 'disabled' if not can_edit else '' }}>
      <option value="0" selected>No</option>
      <option value="1">Yes</option>
    </select>
  </div>
</div>
<div class="row">
  <div>
    <label>Enable TREND</label>
    <select name="trend" {{ 'disabled' if not can_edit else '' }}>
      <option value="1" {{ 'selected' if current.ENABLE_TRENDING=='1' else '' }}>On</option>
      <option value="0" {{ 'selected' if current.ENABLE_TRENDING=='0' else '' }}>Off</option>
    </select>
    <label>SMA Fast / Slow</label>
    <div class="row">
      <input type="number" name="sma_fast" value="{{ current.SMA_FAST }}" min="2" step="1" {{ 'disabled' if not can_edit else '' }}/>
      <input type="number" name="sma_slow" value="{{ current.SMA_SLOW }}" min="3" step="1" {{ 'disabled' if not can_edit else '' }}/>
    </div>
  </div>
  <div>
    <label>Enable CHOP</label>
    <select name="chop" {{ 'disabled' if not can_edit else '' }}>
      <option value="1" {{ 'selected' if current.ENABLE_CHOPPY=='1' else '' }}>On</option>
      <option value="0" {{ 'selected' if current.ENABLE_CHOPPY=='0' else '' }}>Off</option>
    </select>
    <label>BB Window / Std Mult</label>
    <div class="row">
      <input type="number" name="bb_win" value="{{ current.BB_WINDOW }}" min="5" step="1" {{ 'disabled' if not can_edit else '' }}/>
      <input type="number" step="0.1" name="bb_mult" value="{{ current.BB_STD_MULT }}" min="0.1" {{ 'disabled' if not can_edit else '' }}/>
    </div>
  </div>
</div>
<div class="row">
  <div>
    <label>ATR Window</label>
    <input type="number" name="atr_win" value="{{ current.ATR_WINDOW }}" min="2" step="1" {{ 'disabled' if not can_edit else '' }}/>
  </div>
  <div>
    <label>Slope ATR Mult</label>
    <input type="number" step="0.05" name="slope_mult" value="{{ current.SLOPE_ATR_MULT }}" min="0.05" {{ 'disabled' if not can_edit else '' }}/>
    <label>Pullback Min</label>
    <input type="text" name="pullback" value="{{ current.PULLBACK_MIN }}" {{ 'disabled' if not can_edit else '' }}/>
  </div>
</div>
{% if can_edit %}
  <button type="submit">Run Backtest</button>
{% endif %}
</form>

{% if backtest %}
<hr/>
<h4>Backtest Summary</h4>
<div><b>Symbol:</b> <code>{{ backtest.summary.symbol }} ({{ backtest.summary.symbol_pretty }})</code>
 · <b>Period:</b> <code>{{ backtest.summary.start }} → {{ backtest.summary.end }} UTC</code>
 · <b>Gran:</b> <code>{{ backtest.summary.granularity }}s</code>
 · <b>Expiry:</b> <code>{{ backtest.summary.expiry }}m</code></div>
<div style="margin-top:8px"><b>Signals:</b> <code>{{ backtest.summary.signals }}</code>
 · <b>Wins:</b> <code>{{ backtest.summary.wins }}</code>
 · <b>Losses:</b> <code>{{ backtest.summary.losses }}</code>
 · <b>Ties:</b> <code>{{ backtest.summary.ties }}</code>
 · <b>Winrate:</b> <code>{{ "%.2f"|format(backtest.summary.winrate) }}%</code></div>

<h4 style="margin-top:12px">Per Strategy</h4>
<table>
  <tr><th>Strategy</th><th>Count</th><th>Wins</th><th>Losses</th><th>Ties</th><th>Winrate</th></tr>
  {% for name, s in backtest.summary.per_strategy.items() %}
    <tr><td>{{ name }}</td><td>{{ s.count }}</td><td>{{ s.wins }}</td><td>{{ s.losses }}</td><td>{{ s.ties }}</td><td>{{ "%.2f"|format(s.winrate) }}%</td></tr>
  {% endfor %}
</table>

<h4 style="margin-top:12px">Recent Trades (last 50)</h4>
<table>
  <tr><th>Time (UTC)</th><th>Symbol</th><th>Strat</th><th>Dir</th><th>Entry</th><th>Expiry(m)</th><th>Exit</th><th>Result</th></tr>
  {% for t in backtest.recent %}
    <tr><td>{{ t.time_utc }}</td><td>{{ t.symbol_pretty }}</td><td>{{ t.strategy }}</td><td>{{ t.direction }}</td><td>{{ t.entry }}</td><td>{{ t.expiry_min }}</td><td>{{ t.expiry_close }}</td><td>{{ t.result }}</td></tr>
  {% endfor %}
</table>

<p><a href="{{ url_for('download_backtest_csv', key=backtest.key) }}?token={{ token }}"><button class="secondary" type="button">Download CSV</button></a></p>
{% endif %}
</div>

</body></html>
"""

# ---------- ROUTES ----------
@app.route("/", methods=["GET"])
def root():
    return jsonify({
        "name": APP_NAME,
        "status": "ok",
        "utc_time": fmt_dt(now_utc()),
        "local_time": now_local().strftime("%Y-%m-%d %H:%M:%S %Z"),
        "local_day": local_day_key(),
        "trading_tz": get_setting("TRADING_TZ", TRADING_TZ),
        "market_open": local_market_open(now_utc()),
        "local_trading_window": {
            "days": get_setting("LOCAL_TRADING_DAYS", ",".join(LOCAL_TRADING_DAYS)),
            "start": get_setting("LOCAL_TRADING_START", LOCAL_START_LOCAL),
            "end": get_setting("LOCAL_TRADING_END", LOCAL_END_LOCAL)
        },
        "symbols": get_symbols(),
        "today": {
            "signals_total": DBH.get_tally("signals_sent"),
            "trend": DBH.get_tally("signals_trend"),
            "chop": DBH.get_tally("signals_chop"),
            "base": DBH.get_tally("signals_base"),
            "tier": {
                "free": DBH.get_tally("sent_free"),
                "basic": DBH.get_tally("sent_basic"),
                "pro": DBH.get_tally("sent_pro"),
                "vip": DBH.get_tally("sent_vip")
            }
        },
        "limits": {
            "free": get_int("FREE_DAILY_LIMIT", FREE_DAILY_LIMIT),
            "basic": get_int("BASIC_DAILY_LIMIT", BASIC_DAILY_LIMIT),
            "pro": get_int("PRO_DAILY_LIMIT", PRO_DAILY_LIMIT),
            "vip": None
        },
        "granularity_sec": get_int("CANDLE_GRANULARITY", CANDLE_GRANULARITY),
        "message_times": {
            "candle_min": get_int("CANDLE_MIN", CANDLE_MIN),
            "expiry_min": get_int("EXPIRY_MIN", EXPIRY_MIN)
        },
        "strategies": {
            "trend": get_bool("ENABLE_TRENDING", ENABLE_TRENDING),
            "chop": get_bool("ENABLE_CHOPPY", ENABLE_CHOPPY), "base": True
        },
        "backtest_max_days": get_int("BACKTEST_MAX_DAYS", BACKTEST_MAX_DAYS),
        "app_id": get_setting("DERIV_APP_ID", DERIV_APP_ID)
    })

@app.route("/dashboard", methods=["GET","POST"])
def dashboard():
    token = flask_request.args.get("token","") or flask_request.form.get("token","")
    can_edit = admin_ok(flask_request)
    if flask_request.method == "POST":
        if not can_edit:
            return redirect(url_for("dashboard"))
        save_settings_from_form(flask_request.form)
        global TT_TZ; TT_TZ = tz_obj()
        return redirect(url_for("dashboard", token=token))

    with STATE_LOCK:
        candles_map = {}
        for s in get_symbols():
            items = list(CANDLES[s]); trimmed = items[-10:]
            candles_map[s] = [(fmt_dt(dt), c) for (dt, c) in trimmed]

    current = current_settings_snapshot()
    # If there was a recent backtest key in querystring, load & show
    bt_key = flask_request.args.get("bt","")
    backtest=None
    if bt_key:
        summary, csv_bytes = DBH.get_backtest(bt_key)
        if summary:
            # show last 50 trades from csv
            dec = csv_bytes.decode("utf-8", errors="ignore")
            rows = list(csv.DictReader(io.StringIO(dec)))
            recent = rows[-50:]
            backtest = {"summary": summary, "recent": recent, "key": bt_key}

    return render_template_string(
        DASHBOARD_HTML,
        app_name=APP_NAME,
        utc_time=fmt_dt(now_utc()),
        local_time=now_local().strftime("%Y-%m-%d %H:%M:%S %Z"),
        tz=get_setting("TRADING_TZ", TRADING_TZ),
        local_day=local_day_key(),
        market_open=local_market_open(now_utc()),
        window={
            "days": get_setting("LOCAL_TRADING_DAYS", ",".join(LOCAL_TRADING_DAYS)),
            "start": get_setting("LOCAL_TRADING_START", LOCAL_START_LOCAL),
            "end": get_setting("LOCAL_TRADING_END", LOCAL_END_LOCAL)
        },
        symbols=get_symbols(),
        totals={
            "signals": DBH.get_tally("signals_sent"),
            "trend": DBH.get_tally("signals_trend"),
            "chop": DBH.get_tally("signals_chop"),
            "base": DBH.get_tally("signals_base"),
        },
        tiers={
            "free": DBH.get_tally("sent_free"),
            "basic": DBH.get_tally("sent_basic"),
            "pro": DBH.get_tally("sent_pro"),
            "vip": DBH.get_tally("sent_vip"),
        },
        limits={"free": get_int("FREE_DAILY_LIMIT", FREE_DAILY_LIMIT),
                "basic": get_int("BASIC_DAILY_LIMIT", BASIC_DAILY_LIMIT),
                "pro": get_int("PRO_DAILY_LIMIT", PRO_DAILY_LIMIT)},
        granularity=get_int("CANDLE_GRANULARITY", CANDLE_GRANULARITY),
        cooldown=get_int("COOLDOWN_SECONDS", COOLDOWN_SECONDS),
        candle=get_int("CANDLE_MIN", CANDLE_MIN),
        expiry=get_int("EXPIRY_MIN", EXPIRY_MIN),
        candles_map=candles_map,
        can_edit=can_edit,
        token=token,
        current=current,
        backtest=backtest
    )

@app.route("/apply", methods=["GET"])
def apply_and_restart():
    if not admin_ok(flask_request):
        return jsonify({"ok": False, "error": "admin token required"}), 403
    with STATE_LOCK:
        ROLLING.clear(); CANDLES.clear(); LAST_EPOCH.clear(); LAST_SIGNAL_AT.clear()
    restart_ws_thread()
    return redirect(url_for("dashboard", token=flask_request.args.get("token","")))

@app.route("/metrics", methods=["GET"])
def metrics():
    lines = [
        f"service_up 1",
        f"local_day \"{local_day_key()}\"",
        f"market_open {1 if local_market_open(now_utc()) else 0}",
        f"signals_today {DBH.get_tally('signals_sent')}",
        f"signals_trend_today {DBH.get_tally('signals_trend')}",
        f"signals_chop_today {DBH.get_tally('signals_chop')}",
        f"signals_base_today {DBH.get_tally('signals_base')}",
        f"tier_free_sent {DBH.get_tally('sent_free')}",
        f"tier_basic_sent {DBH.get_tally('sent_basic')}",
        f"tier_pro_sent {DBH.get_tally('sent_pro')}",
        f"tier_vip_sent {DBH.get_tally('sent_vip')}",
        f"granularity_seconds {get_int('CANDLE_GRANULARITY', CANDLE_GRANULARITY)}",
        f"sma_fast {get_int('SMA_FAST', SMA_FAST)}",
        f"sma_slow {get_int('SMA_SLOW', SMA_SLOW)}",
        f"trend_enabled {1 if get_bool('ENABLE_TRENDING', ENABLE_TRENDING) else 0}",
        f"chop_enabled {1 if get_bool('ENABLE_CHOPPY', ENABLE_CHOPPY) else 0}",
        f"cooldown_seconds {get_int('COOLDOWN_SECONDS', COOLDOWN_SECONDS)}",
        f"backtest_max_days {get_int('BACKTEST_MAX_DAYS', BACKTEST_MAX_DAYS)}",
    ]
    return Response("\n".join(lines) + "\n", mimetype="text/plain")

# ---------- Backtest endpoints ----------
@app.route("/run_backtest", methods=["POST"])
def run_backtest():
    if not admin_ok(flask_request):
        return jsonify({"ok": False, "error": "admin token required"}), 403
    symbol = flask_request.form.get("symbol", (get_symbols() or ["frxGBPUSD"])[0])
    start = flask_request.form.get("start","").strip()
    end   = flask_request.form.get("end","").strip()
    try:
        # allow simple dates; add time
        _ = datetime.fromisoformat(start)
        _ = datetime.fromisoformat(end if "T" in end else end + "T23:59:59")
    except Exception:
        return jsonify({"ok": False, "error": "Invalid start/end format. Use YYYY-MM-DD"}), 400

    params = {
        "CANDLE_GRANULARITY": int(flask_request.form.get("gran", get_int("CANDLE_GRANULARITY", CANDLE_GRANULARITY))),
        "EXPIRY_MIN": int(flask_request.form.get("expiry", get_int("EXPIRY_MIN", EXPIRY_MIN))),
        "ENABLE_TRENDING": flask_request.form.get("trend", "1") == "1",
        "ENABLE_CHOPPY": flask_request.form.get("chop", "1") == "1",
        "SMA_FAST": int(flask_request.form.get("sma_fast", get_int("SMA_FAST", SMA_FAST))),
        "SMA_SLOW": int(flask_request.form.get("sma_slow", get_int("SMA_SLOW", SMA_SLOW))),
        "BB_WINDOW": int(flask_request.form.get("bb_win", get_int("BB_WINDOW", BB_WINDOW))),
        "BB_STD_MULT": float(flask_request.form.get("bb_mult", get_float("BB_STD_MULT", BB_STD_MULT))),
        "ATR_WINDOW": int(flask_request.form.get("atr_win", get_int("ATR_WINDOW", ATR_WINDOW))),
        "SLOPE_ATR_MULT": float(flask_request.form.get("slope_mult", get_float("SLOPE_ATR_MULT", SLOPE_ATR_MULT))),
        "PULLBACK_MIN": float(flask_request.form.get("pullback", get_float("PULLBACK_MIN", PULLBACK_MIN))),
    }
    tie_is_win = flask_request.form.get("tiewin","0") == "1"

    try:
        summary, trades, csv_bytes = run_backtest_sync(
            symbol, start, end if "T" in end else end, params, tie_is_win=tie_is_win
        )
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 400

    # Cache CSV and redirect back to dashboard with results
    key = f"{symbol}_{start}_{end}_{int(time.time())}"
    DBH.put_backtest(key, summary, csv_bytes)
    return redirect(url_for("dashboard", token=flask_request.form.get("token",""), bt=key))

@app.route("/download_backtest_csv", methods=["GET"])
def download_backtest_csv():
    if not admin_ok(flask_request):
        return jsonify({"ok": False, "error": "admin token required"}), 403
    key = flask_request.args.get("key","")
    summary, csv_bytes = DBH.get_backtest(key)
    if not summary:
        return jsonify({"ok": False, "error": "not found"}), 404
    resp = make_response(csv_bytes)
    resp.headers["Content-Type"] = "text/csv; charset=utf-8"
    fn = f"{summary['symbol']}_{summary['start'].replace(' ','_')}_{summary['end'].replace(' ','_')}.csv"
    resp.headers["Content-Disposition"] = f'attachment; filename="{fn}"'
    return resp

# ---------- TEST ----------
@app.route("/send_test", methods=["GET","POST"])
def send_test():
    sym = flask_request.args.get("sym", "GBP/USD")
    direction = flask_request.args.get("dir", "CALL").upper()
    msg = minimal_message(sym, direction, now_utc())
    ids = TELEGRAM_FREE + TELEGRAM_BASIC + TELEGRAM_PRO + TELEGRAM_VIP
    tg_send_to(ids, msg)
    return jsonify({"ok": True, "sent_to": len(ids)})

# ---------- START/SHUTDOWN ----------
def shutdown(signum=None, frame=None):
    logger.warning("Shutdown signal received; stopping…")
    try: SCHED.shutdown(wait=False)
    except Exception: pass
    time.sleep(0.2); sys.exit(0)

signal.signal(signal.SIGTERM, shutdown)
signal.signal(signal.SIGINT, shutdown)

start_ws_thread(get_symbols())

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT","8000")), debug=(ENV!="production"))
