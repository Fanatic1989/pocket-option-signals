# utils.py
import os, json, sqlite3, threading
from datetime import datetime, timedelta
import pytz
from apscheduler.schedulers.background import BackgroundScheduler

TIMEZONE = os.getenv("TIMEZONE", "America/Port_of_Spain")
TZ = pytz.timezone(TIMEZONE)
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///data.db")
IS_SQLITE = DATABASE_URL.startswith("sqlite:///")
SQLITE_PATH = DATABASE_URL.replace("sqlite:///", "") if IS_SQLITE else None
LOCK = threading.Lock()
scheduler = BackgroundScheduler(timezone=TIMEZONE)

DEFAULT_CONFIG = {
    "window": {"start":"08:00","end":"17:00","timezone":TIMEZONE},
    "strategies":{"BASE":{"enabled":True},"TREND":{"enabled":True},"CHOP":{"enabled":True},"CUSTOM":{"enabled":False}},
    "indicators": {},
    "custom": {"enabled": False, "mode":"SIMPLE", "simple_buy":"", "simple_sell":"", "buy_rule":"", "sell_rule":"", "tol_pct":0.1, "lookback":3},
    "engine":{"symbol":"SYNTH","lookback":400,"cadence_sec":60,"tf":"M5","expiry":"5m"},
    "tiers":{"free":{"daily_limit":10},"premium":{"daily_limit":50},"vip":{"daily_limit":None}}
}

def get_conn():
    if IS_SQLITE:
        os.makedirs(os.path.dirname(SQLITE_PATH or "data.db") or ".", exist_ok=True)
        conn = sqlite3.connect(SQLITE_PATH or "data.db", check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn
    import psycopg2
    return psycopg2.connect(DATABASE_URL)

def exec_sql(sql, params=(), fetch=False):
    with LOCK:
        conn = get_conn()
        try:
            cur = conn.cursor()
            cur.execute(sql, params)
            rows = cur.fetchall() if fetch else None
            conn.commit()
            return rows
        finally:
            conn.close()

def init_db():
    exec_sql("""CREATE TABLE IF NOT EXISTS users (id INTEGER PRIMARY KEY AUTOINCREMENT,
              telegram_id TEXT UNIQUE, username TEXT, tier TEXT DEFAULT 'free', expires_at TEXT)""")
    exec_sql("""CREATE TABLE IF NOT EXISTS quotas (id INTEGER PRIMARY KEY AUTOINCREMENT,
              telegram_id TEXT, date_key TEXT, signals_sent INTEGER DEFAULT 0, UNIQUE(telegram_id, date_key))""")
    exec_sql("""CREATE TABLE IF NOT EXISTS config (id INTEGER PRIMARY KEY AUTOINCREMENT, key TEXT UNIQUE, value TEXT)""")
    exec_sql("""CREATE TABLE IF NOT EXISTS signals (id INTEGER PRIMARY KEY AUTOINCREMENT, created_at TEXT,
              strategy TEXT, direction TEXT, price REAL, meta TEXT)""")
    exec_sql("""CREATE TABLE IF NOT EXISTS live_trades (id INTEGER PRIMARY KEY AUTOINCREMENT, created_at TEXT,
              strategy TEXT, direction TEXT, entry REAL, tf_minutes INTEGER, expiry_minutes INTEGER,
              symbol TEXT, resolved_at TEXT, outcome TEXT, meta TEXT)""")
    exec_sql("""CREATE TABLE IF NOT EXISTS summaries_sent (id INTEGER PRIMARY KEY AUTOINCREMENT, date_key TEXT, type TEXT,
              UNIQUE(date_key,type))""")
    exec_sql("""CREATE TABLE IF NOT EXISTS logs (id INTEGER PRIMARY KEY AUTOINCREMENT, created_at TEXT, level TEXT, message TEXT)""")

def get_config():
    rows = exec_sql("SELECT value FROM config WHERE key=?", ("app",), fetch=True)
    if not rows: return json.loads(json.dumps(DEFAULT_CONFIG))
    try: return json.loads(rows[0][0])
    except: return json.loads(json.dumps(DEFAULT_CONFIG))

def set_config(cfg): exec_sql("INSERT OR REPLACE INTO config(key,value) VALUES(?,?)", ("app", json.dumps(cfg)))

def ensure_config():
    cfg = get_config()
    from indicators import INDICATOR_SPECS
    if not cfg.get("indicators"):
        cfg["indicators"] = {k: {"enabled": False, **v["params"]} for k,v in INDICATOR_SPECS.items()}
        for k in ("EMA","RSI","ADX","BB"):
            if k in cfg["indicators"]: cfg["indicators"][k]["enabled"] = True
        set_config(cfg)

def log(level, message):
    now = datetime.now(TZ).strftime("%Y-%m-%d %H:%M:%S")
    exec_sql("INSERT INTO logs(created_at, level, message) VALUES(?,?,?)", (now, level, message))

def today_key(): return datetime.now(TZ).strftime("%Y-%m-%d")
def week_key(dt): return f"{dt.isocalendar().year}-W{dt.isocalendar().week:02d}"

def within_window(cfg=None):
    cfg = cfg or get_config()
    import pytz
    tz = pytz.timezone(cfg["window"]["timezone"])
    now = datetime.now(tz)
    sh, sm = map(int, cfg["window"]["start"].split(":"))
    eh, em = map(int, cfg["window"]["end"].split(":"))
    start = now.replace(hour=sh, minute=sm, second=0, microsecond=0)
    end = now.replace(hour=eh, minute=em, second=0, microsecond=0)
    return start <= now <= end

def upsert_quota(telegram_id, inc=0):
    dk = today_key()
    rows = exec_sql("SELECT signals_sent FROM quotas WHERE telegram_id=? AND date_key=?", (telegram_id, dk), fetch=True)
    sent = rows[0][0] if rows else 0
    if not rows: exec_sql("INSERT INTO quotas(telegram_id, date_key, signals_sent) VALUES(?,?,?)", (telegram_id, dk, 0))
    if inc:
        sent += inc
        exec_sql("UPDATE quotas SET signals_sent=? WHERE telegram_id=? AND date_key=?", (sent, telegram_id, dk))
    return sent

def get_user_tier_limit(tier, cfg=None):
    cfg = cfg or get_config()
    return cfg["tiers"].get(tier,{}).get("daily_limit")

def tier_ok_to_send(telegram_id, tier, cfg=None):
    limit = get_user_tier_limit(tier, cfg)
    if limit is None: return True
    return upsert_quota(telegram_id, 0) < limit

def schedule_jobs():
    if not scheduler.running: scheduler.start()
