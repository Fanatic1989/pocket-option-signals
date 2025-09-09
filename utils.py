import os, json, sqlite3, threading
from datetime import datetime, time as dtime
import pytz

TIMEZONE = os.getenv("TZ_NAME", "America/Port_of_Spain")
TZ = pytz.timezone(TIMEZONE)

_CFG_PATH = os.getenv("CFG_PATH", "/tmp/pos_config.json")
_cfg_lock = threading.Lock()

_DEFAULT_CFG = {
    "window": {"start": "08:00", "end": "17:00", "timezone": TIMEZONE},
    "strategies": {"BASE":{"enabled": True}, "TREND":{"enabled": False}, "CHOP":{"enabled": False}},
    "indicators": {},
    "symbols": [],
    "custom": {},
    "custom1": {"enabled": False, "mode":"SIMPLE", "simple_buy":"", "simple_sell":"", "buy_rule":"", "sell_rule":"", "tol_pct":0.1, "lookback":3},
    "custom2": {"enabled": False, "mode":"SIMPLE", "simple_buy":"", "simple_sell":"", "buy_rule":"", "sell_rule":"", "tol_pct":0.1, "lookback":3},
    "custom3": {"enabled": False, "mode":"SIMPLE", "simple_buy":"", "simple_sell":"", "buy_rule":"", "sell_rule":"", "tol_pct":0.1, "lookback":3},
    "live_tf": "M5",
    "live_expiry": "5m"
}

def _ensure_cfg_file():
    if not os.path.exists(_CFG_PATH):
        with open(_CFG_PATH, "w") as f:
            json.dump(_DEFAULT_CFG, f)

def get_config():
    with _cfg_lock:
        _ensure_cfg_file()
        with open(_CFG_PATH, "r") as f:
            cfg = json.load(f)
        # ensure backward-compat defaults
        for k,v in _DEFAULT_CFG.items():
            if k not in cfg: cfg[k]=v
        return cfg

def set_config(cfg: dict):
    with _cfg_lock:
        with open(_CFG_PATH, "w") as f:
            json.dump(cfg, f)

def within_window(cfg: dict) -> bool:
    try:
        win = cfg.get("window", {})
        tz = pytz.timezone(win.get("timezone", TIMEZONE))
        now = datetime.now(tz).time()
        s = dtime.fromisoformat(win.get("start","08:00"))
        e = dtime.fromisoformat(win.get("end","17:00"))
        if s <= e:
            return s <= now <= e
        # window across midnight
        return now >= s or now <= e
    except Exception:
        return True

def log(level, msg):
    print(f"[{level}] {msg}", flush=True)

_DB_PATH = os.getenv("DB_PATH", "/tmp/pos_users.db")

def exec_sql(sql, params=(), fetch=False):
    try:
        conn = sqlite3.connect(_DB_PATH)
        cur = conn.cursor()
        cur.execute("CREATE TABLE IF NOT EXISTS users(telegram_id TEXT PRIMARY KEY, tier INTEGER, expires_at TEXT)")
        cur.execute(sql, params)
        rows = cur.fetchall() if fetch else None
        conn.commit()
        conn.close()
        return rows
    except Exception as e:
        log("WARN", f"exec_sql failed: {e}")
        return [] if fetch else None
