# live_engine.py — Live signal loop with tiered per-channel caps & tally
import os, re, time, json, threading
from datetime import datetime
import requests

from utils import exec_sql, get_config, within_window, TZ

# ---------------------------------- Telegram ---------------------------------
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()

# Detect all chats: TELEGRAM_CHAT_ID_*
TELEGRAM_CHAT_KEYS = sorted([k for k in os.environ if k.startswith("TELEGRAM_CHAT_ID_")])
TELEGRAM_CHATS = {k: os.getenv(k, "").strip() for k in TELEGRAM_CHAT_KEYS if os.getenv(k, "").strip()}

# Tier caps by key name
TIER_DEFAULT_CAPS = {"FREE": 3, "BASIC": 6, "PRO": 15, "VIP": None}  # None = unlimited
def _infer_tier_from_key(chat_key: str) -> str:
    k = chat_key.upper()
    for tier in ("FREE","BASIC","PRO","VIP"):
        if re.search(rf"(?:^|_){tier}(?:_|$)", k): return tier
    if "VIP" in k: return "VIP"
    if "PRO" in k: return "PRO"
    if "BASIC" in k: return "BASIC"
    return "FREE"

# Build per-channel caps with optional overrides TELEGRAM_DAILY_CAP_<SUFFIX>
DAILY_CAPS = {}
for chat_key in TELEGRAM_CHAT_KEYS:
    suffix = chat_key.replace("TELEGRAM_CHAT_ID_", "", 1)
    ov_key = f"TELEGRAM_DAILY_CAP_{suffix}"
    raw = os.getenv(ov_key, "").strip()
    if raw != "":
        if raw.lower() in ("none","unlimited","inf","infinite"):
            DAILY_CAPS[chat_key] = None
        else:
            try: DAILY_CAPS[chat_key] = int(raw)
            except: DAILY_CAPS[chat_key] = TIER_DEFAULT_CAPS[_infer_tier_from_key(chat_key)]
    else:
        DAILY_CAPS[chat_key] = TIER_DEFAULT_CAPS[_infer_tier_from_key(chat_key)]

MIN_INTERVAL_S = int(os.getenv("TELEGRAM_MIN_INTERVAL_S", "3"))
COOLDOWN_S    = int(os.getenv("TELEGRAM_COOLDOWN_S", "10"))

# ---------------------------------- Storage ----------------------------------
exec_sql("""CREATE TABLE IF NOT EXISTS tally(
  d TEXT, chat_key TEXT, symbol TEXT, outcome TEXT, ts TEXT)""")
exec_sql("""CREATE TABLE IF NOT EXISTS engine_meta(
  k TEXT PRIMARY KEY, v TEXT)""")

def _meta_get(k, default=None):
    row = exec_sql("SELECT v FROM engine_meta WHERE k=?", (k,), fetch=True)
    if row:
        try: return json.loads(row[0][0])
        except: return row[0][0]
    return default

def _meta_set(k, v):
    exec_sql("INSERT INTO engine_meta(k,v) VALUES(?,?) ON CONFLICT(k) DO UPDATE SET v=excluded.v",
             (k, json.dumps(v)))

def _today_utc(): return datetime.utcnow().date().isoformat()
def _tally_count_today(chat_key):
    row = exec_sql("SELECT COUNT(*) FROM tally WHERE d=? AND chat_key=? AND outcome='sent'",
                   (_today_utc(), chat_key), fetch=True)
    return int(row[0][0]) if row else 0
def _tally_bump(chat_key, symbol, outcome):
    exec_sql("INSERT INTO tally(d,chat_key,symbol,outcome,ts) VALUES(?,?,?,?,?)",
             (_today_utc(), chat_key, symbol, outcome, datetime.utcnow().isoformat(timespec="seconds")))

# --------------------------------- Telegram API -------------------------------
def _send_telegram_raw(chat_id: str, text: str):
    if not TELEGRAM_BOT_TOKEN: return False, "No TELEGRAM_BOT_TOKEN"
    try:
        r = requests.post(f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage",
                          json={"chat_id": chat_id, "text": text, "parse_mode":"HTML",
                                "disable_web_page_preview": True}, timeout=12)
        if not r.ok:
            return False, f"{r.status_code}: {r.text}"
        return True, r.json()
    except Exception as e:
        return False, str(e)

def _can_send_to(chat_key: str):
    cap = DAILY_CAPS.get(chat_key, TIER_DEFAULT_CAPS[_infer_tier_from_key(chat_key)])
    if cap is not None:
        used = _tally_count_today(chat_key)
        if used >= cap:
            return False, f"daily cap {used}/{cap}"
    last = _meta_get("last_send_ts")
    if last:
        try:
            from datetime import datetime as _dt
            if (_dt.utcnow() - _dt.fromisoformat(last)).total_seconds() < MIN_INTERVAL_S:
                return False, f"throttle {MIN_INTERVAL_S}s"
        except: pass
    cfg = get_config()
    if not within_window(cfg): return False, "outside window"
    return True, "ok"

def _format_signal(sym, tf, expiry, direction, strategy, extra=""):
    return (f"📣 <b>{direction}</b> • <code>{sym}</code>\n"
            f"⏱ TF={tf} • Exp={expiry} • Strat={strategy}"
            + (f"\n{extra}" if extra else ""))

def _audiences_from_cfg(cfg):  # can later filter by tier
    return list(TELEGRAM_CHATS.keys())

def _send_telegram(msg: str, symbol: str, audience_keys=None):
    audience_keys = audience_keys or list(TELEGRAM_CHATS.keys())
    if not audience_keys: return False, "no chats configured"
    sent_to, blocked = [], []
    for chat_key in audience_keys:
        ok, why = _can_send_to(chat_key)
        if not ok:
            blocked.append(f"{chat_key}:{why}"); continue
        ok2, info = _send_telegram_raw(TELEGRAM_CHATS[chat_key], msg)
        if ok2:
            sent_to.append(chat_key)
            _tally_bump(chat_key, symbol, "sent")
            _meta_set("last_send_ts", datetime.utcnow().isoformat(timespec="seconds"))
            time.sleep(max(0.0, MIN_INTERVAL_S))
        else:
            blocked.append(f"{chat_key}:ERR {info}")
            time.sleep(COOLDOWN_S)
    return (True, {"sent_to": sent_to, "blocked": blocked}) if sent_to else (False, {"blocked": blocked})

# ------------------------------ Strategy adapter ------------------------------
def evaluate_symbols_for_signals(cfg: dict):
    """Call your generator, fallback empty list."""
    try:
        from strategies import generate_live_signals
        sigs = generate_live_signals(cfg)
        if isinstance(sigs, list): return sigs
    except Exception:
        pass
    return []

# ----------------------------------- Engine ----------------------------------
class _Engine:
    def __init__(self):
        self._lock = threading.Lock()
        self._running = False
        self._thread = None
        self.debug = False
        self._last_error = None
        self._last_loop = None
        self._last_reasons = []
        self._sent = 0

    def start(self):
        with self._lock:
            if self._running: return True, "already running"
            self._running = True
            self._thread = threading.Thread(target=self._run, daemon=True, name="live-engine")
            self._thread.start()
            return True, "engine started"

    def stop(self):
        with self._lock: self._running = False
        return True, "engine stopped"

    def status(self):
        return {"running": self._running, "last_error": self._last_error,
                "last_loop": self._last_loop, "last_reasons": self._last_reasons[-6:],
                "sent": self._sent, "debug": self.debug}

    def tally(self):
        d = _today_utc()
        out = {"date": d, "channels": {}, "wl": {"win":0, "loss":0, "draw":0}}
        for chat_key in TELEGRAM_CHATS:
            out["channels"][chat_key] = {"sent": _tally_count_today(chat_key),
                                         "cap": DAILY_CAPS.get(chat_key)}
        for k in ("win","loss","draw"):
            row = exec_sql("SELECT COUNT(*) FROM tally WHERE d=? AND outcome=?", (d,k), fetch=True)
            out["wl"][k] = int(row[0][0]) if row else 0
        return out

    def _note(self, s):
        self._last_reasons.append(s)
        if len(self._last_reasons) > 50: self._last_reasons = self._last_reasons[-50:]
        if self.debug: print("[ENGINE]", s)

    def _run(self):
        while True:
            with self._lock: run = self._running
            if not run: break
            try:
                cfg = get_config() or {}
                if not within_window(cfg):
                    self._note("outside window"); time.sleep(10); continue

                sigs = evaluate_symbols_for_signals(cfg)
                if self.debug: self._note(f"got {len(sigs)} signals")

                for s in sigs:
                    sym = s.get("symbol")
                    tf  = (s.get("tf") or cfg.get("live_tf") or "M5").upper()
                    exp = (s.get("expiry") or cfg.get("live_expiry") or "5m")
                    direction = (s.get("direction") or "").upper()
                    strat = s.get("strategy") or "BASE"
                    if direction not in ("BUY","SELL"): continue
                    msg = _format_signal(sym, tf, exp, direction, strat, s.get("extra",""))
                    ok, info = _send_telegram(msg, sym, audience_keys=_audiences_from_cfg(cfg))
                    if ok:
                        self._sent += 1
                        self._note(f"sent {direction} {sym} -> {info.get('sent_to')}")
                    else:
                        self._note(f"blocked {direction} {sym}: {info}")

                time.sleep(2)
                self._last_loop = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
            except Exception as e:
                self._last_error = str(e)
                self._note(f"loop error: {e}")
                time.sleep(COOLDOWN_S)

ENGINE = _Engine()

def tg_test():
    if not TELEGRAM_BOT_TOKEN: return False, "No TELEGRAM_BOT_TOKEN"
    if not TELEGRAM_CHATS: return False, "No TELEGRAM_CHAT_ID_*"
    ok, info = _send_telegram("🧪 Test from Pocket Option Signals.", "TEST")
    return ok, info
