# live_engine.py — fully featured, with per-channel caps & tally
import os
import re
import time
import json
import queue
import threading
from datetime import datetime, timedelta, date, timezone

import requests
import pandas as pd

from utils import exec_sql, get_config, within_window, TZ, TIMEZONE

# -----------------------------------------------------------------------------
# Telegram wiring
# -----------------------------------------------------------------------------
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()

# Auto-detect all chat ids: TELEGRAM_CHAT_ID_*, e.g. TELEGRAM_CHAT_ID_VIP, TELEGRAM_CHAT_ID_FREE, TELEGRAM_CHAT_ID_SIGNALS1
TELEGRAM_CHAT_KEYS = sorted([k for k in os.environ.keys() if k.startswith("TELEGRAM_CHAT_ID_")])
TELEGRAM_CHATS = {k: os.getenv(k, "").strip() for k in TELEGRAM_CHAT_KEYS if os.getenv(k, "").strip()}

# Optional per-channel caps: TELEGRAM_DAILY_CAP_<SUFFIX>, e.g. TELEGRAM_DAILY_CAP_VIP=150
# Defaults if not set
DEFAULT_DAILY_CAP = int(os.getenv("TELEGRAM_DAILY_CAP_DEFAULT", "250"))  # sane default
DAILY_CAPS = {}
for k in TELEGRAM_CHAT_KEYS:
    suffix = k.replace("TELEGRAM_CHAT_ID_", "", 1)
    cap_key = f"TELEGRAM_DAILY_CAP_{suffix}"
    try:
        DAILY_CAPS[k] = int(os.getenv(cap_key, str(DEFAULT_DAILY_CAP)))
    except Exception:
        DAILY_CAPS[k] = DEFAULT_DAILY_CAP

# Min interval between any two sends (global throttle), in seconds
MIN_INTERVAL_S = int(os.getenv("TELEGRAM_MIN_INTERVAL_S", "3"))
# Optional cool down if API rate-limits occur
COOLDOWN_S = int(os.getenv("TELEGRAM_COOLDOWN_S", "10"))

# -----------------------------------------------------------------------------
# Database (tally + last send)
# -----------------------------------------------------------------------------
exec_sql("""
CREATE TABLE IF NOT EXISTS tally(
  d TEXT,                 -- UTC date YYYY-MM-DD
  chat_key TEXT,          -- env var key, e.g. TELEGRAM_CHAT_ID_VIP
  symbol TEXT,            -- symbol of the signal
  outcome TEXT,           -- 'sent', 'win', 'loss', 'draw'
  ts TEXT                 -- ISO time
)
""")

exec_sql("""
CREATE TABLE IF NOT EXISTS engine_meta(
  k TEXT PRIMARY KEY,
  v TEXT
)
""")

def _meta_get(k, default=None):
    row = exec_sql("SELECT v FROM engine_meta WHERE k=?", (k,), fetch=True)
    if row:
        try:
            return json.loads(row[0][0])
        except Exception:
            return row[0][0]
    return default

def _meta_set(k, v):
    exec_sql("INSERT INTO engine_meta(k,v) VALUES(?,?) ON CONFLICT(k) DO UPDATE SET v=excluded.v", (k, json.dumps(v)))

def _today_utc():
    return datetime.utcnow().date().isoformat()

def _tally_count_today(chat_key):
    d = _today_utc()
    row = exec_sql("SELECT COUNT(*) FROM tally WHERE d=? AND chat_key=? AND outcome='sent'", (d, chat_key), fetch=True)
    return int(row[0][0]) if row else 0

def _tally_bump(chat_key, symbol, outcome):
    exec_sql("INSERT INTO tally(d, chat_key, symbol, outcome, ts) VALUES(?,?,?,?,?)",
             (_today_utc(), chat_key, symbol, outcome, datetime.utcnow().isoformat(timespec="seconds")))

def reset_daily_if_new_day():
    # Nothing to clear—query is by date—but we keep last_date to know when to do weekly recaps, etc.
    last_date = _meta_get("last_date")
    today = _today_utc()
    if last_date != today:
        _meta_set("last_date", today)

# -----------------------------------------------------------------------------
# Telegram helpers
# -----------------------------------------------------------------------------
def _send_telegram_raw(chat_id: str, text: str, disable_web_page_preview=True):
    if not TELEGRAM_BOT_TOKEN:
        return False, "No TELEGRAM_BOT_TOKEN provided"
    try:
        r = requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage",
            json={
                "chat_id": chat_id,
                "text": text,
                "parse_mode": "HTML",
                "disable_web_page_preview": disable_web_page_preview,
            },
            timeout=12,
        )
        ok = r.ok
        data = r.json() if r.headers.get("content-type","").startswith("application/json") else {"text": r.text}
        if not ok:
            return False, f"{r.status_code}: {data}"
        return True, data
    except Exception as e:
        return False, str(e)

def _can_send_to(chat_key: str) -> (bool, str):
    # enforce daily cap
    cap = DAILY_CAPS.get(chat_key, DEFAULT_DAILY_CAP)
    used = _tally_count_today(chat_key)
    if used >= cap:
        return False, f"daily cap reached ({used}/{cap})"
    # enforce global min interval
    last_send = _meta_get("last_send_ts")
    if last_send:
        try:
            last_dt = datetime.fromisoformat(last_send)
            if (datetime.utcnow() - last_dt).total_seconds() < MIN_INTERVAL_S:
                return False, f"global throttle (min interval {MIN_INTERVAL_S}s)"
        except Exception:
            pass
    # enforce trading window (TT)
    cfg = get_config()
    if not within_window(cfg):
        return False, "outside trading window"
    return True, "ok"

def _format_signal(sym, tf, expiry, direction, strategy, extra=""):
    # short, clean message
    line1 = f"📣 <b>{direction}</b> • <code>{sym}</code>"
    line2 = f"⏱ TF={tf} • Exp={expiry} • Strat={strategy}"
    return line1 + "\n" + line2 + (("\n" + extra) if extra else "")

def _audiences_from_cfg(cfg: dict):
    """
    Decide which chat keys to send to.
    Strategy: send to all configured chats by default, but if cfg has tiers, you can filter here.
    """
    # Example: you can gate by user tiers later; for now return all configured
    return list(TELEGRAM_CHATS.keys())

def _send_telegram(msg: str, symbol: str, audience_keys=None):
    """
    Sends a message to a set of chats respecting per-channel caps & global interval.
    Returns (ok, info)
    """
    audience_keys = audience_keys or list(TELEGRAM_CHATS.keys())
    if not audience_keys:
        return False, "No TELEGRAM_CHAT_ID_* configured"

    sent_to = []
    blocked = []
    for chat_key in audience_keys:
        ok, why = _can_send_to(chat_key)
        if not ok:
            blocked.append(f"{chat_key}:{why}")
            continue
        chat_id = TELEGRAM_CHATS[chat_key]
        ok2, info = _send_telegram_raw(chat_id, msg)
        if ok2:
            sent_to.append(chat_key)
            _tally_bump(chat_key, symbol, "sent")
            _meta_set("last_send_ts", datetime.utcnow().isoformat(timespec="seconds"))
            time.sleep(max(0.0, MIN_INTERVAL_S))  # spacing between channels as well
        else:
            blocked.append(f"{chat_key}:ERR {info}")
            # Optional: back off a bit on error
            time.sleep(COOLDOWN_S)

    if sent_to:
        return True, {"sent_to": sent_to, "blocked": blocked}
    return False, {"blocked": blocked}

# -----------------------------------------------------------------------------
# Strategy evaluation stub
# -----------------------------------------------------------------------------
def evaluate_symbols_for_signals(cfg: dict):
    """
    Return a list of signals to send:
    [{"symbol":"frxEURUSD","tf":"M5","expiry":"5m","direction":"BUY","strategy":"TREND", "extra":""}, ...]
    This function should link to your real strategy engine.
    """
    # Hook your real generator here; keep a safe fallback to avoid crashes
    try:
        from strategies import generate_live_signals
        sigs = generate_live_signals(cfg)
        if isinstance(sigs, list):
            return sigs
    except Exception:
        pass
    return []  # no-op fallback

# -----------------------------------------------------------------------------
# Engine (start/stop/status/loop)
# -----------------------------------------------------------------------------
class _Engine:
    def __init__(self):
        self._lock = threading.Lock()
        self._thread = None
        self._running = False
        self.debug = False
        self._last_error = None
        self._last_loop = None
        self._last_reasons = []
        self._sent_count = 0

    # public API used by routes.py
    def start(self):
        with self._lock:
            if self._running:
                return True, "already running"
            self._running = True
            self._thread = threading.Thread(target=self._run, name="live-engine", daemon=True)
            self._thread.start()
            return True, "engine started"

    def stop(self):
        with self._lock:
            self._running = False
        return True, "engine stopped"

    def status(self):
        return {
            "running": self._running,
            "last_error": self._last_error,
            "last_loop": self._last_loop,
            "last_reasons": self._last_reasons[-6:],  # recent notes
            "sent": self._sent_count,
            "debug": self.debug,
        }

    def tally(self):
        """Daily per-channel send counts + simple W/L/D (if you record outcomes elsewhere)."""
        d = _today_utc()
        out = {"date": d, "channels": {}, "wl": {"win":0, "loss":0, "draw":0}}
        # counts per channel
        for chat_key in TELEGRAM_CHATS.keys():
            out["channels"][chat_key] = {
                "sent": _tally_count_today(chat_key),
                "cap": DAILY_CAPS.get(chat_key, DEFAULT_DAILY_CAP),
            }
        # wl summary
        for k in ("win","loss","draw"):
            row = exec_sql("SELECT COUNT(*) FROM tally WHERE d=? AND outcome=?", (d, k), fetch=True)
            out["wl"][k] = int(row[0][0]) if row else 0
        return out

    # core loop
    def _run(self):
        self._last_error = None
        self._last_reasons = []
        cooldown = 2  # loop sleep baseline
        while True:
            with self._lock:
                running = self._running
            if not running:
                break
            try:
                reset_daily_if_new_day()

                cfg = get_config() or {}
                if not within_window(cfg):
                    self._note("outside window")
                    time.sleep(10)
                    continue

                # Collect signals from strategies
                sigs = evaluate_symbols_for_signals(cfg)

                if self.debug:
                    self._note(f"debug=ON, got {len(sigs)} signals")

                # Send each signal to all allowed chats
                for sig in sigs:
                    sym = sig.get("symbol")
                    tf  = (sig.get("tf") or cfg.get("live_tf") or "M5").upper()
                    exp = (sig.get("expiry") or cfg.get("live_expiry") or "5m")
                    direction = (sig.get("direction") or "").upper()
                    strat = sig.get("strategy") or "BASE"
                    if direction not in ("BUY","SELL"):
                        continue
                    msg = _format_signal(sym, tf, exp, direction, strat, sig.get("extra",""))
                    audience = _audiences_from_cfg(cfg)
                    ok, info = _send_telegram(msg, sym, audience_keys=audience)
                    if ok:
                        self._sent_count += 1
                        self._note(f"sent {direction} {sym} -> {info.get('sent_to')}")
                    else:
                        self._note(f"blocked {direction} {sym}: {info}")

                # soft sleep
                time.sleep(cooldown)
                self._last_loop = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
            except Exception as e:
                self._last_error = str(e)
                self._note(f"loop error: {e}")
                time.sleep(COOLDOWN_S)

    def _note(self, s):
        self._last_reasons.append(s)
        if len(self._last_reasons) > 50:
            self._last_reasons = self._last_reasons[-50:]
        if self.debug:
            print("[ENGINE]", s)

ENGINE = _Engine()

# ----- routes.py import helpers expect these ----------------------------------
def tg_test():
    """Used by /telegram/test route."""
    if not TELEGRAM_BOT_TOKEN:
        return False, "No TELEGRAM_BOT_TOKEN provided"
    if not TELEGRAM_CHATS:
        return False, "No TELEGRAM_CHAT_ID_* configured"
    ok, info = _send_telegram("🧪 Test from Pocket Option Signals.", symbol="TEST")
    return ok, info
