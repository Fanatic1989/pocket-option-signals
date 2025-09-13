# live_engine.py — Telegram + Engine + Caps + Trading window guard
from __future__ import annotations
import os, json, time, threading, requests
from datetime import datetime, date, time as dtime, timezone
from typing import Dict, Optional, Tuple, Any

# ----------------- Telegram config -----------------
BOT_TOKEN = (os.getenv("TELEGRAM_BOT_TOKEN") or "").strip()

def _first_nonempty(*names: str) -> str:
    for n in names:
        v = os.getenv(n, "").strip()
        if v:
            return v
    return ""

CHAT_FREE  = _first_nonempty("TELEGRAM_CHAT_ID_FREE",  "TELEGRAM_CHAT_FREE")
CHAT_BASIC = _first_nonempty("TELEGRAM_CHAT_ID_BASIC", "TELEGRAM_CHAT_BASIC")
CHAT_PRO   = _first_nonempty("TELEGRAM_CHAT_ID_PRO",   "TELEGRAM_CHAT_PRO")
CHAT_VIP   = _first_nonempty("TELEGRAM_CHAT_ID_VIP",   "TELEGRAM_CHAT_VIP")
FALLBACK_CHAT = os.getenv("TELEGRAM_CHAT_ID", "").strip()

TIER_TO_CHAT: Dict[str, str] = {
    "free":  CHAT_FREE  or FALLBACK_CHAT,
    "basic": CHAT_BASIC or FALLBACK_CHAT,
    "pro":   CHAT_PRO   or FALLBACK_CHAT,
    "vip":   CHAT_VIP   or FALLBACK_CHAT,
}

DAILY_CAPS: Dict[str, Optional[int]] = {
    "free":  3,
    "basic": 6,
    "pro":   15,
    "vip":   None,  # unlimited
}

# ----------------- Trading window (Mon–Fri 08:00–17:00 local) -----------------
APP_TZ = os.getenv("APP_TZ", "America/Port_of_Spain")

def _get_tz():
    try:
        import pytz
        return pytz.timezone(APP_TZ)
    except Exception:
        from zoneinfo import ZoneInfo
        return ZoneInfo(APP_TZ)

TZ = _get_tz()

def trading_open_now() -> bool:
    # Weekend guard
    now_local = datetime.now(TZ)
    if now_local.weekday() >= 5:  # 5=Sat, 6=Sun
        return False
    # Window guard (env overrides optional)
    start_s = os.getenv("TRADING_START", "08:00")
    end_s   = os.getenv("TRADING_END", "17:00")
    try:
        s_h, s_m = [int(x) for x in start_s.split(":")[:2]]
        e_h, e_m = [int(x) for x in end_s.split(":")[:2]]
    except Exception:
        s_h, s_m, e_h, e_m = 8, 0, 17, 0
    now_t = now_local.time()
    return dtime(s_h, s_m) <= now_t <= dtime(e_h, e_m)

# ----------------- Low-level Telegram -----------------
def _send_telegram(text: str, tier: str) -> Tuple[bool, Dict[str, Any]]:
    if not BOT_TOKEN:
        return False, {"ok": False, "error": "Missing TELEGRAM_BOT_TOKEN"}
    t = (tier or "vip").lower()
    chat_id = TIER_TO_CHAT.get(t)
    if not chat_id:
        return False, {"ok": False, "error": f"No chat id for tier '{t}'"}
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": text,
        "parse_mode": "HTML",
        "disable_web_page_preview": True,
    }
    try:
        r = requests.post(url, json=payload, timeout=12)
        js = {}
        try: js = r.json()
        except Exception: pass
        ok = r.ok and js.get("ok") is True
        return ok, (js if js else {"ok": ok, "status_code": r.status_code})
    except Exception as e:
        return False, {"ok": False, "error": f"{type(e).__name__}: {e}"}

def _get_me() -> Dict[str, Any]:
    if not BOT_TOKEN:
        return {"ok": False, "error": "Missing TELEGRAM_BOT_TOKEN"}
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/getMe"
    try:
        r = requests.get(url, timeout=10)
        return r.json()
    except Exception as e:
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}

def tg_test() -> Tuple[bool, Dict[str, Any]]:
    """
    Per your requirement: test message only to VIP.
    Returns (ok, diagnostic_json_like_dict)
    """
    getme = _get_me()
    token_masked = (BOT_TOKEN[:9] + "..." + BOT_TOKEN[-6:]) if BOT_TOKEN else ""
    configured = {k: bool(v) for k, v in TIER_TO_CHAT.items()}
    ok, send_res = _send_telegram("🧪 Test message (VIP only)", "vip")
    diag = {
        "configured_chats": configured,
        "getMe": getme,
        "ok": ok,
        "send_result": "sent" if ok else (send_res.get("error") or "failed"),
        "token_masked": token_masked
    }
    return ok, diag

# ----------------- Live engine -----------------
class LiveEngine:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self.debug = False

        today = date.today().isoformat()
        self._tally = {"date": today, "by_tier": {"free": 0, "basic": 0, "pro": 0, "vip": 0}, "total": 0}
        self._sleep_seconds = int(os.getenv("ENGINE_LOOP_SLEEP", "4"))
        self.last_send = {}    # last send response
        self.last_error = None

    # helpers
    def _maybe_reset_tallies(self) -> None:
        with self._lock:
            today = date.today().isoformat()
            if self._tally.get("date") != today:
                self._tally = {"date": today, "by_tier": {"free": 0, "basic": 0, "pro": 0, "vip": 0}, "total": 0}

    def _cap_reached(self, tier: str) -> bool:
        cap = DAILY_CAPS.get(tier)
        if cap is None:  # unlimited
            return False
        return self._tally["by_tier"].get(tier, 0) >= cap

    def _inc_tally(self, tier: str) -> None:
        self._tally["by_tier"][tier] += 1
        self._tally["total"] += 1

    # public
    def start(self) -> Tuple[bool, str]:
        with self._lock:
            if self._running: return True, "already running"
            self._running = True
            self._thread = threading.Thread(target=self._loop, name="live-engine", daemon=True)
            self._thread.start()
            return True, "started"

    def stop(self) -> Tuple[bool, str]:
        with self._lock:
            if not self._running: return True, "already stopped"
            self._running = False
        return True, "stopped"

    def set_debug(self, value: bool) -> None:
        self.debug = bool(value)

    def send_signal(self, tier: str, text: str) -> Dict[str, Any]:
        """
        Enforces trading window and daily caps.
        """
        self._maybe_reset_tallies()
        t = (tier or "vip").lower()
        if t not in ("free","basic","pro","vip"):
            return {"ok": False, "error": f"unknown tier '{t}'"}
        if not TIER_TO_CHAT.get(t):
            return {"ok": False, "error": f"tier '{t}' not configured"}

        if not trading_open_now():
            return {"ok": False, "error": "trading window closed (Mon–Fri 08:00–17:00 local)"}

        if self._cap_reached(t):
            return {"ok": False, "error": f"daily cap reached for {t}"}

        msg = text
        if self.debug:
            stamp = datetime.now(timezone.utc).isoformat(timespec="seconds") + "Z"
            msg = f"[DEBUG]\n{stamp}\n\n{text}"

        ok, res = _send_telegram(msg, t)
        self.last_send = res
        if ok:
            self._inc_tally(t)
        return res | {"ok": ok}

    def status(self) -> Dict[str, Any]:
        self._maybe_reset_tallies()
        return {
            "running": self._running,
            "loop_sleep": self._sleep_seconds,
            "tally": self._tally,
            "debug": self.debug,
            "caps": DAILY_CAPS,
            "configured_chats": {k: bool(v) for k, v in TIER_TO_CHAT.items()},
            "last_send_result": self.last_send,
            "day": self._tally["date"],
        }

    # loop
    def _loop(self) -> None:
        while True:
            with self._lock:
                if not self._running: break
            self._maybe_reset_tallies()
            if self.debug:
                print(f"[ENGINE] tick {datetime.now(timezone.utc).isoformat()}Z | {self._tally}")
            time.sleep(max(1, self._sleep_seconds))

# singleton
ENGINE = LiveEngine()
