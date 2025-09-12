# live_engine.py — Telegram wiring + live loop + daily caps (FREE=3, BASIC=6, PRO=15, VIP=∞)
from __future__ import annotations

import os, json, time, threading
from datetime import datetime, date, timezone
from typing import Optional, Dict, Any, Tuple

import requests
from utils import get_chat_id_for_tier, telegram_send_message

# ------------------ Caps ------------------
DAILY_CAPS: Dict[str, Optional[int]] = {
    "free":  3,
    "basic": 6,
    "pro":   15,
    "vip":   None,   # unlimited
}

BOT_TOKEN = (os.getenv("TELEGRAM_BOT_TOKEN") or "").strip()

# ------------------ Low-level sender ------------------
def _send_telegram_tier(text: str, tier: str) -> Tuple[bool, Dict[str, Any]]:
    if not BOT_TOKEN:
        return False, {"ok": False, "error": "Missing TELEGRAM_BOT_TOKEN"}
    chat_id = get_chat_id_for_tier(tier)
    if not chat_id:
        return False, {"ok": False, "error": f"No chat configured for tier '{tier}'"}
    js = telegram_send_message(chat_id, text)
    return bool(js.get("ok")), js

def tg_test_all() -> Tuple[bool, str]:
    """Send a small test to every configured tier. Returns (ok,json_string)."""
    results: Dict[str, Any] = {}
    for t in ("free","basic","pro","vip"):
        chat_ok = bool(get_chat_id_for_tier(t))
        if chat_ok:
            ok, info = _send_telegram_tier(f"🧪 Bot test ({t.upper()})", t)
            results[t] = {"ok": ok, "info": info}
        else:
            results[t] = {"ok": False, "info": "not configured"}
    return any(v["ok"] for v in results.values()), json.dumps(results, ensure_ascii=False)

# ------------------ Live Engine ------------------
class LiveEngine:
    """
    - start/stop thread
    - per-tier daily tallies with cap enforcement
    - debug flag
    - last_send_result exposed
    """
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self.debug = False
        self._sleep_seconds = int(os.getenv("ENGINE_LOOP_SLEEP", "4"))

        today = date.today().isoformat()
        self._tally = {"date": today, "by_tier": {"free": 0, "basic": 0, "pro": 0, "vip": 0}, "total": 0}
        self.last_send_result: Dict[str, Any] = {}
        self.last_error: Optional[str] = None

    # ----- helpers
    def _maybe_reset_tallies(self):
        with self._lock:
            today = date.today().isoformat()
            if self._tally.get("date") != today:
                self._tally = {"date": today, "by_tier": {"free": 0, "basic": 0, "pro": 0, "vip": 0}, "total": 0}

    def _inc_tally(self, tier: str):
        with self._lock:
            self._tally["by_tier"][tier] += 1
            self._tally["total"] += 1

    def _cap_reached(self, tier: str) -> bool:
        cap = DAILY_CAPS.get(tier)
        if cap is None:  # unlimited
            return False
        with self._lock:
            return self._tally["by_tier"].get(tier, 0) >= cap

    # ----- API
    def start(self) -> Tuple[bool, str]:
        with self._lock:
            if self._running:
                return True, "already running"
            self._running = True
            self._thread = threading.Thread(target=self._loop, name="live-engine", daemon=True)
            self._thread.start()
        return True, "started"

    def stop(self) -> Tuple[bool, str]:
        with self._lock:
            if not self._running:
                return True, "already stopped"
            self._running = False
        for _ in range(40):
            t = self._thread
            if not t or not t.is_alive():
                break
            time.sleep(0.05)
        return True, "stopped"

    def set_debug(self, value: bool) -> None:
        with self._lock:
            self.debug = bool(value)

    def status(self) -> Dict[str, Any]:
        self._maybe_reset_tallies()
        with self._lock:
            return {
                "running": self._running,
                "loop_sleep": self._sleep_seconds,
                "tally": json.loads(json.dumps(self._tally)),
                "debug": self.debug,
                "caps": DAILY_CAPS,
                "configured_chats": {
                    "free":  bool(get_chat_id_for_tier("free")),
                    "basic": bool(get_chat_id_for_tier("basic")),
                    "pro":   bool(get_chat_id_for_tier("pro")),
                    "vip":   bool(get_chat_id_for_tier("vip")),
                },
                "last_send_result": self.last_send_result,
                "token_masked": (BOT_TOKEN[:9] + "..." + BOT_TOKEN[-6:]) if BOT_TOKEN else "",
            }

    def send_signal(self, tier: str, text: str) -> Tuple[bool, str]:
        self._maybe_reset_tallies()
        t = (tier or "vip").lower()
        if t not in ("free","basic","pro","vip"):
            return False, f"unknown tier: {t}"
        if not get_chat_id_for_tier(t):
            return False, f"tier {t} not configured"
        if self._cap_reached(t):
            return False, f"daily cap reached for {t} ({DAILY_CAPS.get(t)})"
        msg = text if not self.debug else f"[DEBUG]\n{datetime.now(timezone.utc).isoformat(timespec='seconds')}Z\n\n{text}"
        ok, res = _send_telegram_tier(msg, t)
        with self._lock:
            self.last_send_result = res
        if ok:
            self._inc_tally(t)
            return True, "sent"
        return False, res.get("description") or res.get("error") or "send failed"

    def send_raw(self, tier: str, text: str) -> Dict[str, Any]:
        """Bypass caps (for your ‘Send Test’ buttons)."""
        ok, res = _send_telegram_tier(text, (tier or "vip").lower())
        with self._lock:
            self.last_send_result = res
        return res

    # compat for older code
    def send_to_tier(self, tier: str, text: str) -> Dict[str, Any]:
        ok, reason = self.send_signal(tier, text)
        if ok:
            d = dict(self.last_send_result) if isinstance(self.last_send_result, dict) else {}
            d.setdefault("ok", True)
            d.setdefault("send_result", "sent")
            return d
        return {"ok": False, "error": reason}

    # ----- loop
    def _loop(self):
        while True:
            with self._lock:
                running = self._running
            if not running:
                break
            self._maybe_reset_tallies()
            if self.debug:
                print("[ENGINE] tick", datetime.utcnow().isoformat()+"Z", "| tally:", self._tally)
            time.sleep(max(1, self._sleep_seconds))

# Singleton for routes
ENGINE = LiveEngine()
