# live_engine.py — Telegram wiring + live loop + daily caps + exports
from __future__ import annotations

import os
import json
import time
import threading
from datetime import datetime, date, timezone
from typing import Dict, Optional, Tuple, Any

import requests

# ===================== Telegram configuration ================================

BOT_TOKEN = (os.getenv("TELEGRAM_BOT_TOKEN") or "").strip()

def _first_nonempty(*names: str) -> str:
    for n in names:
        v = (os.getenv(n) or "").strip()
        if v:
            return v
    return ""

# Support BOTH legacy TELEGRAM_CHAT_ID_* and new TELEGRAM_CHAT_* (any may be set).
CHAT_FREE  = _first_nonempty("TELEGRAM_CHAT_FREE",  "TELEGRAM_CHAT_ID_FREE")
CHAT_BASIC = _first_nonempty("TELEGRAM_CHAT_BASIC", "TELEGRAM_CHAT_ID_BASIC")
CHAT_PRO   = _first_nonempty("TELEGRAM_CHAT_PRO",   "TELEGRAM_CHAT_ID_PRO")
CHAT_VIP   = _first_nonempty("TELEGRAM_CHAT_VIP",   "TELEGRAM_CHAT_ID_VIP")

# Optional single fallback chat if a tier is unset
FALLBACK_CHAT = (os.getenv("TELEGRAM_CHAT_ID") or "").strip()

# >>> Exported mapping expected by app/routes/templates
TIER_TO_CHAT: Dict[str, str] = {
    "free":  CHAT_FREE  or FALLBACK_CHAT,
    "basic": CHAT_BASIC or FALLBACK_CHAT,
    "pro":   CHAT_PRO   or FALLBACK_CHAT,
    "vip":   CHAT_VIP   or FALLBACK_CHAT,
}

# >>> Exported caps (VIP unlimited)
DAILY_CAPS: Dict[str, Optional[int]] = {
    "free":  3,
    "basic": 6,
    "pro":   15,
    "vip":   None,  # unlimited
}

# ===================== Low-level Telegram sender =============================

def _send_message(chat_id: str, text: str) -> Dict[str, Any]:
    if not BOT_TOKEN:
        return {"ok": False, "error": "Missing TELEGRAM_BOT_TOKEN"}
    if not chat_id:
        return {"ok": False, "error": "Missing chat_id"}

    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    try:
        r = requests.post(
            url,
            json={
                "chat_id": chat_id,
                "text": text,
                "parse_mode": "HTML",
                "disable_web_page_preview": True,
            },
            timeout=15,
        )
        js = r.json() if r.headers.get("content-type","").startswith("application/json") \
             else {"ok": r.ok, "text": r.text}
        js["_http"] = r.status_code
        return js
    except Exception as e:
        return {"ok": False, "error": str(e)}

def _send_telegram(text: str, tier: str = "vip") -> Dict[str, Any]:
    t = (tier or "vip").lower()
    chat_id = TIER_TO_CHAT.get(t, "")
    return _send_message(chat_id, text)

# Public helper used by dashboard “Send Test”
def tg_test_all() -> Tuple[bool, str]:
    results: Dict[str, Any] = {}
    if not BOT_TOKEN:
        return False, json.dumps({"error": "Missing TELEGRAM_BOT_TOKEN"})

    if not any(TIER_TO_CHAT.values()):
        return False, json.dumps({"error": "No Telegram channels configured"})

    for t in ("free", "basic", "pro", "vip"):
        if TIER_TO_CHAT.get(t):
            res = _send_telegram(f"🧪 Test from bot ({t.upper()})", t)
            results[t] = res
        else:
            results[t] = {"ok": False, "error": "not configured"}

    try:
        info = json.dumps(results)
    except Exception:
        info = str(results)
    overall = any(v.get("ok") for v in results.values())
    return overall, info

# ===================== Live Engine ===========================================

class LiveEngine:
    """
    Minimal live loop with:
      - start/stop
      - per-tier daily tallies with cap enforcement
      - debug flag
      - thread-safe counters
    Use ENGINE.send_to_tier(tier, text) from your strategy code.
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

    # ---------- Internal helpers ----------
    def _maybe_reset_tallies(self) -> None:
        with self._lock:
            today = date.today().isoformat()
            if self._tally.get("date") != today:
                self._tally = {"date": today, "by_tier": {"free": 0, "basic": 0, "pro": 0, "vip": 0}, "total": 0}

    def _inc_tally(self, tier: str) -> None:
        with self._lock:
            self._tally["by_tier"][tier] += 1
            self._tally["total"] += 1

    def _cap_reached(self, tier: str) -> bool:
        cap = DAILY_CAPS.get(tier)
        if cap is None:
            return False
        with self._lock:
            return self._tally["by_tier"].get(tier, 0) >= cap

    # ---------- Public API ----------
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
        # brief wait for thread to exit
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
            # Back/forward compatible payload:
            tallies = {
                "free": self._tally["by_tier"]["free"],
                "basic": self._tally["by_tier"]["basic"],
                "pro": self._tally["by_tier"]["pro"],
                "vip": self._tally["by_tier"]["vip"],
                "all": self._tally["total"],
                "total": self._tally["total"],
            }
            return {
                "state": "running" if self._running else "stopped",
                "running": self._running,
                "debug": self.debug,
                "loop_sleep": self._sleep_seconds,
                "tally": json.loads(json.dumps(self._tally)),  # deep copy
                "tallies": tallies,
                "caps": DAILY_CAPS,
                "configured_chats": {k: bool(v) for k, v in TIER_TO_CHAT.items()},
                "last_send_result": self.last_send_result,
                "last_error": self.last_error,
                "day": date.today().isoformat(),
            }

    def can_send(self, tier: str) -> bool:
        self._maybe_reset_tallies()
        t = (tier or "").lower()
        cap = DAILY_CAPS.get(t)
        if cap is None:
            return True
        with self._lock:
            return self._tally["by_tier"].get(t, 0) < cap

    def send_to_tier(self, tier: str, text: str) -> Dict[str, Any]:
        """
        Safely send a message respecting daily caps.
        Returns Telegram response JSON (with ok/error).
        """
        self._maybe_reset_tallies()
        t = (tier or "vip").lower()
        if t not in ("free", "basic", "pro", "vip"):
            res = {"ok": False, "error": f"unknown tier: {t}"}
            self.last_send_result = res
            return res
        if not TIER_TO_CHAT.get(t):
            res = {"ok": False, "error": f"tier {t} not configured"}
            self.last_send_result = res
            return res
        if not self.can_send(t):
            cap = DAILY_CAPS.get(t)
            res = {"ok": False, "error": f"daily cap reached for {t} ({cap})"}
            self.last_send_result = res
            return res

        # Debug prefix includes UTC timestamp; message bodies should fit the channel boxes
        msg = text
        if self.debug:
            msg = f"[DEBUG]\n{datetime.now(timezone.utc).isoformat(timespec='seconds')}Z\n\n{text}"

        data = _send_telegram(msg, t)
        self.last_send_result = data
        if data.get("ok"):
            self._inc_tally(t)
        return data

    # ---------- Loop ----------
    def _loop(self) -> None:
        while True:
            with self._lock:
                running = self._running
                dbg = self.debug
                sleep_s = self._sleep_seconds
            if not running:
                break

            self._maybe_reset_tallies()

            if dbg:
                print(f"[ENGINE] tick {datetime.utcnow().isoformat()}Z | tally={self._tally}")

            time.sleep(max(1, sleep_s))

# Singleton exported
ENGINE = LiveEngine()
