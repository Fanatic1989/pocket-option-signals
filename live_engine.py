# live_engine.py
# -----------------------------------------------------------------------------
# Telegram wiring + live loop + per-tier daily caps
# - Env: TELEGRAM_BOT_TOKEN
#        TELEGRAM_CHAT_FREE / TELEGRAM_CHAT_ID_FREE
#        TELEGRAM_CHAT_BASIC / TELEGRAM_CHAT_ID_BASIC
#        TELEGRAM_CHAT_PRO / TELEGRAM_CHAT_ID_PRO
#        TELEGRAM_CHAT_VIP / TELEGRAM_CHAT_ID_VIP
#        (optional fallback) TELEGRAM_CHAT_ID
# - Caps: FREE=3, BASIC=6, PRO=15, VIP=∞
# - Public API expected by app/routes:
#       ENGINE.start()/stop()/set_debug(bool)/status()
#       ENGINE.send_signal(tier, text)        # preferred
#       ENGINE.send_to_tier(tier, text)       # legacy alias
#   Helpers exported for diagnostics:
#       tg_test() -> (ok: bool, diag_json_str)
#       tg_test_all() -> (ok: bool, diag_json_str)   # alias
#       BOT_TOKEN, TIER_TO_CHAT, DAILY_CAPS
# -----------------------------------------------------------------------------

from __future__ import annotations
import os
import json
import time
import threading
from datetime import datetime, date, timezone
from typing import Optional, Dict, Any, Tuple

import requests

# ========================= Telegram configuration ============================

BOT_TOKEN = (os.getenv("TELEGRAM_BOT_TOKEN") or "").strip()


def _first_nonempty(*names: str) -> str:
    for n in names:
        v = os.getenv(n, "")
        if v and v.strip():
            return v.strip()
    return ""


# Accept both new and legacy names; allow a single fallback channel
CHAT_FREE = _first_nonempty("TELEGRAM_CHAT_FREE", "TELEGRAM_CHAT_ID_FREE")
CHAT_BASIC = _first_nonempty("TELEGRAM_CHAT_BASIC", "TELEGRAM_CHAT_ID_BASIC")
CHAT_PRO = _first_nonempty("TELEGRAM_CHAT_PRO", "TELEGRAM_CHAT_ID_PRO")
CHAT_VIP = _first_nonempty("TELEGRAM_CHAT_VIP", "TELEGRAM_CHAT_ID_VIP")
FALLBACK_CHAT = _first_nonempty("TELEGRAM_CHAT_ID")

TIER_TO_CHAT: Dict[str, str] = {
    "free": CHAT_FREE or FALLBACK_CHAT,
    "basic": CHAT_BASIC or FALLBACK_CHAT,
    "pro": CHAT_PRO or FALLBACK_CHAT,
    "vip": CHAT_VIP or FALLBACK_CHAT,
}

# Per-tier caps (None => unlimited)
DAILY_CAPS: Dict[str, Optional[int]] = {
    "free": 3,
    "basic": 6,
    "pro": 15,
    "vip": None,
}


def _send_telegram_raw(chat_id: str, text: str) -> Dict[str, Any]:
    """Low-level Telegram send; returns response JSON or {'ok': False, 'error': ...}."""
    if not BOT_TOKEN:
        return {"ok": False, "error": "Missing TELEGRAM_BOT_TOKEN"}

    if not chat_id:
        return {"ok": False, "error": "Missing chat_id"}

    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": text,
        "parse_mode": "HTML",
        "disable_web_page_preview": True,
    }
    try:
        r = requests.post(url, json=payload, timeout=15)
        # Telegram always returns JSON
        try:
            js = r.json()
        except Exception:
            js = {"ok": False, "error": f"HTTP {r.status_code}", "text": r.text[:300]}
        return js
    except Exception as e:
        return {"ok": False, "error": str(e)}


def _send_telegram_tier(tier: str, text: str) -> Dict[str, Any]:
    t = (tier or "vip").lower()
    chat_id = TIER_TO_CHAT.get(t)
    if not chat_id:
        return {"ok": False, "error": f"No Telegram chat configured for '{t}'"}
    return _send_telegram_raw(chat_id, text)


def tg_test() -> Tuple[bool, str]:
    """
    Send a tiny test to each configured tier; returns (overall_ok, json_string).
    Exported because routes/app import this.
    """
    results: Dict[str, Any] = {}
    if not BOT_TOKEN:
        return False, json.dumps({"ok": False, "error": "Missing TELEGRAM_BOT_TOKEN"})

    any_configured = any(bool(v) for v in TIER_TO_CHAT.values())
    if not any_configured:
        return False, json.dumps({"ok": False, "error": "No Telegram chats configured"})

    for t in ("free", "basic", "pro", "vip"):
        if TIER_TO_CHAT.get(t):
            js = _send_telegram_tier(t, f"🧪 Bot test ({t.upper()})")
            results[t] = js
        else:
            results[t] = {"ok": False, "error": "not configured"}

    overall_ok = any(r.get("ok") is True for r in results.values())
    try:
        s = json.dumps(results)
    except Exception:
        s = str(results)
    return overall_ok, s


def tg_test_all() -> Tuple[bool, str]:
    # Alias kept for older app.py builds
    return tg_test()


# =============================== Live Engine =================================

class LiveEngine:
    """
    Minimal live loop with:
      - start/stop
      - per-tier daily tallies with cap enforcement
      - debug flag
      - thread-safe counters
    NOTE: Signal discovery logic is not here; call .send_signal(tier, text).
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._sleep_seconds = int(os.getenv("ENGINE_LOOP_SLEEP", "4"))
        self.debug = False

        today = date.today().isoformat()
        # Primary shape
        self._tally = {
            "date": today,
            "by_tier": {"free": 0, "basic": 0, "pro": 0, "vip": 0},
            "total": 0,
        }
        # Legacy shape (some templates read 'tallies')
        self._tallies_legacy = {"free": 0, "basic": 0, "pro": 0, "vip": 0, "all": 0}

        self._last_send_result: Dict[str, Any] = {}
        self._last_error: Optional[str] = None

    # ---------- internals ----------
    def _maybe_reset_tallies(self) -> None:
        with self._lock:
            cur = date.today().isoformat()
            if self._tally.get("date") != cur:
                self._tally = {
                    "date": cur,
                    "by_tier": {"free": 0, "basic": 0, "pro": 0, "vip": 0},
                    "total": 0,
                }
                self._tallies_legacy = {"free": 0, "basic": 0, "pro": 0, "vip": 0, "all": 0}

    def _inc_tally(self, tier: str) -> None:
        with self._lock:
            self._tally["by_tier"][tier] += 1
            self._tally["total"] += 1
            self._tallies_legacy[tier] += 1
            self._tallies_legacy["all"] += 1

    def _cap_reached(self, tier: str) -> bool:
        cap = DAILY_CAPS.get(tier)
        if cap is None:
            return False
        with self._lock:
            return self._tally["by_tier"].get(tier, 0) >= cap

    # ---------- public API ----------
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
        # quick join
        for _ in range(40):
            t = self._thread
            if not t or not t.is_alive():
                break
            time.sleep(0.05)
        return True, "stopped"

    def set_debug(self, value: bool) -> None:
        with self._lock:
            self.debug = bool(value)

    def can_send(self, tier: str) -> bool:
        t = (tier or "").lower()
        return not self._cap_reached(t)

    def send_signal(self, tier: str, text: str) -> Dict[str, Any]:
        """
        Respect per-tier daily caps; prepend debug header if enabled.
        Returns Telegram response JSON (or error dict).
        """
        self._maybe_reset_tallies()
        t = (tier or "vip").lower()
        if t not in ("free", "basic", "pro", "vip"):
            out = {"ok": False, "error": f"unknown tier '{t}'"}
            self._last_send_result = out
            return out

        if not TIER_TO_CHAT.get(t):
            out = {"ok": False, "error": f"No Telegram chat configured for '{t}'"}
            self._last_send_result = out
            return out

        if self._cap_reached(t):
            out = {"ok": False, "error": f"Daily cap reached for '{t}'"}
            self._last_send_result = out
            return out

        msg = text or ""
        if self.debug:
            ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
            msg = f"[DEBUG]\n{ts}Z\n\n{msg}"

        res = _send_telegram_tier(t, msg)
        self._last_send_result = res
        if res.get("ok"):
            self._inc_tally(t)
        return res

    # legacy name used by some previous builds
    def send_to_tier(self, tier: str, text: str) -> Dict[str, Any]:
        return self.send_signal(tier, text)

    def status(self) -> Dict[str, Any]:
        """
        Provide a superset so both old and new UIs work:
          - running/debug/loop_sleep
          - tally (by_tier + total)  [new]
          - tallies (legacy with 'all') [old]
          - caps, configured_chats
          - last_send_result, last_error, day
        """
        self._maybe_reset_tallies()
        with self._lock:
            return {
                "running": self._running,
                "debug": self.debug,
                "loop_sleep": self._sleep_seconds,
                "tally": json.loads(json.dumps(self._tally)),  # copy
                "tallies": json.loads(json.dumps(self._tallies_legacy)),
                "caps": DAILY_CAPS,
                "configured_chats": {k: bool(v) for k, v in TIER_TO_CHAT.items()},
                "last_send_result": self._last_send_result,
                "last_error": self._last_error,
                "day": self._tally.get("date"),
            }

    # ---------- loop ----------
    def _loop(self) -> None:
        while True:
            with self._lock:
                running = self._running
                dbg = self.debug
                slp = max(1, int(self._sleep_seconds))
            if not running:
                break
            try:
                self._maybe_reset_tallies()
                if dbg:
                    print(f"[ENGINE] tick {datetime.utcnow().isoformat()}Z | tally={self._tally}")
            except Exception as e:
                with self._lock:
                    self._last_error = f"{type(e).__name__}: {e}"
            time.sleep(slp)


# Singleton used by the app
ENGINE = LiveEngine()
