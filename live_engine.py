# live_engine.py — Telegram wiring + live loop + daily caps + diagnostics
# -----------------------------------------------------------------------------
# Exports used elsewhere:
#   - ENGINE (singleton LiveEngine)
#   - TIER_TO_CHAT (dict)
#   - DAILY_CAPS   (dict)
#   - tg_test_all()      -> diagnostics + sends a test to each configured tier
#   - tg_test_one(tier)  -> send a test to one tier
#   - get_me_diag()      -> returns bot getMe() JSON
#
# Caps you requested: Free=3, Basic=6, Pro=15, VIP=∞

from __future__ import annotations

import os
import json
import time
import threading
from datetime import date
from typing import Dict, Any, Optional, Tuple

import requests


# ===================== Telegram configuration ================================

def _first_nonempty(*names: str) -> str:
    for n in names:
        v = os.getenv(n, "").strip()
        if v:
            return v
    return ""

BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()

# Support BOTH legacy TELEGRAM_CHAT_ID_* and new TELEGRAM_CHAT_* variable names
CHAT_FREE  = _first_nonempty("TELEGRAM_CHAT_ID_FREE",  "TELEGRAM_CHAT_FREE")
CHAT_BASIC = _first_nonempty("TELEGRAM_CHAT_ID_BASIC", "TELEGRAM_CHAT_BASIC")
CHAT_PRO   = _first_nonempty("TELEGRAM_CHAT_ID_PRO",   "TELEGRAM_CHAT_PRO")
CHAT_VIP   = _first_nonempty("TELEGRAM_CHAT_ID_VIP",   "TELEGRAM_CHAT_VIP")

# Optional single fallback (used if a tier’s chat isn’t set)
FALLBACK_CHAT = os.getenv("TELEGRAM_CHAT_ID", "").strip()

TIER_TO_CHAT: Dict[str, str] = {
    "free":  CHAT_FREE  or FALLBACK_CHAT,
    "basic": CHAT_BASIC or FALLBACK_CHAT,
    "pro":   CHAT_PRO   or FALLBACK_CHAT,
    "vip":   CHAT_VIP   or FALLBACK_CHAT,
}

# Per-tier daily caps (None => unlimited)
DAILY_CAPS: Dict[str, Optional[int]] = {
    "free":  3,
    "basic": 6,     # <= you asked for 6 here
    "pro":   15,
    "vip":   None,  # ∞
}


# ===================== Low-level Telegram helpers ============================

def _mask_token(tok: str) -> str:
    if not tok:
        return ""
    if len(tok) <= 8:
        return "***"
    return tok[:6] + "..." + tok[-4:]

def _send_telegram(text: str, tier: str) -> Tuple[bool, str, Dict[str, Any]]:
    """
    Send a message to the Telegram channel for the given tier.
    Returns (ok, info_text, raw_json_or_error).
    """
    if not BOT_TOKEN:
        return False, "Missing TELEGRAM_BOT_TOKEN", {}

    t = (tier or "vip").lower()
    chat_id = TIER_TO_CHAT.get(t)
    if not chat_id:
        return False, f"No Telegram chat configured for tier '{t}'", {}

    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": text,
        "parse_mode": "HTML",
        "disable_web_page_preview": True,
    }
    try:
        r = requests.post(url, json=payload, timeout=15)
        try:
            raw = r.json()
        except Exception:
            raw = {"status_code": r.status_code, "text": r.text[:500]}
        ok = bool(r.ok and raw.get("ok") is True)
        return (True, "sent", raw) if ok else (False, f"HTTP {r.status_code}", raw)
    except Exception as e:
        return False, str(e), {}

def _get_me() -> Dict[str, Any]:
    if not BOT_TOKEN:
        return {"ok": False, "error": "Missing TELEGRAM_BOT_TOKEN"}
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/getMe"
    try:
        r = requests.get(url, timeout=10)
        return r.json() if r.headers.get("content-type","").startswith("application/json") else {"ok": False, "error": r.text[:300]}
    except Exception as e:
        return {"ok": False, "error": str(e)}


# ===================== Diagnostics API (imported by routes/app) ==============

def tg_test_one(tier: str) -> Dict[str, Any]:
    """Send a short test message to a specific tier."""
    t = (tier or "vip").lower()
    ok, info, raw = _send_telegram(f"🧪 Test from bot ({t.upper()})", t)
    return {"tier": t, "ok": ok, "info": info, "raw": raw}

def tg_test_all() -> Dict[str, Any]:
    """
    Sends a short test message to each configured tier (free/basic/pro/vip).
    Returns a diagnostic blob similar to what your dashboard expects.
    """
    results = {}
    for t in ("free", "basic", "pro", "vip"):
        if TIER_TO_CHAT.get(t):
            ok, info, raw = _send_telegram(f"🧪 Test from bot ({t.upper()})", t)
            results[t] = {"ok": ok, "info": info, "raw": raw}
        else:
            results[t] = {"ok": False, "info": "not configured"}

    getme = _get_me()
    diag = {
        "configured_chats": {k: bool(v) for k, v in TIER_TO_CHAT.items()},
        "getMe": getme,
        "ok": any(v.get("ok") for v in results.values()),
        "send_result": "sent" if any(v.get("ok") for v in results.values()) else "none",
        "token_masked": _mask_token(BOT_TOKEN),
        "results": results,
    }
    return diag

def get_me_diag() -> Dict[str, Any]:
    """Expose getMe + configured chats for /api/check_bot route."""
    return {
        "configured_chats": {k: bool(v) for k, v in TIER_TO_CHAT.items()},
        "getMe": _get_me(),
        "token_masked": _mask_token(BOT_TOKEN),
    }


# ===================== Live Engine ===========================================

class LiveEngine:
    """
    Minimal live loop with:
      - start/stop
      - per-tier daily tallies with cap enforcement
      - debug flag
      - thread-safe counters
    Use .send_signal(tier, text) to send respecting caps.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self.debug = False

        today = date.today().isoformat()
        self._tally = {
            "date": today,
            "by_tier": {"free": 0, "basic": 0, "pro": 0, "vip": 0},
            "total": 0
        }

        self._sleep_seconds = int(os.getenv("ENGINE_LOOP_SLEEP", "4"))
        self._last_send: Dict[str, Any] = {}

    # ---------- Internal helpers ----------
    def _maybe_reset_tallies(self) -> None:
        with self._lock:
            today = date.today().isoformat()
            if self._tally.get("date") != today:
                self._tally = {
                    "date": today,
                    "by_tier": {"free": 0, "basic": 0, "pro": 0, "vip": 0},
                    "total": 0
                }

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
    def tally(self) -> Dict[str, Any]:
        self._maybe_reset_tallies()
        with self._lock:
            return json.loads(json.dumps(self._tally))  # deep copy

    def status(self) -> Dict[str, Any]:
        self._maybe_reset_tallies()
        with self._lock:
            return {
                "running": self._running,
                "loop_sleep": self._sleep_seconds,
                "tally": self._tally,
                "debug": self.debug,
                "configured_tiers": {k: bool(v) for k, v in TIER_TO_CHAT.items()},
                "last_send": self._last_send,
                "caps": DAILY_CAPS,
                "day": self._tally["date"],
            }

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
        # wait briefly for thread to exit
        for _ in range(40):
            t = self._thread
            if not t or not t.is_alive():
                break
            time.sleep(0.05)
        return True, "stopped"

    def send_signal(self, tier: str, text: str) -> Tuple[bool, str]:
        """
        Safely send a message respecting daily caps.
        Returns (ok, 'sent' | reason).
        """
        self._maybe_reset_tallies()
        t = (tier or "vip").lower()
        if t not in ("free", "basic", "pro", "vip"):
            return False, f"unknown tier: {t}"
        if not TIER_TO_CHAT.get(t):
            return False, f"tier {t} not configured"

        if self._cap_reached(t):
            cap = DAILY_CAPS.get(t)
            return False, f"daily cap reached for {t} ({'∞' if cap is None else cap})"

        ok, info, raw = _send_telegram(text, t)
        with self._lock:
            self._last_send = {"tier": t, "ok": ok, "info": info, "raw": raw}
        if ok:
            self._inc_tally(t)
        return ok, info

    # ---------- Loop ----------
    def _loop(self) -> None:
        while True:
            with self._lock:
                running = self._running
            if not running:
                break

            self._maybe_reset_tallies()

            if self.debug:
                try:
                    print(f"[ENGINE] tick | date={self._tally['date']} | tally={self._tally}")
                except Exception:
                    pass

            time.sleep(max(1, self._sleep_seconds))


# Singleton used by routes/app
ENGINE = LiveEngine()
