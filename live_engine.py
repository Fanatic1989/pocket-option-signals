# live_engine.py
# Telegram wiring + live loop + daily caps for Pocket-Option Signals

from __future__ import annotations
import os
import json
import time
import threading
from datetime import datetime, date, timezone
from typing import Optional, Dict, Any, Tuple

import requests

# ===================== Telegram configuration ================================

BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()

def _first_nonempty(*names: str) -> str:
    for n in names:
        v = os.getenv(n, "").strip()
        if v:
            return v
    return ""

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
    "basic": 6,
    "pro":   15,
    "vip":   None,  # unlimited
}

# ===================== Low-level Telegram sender =============================

def _send_telegram(text: str, tier: str = "vip") -> Dict[str, Any]:
    """
    Send a message to the Telegram channel for the given tier.
    Returns a result dict similar to Telegram API (ok, error/info, raw if available).
    """
    result: Dict[str, Any] = {"ok": False, "tier": tier, "info": "", "status": 0}

    if not BOT_TOKEN:
        result["info"] = "Missing TELEGRAM_BOT_TOKEN"
        return result

    tier = (tier or "vip").lower()
    chat_id = TIER_TO_CHAT.get(tier)
    if not chat_id:
        result["info"] = f"No chat id configured for tier '{tier}'"
        return result

    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": text,
        "parse_mode": "HTML",
        "disable_web_page_preview": True,
    }
    try:
        r = requests.post(url, json=payload, timeout=15)
        result["status"] = r.status_code
        try:
            js = r.json()
        except Exception:
            js = {}
        result["raw"] = js
        ok = r.ok and bool(js.get("ok"))
        result["ok"] = ok
        result["info"] = "sent" if ok else (js.get("description") or r.text)
        return result
    except Exception as e:
        result["info"] = str(e)
        return result

def tg_test() -> Tuple[bool, str]:
    """
    Sends one short test message to each configured tier (free/basic/pro/vip).
    Returns (ok_any, info_json_string).
    """
    results: Dict[str, Any] = {}
    if not BOT_TOKEN:
        return False, json.dumps({"error": "Missing TELEGRAM_BOT_TOKEN"})

    if not any(TIER_TO_CHAT.values()):
        return False, json.dumps({"error": "No Telegram channels configured"})

    for t in ("free", "basic", "pro", "vip"):
        if TIER_TO_CHAT.get(t):
            results[t] = _send_telegram(f"🧪 Test from bot ({t.upper()})", t)
        else:
            results[t] = {"ok": False, "info": "not configured"}

    ok_any = any(v.get("ok") for v in results.values())
    return ok_any, json.dumps(results)

# ===================== Live Engine ===========================================

class LiveEngine:
    """
    Minimal live loop with:
      - start/stop and debug flag
      - per-tier daily tallies with cap enforcement
      - thread-safe counters
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self.debug = False

        # tallies: {"date": YYYY-MM-DD, "by_tier": {"free": n, "basic": n, ...}, "total": n}
        today = date.today().isoformat()
        self._tally = {"date": today, "by_tier": {"free": 0, "basic": 0, "pro": 0, "vip": 0}, "total": 0}

        # loop timing
        self._sleep_seconds = int(os.getenv("ENGINE_LOOP_SLEEP", "4"))

        # remember last send result for diagnostics
        self._last_send: Dict[str, Any] = {}
        self._last_error: Optional[str] = None

    # ---------- Internal helpers ----------
    def _maybe_reset_tallies(self) -> None:
        with self._lock:
            today = date.today().isoformat()
            if self._tally.get("date") != today:
                if self.debug:
                    print(f"[ENGINE] new day detected; resetting tallies {self._tally} -> 0s")
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
    def tally(self) -> Dict[str, Any]:
        self._maybe_reset_tallies()
        with self._lock:
            # deep copy
            return json.loads(json.dumps(self._tally))

    def status(self) -> Dict[str, Any]:
        self._maybe_reset_tallies()
        with self._lock:
            return {
                "running": self._running,
                "loop_sleep": self._sleep_seconds,
                "tally": self._tally,
                "debug": self.debug,
                "configured_tiers": {k: bool(v) for k, v in TIER_TO_CHAT.items()},
                "last_send_result": self._last_send,
                "day": self._tally.get("date"),
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
        for _ in range(40):
            t = self._thread
            if not t or not t.is_alive():
                break
            time.sleep(0.05)
        return True, "stopped"

    def send_signal(self, tier: str, text: str) -> Tuple[bool, str]:
        """
        Safely send a message respecting daily caps.
        Return (ok, 'sent' | reason).
        """
        self._maybe_reset_tallies()
        tier = (tier or "vip").lower()
        if tier not in ("free","basic","pro","vip"):
            msg = f"unknown tier: {tier}"
            self._last_send = {"ok": False, "info": msg}
            return False, msg
        if not TIER_TO_CHAT.get(tier):
            msg = f"tier {tier} not configured"
            self._last_send = {"ok": False, "info": msg}
            return False, msg
        if self._cap_reached(tier):
            cap = DAILY_CAPS.get(tier)
            msg = f"daily cap reached for {tier} ({cap if cap is not None else '∞'})"
            self._last_send = {"ok": False, "info": msg}
            return False, msg

        # Optional: prepend debug prefix
        text_to_send = text
        if self.debug:
            ts = datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
            text_to_send = f"[DEBUG {ts}]\n\n{text}"

        res = _send_telegram(text_to_send, tier)
        self._last_send = res
        ok = bool(res.get("ok"))
        if ok:
            self._inc_tally(tier)
        return ok, res.get("info") or ("sent" if ok else "failed")

    # ---------- Loop ----------
    def _loop(self) -> None:
        """
        Very light loop: You can wire strategy logic here later.
        Currently just idles and resets tallies each day, optionally logs a heartbeat.
        """
        while True:
            with self._lock:
                running = self._running
            if not running:
                break

            try:
                self._maybe_reset_tallies()
                if self.debug:
                    print(f"[ENGINE] tick {datetime.utcnow().isoformat()}Z | tally={self._tally}")
                time.sleep(max(1, self._sleep_seconds))
            except Exception as e:
                self._last_error = f"{type(e).__name__}: {e}"
                time.sleep(max(1, self._sleep_seconds))

# Singleton exported for routes
ENGINE = LiveEngine()
