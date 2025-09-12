# live_engine.py — Telegram wiring + live loop + daily caps (compatible with GitHub dashboard)
from __future__ import annotations

import os, json, time, threading
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

# Support ALL common naming styles (yours from screenshot included)
CHAT_FREE  = _first_nonempty("TELEGRAM_CHAT_FREE",  "TELEGRAM_CHAT_ID_FREE",  "FREE_CHAT_ID")
CHAT_BASIC = _first_nonempty("TELEGRAM_CHAT_BASIC", "TELEGRAM_CHAT_ID_BASIC", "BASIC_CHAT_ID")
CHAT_PRO   = _first_nonempty("TELEGRAM_CHAT_PRO",   "TELEGRAM_CHAT_ID_PRO",   "PRO_CHAT_ID")
CHAT_VIP   = _first_nonempty("TELEGRAM_CHAT_VIP",   "TELEGRAM_CHAT_ID_VIP",   "VIP_CHAT_ID")

# Optional fallback if you keep one global channel
FALLBACK_CHAT = _first_nonempty("TELEGRAM_CHAT_ID", "CHAT_ID_FALLBACK")

TIER_TO_CHAT: Dict[str, str] = {
    "free":  CHAT_FREE  or FALLBACK_CHAT,
    "basic": CHAT_BASIC or FALLBACK_CHAT,
    "pro":   CHAT_PRO   or FALLBACK_CHAT,
    "vip":   CHAT_VIP   or FALLBACK_CHAT,
}

# Per-tier daily caps (None => unlimited)
DAILY_CAPS: Dict[str, Optional[int]] = {
    "free":  3,
    "basic": 6,     # your GitHub spec
    "pro":   15,
    "vip":   None,  # unlimited
}

# ===================== Low-level Telegram sender =============================

def _send_telegram(text: str, tier: str = "vip") -> Tuple[bool, Dict[str, Any]]:
    """
    Send a message to the Telegram channel for the given tier.
    Returns (ok, raw_result_dict). raw_result_dict mirrors Telegram JSON, or
    {"ok": False, "error": "..."} on failure.
    """
    if not BOT_TOKEN:
        return False, {"ok": False, "error": "Missing TELEGRAM_BOT_TOKEN"}

    tier = (tier or "vip").lower()
    chat_id = TIER_TO_CHAT.get(tier)
    if not chat_id:
        return False, {"ok": False, "error": f"No Telegram channel configured for tier '{tier}'"}

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
            js = r.json()
        except Exception:
            js = {"ok": False, "error": f"Non-JSON response: {r.text[:400]}..."}
        js["_http_status"] = r.status_code
        js["_sent_at"] = int(datetime.now(timezone.utc).timestamp())
        return bool(js.get("ok")), js
    except Exception as e:
        return False, {"ok": False, "error": str(e), "_sent_at": int(datetime.now(timezone.utc).timestamp())}

def tg_test_all() -> Tuple[bool, str]:
    """Send a small test to every configured tier. Returns (ok, json_string)."""
    results: Dict[str, Any] = {}
    if not BOT_TOKEN:
        return False, json.dumps({"error": "Missing TELEGRAM_BOT_TOKEN"})
    if not any(TIER_TO_CHAT.values()):
        return False, json.dumps({"error": "No Telegram channels configured"})

    for t in ("free","basic","pro","vip"):
        if TIER_TO_CHAT.get(t):
            ok, info = _send_telegram(f"🧪 Bot test ({t.upper()})", t)
            results[t] = {"ok": ok, "info": info}
        else:
            results[t] = {"ok": False, "info": "not configured"}
    return any(v["ok"] for v in results.values()), json.dumps(results, ensure_ascii=False)

# ===================== Live Engine ===========================================

class LiveEngine:
    """
    - start/stop loop (lightweight heartbeat)
    - per-tier daily tallies with cap enforcement
    - debug flag
    - thread-safe counters
    - last_send_result exposed for dashboards
    """
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self.debug = False

        today = date.today().isoformat()
        self._tally = {"date": today, "by_tier": {"free": 0, "basic": 0, "pro": 0, "vip": 0}, "total": 0}

        self._sleep_seconds = int(os.getenv("ENGINE_LOOP_SLEEP", "4"))
        self.last_send_result: Dict[str, Any] = {}
        self.last_error: Optional[str] = None

    # ---------- helpers ----------
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

    # ---------- public ----------
    def tally(self) -> Dict[str, Any]:
        self._maybe_reset_tallies()
        with self._lock:
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
                "caps": DAILY_CAPS,
                "last_send_result": self.last_send_result,
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

    def set_debug(self, value: bool) -> None:
        with self._lock:
            self.debug = bool(value)

    def send_signal(self, tier: str, text: str) -> Tuple[bool, str]:
        """Cap-aware send used by normal flows."""
        self._maybe_reset_tallies()
        tier = (tier or "vip").lower()
        if tier not in ("free","basic","pro","vip"):
            return False, f"unknown tier: {tier}"
        if not TIER_TO_CHAT.get(tier):
            return False, f"tier {tier} not configured"
        if self._cap_reached(tier):
            return False, f"daily cap reached for {tier} ({DAILY_CAPS.get(tier)})"

        msg = text if not self.debug else f"[DEBUG]\n{datetime.now(timezone.utc).isoformat(timespec='seconds')}Z\n\n{text}"
        ok, info = _send_telegram(msg, tier)
        with self._lock:
            self.last_send_result = info
        if ok:
            self._inc_tally(tier)
            return True, "sent"
        return False, info.get("description") or info.get("error") or "send failed"

    def send_raw(self, tier: str, text: str) -> Dict[str, Any]:
        """
        Non-cap test sender for /telegram/test — does NOT increment tallies.
        Returns raw Telegram JSON.
        """
        ok, info = _send_telegram(text, tier)
        with self._lock:
            self.last_send_result = info
        return info

    # Compat alias for older routes (expects Telegram-like dict)
    def send_to_tier(self, tier: str, text: str) -> Dict[str, Any]:
        ok, reason = self.send_signal(tier, text)
        if ok:
            with self._lock:
                data = dict(self.last_send_result) if isinstance(self.last_send_result, dict) else {}
            if not data:
                data = {"ok": True, "result": {"status": "sent"}}
            data.setdefault("ok", True)
            return data
        return {"ok": False, "error": reason}

    # ---------- loop ----------
    def _loop(self) -> None:
        while True:
            with self._lock:
                running = self._running
            if not running:
                break
            self._maybe_reset_tallies()
            if self.debug:
                print(f"[ENGINE] tick {datetime.utcnow().isoformat()}Z | tally={self._tally}")
            time.sleep(max(1, self._sleep_seconds))

# Singleton importable by routes
ENGINE = LiveEngine()
