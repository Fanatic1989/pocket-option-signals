# live_engine.py — Telegram routing (4 channels) + daily caps + engine shell

from __future__ import annotations
import os
import json
import time
import threading
from datetime import datetime, timezone

import requests

# --------------------- Environment & constants ---------------------

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()

# 4 channel IDs (can be supergroup/channel IDs like -1001234567890)
CHAT_FREE  = os.getenv("TELEGRAM_CHAT_ID_FREE",  "").strip()
CHAT_BASIC = os.getenv("TELEGRAM_CHAT_ID_BASIC", "").strip()
CHAT_PRO   = os.getenv("TELEGRAM_CHAT_ID_PRO",   "").strip()
CHAT_VIP   = os.getenv("TELEGRAM_CHAT_ID_VIP",   "").strip()

# expose the keys so routes.py /telegram/diag can show what’s set
TELEGRAM_CHAT_KEYS = [
    "TELEGRAM_CHAT_ID_FREE",
    "TELEGRAM_CHAT_ID_BASIC",
    "TELEGRAM_CHAT_ID_PRO",
    "TELEGRAM_CHAT_ID_VIP",
]

# Per-tier daily caps (per channel per UTC day)
DAILY_CAPS: dict[str, int | None] = {
    "free":  3,
    "basic": 6,
    "pro":   15,
    "vip":   None,   # unlimited
}

# Map tier -> chat id (missing values are ignored at send time)
TIER_TO_CHAT = {
    "free":  CHAT_FREE,
    "basic": CHAT_BASIC,
    "pro":   CHAT_PRO,
    "vip":   CHAT_VIP,
}

# --------------------- Telegram low-level helpers ---------------------

def _tg_api(method: str) -> str:
    if not TELEGRAM_BOT_TOKEN:
        raise RuntimeError("Missing TELEGRAM_BOT_TOKEN")
    return f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/{method}"

def _tg_send_message(chat_id: str, text: str, disable_web_page_preview=True) -> tuple[bool, str]:
    """Returns (ok, info). Never raises."""
    if not TELEGRAM_BOT_TOKEN:
        return False, "Missing TELEGRAM_BOT_TOKEN"
    if not chat_id:
        return False, "Missing chat_id"
    try:
        r = requests.post(
            _tg_api("sendMessage"),
            json={
                "chat_id": chat_id,
                "text": text,
                "parse_mode": "HTML",
                "disable_web_page_preview": disable_web_page_preview,
            },
            timeout=15,
        )
        ok = r.ok and r.json().get("ok", False)
        return (ok, r.text if not ok else "ok")
    except Exception as e:
        return False, str(e)

def _send_telegram(text: str, tier: str | None = None) -> tuple[bool, str]:
    """
    Public helper used by routes.telegram_diag().
    If tier is None, broadcasts to all configured channels (FREE/BASIC/PRO/VIP).
    Applies caps when ENGINE is running; for diag/test we still apply caps to avoid overflow.
    """
    targets = []
    if tier:
        t = tier.lower()
        cid = TIER_TO_CHAT.get(t, "")
        if cid:
            targets.append((t, cid))
    else:
        for t, cid in TIER_TO_CHAT.items():
            if cid:
                targets.append((t, cid))

    if not targets:
        return False, "No Telegram channels configured"

    results = []
    all_ok = True
    for t, chat in targets:
        ok, info = ENGINE._send_with_caps(tier=t, chat_id=chat, text=text)
        results.append(f"{t}:{'ok' if ok else 'fail'}")
        if not ok:
            all_ok = False

    return all_ok, ", ".join(results)

# --------------------- Live engine (shell) ---------------------

class LiveEngine:
    """
    Lightweight engine wrapper:
    - start/stop: flips running flag
    - debug: when True, still sends; when debug_only_send_to is set to a tier, only that tier is used
    - per-tier per-day caps enforced in _send_with_caps()
    - tally: totals per tier & overall, reset at UTC day boundary
    """
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._running = False
        self.debug = False
        self.debug_only_send_to: str | None = None  # e.g. "vip" to test only VIP

        # per-UTC-day counters
        self._day_key = self._utc_day_key()
        self._tally = {
            "free": 0, "basic": 0, "pro": 0, "vip": 0,
            "total": 0,
            # optional: deliver per-channel logs last 50
            "last": [],   # list of dicts {ts, tier, ok, info}
        }

        # worker placeholder (if you later add loops)
        self._thread: threading.Thread | None = None

    # ---------- internals ----------
    def _utc_day_key(self) -> str:
        now = datetime.now(timezone.utc)
        return now.strftime("%Y-%m-%d")

    def _rollover_if_needed(self) -> None:
        day = self._utc_day_key()
        if day != self._day_key:
            self._day_key = day
            self._tally = {"free":0,"basic":0,"pro":0,"vip":0,"total":0,"last":[]}

    # ---------- public controls ----------
    def start(self) -> tuple[bool, str]:
        with self._lock:
            if self._running:
                return True, "already running"
            self._running = True
            # (optional) spin worker if you later implement loops
            return True, "started"

    def stop(self) -> tuple[bool, str]:
        with self._lock:
            if not self._running:
                return True, "already stopped"
            self._running = False
            return True, "stopped"

    def status(self) -> dict:
        with self._lock:
            self._rollover_if_needed()
            return {
                "running": self._running,
                "debug": self.debug,
                "day": self._day_key,
                "tally": {k:self._tally[k] for k in ("free","basic","pro","vip","total")},
                "last": list(self._tally["last"]),
            }

    def tally(self) -> dict:
        with self._lock:
            self._rollover_if_needed()
            return {
                "day": self._day_key,
                "per_tier": {k:self._tally[k] for k in ("free","basic","pro","vip")},
                "total": self._tally["total"],
            }

    # ---------- sending with caps ----------
    def _cap_for_tier(self, tier: str) -> int | None:
        return DAILY_CAPS.get(tier.lower())

    def _send_with_caps(self, tier: str, chat_id: str, text: str) -> tuple[bool, str]:
        tier = (tier or "").lower().strip()
        with self._lock:
            self._rollover_if_needed()

            # Debug: restrict to a single tier if set
            if self.debug_only_send_to and tier != self.debug_only_send_to.lower():
                # pretend success but skip actual send to avoid confusion
                info = f"debug_only_send_to={self.debug_only_send_to}, skipped {tier}"
                self._tally["last"].append({"ts": int(time.time()), "tier": tier, "ok": True, "info": info})
                self._trim_log()
                return True, info

            cap = self._cap_for_tier(tier)
            used = self._tally.get(tier, 0)
            if cap is not None and used >= cap:
                info = f"cap reached ({used}/{cap})"
                self._tally["last"].append({"ts": int(time.time()), "tier": tier, "ok": False, "info": info})
                self._trim_log()
                return False, info

        # send outside the lock
        ok, info = _tg_send_message(chat_id, text)

        with self._lock:
            self._rollover_if_needed()
            self._tally["last"].append({"ts": int(time.time()), "tier": tier, "ok": ok, "info": info})
            self._trim_log()
            if ok:
                self._tally[tier] = self._tally.get(tier, 0) + 1
                self._tally["total"] += 1
        return ok, info

    def _trim_log(self):
        if len(self._tally["last"]) > 50:
            self._tally["last"] = self._tally["last"][-50:]

    # ---------- high-level helper you can call from signal logic ----------
    def send_signal(self, text: str, tiers: list[str] | None = None) -> dict:
        """
        Send the same signal text to one or more tiers with cap enforcement.
        tiers=None => send to all configured tiers.
        Returns per-tier results.
        """
        if tiers is None:
            tiers = ["free","basic","pro","vip"]

        out = {}
        for t in tiers:
            chat = TIER_TO_CHAT.get(t.lower())
            if not chat:
                out[t] = {"ok": False, "info": "no chat configured"}
                continue
            ok, info = self._send_with_caps(t, chat, text)
            out[t] = {"ok": ok, "info": info}
        return out


# Singleton instance used by routes.py
ENGINE = LiveEngine()

# Simple test used by routes.telegram_test()
def tg_test() -> tuple[bool, str]:
    msg = "🧪 Test from server — if this appears, routing & caps are wired."
    return _send_telegram(msg)
