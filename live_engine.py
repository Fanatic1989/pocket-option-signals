# live_engine.py — signal engine with rate limits + live tally
import os
import time
from datetime import datetime, timedelta, timezone
from collections import deque, defaultdict
import threading
import requests

from utils import TZ, within_window, get_config

# ====== Telegram config & helpers =============================================
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()

# You can set one or many chat IDs. Any env that starts with TELEGRAM_CHAT_ID will be used.
TELEGRAM_CHAT_KEYS = [k for k in os.environ.keys() if k.upper().startswith("TELEGRAM_CHAT_ID")]
TELEGRAM_CHAT_IDS = [os.getenv(k, "").strip() for k in TELEGRAM_CHAT_KEYS if os.getenv(k, "").strip()]

def _post_json(url, payload, timeout=12):
    try:
        r = requests.post(url, json=payload, timeout=timeout)
        return True, r.json()
    except Exception as e:
        return False, str(e)

def _send_telegram(text: str):
    if not TELEGRAM_BOT_TOKEN:
        return False, "No TELEGRAM_BOT_TOKEN provided"

    ok_all = True
    infos = []
    for chat_id in TELEGRAM_CHAT_IDS:
        ok, info = ENGINE.send_with_limits(chat_id, text)
        ok_all = ok_all and ok
        infos.append((chat_id, info))
    if not TELEGRAM_CHAT_IDS:
        return False, "No TELEGRAM_CHAT_ID* provided"
    return ok_all, infos

def tg_test():
    return _send_telegram("✅ Telegram test from Pocket Option Signals")

# ====== ENGINE ================================================================
class _Engine:
    def __init__(self):
        self._lock = threading.RLock()
        self._running = False
        self.debug = False
        self._thread = None

        # Tally / rate limiting
        self._sent_times_global = deque()  # timestamps (epoch seconds) for all sends
        self._sent_times_by_chat = defaultdict(deque)
        self._sent_count_today = 0
        self._sent_count_today_by_chat = defaultdict(int)
        self._midnight_epoch = self._today_midnight_epoch()

        # rolling status
        self._last_error = None
        self._last_loop = None
        self._last_reasons = []

        # limits (read from env, with safe defaults)
        self.LIMIT_PER_MIN = int(os.getenv("TELEGRAM_LIMIT_PER_MIN", "30"))
        self.LIMIT_PER_HOUR = int(os.getenv("TELEGRAM_LIMIT_PER_HOUR", "500"))
        self.LIMIT_PER_DAY = int(os.getenv("TELEGRAM_LIMIT_PER_DAY", "5000"))

    def _today_midnight_epoch(self):
        now = datetime.now(TZ)
        mid = datetime(now.year, now.month, now.day, tzinfo=TZ)
        return int(mid.timestamp())

    def _maybe_reset_daily(self):
        # Reset counters at local midnight
        now_epoch = int(time.time())
        if now_epoch - self._midnight_epoch >= 86400:
            self._midnight_epoch = self._today_midnight_epoch()
            self._sent_count_today = 0
            self._sent_count_today_by_chat = defaultdict(int)

    def _prune(self):
        # prune >1h from global, and >1h for chats; also compute minute/hour windows
        cutoff_min = time.time() - 60
        cutoff_hr = time.time() - 3600
        while self._sent_times_global and self._sent_times_global[0] < cutoff_hr:
            self._sent_times_global.popleft()
        # prune chat deques
        for dq in self._sent_times_by_chat.values():
            while dq and dq[0] < cutoff_hr:
                dq.popleft()

    def _window_counts(self, dq):
        now = time.time()
        one_min = now - 60
        one_hr = now - 3600
        min_count = sum(1 for t in dq if t >= one_min)
        hr_count = sum(1 for t in dq if t >= one_hr)
        return min_count, hr_count

    def send_with_limits(self, chat_id: str, text: str):
        with self._lock:
            self._maybe_reset_daily()
            self._prune()

            # global windows
            g_min, g_hr = self._window_counts(self._sent_times_global)

            # per-chat windows
            dq = self._sent_times_by_chat[chat_id]
            c_min, c_hr = self._window_counts(dq)

            # per-day (local)
            if self._sent_count_today >= self.LIMIT_PER_DAY:
                reason = f"daily cap reached ({self._sent_count_today}/{self.LIMIT_PER_DAY})"
                self._last_reasons.append(reason)
                return False, reason
            if g_min >= self.LIMIT_PER_MIN or c_min >= self.LIMIT_PER_MIN:
                reason = "per-minute cap"
                self._last_reasons.append(reason)
                return False, reason
            if g_hr >= self.LIMIT_PER_HOUR or c_hr >= self.LIMIT_PER_HOUR:
                reason = "per-hour cap"
                self._last_reasons.append(reason)
                return False, reason

        # actually send
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        ok, info = _post_json(url, {"chat_id": chat_id, "text": text})
        with self._lock:
            if ok and isinstance(info, dict) and info.get("ok"):
                now = time.time()
                self._sent_times_global.append(now)
                self._sent_times_by_chat[chat_id].append(now)
                self._sent_count_today += 1
                self._sent_count_today_by_chat[chat_id] += 1
                return True, "sent"
            else:
                self._last_error = str(info)
                return False, f"send failed: {info}"

    # ============= Loop control ==============================================
    def start(self):
        with self._lock:
            if self._running:
                return True, "already running"
            self._running = True
            self._thread = threading.Thread(target=self._loop, daemon=True)
            self._thread.start()
            return True, "started"

    def stop(self):
        with self._lock:
            if not self._running:
                return True, "already stopped"
            self._running = False
            return True, "stopped"

    def status(self):
        with self._lock:
            return {
                "running": self._running,
                "last_loop": self._last_loop,
                "last_error": self._last_error,
                "last_reasons": list(self._last_reasons[-10:]),
                "sent": self._sent_count_today,
            }

    def tally(self):
        """Return live tallies for UI."""
        with self._lock:
            now = datetime.now(TZ)
            # minute/hour counts from global
            g_min, g_hr = self._window_counts(self._sent_times_global)
            # per chat latest totals
            per_chat = []
            for chat_id, dq in self._sent_times_by_chat.items():
                cmin, chr_ = self._window_counts(dq)
                per_chat.append({
                    "chat_id": chat_id,
                    "last_min": cmin,
                    "last_hour": chr_,
                    "today": self._sent_count_today_by_chat.get(chat_id, 0),
                })
            return {
                "now": now.strftime("%Y-%m-%d %H:%M:%S"),
                "tz": str(TZ),
                "global": {
                    "last_min": g_min,
                    "last_hour": g_hr,
                    "today": self._sent_count_today,
                    "per_min_cap": self.LIMIT_PER_MIN,
                    "per_hour_cap": self.LIMIT_PER_HOUR,
                    "per_day_cap": self.LIMIT_PER_DAY,
                },
                "per_chat": per_chat,
            }

    # ============= Main loop ==================================================
    def _loop(self):
        while True:
            with self._lock:
                if not self._running:
                    break
                self._last_loop = datetime.now(TZ).strftime("%Y-%m-%d %H:%M:%S")
                self._last_reasons = []

            # Respect trading window
            cfg = get_config()
            if not within_window(cfg):
                time.sleep(10)
                continue

            # TODO: plug your live strategy selection & signal generation here.
            # Example (pseudo):
            # signals = generate_signals(cfg)
            signals = []  # keep as no-op until plugged

            # Dispatch
            for sig in signals:
                text = sig.get("text") or sig  # allow string or dict with text
                if not text:
                    continue
                _send_telegram(text)

            # loop throttle
            time.sleep(5)

ENGINE = _Engine()
