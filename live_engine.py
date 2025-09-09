import os
import threading
import time
from datetime import datetime
from typing import List, Tuple, Optional

import pandas as pd
import requests

from utils import get_config, within_window, TZ, log
from strategies import run_backtest_core_binary
from data_fetch import deriv_csv_path  # path helper only; avoids circular imports


# ---- all env keys we support for targets (kept in one tuple) ----
TELEGRAM_CHAT_KEYS = (
    "TELEGRAM_CHAT_ID",          # default
    "TELEGRAM_CHAT_ID_TIER1",    # legacy tiers
    "TELEGRAM_CHAT_ID_TIER2",
    "TELEGRAM_CHAT_ID_TIER3",
    # your tier names:
    "TELEGRAM_CHAT_BASIC",
    "TELEGRAM_CHAT_FREE",
    "TELEGRAM_CHAT_PRO",
    "TELEGRAM_CHAT_VIP",
)


def _send_telegram(text: str) -> Tuple[bool, str]:
    """
    Sends a message to any configured chats.
    Env:
      TELEGRAM_BOT_TOKEN (required)
      And any of: TELEGRAM_CHAT_ID, TELEGRAM_CHAT_ID_TIER1..3,
                  TELEGRAM_CHAT_BASIC, TELEGRAM_CHAT_FREE, TELEGRAM_CHAT_PRO, TELEGRAM_CHAT_VIP
    Returns (ok, summary)
    """
    token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    if not token:
        return False, "Missing TELEGRAM_BOT_TOKEN"

    # Collect chat targets (de-duped, keep order)
    seen = set()
    targets = []
    for key in TELEGRAM_CHAT_KEYS:
        cid = os.getenv(key, "").strip()
        if cid and cid not in seen:
            targets.append((key, cid))
            seen.add(cid)

    if not targets:
        return False, "No TELEGRAM_CHAT_* provided"

    url = f"https://api.telegram.org/bot{token}/sendMessage"
    ok_count = 0
    details = []
    for key, chat in targets:
        try:
            resp = requests.post(
                url,
                json={"chat_id": chat, "text": text, "parse_mode": "HTML", "disable_web_page_preview": True},
                timeout=12,
            )
            data = {}
            try:
                data = resp.json()
            except Exception:
                pass
            if resp.ok and isinstance(data, dict) and data.get("ok") is True:
                ok_count += 1
                details.append(f"{key}:{chat} ok")
            else:
                # common causes: bot not member/admin, wrong chat id (groups need negative id), privacy mode
                err = data.get("description") if isinstance(data, dict) else resp.text
                details.append(f"{key}:{chat} error: {err}")
        except Exception as e:
            details.append(f"{key}:{chat} exception: {e}")

    if ok_count:
        return True, f"sent to {ok_count}/{len(targets)} chats; " + "; ".join(details)
    return False, "; ".join(details) if details else "unknown telegram error"


def _format_signal_msg(sym: str, tf: str, expiry: str, direction: str, tstamp: str, price: Optional[float]) -> str:
    p = f"{price:.5f}" if isinstance(price, (int, float)) else "-"
    return (
        f"📣 <b>Pocket Option Signal</b>\n"
        f"Pair: <b>{sym}</b>\n"
        f"TF: <b>{tf}</b> | Expiry: <b>{expiry}</b>\n"
        f"Direction: <b>{direction}</b>\n"
        f"Time: <code>{tstamp}</code>\n"
        f"Price: <code>{p}</code>"
    )


class LiveEngine:
    """
    Lightweight live loop that:
      - Respects the trading window (utils.within_window)
      - Uses whatever strategy is currently enabled
      - Reads CSVs saved via /deriv_fetch (data_fetch.deriv_csv_path)
      - Derives a signal by running a tiny backtest slice and grabbing the last trade
      - Sends signals to Telegram when present
      - Provides debug notes and a manual "step once"
    """

    def __init__(self) -> None:
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._stop_ev = threading.Event()

        self._last_error: Optional[str] = None
        self._last_loop: Optional[str] = None
        self._sent = 0

        self._debug = False
        self._last_reasons: List[str] = []

        # loop cadence in seconds
        self._period_sec = int(os.getenv("LIVE_LOOP_SECONDS", "30"))

    # -------- public API ----------

    def status(self):
        return {
            "running": self._running,
            "sent": self._sent,
            "last_loop": self._last_loop,
            "last_error": self._last_error,
            "debug": self._debug,
            "last_reasons": self._last_reasons[-50:],  # tail
        }

    def set_debug(self, on: bool):
        self._debug = bool(on)
        self._note(f"debug={'ON' if self._debug else 'OFF'}")

    def start(self):
        if self._running:
            return True, "already running"
        self._stop_ev.clear()
        self._thread = threading.Thread(target=self._run_forever, name="LiveEngine", daemon=True)
        self._running = True
        self._thread.start()
        return True, "started"

    def stop(self):
        if not self._running:
            return True, "already stopped"
        self._stop_ev.set()
        self._running = False
        return True, "stopped"

    def step_once(self):
        """Run exactly one iteration without flipping running state."""
        try:
            self._note("manual step_once invoked")
            self._run_once()
            self._last_loop = datetime.now(TZ).strftime("%Y-%m-%d %H:%M:%S")
            return True, "stepped"
        except Exception as e:
            self._last_error = str(e)
            self._note(f"error: {e}")
            return False, str(e)

    # -------- internals ----------

    def _run_forever(self):
        self._note("loop start")
        while not self._stop_ev.is_set():
            try:
                self._run_once()
                self._last_loop = datetime.now(TZ).strftime("%Y-%m-%d %H:%M:%S")
                self._last_error = None
            except Exception as e:
                self._last_error = str(e)
                self._note(f"loop error: {e}")
            # sleep
            for _ in range(self._period_sec):
                if self._stop_ev.is_set():
                    break
                time.sleep(1)
        self._note("loop end")

    def _run_once(self):
        cfg = get_config()  # live cfg
        # 1) window
        if not within_window(cfg):
            self._note("skip: outside trading window")
            return

        # 2) determine which strategy to run (first enabled wins)
        strat = self._pick_strategy(cfg)
        if not strat:
            self._note("skip: no enabled strategy")
            return

        tf = cfg.get("live_tf", "M5").upper()
        expiry = cfg.get("live_expiry", "5m")
        symbols = cfg.get("symbols") or []
        if not symbols:
            self._note("skip: no active symbols")
            return

        # 3) evaluate each symbol from its latest CSV
        for sym in symbols:
            try:
                df = self._load_df(sym, tf)
            except Exception as e:
                self._note(f"{sym}: load error {e}")
                continue

            if df is None or len(df) < 50:
                self._note(f"{sym}: skip no/short data")
                continue

            # Use the same engine as backtest to get decisions on latest bars
            tail = df.tail(300).reset_index(drop=True)
            try:
                bt = run_backtest_core_binary(tail, strat["core"], strat["cfg"], tf, expiry)
            except Exception as e:
                self._note(f"{sym}: eval error {e}")
                continue

            if not bt.rows:
                self._note(f"{sym}: no signal")
                continue

            last = bt.rows[-1]  # dict with keys like time_in, dir, entry, outcome?
            t_in = last.get("time_in") or last.get("timestamp") or ""
            direction = last.get("dir") or last.get("signal") or ""
            entry = last.get("entry")

            if not direction:
                self._note(f"{sym}: last row has no direction")
                continue

            # Send
            msg = _format_signal_msg(sym, tf, expiry, direction.upper(), str(t_in), entry)
            ok, info = _send_telegram(msg)
            if ok:
                self._sent += 1
                self._note(f"{sym}: SENT {direction.upper()}")
            else:
                self._note(f"{sym}: telegram fail {info}")

    def _pick_strategy(self, cfg) -> Optional[dict]:
        """
        Returns {"core": "BASE"/"TREND"/"CHOP"/"CUSTOM", "cfg": full_cfg_with_custom}
        First enabled among: BASE, TREND, CHOP, CUSTOM1..3
        """
        strat_cfg = cfg.get("strategies", {})
        order = ["BASE", "TREND", "CHOP", "CUSTOM1", "CUSTOM2", "CUSTOM3"]
        for key in order:
            enabled = bool(strat_cfg.get(key, {}).get("enabled"))
            if not enabled:
                continue
            if key.startswith("CUSTOM"):
                slot = key[-1]
                custom = cfg.get(f"custom{slot}", {})
                run_cfg = dict(cfg)
                run_cfg["custom"] = custom
                return {"core": "CUSTOM", "cfg": run_cfg}
            return {"core": key, "cfg": cfg}
        return None

    def _load_df(self, sym: str, tf: str) -> Optional[pd.DataFrame]:
        gran_map = {"M1":60,"M2":120,"M3":180,"M5":300,"M10":600,"M15":900,"M30":1800,"H1":3600,"H4":14400,"D1":86400}
        gran = gran_map.get(tf, 300)
        path = deriv_csv_path(sym, gran)
        if not os.path.exists(path):
            self._note(f"{sym}: no CSV {path}")
            return None
        df = pd.read_csv(path)
        if df.empty:
            return None
        df.columns = [c.strip().lower() for c in df.columns]
        if "timestamp" in df.columns:
            try:
                df["timestamp"] = pd.to_datetime(df["timestamp"])
            except Exception:
                pass
        for c in ("open", "high", "low", "close"):
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df.dropna(subset=["close"]).reset_index(drop=True)
        return df

    def _note(self, msg: str):
        if self._debug:
            self._last_reasons.append(msg)


# Singleton
ENGINE = LiveEngine()


def tg_test() -> Tuple[bool, str]:
    return _send_telegram("✅ Telegram test: Pocket Option Signals dashboard is connected.")
