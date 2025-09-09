import os, threading, time, json, traceback
from datetime import datetime
import pandas as pd
import requests

from utils import TZ, get_config, within_window, log
from strategies import run_backtest_core_binary
from routes import _deriv_csv_path, _fetch_one_symbol  # reuse existing fetchers

BOT = os.getenv("TELEGRAM_BOT_TOKEN", "")
CHAT_TIER = {
    1: os.getenv("TG_CHAT_TIER1", ""),
    2: os.getenv("TG_CHAT_TIER2", ""),
    3: os.getenv("TG_CHAT_TIER3", "")
}
CHAT_ADMIN = os.getenv("TG_CHAT_ADMIN", "")

def tg_send(chat_id: str, text: str, parse_mode: str = "HTML"):
    if not BOT or not chat_id:
        return False, "Missing BOT token or chat_id"
    try:
        url = f"https://api.telegram.org/bot{BOT}/sendMessage"
        r = requests.post(url, timeout=10, json={
            "chat_id": chat_id, "text": text, "parse_mode": parse_mode, "disable_web_page_preview": True
        })
        ok = r.status_code == 200
        if not ok:
            log("ERROR", f"Telegram send fail {r.status_code}: {r.text}")
        return ok, (r.text if not ok else "ok")
    except Exception as e:
        log("ERROR", f"Telegram exception: {e}")
        return False, str(e)

def tg_test():
    return tg_send(CHAT_ADMIN or CHAT_TIER.get(1) or CHAT_TIER.get(2) or CHAT_TIER.get(3),
                   f"✅ Live ping {datetime.now(TZ).strftime('%Y-%m-%d %H:%M:%S')}")

class LiveEngine:
    def __init__(self):
        self._lock = threading.Lock()
        self._thr = None
        self._stop = threading.Event()
        self._status = {"running": False, "last_loop": None, "last_error": None, "sent": 0}
        self._last_sig = {}

    def status(self):
        with self._lock:
            return dict(self._status)

    def start(self):
        with self._lock:
            if self._thr and self._thr.is_alive():
                return False, "already running"
            self._stop.clear()
            self._thr = threading.Thread(target=self._run, daemon=True)
            self._thr.start()
            self._status["running"] = True
            self._status["last_error"] = None
        return True, "started"

    def stop(self):
        with self._lock:
            self._stop.set()
            self._status["running"] = False
        return True, "stopping"

    def _run(self):
        SLEEP = int(os.getenv("LIVE_LOOP_SEC", "55"))
        COUNT = int(os.getenv("LIVE_FETCH_CANDLES", "300"))

        while not self._stop.is_set():
            try:
                cfg = get_config()
                self._status["last_loop"] = datetime.now(TZ).strftime("%Y-%m-%d %H:%M:%S")

                if not within_window(cfg):
                    time.sleep(SLEEP); continue

                symbols = cfg.get("symbols") or []
                if not symbols:
                    time.sleep(SLEEP); continue

                strategy = None
                for name, opts in (cfg.get("strategies") or {}).items():
                    if opts.get("enabled"): strategy = name; break
                if not strategy:
                    time.sleep(SLEEP); continue

                tf = (cfg.get("live_tf") or "M5").upper()
                expiry = (cfg.get("live_expiry") or "5m")
                gran_map = {"M1":60,"M2":120,"M3":180,"M5":300,"M10":600,"M15":900,"M30":1800,"H1":3600,"H4":14400,"D1":86400}
                gran = gran_map.get(tf, 300)

                cfg_for_run = dict(cfg)
                if strategy.upper() in ("CUSTOM1","CUSTOM2","CUSTOM3"):
                    sid = strategy[-1]
                    cfg_for_run["custom"] = cfg.get(f"custom{sid}", {})
                    core_strategy = "CUSTOM"
                else:
                    core_strategy = strategy

                sent_now = 0
                for sym in symbols:
                    try:
                        _fetch_one_symbol(os.getenv("DERIV_APP_ID", "1089"), sym, gran, COUNT)
                        path = _deriv_csv_path(sym, gran)
                        df = pd.read_csv(path)
                        df.columns = [c.strip().lower() for c in df.columns]
                        if "timestamp" in df.columns:
                            df["timestamp"] = pd.to_datetime(df["timestamp"])
                        for c in ("open","high","low","close"):
                            if c in df.columns:
                                df[c] = pd.to_numeric(df[c], errors="coerce")
                        df = df.dropna(subset=["close"]).sort_values("timestamp").reset_index(drop=True)
                        if len(df) < 50: continue

                        bt = run_backtest_core_binary(df, core_strategy, cfg_for_run, tf, expiry)
                        last = bt.rows[-1] if bt.rows else None
                        if not last: continue

                        key = (sym, tf)
                        last_key = f"{last.get('time_in') or last.get('idx')}"
                        prev_key = self._last_sig.get(key)
                        if prev_key == last_key: continue

                        tier = int(os.getenv("LIVE_TIER", "2"))
                        chat_id = CHAT_TIER.get(tier) or CHAT_ADMIN
                        if not chat_id: continue

                        dir_txt = str(last.get("dir","")).upper()
                        entry = last.get("entry")
                        expiry_time = last.get("time_out") or ""
                        msg = (
                            f"📢 <b>Signal</b>\n"
                            f"• Pair: <b>{sym}</b>\n"
                            f"• TF: <b>{tf}</b>  • Expiry: <b>{expiry}</b>\n"
                            f"• Strategy: <b>{strategy}</b>\n"
                            f"• Action: <b>{dir_txt or 'N/A'}</b>\n"
                            f"• Entry: <code>{entry if entry is not None else '—'}</code>\n"
                            f"• Expires: <code>{expiry_time}</code>\n"
                            f"• Time: <code>{datetime.now(TZ).strftime('%Y-%m-%d %H:%M:%S')}</code>"
                        )
                        ok, err = tg_send(chat_id, msg)
                        if ok:
                            self._last_sig[key] = last_key
                            sent_now += 1
                            self._status["sent"] += 1
                            log("INFO", f"TELEGRAM SENT [{tier}] {sym} {tf} {dir_txt}")
                        else:
                            self._status["last_error"] = f"Telegram: {err}"
                    except Exception as e:
                        self._status["last_error"] = f"{sym} error: {e}"
                        log("ERROR", f"Live loop error {sym}: {e}\n{traceback.format_exc()}")

                time.sleep(SLEEP if sent_now == 0 else 3)
            except Exception as e:
                self._status["last_error"] = f"loop crash: {e}"
                log("ERROR", f"Live loop crash: {e}\n{traceback.format_exc()}")
                time.sleep(5)

ENGINE = LiveEngine()
