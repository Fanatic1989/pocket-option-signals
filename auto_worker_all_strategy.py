# auto_worker_all_strategy.py
# Fully automated: fetch candles (Deriv), evaluate strategy.py, broadcast to ALL tiers.
from __future__ import annotations
import os, time, json, requests, traceback
from datetime import datetime, timezone
from typing import Dict, Any, List
import pandas as pd

# ===== Web service endpoints (already running on Render) =====
CORE_SEND_URL   = os.getenv("CORE_SEND_URL",  "http://127.0.0.1:8000/api/core/send").strip()
CORE_STATUS_URL = CORE_SEND_URL.replace("/send", "/status")
CORE_SEND_KEY   = os.getenv("CORE_SEND_KEY", "").strip()
HEADERS = {"Content-Type": "application/json"}
if CORE_SEND_KEY:
    HEADERS["X-API-Key"] = CORE_SEND_KEY

POLL_SECONDS = int(os.getenv("POLL_SECONDS", "5"))
MIN_GAP_SEC  = int(os.getenv("MIN_GAP_SEC", "3"))
MAX_PER_MIN  = int(os.getenv("MAX_PER_MIN", "20"))

# ===== Your modules =====
from data_fetch import fetch_latest_candles
# Import directly from your provided strategy.py
from strategy import (
    _ensure_cols, _prep_indicators,
    _base_conditions, _trend_conditions, _chop_conditions, _eval_custom,
    _expiry_to_bars,
)

# If you keep strategies in rules.py, import them; otherwise define below.
try:
    from rules import get_symbol_strategies  # returns list[dict]
except Exception:
    def get_symbol_strategies():
        # Minimal default if rules.py is absent; edit your rules.py instead.
        return [{
            "symbol": "EURUSD", "tf": "M1", "expiry": "1m",
            "core": "BASE", "name": "EURUSD M1 BASE",
            "cfg": { "indicators": { "sma": {"period":50}, "rsi":{"period":14}, "stoch":{"k":14,"d":3} } }
        }]

# ===== Throttle + de-dup =====
_last_sends_ts: List[float] = []
_last_sent_key: Dict[str, str] = {}   # per symbol

def _throttle_ok() -> bool:
    now = time.time()
    while _last_sends_ts and (now - _last_sends_ts[0] > 60):
        _last_sends_ts.pop(0)
    if len(_last_sends_ts) >= MAX_PER_MIN: return False
    if _last_sends_ts and (now - _last_sends_ts[-1] < MIN_GAP_SEC): return False
    return True

def _mark_sent(): _last_sends_ts.append(time.time())

def _get_status():
    try: return requests.get(CORE_STATUS_URL, timeout=10).json()
    except Exception as e: return {"ok": False, "error": str(e)}

def _send_all(text: str) -> Dict[str, Any]:
    try:
        r = requests.post(CORE_SEND_URL, json={"tier": "all", "text": text}, headers=HEADERS, timeout=15)
        try: return r.json()
        except Exception: return {"ok": False, "error": f"HTTP {r.status_code}", "raw": r.text[:300]}
    except Exception as e:
        return {"ok": False, "error": str(e)}

# ===== Evaluate strategy.py at latest completed bar =====
def _detect_latest_signal(df: pd.DataFrame, core: str, cfg: Dict[str, Any], tf: str, expiry: str):
    """
    Returns ('BUY'|'SELL'|None, ref_ts:int, context:dict).
    Uses your strategy.py core blocks and CUSTOM evaluator.
    """
    coreU = (core or "BASE").upper()
    d = _ensure_cols(df)
    if len(d) < 30:
        return None, None, {"reason": "insufficient_bars", "len": len(d)}

    # indicator params compatible with strategy.py
    ind = cfg.get("indicators", {}) if isinstance(cfg.get("indicators", {}), dict) else {}
    sma = ind.get("sma", {}) if isinstance(ind.get("sma", {}), dict) else {}
    rsi = ind.get("rsi", {}) if isinstance(ind.get("rsi", {}), dict) else {}
    st  = ind.get("stoch", {}) if isinstance(ind.get("stoch", {}), dict) else {}
    ip = {
        "sma_period": int(sma.get("period", 50) or 50),
        "rsi_period": int(rsi.get("period", 14) or 14),
        "stoch_k":    int(st.get("k", 14) or 14),
        "stoch_d":    int(st.get("d", 3) or 3),
    }
    d = _prep_indicators(d, ip)

    if coreU == "TREND":
        buy_sig, sell_sig = _trend_conditions(d)
    elif coreU == "CHOP":
        buy_sig, sell_sig = _chop_conditions(d)
    elif coreU == "CUSTOM":
        # Your strategy.py expects dict with {buy_rule/sell_rule} booleans/thresholds
        custom = cfg.get("custom", {}) if isinstance(cfg.get("custom", {}), dict) else {}
        buy_sig, sell_sig = _eval_custom(d, custom)
    else:
        buy_sig, sell_sig = _base_conditions(d)

    i = len(d) - 2  # latest completed bar
    if i < 1: return None, None, {"reason": "short_after_prep"}

    if bool(buy_sig.iloc[i]):
        sig = "BUY"
    elif bool(sell_sig.iloc[i]):
        sig = "SELL"
    else:
        sig = None

    ctx = {
        "last_close": float(d["close"].iloc[i]),
        "sma": float(d["sma"].iloc[i]) if "sma" in d else None,
        "rsi": float(d["rsi"].iloc[i]) if "rsi" in d else None,
        "bars_to_expiry": _expiry_to_bars(tf, expiry),
    }
    return sig, int(d["timestamp"].iloc[i]), ctx

# ===== Main loop =====
def main():
    print("[AUTO-STRAT] worker up | send=", CORE_SEND_URL)
    backoff = 2
    while True:
        try:
            st = _get_status()
            if st.get("running") is not True:
                print("[AUTO-STRAT] engine not running; retry", st)
                time.sleep(backoff); backoff = min(backoff*2, 30)
                continue
            backoff = 2

            for s in get_symbol_strategies():
                symbol = s.get("symbol", "EURUSD")
                tf     = (s.get("tf") or "M1").upper()
                expiry = s.get("expiry", "1m")
                core   = (s.get("core") or "BASE").upper()
                name   = s.get("name") or f"{symbol} {tf} {core}"
                cfg    = s.get("cfg", {}) or {}

                df = fetch_latest_candles(symbol, tf, limit=300)
                if df is None or df.empty or "close" not in df.columns:
                    continue

                sig, ref_ts, ctx = _detect_latest_signal(df, core, cfg, tf, expiry)
                if not sig:
                    continue

                # de-dup per symbol+tf+core+ref_ts+side
                key = f"{symbol}:{tf}:{core}:{ref_ts}:{sig}"
                if _last_sent_key.get(symbol) == key:
                    continue

                if not _throttle_ok():
                    time.sleep(max(0.5, MIN_GAP_SEC))
                    continue

                nowz = datetime.now(timezone.utc).isoformat(timespec="seconds")
                msg = (
                    f"⚡ <b>{symbol}</b> • {tf} • {name}\n"
                    f"Signal: <b>{'CALL/BUY' if sig=='BUY' else 'PUT/SELL'}</b>\n"
                    f"Expiry: {expiry}\n"
                    f"Ref bar ts: {ref_ts}\n"
                    f"Now: {nowz}Z\n"
                    f"RSI: {ctx.get('rsi')} | SMA: {ctx.get('sma')} | bars→exp: {ctx.get('bars_to_expiry')}\n"
                )

                res = _send_all(msg)
                print("[AUTO-STRAT] SEND ALL =>", json.dumps(res))
                if res.get("ok") is True or isinstance(res.get("results"), dict):
                    _mark_sent()
                    _last_sent_key[symbol] = key

            time.sleep(POLL_SECONDS)
        except KeyboardInterrupt:
            print("[AUTO-STRAT] stop (keyboard)"); break
        except Exception as e:
            print("[AUTO-STRAT] ERROR:", type(e).__name__, str(e))
            traceback.print_exc()
            time.sleep(3)

if __name__ == "__main__":
    main()
