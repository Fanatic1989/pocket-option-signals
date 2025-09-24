# auto_worker_all.py
# Fully automated worker that:
#  - fetches latest candles
#  - evaluates your strategies (BASE/TREND/CHOP/CUSTOM with custom rules)
#  - broadcasts signals to ALL tiers via /api/core/send (caps enforced per tier)
#  - handles throttling, backoff, de-duplication (per symbol/tf)
#
# Env:
#   CORE_SEND_URL=https://.../api/core/send
#   CORE_SEND_KEY=<optional secret>
#   SYMBOLS_JSON='[{"symbol":"EURUSD","tf":"M1","expiry":"1m","core":"BASE","cfg":{...}}, ...]'
#   POLL_SECONDS=5
#   MAX_PER_MIN=20
#   MIN_GAP_SEC=3
#   DATA_SOURCE=local|custom   (default: local stub)
#
# You MUST implement fetch_latest_candles() for your live source.
# A simple stub is provided; replace it with your Pocket Options feed or broker REST.
# ------------------------------------------------------------------------------

from __future__ import annotations
import os, time, json, math, random, requests, traceback
from datetime import datetime, timezone
from typing import Dict, Any, List, Tuple
import pandas as pd

# === API config (broadcast to ALL tiers) ===
CORE_SEND_URL  = os.getenv("CORE_SEND_URL",  "http://127.0.0.1:8000/api/core/send").strip()
CORE_STATUS_URL= CORE_SEND_URL.replace("/send", "/status")
CORE_SEND_KEY  = os.getenv("CORE_SEND_KEY", "").strip()
HEADERS = {"Content-Type":"application/json"}
if CORE_SEND_KEY:
    HEADERS["X-API-Key"] = CORE_SEND_KEY

POLL_SECONDS   = int(os.getenv("POLL_SECONDS", "5"))
MAX_PER_MIN    = int(os.getenv("MAX_PER_MIN", "20"))
MIN_GAP_SEC    = int(os.getenv("MIN_GAP_SEC", "3"))
DATA_SOURCE    = os.getenv("DATA_SOURCE", "local").lower()

# Your strategy code must be available as strategy_core.py
from strategy_core import (
    _ensure_cols, _prep_indicators,
    _base_conditions, _trend_conditions, _chop_conditions, _eval_custom,
    _expiry_to_bars,
)

# ------------- Strategy evaluation at the latest bar -------------
def detect_latest_signal(df: pd.DataFrame, core: str, cfg: Dict[str, Any], tf: str, expiry: str):
    """
    Evaluate at the latest completed bar and return:
      (signal: 'BUY'|'SELL'|None, entry_ts (any, from df['timestamp']), ctx: dict)
    Uses the same internals your backtester uses, but only checks the newest index.
    """
    if not isinstance(cfg, dict):
        try: cfg = json.loads(cfg) if isinstance(cfg, str) else {}
        except Exception: cfg = {}
    core = (core or "BASE").upper()
    tf   = (tf or "M5").upper()
    expiry = (expiry or "5m")

    d = _ensure_cols(df)
    if len(d) < 30:
        return None, None, {"reason":"insufficient_bars", "len":len(d)}

    # Indicator params (same shape you used in backtest)
    raw_inds = cfg.get("indicators", {}) if isinstance(cfg.get("indicators", {}), dict) else {}
    sma_obj  = raw_inds.get("sma",  {}) if isinstance(raw_inds.get("sma", {}), dict)  else {}
    rsi_obj  = raw_inds.get("rsi",  {}) if isinstance(raw_inds.get("rsi", {}), dict)  else {}
    st_obj   = raw_inds.get("stoch",{}) if isinstance(raw_inds.get("stoch",{}), dict) else {}

    ip = {
        "sma_period": int(sma_obj.get("period", 50) or 50),
        "rsi_period": int(rsi_obj.get("period", 14) or 14),
        "stoch_k":    int(st_obj.get("k", 14) or 14),
        "stoch_d":    int(st_obj.get("d", 3) or 3),
    }
    d = _prep_indicators(d, ip)
    bars = _expiry_to_bars(tf, expiry)

    # choose signals
    if core == "TREND":
        buy_sig, sell_sig = _trend_conditions(d)
    elif core == "CHOP":
        buy_sig, sell_sig = _chop_conditions(d)
    elif core == "CUSTOM":
        custom = cfg.get("custom", {}) if isinstance(cfg.get("custom", {}), dict) else {}
        buy_sig, sell_sig = _eval_custom(d, custom)
    else:
        buy_sig, sell_sig = _base_conditions(d)

    # we act on the *latest completed* bar (index = len(d)-2), enter at next bar
    i = len(d) - 2
    if i < 1:
        return None, None, {"reason":"too_short_after_prep"}

    sig = None
    if bool(buy_sig.iloc[i]):
        sig = "BUY"
    elif bool(sell_sig.iloc[i]):
        sig = "SELL"

    ts_val = d["timestamp"].iloc[i]
    ctx = {
        "core": core, "tf": tf, "expiry": expiry, "bars_to_expiry": bars,
        "entry_index": i+1,
        "ref_index": i,
        "last_close": float(d["close"].iloc[i]),
        "sma": float(d["sma"].iloc[i]) if "sma" in d else None,
        "rsi": float(d["rsi"].iloc[i]) if "rsi" in d else None,
    }
    return sig, ts_val, ctx

# ------------- Data source (YOU must implement real fetching) -------------
# Replace this with your real feed (Pocket Options/broker/Exchange).
# Must return a DataFrame with at least: ['timestamp','open','high','low','close'].
def fetch_latest_candles(symbol: str, tf: str, limit: int = 300) -> pd.DataFrame:
    """
    STUB implementation: this returns an empty df by default.
    Replace with your real data fetcher. Examples you can wire:
      - Pocket Options API/WS → assemble OHLCV
      - MT5 → python-mt5 to pull copy_rates_from_pos
      - Any broker REST returning klines
    """
    # Example structure (empty):
    return pd.DataFrame(columns=["timestamp","open","high","low","close"])

# ------------- Broadcast sender (ALL tiers) -------------
def send_to_all(text: str) -> Dict[str, Any]:
    payload = {"tier": "all", "text": text}
    try:
        r = requests.post(CORE_SEND_URL, json=payload, headers=HEADERS, timeout=15)
        try:
            return r.json()
        except Exception:
            return {"ok": False, "error": f"HTTP {r.status_code}", "raw": r.text[:300]}
    except Exception as e:
        return {"ok": False, "error": str(e)}

def get_status():
    try:
        r = requests.get(CORE_STATUS_URL, timeout=10)
        return r.json()
    except Exception as e:
        return {"ok": False, "error": str(e)}

# ------------- De-dup + throttle -------------
_last_sent_key_by_symbol: Dict[str, Any] = {}
_last_sends_ts: List[float] = []

def throttle_ok() -> bool:
    import time as _t
    now = _t.time()
    # sliding 60s window
    while _last_sends_ts and now - _last_sends_ts[0] > 60:
        _last_sends_ts.pop(0)
    if len(_last_sends_ts) >= MAX_PER_MIN:
        return False
    if _last_sends_ts and (now - _last_sends_ts[-1] < MIN_GAP_SEC):
        return False
    return True

def mark_sent():
    import time as _t
    _last_sends_ts.append(_t.time())

# ------------- Symbol config -------------
def _load_symbols() -> List[Dict[str, Any]]:
    # Default: one demo entry—replace with your list
    raw = os.getenv("SYMBOLS_JSON", "").strip()
    if not raw:
        return [
            {"symbol":"EURUSD", "tf":"M1", "expiry":"1m", "core":"BASE", "cfg": {}},
        ]
    try:
        arr = json.loads(raw)
        return arr if isinstance(arr, list) else []
    except Exception:
        return []

# ------------- Main loop -------------
def main():
    print("[AUTO] worker starting | url=", CORE_SEND_URL, "| source=", DATA_SOURCE)
    symbols = _load_symbols()
    if not symbols:
        print("[AUTO] No symbols configured via SYMBOLS_JSON; using default EURUSD/M1.")
        symbols = [{"symbol":"EURUSD", "tf":"M1", "expiry":"1m", "core":"BASE", "cfg": {}}]

    backoff = 2
    while True:
        try:
            status = get_status()
            if not status or status.get("running") is not True:
                print("[AUTO] Engine not running or status error, retrying soon:", status)
                time.sleep(backoff)
                backoff = min(backoff*2, 30)
                continue
            backoff = 2

            for s in symbols:
                sym  = s.get("symbol","EURUSD")
                tf   = (s.get("tf") or "M1").upper()
                exp  = s.get("expiry","1m")
                core = (s.get("core") or "BASE").upper()
                cfg  = s.get("cfg") or {}

                df = fetch_latest_candles(sym, tf, limit=300)
                if df is None or df.empty or "close" not in df.columns:
                    continue

                sig, sig_ts, ctx = detect_latest_signal(df, core, cfg, tf, exp)
                if not sig:
                    continue

                # De-dup per symbol/strategy at same reference bar
                dedup_key = f"{sym}:{tf}:{core}:{sig_ts}"
                if _last_sent_key_by_symbol.get(sym) == dedup_key:
                    continue

                # Throttle
                if not throttle_ok():
                    # be gentle: wait a bit and skip this cycle
                    time.sleep(max(0.5, MIN_GAP_SEC))
                    continue

                # Build the message (broadcast to all tiers)
                nowz = datetime.now(timezone.utc).isoformat(timespec="seconds")
                msg = (
                    f"⚡ <b>{sym}</b> • {tf} • {core}\n"
                    f"Signal: <b>{'CALL/BUY' if sig=='BUY' else 'PUT/SELL'}</b>\n"
                    f"Expiry: {exp}\n"
                    f"Ref bar ts: {sig_ts}\n"
                    f"Now: {nowz}Z\n"
                    f"RSI: {ctx.get('rsi')}\n"
                    f"SMA: {ctx.get('sma')}\n"
                )

                res = send_to_all(msg)
                print("[AUTO] SEND_ALL:", json.dumps(res))
                if (res.get("ok") is True) or (isinstance(res.get("results"), dict)):
                    mark_sent()
                    _last_sent_key_by_symbol[sym] = dedup_key

            time.sleep(POLL_SECONDS)
        except KeyboardInterrupt:
            print("[AUTO] stopping (keyboard)")
            break
        except Exception as e:
            print("[AUTO] ERROR:", type(e).__name__, str(e))
            traceback.print_exc()
            time.sleep(3)

if __name__ == "__main__":
    main()
