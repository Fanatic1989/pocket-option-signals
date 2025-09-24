# data_fetch.py
# Fetch OHLC candles from Deriv WebSocket API.
# Falls back to stub random candles if Deriv is unavailable.
#
# Env:
#   DATA_MODE=deriv | stub      (default: stub)
#   DERIV_APP_ID=1089           (public demo app_id OK for testing)
#   DERIV_ENDPOINT=wss://ws.deriv.com/websockets/v3  (default)
#
# Returns pandas.DataFrame with columns: ['timestamp','open','high','low','close']

import os, json, time
from typing import Dict, Any
import pandas as pd
import numpy as np

MODE = os.getenv("DATA_MODE", "deriv").lower()
DERIV_APP_ID = os.getenv("DERIV_APP_ID", "1089").strip()
DERIV_ENDPOINT = os.getenv("DERIV_ENDPOINT", "wss://ws.deriv.com/websockets/v3").strip()

# --- light dependency handling: websocket-client is widely available ---
# pip install websocket-client
try:
    import websocket
except Exception:
    websocket = None

# ---------- Symbol mapping (your strategy can still use "EURUSD") ----------
_DERIV_SYMBOLS = {
    # FX majors
    "EURUSD": "frxEURUSD",
    "GBPUSD": "frxGBPUSD",
    "USDJPY": "frxUSDJPY",
    "AUDUSD": "frxAUDUSD",
    "USDCAD": "frxUSDCAD",
    "USDCHF": "frxUSDCHF",
    "NZDUSD": "frxNZDUSD",
    # add more as needed; see Deriv's active_symbols list for full catalog
}

# ---------- TF → Deriv granularity (seconds) ----------
_TF_TO_SEC = {
    "M1": 60, "M2": 120, "M3": 180, "M5": 300, "M10": 600, "M15": 900,
    "M30": 1800, "H1": 3600, "H4": 14400, "D1": 86400
}

# ==================== Deriv candles via WebSocket ====================

def _deriv_symbol(sym: str) -> str:
    s = (sym or "").upper().replace("/", "")
    return _DERIV_SYMBOLS.get(s, s)  # allow passing a Deriv symbol directly

def _granularity(tf: str) -> int:
    return _TF_TO_SEC.get((tf or "M1").upper(), 60)

def _fetch_deriv_candles(symbol: str, tf: str, limit: int = 300) -> pd.DataFrame:
    if websocket is None:
        # No websocket-client library installed -> fall back
        return _stub_df(symbol, tf, limit)

    d_symbol = _deriv_symbol(symbol)
    gran = _granularity(tf)
    url = f"{DERIV_ENDPOINT}?app_id={DERIV_APP_ID}"

    # Build request (ticks_history) for candles
    req = {
        "ticks_history": d_symbol,
        "style": "candles",
        "granularity": gran,
        "adjust_start_time": 1,
        "count": int(limit),
        "end": "latest"
    }

    ws = None
    try:
        ws = websocket.create_connection(url, timeout=10)
        ws.send(json.dumps(req))
        # Deriv may return a single big message or a few chunks; read until we get 'candles'
        deadline = time.time() + 10
        candles = None
        while time.time() < deadline:
            raw = ws.recv()
            if not raw:
                break
            msg = json.loads(raw)
            if "error" in msg:
                # e.g., invalid symbol or granularity
                # print for diagnostics but return stub to keep worker alive
                # print("Deriv error:", msg["error"])
                break
            if "candles" in msg:
                candles = msg["candles"]
                break
        if not candles:
            return _stub_df(symbol, tf, limit)

        rows = []
        for c in candles:
            # Deriv fields: epoch, open, high, low, close
            rows.append({
                "timestamp": int(c["epoch"]),
                "open": float(c["open"]),
                "high": float(c["high"]),
                "low":  float(c["low"]),
                "close": float(c["close"]),
            })
        df = pd.DataFrame(rows)
        # make sure we only keep the expected shape and chronological order
        if not df.empty:
            df = df.sort_values("timestamp").reset_index(drop=True)
            df = df[["timestamp", "open", "high", "low", "close"]]
        return df
    except Exception:
        return _stub_df(symbol, tf, limit)
    finally:
        try:
            if ws:
                ws.close()
        except Exception:
            pass

# ==================== Stub/random-walk fallback ====================

_state = {}
def _stub_df(symbol: str, tf: str, limit: int = 300) -> pd.DataFrame:
    key = f"{symbol}:{tf}"
    last = _state.get(key, 1.0)
    rows = []
    step = _granularity(tf)
    now = int(time.time())
    start = now - limit * step
    for i in range(limit):
        ts = start + i * step
        drift = np.random.normal(0, 0.0008)
        openp = last
        closep = max(0.0001, openp + drift)
        high = max(openp, closep) + abs(np.random.normal(0, 0.0004))
        low  = min(openp, closep) - abs(np.random.normal(0, 0.0004))
        rows.append({"timestamp": ts, "open": openp, "high": high, "low": low, "close": closep})
        last = closep
    _state[key] = last
    return pd.DataFrame(rows)

# ==================== Public API ====================

def fetch_latest_candles(symbol: str, tf: str, limit: int = 300) -> pd.DataFrame:
    """
    Main entrypoint used by auto_worker_all.py.
    """
    mode = MODE
    if mode == "deriv":
        return _fetch_deriv_candles(symbol, tf, limit)
    # future: if you add "mt5"/"rest", branch here
    return _stub_df(symbol, tf, limit)
