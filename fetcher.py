# deriv_fetcher.py
# -----------------------------------------------------------------------------
# Deriv market data fetcher (WebSocket, no auth needed for public history).
# - Uses ticks_history with style="candles"
# - Granularity in seconds: 60 (M1), 300 (M5), 900 (M15), etc.
# - Safe retries + timeout + basic validation
# - Returns pandas DataFrame with columns: time,o,h,l,c,volume,complete
#
# Env:
#   DERIV_APP_ID      = 12345                (required by Deriv to tag your app)
#   DERIV_ENDPOINT    = wss://ws.deriv.com/websockets/v3?app_id=${DERIV_APP_ID}
#
# Example usage:
#   from deriv_fetcher import deriv_m1, last_completed_idx
#   df = await deriv_m1("frxEURUSD", 400)
#   idx = last_completed_idx(df)
# -----------------------------------------------------------------------------

from __future__ import annotations

import os
import json
import asyncio
import logging
from typing import Optional, Dict, Any

import pandas as pd
import websockets
from websockets.exceptions import ConnectionClosed, InvalidStatusCode, InvalidURI

log = logging.getLogger("deriv-fetcher")

DERIV_ENDPOINT = os.getenv("DERIV_ENDPOINT", "").strip()
if not DERIV_ENDPOINT:
    # You can set this at runtime; we only warn here to help during local dev
    log.warning("DERIV_ENDPOINT is empty. Set wss endpoint, e.g. wss://ws.deriv.com/websockets/v3?app_id=XXXX")

# ---------------------------
# Internal helpers
# ---------------------------

def _validate_candle_payload(resp: Dict[str, Any]) -> None:
    """Raise if Deriv response indicates an error or missing fields."""
    if "error" in resp and resp["error"]:
        raise RuntimeError(f"Deriv error {resp['error'].get('code')}: {resp['error'].get('message')}")
    if ("candles" not in resp) or (resp["candles"] is None):
        # Some responses wrap candles differently; ticks_history always returns 'candles' for style=candles
        raise RuntimeError("No 'candles' in Deriv response")

def _rows_from_candles(candles: list[dict]) -> list[dict]:
    rows = []
    for c in candles:
        try:
            ts = pd.to_datetime(c["epoch"], unit="s", utc=True)
            rows.append({
                "time": ts,
                "o": float(c["open"]),
                "h": float(c["high"]),
                "l": float(c["low"]),
                "c": float(c["close"]),
                "volume": int(c.get("tick_count", 0)),
                "complete": True  # Deriv returns closed candles
            })
        except Exception:
            continue
    return rows

async def _ws_request(payload: Dict[str, Any], endpoint: Optional[str] = None, timeout: float = 20.0) -> Dict[str, Any]:
    """Send a single request and return one response message."""
    url = (endpoint or DERIV_ENDPOINT)
    if not url:
        raise RuntimeError("DERIV_ENDPOINT not set")

    async with websockets.connect(url, ping_interval=20, ping_timeout=20) as ws:
        await asyncio.wait_for(ws.send(json.dumps(payload)), timeout=timeout)
        raw = await asyncio.wait_for(ws.recv(), timeout=timeout)
        return json.loads(raw)

async def _ws_request_retry(payload: Dict[str, Any], endpoint: Optional[str] = None, attempts: int = 4, base_delay: float = 0.8) -> Dict[str, Any]:
    """Retry wrapper for transient WS errors."""
    last_err: Optional[Exception] = None
    for i in range(attempts):
        try:
            return await _ws_request(payload, endpoint=endpoint)
        except (ConnectionClosed, InvalidStatusCode, InvalidURI, asyncio.TimeoutError) as e:
            last_err = e
            delay = base_delay * (2 ** i)
            log.warning(f"Deriv WS transient error ({e}); retry {i+1}/{attempts} in {delay:.1f}s")
            await asyncio.sleep(delay)
        except Exception as e:
            # Non-transient errors
            raise
    # Retries exhausted
    raise RuntimeError(f"Deriv WS failed after {attempts} attempts: {last_err}")

# ---------------------------
# Public API
# ---------------------------

async def deriv_candles(
    symbol: str,
    granularity_sec: int = 60,
    count: int = 500,
    endpoint: Optional[str] = None
) -> pd.DataFrame:
    """
    Fetch recent candles from Deriv (no authorization required).
      - symbol: e.g., 'frxEURUSD'
      - granularity_sec: 60 (M1), 300 (M5), 900 (M15), etc.
      - count: number of candles

    Returns:
      DataFrame sorted by 'time' with columns: time,o,h,l,c,volume,complete
    """
    if not symbol:
        raise ValueError("symbol is required")

    payload = {
        "ticks_history": symbol,
        "granularity": granularity_sec,
        "count": count,
        "end": "latest",
        "style": "candles"
    }

    resp = await _ws_request_retry(payload, endpoint=endpoint)
    _validate_candle_payload(resp)
    candles = resp.get("candles") or []
    rows = _rows_from_candles(candles)

    df = pd.DataFrame(rows).dropna().sort_values("time").reset_index(drop=True)
    return df

async def deriv_m1(symbol: str, count: int = 500, endpoint: Optional[str] = None) -> pd.DataFrame:
    return await deriv_candles(symbol, 60, count, endpoint=endpoint)

async def deriv_m5(symbol: str, count: int = 500, endpoint: Optional[str] = None) -> pd.DataFrame:
    return await deriv_candles(symbol, 300, count, endpoint=endpoint)

async def deriv_m15(symbol: str, count: int = 500, endpoint: Optional[str] = None) -> pd.DataFrame:
    return await deriv_candles(symbol, 900, count, endpoint=endpoint)

def last_completed_idx(df: pd.DataFrame) -> Optional[int]:
    """
    For consistency with your OANDA code:
      - If DF is empty: None
      - All Deriv candles returned are complete, so the last row is fine.
    """
    if df is None or df.empty:
        return None
    return int(df.index[-1])

# Optional: simple connectivity test (run as a script)
if __name__ == "__main__":
    import asyncio as _asyncio

    async def _demo():
        sym = os.getenv("DERIV_DEMO_SYMBOL", "frxEURUSD")
        print(f"Fetching last 10 M1 candles for {sym} ...")
        df = await deriv_m1(sym, 10)
        print(df.tail(3))

    _asyncio.run(_demo())
