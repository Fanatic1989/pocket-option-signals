# --- Deriv fetcher.py snippet (async) ---
import asyncio, json
import pandas as pd
import websockets

DERIV_ENDPOINT = os.getenv("DERIV_ENDPOINT", "").strip()

async def deriv_candles(symbol: str, granularity_sec: int = 60, count: int = 500) -> pd.DataFrame:
    """
    Fetch recent candles from Deriv (no auth needed):
    - symbol: e.g., 'frxEURUSD'
    - granularity_sec: 60 for M1, 300 for M5, 900 for M15, etc.
    """
    if not DERIV_ENDPOINT:
        raise RuntimeError("DERIV_ENDPOINT not set")

    req = {
        "ticks_history": symbol,
        "granularity": granularity_sec,
        "count": count,
        "end": "latest",
        "style": "candles"
    }

    async with websockets.connect(DERIV_ENDPOINT, ping_interval=20, ping_timeout=20) as ws:
        await ws.send(json.dumps(req))
        resp = json.loads(await ws.recv())

    candles = resp.get("candles") or []
    rows = []
    for c in candles:
        # Deriv returns epoch seconds + o/h/l/c
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
    df = pd.DataFrame(rows).sort_values("time").reset_index(drop=True)
    return df

async def deriv_m1(symbol: str, count: int = 500) -> pd.DataFrame:
    return await deriv_candles(symbol, 60, count)

async def deriv_m5(symbol: str, count: int = 500) -> pd.DataFrame:
    return await deriv_candles(symbol, 300, count)

async def deriv_m15(symbol: str, count: int = 500) -> pd.DataFrame:
    return await deriv_candles(symbol, 900, count)
