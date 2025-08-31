import os, requests
from datetime import datetime, timezone, timedelta
import pandas as pd

OANDA_API_KEY = os.getenv("OANDA_API_KEY","")
OANDA_ENV     = os.getenv("OANDA_ENV","practice").lower()
HOST = "https://api-fxtrade.oanda.com" if OANDA_ENV=="live" else "https://api-fxpractice.oanda.com"
HDR  = {"Authorization": f"Bearer {OANDA_API_KEY}"} if OANDA_API_KEY else {}

def _to_dt(ts: str) -> datetime:
    # Samples: "2025-08-29T20:59:00.000000000Z"
    return datetime.fromisoformat(ts.replace("Z","+00:00")).astimezone(timezone.utc)

def oanda_fetch_m1(instrument: str, count: int = 220) -> pd.DataFrame:
    """Return M1 OHLC dataframe with UTC datetime index."""
    url = f"{HOST}/v3/instruments/{instrument}/candles"
    r = requests.get(url, params={"granularity":"M1","count":count,"price":"M"}, headers=HDR, timeout=20)
    r.raise_for_status()
    js = r.json()
    rows = []
    for c in js.get("candles", []):
        if not c.get("complete"): 
            continue
        mid = c.get("mid", {})
        rows.append({
            "time": _to_dt(c["time"]),
            "open": float(mid["o"]),
            "high": float(mid["h"]),
            "low":  float(mid["l"]),
            "close":float(mid["c"]),
        })
    if not rows:
        return pd.DataFrame(columns=["open","high","low","close"])
    df = pd.DataFrame(rows).set_index("time").sort_index()
    return df

def oanda_price_at(instrument: str, dt_utc: datetime) -> float | None:
    """
    Return the candle close near dt_utc (M1). No 'granularity' kw to avoid past errors.
    """
    if dt_utc.tzinfo is None:
        dt_utc = dt_utc.replace(tzinfo=timezone.utc)
    # Pull a small window around dt_utc
    df = oanda_fetch_m1(instrument, count=20)
    if df.empty:
        return None
    # pick the candle at or just after dt_utc, else last known before
    ix = df.index.get_indexer([dt_utc], method="nearest")
    idx = ix[0] if len(ix) else -1
    try:
        return float(df.iloc[idx]["close"])
    except Exception:
        return float(df["close"].iloc[-1])
