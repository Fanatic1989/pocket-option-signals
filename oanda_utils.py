import os, requests
from datetime import datetime, timezone
import pandas as pd

OANDA_API_KEY = os.getenv("OANDA_API_KEY","")
OANDA_ENV     = os.getenv("OANDA_ENV","practice").lower()  # "practice" or "live"
OANDA_HOST    = "https://api-fxpractice.oanda.com" if OANDA_ENV != "live" else "https://api-fxtrade.oanda.com"
HEADERS       = {"Authorization": f"Bearer {OANDA_API_KEY}"}

def oanda_fetch_m1(instrument: str, count: int = 220) -> pd.DataFrame:
    """Return a DataFrame of M1 candles (UTC index, columns: open/high/low/close) for an OANDA instrument like 'EUR_USD'."""
    if not OANDA_API_KEY:
        raise RuntimeError("OANDA_API_KEY not set")
    url = f"{OANDA_HOST}/v3/instruments/{instrument}/candles"
    r = requests.get(url, params={"granularity":"M1","count":count,"price":"M"}, headers=HEADERS, timeout=20)
    r.raise_for_status()
    candles = r.json().get("candles", [])
    if not candles:
        return pd.DataFrame(columns=["open","high","low","close"])
    rows = []
    for c in candles:
        t = datetime.fromisoformat(c["time"].replace("Z","+00:00")).astimezone(timezone.utc)
        m = c.get("mid") or {}
        rows.append({"time": t,
                     "open": float(m.get("o", "nan")),
                     "high": float(m.get("h", "nan")),
                     "low":  float(m.get("l", "nan")),
                     "close":float(m.get("c", "nan"))})
    df = pd.DataFrame(rows).dropna()
    if not df.empty:
        df = df.set_index("time").sort_index()
    return df

def oanda_last_close(instrument: str) -> float | None:
    df = oanda_fetch_m1(instrument, count=2)
    if df is None or df.empty:
        return None
    return float(df["close"].iloc[-1])

def oanda_price_at(symbol: str, when) -> float:
    """
    Fetch the approximate price of `symbol` at datetime `when` (UTC).
    Falls back to nearest available candle close.
    """
    import requests
    from datetime import timedelta

    if not OANDA_API_KEY:
        raise RuntimeError("OANDA_API_KEY not set")
    # Convert datetime to RFC3339
    end = when.isoformat().replace("+00:00", "Z")
    start = (when - timedelta(minutes=5)).isoformat().replace("+00:00", "Z")

    url = f"{OANDA_HOST}/v3/instruments/{symbol}/candles"
    params = dict(granularity="M1", from_=start, to=end, price="M")
    r = requests.get(url, params=params,
                     headers={"Authorization": f"Bearer {OANDA_API_KEY}"},
                     timeout=20)
    r.raise_for_status()
    data = r.json()
    candles = data.get("candles", [])
    if not candles:
        raise RuntimeError(f"No candles found for {symbol} at {when}")
    return float(candles[-1]["mid"]["c"])
