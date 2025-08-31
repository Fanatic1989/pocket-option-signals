import os, requests
from datetime import datetime

OANDA_API_KEY = os.getenv("OANDA_API_KEY", "")
OANDA_ENV     = os.getenv("OANDA_ENV", "practice").lower()
OANDA_HOST    = "https://api-fxpractice.oanda.com" if OANDA_ENV != "live" else "https://api-fxtrade.oanda.com"

def oanda_get_candle(symbol: str):
    """
    Fetch the latest M1 candle for a given instrument (e.g., EUR_USD).
    Returns float price or raises Exception.
    """
    if not OANDA_API_KEY:
        raise RuntimeError("OANDA_API_KEY not set")

    url = f"{OANDA_HOST}/v3/instruments/{symbol}/candles"
    params = {"granularity": "M1", "count": 1, "price": "M"}
    headers = {"Authorization": f"Bearer {OANDA_API_KEY}"}

    r = requests.get(url, headers=headers, params=params, timeout=20)
    r.raise_for_status()
    data = r.json()

    candle = data["candles"][0]
    price = float(candle["mid"]["c"])
    time = candle["time"]
    return {"symbol": symbol, "price": price, "time": time}
