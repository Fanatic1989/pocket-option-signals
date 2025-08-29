import os, requests, datetime as dt

OANDA_API_KEY = os.getenv("OANDA_API_KEY", "")
OANDA_ENV     = os.getenv("OANDA_ENV", "practice").lower()  # "practice" or "live"
BASE = "https://api-fxpractice.oanda.com" if OANDA_ENV!="live" else "https://api-fxtrade.oanda.com"

def _hdr():
    if not OANDA_API_KEY: raise RuntimeError("OANDA_API_KEY not set")
    return {"Authorization": f"Bearer {OANDA_API_KEY}"}

def oanda_get_candle(instrument: str, granularity="M1"):
    """Latest mid close."""
    url = f"{BASE}/v3/instruments/{instrument}/candles"
    r = requests.get(url, headers=_hdr(), params={"granularity":granularity,"count":1,"price":"M"}, timeout=20)
    r.raise_for_status()
    c = r.json()["candles"][-1]
    return float(c["mid"]["c"])

def oanda_price_at(instrument: str, when_iso: str, granularity="M1"):
    """
    Close at/after a target UTC timestamp. We fetch a small window and
    pick the first candle >= when.
    """
    when = dt.datetime.fromisoformat(when_iso.replace("Z","")).replace(tzinfo=dt.timezone.utc)
    # 10-minute window around eval time
    start = (when - dt.timedelta(minutes=5)).isoformat().replace("+00:00","Z")
    end   = (when + dt.timedelta(minutes=10)).isoformat().replace("+00:00","Z")
    url = f"{BASE}/v3/instruments/{instrument}/candles"
    r = requests.get(url, headers=_hdr(), params={
        "from": start, "to": end, "granularity": granularity, "price":"M", "includeFirst":"true"
    }, timeout=25)
    r.raise_for_status()
    candles = r.json().get("candles", [])
    if not candles:
        raise RuntimeError("No OANDA candles returned")
    # find first >= eval time
    for c in candles:
        t = dt.datetime.fromisoformat(c["time"].replace("Z","+00:00"))
        if t >= when and c.get("complete", True):
            return float(c["mid"]["c"])
    # fallback to last complete
    for c in reversed(candles):
        if c.get("complete", True):
            return float(c["mid"]["c"])
    return float(candles[-1]["mid"]["c"])
