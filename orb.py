import os, requests
from datetime import datetime, timezone, timedelta
import pandas as pd
import pandas_ta as ta

OANDA_API_KEY = os.getenv("OANDA_API_KEY","")
OANDA_ENV     = os.getenv("OANDA_ENV","practice").lower()  # "practice" or "live"
OANDA_HOST    = "https://api-fxpractice.oanda.com" if OANDA_ENV!="live" else "https://api-fxtrade.oanda.com"

# ORB params (env overrideable)
ORB_ENABLED        = int(os.getenv("ORB_ENABLED","1"))
ORB_WINDOW_MIN     = int(os.getenv("ORB_WINDOW_MIN","5"))   # first 5min of the hour
ORB_RETEST_BARS    = int(os.getenv("ORB_RETEST_BARS","3"))
ORB_MIN_BREAK_PCT  = float(os.getenv("ORB_MIN_BREAK_PCT","0.02"))  # 0.02% beyond range
ORB_REQUIRE_TREND  = int(os.getenv("ORB_REQUIRE_TREND","1"))
ORB_REQUIRE_RSI    = int(os.getenv("ORB_REQUIRE_RSI","0"))
RSI_OB             = int(os.getenv("RSI_SELL","70"))
RSI_OS             = int(os.getenv("RSI_BUY","30"))

def _oanda_get_m1(symbol:str, count:int=180) -> pd.DataFrame:
    """Fetch recent M1 candles (last 'count') for instrument like 'EUR_USD'."""
    if not OANDA_API_KEY:
        raise RuntimeError("OANDA_API_KEY not set")
    url = f"{OANDA_HOST}/v3/instruments/{symbol}/candles"
    params = dict(granularity="M1", count=count, price="M")
    r = requests.get(url, params=params, headers={"Authorization": f"Bearer {OANDA_API_KEY}"}, timeout=20)
    r.raise_for_status()
    js = r.json()
    rows=[]
    for c in js.get("candles",[]):
        if not c.get("complete",False): 
            continue
        t = datetime.fromisoformat(c["time"].replace("Z","+00:00"))
        m = c["mid"]
        rows.append((t, float(m["o"]), float(m["h"]), float(m["l"]), float(m["c"])))
    if not rows:
        return pd.DataFrame()
    df=pd.DataFrame(rows, columns=["time","open","high","low","close"])
    df=df.sort_values("time").reset_index(drop=True)
    return df

def _opening_range_of_hour(df: pd.DataFrame, now_utc: datetime, window_min: int):
    hour_start = now_utc.replace(minute=0, second=0, microsecond=0)
    orb_end = hour_start + timedelta(minutes=window_min)
    dfh = df[(df["time"] >= hour_start) & (df["time"] < hour_start + timedelta(hours=1))].copy()
    if dfh.empty: 
        return None
    orbf = dfh[(dfh["time"] >= hour_start) & (dfh["time"] < orb_end)].copy()
    if len(orbf) < window_min: 
        return None
    lo = orbf["low"].min()
    hi = orbf["high"].max()
    after = dfh[dfh["time"] >= orb_end].copy()
    return (hi, lo, hour_start, orb_end, dfh, after)

def _trend_filters(dfh: pd.DataFrame):
    dfh = dfh.copy()
    dfh["ema50"] = ta.ema(dfh["close"], length=50)
    dfh["ema200"]= ta.ema(dfh["close"], length=200)
    dfh["rsi14"] = ta.rsi(dfh["close"], length=14)
    dfh = dfh.dropna()
    if dfh.empty: 
        return None
    last = dfh.iloc[-1]
    uptrend = last["ema50"] > last["ema200"]
    rsi = float(last["rsi14"])
    return uptrend, rsi

def orb_signal(symbol:str, pretty:str, expiry_min:int=5):
    """
    Returns (signal_dict or None, debug_reason).
    signal_dict: {symbol_yf, symbol_pretty, signal, price, score, why}
    """
    if not ORB_ENABLED:
        return None, "ORB disabled"
    now = datetime.now(timezone.utc)
    df = _oanda_get_m1(symbol, count=180)
    if df.empty: 
        return None, "no data"
    rng = _opening_range_of_hour(df, now, ORB_WINDOW_MIN)
    if not rng: 
        return None, "ORB window not ready"
    hi, lo, hour_start, orb_end, dfh, after = rng
    if after.empty:
        return None, "no post-ORB bars"

    recent = after.tail(ORB_RETEST_BARS+5).reset_index(drop=True)
    if recent.empty:
        return None, "no recent bars"

    score = 0
    why = []
    tf = _trend_filters(dfh)
    uptrend, rsi = (None, None)
    if tf:
        uptrend, rsi = tf
        why.append("EMA50>EMA200" if uptrend else "EMA50<EMA200"); score+=1

    last = recent.iloc[-1]
    px = float(last["close"])
    hi_break = hi * (1 + ORB_MIN_BREAK_PCT/100.0)
    lo_break = lo * (1 - ORB_MIN_BREAK_PCT/100.0)

    side=None
    # BUY: break above & retest near HI
    broke_up = any(recent["close"] > hi_break)
    retest_up = any(abs(recent.iloc[-1-i]["low"] - hi) / hi <= (ORB_MIN_BREAK_PCT/100.0) for i in range(min(ORB_RETEST_BARS, len(recent)-1)))
    if broke_up and retest_up:
        if (not ORB_REQUIRE_TREND) or (uptrend is True):
            if (not ORB_REQUIRE_RSI) or (rsi is not None and rsi <= RSI_OB):
                side="BUY"; why.append("ORB↑ + retest"); score+=1

    # SELL: break below & retest near LO
    if side is None:
        broke_dn = any(recent["close"] < lo_break)
        retest_dn = any(abs(recent.iloc[-1-i]["high"] - lo) / lo <= (ORB_MIN_BREAK_PCT/100.0) for i in range(min(ORB_RETEST_BARS, len(recent)-1)))
        if broke_dn and retest_dn:
            if (not ORB_REQUIRE_TREND) or (uptrend is False):
                if (not ORB_REQUIRE_RSI) or (rsi is not None and rsi >= RSI_OS):
                    side="SELL"; why.append("ORB↓ + retest"); score+=1

    if side:
        return ({
            "symbol_yf": symbol,
            "symbol_pretty": pretty,
            "signal": side,
            "price": f"{px:.5f}",
            "score": score,
            "why": ", ".join(why) if why else "ORB"
        }, f"OK {side}")
    return None, "no ORB setup"
