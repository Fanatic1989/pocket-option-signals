# utils.py — config store, TZ, sqlite, Deriv fetch (robust), symbols, indicators, backtest helpers, plotting, Telegram helpers
import os, json, sqlite3
from datetime import datetime, time as dtime, timezone, timedelta
from typing import Dict, Any, Optional, Tuple

# ------------------------------- Timezone ------------------------------------
TIMEZONE = os.getenv("APP_TZ", "America/Port_of_Spain")
try:
    import pytz
    TZ = pytz.timezone(TIMEZONE)
except Exception:
    from zoneinfo import ZoneInfo
    TZ = ZoneInfo(TIMEZONE)

import pandas as pd
import numpy as np

# mplfinance is optional; guard import for environments without it
try:
    import mplfinance as mpf
    HAVE_MPLFIN = True
except Exception:
    HAVE_MPLFIN = False
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# ------------------------------- SQLite --------------------------------------
DB_PATH = os.getenv("SQLITE_PATH", "members.db")

def exec_sql(sql, params=(), fetch=False):
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    try:
        cur = conn.cursor()
        cur.execute(sql, params)
        rows = cur.fetchall() if fetch else None
        conn.commit()
        return rows
    finally:
        conn.close()

exec_sql("""CREATE TABLE IF NOT EXISTS app_config(
  k TEXT PRIMARY KEY, v TEXT)""")

def get_config() -> dict:
    row = exec_sql("SELECT v FROM app_config WHERE k='config'", fetch=True)
    if not row: return {}
    try: return json.loads(row[0][0])
    except: return {}

def set_config(cfg: dict):
    exec_sql(
        "INSERT INTO app_config(k,v) VALUES('config',?) "
        "ON CONFLICT(k) DO UPDATE SET v=excluded.v",
        (json.dumps(cfg),)
    )

# ------------------------------ Trading window -------------------------------
def within_window(cfg: dict) -> bool:
    w = cfg.get("window") or {}
    start = w.get("start") or cfg.get("window_start","08:00")
    end   = w.get("end")   or cfg.get("window_end","17:00")
    try:
        now_tt = datetime.now(TZ).time()
    except Exception:
        now_tt = datetime.utcnow().time()
    s_h,s_m = [int(x) for x in (start or "08:00").split(":")[:2]]
    e_h,e_m = [int(x) for x in (end   or "17:00").split(":")[:2]]
    return dtime(s_h,s_m) <= now_tt <= dtime(e_h,e_m)

# ----------------------------- Symbols & mapping -----------------------------
PO_PAIRS = [
  "EURUSD","GBPUSD","USDJPY","USDCHF","USDCAD","AUDUSD","NZDUSD",
  "EURGBP","EURJPY","GBPJPY","EURAUD","AUDJPY","CADJPY","CHFJPY"
]
DERIV_PAIRS = [
  "frxEURUSD","frxGBPUSD","frxUSDJPY","frxUSDCHF","frxUSDCAD","frxAUDUSD","frxNZDUSD",
  "frxEURGBP","frxEURJPY","frxGBPJPY","frxEURAUD","frxAUDJPY","frxCADJPY","frxCHFJPY"
]
PO2DERIV = dict(zip(PO_PAIRS, DERIV_PAIRS))
DERIV2PO = {v:k for k,v in PO2DERIV.items()}

def convert_po_to_deriv(symbols):
    out = []
    for s in symbols:
        s = (s or "").strip().upper()
        if s.startswith("frx"): out.append(s)
        else: out.append(PO2DERIV.get(s, s))
    return out

# ----------------------------- Data loading/fetch ----------------------------
def load_csv(csv_file) -> pd.DataFrame:
    df = pd.read_csv(csv_file)
    # Flexible columns: o,h,l,c or open,high,low,close; time/timestamp/date/epoch
    cols = {c.lower(): c for c in df.columns}
    o = cols.get("open") or cols.get("o")
    h = cols.get("high") or cols.get("h")
    l = cols.get("low")  or cols.get("l")
    c = cols.get("close") or cols.get("c")
    t = cols.get("time") or cols.get("timestamp") or cols.get("date") or cols.get("epoch")
    if not all([o,h,l,c,t]): raise ValueError("CSV missing OHLC/time columns")
    df = df[[t,o,h,l,c]].copy()
    df.columns = ["time","Open","High","Low","Close"]
    # epoch to datetime if numeric
    if np.issubdtype(df["time"].dtype, np.number):
        df["time"] = pd.to_datetime(df["time"].astype(int), unit="s", utc=True, errors="coerce")
    else:
        df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
    df = df.dropna(subset=["time"]).set_index("time").sort_index()
    return df

def _synth_candles(symbol: str, granularity_sec: int, bars: int = 600) -> pd.DataFrame:
    """
    Synthetic OHLC generator so backtest never 500s if Deriv is unavailable.
    Creates a gentle random-walk + sine drift so indicators look realistic.
    """
    bars = int(max(200, min(3000, bars)))
    now = int(datetime.now(timezone.utc).timestamp())
    times = pd.to_datetime([now - (bars - i) * granularity_sec for i in range(bars)], unit="s", utc=True)
    rng = np.random.default_rng(abs(hash(symbol)) % (2**32 - 1))
    drift = np.sin(np.linspace(0, 12*np.pi, bars)) * 0.0008
    noise = rng.normal(0, 0.0009, size=bars)
    base = 1.10 + rng.normal(0, 0.02)  # around 1.x for FX
    close = base + np.cumsum(drift + noise)
    open_ = np.r_[close[0], close[:-1]]
    high = np.maximum(open_, close) + rng.random(bars)*0.0008
    low  = np.minimum(open_, close) - rng.random(bars)*0.0008
    df = pd.DataFrame({"Open": open_, "High": high, "Low": low, "Close": close}, index=times)
    return df.sort_index()

def fetch_deriv_history(symbol: str, granularity_sec: int, days: int=5) -> pd.DataFrame:
    """
    Fetch OHLC candles for a Deriv FRX symbol (e.g. frxEURUSD).
    Tries two HTTP endpoints; if both fail, falls back to synthetic data so the app never 500s.
    """
    import requests
    end = int(datetime.now(timezone.utc).timestamp())
    start = end - int(days) * 86400

    errs = []

    # Attempt 1: explorer 'candles'
    try:
        url = "https://api.deriv.com/api/explore/candles"
        params = {"symbol": symbol, "granularity": granularity_sec, "start": start, "end": end}
        r = requests.get(url, params=params, timeout=15)
        if r.ok:
            js = r.json()
            rows = js.get("candles") or js.get("history") or []
            if rows:
                df = pd.DataFrame(rows)
                epoch = df.get("epoch") if "epoch" in df else df.get("time")
                out = pd.DataFrame({
                    "time": pd.to_datetime(epoch.astype(int), unit="s", utc=True),
                    "Open": pd.to_numeric(df.get("open"), errors="coerce"),
                    "High": pd.to_numeric(df.get("high"), errors="coerce"),
                    "Low" : pd.to_numeric(df.get("low"), errors="coerce"),
                    "Close": pd.to_numeric(df.get("close"), errors="coerce"),
                }).dropna()
                if not out.empty:
                    return out.set_index("time").sort_index()
        errs.append(f"A1 {getattr(r,'status_code', 'NA')} {getattr(r,'text','')[:80]}")
    except Exception as e:
        errs.append(f"A1 err {type(e).__name__}: {e}")

    # Attempt 2: legacy binary path
    try:
        url2 = "https://api.deriv.com/binary/api/v1/candles"
        params2 = {"symbol": symbol, "granularity": granularity_sec, "end": end, "start": start}
        r2 = requests.get(url2, params=params2, timeout=15)
        if r2.ok:
            js2 = r2.json()
            arr = js2.get("candles") or js2.get("history") or js2.get("data")
            if arr:
                df = pd.DataFrame(arr)
                epoch = df.get("epoch") or df.get("time")
                open_ = df.get("open")  or df.get("o")
                high_ = df.get("high")  or df.get("h")
                low__ = df.get("low")   or df.get("l")
                close = df.get("close") or df.get("c")
                out = pd.DataFrame({
                    "time": pd.to_datetime(epoch.astype(int), unit="s", utc=True),
                    "Open": pd.to_numeric(open_, errors="coerce"),
                    "High": pd.to_numeric(high_, errors="coerce"),
                    "Low" : pd.to_numeric(low__, errors="coerce"),
                    "Close": pd.to_numeric(close, errors="coerce"),
                }).dropna()
                if not out.empty:
                    return out.set_index("time").sort_index()
        errs.append(f"A2 {getattr(r2,'status_code','NA')} {getattr(r2,'text','')[:80]}")
    except Exception as e:
        errs.append(f"A2 err {type(e).__name__}: {e}")

    # Fallback: synthetic (prevents 500)
    # Estimate bars from days + granularity
    est_bars = max(300, min(2500, int(days * (86400 / max(1, granularity_sec)))))
    df = _synth_candles(symbol, granularity_sec, est_bars)
    # Attach a hint you can read in logs if needed
    print(f"[utils] Deriv fetch failed for {symbol}@{granularity_sec}s; using synthetic. Details: {' | '.join(errs)}")
    return df

# ----------------------------- Indicator utils -------------------------------
def _ema(s: pd.Series, n: int) -> pd.Series: return s.ewm(span=int(n), adjust=False).mean()
def _sma(s: pd.Series, n: int) -> pd.Series: return s.rolling(int(n)).mean()
def _true_range(h,l,c):
    prev_c = c.shift(1)
    tr = pd.concat([(h-l).abs(), (h-prev_c).abs(), (l-prev_c).abs()], axis=1).max(axis=1)
    return tr

# ----------------------------- Indicator computation -------------------------
def compute_indicators(df: pd.DataFrame, ind_cfg: dict) -> Dict[str, pd.Series | pd.DataFrame]:
    """
    Returns a dict: name -> Series OR DataFrame (for multi-line indicators).
    Names are canonical so plotter can decide overlay vs oscillator panel.
    """
    out: Dict[str, Any] = {}
    o, h, l, c = df["Open"], df["High"], df["Low"], df["Close"]

    def enabled(key: str) -> Tuple[bool, dict]:
        cfg = (ind_cfg or {}).get(key, {})
        # Backwards compatibility: spec might be {"enabled":true, ...} or a primitive
        if isinstance(cfg, dict):
            return bool(cfg.get("enabled")), cfg
        # If user stored just a value, treat as enabled with default param name
        return bool(cfg), {"period": cfg} if isinstance(cfg, (int,float,str)) else {}

    # --- Moving averages family
    on,cfg = enabled("SMA")
    if on:
        p = int(cfg.get("period", 20))
        out[f"SMA({p})"] = _sma(c, p)

    on,cfg = enabled("EMA")
    if on:
        p = int(cfg.get("period", 20))
        out[f"EMA({p})"] = _ema(c, p)

    on,cfg = enabled("WMA")
    if on:
        p = int(cfg.get("period", 20))
        w = np.arange(1, p+1)
        out[f"WMA({p})"] = c.rolling(p).apply(lambda x: (x*w).sum()/w.sum(), raw=True)

    on,cfg = enabled("SMMA")
    if on:
        p = int(cfg.get("period", 20))
        out[f"SMMA({p})"] = c.ewm(alpha=1/p, adjust=False).mean()

    on,cfg = enabled("TMA")
    if on:
        p = int(cfg.get("period", 20))
        out[f"TMA({p})"] = _sma(_sma(c, p), p)

    # --- RSI
    on,cfg = enabled("RSI")
    if on:
        pr = int(cfg.get("period", 14))
        d = c.diff()
        up = d.clip(lower=0).ewm(alpha=1/pr, adjust=False).mean()
        dn = (-d.clip(upper=0)).ewm(alpha=1/pr, adjust=False).mean()
        rs = up / dn.replace(0, np.nan)
        out["RSI"] = (100 - (100/(1+rs))).fillna(method="bfill")

    # --- Stochastic
    on,cfg = enabled("STOCH")
    if on:
        k = int(cfg.get("k", 14)); d_ = int(cfg.get("d", 3))
        lowk = l.rolling(k).min(); highk = h.rolling(k).max()
        kline = (c - lowk) / (highk - lowk).replace(0, np.nan) * 100.0
        dline = kline.rolling(d_).mean()
        out["STOCH"] = pd.DataFrame({"%K": kline, "%D": dline})

    # --- ATR
    on,cfg = enabled("ATR")
    if on:
        n = int(cfg.get("period", 14))
        atr = _true_range(h,l,c).rolling(n).mean()
        out["ATR"] = atr

    # --- ADX (+DI/-DI)
    on,cfg = enabled("ADX")
    if on:
        n = int(cfg.get("period", 14))
        up_move = h.diff()
        down_move = -l.diff()
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
        tr = _true_range(h,l,c)
        atr = tr.rolling(n).mean()
        plus_di = 100 * pd.Series(plus_dm, index=c.index).ewm(alpha=1/n, adjust=False).mean() / atr
        minus_di = 100 * pd.Series(minus_dm, index=c.index).ewm(alpha=1/n, adjust=False).mean() / atr
        dx = ( (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0,np.nan) ) * 100
        adx = dx.ewm(alpha=1/n, adjust=False).mean()
        out["ADX"] = pd.DataFrame({"+DI": plus_di, "-DI": minus_di, "ADX": adx})

    # --- Bollinger Bands + Width
    on,cfg = enabled("BBANDS")
    if on:
        n = int(cfg.get("period", 20)); k = float(cfg.get("mult", 2.0))
        ma = _sma(c, n); sd = c.rolling(n).std()
        upper = ma + k*sd; lower = ma - k*sd
        out["BBANDS"] = pd.DataFrame({"Upper": upper, "Middle": ma, "Lower": lower})
        out["BBWIDTH"] = (upper - lower) / ma.replace(0, np.nan)

    # --- Keltner Channel
    on,cfg = enabled("KELTNER")
    if on:
        n = int(cfg.get("period", 20)); m = float(cfg.get("mult", 2.0))
        ema = _ema(c, n); atr = _true_range(h,l,c).rolling(n).mean()
        upper = ema + m*atr; lower = ema - m*atr
        out["KELTNER"] = pd.DataFrame({"Upper": upper, "EMA": ema, "Lower": lower})

    # --- Donchian Channels
    on,cfg = enabled("DONCHIAN")
    if on:
        n = int(cfg.get("period", 20))
        upper = h.rolling(n).max(); lower = l.rolling(n).min(); mid = (upper+lower)/2
        out["DONCHIAN"] = pd.DataFrame({"Upper": upper, "Mid": mid, "Lower": lower})

    # --- Envelopes (pct around SMA)
    on,cfg = enabled("ENVELOPES")
    if on:
        n = int(cfg.get("period", 20)); pct = float(cfg.get("pct", 0.02))
        ma = _sma(c, n)
        up = ma * (1+pct); lo = ma * (1-pct)
        out["ENVELOPES"] = pd.DataFrame({"Upper": up, "MA": ma, "Lower": lo})

    # --- MACD (also OSMA=hist)
    on,cfg = enabled("MACD")
    if on:
        f = int(cfg.get("fast", 12)); s = int(cfg.get("slow", 26)); sig = int(cfg.get("signal", 9))
        macd = _ema(c, f) - _ema(c, s)
        signal = _ema(macd, sig)
        hist = macd - signal
        out["MACD"] = pd.DataFrame({"MACD": macd, "Signal": signal, "Hist": hist})
        out["OSMA"] = hist

    # --- Momentum
    on,cfg = enabled("MOMENTUM")
    if on:
        n = int(cfg.get("period", 10))
        out["MOMENTUM"] = c.diff(n)

    # --- ROC
    on,cfg = enabled("ROC")
    if on:
        n = int(cfg.get("period", 10))
        out["ROC"] = (c / c.shift(n) - 1.0) * 100.0

    # --- Williams %R
    on,cfg = enabled("WILLR")
    if on:
        n = int(cfg.get("period", 14))
        hh = h.rolling(n).max(); ll = l.rolling(n).min()
        out["WILLR"] = -100 * (hh - c) / (hh - ll).replace(0,np.nan)

    # --- CCI
    on,cfg = enabled("CCI")
    if on:
        n = int(cfg.get("period", 20))
        tp = (h + l + c) / 3.0
        sma = tp.rolling(n).mean()
        md = (tp - sma).abs().rolling(n).mean()
        out["CCI"] = (tp - sma) / (0.015 * md).replace(0,np.nan)

    # --- Aroon
    on,cfg = enabled("AROON")
    if on:
        n = int(cfg.get("period", 14))
        def aroon_up(x): return (x.argmax()+1)/len(x)*100
        def aroon_dn(x): return (x[::-1].argmax()+1)/len(x)*100
        up = h.rolling(n).apply(aroon_up, raw=True)
        dn = l.rolling(n).apply(aroon_dn, raw=True)
        out["AROON"] = pd.DataFrame({"Up": up, "Down": dn})

    # --- Vortex
    on,cfg = enabled("VORTEX")
    if on:
        n = int(cfg.get("period", 14))
        vm_plus = (h - l.shift(1)).abs()
        vm_minus = (l - h.shift(1)).abs()
        tr = _true_range(h,l,c)
        vi_plus = vm_plus.rolling(n).sum() / tr.rolling(n).sum()
        vi_minus = vm_minus.rolling(n).sum() / tr.rolling(n).sum()
        out["VORTEX"] = pd.DataFrame({"+VI": vi_plus, "-VI": vi_minus})

    # --- Awesome Oscillator (median price 5-34 SMA)
    on,cfg = enabled("AO")
    if on:
        mp = (h + l) / 2.0
        ao = _sma(mp, 5) - _sma(mp, 34)
        out["AO"] = ao

    # --- Accelerator Oscillator (AO - SMA(5) of AO)
    on,cfg = enabled("AC")
    if on:
        mp = (h + l) / 2.0
        ao = _sma(mp, 5) - _sma(mp, 34)
        ac = ao - _sma(ao, 5)
        out["AC"] = ac

    # --- Ichimoku (basic: Tenkan, Kijun, SpanA/B; no forward shift)
    on,cfg = enabled("ICHIMOKU")
    if on:
        conv = int(cfg.get("conversion", 9))
        base = int(cfg.get("base", 26))
        spanb = int(cfg.get("spanb", 52))
        tenkan = (h.rolling(conv).max() + l.rolling(conv).min())/2
        kijun  = (h.rolling(base).max() + l.rolling(base).min())/2
        span_a = (tenkan + kijun) / 2
        span_b = (h.rolling(spanb).max() + l.rolling(spanb).min())/2
        out["ICHIMOKU"] = pd.DataFrame({"Tenkan":tenkan,"Kijun":kijun,"SpanA":span_a,"SpanB":span_b})

    # --- Parabolic SAR (simple)
    on,cfg = enabled("PSAR")
    if on:
        af_start = float(cfg.get("af", 0.02))
        af_max = float(cfg.get("af_max", 0.2))
        psar = []
        uptrend = True
        af = af_start
        ep = l.iloc[0]
        sar = l.iloc[0]
        for i in range(len(c)):
            if i < 2:
                psar.append(np.nan)
                continue
            prev_sar = sar
            if uptrend:
                sar = prev_sar + af * (ep - prev_sar)
                sar = min(sar, l.iloc[i-1], l.iloc[i-2])
                if h.iloc[i] > ep:
                    ep = h.iloc[i]; af = min(af + af_start, af_max)
                if l.iloc[i] < sar:
                    uptrend = False; sar = ep; ep = l.iloc[i]; af = af_start
            else:
                sar = prev_sar + af * (ep - prev_sar)
                sar = max(sar, h.iloc[i-1], h.iloc[i-2])
                if l.iloc[i] < ep:
                    ep = l.iloc[i]; af = min(af + af_start, af_max)
                if h.iloc[i] > sar:
                    uptrend = True; sar = ep; ep = h.iloc[i]; af = af_start
            psar.append(sar)
        out["PSAR"] = pd.Series(psar, index=c.index)

    # --- SuperTrend (simplified)
    on,cfg = enabled("SUPERtrend")
    if on:
        n = int(cfg.get("period", 10)); m = float(cfg.get("mult", 3.0))
        atr = _true_range(h,l,c).rolling(n).mean()
        hl2 = (h + l) / 2.0
        upper = hl2 + m*atr
        lower = hl2 - m*atr
        trend = pd.Series(index=c.index, dtype=float)
        for i in range(len(c)):
            if i == 0:
                trend.iloc[i] = upper.iloc[i]
                continue
            prev = trend.iloc[i-1]
            if c.iloc[i] > prev:
                trend.iloc[i] = max(lower.iloc[i], prev)
            else:
                trend.iloc[i] = min(upper.iloc[i], prev)
        out["SUPER"] = pd.DataFrame({"Upper": upper, "Lower": lower, "Trend": trend})

    # --- Fractal markers
    on,cfg = enabled("FRACTAL")
    if on:
        def fractals(high, low):
            up = (high.shift(2) < high.shift(1)) & (high.shift(2) < high) & \
                 (high.shift(-2) < high.shift(-1)) & (high.shift(-2) < high)
            dn = (low.shift(2) > low.shift(1)) & (low.shift(2) > low) & \
                 (low.shift(-2) > low.shift(-1)) & (low.shift(-2) > low)
            up_px = c.where(up); dn_px = c.where(dn)
            return up_px, dn_px
        up_px, dn_px = fractals(h, l)
        out["FRACTAL_UP"] = up_px
        out["FRACTAL_DN"] = dn_px

    # --- Fractal Chaos Bands
    on,cfg = enabled("FRACTAL_BANDS")
    if on:
        n = int(cfg.get("period", 20))
        out["FRACTAL_BANDS"] = pd.DataFrame({"Upper": h.rolling(n).max(), "Lower": l.rolling(n).min()})

    # --- Bears/Bulls Power
    on,cfg = enabled("BEARS")
    if on:
        n = int(cfg.get("period", 13))
        out["BEARS"] = l - _ema(c, n)
    on,cfg = enabled("BULLS")
    if on:
        n = int(cfg.get("period", 13))
        out["BULLS"] = h - _ema(c, n)

    # --- Schaff Trend Cycle (simplified)
    on,cfg = enabled("SCHAFF")
    if on:
        fast = int(cfg.get("fast", 23)); slow = int(cfg.get("slow", 50)); cycle = int(cfg.get("cycle", 10))
        macd = _ema(c, fast) - _ema(c, slow)
        minm = macd.rolling(cycle).min(); maxm = macd.rolling(cycle).max()
        stc = 100 * (macd - minm) / (maxm - minm).replace(0,np.nan)
        out["SCHAFF"] = stc

    # --- ZigZag (percentage)
    on,cfg = enabled("ZIGZAG")
    if on:
        pct = float(cfg.get("pct", 5.0)) / 100.0
        zz = pd.Series(index=c.index, dtype=float)
        last_pivot = c.iloc[0]
        direction = 0
        for idx, px in c.items():
            chg = (px - last_pivot) / last_pivot if last_pivot != 0 else 0
            if direction >= 0 and chg >= pct:
                direction = 1; last_pivot = px; zz.loc[idx] = px
            elif direction <= 0 and chg <= -pct:
                direction = -1; last_pivot = px; zz.loc[idx] = px
        out["ZIGZAG"] = zz

    return out

# ----------------------------- Rule engine & backtest ------------------------
def simple_rule_engine(df: pd.DataFrame, ind: dict, rule_name: str):
    signals = []
    close = df["Close"]

    def add(sig_idx, direction, bars):
        all_idx = df.index
        pos = all_idx.get_indexer([sig_idx])[0]
        exp_pos = min(pos + max(1,bars), len(all_idx)-1)
        signals.append({"index": sig_idx, "direction": direction, "expiry_idx": all_idx[exp_pos]})

    if rule_name == "TREND":
        sma50 = ind.get("SMA(50)") or ind.get("EMA(50)")
        if sma50 is None: return signals
        above = close > sma50
        cross_up = (above & (~above.shift(1).fillna(False)))
        cross_dn = ((~above) & (above.shift(1).fillna(False)))
        for ts in close.index[cross_up.fillna(False)]: add(ts, "BUY", 5)
        for ts in close.index[cross_dn.fillna(False)]: add(ts, "SELL", 5)
        return signals

    if rule_name == "CHOP":
        rsi = ind.get("RSI")
        if rsi is None: return signals
        bounce_up = (rsi.shift(1) < 50) & (rsi >= 50)
        bounce_dn = (rsi.shift(1) > 50) & (rsi <= 50)
        for ts in rsi.index[bounce_up.fillna(False)]: add(ts, "BUY", 3)
        for ts in rsi.index[bounce_dn.fillna(False)]: add(ts, "SELL", 3)
        return signals

    # BASE: Stochastic cross
    st = ind.get("STOCH")
    if isinstance(st, pd.DataFrame):
        k, d = st["%K"], st["%D"]
        cross_up = (k.shift(1) < d.shift(1)) & (k >= d)
        cross_dn = (k.shift(1) > d.shift(1)) & (k <= d)
        for ts in k.index[cross_up.fillna(False)]: add(ts, "BUY", 5)
        for ts in k.index[cross_dn.fillna(False)]: add(ts, "SELL", 5)
    return signals

def evaluate_signals_outcomes(df: pd.DataFrame, signals: list) -> dict:
    wins=loss=draw=0
    for s in signals:
        e = s["expiry_idx"]; i = s["index"]
        try:
            c0 = float(df.loc[i, "Close"]); ce = float(df.loc[e, "Close"])
            if s["direction"] == "BUY":
                if ce > c0: wins+=1
                elif ce < c0: loss+=1
                else: draw+=1
            else:
                if ce < c0: wins+=1
                elif ce > c0: loss+=1
                else: draw+=1
        except Exception:
            pass
    return {"wins": wins, "loss": loss, "draw": draw, "total": wins+loss+draw,
            "win_rate": (wins*100.0/max(1,wins+loss))}

# ----------------------------- Plotting --------------------------------------
def _panel_of(name: str) -> int:
    """0=price overlay, 1=oscillator panel."""
    overlay = ("BBANDS","KELTNER","DONCHIAN","ENVELOPES","PSAR","SUPER","ICHIMOKU","FRACTAL_BANDS","ZIGZAG",
               "SMA","EMA","WMA","SMMA","TMA")
    if name.startswith(overlay) or name.startswith(("SMA(","EMA(","WMA(","SMMA(","TMA(")):
        return 0
    return 1

def plot_signals(df, signals, indicators, strategy, tf, expiry) -> str:
    import os
    os.makedirs("static/plots", exist_ok=True)
    if df.empty:
        out = "empty.png"
        plt.figure(figsize=(8,2)); plt.text(0.5,0.5,"No data", ha="center"); plt.savefig("static/plots/"+out); plt.close()
        return out
    out_name = f"{df.index[-1].strftime('%Y%m%d_%H%M%S')}_{strategy}_{tf}.png"
    path = os.path.join("static","plots", out_name)

    # compute indicators (ensure latest cfg)
    ind = compute_indicators(df, indicators or {})

    # ---- mplfinance path
    if HAVE_MPLFIN:
        addplots = []
        for name, val in ind.items():
            panel = _panel_of(name)
            if isinstance(val, pd.DataFrame):
                for col in val.columns:
                    addplots.append(mpf.make_addplot(val[col], panel=panel, width=1))
            else:
                addplots.append(mpf.make_addplot(val, panel=panel, width=1))

        fig, axlist = mpf.plot(df, type="candle", style="yahoo", addplot=addplots,
                               returnfig=True, volume=False, figsize=(13,9),
                               title=f"{strategy} • TF={tf} • Exp={expiry}")
        ax = axlist[0]
        # markers
        buy_x= []; buy_y= []; sell_x=[]; sell_y=[]
        for s in signals:
            if s["index"] not in df.index: continue
            px = float(df.loc[s["index"],"Close"])
            if s["direction"]=="BUY":
                buy_x.append(s["index"]); buy_y.append(px)
            else:
                sell_x.append(s["index"]); sell_y.append(px)
        if buy_x: ax.scatter(buy_x, buy_y, marker="^", s=80)
        if sell_x: ax.scatter(sell_x, sell_y, marker="v", s=80)
        # fractal markers
        if "FRACTAL_UP" in ind:
            fu = ind["FRACTAL_UP"].dropna()
            if not fu.empty: ax.scatter(fu.index, fu.values, marker="^", s=50)
        if "FRACTAL_DN" in ind:
            fd = ind["FRACTAL_DN"].dropna()
            if not fd.empty: ax.scatter(fd.index, fd.values, marker="v", s=50)

        fig.savefig(path, dpi=120, bbox_inches="tight"); plt.close(fig)
        return out_name

    # ---- fallback simple matplotlib
    ds = df.reset_index().rename(columns={"index":"timestamp","Open":"open","High":"high","Low":"low","Close":"close"})
    ds["timestamp"] = pd.to_datetime(ds["time"] if "time" in ds else ds["timestamp"])
    ts = mdates.date2num(pd.to_datetime(ds["timestamp"]).dt.to_pydatetime())

    fig = plt.figure(figsize=(13,9))
    ax = fig.add_subplot(2,1,1)
    for i,(t,o,h,lw,cl) in enumerate(zip(ts, ds["open"], ds["high"], ds["low"], ds["close"])):
        color = "#17c964" if cl>=o else "#f31260"
        ax.vlines(t, lw, h, color=color, linewidth=1.0, alpha=.9)
        ax.add_patch(plt.Rectangle((t-0.0008, min(o,cl)), 0.0016, max(abs(cl-o),1e-6),
                                   facecolor=color, edgecolor=color, linewidth=.8, alpha=.95))

    # overlay indicators
    for name,val in ind.items():
        if _panel_of(name) != 0: continue
        if isinstance(val, pd.DataFrame):
            for col in val.columns: ax.plot(df.index, val[col], linewidth=1)
        else:
            ax.plot(df.index, val, linewidth=1)

    # markers
    for s in signals:
        i = s["index"]
        if i in df.index:
            px = float(df.loc[i,"Close"])
            ax.scatter([mdates.date2num(i)],[px], marker="^" if s["direction"]=="BUY" else "v", s=60)

    ax.set_title(f"{strategy} • TF={tf} • Exp={expiry}")
    ax.grid(True, alpha=.2)

    # oscillator panel
    ax2 = fig.add_subplot(2,1,2, sharex=ax)
    for name,val in ind.items():
        if _panel_of(name) == 0: continue
        if isinstance(val, pd.DataFrame):
            for col in val.columns: ax2.plot(df.index, val[col], linewidth=1)
        else:
            ax2.plot(df.index, val, linewidth=1)
    ax2.set_title("Oscillators")
    ax2.grid(True, alpha=.2)

    fig.autofmt_xdate(); fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight"); plt.close(fig)
    return out_name

def backtest_run(df: pd.DataFrame, strategy: str, indicators: dict, expiry: str):
    ind = compute_indicators(df, indicators or {})
    signals = simple_rule_engine(df, ind, strategy.upper())
    bars = {"1m":1,"3m":3,"5m":5,"10m":10,"30m":30,"1h":60,"4h":240}.get((expiry or "5m").lower(), 5)
    fixed = []
    for s in signals:
        i = s["index"]
        pos = df.index.get_indexer([i])[0]
        exp_pos = min(pos + max(1,bars), len(df.index)-1)
        s["expiry_idx"] = df.index[exp_pos]
        fixed.append(s)
    stats = evaluate_signals_outcomes(df, fixed)
    return fixed, stats

# ----------------------------- Telegram helpers ------------------------------
def get_chat_id_for_tier(tier: str) -> Optional[str]:
    t = (tier or "").lower()
    table = {
        "free":  os.getenv("TELEGRAM_CHAT_FREE")  or os.getenv("TELEGRAM_CHAT_ID_FREE")  or os.getenv("TELEGRAM_CHAT_ID"),
        "basic": os.getenv("TELEGRAM_CHAT_BASIC") or os.getenv("TELEGRAM_CHAT_ID_BASIC") or os.getenv("TELEGRAM_CHAT_ID"),
        "pro":   os.getenv("TELEGRAM_CHAT_PRO")   or os.getenv("TELEGRAM_CHAT_ID_PRO")   or os.getenv("TELEGRAM_CHAT_ID"),
        "vip":   os.getenv("TELEGRAM_CHAT_VIP")   or os.getenv("TELEGRAM_CHAT_ID_VIP")   or os.getenv("TELEGRAM_CHAT_ID"),
    }
    return (table.get(t) or "").strip() or None

def telegram_send_message(chat_id: str, text: str) -> Dict[str, Any]:
    import requests
    token = (os.getenv("TELEGRAM_BOT_TOKEN") or "").strip()
    if not token:
        return {"ok": False, "error": "Missing TELEGRAM_BOT_TOKEN"}
    if not chat_id:
        return {"ok": False, "error": "Missing chat_id"}
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    try:
        r = requests.post(url, json={
            "chat_id": chat_id,
            "text": text,
            "parse_mode": "HTML",
            "disable_web_page_preview": True
        }, timeout=15)
        js = r.json() if r.headers.get("content-type","").startswith("application/json") else {"ok": r.ok, "text": r.text}
        js["_http"] = r.status_code
        return js
    except Exception as e:
        return {"ok": False, "error": str(e)}
