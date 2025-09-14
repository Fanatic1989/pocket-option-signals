# utils.py — config store, TZ, sqlite, FX fetch (Deriv + Stooq + Synthetic), indicators, backtest, plotting
import os, json, sqlite3, math, time
from datetime import datetime, time as dtime, timezone, timedelta

# --------------------------- Timezone ----------------------------------------
TIMEZONE = os.getenv("APP_TZ", "America/Port_of_Spain")
try:
    import pytz
    TZ = pytz.timezone(TIMEZONE)
except Exception:
    from zoneinfo import ZoneInfo
    TZ = ZoneInfo(TIMEZONE)

# --------------------------- Std libs ----------------------------------------
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

DB_PATH = os.getenv("SQLITE_PATH", "members.db")

# ============================== SQLite =======================================
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
    exec_sql("INSERT INTO app_config(k,v) VALUES('config',?) ON CONFLICT(k) DO UPDATE SET v=excluded.v",
             (json.dumps(cfg),))

# ============================== Trading window ===============================
def within_window(cfg: dict) -> bool:
    """
    True only Mon–Fri AND between configured start/end (local TZ).
    """
    w = cfg.get("window") or {}
    start = w.get("start") or cfg.get("window_start","08:00")
    end   = w.get("end")   or cfg.get("window_end","17:00")
    try:
        now_dt = datetime.now(TZ); now_tt = now_dt.time()
    except Exception:
        now_dt = datetime.utcnow(); now_tt = now_dt.time()
    if now_dt.weekday() >= 5:  # 5=Sat, 6=Sun
        return False
    s_h,s_m = [int(x) for x in (start or "08:00").split(":")[:2]]
    e_h,e_m = [int(x) for x in (end   or "17:00").split(":")[:2]]
    return dtime(s_h,s_m) <= now_tt <= dtime(e_h,e_m)

# ============================== Symbols mapping ==============================
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
        s = (s or "").strip()
        if not s: continue
        if s.lower().startswith("frx"):
            out.append("frx" + s[3:].upper())
        else:
            out.append(PO2DERIV.get(s.upper(), s))
    return out

def _sanitize_symbol(sym: str) -> str:
    """Take only the first token (handles 'EURUSD GBPUSD'). Also normalize frx casing."""
    if not sym: return sym
    tok = sym.replace(",", " ").split()
    s = tok[0]
    if s.lower().startswith("frx"):
        return "frx" + s[3:].upper()
    return s.upper()

# ============================== Data loading/fetch ============================
def load_csv(csv_file) -> pd.DataFrame:
    df = pd.read_csv(csv_file)
    cols = {c.lower(): c for c in df.columns}
    o = cols.get("open"); h = cols.get("high"); l = cols.get("low"); c = cols.get("close")
    t = cols.get("time") or cols.get("timestamp") or cols.get("date") or cols.get("epoch")
    if not all([o,h,l,c,t]): raise ValueError("CSV missing OHLC/time columns")
    df = df[[t,o,h,l,c]].copy()
    df.columns = ["time","Open","High","Low","Close"]
    # epoch seconds or ms
    try:
        if str(df["time"].iloc[0]).isdigit():
            v = str(int(df["time"].iloc[0]))
            unit = "ms" if len(v) == 13 else "s"
            df["time"] = pd.to_datetime(df["time"].astype(int), unit=unit, utc=True, errors="coerce")
        else:
            df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
    except Exception:
        df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
    df = df.dropna(subset=["time"]).set_index("time").sort_index()
    return df

# -------- Deriv HTTP (may 404) ----------
def _deriv_price_history(symbol: str, granularity: int, count: int=300, app_id: str|None=None):
    import requests
    attempts = []
    headers = {"User-Agent": "Mozilla/5.0 (RenderFetch/1.0)"}
    qs = {"ticks_history": symbol, "style": "candles", "granularity": granularity, "count": count}
    if app_id: qs["app_id"] = app_id

    for base in ("https://api.deriv.com/api/v4/price_history",
                 "https://api.deriv.com/api/v3/price_history"):
        try:
            r = requests.get(base, params=qs, timeout=15, headers=headers)
            attempts.append((base, r.status_code))
            if r.ok and r.headers.get("content-type","").startswith("application/json"):
                js = r.json()
                candles = js.get("candles") or js.get("history")
                if candles:
                    d = pd.DataFrame(candles)
                    epoch = d.get("epoch") if "epoch" in d else d.get("time")
                    out = pd.DataFrame({
                        "time": pd.to_datetime(epoch.astype(int), unit="s", utc=True),
                        "Open": pd.to_numeric(d.get("open"), errors="coerce"),
                        "High": pd.to_numeric(d.get("high"), errors="coerce"),
                        "Low" : pd.to_numeric(d.get("low"), errors="coerce"),
                        "Close": pd.to_numeric(d.get("close"), errors="coerce"),
                    }).dropna()
                    if not out.empty:
                        return out.set_index("time").sort_index(), attempts
        except Exception:
            attempts.append((base, "EXC"))

    # Explore fallback
    try:
        url = "https://api.deriv.com/api/explore/candles"
        r = requests.get(url, params={"symbol": symbol, "granularity": granularity, "count": count}, timeout=15, headers=headers)
        attempts.append((url, r.status_code))
        if r.ok and "json" in r.headers.get("content-type",""):
            js = r.json()
            rows = js.get("candles") or js.get("history") or []
            d = pd.DataFrame(rows)
            if not d.empty:
                epoch = d.get("epoch") if "epoch" in d else d.get("time")
                out = pd.DataFrame({
                    "time": pd.to_datetime(epoch.astype(int), unit="s", utc=True),
                    "Open": pd.to_numeric(d.get("open"), errors="coerce"),
                    "High": pd.to_numeric(d.get("high"), errors="coerce"),
                    "Low" : pd.to_numeric(d.get("low"), errors="coerce"),
                    "Close": pd.to_numeric(d.get("close"), errors="coerce"),
                }).dropna()
                if not out.empty:
                    return out.set_index("time").sort_index(), attempts
    except Exception:
        attempts.append(("explore/candles", "EXC"))

    return None, attempts

# -------- Stooq fallback (intraday + daily) ----------
def _to_stooq_pair(symbol: str) -> str:
    s = symbol.strip()
    if s.lower().startswith("frx"):
        s = s[3:]
    return s.replace("_","").lower()

def _stooq_intraday(symbol: str, granularity: int, count: int=300):
    import requests
    pair = _to_stooq_pair(symbol)
    mins = max(1, int(granularity/60))
    if mins <= 7: i = 5
    elif mins <= 12: i = 10
    elif mins <= 22: i = 15
    else: i = 60
    url = "https://stooq.com/q/d/l/"
    r = requests.get(url, params={"s": pair, "i": i}, timeout=15, headers={"User-Agent":"Mozilla/5.0"})
    if not r.ok or "Date,Time,Open,High,Low,Close,Volume" not in r.text.splitlines()[0]:
        return None
    from io import StringIO
    df = pd.read_csv(StringIO(r.text))
    df["time"] = pd.to_datetime(df["Date"].astype(str)+" "+df["Time"].astype(str), utc=True, errors="coerce")
    df = df.dropna(subset=["time"]).rename(columns={"Open":"Open","High":"High","Low":"Low","Close":"Close"})
    df = df[["time","Open","High","Low","Close"]].set_index("time").sort_index()
    return df.iloc[-count:] if count and len(df)>count else df

def _stooq_daily(symbol: str, count: int=300):
    import requests
    pair = _to_stooq_pair(symbol)
    url = "https://stooq.com/q/d/l/"
    r = requests.get(url, params={"s": pair}, timeout=15, headers={"User-Agent":"Mozilla/5.0"})
    if not r.ok or "Date,Open,High,Low,Close,Volume" not in r.text.splitlines()[0]:
        return None
    from io import StringIO
    df = pd.read_csv(StringIO(r.text))
    df["time"] = pd.to_datetime(df["Date"].astype(str), utc=True, errors="coerce")
    df = df.dropna(subset=["time"]).rename(columns={"Open":"Open","High":"High","Low":"Low","Close":"Close"})
    df = df[["time","Open","High","Low","Close"]].set_index("time").sort_index()
    return df.iloc[-count:] if count and len(df)>count else df

# -------- Synthetic last-resort ----------
def _synthetic_history(symbol: str, granularity_sec: int, count: int=300):
    """
    Deterministic GBM candles so UI/backtest still works when all remotes fail.
    Seeded by symbol+granularity for stability across runs.
    """
    seed = abs(hash((symbol, granularity_sec))) % (2**32)
    rng = np.random.default_rng(seed)
    # build equally spaced UTC index, ending 'now'
    end = pd.Timestamp.utcnow().floor('S')
    idx = pd.date_range(end=end, periods=count, freq=f"{granularity_sec}S", tz="UTC")
    # GBM mid price
    mu = 0.00002; sigma = 0.0008
    ret = rng.normal(mu, sigma, size=count)
    price = 1.0 * np.exp(np.cumsum(ret))
    # build candles from mid with small ranges
    base = pd.Series(price, index=idx)
    close = base
    open_ = base.shift(1).fillna(base.iloc[0])
    spread = np.maximum(1e-5, (base.rolling(10).std().fillna(0) + 1e-3) * 0.2)
    high = np.maximum(open_, close) + spread.values
    low  = np.minimum(open_, close) - spread.values
    df = pd.DataFrame({"Open":open_.values, "High":high.values, "Low":low.values, "Close":close.values}, index=idx)
    return df

def fetch_deriv_history(symbol: str, granularity_sec: int, count: int=300) -> pd.DataFrame:
    """
    Backward-compatible fetch that tries:
    1) Deriv HTTP (v4/v3/explore)
    2) Stooq intraday CSV
    3) Stooq daily CSV
    4) Synthetic GBM candles (if ALLOW_SYNTHETIC_DATA != '0')
    """
    s = _sanitize_symbol(symbol)
    app_id = os.getenv("DERIV_APP_ID") or os.getenv("DERIV_APPID") or os.getenv("DERIV_APP")

    df, attempts = _deriv_price_history(s, granularity_sec, count=count, app_id=app_id)
    if df is not None and not df.empty:
        return df

    try:
        sti = _stooq_intraday(s, granularity_sec, count=count)
        if sti is not None and not sti.empty:
            return sti
    except Exception:
        pass

    try:
        std = _stooq_daily(s, count=max(120, count//24))
        if std is not None and not std.empty:
            # Expand daily into intraday bars by simple interpolation (keeps trend for plotting/backtest demo)
            std = std.asfreq("D")
            intr_idx = pd.date_range(std.index.min(), std.index.max() + timedelta(days=1),
                                     freq=f"{granularity_sec}S", tz="UTC", inclusive="left")
            out = pd.DataFrame(index=intr_idx)
            for col in ["Open","High","Low","Close"]:
                out[col] = std[col].reindex(std.index).interpolate("time").reindex(intr_idx).ffill()
            return out.iloc[-count:]
    except Exception:
        pass

    if os.getenv("ALLOW_SYNTHETIC_DATA", "1") != "0":
        return _synthetic_history(s, granularity_sec, count=count)

    # If synthetic disabled, raise concise error
    def _short_attempts(atts):
        try:
            return [(a[0].split("//")[-1].split("/")[0], a[1]) for a in atts][-3:]
        except Exception:
            return atts
    short = _short_attempts(attempts or [])
    raise RuntimeError(
        f"Fetch failed for {s} @{granularity_sec}s. Tried Deriv {short} and Stooq. "
        "Tip: enable ALLOW_SYNTHETIC_DATA=1 or upload a CSV."
    )

# ============================== Indicator helpers ============================
def _sma(s: pd.Series, p: int): return s.rolling(p).mean()
def _ema(s: pd.Series, p: int): return s.ewm(span=p, adjust=False).mean()
def _wma(s: pd.Series, p: int):
    w = np.arange(1, p+1)
    return s.rolling(p).apply(lambda x: (x*w).sum()/w.sum(), raw=True)
def _smma(s: pd.Series, p: int):
    alpha = 1/float(p)
    return s.ewm(alpha=alpha, adjust=False).mean()
def _tma(s: pd.Series, p: int):
    return s.rolling(p).mean().rolling(p).mean()

def _true_range(df):
    high, low, close = df["High"], df["Low"], df["Close"]
    prev_close = close.shift(1)
    tr = pd.concat([(high - low).abs(),
                    (high - prev_close).abs(),
                    (low - prev_close).abs()], axis=1).max(axis=1)
    return tr

def _rsi(close, period=14):
    delta = close.diff()
    up = delta.clip(lower=0).ewm(alpha=1/period, adjust=False).mean()
    down = (-delta.clip(upper=0)).ewm(alpha=1/period, adjust=False).mean()
    rs = up / down.replace(0, np.nan)
    return (100 - (100/(1+rs))).fillna(method="bfill")

def _stoch(df, k=14, d=3):
    lowk = df["Low"].rolling(k).min()
    highk = df["High"].rolling(k).max()
    kline = (df["Close"] - lowk) / (highk - lowk).replace(0, np.nan) * 100.0
    dline = kline.rolling(d).mean()
    return kline, dline

def _adx(df, period=14):
    high, low, close = df["High"], df["Low"], df["Close"]
    upmove = high.diff()
    downmove = -low.diff()
    plus_dm = np.where((upmove > downmove) & (upmove > 0), upmove, 0.0)
    minus_dm = np.where((downmove > upmove) & (downmove > 0), downmove, 0.0)
    tr = _true_range(df)
    atr = tr.ewm(alpha=1/period, adjust=False).mean()
    plus_di = 100 * pd.Series(plus_dm, index=close.index).ewm(alpha=1/period, adjust=False).mean() / atr
    minus_di = 100 * pd.Series(minus_dm, index=close.index).ewm(alpha=1/period, adjust=False).mean() / atr
    dx = (100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)).fillna(0)
    adx = dx.ewm(alpha=1/period, adjust=False).mean()
    return adx, plus_di, minus_di

def _atr(df, period=14):
    tr = _true_range(df)
    return tr.ewm(alpha=1/period, adjust=False).mean()

def _bb(close, period=20, mult=2.0):
    ma = close.rolling(period).mean()
    sd = close.rolling(period).std()
    upper = ma + mult*sd
    lower = ma - mult*sd
    width = (upper - lower)
    return ma, upper, lower, width

def _cci(df, period=20):
    tp = (df["High"] + df["Low"] + df["Close"]) / 3.0
    sma = tp.rolling(period).mean()
    md = (tp - sma).abs().rolling(period).mean()
    return (tp - sma) / (0.015*md).replace(0, np.nan)

def _donchian(df, period=20):
    up = df["High"].rolling(period).max()
    dn = df["Low"].rolling(period).min()
    mid = (up + dn) / 2
    return up, dn, mid

def _keltner(df, period=20, mult=2.0):
    ema = _ema(df["Close"], period)
    atr = _atr(df, period)
    upper = ema + mult*atr
    lower = ema - mult*atr
    return ema, upper, lower

def _envelopes(close, period=20, pct=2.0):
    ma = close.rolling(period).mean()
    up = ma * (1 + pct/100.0)
    dn = ma * (1 - pct/100.0)
    return ma, up, dn

def _ichimoku(df):
    high, low = df["High"], df["Low"]
    conv = (high.rolling(9).max() + low.rolling(9).min()) / 2
    base = (high.rolling(26).max() + low.rolling(26).min()) / 2
    span_a = (conv + base)/2
    span_b = (high.rolling(52).max() + low.rolling(52).min()) / 2
    lag = df["Close"].shift(-26)
    return conv, base, span_a.shift(26), span_b.shift(26), lag

def _macd(close, fast=12, slow=26, signal=9):
    macd = _ema(close, fast) - _ema(close, slow)
    sig = _ema(macd, signal)
    hist = macd - sig
    return macd, sig, hist

def _psar(df, af_step=0.02, af_max=0.2):
    high, low = df["High"].values, df["Low"].values
    length = len(df)
    psar = np.zeros(length)
    bull = True
    af = af_step
    ep = low[0]
    psar[0] = low[0]
    for i in range(1, length):
        prev = i-1
        psar[i] = psar[prev] + af*(ep - psar[prev])
        if bull:
            psar[i] = min(psar[i], low[prev], low[i])
            if high[i] > ep:
                ep = high[i]; af = min(af+af_step, af_max)
            if low[i] < psar[i]:
                bull = False; psar[i] = ep; ep = low[i]; af = af_step
        else:
            psar[i] = max(psar[i], high[prev], high[i])
            if low[i] < ep:
                ep = low[i]; af = min(af+af_step, af_max)
            if high[i] > psar[i]:
                bull = True; psar[i] = ep; ep = high[i]; af = af_step
    return pd.Series(psar, index=df.index)

def _momentum(close, period=10): return (close / close.shift(period) - 1) * 100
def _roc(close, period=10): return (close - close.shift(period)) / close.shift(period) * 100
def _willr(df, period=14):
    high, low, close = df["High"], df["Low"], df["Close"]
    return -100 * (high.rolling(period).max() - close) / (high.rolling(period).max() - low.rolling(period).min()).replace(0,np.nan)

def _supertrend(df, period=10, mult=3.0):
    atr = _atr(df, period)
    hl2 = (df["High"] + df["Low"]) / 2
    upper = hl2 + mult*atr
    lower = hl2 - mult*atr
    st = pd.Series(index=df.index, dtype=float)
    dir_up = True
    for i in range(len(df)):
        if i==0:
            st.iloc[i] = upper.iloc[i]; dir_up = True; continue
        if df["Close"].iloc[i] > st.iloc[i-1]: dir_up = True
        elif df["Close"].iloc[i] < st.iloc[i-1]: dir_up = False
        st.iloc[i] = max(lower.iloc[i], st.iloc[i-1]) if dir_up else min(upper.iloc[i], st.iloc[i-1])
    return st, upper, lower

def _vortex(df, period=14):
    tr = _true_range(df)
    vm_pos = (df["High"] - df["Low"].shift(1)).abs()
    vm_neg = (df["Low"] - df["High"].shift(1)).abs()
    vip = vm_pos.rolling(period).sum() / tr.rolling(period).sum()
    vin = vm_neg.rolling(period).sum() / tr.rolling(period).sum()
    return vip, vin

def _ao(df):
    mp = (df["High"] + df["Low"]) / 2
    return _sma(mp, 5) - _sma(mp, 34)

def _ac(df):
    ao = _ao(df)
    return ao - _sma(ao, 5)

def _bears_power(df, period=13): return df["Low"] - _ema(df["Close"], period)
def _bulls_power(df, period=13): return df["High"] - _ema(df["Close"], period)

def _demarker(df, period=14):
    up = (df["High"].diff().clip(lower=0)).rolling(period).sum()
    dn = (-df["Low"].diff().clip(upper=0)).rolling(period).sum()
    dem = up / (up + dn).replace(0, np.nan)
    return dem

def _osma(close, fast=12, slow=26, signal=9):
    macd, sig, _ = _macd(close, fast, slow, signal)
    return macd - sig

def _zigzag(close, pct=1.0):
    zz = pd.Series(index=close.index, dtype=float)
    if close.empty: return zz
    last_pivot = close.iloc[0]; last_dir = 0; zz.iloc[0]=last_pivot
    for i, p in enumerate(close):
        chg = (p-last_pivot)/last_pivot*100 if last_pivot!=0 else 0
        if last_dir>=0 and chg<=-pct:
            last_dir=-1; last_pivot=p; zz.iloc[i]=p
        elif last_dir<=0 and chg>=pct:
            last_dir=1; last_pivot=p; zz.iloc[i]=p
    return zz

# ---------- Alligator / Aroon / Fractal Bands / STC ----------
def _smma_series(s: pd.Series, period: int) -> pd.Series:
    alpha = 1.0 / float(period)
    return s.ewm(alpha=alpha, adjust=False).mean()

def _alligator(df: pd.DataFrame, jaw=13, teeth=8, lips=5, jaw_shift=8, teeth_shift=5, lips_shift=3):
    price = (df["High"] + df["Low"]) / 2.0
    jaw_line   = _smma_series(price, int(jaw)).shift(int(jaw_shift))
    teeth_line = _smma_series(price, int(teeth)).shift(int(teeth_shift))
    lips_line  = _smma_series(price, int(lips)).shift(int(lips_shift))
    return jaw_line, teeth_line, lips_line

def _aroon(df: pd.DataFrame, period=14):
    high = df["High"]; low = df["Low"]
    roll_high_idx = high.rolling(period+1).apply(lambda x: period - np.argmax(x[::-1]), raw=True)
    roll_low_idx  = low.rolling(period+1).apply(lambda x: period - np.argmin(x[::-1]),  raw=True)
    a_up   = (period - roll_high_idx) / period * 100.0
    a_dn   = (period - roll_low_idx)  / period * 100.0
    return a_up, a_dn

def _fractal_bands(df: pd.DataFrame, lookback=2, smooth=5):
    H = df["High"]; L = df["Low"]
    hi = (H.shift(2) < H.shift(1)) & (H.shift(1) < H) & (H.shift(-1) < H) & (H.shift(-2) < H)
    lo = (L.shift(2) > L.shift(1)) & (L.shift(1) > L) & (L.shift(-1) > L) & (L.shift(-2) > L)
    up  = H.where(hi).ffill().rolling(int(smooth)).max()
    dn  = L.where(lo).ffill().rolling(int(smooth)).min()
    return up, dn

def _stc(close: pd.Series, fast=23, slow=50, cycle=10):
    macd = _ema(close, int(fast)) - _ema(close, int(slow))
    lowk = macd.rolling(int(cycle)).min()
    highk = macd.rolling(int(cycle)).max()
    raw = (macd - lowk) / (highk - lowk).replace(0, np.nan) * 100.0
    stc1 = raw.ewm(span=3, adjust=False).mean()
    stc2 = stc1.ewm(span=3, adjust=False).mean()
    return stc2.clip(0, 100)

# ============================== Compute indicators ===========================
def compute_indicators(df: pd.DataFrame, ind_cfg: dict) -> dict:
    out = {}
    if df is None or df.empty: return out
    close = df["Close"]

    # Overlays
    if (cfg := ind_cfg.get("SMA", {})).get("enabled"):
        p = int(cfg.get("period", 50)); out["SMA"] = _sma(close, p)
    if (cfg := ind_cfg.get("EMA", {})).get("enabled"):
        p = int(cfg.get("period", 20)); out["EMA"] = _ema(close, p)
    if (cfg := ind_cfg.get("WMA", {})).get("enabled"):
        p = int(cfg.get("period", 20)); out["WMA"] = _wma(close, p)
    if (cfg := ind_cfg.get("SMMA", {})).get("enabled"):
        p = int(cfg.get("period", 20)); out["SMMA"] = _smma(close, p)
    if (cfg := ind_cfg.get("TMA", {})).get("enabled"):
        p = int(cfg.get("period", 20)); out["TMA"] = _tma(close, p)

    if (cfg := ind_cfg.get("BOLL", {})).get("enabled"):
        p = int(cfg.get("period",20)); mult=float(cfg.get("mult",2))
        ma, up, dn, width = _bb(close, p, mult)
        out["BB_MA"]=ma; out["BB_UP"]=up; out["BB_DN"]=dn; out["BB_WIDTH"]=width

    if (cfg := ind_cfg.get("KELTNER", {})).get("enabled"):
        p=int(cfg.get("period",20)); mult=float(cfg.get("mult",2))
        ma, up, dn = _keltner(df, p, mult)
        out["KC_MA"]=ma; out["KC_UP"]=up; out["KC_DN"]=dn

    if (cfg := ind_cfg.get("DONCHIAN", {})).get("enabled"):
        p=int(cfg.get("period",20)); up,dn,mid=_donchian(df,p); out["DON_UP"]=up; out["DON_DN"]=dn; out["DON_MID"]=mid

    if (cfg := ind_cfg.get("ENVELOPES", {})).get("enabled"):
        p=int(cfg.get("period",20)); pct=float(cfg.get("pct",2))
        ma, up, dn = _envelopes(close, p, pct)
        out["ENV_MA"]=ma; out["ENV_UP"]=up; out["ENV_DN"]=dn

    if (cfg := ind_cfg.get("ICHIMOKU", {})).get("enabled"):
        conv, base, span_a, span_b, lag = _ichimoku(df)
        out["ICH_CONV"]=conv; out["ICH_BASE"]=base; out["ICH_SA"]=span_a; out["ICH_SB"]=span_b; out["ICH_LAG"]=lag

    if (cfg := ind_cfg.get("PSAR", {})).get("enabled"):
        af=float(cfg.get("step",0.02)); afm=float(cfg.get("max",0.2))
        out["PSAR"]=_psar(df, af_step=af, af_max=afm)

    if (cfg := ind_cfg.get("SUPERTREND", {})).get("enabled"):
        p=int(cfg.get("period",10)); mult=float(cfg.get("mult",3))
        st, up, dn = _supertrend(df, p, mult)
        out["ST"]=st; out["ST_UP"]=up; out["ST_DN"]=dn

    # Alligator
    if (cfg := ind_cfg.get("ALLIGATOR", {})).get("enabled"):
        jaw  = int(cfg.get("jaw", 13))
        teeth= int(cfg.get("teeth", 8))
        lips = int(cfg.get("lips", 5))
        jsh  = int(cfg.get("jaw_shift", 8))
        tsh  = int(cfg.get("teeth_shift", 5))
        lsh  = int(cfg.get("lips_shift", 3))
        jaw_line, teeth_line, lips_line = _alligator(df, jaw, teeth, lips, jsh, tsh, lsh)
        out["ALLIG_JAW"] = jaw_line
        out["ALLIG_TEETH"] = teeth_line
        out["ALLIG_LIPS"] = lips_line

    # Fractal Chaos Bands
    if (cfg := ind_cfg.get("FRACTAL", {})).get("enabled"):
        look = int(cfg.get("lookback", 2))
        sm   = int(cfg.get("smooth", 5))
        up, dn = _fractal_bands(df, look, sm)
        out["FRACTAL_UP"] = up
        out["FRACTAL_DN"] = dn

    # Oscillators
    if (cfg := ind_cfg.get("RSI", {})).get("enabled", cfg.get("show", True)):
        p=int(cfg.get("period",14)); out["RSI"]=_rsi(close,p)

    if (cfg := ind_cfg.get("STOCH", {})).get("enabled", cfg.get("show", True)):
        k=int(cfg.get("k",14)); d=int(cfg.get("d",3))
        kline,dline = _stoch(df,k,d); out["STOCH_K"]=kline; out["STOCH_D"]=dline

    if (cfg := ind_cfg.get("ATR", {})).get("enabled", cfg.get("show", False)):
        p=int(cfg.get("period",14)); out["ATR"]=_atr(df,p)

    if (cfg := ind_cfg.get("ADX", {})).get("enabled", cfg.get("show", False)):
        p=int(cfg.get("period",14)); adx, pdm, ndm = _adx(df,p); out["ADX"]=adx; out["+DI"]=pdm; out["-DI"]=ndm

    if (cfg := ind_cfg.get("CCI", {})).get("enabled", False):
        p=int(cfg.get("period",20)); out["CCI"]=_cci(df,p)

    if (cfg := ind_cfg.get("MOMENTUM", {})).get("enabled", False):
        p=int(cfg.get("period",10)); out["MOM"]=_momentum(close,p)

    if (cfg := ind_cfg.get("ROC", {})).get("enabled", False):
        p=int(cfg.get("period",10)); out["ROC"]=_roc(close,p)

    if (cfg := ind_cfg.get("WILLR", {})).get("enabled", False):
        p=int(cfg.get("period",14)); out["WILLR"]=_willr(df,p)

    if (cfg := ind_cfg.get("VORTEX", {})).get("enabled", False):
        p=int(cfg.get("period",14)); vip, vin = _vortex(df,p); out["VI+"]=vip; out["VI-"]=vin

    if (cfg := ind_cfg.get("MACD", {})).get("enabled", False):
        f=int(cfg.get("fast",12)); s=int(cfg.get("slow",26)); g=int(cfg.get("signal",9))
        macd, sig, hist = _macd(close,f,s,g); out["MACD"]=macd; out["MACD_SIG"]=sig; out["MACD_HIST"]=hist

    if (cfg := ind_cfg.get("AO", {})).get("enabled", False):
        out["AO"]=_ao(df)
    if (cfg := ind_cfg.get("AC", {})).get("enabled", False):
        out["AC"]=_ac(df)

    if (cfg := ind_cfg.get("BEARS", {})).get("enabled", False):
        p=int(cfg.get("period",13)); out["BEARS"]=_bears_power(df,p)
    if (cfg := ind_cfg.get("BULLS", {})).get("enabled", False):
        p=int(cfg.get("period",13)); out["BULLS"]=_bulls_power(df,p)

    if (cfg := ind_cfg.get("DEMARKER", {})).get("enabled", False):
        p=int(cfg.get("period",14)); out["DEMARKER"]=_demarker(df,p)

    if (cfg := ind_cfg.get("OSMA", {})).get("enabled", False):
        out["OSMA"]=_osma(close)

    if (cfg := ind_cfg.get("ZIGZAG", {})).get("enabled", False):
        pct=float(cfg.get("pct",1.0)); out["ZZ"]=_zigzag(close,pct)

    # Aroon + STC
    if (cfg := ind_cfg.get("AROON", {})).get("enabled"):
        p = int(cfg.get("period", 14))
        up, dn = _aroon(df, p)
        out["AROON_UP"] = up; out["AROON_DN"] = dn
    if (cfg := ind_cfg.get("STC", {})).get("enabled"):
        f = int(cfg.get("fast", 23)); s = int(cfg.get("slow", 50)); c = int(cfg.get("cycle", 10))
        out["STC"] = _stc(close, f, s, c)

    return out

# ============================== Strategy / backtest ==========================
def simple_rule_engine(df: pd.DataFrame, ind: dict, rule_name: str):
    signals = []
    close = df["Close"]

    def add(sig_idx, direction, bars):
        all_idx = df.index
        pos = all_idx.get_indexer([sig_idx])[0]
        exp_pos = min(pos + max(1,bars), len(all_idx)-1)
        signals.append({"index": sig_idx, "direction": direction, "expiry_idx": all_idx[exp_pos]})

    rule = (rule_name or "BASE").upper()

    if rule == "TREND":
        sma = ind.get("SMA") or _sma(close,50)
        cross_up = (close.shift(1) <= sma.shift(1)) & (close > sma)
        cross_dn = (close.shift(1) >= sma.shift(1)) & (close < sma)
        for ts in close.index[cross_up.fillna(False)]: add(ts,"BUY",5)
        for ts in close.index[cross_dn.fillna(False)]: add(ts,"SELL",5)
        return signals

    if rule == "CHOP":
        rsi = ind.get("RSI") or _rsi(close,14)
        bounce_up = (rsi.shift(1) < 50) & (rsi >= 50)
        bounce_dn = (rsi.shift(1) > 50) & (rsi <= 50)
        for ts in rsi.index[bounce_up.fillna(False)]: add(ts,"BUY",3)
        for ts in rsi.index[bounce_dn.fillna(False)]: add(ts,"SELL",3)
        return signals

    # BASE: Stochastic cross
    k = ind.get("STOCH_K"); d = ind.get("STOCH_D")
    if k is not None and d is not None:
        cross_up = (k.shift(1) < d.shift(1)) & (k >= d)
        cross_dn = (k.shift(1) > d.shift(1)) & (k <= d)
        for ts in k.index[cross_up.fillna(False)]: add(ts,"BUY",5)
        for ts in k.index[cross_dn.fillna(False)]: add(ts,"SELL",5)
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

# ============================== Plotting =====================================
def plot_signals(df, signals, indicators_cfg, strategy, tf, expiry) -> str:
    os.makedirs("static/plots", exist_ok=True)
    if df is None or df.empty:
        out = "empty.png"
        plt.figure(figsize=(10,2)); plt.text(0.5,0.5,"No data", ha="center"); plt.savefig("static/plots/"+out); plt.close()
        return out

    ind = compute_indicators(df, indicators_cfg or {})
    ts = df.index

    want_rsi = "RSI" in ind
    want_sto = ("STOCH_K" in ind and "STOCH_D" in ind) or ("AROON_UP" in ind) or ("STC" in ind)
    rows = 1 + (1 if want_rsi else 0) + (1 if want_sto else 0)
    fig_h = 6 + (2 if want_rsi else 0) + (2 if want_sto else 0)
    fig, axes = plt.subplots(rows, 1, figsize=(14, fig_h), sharex=True,
                             gridspec_kw={"height_ratios":[4] + ([1] if want_rsi else []) + ([1] if want_sto else [])})
    if rows == 1: axes = [axes]
    ax_price = axes[0]
    ax_price.set_facecolor("#0b0f17")
    fig.patch.set_facecolor("#0b0f17")

    # --- Candles ---
    dt = mdates.date2num(pd.to_datetime(ts).to_pydatetime())
    for i,(t,o,h,l,c) in enumerate(zip(dt, df["Open"], df["High"], df["Low"], df["Close"])):
        color = "#17c964" if c>=o else "#f44336"
        ax_price.vlines(t, l, h, color=color, linewidth=0.8, alpha=.9, zorder=2)
        ax_price.add_patch(plt.Rectangle((t-0.001, min(o,c)), 0.002, max(abs(c-o),1e-6),
                                         facecolor=color, edgecolor=color, linewidth=.6, alpha=.95, zorder=3))
    ax_price.set_ylabel("Price", color="#e8edf7")

    overlay_colors = {
        "SMA":"#ffd166", "EMA":"#60a5fa", "WMA":"#f59e0b", "SMMA":"#a78bfa","TMA":"#ef4444",
        "BB_MA":"#9ca3af","BB_UP":"#374151","BB_DN":"#374151",
        "KC_MA":"#94a3b8","KC_UP":"#475569","KC_DN":"#475569",
        "DON_UP":"#4b5563","DON_DN":"#4b5563","DON_MID":"#64748b",
        "ENV_MA":"#9ca3af","ENV_UP":"#52525b","ENV_DN":"#52525b",
        "ICH_CONV":"#10b981","ICH_BASE":"#ef4444","ICH_SA":"#22d3ee","ICH_SB":"#fb7185",
        "PSAR":"#f472b6","ST":"#22c55e",
        "ALLIG_JAW":"#60a5fa","ALLIG_TEETH":"#ef4444","ALLIG_LIPS":"#22c55e",
        "FRACTAL_UP":"#94a3b8","FRACTAL_DN":"#94a3b8",
    }
    for key, col in overlay_colors.items():
        if key in ind:
            ax_price.plot(ts, ind[key], color=col, linewidth=1.0, alpha=0.9, label=key)

    if "PSAR" in ind:
        ax_price.scatter(ts, ind["PSAR"], s=10, color="#f472b6", alpha=.8, zorder=5)

    if signals:
        buy_x=[]; buy_y=[]; sell_x=[]; sell_y=[]
        for s in signals:
            if s["index"] in df.index:
                y = float(df.loc[s["index"],"Close"])
                if s["direction"]=="BUY": buy_x.append(s["index"]); buy_y.append(y)
                else: sell_x.append(s["index"]); sell_y.append(y)
        if buy_x: ax_price.scatter(buy_x, buy_y, marker="^", s=60, color="#16a34a", label="BUY", zorder=6)
        if sell_x: ax_price.scatter(sell_x, sell_y, marker="v", s=60, color="#ef4444", label="SELL", zorder=6)
        ax_price.legend(loc="upper left", fontsize=9)

    cur_row = 1
    if want_rsi:
        ax_rsi = axes[cur_row]
        ax_rsi.set_facecolor("#0b0f17")
        ax_rsi.plot(ts, ind["RSI"], color="#60a5fa", linewidth=1.2, label="RSI")
        ax_rsi.axhline(70, color="#6b7280", linewidth=.8, linestyle="--")
        ax_rsi.axhline(30, color="#6b7280", linewidth=.8, linestyle="--")
        ax_rsi.set_ylim(0,100); ax_rsi.set_yticks([0,30,50,70,100])
        ax_rsi.legend(loc="upper left", fontsize=9)
        cur_row += 1

    if want_sto:
        ax_st = axes[cur_row]
        ax_st.set_facecolor("#0b0f17")
        if "STOCH_K" in ind: ax_st.plot(ts, ind["STOCH_K"], color="#22c55e", linewidth=1.2, label="%K")
        if "STOCH_D" in ind: ax_st.plot(ts, ind["STOCH_D"], color="#f59e0b", linewidth=1.0, label="%D")
        if "AROON_UP" in ind: ax_st.plot(ts, ind["AROON_UP"], color="#93c5fd", linewidth=.9, alpha=.9, label="Aroon Up")
        if "AROON_DN" in ind: ax_st.plot(ts, ind["AROON_DN"], color="#fca5a5", linewidth=.9, alpha=.9, label="Aroon Down")
        if "STC" in ind: ax_st.plot(ts, ind["STC"], color="#a78bfa", linewidth=1.0, alpha=.9, label="STC")
        ax_st.axhline(80, color="#6b7280", linewidth=.8, linestyle="--")
        ax_st.axhline(20, color="#6b7280", linewidth=.8, linestyle="--")
        ax_st.set_ylim(0,100); ax_st.set_yticks([0,20,50,80,100])
        ax_st.legend(loc="upper left", fontsize=9)

    ax_price.grid(color="#111827", linestyle="--", linewidth=0.5)
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    for ax in axes:
        ax.tick_params(colors="#cbd5e1")
        for spine in ax.spines.values(): spine.set_color("#1f2937")

    fig.suptitle(f"{strategy} • TF={tf} • Exp={expiry}", color="#e5e7eb", fontsize=14, fontweight="bold", y=0.98)
    fig.autofmt_xdate()
    fig.tight_layout(rect=[0,0,1,0.96])

    out_name = f"{df.index[-1].strftime('%Y%m%d_%H%M%S')}_{strategy}_{tf}.png"
    path = os.path.join("static","plots", out_name)
    fig.savefig(path, dpi=140, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    return out_name

# ============================== Backtest =====================================
EXPIRY_TO_BARS = {"1m":1,"3m":3,"5m":5,"10m":10,"30m":30,"1h":60,"4h":240}

def backtest_run(df: pd.DataFrame, strategy: str, indicators: dict, expiry: str):
    ind = compute_indicators(df, indicators or {})
    signals = simple_rule_engine(df, ind, (strategy or "BASE").upper())
    bars = EXPIRY_TO_BARS.get((expiry or "5m").lower(), 5)
    fixed = []
    for s in signals:
        i = s["index"]
        pos = df.index.get_indexer([i])[0]
        exp_pos = min(pos + max(1,bars), len(df.index)-1)
        s["expiry_idx"] = df.index[exp_pos]
        fixed.append(s)
    stats = evaluate_signals_outcomes(df, fixed)
    return fixed, stats
