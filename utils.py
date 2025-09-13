# utils.py — config store, TZ, sqlite, Deriv fetch, symbols, indicators, backtest & plotting
import os, json, sqlite3, math
from datetime import datetime, date, time as dtime, timedelta, timezone

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
def _parse_hhmm(s: str, default: str) -> dtime:
    try:
        hh, mm = [int(x) for x in (s or default).split(":")[:2]]
        return dtime(hh, mm)
    except Exception:
        return dtime(8, 0)

def within_window(cfg: dict) -> bool:
    """
    True if NOW is within configured window AND Mon-Fri.
    Defaults: 08:00–17:00 in APP_TZ, weekdays only.
    """
    w = cfg.get("window") or {}
    start = _parse_hhmm(w.get("start"), "08:00")
    end   = _parse_hhmm(w.get("end"), "17:00")
    mf_only = w.get("monday_friday_only", True)

    try:
        now_local = datetime.now(TZ)
    except Exception:
        now_local = datetime.utcnow().replace(tzinfo=timezone.utc)
    wd = now_local.weekday()  # 0=Mon, 6=Sun
    if mf_only and wd >= 5:
        return False
    tt = now_local.time()
    return (start <= tt <= end)

def trading_open_now(cfg: dict) -> bool:
    """Alias used by routes to gate sending."""
    return within_window(cfg)

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
        if s.startswith("frx"):
            out.append(s)
        elif s.upper().startswith("FRX"):
            out.append("frx" + s[3:])
        else:
            out.append(PO2DERIV.get(s.upper(), s))
    return out

# ============================== Data loading/fetch ============================
def load_csv(csv_file) -> pd.DataFrame:
    df = pd.read_csv(csv_file)
    cols = {c.lower(): c for c in df.columns}
    o = cols.get("open"); h = cols.get("high"); l = cols.get("low"); c = cols.get("close")
    t = cols.get("time") or cols.get("timestamp") or cols.get("date")
    if not all([o,h,l,c,t]): raise ValueError("CSV missing OHLC/time columns")
    df = df[[t,o,h,l,c]].copy()
    df.columns = ["time","Open","High","Low","Close"]
    df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
    df = df.dropna(subset=["time"]).set_index("time").sort_index()
    return df

def _deriv_price_history(symbol: str, granularity: int, count: int=600, app_id: str|None=None):
    """
    Only Deriv endpoints (no Yahoo). Tries v4 -> v3 -> explore.
    Returns pandas DataFrame or raises RuntimeError with info.
    """
    import requests
    attempts = []
    qs = {"ticks_history": symbol, "style": "candles", "granularity": granularity, "count": int(count)}
    if app_id:
        qs["app_id"] = app_id

    for base in ("https://api.deriv.com/api/v4/price_history",
                 "https://api.deriv.com/api/v3/price_history"):
        try:
            r = requests.get(base, params=qs, timeout=15)
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
                        return out.set_index("time").sort_index()
        except Exception:
            attempts.append((base, "EXC"))

    # Explore fallback
    try:
        url = "https://api.deriv.com/api/explore/candles"
        r = requests.get(url, params={"symbol": symbol, "granularity": granularity, "count": int(count)}, timeout=15)
        attempts.append((url, r.status_code))
        if r.ok:
            js = r.json()
            rows = js.get("candles") or js.get("history") or []
            d = pd.DataFrame(rows)
            epoch = d.get("epoch") if "epoch" in d else d.get("time")
            out = pd.DataFrame({
                "time": pd.to_datetime(epoch.astype(int), unit="s", utc=True),
                "Open": pd.to_numeric(d.get("open"), errors="coerce"),
                "High": pd.to_numeric(d.get("high"), errors="coerce"),
                "Low" : pd.to_numeric(d.get("low"), errors="coerce"),
                "Close": pd.to_numeric(d.get("close"), errors="coerce"),
            }).dropna()
            if not out.empty:
                return out.set_index("time").sort_index()
    except Exception:
        attempts.append(("explore/candles", "EXC"))

    app = f" (DERIV_APP_ID={app_id})" if app_id else ""
    raise RuntimeError(f"Deriv fetch failed for symbol={symbol}, granularity={granularity}. Attempts: {attempts}{app}")

def fetch_deriv_history(symbol: str, granularity_sec: int, count: int=600) -> pd.DataFrame:
    app_id = os.getenv("DERIV_APP_ID") or os.getenv("DERIV_APPID") or os.getenv("DERIV_APP")
    return _deriv_price_history(symbol, int(granularity_sec), count=int(count), app_id=app_id)

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
            st.iloc[i] = upper.iloc[i]
            dir_up = True
            continue
        if df["Close"].iloc[i] > st.iloc[i-1]:
            dir_up = True
        elif df["Close"].iloc[i] < st.iloc[i-1]:
            dir_up = False
        if dir_up:
            st.iloc[i] = max(lower.iloc[i], st.iloc[i-1])
        else:
            st.iloc[i] = min(upper.iloc[i], st.iloc[i-1])
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
        conv, base, sa, sb, lag = _ichimoku(df)
        out["ICH_CONV"]=conv; out["ICH_BASE"]=base; out["ICH_SA"]=sa; out["ICH_SB"]=sb; out["ICH_LAG"]=lag

    if (cfg := ind_cfg.get("PSAR", {})).get("enabled"):
        af=float(cfg.get("step",0.02)); afm=float(cfg.get("max",0.2))
        out["PSAR"]=_psar(df, af_step=af, af_max=afm)

    if (cfg := ind_cfg.get("SUPERTREND", {})).get("enabled"):
        p=int(cfg.get("period",10)); mult=float(cfg.get("mult",3))
        st, up, dn = _supertrend(df, p, mult)
        out["ST"]=st; out["ST_UP"]=up; out["ST_DN"]=dn

    # Oscillators
    if (cfg := ind_cfg.get("RSI", {})).get("enabled"):
        p=int(cfg.get("period",14)); out["RSI"]=_rsi(close,p)

    if (cfg := ind_cfg.get("STOCH", {})).get("enabled"):
        k=int(cfg.get("k",14)); d=int(cfg.get("d",3))
        kline,dline = _stoch(df,k,d); out["STOCH_K"]=kline; out["STOCH_D"]=dline

    if (cfg := ind_cfg.get("ATR", {})).get("enabled"):
        p=int(cfg.get("period",14)); out["ATR"]=_atr(df,p)

    if (cfg := ind_cfg.get("ADX", {})).get("enabled"):
        p=int(cfg.get("period",14)); adx, pdm, ndm = _adx(df,p); out["ADX"]=adx; out["+DI"]=pdm; out["-DI"]=ndm

    if (cfg := ind_cfg.get("CCI", {})).get("enabled"):
        p=int(cfg.get("period",20)); out["CCI"]=_cci(df,p)

    if (cfg := ind_cfg.get("MOMENTUM", {})).get("enabled"):
        p=int(cfg.get("period",10)); out["MOM"]=_momentum(close,p)

    if (cfg := ind_cfg.get("ROC", {})).get("enabled"):
        p=int(cfg.get("period",10)); out["ROC"]=_roc(close,p)

    if (cfg := ind_cfg.get("WILLR", {})).get("enabled"):
        p=int(cfg.get("period",14)); out["WILLR"]=_willr(df,p)

    if (cfg := ind_cfg.get("VORTEX", {})).get("enabled"):
        p=int(cfg.get("period",14)); vip, vin = _vortex(df,p); out["VI+"]=vip; out["VI-"]=vin

    if (cfg := ind_cfg.get("MACD", {})).get("enabled"):
        f=int(cfg.get("fast",12)); s=int(cfg.get("slow",26)); g=int(cfg.get("signal",9))
        macd, sig, hist = _macd(close,f,s,g); out["MACD"]=macd; out["MACD_SIG"]=sig; out["MACD_HIST"]=hist

    if (cfg := ind_cfg.get("AO", {})).get("enabled"): out["AO"]=_ao(df)
    if (cfg := ind_cfg.get("AC", {})).get("enabled"): out["AC"]=_ac(df)

    if (cfg := ind_cfg.get("BEARS", {})).get("enabled"):
        p=int(cfg.get("period",13)); out["BEARS"]=_bears_power(df,p)
    if (cfg := ind_cfg.get("BULLS", {})).get("enabled"):
        p=int(cfg.get("period",13)); out["BULLS"]=_bulls_power(df,p)

    if (cfg := ind_cfg.get("DEMARKER", {})).get("enabled"):
        p=int(cfg.get("period",14)); out["DEMARKER"]=_demarker(df,p)

    if (cfg := ind_cfg.get("OSMA", {})).get("enabled"): out["OSMA"]=_osma(close)

    if (cfg := ind_cfg.get("ZIGZAG", {})).get("enabled"):
        pct=float(cfg.get("pct",1.0)); out["ZZ"]=_zigzag(close,pct)

    return out

# ============================== Simple strategy / signals ====================
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
    want_sto = "STOCH_K" in ind and "STOCH_D" in ind
    rows = 1 + (1 if want_rsi else 0) + (1 if want_sto else 0)
    fig_h = 6 + (2 if want_rsi else 0) + (2 if want_sto else 0)
    fig, axes = plt.subplots(rows, 1, figsize=(14, fig_h), sharex=True,
                             gridspec_kw={"height_ratios":[4] + ([1] if want_rsi else []) + ([1] if want_sto else [])})
    if rows == 1: axes = [axes]
    ax_price = axes[0]
    ax_price.set_facecolor("#0b0f17")
    fig.patch.set_facecolor("#0b0f17")

    # Candles
    dt = mdates.date2num(pd.to_datetime(ts).to_pydatetime())
    for i,(t,o,h,l,c) in enumerate(zip(dt, df["Open"], df["High"], df["Low"], df["Close"])):
        color = "#17c964" if c>=o else "#f44336"
        ax_price.vlines(t, l, h, color=color, linewidth=0.8, alpha=.9, zorder=2)
        ax_price.add_patch(plt.Rectangle((t-0.001, min(o,c)), 0.002, max(abs(c-o),1e-6),
                                         facecolor=color, edgecolor=color, linewidth=.6, alpha=.95, zorder=3))
    ax_price.set_ylabel("Price", color="#e8edf7")

    # Overlays
    overlay_colors = {
        "SMA":"#ffd166", "EMA":"#60a5fa", "WMA":"#f59e0b", "SMMA":"#a78bfa","TMA":"#ef4444",
        "BB_MA":"#9ca3af","BB_UP":"#374151","BB_DN":"#374151",
        "KC_MA":"#94a3b8","KC_UP":"#475569","KC_DN":"#475569",
        "DON_UP":"#4b5563","DON_DN":"#4b5563","DON_MID":"#64748b",
        "ENV_MA":"#9ca3af","ENV_UP":"#52525b","ENV_DN":"#52525b",
        "ICH_CONV":"#10b981","ICH_BASE":"#ef4444","ICH_SA":"#22d3ee","ICH_SB":"#fb7185",
        "PSAR":"#f472b6","ST":"#22c55e","ZZ":"#93c5fd"
    }
    for key, col in overlay_colors.items():
        if key in ind:
            ax_price.plot(ts, ind[key], color=col, linewidth=1.0, alpha=0.9, label=key)
    if "PSAR" in ind:
        ax_price.scatter(ts, ind["PSAR"], s=10, color="#f472b6", alpha=.8, zorder=5)

    # Markers
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

    # RSI panel
    cur = 1
    if want_rsi:
        ax = axes[cur]; cur += 1
        ax.set_facecolor("#0b0f17")
        ax.plot(ts, ind["RSI"], color="#60a5fa", linewidth=1.2, label="RSI")
        ax.axhline(70, color="#6b7280", linewidth=.8, linestyle="--")
        ax.axhline(30, color="#6b7280", linewidth=.8, linestyle="--")
        ax.set_ylim(0,100); ax.set_yticks([0,30,50,70,100]); ax.legend(loc="upper left", fontsize=9)

    # Stoch panel
    if want_sto:
        ax = axes[cur]
        ax.set_facecolor("#0b0f17")
        ax.plot(ts, ind["STOCH_K"], color="#22c55e", linewidth=1.2, label="%K")
        ax.plot(ts, ind["STOCH_D"], color="#f59e0b", linewidth=1.0, label="%D")
        ax.axhline(80, color="#6b7280", linewidth=.8, linestyle="--")
        ax.axhline(20, color="#6b7280", linewidth=.8, linestyle="--")
        ax.set_ylim(0,100); ax.set_yticks([0,20,50,80,100]); ax.legend(loc="upper left", fontsize=9)

    ax_price.grid(color="#111827", linestyle="--", linewidth=0.5)
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    for ax in axes:
        ax.tick_params(colors="#cbd5e1")
        for spine in ax.spines.values(): spine.set_color("#1f2937")

    fig.suptitle(f"{strategy} • TF={tf} • Exp={expiry}", color="#e5e7eb", fontsize=14, fontweight="bold", y=0.98)
    fig.autofmt_xdate(); fig.tight_layout(rect=[0,0,1,0.96])

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
