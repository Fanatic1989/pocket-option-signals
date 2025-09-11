# utils.py — config store, Deriv fetch, PO/Deriv symbols, backtest + plotting
import os, json, sqlite3, math
from datetime import datetime, time as dtime, timedelta, timezone
import pytz
import pandas as pd
import numpy as np
import mplfinance as mpf
import matplotlib.pyplot as plt

DB_PATH = os.getenv("SQLITE_PATH", "members.db")
TZ = pytz.timezone(os.getenv("APP_TZ", "America/Port_of_Spain"))

# ------------------------------- SQLite helpers ------------------------------
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

# ------------------------------ Trading window -------------------------------
def within_window(cfg: dict) -> bool:
    start = cfg.get("window_start", "08:00")
    end   = cfg.get("window_end", "17:00")
    now_tt = datetime.now(TZ).time()
    s_h,s_m = [int(x) for x in start.split(":")]
    e_h,e_m = [int(x) for x in end.split(":")]
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
        s = s.strip().upper()
        out.append(PO2DERIV.get(s, s))  # leave as-is if already frx*
    return out

# ----------------------------- Data loading/fetch ----------------------------
def load_csv(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # Expected columns: time,open,high,low,close (any case)
    cols = {c.lower(): c for c in df.columns}
    o = cols.get("open"); h = cols.get("high"); l = cols.get("low"); c = cols.get("close")
    t = cols.get("time") or cols.get("timestamp") or cols.get("date")
    if not all([o,h,l,c,t]): raise ValueError("CSV missing OHLC/time columns")
    df = df[[t,o,h,l,c]].copy()
    df.columns = ["time","Open","High","Low","Close"]
    # parse time
    df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
    df = df.dropna(subset=["time"]).set_index("time").sort_index()
    return df

def fetch_deriv_history(symbol: str, granularity_sec: int, days: int=5) -> pd.DataFrame:
    """
    Simple HTTP fetch using Deriv API (candles endpoint).
    NOTE: If Deriv blocks direct requests from Render free tier, keep CSV fallback.
    """
    import requests
    end = int(datetime.now(timezone.utc).timestamp())
    start = end - days*86400
    url = "https://api.deriv.com/api/explore/candles"
    # Alternative documented REST-like endpoint via API Gateway is not official; fallback to CSV uploader if blocked.
    params = {"symbol": symbol, "granularity": granularity_sec, "start": start, "end": end}
    try:
        r = requests.get(url, params=params, timeout=15)
        if not r.ok:
            raise RuntimeError(f"Deriv HTTP {r.status_code}")
        js = r.json()
        rows = js.get("candles") or js.get("history") or []
        if not rows: raise RuntimeError("empty candles")
        df = pd.DataFrame(rows)
        # Normalize
        # Expect fields: epoch, open, high, low, close
        cand_time = (df.get("epoch") or df.get("time")).astype(int)
        out = pd.DataFrame({
            "time": pd.to_datetime(cand_time, unit="s", utc=True),
            "Open": pd.to_numeric(df.get("open"), errors="coerce"),
            "High": pd.to_numeric(df.get("high"), errors="coerce"),
            "Low" : pd.to_numeric(df.get("low"), errors="coerce"),
            "Close": pd.to_numeric(df.get("close"), errors="coerce"),
        }).dropna()
        return out.set_index("time").sort_index()
    except Exception as e:
        raise RuntimeError(f"Deriv fetch failed: {e}")

# ----------------------------- Backtest helpers ------------------------------
EXPIRY_TO_BARS = {
    "1m":1,"3m":3,"5m":5,"10m":10,"30m":30,
    "1h":60,"4h":240
}

def compute_indicators(df: pd.DataFrame, ind_cfg: dict) -> dict:
    out = {}
    close = df["Close"]
    if "SMA" in ind_cfg:
        for p in (ind_cfg["SMA"] if isinstance(ind_cfg["SMA"], (list,tuple)) else [ind_cfg["SMA"]]):
            out[f"SMA({p})"] = close.rolling(int(p)).mean()
    if "EMA" in ind_cfg:
        for p in (ind_cfg["EMA"] if isinstance(ind_cfg["EMA"], (list,tuple)) else [ind_cfg["EMA"]]):
            out[f"EMA({p})"] = close.ewm(span=int(p), adjust=False).mean()
    if "WMA" in ind_cfg:
        for p in (ind_cfg["WMA"] if isinstance(ind_cfg["WMA"], (list,tuple)) else [ind_cfg["WMA"]]):
            w = np.arange(1, int(p)+1)
            out[f"WMA({p})"] = close.rolling(int(p)).apply(lambda x: (x*w).sum()/w.sum(), raw=True)
    if "SMMA" in ind_cfg:
        for p in (ind_cfg["SMMA"] if isinstance(ind_cfg["SMMA"], (list,tuple)) else [ind_cfg["SMMA"]]):
            out[f"SMMA({p})"] = close.ewm(alpha=1/int(p), adjust=False).mean()
    if "TMA" in ind_cfg:
        for p in (ind_cfg["TMA"] if isinstance(ind_cfg["TMA"], (list,tuple)) else [ind_cfg["TMA"]]):
            out[f"TMA({p})"] = close.rolling(int(p)).mean().rolling(int(p)).mean()
    if "RSI" in ind_cfg and isinstance(ind_cfg["RSI"], dict) and ind_cfg["RSI"].get("show", True):
        pr = int(ind_cfg["RSI"].get("period", 14))
        delta = close.diff()
        up = delta.clip(lower=0).ewm(alpha=1/pr, adjust=False).mean()
        down = (-delta.clip(upper=0)).ewm(alpha=1/pr, adjust=False).mean()
        rs = up / down
        out["RSI"] = 100 - (100/(1+rs))
    if "STOCH" in ind_cfg and isinstance(ind_cfg["STOCH"], dict) and ind_cfg["STOCH"].get("show", True):
        k = int(ind_cfg["STOCH"].get("k", 14)); d = int(ind_cfg["STOCH"].get("d",3))
        lowk = df["Low"].rolling(k).min(); highk = df["High"].rolling(k).max()
        kline = (close - lowk) / (highk - lowk) * 100.0
        dline = kline.rolling(d).mean()
        out["STOCH_K"] = kline; out["STOCH_D"] = dline
    return out

def simple_rule_engine(df: pd.DataFrame, ind: dict, rule_name: str):
    """
    Three built-ins: BASE / TREND / CHOP.
    You can expand to parse your English rules elsewhere; here we keep something consistent.
    Returns a list of signals: [{"index": ts, "direction": "BUY"/"SELL", "expiry_idx": ts2}, ...]
    """
    signals = []
    close = df["Close"]

    def add(sig_idx, direction, bars):
        # expiry index bars ahead (clip to last)
        all_idx = df.index
        pos = all_idx.get_indexer([sig_idx])[0]
        exp_pos = min(pos + max(1,bars), len(all_idx)-1)
        signals.append({"index": sig_idx, "direction": direction, "expiry_idx": all_idx[exp_pos]})

    if rule_name == "TREND":
        # Price > SMA50 => buy dips; Price < SMA50 => sell rallies
        sma50 = ind.get("SMA(50)")
        if sma50 is None: return signals
        above = close > sma50
        cross_up = (above & (~above.shift(1).fillna(False)))
        cross_dn = ((~above) & (above.shift(1).fillna(False)))
        for ts, v in cross_up[cross_up].items():
            add(ts, "BUY", 5)
        for ts, v in cross_dn[cross_dn].items():
            add(ts, "SELL", 5)
        return signals

    if rule_name == "CHOP":
        # RSI midline bounces around 50 generate alternating signals
        rsi = ind.get("RSI")
        if rsi is None: return signals
        bounce_up = (rsi.shift(1) < 50) & (rsi >= 50)
        bounce_dn = (rsi.shift(1) > 50) & (rsi <= 50)
        for ts in rsi.index[bounce_up.fillna(False)]:
            add(ts, "BUY", 3)
        for ts in rsi.index[bounce_dn.fillna(False)]:
            add(ts, "SELL", 3)
        return signals

    # BASE: Stoch cross
    k = ind.get("STOCH_K"); d = ind.get("STOCH_D")
    if k is not None and d is not None:
        cross_up = (k.shift(1) < d.shift(1)) & (k >= d)
        cross_dn = (k.shift(1) > d.shift(1)) & (k <= d)
        for ts in k.index[cross_up.fillna(False)]:
            add(ts, "BUY", 5)
        for ts in k.index[cross_dn.fillna(False)]:
            add(ts, "SELL", 5)
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

def plot_signals(df, signals, indicators, strategy, tf, expiry) -> str:
    if df.empty:
        out = "empty.png"
        os.makedirs("static/plots", exist_ok=True)
        plt.figure(figsize=(8,2)); plt.text(0.5,0.5,"No data", ha="center"); plt.savefig("static/plots/"+out); plt.close()
        return out

    addplots = []
    # overlay MAs
    for name, val in indicators.items():
        upper = name.upper()
        if upper in ("SMA","EMA","WMA","SMMA","TMA"):
            periods = val if isinstance(val, (list,tuple)) else [val]
            for p in periods:
                p = int(p)
                if upper=="SMA":
                    ser = df["Close"].rolling(p).mean()
                elif upper=="EMA":
                    ser = df["Close"].ewm(span=p, adjust=False).mean()
                elif upper=="WMA":
                    w = np.arange(1,p+1)
                    ser = df["Close"].rolling(p).apply(lambda x: (x*w).sum()/w.sum(), raw=True)
                elif upper=="SMMA":
                    ser = df["Close"].ewm(alpha=1/p, adjust=False).mean()
                else: # TMA
                    ser = df["Close"].rolling(p).mean().rolling(p).mean()
                addplots.append(mpf.make_addplot(ser, panel=0, width=1))

    # RSI panel
    rsi_spec = indicators.get("RSI")
    if isinstance(rsi_spec, dict) and rsi_spec.get("show", True):
        pr = int(rsi_spec.get("period",14))
        delta = df["Close"].diff()
        up = delta.clip(lower=0).ewm(alpha=1/pr, adjust=False).mean()
        down = (-delta.clip(upper=0)).ewm(alpha=1/pr, adjust=False).mean()
        rs = up / down
        rsi = 100 - (100/(1+rs))
        addplots.append(mpf.make_addplot(rsi, panel=1, ylabel="RSI"))

    # Stoch panel
    st = indicators.get("STOCH")
    if isinstance(st, dict) and st.get("show", True):
        k = int(st.get("k",14)); d = int(st.get("d",3))
        lowk = df["Low"].rolling(k).min(); highk = df["High"].rolling(k).max()
        kline = (df["Close"] - lowk) / (highk - lowk) * 100.0
        dline = kline.rolling(d).mean()
        addplots.append(mpf.make_addplot(kline, panel=2, ylabel="%K"))
        addplots.append(mpf.make_addplot(dline, panel=2))

    fig, axlist = mpf.plot(df, type="candle", style="yahoo", addplot=addplots,
                           returnfig=True, volume=False, figsize=(13,8),
                           title=f"{strategy} • TF={tf} • Exp={expiry}")
    ax = axlist[0]

    # markers & expiry shading
    buy_x= []; buy_y= []; sell_x=[]; sell_y=[]
    for s in signals:
        if s["index"] not in df.index: continue
        px = float(df.loc[s["index"],"Close"])
        if s["direction"]=="BUY":
            buy_x.append(s["index"]); buy_y.append(px)
        else:
            sell_x.append(s["index"]); sell_y.append(px)
        ex = s.get("expiry_idx")
        if ex in df.index:
            ax.axvspan(ex, ex, alpha=0.20)

    if buy_x: ax.scatter(buy_x, buy_y, marker="^", s=80)
    if sell_x: ax.scatter(sell_x, sell_y, marker="v", s=80)

    os.makedirs("static/plots", exist_ok=True)
    out_name = f"{df.index[-1].strftime('%Y%m%d_%H%M%S')}_{strategy}_{tf}.png"
    fig.savefig(os.path.join("static","plots", out_name), dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out_name

def backtest_run(df: pd.DataFrame, strategy: str, indicators: dict, expiry: str) -> tuple[list, dict]:
    # Compute ind, run rule engine, compute outcomes
    ind = compute_indicators(df, indicators)
    from utils import simple_rule_engine  # self-import ok
    signals = simple_rule_engine(df, ind, strategy.upper())
    # adjust expiry bars according to selection
    bars = EXPIRY_TO_BARS.get(expiry.lower(), 5)
    # already set in simple_rule_engine, but ensure bars minimum
    fixed = []
    for s in signals:
        i = s["index"]
        pos = df.index.get_indexer([i])[0]
        exp_pos = min(pos + max(1,bars), len(df.index)-1)
        s["expiry_idx"] = df.index[exp_pos]
        fixed.append(s)
    stats = evaluate_signals_outcomes(df, fixed)
    return fixed, stats
