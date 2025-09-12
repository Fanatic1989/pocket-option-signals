# utils.py — config store, TZ, sqlite, Deriv fetch, symbols, backtest helpers, plotting
import os, json, sqlite3
from datetime import datetime, time as dtime, timezone
import pandas as pd
import numpy as np

# -----------------------------------------------------------------------------
# Timezone
# -----------------------------------------------------------------------------
TIMEZONE = os.getenv("APP_TZ", "America/Port_of_Spain")
try:
    import pytz
    TZ = pytz.timezone(TIMEZONE)
except Exception:
    from zoneinfo import ZoneInfo
    TZ = ZoneInfo(TIMEZONE)

# -----------------------------------------------------------------------------
# Optional mplfinance
# -----------------------------------------------------------------------------
try:
    import mplfinance as mpf
    HAVE_MPLFIN = True
except Exception:
    HAVE_MPLFIN = False
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
# App / DB
# -----------------------------------------------------------------------------
DB_PATH = os.getenv("SQLITE_PATH", "members.db")
DERIV_APP_ID = os.getenv("DERIV_APP_ID", "99185").strip()  # <- your app id

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

# -----------------------------------------------------------------------------
# Trading window
# -----------------------------------------------------------------------------
def within_window(cfg: dict) -> bool:
    w = cfg.get("window") or {}
    start = w.get("start") or cfg.get("window_start","08:00")
    end   = w.get("end")   or cfg.get("window_end","17:00")
    now_tt = datetime.now(TZ).time()
    s_h,s_m = [int(x) for x in (start or "08:00").split(":")[:2]]
    e_h,e_m = [int(x) for x in (end   or "17:00").split(":")[:2]]
    return dtime(s_h,s_m) <= now_tt <= dtime(e_h,e_m)

# -----------------------------------------------------------------------------
# Symbols & mapping
# -----------------------------------------------------------------------------
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
        if s.startswith(("FRX","frx")): out.append("frx"+s[-6:])
        else: out.append(PO2DERIV.get(s, s))
    return out

# -----------------------------------------------------------------------------
# CSV loader
# -----------------------------------------------------------------------------
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

# -----------------------------------------------------------------------------
# Deriv fetch (robust, adds app_id to every request)
# -----------------------------------------------------------------------------
def _series_from_maybe_dict_col(col):
    s = pd.Series(col)
    if s.dtype == "O":
        def pick(v):
            if isinstance(v, (int, float, np.integer, np.floating, str)): return v
            if isinstance(v, dict):
                for k in ("epoch","time","t","open","o","high","h","low","l","close","c","value","v"):
                    if k in v: return v[k]
            return np.nan
        return s.map(pick)
    return s

def _normalize_ohlc_frame(js: dict) -> pd.DataFrame:
    rows = js.get("candles") or js.get("history") or js.get("prices") or js.get("data") or []
    if not rows: raise RuntimeError("empty candles")
    df = pd.DataFrame(rows)

    epoch = None
    for k in ("epoch","time","t"):
        if k in df:
            epoch = _series_from_maybe_dict_col(df[k]); break
    if epoch is None and len(df.columns)==1 and isinstance(df.iloc[0,0], dict):
        sub = pd.DataFrame(df.iloc[:,0].tolist())
        for k in ("epoch","time","t"):
            if k in sub:
                epoch = _series_from_maybe_dict_col(sub[k]); df = sub; break
    if epoch is None: raise RuntimeError("no epoch/time column")

    def grab(name_list):
        for n in name_list:
            if n in df: return _series_from_maybe_dict_col(df[n])
        return pd.Series(np.nan, index=df.index)

    o = grab(["open","o"]); h = grab(["high","h"]); l = grab(["low","l"]); c = grab(["close","c"])
    out = pd.DataFrame({
        "time": pd.to_datetime(pd.to_numeric(epoch, errors="coerce"), unit="s", utc=True),
        "Open": pd.to_numeric(o, errors="coerce"),
        "High": pd.to_numeric(h, errors="coerce"),
        "Low" : pd.to_numeric(l, errors="coerce"),
        "Close": pd.to_numeric(c, errors="coerce")
    }).dropna()
    if out.empty: raise RuntimeError("parsed empty candles")
    return out.set_index("time").sort_index()

def fetch_deriv_history(symbol: str, granularity_sec: int, days: int = 5) -> pd.DataFrame:
    """
    Fetch OHLC candles for a Deriv (frx*) symbol.
      - clamps granularity to allowed set
      - tries symbol casings: frxPAIR and FRXPAIR
      - tries v4/v3 price_history (count) then explore/candles (count or start/end)
      - adds DERIV_APP_ID to every request (fixes 404 in many regions)
    """
    import requests

    # Normalize symbol
    sym = (symbol or "").strip()
    if not sym: raise RuntimeError("missing symbol")
    if not sym.lower().startswith("frx"):
        sym = PO2DERIV.get(sym.upper(), sym)
    if not sym.lower().startswith("frx"):
        raise RuntimeError(f"unsupported symbol '{symbol}' (expect frx* or PO major)")
    sym_lc = "frx" + sym[-6:].upper()
    sym_uc = sym_lc.upper()
    candidates = [sym_lc, sym_uc] if sym_lc != sym_uc else [sym_lc]

    # Clamp TF
    allowed = [60, 120, 180, 300, 600, 900, 1800, 3600, 14400, 86400]
    try: g_in = int(granularity_sec)
    except Exception: g_in = 300
    gran = min(allowed, key=lambda g: abs(g - g_in))

    approx = min(2000, max(300, int(days * 86400 // gran)))
    attempts = []

    def try_request(url, params):
        params = dict(params or {})
        if DERIV_APP_ID: params["app_id"] = DERIV_APP_ID  # <-- important
        r = requests.get(url, params=params, timeout=15)
        attempts.append((url, r.status_code))
        if not r.ok: return None
        try:
            return _normalize_ohlc_frame(r.json())
        except Exception:
            return None

    # v4/v3 with count
    for s in candidates:
        for base in ("https://api.deriv.com/api/v4/price_history",
                     "https://api.deriv.com/api/v3/price_history"):
            df = try_request(base, {"symbol": s, "granularity": gran, "count": approx})
            if df is not None and not df.empty: return df

    # explore/candles (count then range)
    for s in candidates:
        df = try_request("https://api.deriv.com/api/explore/candles",
                         {"symbol": s, "granularity": gran, "count": approx})
        if df is not None and not df.empty: return df
        end = int(datetime.now(timezone.utc).timestamp())
        start = end - days*86400
        df = try_request("https://api.deriv.com/api/explore/candles",
                         {"symbol": s, "granularity": gran, "start": start, "end": end})
        if df is not None and not df.empty: return df

    raise RuntimeError(
        f"Deriv fetch failed for symbol={symbol}→{candidates}, granularity={gran}. "
        f"Attempts: {attempts}. Tip: ensure DERIV_APP_ID is set and pair exists; "
        f"or upload CSV."
    )

# -----------------------------------------------------------------------------
# Backtest helpers
# -----------------------------------------------------------------------------
EXPIRY_TO_BARS = {"1m":1,"3m":3,"5m":5,"10m":10,"30m":30,"1h":60,"4h":240}

def compute_indicators(df: pd.DataFrame, ind_cfg: dict) -> dict:
    out = {}
    close = df["Close"]
    if "SMA" in ind_cfg:
        vals = ind_cfg["SMA"] if isinstance(ind_cfg["SMA"], (list,tuple)) else [ind_cfg["SMA"]]
        for p in vals: out[f"SMA({p})"] = close.rolling(int(p)).mean()
    if "EMA" in ind_cfg:
        vals = ind_cfg["EMA"] if isinstance(ind_cfg["EMA"], (list,tuple)) else [ind_cfg["EMA"]]
        for p in vals: out[f"EMA({p})"] = close.ewm(span=int(p), adjust=False).mean()
    if "WMA" in ind_cfg:
        vals = ind_cfg["WMA"] if isinstance(ind_cfg["WMA"], (list,tuple)) else [ind_cfg["WMA"]]
        for p in vals:
            w = np.arange(1, int(p)+1)
            out[f"WMA({p})"] = close.rolling(int(p)).apply(lambda x: (x*w).sum()/w.sum(), raw=True)
    if "SMMA" in ind_cfg:
        vals = ind_cfg["SMMA"] if isinstance(ind_cfg["SMMA"], (list,tuple)) else [ind_cfg["SMMA"]]
        for p in vals: out[f"SMMA({p})"] = close.ewm(alpha=1/int(p), adjust=False).mean()
    if "TMA" in ind_cfg:
        vals = ind_cfg["TMA"] if isinstance(ind_cfg["TMA"], (list,tuple)) else [ind_cfg["TMA"]]
        for p in vals: out[f"TMA({p})"] = close.rolling(int(p)).mean().rolling(int(p)).mean()
    if "RSI" in ind_cfg and isinstance(ind_cfg["RSI"], dict) and ind_cfg["RSI"].get("show", True):
        pr = int(ind_cfg["RSI"].get("period", 14))
        delta = close.diff()
        up = delta.clip(lower=0).ewm(alpha=1/pr, adjust=False).mean()
        down = (-delta.clip(upper=0)).ewm(alpha=1/pr, adjust=False).mean()
        rs = up / down.replace(0, np.nan)
        out["RSI"] = (100 - (100/(1+rs))).fillna(method="bfill")
    if "STOCH" in ind_cfg and isinstance(ind_cfg["STOCH"], dict) and ind_cfg["STOCH"].get("show", True):
        k = int(ind_cfg["STOCH"].get("k", 14)); d = int(ind_cfg["STOCH"].get("d",3))
        lowk = df["Low"].rolling(k).min(); highk = df["High"].rolling(k).max()
        kline = (close - lowk) / (highk - lowk).replace(0, np.nan) * 100.0
        dline = kline.rolling(d).mean()
        out["STOCH_K"] = kline; out["STOCH_D"] = dline
    return out

def simple_rule_engine(df: pd.DataFrame, ind: dict, rule_name: str):
    signals = []
    close = df["Close"]

    def add(sig_idx, direction, bars):
        all_idx = df.index
        pos = all_idx.get_indexer([sig_idx])[0]
        exp_pos = min(pos + max(1,bars), len(all_idx)-1)
        signals.append({"index": sig_idx, "direction": direction, "expiry_idx": all_idx[exp_pos]})

    if rule_name == "TREND":
        sma50 = ind.get("SMA(50)")
        if sma50 is None: return signals
        above = close > sma50
        cross_up = (above & (~above.shift(1).fillna(False)))
        cross_dn = ((~above) & (above.shift(1).fillna(False)))
        for ts, _ in cross_up[cross_up].items(): add(ts, "BUY", 5)
        for ts, _ in cross_dn[cross_dn].items(): add(ts, "SELL", 5)
        return signals

    if rule_name == "CHOP":
        rsi = ind.get("RSI")
        if rsi is None: return signals
        bounce_up = (rsi.shift(1) < 50) & (rsi >= 50)
        bounce_dn = (rsi.shift(1) > 50) & (rsi <= 50)
        for ts in rsi.index[bounce_up.fillna(False)]: add(ts, "BUY", 3)
        for ts in rsi.index[bounce_dn.fillna(False)]: add(ts, "SELL", 3)
        return signals

    # BASE: Stoch cross
    k = ind.get("STOCH_K"); d = ind.get("STOCH_D")
    if k is not None and d is not None:
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

# -----------------------------------------------------------------------------
# Plotting (clear)
# -----------------------------------------------------------------------------
def _limit_df(df: pd.DataFrame, last_n: int = 250) -> pd.DataFrame:
    if df is None or df.empty: return df
    if len(df) <= last_n: return df
    return df.iloc[-last_n:].copy()

def plot_signals(df, signals, indicators, strategy, tf, expiry) -> str:
    os.makedirs("static/plots", exist_ok=True)
    if df is None or df.empty:
        out = "empty.png"
        plt.figure(figsize=(8,2)); plt.text(0.5,0.5,"No data", ha="center"); plt.savefig("static/plots/"+out); plt.close()
        return out

    df = _limit_df(df, 250)
    ind = compute_indicators(df, indicators or {})
    out_name = f"{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{strategy}_{tf}.png"
    path = os.path.join("static","plots", out_name)

    if HAVE_MPLFIN:
        addplots = []
        for name in list(ind.keys()):
            if name.startswith(("SMA(","EMA(","WMA(","SMMA(","TMA(")):
                addplots.append(mpf.make_addplot(ind[name], panel=0, width=1.2, alpha=0.9))
        osc_used = False
        if "RSI" in ind:
            addplots.append(mpf.make_addplot(ind["RSI"], panel=1, ylim=(0,100), width=1.2)); osc_used = True
        if "STOCH_K" in ind and "STOCH_D" in ind:
            addplots.append(mpf.make_addplot(ind["STOCH_K"], panel=1, width=1.0))
            addplots.append(mpf.make_addplot(ind["STOCH_D"], panel=1, width=1.0)); osc_used = True

        panels = 2 if osc_used else 1
        fig, axlist = mpf.plot(
            df, type="candle",
            style=mpf.make_mpf_style(
                base_mpf_style="yahoo",
                marketcolors=mpf.make_marketcolors(up="#16a34a", down="#dc2626", wick="inherit", edge="inherit")
            ),
            addplot=addplots, returnfig=True, volume=False,
            figsize=(14, 8 if panels==2 else 6),
            title=f"{strategy} • TF={tf} • Exp={expiry}",
            panel_ratios=(3,1) if panels==2 else None
        )
        ax_price = axlist[0]

        if signals:
            buys_x, buys_y, sells_x, sells_y = [], [], [], []
            for s in signals:
                if s["index"] not in df.index: continue
                y = float(df.loc[s["index"],"Close"])
                (buys_x if s["direction"]=="BUY" else sells_x).append(s["index"])
                (buys_y if s["direction"]=="BUY" else sells_y).append(y)
                if s.get("expiry_idx") in df.index:
                    ax_price.axvspan(s["index"], s["expiry_idx"], alpha=.12,
                                     color="#22c55e" if s["direction"]=="BUY" else "#ef4444")
            if buys_x: ax_price.scatter(buys_x, buys_y, marker="^", s=90, zorder=5)
            if sells_x: ax_price.scatter(sells_x, sells_y, marker="v", s=90, zorder=5)

        fig.tight_layout(); fig.savefig(path, dpi=160, bbox_inches="tight"); plt.close(fig)
        return out_name

    # Fallback matplotlib
    ds = df.reset_index()
    import matplotlib.dates as mdates
    ts = mdates.date2num(pd.to_datetime(ds["time"]).dt.to_pydatetime())
    fig, ax = plt.subplots(1,1,figsize=(14,6))
    for t,o,h,l,c in zip(ts, ds["Open"], ds["High"], ds["Low"], ds["Close"]):
        color = "#16a34a" if c>=o else "#dc2626"
        ax.vlines(t, l, h, color=color, linewidth=1.0, alpha=.9)
        ax.add_patch(plt.Rectangle((t-0.0018, min(o,c)), 0.0036, max(abs(c-o),1e-6),
                                   facecolor=color, edgecolor=color, linewidth=.8, alpha=.95))
    for name, series in ind.items():
        if name.startswith(("SMA(","EMA(","WMA(","SMMA(","TMA(")):
            ax.plot(series.index, series.values, linewidth=1.2, alpha=.9)
    if signals:
        for s in signals:
            i = s["index"]
            if i in df.index:
                y = float(df.loc[i,"Close"])
                ax.scatter([mdates.date2num(i)],[y], marker="^" if s["direction"]=="BUY" else "v", s=90, zorder=5)
                if s.get("expiry_idx") in df.index:
                    ax.axvspan(i, s["expiry_idx"], alpha=.12, color="#22c55e" if s["direction"]=="BUY" else "#ef4444")
    ax.set_title(f"{strategy} • TF={tf} • Exp={expiry}"); ax.set_ylabel("Price"); ax.grid(alpha=.25)
    fig.autofmt_xdate(); fig.tight_layout(); fig.savefig(path, dpi=160, bbox_inches="tight"); plt.close(fig)
    return out_name

# -----------------------------------------------------------------------------
# Backtest run
# -----------------------------------------------------------------------------
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
