# utils.py — config store, TZ, sqlite, Deriv fetch, symbols, backtest helpers, plotting
import os, json, sqlite3, math
from datetime import datetime, time as dtime, timedelta, timezone

# Timezone exports expected by routes.py
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

DB_PATH = os.getenv("SQLITE_PATH", "members.db")

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
    # Expected columns: time/timestamp/date, open, high, low, close
    cols = {c.lower(): c for c in df.columns}
    o = cols.get("open"); h = cols.get("high"); l = cols.get("low"); c = cols.get("close")
    t = cols.get("time") or cols.get("timestamp") or cols.get("date")
    if not all([o,h,l,c,t]): raise ValueError("CSV missing OHLC/time columns")
    df = df[[t,o,h,l,c]].copy()
    df.columns = ["time","Open","High","Low","Close"]
    df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
    df = df.dropna(subset=["time"]).set_index("time").sort_index()
    return df

def fetch_deriv_history(symbol: str, granularity_sec: int, days: int=5) -> pd.DataFrame:
    import requests
    end = int(datetime.now(timezone.utc).timestamp())
    start = end - days*86400
    # Deriv public candles endpoint (docs: app/explore). 404s → caller handles.
    url = "https://api.deriv.com/api/explore/candles"
    params = {"symbol": symbol, "granularity": granularity_sec, "start": start, "end": end}
    r = requests.get(url, params=params, timeout=15)
    if not r.ok:
        raise RuntimeError(f"Deriv HTTP {r.status_code}")
    js = r.json()
    rows = js.get("candles") or js.get("history") or []
    if not rows: raise RuntimeError("empty candles")
    df = pd.DataFrame(rows)
    # Normalize common fields
    epoch = df.get("epoch") if "epoch" in df else df.get("time")
    out = pd.DataFrame({
        "time": pd.to_datetime(epoch.astype(int), unit="s", utc=True),
        "Open": pd.to_numeric(df.get("open"), errors="coerce"),
        "High": pd.to_numeric(df.get("high"), errors="coerce"),
        "Low" : pd.to_numeric(df.get("low"), errors="coerce"),
        "Close": pd.to_numeric(df.get("close"), errors="coerce"),
    }).dropna()
    return out.set_index("time").sort_index()

# ----------------------------- Backtest helpers ------------------------------
EXPIRY_TO_BARS = {
    "1m":1,"3m":3,"5m":5,"10m":10,"30m":30,
    "1h":60,"4h":240
}

def compute_indicators(df: pd.DataFrame, ind_cfg: dict) -> dict:
    out = {}
    close = df["Close"]
    if "SMA" in ind_cfg:
        vals = ind_cfg["SMA"] if isinstance(ind_cfg["SMA"], (list,tuple)) else [ind_cfg["SMA"]]
        for p in vals:
            out[f"SMA({p})"] = close.rolling(int(p)).mean()
    if "EMA" in ind_cfg:
        vals = ind_cfg["EMA"] if isinstance(ind_cfg["EMA"], (list,tuple)) else [ind_cfg["EMA"]]
        for p in vals:
            out[f"EMA({p})"] = close.ewm(span=int(p), adjust=False).mean()
    if "WMA" in ind_cfg:
        vals = ind_cfg["WMA"] if isinstance(ind_cfg["WMA"], (list,tuple)) else [ind_cfg["WMA"]]
        for p in vals:
            w = np.arange(1, int(p)+1)
            out[f"WMA({p})"] = close.rolling(int(p)).apply(lambda x: (x*w).sum()/w.sum(), raw=True)
    if "SMMA" in ind_cfg:
        vals = ind_cfg["SMMA"] if isinstance(ind_cfg["SMMA"], (list,tuple)) else [ind_cfg["SMMA"]]
        for p in vals:
            out[f"SMMA({p})"] = close.ewm(alpha=1/int(p), adjust=False).mean()
    if "TMA" in ind_cfg:
        vals = ind_cfg["TMA"] if isinstance(ind_cfg["TMA"], (list,tuple)) else [ind_cfg["TMA"]]
        for p in vals:
            out[f"TMA({p})"] = close.rolling(int(p)).mean().rolling(int(p)).mean()
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
        for ts, v in cross_up[cross_up].items():
            add(ts, "BUY", 5)
        for ts, v in cross_dn[cross_dn].items():
            add(ts, "SELL", 5)
        return signals

    if rule_name == "CHOP":
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

# ----------------------------- Plotting (improved) ---------------------------
def _limit_df(df: pd.DataFrame, last_n: int = 250) -> pd.DataFrame:
    """Keep only the most recent N rows for a readable chart."""
    if df is None or df.empty: return df
    if len(df) <= last_n: return df
    return df.iloc[-last_n:].copy()

def plot_signals(df, signals, indicators, strategy, tf, expiry) -> str:
    """
    High-clarity plot:
      - last 250 bars
      - price panel with candlesticks + MA overlays (if enabled)
      - oscillator panel (RSI + Stoch if enabled)
      - BUY/SELL arrows at entries
      - shaded expiry windows from entry → expiry
    Returns filename saved under static/plots/.
    """
    os.makedirs("static/plots", exist_ok=True)

    if df is None or df.empty:
        out = "empty.png"
        plt.figure(figsize=(8,2)); plt.text(0.5,0.5,"No data", ha="center"); plt.savefig("static/plots/"+out); plt.close()
        return out

    # keep last N bars for readability
    df = _limit_df(df, 250)
    ind = compute_indicators(df, indicators or {})

    out_name = f"{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{strategy}_{tf}.png"
    path = os.path.join("static","plots", out_name)

    # ---------- mplfinance version ----------
    if HAVE_MPLFIN:
        # Prepare overlays
        addplots = []
        # MAs / overlays
        for name in list(ind.keys()):
            if name.startswith(("SMA(","EMA(","WMA(","SMMA(","TMA(")):
                addplots.append(mpf.make_addplot(ind[name], panel=0, width=1.2, alpha=0.9))
        # Oscillators on panel 1
        osc_used = False
        if "RSI" in ind:
            addplots.append(mpf.make_addplot(ind["RSI"], panel=1, ylim=(0,100), width=1.2))
            osc_used = True
        if "STOCH_K" in ind and "STOCH_D" in ind:
            addplots.append(mpf.make_addplot(ind["STOCH_K"], panel=1, width=1.0))
            addplots.append(mpf.make_addplot(ind["STOCH_D"], panel=1, width=1.0))
            osc_used = True

        panels = 2 if osc_used else 1
        fig, axlist = mpf.plot(
            df, type="candle",
            style=mpf.make_mpf_style(base_mpf_style="yahoo", marketcolors=mpf.make_marketcolors(up="#16a34a", down="#dc2626", wick="inherit", edge="inherit")),
            addplot=addplots,
            returnfig=True, volume=False,
            figsize=(14, 8 if panels==2 else 6),
            title=f"{strategy} • TF={tf} • Exp={expiry}",
            panel_ratios=(3,1) if panels==2 else None
        )
        ax_price = axlist[0]
        ax_osc = axlist[2] if panels==2 else None

        # Arrows at entry
        buy_x=[]; buy_y=[]; sell_x=[]; sell_y=[]
        for s in signals or []:
            if s["index"] not in df.index: continue
            px = float(df.loc[s["index"],"Close"])
            if s["direction"]=="BUY":
                buy_x.append(s["index"]); buy_y.append(px)
            else:
                sell_x.append(s["index"]); sell_y.append(px)
            # shade expiry window
            if s.get("expiry_idx") in df.index:
                ax_price.axvspan(s["index"], s["expiry_idx"], alpha=0.12, color="#22c55e" if s["direction"]=="BUY" else "#ef4444")

        if buy_x: ax_price.scatter(buy_x, buy_y, marker="^", s=90, zorder=5)
        if sell_x: ax_price.scatter(sell_x, sell_y, marker="v", s=90, zorder=5)

        # Guide lines on RSI/Stoch panel (if present)
        if ax_osc is not None:
            ax_osc.axhline(70, color="#ef4444", lw=1, ls="--", alpha=.4)
            ax_osc.axhline(30, color="#22c55e", lw=1, ls="--", alpha=.4)
            ax_osc.set_ylim(0,100)

        fig.tight_layout()
        fig.savefig(path, dpi=160, bbox_inches="tight")
        plt.close(fig)
        return out_name

    # ---------- Fallback matplotlib (no mplfinance) ----------
    # Minimal but clear price+osc setup
    ds = df.reset_index().rename(columns={"index":"timestamp","Open":"open","High":"high","Low":"low","Close":"close"})
    ds["timestamp"] = pd.to_datetime(ds["time"] if "time" in ds else ds["timestamp"])
    import matplotlib.dates as mdates
    ts = mdates.date2num(pd.to_datetime(ds["timestamp"]).dt.to_pydatetime())

    # Decide rows: 2 if we have RSI/Stoch
    osc_used = ("RSI" in ind) or ("STOCH_K" in ind and "STOCH_D" in ind)
    fig_h = 8 if osc_used else 6
    fig, ax = plt.subplots(2 if osc_used else 1, 1, figsize=(14, fig_h), sharex=True,
                           gridspec_kw={"height_ratios":[3,1]} if osc_used else None)
    ax_price = ax[0] if osc_used else ax

    # draw candles
    for i,(t,o,h,l,c) in enumerate(zip(ts, ds["open"], ds["high"], ds["low"], ds["close"])):
        color = "#16a34a" if c>=o else "#dc2626"
        ax_price.vlines(t, l, h, color=color, linewidth=1.0, alpha=.9)
        ax_price.add_patch(plt.Rectangle((t-0.0018, min(o,c)), 0.0036, max(abs(c-o),1e-6),
                                   facecolor=color, edgecolor=color, linewidth=.8, alpha=.95))
    # overlays
    for name, series in ind.items():
        if name.startswith(("SMA(","EMA(","WMA(","SMMA(","TMA(")):
            ax_price.plot(series.index, series.values, linewidth=1.2, alpha=.9)

    # entries + shaded expiry
    for s in signals or []:
        i = s["index"]
        if i in df.index:
            px = float(df.loc[i,"Close"])
            ax_price.scatter([mdates.date2num(i)],[px], marker="^" if s["direction"]=="BUY" else "v", s=90, zorder=5)
            if s.get("expiry_idx") in df.index:
                ax_price.axvspan(i, s["expiry_idx"], alpha=0.12, color="#22c55e" if s["direction"]=="BUY" else "#ef4444")

    ax_price.set_title(f"{strategy} • TF={tf} • Exp={expiry}")
    ax_price.set_ylabel("Price")
    ax_price.grid(alpha=.25)

    if osc_used:
        ax_osc = ax[1]
        if "RSI" in ind:
            ax_osc.plot(ind["RSI"].index, ind["RSI"].values, linewidth=1.2)
        if "STOCH_K" in ind and "STOCH_D" in ind:
            ax_osc.plot(ind["STOCH_K"].index, ind["STOCH_K"].values, linewidth=1.0)
            ax_osc.plot(ind["STOCH_D"].index, ind["STOCH_D"].values, linewidth=1.0)
        ax_osc.axhline(70, lw=1, ls="--", alpha=.4)
        ax_osc.axhline(30, lw=1, ls="--", alpha=.4)
        ax_osc.set_ylim(0,100)
        ax_osc.set_ylabel("Osc")
        ax_osc.grid(alpha=.25)

    fig.autofmt_xdate(); fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight"); plt.close(fig)
    return out_name

def backtest_run(df: pd.DataFrame, strategy: str, indicators: dict, expiry: str):
    from utils import simple_rule_engine  # self-import ok
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
