# utils.py — unified config, timezone, symbols, Deriv fetch, CSV, backtest & plotting
import os, json, sqlite3
from datetime import datetime, time as dtime, timedelta, timezone
import math

import pytz
import requests
import pandas as pd
import numpy as np
import mplfinance as mpf
import matplotlib.pyplot as plt

# ========================= Timezone ==========================================
# Expose both a string and a tz object (routes import TIMEZONE, TZ)
TIMEZONE = os.getenv("APP_TZ", os.getenv("TIMEZONE", "America/Port_of_Spain"))
TZ = pytz.timezone(TIMEZONE)

# ========================= SQLite config store ===============================
DB_PATH = os.getenv("SQLITE_PATH", os.getenv("DB_PATH", "members.db"))

def exec_sql(sql: str, params: tuple | list = (), fetch: bool = False):
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    try:
        cur = conn.cursor()
        cur.execute(sql, params or ())
        rows = cur.fetchall() if fetch else None
        conn.commit()
        return rows
    finally:
        conn.close()

# single-row config table (keyed)
exec_sql("""CREATE TABLE IF NOT EXISTS app_config(
  k TEXT PRIMARY KEY,
  v TEXT
)""")

def get_config() -> dict:
    row = exec_sql("SELECT v FROM app_config WHERE k='config'", fetch=True)
    if not row:
        return {}
    try:
        return json.loads(row[0][0])
    except Exception:
        return {}

def set_config(cfg: dict):
    exec_sql(
        "INSERT INTO app_config(k,v) VALUES('config',?) "
        "ON CONFLICT(k) DO UPDATE SET v=excluded.v",
        (json.dumps(cfg),),
    )

# ========================= Trading window helper =============================
def within_window(cfg: dict | None) -> bool:
    """
    True if now (in TIMEZONE) is inside the configured trading window.

    Supports either:
      cfg = {"window": {"start":"08:00","end":"17:00","timezone":"America/Port_of_Spain"}}
    OR flat keys:
      cfg = {"window_start":"08:00","window_end":"17:00"}
    """
    cfg = cfg or {}
    w = cfg.get("window") or {}

    # Prefer nested window if present
    start_s = w.get("start") or cfg.get("window_start") or "08:00"
    end_s   = w.get("end")   or cfg.get("window_end")   or "17:00"
    tz_name = w.get("timezone") or TIMEZONE

    tz = pytz.timezone(tz_name)
    now_tt = datetime.now(tz).time()

    def _parse_hhmm(s: str) -> dtime:
        hh, mm = (s or "00:00").split(":")[:2]
        return dtime(int(hh), int(mm))

    start_t = _parse_hhmm(start_s)
    end_t   = _parse_hhmm(end_s)

    if start_t <= end_t:
        return start_t <= now_tt <= end_t
    # window passes midnight
    return (now_tt >= start_t) or (now_tt <= end_t)

# ========================= Symbols & mapping =================================
PO_PAIRS = [
    "EURUSD","GBPUSD","USDJPY","USDCHF","USDCAD","AUDUSD","NZDUSD",
    "EURGBP","EURJPY","GBPJPY","EURAUD","AUDJPY","CADJPY","CHFJPY"
]
DERIV_PAIRS = [
    "frxEURUSD","frxGBPUSD","frxUSDJPY","frxUSDCHF","frxUSDCAD","frxAUDUSD","frxNZDUSD",
    "frxEURGBP","frxEURJPY","frxGBPJPY","frxEURAUD","frxAUDJPY","frxCADJPY","frxCHFJPY"
]
# Aliases used by other builds
PO_MAJOR = PO_PAIRS[:]           # alias
DERIV_FRX = DERIV_PAIRS[:]       # alias

PO2DERIV = dict(zip(PO_PAIRS, DERIV_PAIRS))
DERIV2PO = {v: k for k, v in PO2DERIV.items()}

def convert_po_to_deriv(symbols: list[str]) -> list[str]:
    out = []
    for s in symbols or []:
        s = (s or "").strip().upper().replace("/", "")
        out.append(PO2DERIV.get(s, s))  # keep frx* as-is
    return out

# ========================= Data loading / fetch ==============================
def load_csv(csv_file_or_path) -> pd.DataFrame:
    """
    Accepts a file-like (Werkzeug FileStorage) or a path.
    Expects columns: time/timestamp/date + open/high/low/close (any case).
    Returns a DataFrame indexed by UTC time with columns Open/High/Low/Close.
    """
    if hasattr(csv_file_or_path, "read"):
        df = pd.read_csv(csv_file_or_path)
    else:
        df = pd.read_csv(str(csv_file_or_path))

    cols = {c.lower(): c for c in df.columns}
    o = cols.get("open"); h = cols.get("high"); l = cols.get("low"); c = cols.get("close")
    t = cols.get("time") or cols.get("timestamp") or cols.get("date")
    if not all([o, h, l, c, t]):
        raise ValueError("CSV missing OHLC/time columns")

    df = df[[t, o, h, l, c]].copy()
    df.columns = ["time", "Open", "High", "Low", "Close"]
    df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
    df = df.dropna(subset=["time"]).set_index("time").sort_index()
    # ensure numeric
    for col in ["Open", "High", "Low", "Close"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.dropna(subset=["Close"])

def fetch_deriv_history(symbol: str, granularity_sec: int, days: int = 5) -> pd.DataFrame:
    """
    Lightweight Deriv candle fetch. If Deriv blocks the endpoint on your plan,
    keep using CSV uploads or your separate data_fetch helper.
    """
    end = int(datetime.now(timezone.utc).timestamp())
    start = end - int(days) * 86400
    url = "https://api.deriv.com/api/explore/candles"
    params = {"symbol": symbol, "granularity": int(granularity_sec), "start": start, "end": end}

    r = requests.get(url, params=params, timeout=15)
    if not r.ok:
        raise RuntimeError(f"Deriv HTTP {r.status_code}")

    js = r.json()
    rows = js.get("candles") or js.get("history") or []
    if not rows:
        raise RuntimeError("empty candles")

    df = pd.DataFrame(rows)
    # Normalize common fields
    cand_time = (df.get("epoch") if "epoch" in df else df.get("time")).astype(int)
    out = pd.DataFrame({
        "time": pd.to_datetime(cand_time, unit="s", utc=True),
        "Open": pd.to_numeric(df.get("open"), errors="coerce"),
        "High": pd.to_numeric(df.get("high"), errors="coerce"),
        "Low" : pd.to_numeric(df.get("low"), errors="coerce"),
        "Close": pd.to_numeric(df.get("close"), errors="coerce"),
    }).dropna()
    return out.set_index("time").sort_index()

# ========================= Simple backtest helpers ===========================
# (Kept for compatibility with the leaner routes variant; your richer build can
#  still rely on indicators.py/pandas_ta through routes.)

EXPIRY_TO_BARS = {
    "1m": 1, "3m": 3, "5m": 5, "10m": 10, "30m": 30,
    "1h": 60, "4h": 240
}

def compute_indicators(df: pd.DataFrame, ind_cfg: dict) -> dict:
    out = {}
    close = df["Close"]

    def _as_list(v):
        return v if isinstance(v, (list, tuple)) else [v]

    if "SMA" in ind_cfg:
        for p in _as_list(ind_cfg["SMA"]):
            p = int(p); out[f"SMA({p})"] = close.rolling(p).mean()
    if "EMA" in ind_cfg:
        for p in _as_list(ind_cfg["EMA"]):
            p = int(p); out[f"EMA({p})"] = close.ewm(span=p, adjust=False).mean()
    if "WMA" in ind_cfg:
        for p in _as_list(ind_cfg["WMA"]):
            p = int(p); w = np.arange(1, p + 1)
            out[f"WMA({p})"] = close.rolling(p).apply(lambda x: (x * w).sum() / w.sum(), raw=True)
    if "SMMA" in ind_cfg:
        for p in _as_list(ind_cfg["SMMA"]):
            p = int(p); out[f"SMMA({p})"] = close.ewm(alpha=1 / p, adjust=False).mean()
    if "TMA" in ind_cfg:
        for p in _as_list(ind_cfg["TMA"]):
            p = int(p); out[f"TMA({p})"] = close.rolling(p).mean().rolling(p).mean()

    rsi_spec = ind_cfg.get("RSI")
    if isinstance(rsi_spec, dict) and rsi_spec.get("show", True):
        pr = int(rsi_spec.get("period", 14))
        delta = close.diff()
        up = delta.clip(lower=0).ewm(alpha=1 / pr, adjust=False).mean()
        down = (-delta.clip(upper=0)).ewm(alpha=1 / pr, adjust=False).mean()
        rs = up / down.replace(0, np.nan)
        out["RSI"] = (100 - (100 / (1 + rs))).fillna(method="bfill")

    st = ind_cfg.get("STOCH")
    if isinstance(st, dict) and st.get("show", True):
        k = int(st.get("k", 14)); d = int(st.get("d", 3))
        lowk = df["Low"].rolling(k).min(); highk = df["High"].rolling(k).max()
        kline = (close - lowk) / (highk - lowk).replace(0, np.nan) * 100.0
        dline = kline.rolling(d).mean()
        out["STOCH_K"] = kline; out["STOCH_D"] = dline

    return out

def simple_rule_engine(df: pd.DataFrame, ind: dict, rule_name: str):
    """
    Three built-ins: BASE / TREND / CHOP.
    Returns a list of signals: {"index": ts, "direction": "BUY"/"SELL", "expiry_idx": ts2}
    """
    signals = []
    close = df["Close"]

    def _add(sig_idx, direction, bars):
        all_idx = df.index
        pos = all_idx.get_indexer([sig_idx])[0]
        exp_pos = min(pos + max(1, bars), len(all_idx) - 1)
        signals.append({"index": sig_idx, "direction": direction, "expiry_idx": all_idx[exp_pos]})

    rn = (rule_name or "BASE").upper()

    if rn == "TREND":
        sma50 = ind.get("SMA(50)")
        if sma50 is None:
            return signals
        above = close > sma50
        cross_up = (above & (~above.shift(1).fillna(False)))
        cross_dn = ((~above) & (above.shift(1).fillna(False)))
        for ts in above.index[cross_up.fillna(False)]:
            _add(ts, "BUY", 5)
        for ts in above.index[cross_dn.fillna(False)]:
            _add(ts, "SELL", 5)
        return signals

    if rn == "CHOP":
        rsi = ind.get("RSI")
        if rsi is None:
            return signals
        bounce_up = (rsi.shift(1) < 50) & (rsi >= 50)
        bounce_dn = (rsi.shift(1) > 50) & (rsi <= 50)
        for ts in rsi.index[bounce_up.fillna(False)]:
            _add(ts, "BUY", 3)
        for ts in rsi.index[bounce_dn.fillna(False)]:
            _add(ts, "SELL", 3)
        return signals

    # BASE: Stoch cross
    k = ind.get("STOCH_K"); d = ind.get("STOCH_D")
    if k is not None and d is not None:
        cross_up = (k.shift(1) < d.shift(1)) & (k >= d)
        cross_dn = (k.shift(1) > d.shift(1)) & (k <= d)
        for ts in k.index[cross_up.fillna(False)]:
            _add(ts, "BUY", 5)
        for ts in k.index[cross_dn.fillna(False)]:
            _add(ts, "SELL", 5)
    return signals

def evaluate_signals_outcomes(df: pd.DataFrame, signals: list) -> dict:
    wins = loss = draw = 0
    for s in signals:
        i = s["index"]; e = s["expiry_idx"]
        try:
            c0 = float(df.loc[i, "Close"]); ce = float(df.loc[e, "Close"])
            if s["direction"] == "BUY":
                if ce > c0: wins += 1
                elif ce < c0: loss += 1
                else: draw += 1
            else:
                if ce < c0: wins += 1
                elif ce > c0: loss += 1
                else: draw += 1
        except Exception:
            pass
    total = wins + loss + draw
    return {
        "wins": wins, "losses": loss, "draws": draw,
        "total": total,
        "win_rate": (wins * 100.0 / max(1, wins + loss))
    }

def plot_signals(df: pd.DataFrame, signals: list, indicators: dict, strategy: str, tf: str, expiry: str) -> str:
    """
    Produce a multi-panel screenshot with candles + overlays + RSI/STOCH panels and BUY/SELL markers.
    """
    os.makedirs("static/plots", exist_ok=True)

    if df.empty:
        out = "empty.png"
        plt.figure(figsize=(8, 2))
        plt.text(0.5, 0.5, "No data", ha="center")
        plt.savefig(os.path.join("static", "plots", out))
        plt.close()
        return out

    addplots = []
    # overlay MAs
    for name, val in indicators.items():
        upper = (name or "").upper()
        if upper in ("SMA", "EMA", "WMA", "SMMA", "TMA"):
            periods = val if isinstance(val, (list, tuple)) else [val]
            for p in periods:
                p = int(p)
                if upper == "SMA":
                    ser = df["Close"].rolling(p).mean()
                elif upper == "EMA":
                    ser = df["Close"].ewm(span=p, adjust=False).mean()
                elif upper == "WMA":
                    w = np.arange(1, p + 1)
                    ser = df["Close"].rolling(p).apply(lambda x: (x * w).sum() / w.sum(), raw=True)
                elif upper == "SMMA":
                    ser = df["Close"].ewm(alpha=1 / p, adjust=False).mean()
                else:  # TMA
                    ser = df["Close"].rolling(p).mean().rolling(p).mean()
                addplots.append(mpf.make_addplot(ser, panel=0, width=1))

    # RSI panel
    rsi_spec = indicators.get("RSI")
    if isinstance(rsi_spec, dict) and rsi_spec.get("show", True):
        pr = int(rsi_spec.get("period", 14))
        delta = df["Close"].diff()
        up = delta.clip(lower=0).ewm(alpha=1 / pr, adjust=False).mean()
        down = (-delta.clip(upper=0)).ewm(alpha=1 / pr, adjust=False).mean()
        rs = up / down.replace(0, np.nan)
        rsi = (100 - (100 / (1 + rs))).fillna(method="bfill")
        addplots.append(mpf.make_addplot(rsi, panel=1, ylabel="RSI"))

    # Stoch panel
    st = indicators.get("STOCH")
    if isinstance(st, dict) and st.get("show", True):
        k = int(st.get("k", 14)); d = int(st.get("d", 3))
        lowk = df["Low"].rolling(k).min(); highk = df["High"].rolling(k).max()
        kline = (df["Close"] - lowk) / (highk - lowk).replace(0, np.nan) * 100.0
        dline = kline.rolling(d).mean()
        addplots.append(mpf.make_addplot(kline, panel=2, ylabel="%K"))
        addplots.append(mpf.make_addplot(dline, panel=2))

    fig, axlist = mpf.plot(
        df, type="candle", style="yahoo", addplot=addplots,
        returnfig=True, volume=False, figsize=(13, 8),
        title=f"{(strategy or 'BASE').upper()} • TF={tf} • Exp={expiry}"
    )
    ax = axlist[0]

    # Plot buy/sell markers + expiry markers
    buy_x, buy_y, sell_x, sell_y = [], [], [], []
    for s in signals or []:
        idx = s.get("index")
        if idx not in df.index:  # safety
            continue
        px = float(df.loc[idx, "Close"])
        if s.get("direction") == "BUY":
            buy_x.append(idx); buy_y.append(px)
        else:
            sell_x.append(idx); sell_y.append(px)
        ex = s.get("expiry_idx")
        if ex in df.index:
            ax.axvspan(ex, ex, alpha=0.18)

    if buy_x:
        ax.scatter(buy_x, buy_y, marker="^", s=80)
    if sell_x:
        ax.scatter(sell_x, sell_y, marker="v", s=80)

    out_name = f"{df.index[-1].strftime('%Y%m%d_%H%M%S')}_{(strategy or 'BASE').upper()}_{tf}.png"
    fig.savefig(os.path.join("static", "plots", out_name), dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out_name

def backtest_run(df: pd.DataFrame, strategy: str, indicators: dict, expiry: str):
    """
    Compute indicators, run simple rules, adjust expiry, compute outcomes.
    """
    ind = compute_indicators(df, indicators or {})
    signals = simple_rule_engine(df, ind, (strategy or "BASE").upper())
    bars = EXPIRY_TO_BARS.get((expiry or "5m").lower(), 5)

    fixed = []
    for s in signals:
        i = s["index"]
        pos = df.index.get_indexer([i])[0]
        exp_pos = min(pos + max(1, bars), len(df.index) - 1)
        s["expiry_idx"] = df.index[exp_pos]
        fixed.append(s)

    stats = evaluate_signals_outcomes(df, fixed)
    return fixed, stats
