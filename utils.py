# utils.py — config store, TZ, sqlite, Deriv fetch (REST + WebSocket fallback),
# symbols, backtest helpers, plotting (clear markers + shaded expiry)
import os, json, sqlite3
from datetime import datetime, time as dtime, timezone
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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

# -----------------------------------------------------------------------------
# App / DB
# -----------------------------------------------------------------------------
DB_PATH = os.getenv("SQLITE_PATH", "members.db")
DERIV_APP_ID = os.getenv("DERIV_APP_ID", "99185")

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
# Safe coercion helpers (avoid int(dict) crashes)
# -----------------------------------------------------------------------------
def safe_int(x, default=0):
    """Coerce possibly dict/str/float to int; supports {'sec':60}, {'granularity':300}, etc."""
    if x is None:
        return int(default)
    if isinstance(x, (int, np.integer)):
        return int(x)
    if isinstance(x, (float, np.floating)):
        return int(x)
    if isinstance(x, str):
        try:
            return int(float(x.strip()))
        except Exception:
            return int(default)
    if isinstance(x, dict):
        for k in ("sec","secs","seconds","granularity","value","v"):
            if k in x:
                return safe_int(x[k], default)
        for v in x.values():
            try:
                return safe_int(v, default)
            except Exception:
                pass
        return int(default)
    try:
        return int(x)
    except Exception:
        return int(default)

def safe_app_id(s):
    if not s:
        return "99185"
    if isinstance(s, (int, np.integer)):
        return str(int(s))
    if isinstance(s, str):
        digits = "".join(ch for ch in s if ch.isdigit())
        return digits or "99185"
    return "99185"

DERIV_APP_ID = safe_app_id(DERIV_APP_ID)

# -----------------------------------------------------------------------------
# Trading window
# -----------------------------------------------------------------------------
def within_window(cfg: dict) -> bool:
    w = cfg.get("window") or {}
    start = w.get("start") or cfg.get("window_start","08:00")
    end   = w.get("end")   or cfg.get("window_end","17:00")
    now_tt = datetime.now(TZ).time()
    def hhmm(v, fallback):
        try:
            h,m = [int(x) for x in (v or fallback).split(":")[:2]]
            return h,m
        except Exception:
            return (8,0) if fallback=="08:00" else (17,0)
    s_h,s_m = hhmm(start,"08:00"); e_h,e_m = hhmm(end,"17:00")
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
# Deriv REST + WebSocket fallback (Deriv-only)
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
    if epoch is None and len(df.columns)==1 and df.shape[0] and isinstance(df.iloc[0,0], dict):
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

def _fetch_deriv_rest(symbol: str, granularity_sec, days, attempts_log: list) -> pd.DataFrame | None:
    import requests
    sym = (symbol or "").strip()
    if not sym.lower().startswith("frx"):
        sym = PO2DERIV.get(sym.upper(), sym)
    if not sym.lower().startswith("frx"):
        return None
    sym_lc = "frx" + sym[-6:].upper()
    sym_uc = sym_lc.upper()
    candidates = [sym_lc, sym_uc] if sym_lc != sym_uc else [sym_lc]

    gran = safe_int(granularity_sec, 300)
    allowed = [60, 120, 180, 300, 600, 900, 1800, 3600, 14400, 86400]
    gran = min(allowed, key=lambda g: abs(g - gran))
    d = safe_int(days, 5)
    approx = min(2000, max(300, int(d * 86400 // gran)))

    def try_request(url, params):
        params = dict(params or {})
        if DERIV_APP_ID: params["app_id"] = DERIV_APP_ID
        r = requests.get(url, params=params, timeout=15)
        attempts_log.append((url, r.status_code))
        if not r.ok: return None
        try:
            return _normalize_ohlc_frame(r.json())
        except Exception:
            return None

    for s in candidates:
        for base in ("https://api.deriv.com/api/v4/price_history",
                     "https://api.deriv.com/api/v3/price_history"):
            df = try_request(base, {"symbol": s, "granularity": gran, "count": approx})
            if df is not None and not df.empty: return df

    for s in candidates:
        df = try_request("https://api.deriv.com/api/explore/candles",
                         {"symbol": s, "granularity": gran, "count": approx})
        if df is not None and not df.empty: return df
        end = int(datetime.now(timezone.utc).timestamp())
        start = end - d*86400
        df = try_request("https://api.deriv.com/api/explore/candles",
                         {"symbol": s, "granularity": gran, "start": start, "end": end})
        if df is not None and not df.empty: return df
    return None

def _fetch_deriv_ws(symbol: str, granularity_sec, days, attempts_log: list) -> pd.DataFrame | None:
    try:
        from websocket import create_connection
    except Exception as e:
        attempts_log.append(("websocket-client-missing", str(e)))
        return None

    sym = (symbol or "").strip()
    if not sym.lower().startswith("frx"):
        sym = PO2DERIV.get(sym.upper(), sym)
    if not sym.lower().startswith("frx"):
        attempts_log.append(("ws_symbol_invalid", symbol))
        return None
    sym = ("FRX" + sym[-6:].upper())

    gran = safe_int(granularity_sec, 300)
    allowed = [60, 120, 180, 300, 600, 900, 1800, 3600, 14400, 86400]
    gran = min(allowed, key=lambda g: abs(g - gran))
    d = safe_int(days, 5)
    count = min(2000, max(300, int(d * 86400 // gran)))

    appid = safe_app_id(DERIV_APP_ID)
    ws_url = f"wss://ws.derivws.com/websockets/v3?app_id={appid}"

    req = {
        "ticks_history": sym,
        "adjust_start_time": 1,
        "count": int(count),
        "end": "latest",
        "granularity": int(gran),
        "style": "candles"
    }

    try:
        ws = create_connection(ws_url, timeout=15)
        ws.send(json.dumps(req))
        raw = ws.recv()
        ws.close()
    except Exception as e:
        attempts_log.append(("ws_connect_error", str(e)))
        return None

    try:
        js = json.loads(raw)
        if js.get("error"):
            attempts_log.append(("ws_error", js["error"]))
            return None
        df = _normalize_ohlc_frame(js)
        return df
    except Exception as e:
        attempts_log.append(("ws_parse_error", str(e)))
        return None

def fetch_deriv_history(symbol: str, granularity_sec, days=5) -> pd.DataFrame:
    """Deriv-only fetch: REST first, then WebSocket fallback."""
    attempts = []
    df = _fetch_deriv_rest(symbol, granularity_sec, days, attempts)
    if df is None or df.empty:
        df = _fetch_deriv_ws(symbol, granularity_sec, days, attempts)
    if df is None or df.empty:
        raise RuntimeError(
            f"Deriv fetch failed for symbol={symbol}, tf={safe_int(granularity_sec,300)}. "
            f"Attempts: {attempts}. Tip: set DERIV_APP_ID and ensure pair exists (e.g. frxEURUSD). "
            f"You can also uncheck 'Use Deriv server fetch' and upload a CSV."
        )
    return df

# -----------------------------------------------------------------------------
# Backtest helpers
# -----------------------------------------------------------------------------
EXPIRY_TO_BARS = {"1m":1,"3m":3,"5m":5,"10m":10,"30m":30,"1h":60,"4h":240}

def compute_indicators(df: pd.DataFrame, ind_cfg: dict) -> dict:
    """Robust indicator builder (numbers/strings/lists/tuples/dicts)."""
    out = {}
    close = df["Close"]

    def _to_period_list(v):
        if v is None:
            return []
        if isinstance(v, (int, float, np.integer, np.floating, str)):
            try:
                return [int(float(str(v)))]
            except Exception:
                return []
        if isinstance(v, (list, tuple)):
            outv = []
            for x in v:
                try:
                    outv.append(int(float(str(x))))
                except Exception:
                    pass
            return outv
        if isinstance(v, dict):
            outv = []
            for x in v.values():
                try:
                    outv.append(int(float(str(x))))
                except Exception:
                    pass
            return outv
        try:
            return [int(v)]
        except Exception:
            return []

    # MAs
    if "SMA" in ind_cfg:
        for p in _to_period_list(ind_cfg["SMA"]):
            if p > 0:
                out[f"SMA({p})"] = close.rolling(p).mean()
    if "EMA" in ind_cfg:
        for p in _to_period_list(ind_cfg["EMA"]):
            if p > 0:
                out[f"EMA({p})"] = close.ewm(span=p, adjust=False).mean()
    if "WMA" in ind_cfg:
        for p in _to_period_list(ind_cfg["WMA"]):
            if p > 0:
                w = np.arange(1, p + 1)
                out[f"WMA({p})"] = close.rolling(p).apply(
                    lambda x: (x * w).sum() / w.sum(), raw=True
                )
    if "SMMA" in ind_cfg:
        for p in _to_period_list(ind_cfg["SMMA"]):
            if p > 0:
                out[f"SMMA({p})"] = close.ewm(alpha=1 / p, adjust=False).mean()
    if "TMA" in ind_cfg:
        for p in _to_period_list(ind_cfg["TMA"]):
            if p > 0:
                out[f"TMA({p})"] = close.rolling(p).mean().rolling(p).mean()

    # Oscillators
    if "RSI" in ind_cfg and isinstance(ind_cfg["RSI"], dict) and ind_cfg["RSI"].get("show", True):
        pr = 14
        try:
            pr_list = _to_period_list(ind_cfg["RSI"].get("period", 14))
            pr = max(1, pr_list[0]) if pr_list else 14
        except Exception:
            pr = 14
        delta = close.diff()
        up = delta.clip(lower=0).ewm(alpha=1 / pr, adjust=False).mean()
        down = (-delta.clip(upper=0)).ewm(alpha=1 / pr, adjust=False).mean()
        rs = up / down.replace(0, np.nan)
        out["RSI"] = (100 - (100 / (1 + rs))).fillna(method="bfill")

    if "STOCH" in ind_cfg and isinstance(ind_cfg["STOCH"], dict) and ind_cfg["STOCH"].get("show", True):
        k_list = _to_period_list(ind_cfg["STOCH"].get("k", 14)) or [14]
        d_list = _to_period_list(ind_cfg["STOCH"].get("d", 3)) or [3]
        k = max(1, k_list[0]); d = max(1, d_list[0])
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

    rule = (rule_name or "").upper()

    if rule == "TREND":
        sma50 = ind.get("SMA(50)")
        if sma50 is None: return signals
        above = close > sma50
        cross_up = (above & (~above.shift(1).fillna(False)))
        cross_dn = ((~above) & (above.shift(1).fillna(False)))
        for ts, _ in cross_up[cross_up].items(): add(ts, "BUY", 5)
        for ts, _ in cross_dn[cross_dn].items(): add(ts, "SELL", 5)
        return signals

    if rule == "CHOP":
        rsi = ind.get("RSI")
        if rsi is None: return signals
        bounce_up = (rsi.shift(1) < 50) & (rsi >= 50)
        bounce_dn = (rsi.shift(1) > 50) & (rsi <= 50)
        for ts in rsi.index[bounce_up.fillna(False)]: add(ts, "BUY", 3)
        for ts in rsi.index[bounce_dn.fillna(False)]: add(ts, "SELL", 3)
        return signals

    # BASE / CUSTOM: Stochastic cross
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
# Plotting — clear, high-contrast (last 180 candles, markers + shaded expiry)
# -----------------------------------------------------------------------------
def _limit_df(df: pd.DataFrame, last_n: int = 180) -> pd.DataFrame:
    if df is None or df.empty: return df
    if len(df) <= last_n: return df
    return df.iloc[-last_n:].copy()

def plot_signals(df, signals, indicators, strategy, tf, expiry) -> str:
    """
    Clean plot:
      • last 180 candles
      • BUY(▲)/SELL(▼) markers (large, white edge)
      • shaded entry→expiry region
      • RSI/Stochastic lower panel (only if enabled)
    """
    import matplotlib.dates as mdates
    os.makedirs("static/plots", exist_ok=True)

    if df is None or df.empty:
        out = "empty.png"
        plt.figure(figsize=(8,2), dpi=160)
        plt.text(0.5,0.5,"No data", ha="center")
        plt.savefig(os.path.join("static","plots",out)); plt.close()
        return out

    df = _limit_df(df, 180)
    ind = compute_indicators(df, indicators or {})

    out_name = f"{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{strategy}_{tf}.png"
    path = os.path.join("static","plots", out_name)

    # Prefer mplfinance
    if HAVE_MPLFIN:
        addplots = []
        for name, series in ind.items():
            if name.startswith(("SMA(","EMA(","WMA(","SMMA(","TMA(")):
                addplots.append(mpf.make_addplot(series, panel=0, width=1.15, alpha=0.95))

        use_osc = ("RSI" in ind) or ("STOCH_K" in ind and "STOCH_D" in ind)
        if "RSI" in ind:
            addplots.append(mpf.make_addplot(ind["RSI"], panel=1, ylim=(0,100),
                                             width=1.1, secondary_y=False, color="#60a5fa"))
        if "STOCH_K" in ind and "STOCH_D" in ind:
            addplots.append(mpf.make_addplot(ind["STOCH_K"], panel=1, width=1.0, color="#22c55e"))
            addplots.append(mpf.make_addplot(ind["STOCH_D"], panel=1, width=1.0, color="#f59e0b"))

        style = mpf.make_mpf_style(
            base_mpf_style="yahoo",
            facecolor="#0b0f17",
            gridstyle="-",
            gridcolor="#192132",
            marketcolors=mpf.make_marketcolors(up="#16a34a", down="#ef4444", wick="inherit", edge="inherit")
        )

        fig, axlist = mpf.plot(
            df, type="candle", style=style, addplot=addplots, returnfig=True,
            volume=False, figsize=(15, 9 if use_osc else 7),
            panel_ratios=(3,1) if use_osc else None,
            tight_layout=True, title=f"{strategy} • TF={tf} • Exp={expiry}"
        )
        ax_price = axlist[0]

        if signals:
            buy_x, buy_y, sell_x, sell_y = [], [], [], []
            for s in signals:
                idx = s["index"]
                if idx not in df.index: continue
                y = float(df.loc[idx,"Close"])
                if s["direction"] == "BUY":
                    buy_x.append(idx); buy_y.append(y)
                else:
                    sell_x.append(idx); sell_y.append(y)
                ex = s.get("expiry_idx")
                if ex in df.index:
                    ax_price.axvspan(idx, ex, alpha=.15,
                                     color="#22c55e" if s["direction"]=="BUY" else "#ef4444")
            if buy_x:
                ax_price.scatter(buy_x, buy_y, marker="^", s=120, color="#22c55e",
                                 edgecolors="white", linewidths=0.6, label="BUY")
            if sell_x:
                ax_price.scatter(sell_x, sell_y, marker="v", s=120, color="#ef4444",
                                 edgecolors="white", linewidths=0.6, label="SELL")

        ax_price.grid(True, alpha=.25)
        ax_price.legend(loc="upper left", frameon=False)
        fig.savefig(path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        return out_name

    # Fallback matplotlib
    fig, ax = plt.subplots(1,1,figsize=(15,7), dpi=200)
    import matplotlib.dates as mdates
    ts = mdates.date2num(df.index.to_pydatetime())
    for t,(o,h,l,c) in enumerate(zip(df["Open"], df["High"], df["Low"], df["Close"])):
        x = ts[t]
        color = "#16a34a" if c>=o else "#ef4444"
        ax.vlines(x, l, h, color=color, linewidth=1.0, alpha=.95)
        ax.add_patch(plt.Rectangle((x-0.0022, min(o,c)), 0.0044, max(abs(c-o),1e-6),
                                   facecolor=color, edgecolor=color, linewidth=.9, alpha=.95))
    for name, series in ind.items():
        if name.startswith(("SMA(","EMA(","WMA(","SMMA(","TMA(")):
            ax.plot(series.index, series.values, linewidth=1.25, alpha=.95)
    if signals:
        for s in signals:
            idx = s["index"]
            if idx in df.index:
                y = float(df.loc[idx,"Close"])
                ax.scatter([mdates.date2num(idx)], [y],
                           marker="^" if s["direction"]=="BUY" else "v",
                           s=140, color="#22c55e" if s["direction"]=="BUY" else "#ef4444",
                           edgecolors="white", linewidths=.6, zorder=5)
                ex = s.get("expiry_idx")
                if ex in df.index:
                    ax.axvspan(mdates.date2num(idx), mdates.date2num(ex),
                               alpha=.15, color="#22c55e" if s["direction"]=="BUY" else "#ef4444")
    ax.set_title(f"{strategy} • TF={tf} • Exp={expiry}")
    ax.set_ylabel("Price"); ax.grid(alpha=.25)
    fig.autofmt_xdate(); fig.tight_layout(); fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return out_name

# -----------------------------------------------------------------------------
# Strategy requirements → always generate entries/exits for chosen strategy
# -----------------------------------------------------------------------------
def ensure_strategy_requirements(ind_cfg: dict, strategy: str) -> dict:
    """
    Ensure the minimal indicators required by a strategy are present
    for the current run (does not mutate persisted config).
    """
    cfg = {k: (v.copy() if isinstance(v, dict) else v) for k, v in (ind_cfg or {}).items()}
    s = (strategy or "").upper()

    if s in ("BASE", "CUSTOM1", "CUSTOM2", "CUSTOM3"):
        st = cfg.get("STOCH")
        if not isinstance(st, dict):
            st = {"show": True, "k": 14, "d": 3}
        else:
            st.setdefault("show", True)
            st.setdefault("k", 14)
            st.setdefault("d", 3)
        cfg["STOCH"] = st

    if s == "TREND":
        sma = cfg.get("SMA")
        if sma is None:
            cfg["SMA"] = [50]
        elif isinstance(sma, (int, float, str)):
            p = int(float(str(sma)))
            cfg["SMA"] = sorted({p, 50})
        elif isinstance(sma, (list, tuple, set)):
            cfg["SMA"] = sorted({int(float(str(x))) for x in sma} | {50})
        elif isinstance(sma, dict):
            vals = {50}
            for v in sma.values():
                try: vals.add(int(float(str(v))))
                except: pass
            cfg["SMA"] = sorted(vals)

    if s == "CHOP":
        rsi = cfg.get("RSI")
        if not isinstance(rsi, dict):
            rsi = {"show": True, "period": 14}
        else:
            rsi.setdefault("show", True)
            rsi.setdefault("period", 14)
        cfg["RSI"] = rsi

    return cfg

# -----------------------------------------------------------------------------
# Backtest run (uses ensure_strategy_requirements)
# -----------------------------------------------------------------------------
def backtest_run(df: pd.DataFrame, strategy: str, indicators: dict, expiry: str):
    ind_cfg = ensure_strategy_requirements(indicators or {}, strategy)
    ind = compute_indicators(df, ind_cfg)
    signals = simple_rule_engine(df, ind, (strategy or "").upper())

    bars_map = {"1m":1,"3m":3,"5m":5,"10m":10,"30m":30,"1h":60,"4h":240}
    bars = bars_map.get((expiry or "5m").lower(), 5)

    fixed = []
    for s in signals:
        i = s["index"]
        pos = df.index.get_indexer([i])[0]
        exp_pos = min(pos + max(1,bars), len(df.index)-1)
        s["expiry_idx"] = df.index[exp_pos]
        fixed.append(s)

    stats = evaluate_signals_outcomes(df, fixed)
    return fixed, stats
