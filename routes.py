# routes.py — full file (cache-busting images + defaults + entries/expiry markers)
import os, re, json, math, uuid
from io import StringIO
from datetime import datetime, timedelta

import pandas as pd
import requests
from flask import Blueprint, render_template, request, redirect, url_for, session, flash, jsonify

# --- Matplotlib headless plotting ---
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# ---------------------------------------------------------------------
# Blueprint
# ---------------------------------------------------------------------
bp = Blueprint("dashboard", __name__, template_folder="templates", static_folder="static")

# ---------------------------------------------------------------------
# Local modules (one-way imports to avoid cycles)
# ---------------------------------------------------------------------
from utils import exec_sql, get_config, set_config, within_window, TZ, TIMEZONE
from indicators import INDICATOR_SPECS
from strategies import run_backtest_core_binary
from rules import parse_natural_rule

# Standalone fetchers
from data_fetch import deriv_csv_path as _deriv_csv_path, fetch_one_symbol as _fetch_one_symbol

# Live engine (must not import routes)
from live_engine import ENGINE, tg_test

# ---------------------------------------------------------------------
# Symbol sets / helpers
# ---------------------------------------------------------------------
DERIV_FRX = [
    "frxEURUSD","frxGBPUSD","frxUSDJPY","frxUSDCHF","frxUSDCAD","frxAUDUSD","frxNZDUSD",
    "frxEURGBP","frxEURJPY","frxEURCHF","frxEURAUD","frxGBPAUD","frxGBPJPY","frxGBPNZD",
    "frxAUDJPY","frxAUDCAD","frxAUDCHF","frxCADJPY","frxCADCHF","frxCHFJPY","frxNZDJPY",
    "frxEURNZD","frxEURCAD","frxGBPCAD","frxGBPCHF","frxNZDCHF","frxNZDCAD"
]
PO_MAJOR = [
    "EURUSD","GBPUSD","USDJPY","USDCHF","USDCAD","AUDUSD","NZDUSD","EURGBP","EURJPY","GBPJPY"
]
AVAILABLE_GROUPS = [
    {"label": "Deriv (frx*)", "items": DERIV_FRX},
    {"label": "Pocket Option majors", "items": PO_MAJOR},
]

def _cfg_dict(x):
    if isinstance(x, dict): return x
    if isinstance(x, str):
        try:
            j = json.loads(x)
            return j if isinstance(j, dict) else {}
        except Exception:
            return {}
    return {}

def _merge_unique(seq):
    seen, out = set(), []
    for x in seq:
        if not x: continue
        if x not in seen:
            out.append(x); seen.add(x)
    return out

def _is_po_symbol(sym: str) -> bool:
    s = (sym or "").upper()
    return s in PO_MAJOR or bool(re.fullmatch(r"[A-Z]{6}", s))

def _to_deriv(sym: str) -> str:
    if not sym: return sym
    s = sym.strip()
    if s.startswith("frx"): return s
    sU = s.upper().replace("/", "")
    if _is_po_symbol(sU): return "frx" + sU
    return s

def _expand_all_symbols(tokens):
    out = []
    for t in tokens:
        tt = (t or "").strip().upper()
        if tt in ("ALL", "__ALL__", "__ALL_DERIV__", "ALL_DERIV"):
            out.extend(DERIV_FRX)
        elif tt in ("__ALL_PO__", "ALL_PO", "PO_ALL"):
            out.extend(PO_MAJOR)
        else:
            out.append(t)
    seen, unique = set(), []
    for s in out:
        if s and s not in seen:
            unique.append(s); seen.add(s)
    return unique

# ---------------------------------------------------------------------
# Minimal indicator fns (for plotting overlays)
# ---------------------------------------------------------------------
def _sma(series: pd.Series, period: int):
    return series.rolling(period, min_periods=1).mean()

def _ema(series: pd.Series, period: int):
    return series.ewm(span=period, adjust=False).mean()

def _wma(series: pd.Series, period: int):
    weights = pd.Series(range(1, period+1), dtype=float)
    return series.rolling(period).apply(lambda x: (weights.to_numpy()*x).sum()/weights.sum(), raw=True)

def _smma(series: pd.Series, period: int):
    alpha = 1.0/period
    return series.ewm(alpha=alpha, adjust=False).mean()

def _tma(series: pd.Series, period: int):
    p1 = max(1, int(math.ceil(period/2)))
    return _sma(_sma(series, p1), period)

def _rsi(close: pd.Series, period: int = 14):
    delta = close.diff()
    gain = (delta.clip(lower=0)).ewm(alpha=1/period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1/period, adjust=False).mean()
    rs = gain / loss.replace(0, pd.NA)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(method="bfill")

def _stochastic(high: pd.Series, low: pd.Series, close: pd.Series, k_period=14, d_period=3, smooth_k=3):
    lowest_low = low.rolling(k_period).min()
    highest_high = high.rolling(k_period).max()
    k = (close - lowest_low) / (highest_high - lowest_low).replace(0, pd.NA) * 100
    k = k.rolling(smooth_k).mean()
    d = k.rolling(d_period).mean()
    return k, d

def _any_ind_enabled(d):
    d = _cfg_dict(d)
    for k in ("sma","ema","wma","smma","tma","rsi","stoch"):
        if _cfg_dict(d.get(k)).get("enabled"): return True
    return False

# ---- defaults if all indicators disabled ----
def _compute_plot_lines(df: pd.DataFrame, inds_cfg: dict):
    inds_cfg = _cfg_dict(inds_cfg)
    if not _any_ind_enabled(inds_cfg):
        inds_cfg = {
            "sma":   {"enabled": True, "period": 50},
            "rsi":   {"enabled": True, "period": 14},
            "stoch": {"enabled": True, "k": 14, "d": 3, "smooth_k": 3},
        }

    close = df["close"]; high = df["high"]; low = df["low"]
    out = {}

    if inds_cfg.get("sma",{}).get("enabled"):
        p = int(inds_cfg["sma"].get("period", 50))
        out[f"SMA({p})"] = _sma(close, p)
    if inds_cfg.get("ema",{}).get("enabled"):
        p = int(inds_cfg["ema"].get("period", 50))
        out[f"EMA({p})"] = _ema(close, p)
    if inds_cfg.get("wma",{}).get("enabled"):
        p = int(inds_cfg["wma"].get("period", 50))
        out[f"WMA({p})"] = _wma(close, p)
    if inds_cfg.get("smma",{}).get("enabled"):
        p = int(inds_cfg["smma"].get("period", 50))
        out[f"SMMA({p})"] = _smma(close, p)
    if inds_cfg.get("tma",{}).get("enabled"):
        p = int(inds_cfg["tma"].get("period", 50))
        out[f"TMA({p})"] = _tma(close, p)

    if inds_cfg.get("rsi",{}).get("enabled"):
        p = int(inds_cfg["rsi"].get("period", 14))
        out[f"RSI({p})"] = _rsi(close, p)

    if inds_cfg.get("stoch",{}).get("enabled"):
        kp = int(inds_cfg["stoch"].get("k", 14))
        dp = int(inds_cfg["stoch"].get("d", 3))
        sp = int(inds_cfg["stoch"].get("smooth_k", 3))
        k, d = _stochastic(high, low, close, kp, dp, sp)
        out[f"StochK({kp},{dp},{sp})"] = k
        out[f"StochD({kp},{dp},{sp})"] = d

    return out

# ---------------------------------------------------------------------
# Candlestick plotting + entries
# ---------------------------------------------------------------------
def _draw_candles(ax, df, ts_col="timestamp", o="open", h="high", l="low", c="close"):
    ts = pd.to_datetime(df[ts_col])
    opens  = pd.to_numeric(df[o], errors="coerce")
    highs  = pd.to_numeric(df[h], errors="coerce")
    lows   = pd.to_numeric(df[l], errors="coerce")
    closes = pd.to_numeric(df[c], errors="coerce")

    if len(ts) > 1:
        deltas = ts.diff().dropna().dt.total_seconds()
        step = (deltas.median() if not deltas.empty else 60.0)
    else:
        step = 60.0
    width = step * 0.6  # seconds → body width proportion

    x = mdates.date2num(ts.dt.to_pydatetime())

    for xi, o_, h_, l_, c_ in zip(x, opens, highs, lows, closes):
        if pd.isna(o_) or pd.isna(h_) or pd.isna(l_) or pd.isna(c_):
            continue
        up = c_ >= o_
        color = "#17c964" if up else "#f31260"
        # wick
        ax.vlines(xi, l_, h_, color=color, linewidth=1.0, alpha=0.9)
        # body
        halfw = (width / (24*3600)) * 0.45  # seconds → days
        y0, y1 = (o_, c_) if up else (c_, o_)
        ax.add_patch(plt.Rectangle(
            (xi - halfw, y0), 2*halfw, max(y1 - y0, 1e-9),
            facecolor=color, edgecolor=color, linewidth=0.8, alpha=0.95
        ))
    ax.grid(True, alpha=.15)

def _expiry_seconds(expiry: str) -> int:
    s = (expiry or "").lower().strip()
    if s.endswith("m"):
        return int(re.sub(r"[^0-9]", "", s)) * 60
    if s.endswith("h"):
        return int(re.sub(r"[^0-9]", "", s)) * 3600
    if s.endswith("d"):
        return int(re.sub(r"[^0-9]", "", s)) * 86400
    if s.isdigit():
        return int(s) * 60
    return 300

def _extract_trades(bt, df: pd.DataFrame, expiry: str):
    out = []
    exp_s = _expiry_seconds(expiry)

    def to_ts(x):
        if isinstance(x, (int, float)):
            i = int(x)
            if 0 <= i < len(df): return pd.to_datetime(df["timestamp"].iloc[i])
            return None
        try:
            return pd.to_datetime(x)
        except Exception:
            return None

    if hasattr(bt, "trades_df") and isinstance(bt.trades_df, pd.DataFrame):
        tdf = bt.trades_df
        for _, r in tdf.iterrows():
            t = to_ts(r.get("entry_time") or r.get("time") or r.get("t"))
            side = str(r.get("side") or r.get("direction") or "").upper()
            px = r.get("entry_price") or r.get("price") or None
            exp = r.get("expiry_time") or r.get("exit_time")
            exp = to_ts(exp) if exp is not None else None
            if t is not None and side in ("BUY","SELL"):
                if exp is None: exp = t + timedelta(seconds=exp_s)
                out.append({"t": t, "side": side, "expiry": exp, "price": px})
    elif hasattr(bt, "entries") and isinstance(bt.entries, list):
        for e in bt.entries:
            t = to_ts(e.get("time") or e.get("t") or e.get("entry_time"))
            side = str(e.get("side") or e.get("direction") or "").upper()
            px = e.get("price") or e.get("entry_price") or None
            exp = e.get("expiry_time")
            exp = to_ts(exp) if exp is not None else None
            if t is not None and side in ("BUY","SELL"):
                if exp is None: exp = t + timedelta(seconds=exp_s)
                out.append({"t": t, "side": side, "expiry": exp, "price": px})
    elif hasattr(bt, "signals") and isinstance(bt.signals, list):
        for e in bt.signals:
            t = to_ts(e.get("time") or e.get("t"))
            side = str(e.get("side") or e.get("signal") or "").upper()
            px = e.get("price") or None
            if t is not None and side in ("BUY","SELL"):
                out.append({"t": t, "side": side, "expiry": t + timedelta(seconds=exp_s), "price": px})
    elif hasattr(bt, "trades"):
        try:
            for e in bt.trades:
                t = to_ts(getattr(e, "entry_time", None) or getattr(e, "time", None) or (e.get("entry_time") if isinstance(e, dict) else None))
                side = (
                    getattr(e, "side", None) or getattr(e, "direction", None) or
                    (e.get("side") if isinstance(e, dict) else None) or
                    (e.get("direction") if isinstance(e, dict) else None)
                )
                px = getattr(e, "entry_price", None) or (e.get("entry_price") if isinstance(e, dict) else None)
                side = str(side or "").upper()
                if t is not None and side in ("BUY","SELL"):
                    out.append({"t": t, "side": side, "expiry": t + timedelta(seconds=exp_s), "price": px})
        except Exception:
            pass

    return out

def _save_backtest_plot(sym: str, tf: str, expiry: str, df: pd.DataFrame, inds_cfg: dict,
                        trades: list | None,
                        outdir="static/plots", bars=200):
    os.makedirs(outdir, exist_ok=True)

    ds = df.copy()
    if "timestamp" not in ds.columns: raise ValueError("CSV missing 'timestamp' column")
    ds["timestamp"] = pd.to_datetime(ds["timestamp"])
    for col in ("open","high","low","close"):
        if col not in ds.columns: raise ValueError(f"CSV missing '{col}' column")
        ds[col] = pd.to_numeric(ds[col], errors="coerce")
    ds = ds.dropna(subset=["open","high","low","close"]).sort_values("timestamp").tail(bars)

    ts = ds["timestamp"]
    lines = _compute_plot_lines(ds, inds_cfg)
    has_rsi = any(k.startswith("RSI(") for k in lines)
    has_sto = any(k.startswith("Stoch") for k in lines)
    rows    = 1 + (1 if has_rsi else 0) + (1 if has_sto else 0)

    fig_height = 3.2 * rows + 0.6
    fig, axes = plt.subplots(rows, 1, figsize=(13.5, fig_height), sharex=True)
    if rows == 1: axes = [axes]

    # price
    axp = axes[0]
    _draw_candles(axp, ds)
    prefixes = ("SMA(", "EMA(", "WMA(", "SMMA(", "TMA(")
    for name, s in lines.items():
        if name.startswith(prefixes):
            axp.plot(ts, s, label=name, linewidth=1.1)
    axp.set_title(f"{sym}  •  TF={tf}  •  Expiry={expiry}")
    # legend only if there are handles
    if axp.get_legend_handles_labels()[0]:
        axp.legend(loc="upper left", fontsize=8, ncols=3)

    # -- entries / expiry shading
    if trades:
        for tr in trades:
            t  = pd.to_datetime(tr["t"])
            te = pd.to_datetime(tr["expiry"]) if tr.get("expiry") is not None else None
            side = tr.get("side", "BUY").upper()
            try:
                row = ds.iloc[ds["timestamp"].searchsorted(t, side="left")]
                y = float(row["close"])
            except Exception:
                y = float(ds["close"].iloc[-1])

            marker = "▲" if side == "BUY" else "▼"
            color  = "#17c964" if side == "BUY" else "#f31260"
            axp.annotate(marker, (t, y), color=color, fontsize=11, ha="center", va="bottom" if side=="BUY" else "top")

            if te is not None and te > t:
                axp.axvspan(t, te, color=color, alpha=0.10)

    # rsi
    idx = 1
    if has_rsi:
        axr = axes[idx]; idx += 1
        for name, s in lines.items():
            if name.startswith("RSI("):
                axr.plot(ts, s, label=name, linewidth=1.1)
        axr.axhline(50, color="#888888", linewidth=.9)
        axr.axhline(70, color="#aa4444", linewidth=.7)
        axr.axhline(30, color="#44aa44", linewidth=.7)
        axr.set_ylim(0, 100); axr.set_ylabel("RSI")
        axr.grid(True, alpha=.15)
        if axr.get_legend_handles_labels()[0]:
            axr.legend(loc="upper left", fontsize=8)

    # stoch
    if has_sto:
        axs = axes[idx]
        for name, s in lines.items():
            if name.startswith("StochK("):
                axs.plot(ts, s, label=name, linewidth=1.1)
        for name, s in lines.items():
            if name.startswith("StochD("):
                axs.plot(ts, s, label=name, linewidth=1.1, linestyle="--")
        axs.axhline(80, color="#aa4444", linewidth=.7)
        axs.axhline(20, color="#44aa44", linewidth=.7)
        axs.set_ylim(0, 100); axs.set_ylabel("Stoch")
        axs.grid(True, alpha=.15)
        if axs.get_legend_handles_labels()[0]:
            axs.legend(loc="upper left", fontsize=8)

    fig.autofmt_xdate(); fig.tight_layout()

    # ---- CACHE BUSTING (unique filename each run) ----
    stamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    uid = uuid.uuid4().hex[:6]
    fname = f"{sym.replace('/','_')}_{tf}_{expiry}_{stamp}_{uid}.png"
    fpath = os.path.join(outdir, fname)

    plt.savefig(fpath, dpi=140); plt.close(fig)
    return "/" + fpath

# ---------------------------------------------------------------------
# Auth & Health
# ---------------------------------------------------------------------
def require_login(func):
    from functools import wraps
    @wraps(func)
    def wrapper(*args, **kwargs):
        if not session.get("admin"):
            return redirect(url_for("dashboard.login", next=request.path))
        return func(*args, **kwargs)
    return wrapper

@bp.route("/_up", methods=["GET","HEAD"])
def up_check(): return "OK", 200

@bp.route("/login", methods=["GET","POST"])
def login():
    if request.method == "POST":
        if request.form.get("password") == os.getenv("ADMIN_PASSWORD","admin123"):
            session["admin"] = True; flash("Logged in.")
            return redirect(request.args.get("next") or url_for("dashboard.dashboard"))
        flash("Invalid password","error")
    cfg = _cfg_dict(get_config())
    return render_template("dashboard.html", view="login", window=cfg.get("window",{}), tz=TIMEZONE)

@bp.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("dashboard.index"))

@bp.route("/")
def index():
    cfg = _cfg_dict(get_config())
    return render_template("dashboard.html", view="index",
                           within=within_window(cfg),
                           now=datetime.now(TZ).strftime("%Y-%m-%d %H:%M:%S"),
                           tz=TIMEZONE, window=cfg.get("window", {}))

# ---------------------------------------------------------------------
# Dashboard
# ---------------------------------------------------------------------
@bp.route("/dashboard")
@require_login
def dashboard():
    exec_sql("""
      CREATE TABLE IF NOT EXISTS users(
        telegram_id TEXT PRIMARY KEY,
        tier TEXT,
        expires_at TEXT
      )
    """)
    cfg = _cfg_dict(get_config())
    rows = exec_sql("SELECT telegram_id, tier, COALESCE(expires_at,'') FROM users", fetch=True) or []
    users = [{"telegram_id": r[0], "tier": r[1], "expires_at": r[2] or None} for r in rows]

    customs = [
        dict(_cfg_dict(cfg.get("custom1")), _idx=1),
        dict(_cfg_dict(cfg.get("custom2")), _idx=2),
        dict(_cfg_dict(cfg.get("custom3")), _idx=3),
    ]

    strategies_core = _cfg_dict(cfg.get("strategies")) or {
        "BASE":{"enabled": True},
        "TREND":{"enabled": False},
        "CHOP":{"enabled": False},
    }
    strategies_all = dict(strategies_core)
    for i, c in enumerate(customs, start=1):
        strategies_all[f"CUSTOM{i}"] = {"enabled": bool(c.get("enabled"))}

    bt = session.get("bt", {})

    return render_template("dashboard.html", view="dashboard",
                           window=cfg.get("window", {}),
                           strategies_all=strategies_all,
                           strategies=strategies_core,
                           indicators=_cfg_dict(cfg.get("indicators")),
                           specs=INDICATOR_SPECS,
                           customs=customs,
                           active_symbols=cfg.get("symbols") or [],
                           symbols_raw=cfg.get("symbols_raw") or [],
                           available_groups=AVAILABLE_GROUPS,
                           users=users,
                           bt=bt,
                           live_tf=cfg.get("live_tf","M5"),
                           live_expiry=cfg.get("live_expiry","5m"),
                           now=datetime.now(TZ).strftime("%Y-%m-%d %H:%M:%S"),
                           tz=TIMEZONE)

# ---------------------------------------------------------------------
# Config updaters
# ---------------------------------------------------------------------
@bp.route("/update_window", methods=["POST"])
@require_login
def update_window():
    cfg = _cfg_dict(get_config())
    cfg.setdefault("window", {"start":"08:00","end":"17:00","timezone":TIMEZONE})
    cfg["window"]["start"] = request.form.get("start", cfg["window"]["start"])
    cfg["window"]["end"]   = request.form.get("end",   cfg["window"]["end"])
    cfg["window"]["timezone"] = request.form.get("timezone", cfg["window"]["timezone"])
    if request.form.get("live_tf"): cfg["live_tf"] = request.form.get("live_tf").upper()
    if request.form.get("live_expiry"): cfg["live_expiry"] = request.form.get("live_expiry")
    set_config(cfg); flash("Trading window & live defaults updated.")
    return redirect(url_for("dashboard.dashboard"))

@bp.route("/update_strategies", methods=["POST"])
@require_login
def update_strategies():
    cfg = _cfg_dict(get_config())
    cfg.setdefault("strategies", {"BASE":{"enabled":True},"TREND":{"enabled":False},"CHOP":{"enabled":False}})
    for name in list(cfg["strategies"].keys()):
        cfg["strategies"][name]["enabled"] = bool(request.form.get(f"s_{name}"))
    for i in (1,2,3):
        box = bool(request.form.get(f"s_CUSTOM{i}"))
        key = f"custom{i}"
        cfg.setdefault(key, {})
        cfg[key]["enabled"] = box
        cfg["strategies"][f"CUSTOM{i}"] = {"enabled": box}
    set_config(cfg)
    flash("Strategies (including CUSTOM) updated.")
    return redirect(url_for("dashboard.dashboard"))

@bp.route("/update_symbols", methods=["POST"])
@require_login
def update_symbols():
    cfg = _cfg_dict(get_config())
    sel_po     = request.form.getlist("symbols_po_multi")
    sel_deriv  = request.form.getlist("symbols_deriv_multi")
    raw = (request.form.get("symbols_text") or "").strip()
    text_syms = [s.strip() for s in re.split(r"[,\s]+", raw) if s.strip()]
    convert_po = bool(request.form.get("convert_po"))

    merged = _merge_unique(sel_po + sel_deriv + text_syms)
    normalized = [_to_deriv(s) if convert_po else s for s in merged]

    cfg["symbols"] = normalized
    cfg["symbols_raw"] = merged
    set_config(cfg)

    shown = ", ".join(normalized) if normalized else "(none)"
    flash(("Active symbols saved (PO→Deriv conversion ON): " if convert_po else "Active symbols saved: ") + shown)
    return redirect(url_for("dashboard.dashboard"))

@bp.route("/update_indicators", methods=["POST"])
@require_login
def update_indicators():
    cfg = _cfg_dict(get_config())
    inds = _cfg_dict(cfg.get("indicators"))
    for key, spec in INDICATOR_SPECS.items():
        enabled = bool(request.form.get(f"ind_{key}_enabled"))
        inds.setdefault(key, {})
        inds[key]["enabled"] = enabled
        for p in spec.get("params", {}).keys():
            form_key = f"ind_{key}_{p}"
            if form_key in request.form and request.form.get(form_key) != "":
                val = request.form.get(form_key)
                try:
                    inds[key][p] = int(val) if re.fullmatch(r"-?\d+", val) else float(val)
                except Exception:
                    inds[key][p] = val
        if key == "sma" and "period" not in inds[key]:
            inds[key]["period"] = spec["params"]["period"]
    cfg["indicators"] = inds
    set_config(cfg)
    flash("Indicators updated.")
    return redirect(url_for("dashboard.dashboard"))

@bp.route("/update_custom", methods=["POST"])
@require_login
def update_custom():
    slot = (request.form.get("slot") or "1").strip()
    field_prefix = f"custom{slot}"
    cfg = _cfg_dict(get_config())
    cfg.setdefault(field_prefix, {})

    mode = (request.form.get("mode", "SIMPLE") or "SIMPLE").upper()
    cfg[field_prefix]["enabled"] = bool(request.form.get("enabled"))
    cfg[field_prefix]["mode"] = mode

    if mode == "SIMPLE":
        cfg[field_prefix]["simple_buy"]  = request.form.get("simple_buy", "")
        cfg[field_prefix]["simple_sell"] = request.form.get("simple_sell", "")
        cfg[field_prefix]["buy_rule"]  = parse_natural_rule(cfg[field_prefix]["simple_buy"])
        cfg[field_prefix]["sell_rule"] = parse_natural_rule(cfg[field_prefix]["simple_sell"])
    else:
        try:
            cfg[field_prefix]["buy_rule"]  = json.loads(request.form.get("buy_rule_json","{}"))
        except Exception: cfg[field_prefix]["buy_rule"] = {}
        try:
            cfg[field_prefix]["sell_rule"] = json.loads(request.form.get("sell_rule_json","{}"))
        except Exception: cfg[field_prefix]["sell_rule"] = {}

    try: cfg[field_prefix]["tol_pct"] = float(request.form.get("tol_pct", cfg[field_prefix].get("tol_pct", 0.1)))
    except Exception: pass
    try: cfg[field_prefix]["lookback"] = int(request.form.get("lookback", cfg[field_prefix].get("lookback", 3)))
    except Exception: pass

    cfg.setdefault("strategies", {})
    cfg["strategies"][f"CUSTOM{slot}"] = {"enabled": bool(cfg[field_prefix]["enabled"])}

    set_config(cfg)
    flash(f"Custom #{slot} saved.")
    return redirect(url_for("dashboard.dashboard"))

# ---------------------------------------------------------------------
# Users
# ---------------------------------------------------------------------
@bp.route("/users/add", methods=["POST"])
@require_login
def users_add():
    telegram_id = (request.form.get("telegram_id") or "").strip()
    tier = (request.form.get("tier") or "free").strip()
    expires_at = (request.form.get("expires_at") or "").strip() or None
    if not telegram_id:
        flash("Telegram ID required.", "error")
        return redirect(url_for("dashboard.dashboard"))
    exec_sql("""
      CREATE TABLE IF NOT EXISTS users(
        telegram_id TEXT PRIMARY KEY,
        tier TEXT,
        expires_at TEXT
      )
    """)
    exec_sql("""
      INSERT INTO users(telegram_id, tier, expires_at)
      VALUES(?,?,?)
      ON CONFLICT(telegram_id) DO UPDATE SET
        tier=excluded.tier,
        expires_at=excluded.expires_at
    """, (telegram_id, tier, expires_at))
    flash(f"User saved: {telegram_id} ({tier})")
    return redirect(url_for("dashboard.dashboard"))

@bp.route("/users/delete", methods=["POST"])
@require_login
def users_delete():
    telegram_id = (request.form.get("telegram_id") or "").strip()
    if not telegram_id:
        flash("Telegram ID required.", "error")
        return redirect(url_for("dashboard.dashboard"))
    exec_sql("DELETE FROM users WHERE telegram_id = ?", (telegram_id,))
    flash(f"User deleted: {telegram_id}")
    return redirect(url_for("dashboard.dashboard"))

# ---------------------------------------------------------------------
# Deriv fetch
# ---------------------------------------------------------------------
@bp.route("/deriv_fetch", methods=["POST"])
@require_login
def deriv_fetch():
    app_id = os.getenv("DERV_APP_ID", None) or os.getenv("DERIV_APP_ID", "1089")
    symbols = (request.form.get("fetch_symbols") or "").strip()
    tf = (request.form.get("fetch_tf") or "M5").upper()
    count = int(request.form.get("fetch_count") or "300")
    convert_po = bool(request.form.get("convert_po_fetch"))

    gran_map = {"M1":60,"M2":120,"M3":180,"M5":300,"M10":600,"M15":900,"M30":1800,"H1":3600,"H4":14400,"D1":86400}
    gran = gran_map.get(tf, 300)

    pairs_in = [s.strip() for s in re.split(r"[,\s]+", symbols) if s.strip()]
    pairs = [_to_deriv(s) if convert_po else s for s in pairs_in]

    ok = 0; fail = []
    for sym in pairs:
        try:
            _fetch_one_symbol(app_id, sym, gran, count)
            ok += 1
        except Exception as e:
            fail.append(f"{sym}: {e}")

    if ok: flash(f"Deriv: saved {ok} symbol(s) @ {tf}")
    if fail: flash("Errors: " + "; ".join(fail))
    return redirect(url_for("dashboard.dashboard"))

# ---------------------------------------------------------------------
# Backtest (with candlestick + entries plot)
# ---------------------------------------------------------------------
@bp.route("/backtest", methods=["POST"])
@require_login
def backtest():
    cfg = _cfg_dict(get_config())
    tf = (request.form.get("bt_tf") or "M5").upper()
    expiry = request.form.get("bt_expiry") or "5m"
    strategy = (request.form.get("bt_strategy") or "BASE").upper()
    use_server = bool(request.form.get("use_server"))
    app_id = os.getenv("DERV_APP_ID", None) or os.getenv("DERIV_APP_ID", "1089")
    count = int(request.form.get("bt_count") or "300")
    convert_po = bool(request.form.get("convert_po_bt"))

    gran_map = {"M1":60,"M2":120,"M3":180,"M5":300,"M10":600,"M15":900,"M30":1800,"H1":3600,"H4":14400,"D1":86400}
    gran = gran_map.get(tf, 300)

    raw_syms_text = request.form.get("bt_symbols") or " ".join(cfg.get("symbols") or [])
    text_syms = [s.strip() for s in re.split(r"[,\s]+", raw_syms_text) if s.strip()]
    from_multi = request.form.getlist("bt_symbols_multi")
    chosen = from_multi if from_multi else text_syms
    chosen = _expand_all_symbols(chosen)
    symbols = [_to_deriv(s) if convert_po else s for s in chosen]

    if not symbols:
        session["bt"] = {"error": "No symbols selected. Choose pairs or use ALL/ALL_DERIV/ALL_PO."}
        flash("Backtest error: no symbols selected.")
        return redirect(url_for("dashboard.dashboard"))

    uploaded = request.files.get("bt_csv")
    results = []
    summary = {"trades":0,"wins":0,"losses":0,"draws":0,"winrate":0.0}
    warnings = []
    plot_url = None

    def run_one(sym, df):
        nonlocal summary, results, plot_url
        if strategy in ("CUSTOM1","CUSTOM2","CUSTOM3"):
            sid = strategy[-1]
            cfg_run = dict(_cfg_dict(cfg))
            cfg_run["custom"] = _cfg_dict(cfg.get(f"custom{sid}"))
            core = "CUSTOM"
        else:
            cfg_run = dict(_cfg_dict(cfg))
            core = strategy

        try:
            bt = run_backtest_core_binary(df, core, cfg_run, tf, expiry)
        except Exception as e:
            if "object has no attribute 'get'" in str(e):
                bt = run_backtest_core_binary(df, core, {}, tf, expiry)
            else:
                raise

        results.append({
            "symbol": sym,
            "trades": getattr(bt, "trades", 0) if isinstance(getattr(bt, "trades", 0), (int, float)) else (getattr(bt, "trades_count", 0) or 0),
            "wins": getattr(bt, "wins", 0),
            "losses": getattr(bt, "losses", 0),
            "draws": getattr(bt, "draws", 0),
            "winrate": round((getattr(bt, "winrate", 0.0) or 0.0)*100,2) if isinstance(getattr(bt, "winrate", 0.0), (float,int)) else 0.0,
        })
        summary["trades"] += results[-1]["trades"]
        summary["wins"]   += results[-1]["wins"]
        summary["losses"] += results[-1]["losses"]
        summary["draws"]  += results[-1]["draws"]

        # defaults if indicators all off
        inds_cfg = _cfg_dict(cfg.get("indicators") or {})
        if not _any_ind_enabled(inds_cfg):
            inds_cfg = {"sma":{"enabled":True,"period":50},
                        "rsi":{"enabled":True,"period":14},
                        "stoch":{"enabled":True,"k":14,"d":3,"smooth_k":3}}

        trades = _extract_trades(bt, df, expiry)
        try:
            plot_url_local = _save_backtest_plot(sym, tf, expiry, df, inds_cfg, trades, outdir="static/plots", bars=200)
            if plot_url is None:
                plot_url = plot_url_local
        except Exception as pe:
            warnings.append(f"{sym}: plot error {pe}")

    try:
        if uploaded and uploaded.filename:
            data = uploaded.read().decode("utf-8", errors="ignore")
            df = pd.read_csv(StringIO(data))
            df.columns = [c.strip().lower() for c in df.columns]
            if "timestamp" in df.columns: df["timestamp"] = pd.to_datetime(df["timestamp"])
            for c in ("open","high","low","close"):
                df[c] = pd.to_numeric(df[c], errors="coerce")
            df = df.dropna(subset=["close"]).sort_values("timestamp").reset_index(drop=True)
            run_one(symbols[0] if symbols else "CSV", df)
        else:
            for sym in symbols:
                try:
                    if use_server:
                        _fetch_one_symbol(app_id, sym, gran, count)
                    path = _deriv_csv_path(sym, gran)
                    if not os.path.exists(path):
                        warnings.append(f"{sym}: no server CSV @ {tf} (path: {path})")
                        continue
                    df = pd.read_csv(path)
                    df.columns = [c.strip().lower() for c in df.columns]
                    if "timestamp" in df.columns: df["timestamp"] = pd.to_datetime(df["timestamp"])
                    for c in ("open","high","low","close"):
                        df[c] = pd.to_numeric(df[c], errors="coerce")
                    df = df.dropna(subset=["close"]).sort_values("timestamp").reset_index(drop=True)
                    if len(df) < 30:
                        warnings.append(f"{sym}: not enough candles ({len(df)})")
                        continue
                    run_one(sym, df)
                except Exception as e:
                    warnings.append(f"{sym}: {e}")

        summary["winrate"] = round((summary["wins"]/summary["trades"])*100,2) if summary["trades"] else 0.0
        payload = {"summary": summary, "results": results, "tf": tf, "expiry": expiry, "strategy": strategy}
        if plot_url: payload["plot_url"] = plot_url
        if warnings:
            payload["warnings"] = warnings
            flash("Backtest completed with warnings.")
        else:
            flash("Backtest complete.")
        session["bt"] = payload

    except Exception as e:
        session["bt"] = {"error": str(e), "warnings": warnings}
        flash(f"Backtest error: {e}", "error")

    return redirect(url_for("dashboard.dashboard"))

# ---------------------------------------------------------------------
# Live engine endpoints
# ---------------------------------------------------------------------
@bp.route("/live/status")
def live_status():
    return jsonify({"ok": True, "status": ENGINE.status()})

@bp.route("/live/start", methods=["POST","GET"])
def live_start():
    ok, msg = ENGINE.start()
    if request.method == "GET": return jsonify({"ok": ok, "msg": msg})
    flash(f"Live: {msg}")
    return redirect(url_for("dashboard.dashboard"))

@bp.route("/live/stop", methods=["POST","GET"])
def live_stop():
    ok, msg = ENGINE.stop()
    if request.method == "GET": return jsonify({"ok": ok, "msg": msg})
    flash(f"Live: {msg}")
    return redirect(url_for("dashboard.dashboard"))

@bp.route("/live/debug/<state>")
def live_debug(state):
    s = (state or "").lower()
    ENGINE.debug = (s == "on")
    return jsonify({"ok": True, "debug": ENGINE.debug})

# ---------------------------------------------------------------------
# Telegram diagnostics
# ---------------------------------------------------------------------
@bp.route("/telegram/test", methods=["POST","GET"])
def telegram_test():
    ok, msg = tg_test()
    if request.method == "GET":
        return jsonify({"ok": ok, "msg": msg})
    flash("Telegram OK" if ok else f"Telegram error: {msg}")
    return redirect(url_for("dashboard.dashboard"))

@bp.route("/telegram/diag")
def telegram_diag():
    from live_engine import _send_telegram, TELEGRAM_CHAT_KEYS
    token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    if not token:
        return jsonify({"ok": False, "error": "Missing TELEGRAM_BOT_TOKEN"}), 200
    try:
        r = requests.get(f"https://api.telegram.org/bot{token}/getMe", timeout=8)
        getme = r.json()
    except Exception as e:
        getme = {"ok": False, "error": str(e)}
    configured = {k: os.getenv(k, "").strip() for k in TELEGRAM_CHAT_KEYS if os.getenv(k, "").strip()}
    ok, info = _send_telegram("🧪 Telegram DIAG: test message from Pocket Option Signals.")
    masked = token[:9] + "..." + token[-6:] if len(token) > 18 else "***"
    return jsonify({"ok": ok, "token_masked": masked, "getMe": getme, "configured_chats": configured, "send_result": info})
