# routes.py — FULL BUILD (fixed + upgraded)
# Features:
# - Dashboard: window (08:00–17:00 TT), TF/expiry, strategies (BASE/TREND/CHOP + CUSTOM1..3),
#   indicators (full catalog from indicators.py), only-enabled get plotted,
#   multi-symbol selectors (PO majors + Deriv frx*) + free text + PO→Deriv auto-convert.
# - Backtest: multi-symbol, server CSV or uploaded CSV, candles + overlays/panels,
#   BUY/SELL markers + shaded expiry, JSON/CSV export, plot served with no-cache.
# - Live engine: start/stop/debug/status, tally endpoint.
# - Telegram: test & diag (reads envs for 4 tiers).
# - Users: add/delete (tier, expiry).
# - Deriv fetch: optional (data_fetch.py); if missing, message shown.
# - UptimeRobot HEAD /_up.

import os
import re
import csv
import json
import math
import uuid
from io import StringIO
from datetime import datetime, timedelta

import pandas as pd
import numpy as np
import requests

from flask import (
    Blueprint, render_template, request, redirect, url_for, session, flash, jsonify,
    send_from_directory, make_response, Response
)

# Matplotlib headless
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

bp = Blueprint("dashboard", __name__, template_folder="templates", static_folder="static")

# --- Project modules ----------------------------------------------------------
from utils import exec_sql, get_config, set_config, within_window, TZ
TIMEZONE = getattr(TZ, "zone", os.getenv("TIMEZONE", "UTC"))

from indicators import INDICATOR_SPECS, apply_indicators
from strategies import run_backtest_core_binary
from rules import parse_natural_rule
from live_engine import ENGINE, tg_test

# Optional data fetch helpers (safe import)
try:
    from data_fetch import deriv_csv_path as _deriv_csv_path, fetch_one_symbol as _fetch_one_symbol
except Exception:
    _deriv_csv_path = None
    _fetch_one_symbol = None

# Ensure plot dir exists
os.makedirs("static/plots", exist_ok=True)

# ===== Symbol groups ==========================================================
DERIV_FRX = [
    "frxEURUSD","frxGBPUSD","frxUSDJPY","frxUSDCHF","frxUSDCAD","frxAUDUSD","frxNZDUSD",
    "frxEURGBP","frxEURJPY","frxEURCHF","frxEURAUD","frxGBPAUD","frxGBPJPY","frxGBPNZD",
    "frxAUDJPY","frxAUDCAD","frxAUDCHF","frxCADJPY","frxCADCHF","frxCHFJPY","frxNZDJPY",
    "frxEURNZD","frxEURCAD","frxGBPCAD","frxGBPCHF","frxNZDCHF","frxNZDCAD"
]
PO_MAJOR = [
    "EURUSD","GBPUSD","USDJPY","USDCHF","USDCAD","AUDUSD","NZDUSD",
    "EURGBP","EURJPY","GBPJPY","EURAUD","AUDJPY","CADJPY","CHFJPY"
]
AVAILABLE_GROUPS = [
    {"label": "Deriv (frx*)", "items": DERIV_FRX},
    {"label": "Pocket Option majors", "items": PO_MAJOR},
]

# ===== Small helpers ==========================================================
def _cfg_dict(x):
    if isinstance(x, dict):
        return x
    if isinstance(x, str):
        try:
            j = json.loads(x)
            return j if isinstance(j, dict) else {}
        except Exception:
            return {}
    return {}

def _merge_unique(xs):
    out, seen = [], set()
    for x in xs or []:
        if not x:
            continue
        if x not in seen:
            out.append(x)
            seen.add(x)
    return out

def _is_po_symbol(sym: str) -> bool:
    s = (sym or "").upper()
    return s in PO_MAJOR or bool(re.fullmatch(r"[A-Z]{6}", s))

def _to_deriv(sym: str) -> str:
    if not sym:
        return sym
    s = sym.strip()
    if s.startswith("frx"):
        return s
    sU = s.upper().replace("/", "")
    return "frx" + sU if _is_po_symbol(sU) else s

def _expand_all_symbols(tokens):
    out = []
    for t in tokens or []:
        tt = (t or "").strip().upper()
        if tt in ("ALL","ALL_DERIV","__ALL__","__ALL_DERIV__"):
            out += DERIV_FRX
        elif tt in ("ALL_PO","__ALL_PO__","PO_ALL"):
            out += PO_MAJOR
        else:
            out.append(t)
    return _merge_unique(out)

def _ena(ind_cfg, key: str) -> bool:
    return bool((_cfg_dict(ind_cfg).get(key) or {}).get("enabled"))

# ===== Candle plotting & trades ==============================================
def _draw_candles(ax, df):
    ts = pd.to_datetime(df["timestamp"])
    o,h,l,c = [pd.to_numeric(df[k], errors="coerce") for k in ("open","high","low","close")]
    if len(ts) > 1:
        step = ts.diff().dropna().dt.total_seconds().median() or 60.0
    else:
        step = 60.0
    width = (step/(24*3600))*0.9
    # FutureWarning-safe conversion:
    x = mdates.date2num(np.array(ts.dt.to_pydatetime().tolist()))
    for xi, oo, hh, ll, cc in zip(x,o,h,l,c):
        if pd.isna([oo,hh,ll,cc]).any():
            continue
        up = cc >= oo
        color = "#17c964" if up else "#f31260"
        ax.vlines(xi, ll, hh, color=color, linewidth=1.0, alpha=.9)
        ax.add_patch(plt.Rectangle((xi-width/2, min(oo,cc)), width, max(abs(cc-oo),1e-9),
                                   facecolor=color, edgecolor=color, linewidth=.8, alpha=.95))
    ax.grid(True, alpha=.15)

def _expiry_seconds(expiry):
    s=(expiry or "").lower().strip()
    if s.endswith("m"): return int(re.sub(r"\D","",s))*60
    if s.endswith("h"): return int(re.sub(r"\D","",s))*3600
    if s.endswith("d"): return int(re.sub(r"\D","",s))*86400
    if s.isdigit():     return int(s)*60
    return 300  # default 5m

def _extract_trades(bt, df, expiry):
    out=[]; exp_s=_expiry_seconds(expiry)

    def to_ts(x):
        if isinstance(x,(int,float)):
            i=int(x)
            if 0<=i<len(df): return pd.to_datetime(df["timestamp"].iloc[i])
            return None
        try: return pd.to_datetime(x)
        except Exception: return None

    if hasattr(bt,"trades_df") and isinstance(bt.trades_df, pd.DataFrame):
        for _,r in bt.trades_df.iterrows():
            t=to_ts(r.get("entry_time") or r.get("time")); side=(r.get("side") or r.get("direction") or "").upper()
            if t is None or side not in ("BUY","SELL"): continue
            exp=to_ts(r.get("expiry_time")) or t+timedelta(seconds=exp_s)
            out.append({"t":t,"expiry":exp,"side":side})
    elif hasattr(bt,"entries") and isinstance(bt.entries, list):
        for e in bt.entries:
            t=to_ts(e.get("time") or e.get("entry_time")); side=(e.get("side") or e.get("direction") or "").upper()
            if t is None or side not in ("BUY","SELL"): continue
            exp=to_ts(e.get("expiry_time")) or t+timedelta(seconds=exp_s)
            out.append({"t":t,"expiry":exp,"side":side})
    elif hasattr(bt,"signals") and isinstance(bt.signals, list):
        for e in bt.signals:
            t=to_ts(e.get("time")); side=(e.get("side") or e.get("signal") or "").upper()
            if t is None or side not in ("BUY","SELL"): continue
            out.append({"t":t,"expiry":t+timedelta(seconds=exp_s),"side":side})
    return out

def _save_plot(sym, tf, expiry, df, ind_cfg, trades, outdir="static/plots", bars=200):
    os.makedirs(outdir, exist_ok=True)

    # Window & normalize
    ds = df.copy()
    ds["timestamp"] = pd.to_datetime(ds["timestamp"])
    for k in ("open","high","low","close"):
        ds[k] = pd.to_numeric(ds[k], errors="coerce")
    ds = ds.dropna(subset=["open","high","close"]).sort_values("timestamp").tail(bars).reset_index(drop=True)

    # Compute indicators using indicators.py (only-enabled plotting)
    ta_in = pd.DataFrame({
        "time": ds["timestamp"],
        "Open": ds["open"], "High": ds["high"], "Low": ds["low"], "Close": ds["close"]
    }).set_index("time")

    try:
        ta_full = apply_indicators(ta_in, {"indicators": ind_cfg})
    except Exception:
        # If apply_indicators blows up, fall back to plain price-only plot
        ta_full = ta_in.copy()
    ta_full = ta_full.reset_index().rename(columns={"index":"time"})

    # Determine which panels are needed (based on enabled + column existence)
    def has(cols): return all(c in ta_full.columns for c in cols)

    want_rsi   = _ena(ind_cfg,"RSI")    and has(["rsi"])
    want_stoch = _ena(ind_cfg,"STOCH")  and has(["stoch_k","stoch_d"])
    want_macd  = _ena(ind_cfg,"MACD")   and has(["macd","macd_signal","macd_hist"])
    want_osma  = _ena(ind_cfg,"OSMA")   and has(["macd","macd_signal"])
    want_adx   = _ena(ind_cfg,"ADX")    and has(["adx"])
    want_atr   = _ena(ind_cfg,"ATR")    and has(["atr"])
    want_aroon = _ena(ind_cfg,"AROON")  and any(c in ta_full.columns for c in ["aroon_up","aroon_down","aroon_osc"])
    want_vtx   = _ena(ind_cfg,"VORTEX") and has(["vortex_pos","vortex_neg"])
    want_cci   = _ena(ind_cfg,"CCI")    and has(["cci"])
    want_roc   = _ena(ind_cfg,"ROC")    and has(["roc"])
    want_mom   = _ena(ind_cfg,"MOM")    and has(["mom"])
    want_wr    = _ena(ind_cfg,"WILLR")  and has(["willr"])
    want_stc   = _ena(ind_cfg,"STC")    and has(["stc"])
    want_ao    = _ena(ind_cfg,"AO")     and has(["ao"])
    want_ac    = _ena(ind_cfg,"AC")     and has(["ac"])
    want_bbw   = _ena(ind_cfg,"BBW")    and has(["bb_width"])
    want_bulls = _ena(ind_cfg,"BULL_POWER") and has(["bull_power"])
    want_bears = _ena(ind_cfg,"BEAR_POWER") and has(["bear_power"])

    panels = [
        want_rsi, want_stoch, (want_macd or want_osma), want_adx, want_atr,
        want_aroon, want_vtx, want_cci, want_roc, want_mom, want_wr, want_stc,
        want_ao or want_ac, want_bbw, want_bulls or want_bears
    ]
    rows = 1 + sum(1 for x in panels if x)

    fig, axes = plt.subplots(rows, 1, figsize=(14, 3.0*rows+0.8), sharex=True)
    if rows == 1:
        axes = [axes]

    # Price panel
    axp = axes[0]
    _draw_candles(axp, ds)
    ts = ds["timestamp"]

    def maybe_line(series_name, label=None, **kw):
        if series_name in ta_full.columns:
            axp.plot(ts, pd.to_numeric(ta_full[series_name], errors="coerce"),
                     label=label or series_name, linewidth=1.05, **kw)

    # MAs
    if _ena(ind_cfg,"EMA"):   maybe_line("ema",   label=f"EMA({(_cfg_dict(ind_cfg).get('EMA') or {}).get('period',20)})")
    if _ena(ind_cfg,"SMA"):   maybe_line("sma",   label=f"SMA({(_cfg_dict(ind_cfg).get('SMA') or {}).get('period',20)})")
    if _ena(ind_cfg,"WMA"):   maybe_line("wma",   label=f"WMA({(_cfg_dict(ind_cfg).get('WMA') or {}).get('period',20)})")
    if _ena(ind_cfg,"SMMA"):  maybe_line("smma",  label=f"SMMA({(_cfg_dict(ind_cfg).get('SMMA') or {}).get('period',20)})")
    if _ena(ind_cfg,"TMA"):   maybe_line("tma",   label=f"TMA({(_cfg_dict(ind_cfg).get('TMA') or {}).get('period',20)})")
    if _ena(ind_cfg,"MA"):    maybe_line("ma",    label=f"MA({(_cfg_dict(ind_cfg).get('MA') or {}).get('period',50)})")

    # Bands & channels
    if _ena(ind_cfg,"BB") and has(["bb_low","bb_mid","bb_high"]):
        axp.fill_between(ts, ta_full["bb_low"], ta_full["bb_high"], alpha=.12, label="BB")
        maybe_line("bb_mid", "BB mid")
    if _ena(ind_cfg,"KC") and has(["kc_low","kc_mid","kc_high"]):
        axp.fill_between(ts, ta_full["kc_low"], ta_full["kc_high"], alpha=.08, label="KC")
        maybe_line("kc_mid", "KC mid")
    if _ena(ind_cfg,"DON") and has(["don_low","don_mid","don_high"]):
        axp.fill_between(ts, ta_full["don_low"], ta_full["don_high"], alpha=.06, label="Donchian")
        maybe_line("don_mid", "Don mid")
    if _ena(ind_cfg,"ENV") and has(["env_low","env_mid","env_high"]):
        axp.fill_between(ts, ta_full["env_low"], ta_full["env_high"], alpha=.06, label="Envelopes")
        maybe_line("env_mid", "Env mid")

    # Ichimoku / SuperTrend / PSAR
    if _ena(ind_cfg,"ICHI") and has(["ichi_senkou_a","ichi_senkou_b"]):
        maybe_line("ichi_senkou_a","Senkou A", alpha=.8)
        maybe_line("ichi_senkou_b","Senkou B", alpha=.8)
    if _ena(ind_cfg,"SUPERTREND"):
        if "supertrend_upper" in ta_full.columns and "supertrend_lower" in ta_full.columns:
            axp.fill_between(ts, ta_full["supertrend_lower"], ta_full["supertrend_upper"], alpha=.07, label="SuperTrend")
        if "supertrend_dir" in ta_full.columns:
            maybe_line("supertrend_dir","ST dir", alpha=.5)
    if _ena(ind_cfg,"PSAR") and "psar" in ta_full.columns:
        axp.scatter(ts, ta_full["psar"], s=8, marker='.', alpha=.8, label="PSAR")

    # Fractals / FCB / ZigZag / Alligator
    if _ena(ind_cfg,"FRACTAL"):
        if "fractal_high" in ta_full.columns:
            axp.scatter(ts, ta_full["fractal_high"], s=18, marker='^', alpha=.7, label="FractalH")
        if "fractal_low" in ta_full.columns:
            axp.scatter(ts, ta_full["fractal_low"], s=18, marker='v', alpha=.7, label="FractalL")
    if _ena(ind_cfg,"FCB") and has(["fcb_low","fcb_high"]):
        axp.fill_between(ts, ta_full["fcb_low"], ta_full["fcb_high"], alpha=.06, label="FCB")
    if _ena(ind_cfg,"ZIGZAG") and "zigzag" in ta_full.columns:
        maybe_line("zigzag","ZigZag", alpha=.9)
    if _ena(ind_cfg,"ALLIGATOR"):
        for col, lab in (("jaw","Jaw"),("teeth","Teeth"),("lips","Lips")):
            if col in ta_full.columns: maybe_line(col, f"Alligator {lab}", alpha=.9)

    # Trades (markers + shaded expiry)
    for tr in (trades or []):
        t  = pd.to_datetime(tr["t"]); te = pd.to_datetime(tr["expiry"])
        side = (tr["side"] or "").upper()
        marker = "▲" if side=="BUY" else "▼"
        color  = "#17c964" if side=="BUY" else "#f31260"
        try:
            idx = ds["timestamp"].searchsorted(t)
            y   = float(ds["close"].iloc[max(0, min(idx, len(ds)-1))])
        except Exception:
            y   = float(ds["close"].iloc[-1])
        axp.annotate(marker, (t,y), color=color, fontsize=11,
                     ha="center", va="bottom" if side=="BUY" else "top")
        axp.axvspan(t, te, color=color, alpha=.10)

    axp.set_title(f"{sym} • TF={tf} • Expiry={expiry}")
    if axp.get_legend_handles_labels()[0]:
        axp.legend(loc="upper left", fontsize=8, ncols=3)

    # Panels
    rows_i = 1
    def next_ax():
        nonlocal rows_i
        ax = axes[rows_i]; rows_i += 1
        ax.grid(True, alpha=.15)
        return ax

    if want_rsi:
        ax = next_ax()
        ax.plot(ts, ta_full["rsi"], label="RSI")
        ax.axhline(70, color="#aa4444", linewidth=.7)
        ax.axhline(50, color="#888", linewidth=.7)
        ax.axhline(30, color="#44aa44", linewidth=.7)
        ax.set_ylim(0,100); ax.set_ylabel("RSI"); ax.legend(loc="upper left", fontsize=8)

    if want_stoch:
        ax = next_ax()
        ax.plot(ts, ta_full["stoch_k"], label="%K")
        ax.plot(ts, ta_full["stoch_d"], label="%D", linestyle="--")
        ax.axhline(80, color="#aa4444", linewidth=.7)
        ax.axhline(20, color="#44aa44", linewidth=.7)
        ax.set_ylim(0,100); ax.set_ylabel("Stoch"); ax.legend(loc="upper left", fontsize=8)

    if want_macd or want_osma:
        ax = next_ax()
        if want_macd:
            ax.plot(ts, ta_full["macd"], label="MACD")
            ax.plot(ts, ta_full["macd_signal"], label="Signal", linestyle="--")
            if "macd_hist" in ta_full.columns:
                ax.fill_between(ts, 0, ta_full["macd_hist"], alpha=.25, label="Hist")
        if want_osma and "macd" in ta_full.columns and "macd_signal" in ta_full.columns:
            ax.plot(ts, ta_full["macd"] - ta_full["macd_signal"], label="OsMA")
        ax.set_ylabel("MACD/OsMA"); ax.legend(loc="upper left", fontsize=8)

    if want_ao or want_ac:
        ax = next_ax()
        if want_ao and "ao" in ta_full.columns:
            ax.plot(ts, ta_full["ao"], label="AO")
        if want_ac and "ac" in ta_full.columns:
            ax.plot(ts, ta_full["ac"], label="AC")
        ax.set_ylabel("AO/AC"); ax.legend(loc="upper left", fontsize=8)

    if want_adx:
        ax = next_ax()
        ax.plot(ts, ta_full["adx"], label="ADX")
        ax.axhline(20, color="#888", linewidth=.7)
        ax.set_ylabel("ADX"); ax.legend(loc="upper left", fontsize=8)

    if want_atr:
        ax = next_ax()
        ax.plot(ts, ta_full["atr"], label="ATR")
        ax.set_ylabel("ATR"); ax.legend(loc="upper left", fontsize=8)

    if want_aroon:
        ax = next_ax()
        if "aroon_up" in ta_full.columns:   ax.plot(ts, ta_full["aroon_up"], label="Aroon Up")
        if "aroon_down" in ta_full.columns: ax.plot(ts, ta_full["aroon_down"], label="Aroon Down")
        if "aroon_osc" in ta_full.columns:  ax.plot(ts, ta_full["aroon_osc"], label="Osc")
        ax.set_ylabel("Aroon"); ax.legend(loc="upper left", fontsize=8)

    if want_vtx:
        ax = next_ax()
        ax.plot(ts, ta_full["vortex_pos"], label="+VI")
        ax.plot(ts, ta_full["vortex_neg"], label="-VI")
        ax.set_ylabel("Vortex"); ax.legend(loc="upper left", fontsize=8)

    if want_cci:
        ax = next_ax()
        ax.plot(ts, ta_full["cci"], label="CCI")
        ax.set_ylabel("CCI"); ax.legend(loc="upper left", fontsize=8)

    if want_roc:
        ax = next_ax()
        ax.plot(ts, ta_full["roc"], label="ROC")
        ax.set_ylabel("ROC"); ax.legend(loc="upper left", fontsize=8)

    if want_mom:
        ax = next_ax()
        ax.plot(ts, ta_full["mom"], label="Momentum")
        ax.set_ylabel("Momentum"); ax.legend(loc="upper left", fontsize=8)

    if want_wr:
        ax = next_ax()
        ax.plot(ts, ta_full["willr"], label="Williams %R")
        ax.set_ylabel("%R"); ax.legend(loc="upper left", fontsize=8)

    if want_stc:
        ax = next_ax()
        ax.plot(ts, ta_full["stc"], label="STC")
        ax.set_ylabel("STC"); ax.legend(loc="upper left", fontsize=8)

    if want_bbw:
        ax = next_ax()
        ax.plot(ts, ta_full["bb_width"], label="BB Width")
        ax.set_ylabel("BBW"); ax.legend(loc="upper left", fontsize=8)

    if want_bulls or want_bears:
        ax = next_ax()
        if want_bulls and "bull_power" in ta_full.columns:
            ax.plot(ts, ta_full["bull_power"], label="Bulls")
        if want_bears and "bear_power" in ta_full.columns:
            ax.plot(ts, ta_full["bear_power"], label="Bears")
        ax.set_ylabel("Bulls/Bears"); ax.legend(loc="upper left", fontsize=8)

    fig.autofmt_xdate(); fig.tight_layout()
    stamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    name  = f"{sym.replace('/','_')}_{tf}_{expiry}_{stamp}_{uuid.uuid4().hex[:6]}.png"
    path  = os.path.join(outdir, name)
    plt.savefig(path, dpi=140); plt.close(fig)
    return name

# ===== Auth & basic pages =====================================================
def require_login(func):
    from functools import wraps
    @wraps(func)
    def wrapper(*a, **kw):
        if not session.get("admin"):
            return redirect(url_for("dashboard.login", next=request.path))
        return func(*a, **kw)
    return wrapper

@bp.route("/_up", methods=["GET","HEAD"])
def up_check():
    return "OK", 200

@bp.route("/login", methods=["GET","POST"])
def login():
    if request.method=="POST":
        if request.form.get("password")==os.getenv("ADMIN_PASSWORD","admin123"):
            session["admin"]=True; flash("Logged in.")
            return redirect(request.args.get("next") or url_for("dashboard.dashboard"))
        flash("Invalid password","error")
    cfg=_cfg_dict(get_config())
    return render_template("dashboard.html", view="login", window=cfg.get("window",{}), tz=TIMEZONE)

@bp.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("dashboard.index"))

@bp.route("/")
def index():
    cfg=_cfg_dict(get_config())
    return render_template("dashboard.html", view="index",
                           within=within_window(cfg),
                           now=datetime.now(TZ).strftime("%Y-%m-%d %H:%M:%S"),
                           tz=TIMEZONE, window=cfg.get("window", {}))

# ===== Dashboard ==============================================================
@bp.route("/dashboard")
@require_login
def dashboard():
    exec_sql("""CREATE TABLE IF NOT EXISTS users(telegram_id TEXT PRIMARY KEY, tier TEXT, expires_at TEXT)""")
    rows=exec_sql("SELECT telegram_id,tier,COALESCE(expires_at,'') FROM users", fetch=True) or []
    users=[{"telegram_id":r[0],"tier":r[1],"expires_at":r[2] or None} for r in rows]

    cfg=_cfg_dict(get_config())
    customs=[dict(_cfg_dict(cfg.get("custom1")),_idx=1),
             dict(_cfg_dict(cfg.get("custom2")),_idx=2),
             dict(_cfg_dict(cfg.get("custom3")),_idx=3)]
    strategies_core=_cfg_dict(cfg.get("strategies")) or {"BASE":{"enabled":True},"TREND":{"enabled":False},"CHOP":{"enabled":False}}
    strategies_all=dict(strategies_core)
    for i,c in enumerate(customs, start=1): strategies_all[f"CUSTOM{i}"]={"enabled": bool(c.get("enabled"))}

    bt=session.get("bt", {})

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

# ===== Settings updates =======================================================
@bp.route("/update_window", methods=["POST"])
@require_login
def update_window():
    cfg=_cfg_dict(get_config()); cfg.setdefault("window", {"start":"08:00","end":"17:00","timezone":TIMEZONE})
    for k in ("start","end","timezone"): cfg["window"][k]=request.form.get(k, cfg["window"][k])
    if request.form.get("live_tf"): cfg["live_tf"]=request.form.get("live_tf").upper()
    if request.form.get("live_expiry"): cfg["live_expiry"]=request.form.get("live_expiry")
    set_config(cfg); flash("Trading window & live defaults updated."); return redirect(url_for("dashboard.dashboard"))

@bp.route("/update_strategies", methods=["POST"])
@require_login
def update_strategies():
    cfg=_cfg_dict(get_config()); cfg.setdefault("strategies", {"BASE":{"enabled":True},"TREND":{"enabled":False},"CHOP":{"enabled":False}})
    for name in list(cfg["strategies"].keys()):
        cfg["strategies"][name]["enabled"]=bool(request.form.get(f"s_{name}"))
    for i in (1,2,3):
        box=bool(request.form.get(f"s_CUSTOM{i}")); key=f"custom{i}"
        cfg.setdefault(key, {}); cfg[key]["enabled"]=box; cfg["strategies"][f"CUSTOM{i}"]={"enabled":box}
    # backtest/live TF & expiry
    if request.form.get("bt_tf"): cfg["bt_tf"]=request.form.get("bt_tf").upper()
    if request.form.get("bt_expiry"): cfg["bt_expiry"]=request.form.get("bt_expiry")
    if request.form.get("live_tf"): cfg["live_tf"]=request.form.get("live_tf").upper()
    if request.form.get("live_expiry"): cfg["live_expiry"]=request.form.get("live_expiry")
    set_config(cfg); flash("Strategies (including CUSTOM) updated."); return redirect(url_for("dashboard.dashboard"))

@bp.route("/update_symbols", methods=["POST"])
@require_login
def update_symbols():
    cfg=_cfg_dict(get_config())
    sel_po=request.form.getlist("symbols_po_multi"); sel_deriv=request.form.getlist("symbols_deriv_multi")
    raw=(request.form.get("symbols_text") or "").strip()
    text_syms=[s.strip() for s in re.split(r"[,\s]+", raw) if s.strip()]
    convert_po=bool(request.form.get("convert_po"))
    merged=_merge_unique(sel_po + sel_deriv + text_syms)
    normalized=[_to_deriv(s) if convert_po else s for s in merged]
    cfg["symbols"]=normalized; cfg["symbols_raw"]=merged; set_config(cfg)
    flash(("Active symbols saved (PO→Deriv conversion ON): " if convert_po else "Active symbols saved: ")+ (", ".join(normalized) if normalized else "(none)"))
    return redirect(url_for("dashboard.dashboard"))

@bp.route("/update_indicators", methods=["POST"])
@require_login
def update_indicators():
    cfg=_cfg_dict(get_config()); inds=_cfg_dict(cfg.get("indicators"))
    # Walk full catalog; expect fields enabled + params
    for key, spec in INDICATOR_SPECS.items():
        enabled=bool(request.form.get(f"ind_{key}_enabled"))
        inds.setdefault(key, {}); inds[key]["enabled"]=enabled
        for p in spec.get("params", {}).keys():
            fk=f"ind_{key}_{p}"
            if fk in request.form and request.form.get(fk)!="":
                val=request.form.get(fk)
                # numeric if possible
                try:
                    if re.fullmatch(r"-?\d+", val):
                        inds[key][p]=int(val)
                    else:
                        inds[key][p]=float(val)
                except Exception:
                    inds[key][p]=val
    cfg["indicators"]=inds; set_config(cfg); flash("Indicators updated.")
    return redirect(url_for("dashboard.dashboard"))

@bp.route("/update_custom", methods=["POST"])
@require_login
def update_custom():
    slot=(request.form.get("slot") or "1").strip(); prefix=f"custom{slot}"
    cfg=_cfg_dict(get_config()); cfg.setdefault(prefix, {})
    mode=(request.form.get("mode","SIMPLE") or "SIMPLE").upper()
    cfg[prefix]["enabled"]=bool(request.form.get("enabled")); cfg[prefix]["mode"]=mode
    if mode=="SIMPLE":
        cfg[prefix]["simple_buy"]=request.form.get("simple_buy","")
        cfg[prefix]["simple_sell"]=request.form.get("simple_sell","")
        cfg[prefix]["buy_rule"]=parse_natural_rule(cfg[prefix]["simple_buy"])
        cfg[prefix]["sell_rule"]=parse_natural_rule(cfg[prefix]["simple_sell"])
    else:
        try: cfg[prefix]["buy_rule"]=json.loads(request.form.get("buy_rule_json","{}"))
        except Exception: cfg[prefix]["buy_rule"]={}
        try: cfg[prefix]["sell_rule"]=json.loads(request.form.get("sell_rule_json","{}"))
        except Exception: cfg[prefix]["sell_rule"]={}
    try: cfg[prefix]["tol_pct"]=float(request.form.get("tol_pct", cfg[prefix].get("tol_pct", .1)))
    except Exception: pass
    try: cfg[prefix]["lookback"]=int(request.form.get("lookback", cfg[prefix].get("lookback", 3)))
    except Exception: pass
    cfg.setdefault("strategies", {}); cfg["strategies"][f"CUSTOM{slot}"]={"enabled": bool(cfg[prefix]["enabled"])}
    set_config(cfg); flash(f"Custom #{slot} saved."); return redirect(url_for("dashboard.dashboard"))

# ===== Users ==================================================================
@bp.route("/users/add", methods=["POST"])
@require_login
def users_add():
    telegram_id=(request.form.get("telegram_id") or "").strip()
    tier=(request.form.get("tier") or "free").strip()
    expires_at=(request.form.get("expires_at") or "").strip() or None
    if not telegram_id: flash("Telegram ID required.","error"); return redirect(url_for("dashboard.dashboard"))
    exec_sql("""CREATE TABLE IF NOT EXISTS users(telegram_id TEXT PRIMARY KEY, tier TEXT, expires_at TEXT)""")
    exec_sql("""INSERT INTO users(telegram_id,tier,expires_at) VALUES(?,?,?)
                ON CONFLICT(telegram_id) DO UPDATE SET tier=excluded.tier, expires_at=excluded.expires_at""",
             (telegram_id,tier,expires_at))
    flash(f"User saved: {telegram_id} ({tier})"); return redirect(url_for("dashboard.dashboard"))

@bp.route("/users/delete", methods=["POST"])
@require_login
def users_delete():
    telegram_id=(request.form.get("telegram_id") or "").strip()
    if not telegram_id: flash("Telegram ID required.","error"); return redirect(url_for("dashboard.dashboard"))
    exec_sql("DELETE FROM users WHERE telegram_id=?", (telegram_id,))
    flash(f"User deleted: {telegram_id}"); return redirect(url_for("dashboard.dashboard"))

# ===== Deriv fetch (optional) =================================================
@bp.route("/deriv_fetch", methods=["POST"])
@require_login
def deriv_fetch():
    if not (_fetch_one_symbol and _deriv_csv_path):
        flash("Deriv fetch helper (data_fetch.py) not found in this build.","error")
        return redirect(url_for("dashboard.dashboard"))
    app_id=os.getenv("DERV_APP_ID") or os.getenv("DERIV_APP_ID","1089")
    tf=(request.form.get("fetch_tf") or "M5").upper()
    count=int(request.form.get("fetch_count") or "300")
    convert_po=bool(request.form.get("convert_po_fetch"))
    gran={"M1":60,"M2":120,"M3":180,"M5":300,"M10":600,"M15":900,"M30":1800,"H1":3600,"H4":14400,"D1":86400}.get(tf,300)
    tokens=[s.strip() for s in re.split(r"[,\s]+", (request.form.get("fetch_symbols") or "")) if s.strip()]
    pairs=[_to_deriv(s) if convert_po else s for s in tokens]
    ok=0; fail=[]
    for sym in pairs:
        try: _fetch_one_symbol(app_id, sym, gran, count); ok+=1
        except Exception as e: fail.append(f"{sym}: {e}")
    if ok: flash(f"Deriv: saved {ok} symbol(s) @ {tf}")
    if fail: flash("Errors: " + "; ".join(fail))
    return redirect(url_for("dashboard.dashboard"))

# ===== Backtest ===============================================================
@bp.route("/backtest", methods=["POST"])
@require_login
def backtest():
    cfg=_cfg_dict(get_config())
    tf=(request.form.get("bt_tf") or cfg.get("bt_tf") or "M5").upper()
    expiry=(request.form.get("bt_expiry") or cfg.get("bt_expiry") or "5m").lower()
    strategy=(request.form.get("bt_strategy") or "BASE").upper()

    use_server=bool(request.form.get("use_server"))
    app_id=os.getenv("DERV_APP_ID") or os.getenv("DERIV_APP_ID","1089")
    count=int(request.form.get("bt_count") or "300")
    convert_po=bool(request.form.get("convert_po_bt"))
    gran={"M1":60,"M2":120,"M3":180,"M5":300,"M10":600,"M15":900,"M30":1800,"H1":3600,"H4":14400,"D1":86400}.get(tf,300)

    # symbols
    raw_syms=(request.form.get("bt_symbols") or " ".join(cfg.get("symbols") or [])).strip()
    text_syms=[s.strip() for s in re.split(r"[,\s]+", raw_syms) if s.strip()]
    from_multi=request.form.getlist("bt_symbols_multi")
    chosen=_expand_all_symbols(from_multi if from_multi else text_syms)
    symbols=[_to_deriv(s) if convert_po else s for s in chosen]
    if not symbols:
        session["bt"]={"error":"No symbols selected."}; flash("Backtest error: no symbols."); return redirect(url_for("dashboard.dashboard"))

    uploaded=request.files.get("bt_csv")
    results=[]; summary={"trades":0,"wins":0,"losses":0,"draws":0,"winrate":0.0}
    warnings=[]; first_plot_name=None

    def run_one(sym, df):
        nonlocal results, summary, first_plot_name
        core = "CUSTOM" if strategy.startswith("CUSTOM") else strategy
        cfg_run = dict(cfg)  # pass full cfg to strategy core
        try:
            bt=run_backtest_core_binary(df, core, cfg_run, tf, expiry)
        except Exception:
            # guard odd cases like "'str' object has no attribute 'get'"
            bt=run_backtest_core_binary(df, core, {}, tf, expiry)
        r={"symbol":sym,
           "trades": getattr(bt,"trades",0) if isinstance(getattr(bt,"trades",0),(int,float)) else (getattr(bt,"trades_count",0) or 0),
           "wins": getattr(bt,"wins",0), "losses": getattr(bt,"losses",0),
           "draws": getattr(bt,"draws",0),
           "winrate": round((getattr(bt,"winrate",0.0) or 0.0)*100,2) if isinstance(getattr(bt,"winrate",0.0),(int,float)) else 0.0}
        results.append(r)
        for k in ("trades","wins","losses","draws"): summary[k]+=r[k]
        # indicators from cfg (only enabled get drawn)
        ind_cfg=_cfg_dict(cfg.get("indicators") or {})
        trades_list=_extract_trades(bt, df, expiry)
        name=_save_plot(sym, tf, expiry, df, ind_cfg, trades_list, outdir="static/plots", bars=200)
        if first_plot_name is None: first_plot_name=name

    try:
        if uploaded and uploaded.filename:
            data=uploaded.read().decode("utf-8", errors="ignore")
            df=pd.read_csv(StringIO(data)); df.columns=[c.strip().lower() for c in df.columns]
            if "timestamp" in df.columns: df["timestamp"]=pd.to_datetime(df["timestamp"])
            else:
                # try other time column names
                tcol = next((c for c in ("time","date","epoch") if c in df.columns), None)
                if tcol:
                    if tcol=="epoch":
                        df["timestamp"]=pd.to_datetime(pd.to_numeric(df[tcol], errors="coerce"), unit="s")
                    else:
                        df["timestamp"]=pd.to_datetime(df[tcol])
            for c in ("open","high","low","close"): 
                if c in df.columns: df[c]=pd.to_numeric(df[c], errors="coerce")
            df=df.dropna(subset=["close","timestamp"]).sort_values("timestamp").reset_index(drop=True)
            run_one(symbols[0] if symbols else "CSV", df)
        else:
            for sym in symbols:
                try:
                    if use_server:
                        if not _fetch_one_symbol: raise RuntimeError("data_fetch helper not present")
                        _fetch_one_symbol(app_id, sym, gran, count)
                    if not _deriv_csv_path: raise RuntimeError("data_fetch helper not present")
                    path=_deriv_csv_path(sym, gran)
                    if not os.path.exists(path): warnings.append(f"{sym}: no CSV"); continue
                    df=pd.read_csv(path); df.columns=[c.strip().lower() for c in df.columns]
                    if "timestamp" in df.columns: df["timestamp"]=pd.to_datetime(df["timestamp"])
                    for c in ("open","high","low","close"): df[c]=pd.to_numeric(df[c], errors="coerce")
                    df=df.dropna(subset=["close","timestamp"]).sort_values("timestamp").reset_index(drop=True)
                    if len(df)<30: warnings.append(f"{sym}: not enough candles ({len(df)})"); continue
                    run_one(sym, df)
                except Exception as e:
                    warnings.append(f"{sym}: {e}")
        summary["winrate"]=round((summary["wins"]/summary["trades"])*100,2) if summary["trades"] else 0.0
        payload={"summary":summary, "results":results, "tf":tf, "expiry":expiry, "strategy":strategy}
        if first_plot_name: payload["plot_name"]=first_plot_name
        if warnings: payload["warnings"]=warnings; flash("Backtest completed with warnings.")
        else: flash("Backtest complete.")
        session["bt"]=payload
    except Exception as e:
        session["bt"]={"error": str(e), "warnings": warnings}; flash(f"Backtest error: {e}", "error")
    return redirect(url_for("dashboard.dashboard"))

@bp.route("/backtest/last.json")
@require_login
def backtest_last_json():
    return jsonify(session.get("bt") or {"error":"no backtest"})

@bp.route("/backtest/last.csv")
@require_login
def backtest_last_csv():
    bt=session.get("bt") or {}
    rows=bt.get("results") or []
    si=StringIO(); w=csv.writer(si)
    w.writerow(["symbol","trades","wins","losses","draws","winrate"])
    for r in rows: w.writerow([r["symbol"],r["trades"],r["wins"],r["losses"],r["draws"],r["winrate"]])
    return Response(si.getvalue(), mimetype="text/csv",
                    headers={"Content-Disposition":"attachment; filename=backtest_results.csv"})

# ===== No-cache plot serving ==================================================
@bp.route("/plots/<name>")
def plot_file(name):
    resp = make_response(send_from_directory("static/plots", name, max_age=0))
    resp.headers["Cache-Control"]="no-store, no-cache, must-revalidate, max-age=0"
    resp.headers["Pragma"]="no-cache"
    resp.headers["Expires"]="0"
    return resp

# ===== Live engine controls + tally ==========================================
@bp.route("/live/status")
def live_status():
    return jsonify({"ok":True, "status": ENGINE.status()})

@bp.route("/live/tally")
def live_tally():
    return jsonify({"ok": True, "tally": ENGINE.tally()})

@bp.route("/live/start", methods=["POST","GET"])
def live_start():
    ok, msg = ENGINE.start()
    if request.method=="GET": return jsonify({"ok":ok, "msg":msg})
    flash(f"Live: {msg}"); return redirect(url_for("dashboard.dashboard"))

@bp.route("/live/stop", methods=["POST","GET"])
def live_stop():
    ok, msg = ENGINE.stop()
    if request.method=="GET": return jsonify({"ok":ok, "msg":msg})
    flash(f"Live: {msg}"); return redirect(url_for("dashboard.dashboard"))

@bp.route("/live/debug/<state>")
def live_debug(state):
    ENGINE.debug = (state or "").lower()=="on"
    return jsonify({"ok":True, "debug": ENGINE.debug})

# ===== Telegram diagnostics ===================================================
@bp.route("/telegram/test", methods=["POST","GET"])
def telegram_test():
    ok, msg = tg_test()
    if request.method=="GET": return jsonify({"ok":ok, "msg":msg})
    flash("Telegram OK" if ok else f"Telegram error: {msg}")
    return redirect(url_for("dashboard.dashboard"))

@bp.route("/telegram/diag")
def telegram_diag():
    # Expect env:
    # TELEGRAM_BOT_TOKEN
    # TELEGRAM_CHAT_ID_FREE, TELEGRAM_CHAT_ID_BASIC, TELEGRAM_CHAT_ID_PRO, TELEGRAM_CHAT_ID_VIP
    token=os.getenv("TELEGRAM_BOT_TOKEN","").strip()
    if not token: return jsonify({"ok":False,"error":"Missing TELEGRAM_BOT_TOKEN"}),200
    try: getme=requests.get(f"https://api.telegram.org/bot{token}/getMe", timeout=8).json()
    except Exception as e: getme={"ok":False,"error":str(e)}
    configured={}
    for k in ("TELEGRAM_CHAT_ID_FREE","TELEGRAM_CHAT_ID_BASIC","TELEGRAM_CHAT_ID_PRO","TELEGRAM_CHAT_ID_VIP"):
        v=os.getenv(k,"").strip()
        if v: configured[k]=v
    ok = bool(configured)
    # quick one-shot send (via ENGINE helper)
    try:
        from live_engine import _send_telegram
        ok_send, info = _send_telegram("🧪 Telegram DIAG: test message.")
    except Exception as e:
        ok_send, info = False, str(e)
    masked = token[:9]+"..."+token[-6:] if len(token)>18 else "***"
    return jsonify({"ok":ok and ok_send,"token_masked":masked,"getMe":getme,"configured_chats":configured,"send_result":info})
