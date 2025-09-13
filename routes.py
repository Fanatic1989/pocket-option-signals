# routes.py — dashboard blueprint, indicator catalog, backtest & live wiring

from __future__ import annotations
import io
import json
import os
from datetime import datetime
from typing import Dict, Any, List

from flask import (
    Blueprint, render_template, request, redirect, url_for, flash,
    session, send_file, jsonify
)

# ---- Local modules
from utils import (
    TIMEZONE, TZ, get_config, set_config, within_window,
    PO_PAIRS, DERIV_PAIRS, convert_po_to_deriv,
    load_csv, fetch_deriv_history, compute_indicators, backtest_run,
    plot_signals
)

# Live engine & Telegram helpers
try:
    from live_engine import ENGINE, tg_test as TG_TEST, BOT_TOKEN, TIER_TO_CHAT, DAILY_CAPS
except Exception:
    # Fallback if names differ in your live_engine
    from live_engine import ENGINE, BOT_TOKEN, TIER_TO_CHAT, DAILY_CAPS
    def TG_TEST():
        # minimal compatibility
        ok, info = ENGINE.send_signal("vip", "Test ping from dashboard")
        return (ok, json.dumps({"vip":{"ok":ok,"info":info}}))

bp = Blueprint("dashboard", __name__)

# ====================================================================================
# FULL indicator catalog (matches your TradingView screenshot). Only enabled ones plot.
# ====================================================================================
INDICATOR_SPECS: Dict[str, Dict[str, Any]] = {
    # Moving averages (overlays)
    "SMA":        {"name":"Simple MA","kind":"overlay","params":{"period":50}, "enabled":False},
    "EMA":        {"name":"Exponential MA","kind":"overlay","params":{"period":20}, "enabled":False},
    "WMA":        {"name":"Weighted MA","kind":"overlay","params":{"period":20}, "enabled":False},
    "SMMA":       {"name":"Smoothed MA","kind":"overlay","params":{"period":20}, "enabled":False},
    "TMA":        {"name":"Triangular MA","kind":"overlay","params":{"period":20}, "enabled":False},

    # Bands / channels (overlays)
    "BOLL":       {"name":"Bollinger Bands","kind":"overlay","params":{"period":20,"mult":2.0}, "enabled":False},
    "BOLL_WIDTH": {"name":"Bollinger Bands Width","kind":"oscillator","params":{"period":20,"mult":2.0}, "enabled":False},
    "DONCHIAN":   {"name":"Donchian Channels","kind":"overlay","params":{"period":20}, "enabled":False},
    "KELTNER":    {"name":"Keltner Channel","kind":"overlay","params":{"period":20,"mult":2.0}, "enabled":False},
    "ENVELOPES":  {"name":"Envelopes","kind":"overlay","params":{"period":20,"pct":2.0}, "enabled":False},

    # Trend / structure (overlays)
    "ICHIMOKU":   {"name":"Ichimoku Kinko Hyo","kind":"overlay","params":{}, "enabled":False},
    "PSAR":       {"name":"Parabolic SAR","kind":"overlay","params":{"step":0.02,"max":0.2}, "enabled":False},
    "SUPERTREND": {"name":"SuperTrend","kind":"overlay","params":{"period":10,"mult":3.0}, "enabled":False},
    "ZIGZAG":     {"name":"ZigZag","kind":"overlay","params":{"pct":1.0}, "enabled":False},
    "FRACTAL":    {"name":"Fractal (display-only)","kind":"overlay","params":{}, "enabled":False},

    # Oscillators & momentum (own panels)
    "RSI":        {"name":"RSI","kind":"oscillator","params":{"period":14}, "enabled":True},
    "STOCH":      {"name":"Stochastic Oscillator","kind":"oscillator","params":{"k":14,"d":3}, "enabled":True},
    "ATR":        {"name":"ATR","kind":"oscillator","params":{"period":14}, "enabled":False},
    "ADX":        {"name":"ADX/+DI/-DI","kind":"oscillator","params":{"period":14}, "enabled":False},
    "CCI":        {"name":"CCI","kind":"oscillator","params":{"period":20}, "enabled":False},
    "MOMENTUM":   {"name":"Momentum","kind":"oscillator","params":{"period":10}, "enabled":False},
    "ROC":        {"name":"Rate of Change","kind":"oscillator","params":{"period":10}, "enabled":False},
    "WILLR":      {"name":"Williams %R","kind":"oscillator","params":{"period":14}, "enabled":False},
    "VORTEX":     {"name":"Vortex","kind":"oscillator","params":{"period":14}, "enabled":False},

    # MACD family
    "MACD":       {"name":"MACD","kind":"oscillator","params":{"fast":12,"slow":26,"signal":9}, "enabled":False},
    "OSMA":       {"name":"OsMA","kind":"oscillator","params":{"fast":12,"slow":26,"signal":9}, "enabled":False},

    # Awesome/Accelerator
    "AO":         {"name":"Awesome Oscillator","kind":"oscillator","params":{}, "enabled":False},
    "AC":         {"name":"Accelerator Oscillator","kind":"oscillator","params":{}, "enabled":False},

    # Power, DeMarker
    "BEARS":      {"name":"Bears Power","kind":"oscillator","params":{"period":13}, "enabled":False},
    "BULLS":      {"name":"Bulls Power","kind":"oscillator","params":{"period":13}, "enabled":False},
    "DEMARKER":   {"name":"DeMarker","kind":"oscillator","params":{"period":14}, "enabled":False},

    # Labels present in your menu (kept for parity; mapped to close equivalents)
    "ALLIGATOR":  {"name":"Alligator (use SMA triplet)","kind":"overlay","params":{"jaws":13,"teeth":8,"lips":5}, "enabled":False},
    "SCHAFF":     {"name":"Schaff Trend Cycle","kind":"oscillator","params":{"fast":23,"slow":50,"cycle":10}, "enabled":False},
}

# Default strategies
DEFAULT_STRATEGIES = {
    "BASE":   {"enabled": True},
    "TREND":  {"enabled": True},
    "CHOP":   {"enabled": False},
    "CUSTOM1":{"enabled": True},
    "CUSTOM2":{"enabled": False},
    "CUSTOM3":{"enabled": False},
}

# ====================================================================================
# Helpers
# ====================================================================================

def _cfg() -> Dict[str, Any]:
    return get_config() or {}

def _save_cfg(obj: Dict[str, Any]) -> None:
    set_config(obj or {})

def _now_local_str() -> str:
    try:
        return datetime.now(TZ).strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")

def _granularity(tf: str) -> int:
    tf = (tf or "M1").upper()
    return {
        "M1":60, "M2":120, "M3":180, "M5":300, "M10":600, "M15":900, "M30":1800,
        "H1":3600, "H4":14400, "D1":86400
    }.get(tf, 60)

def _expiry_bars(exp: str) -> int:
    return {
        "1m":1,"3m":3,"5m":5,"10m":10,"30m":30,"1h":60,"4h":240
    }.get((exp or "5m").lower(), 5)

# ====================================================================================
# Auth (simple)
# ====================================================================================

def _require_admin(view):
    def wrap(*a, **kw):
        if not session.get("admin"):
            return redirect(url_for("dashboard.login"))
        return view(*a, **kw)
    wrap.__name__ = view.__name__
    return wrap

@bp.route("/login", methods=["GET","POST"])
def login():
    if request.method == "POST":
        pw = request.form.get("password","").strip()
        admin_pw = os.getenv("ADMIN_PASSWORD","admin")
        if pw == admin_pw:
            session["admin"]=True
            return redirect(url_for("dashboard.view"))
        flash("Invalid password", "error")
    ctx = _base_ctx("login")
    return render_template("dashboard.html", **ctx)

@bp.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("dashboard.view"))

# ====================================================================================
# Views
# ====================================================================================

def _base_ctx(view_name: str) -> Dict[str, Any]:
    cfg = _cfg()
    now = _now_local_str()
    within = within_window({"window": cfg.get("window", {}), "window_start":None, "window_end":None})
    strategies = cfg.get("strategies") or DEFAULT_STRATEGIES
    indicators = cfg.get("indicators") or {}
    customs = cfg.get("customs") or [{"_idx":1},{"_idx":2},{"_idx":3}]
    bt = cfg.get("last_bt")

    return {
        "view": view_name,
        "tz": TIMEZONE,
        "now": now,
        "within": within,
        "window": cfg.get("window") or {},
        "live_tf":  cfg.get("live_tf",  "M1"),
        "live_expiry": cfg.get("live_expiry", "5m"),
        "available_groups":[
            {"title":"Deriv (frx*)", "items": DERIV_PAIRS},
            {"title":"PO majors", "items": PO_PAIRS},
        ],
        "active_symbols": cfg.get("active_symbols") or [],
        "symbols_raw": cfg.get("symbols_raw") or [],
        "specs": INDICATOR_SPECS,
        "indicators": indicators,
        "strategies_all": DEFAULT_STRATEGIES,
        "strategies": strategies,
        "customs": customs,
        "bt": bt,
        "session": {"admin": bool(session.get("admin"))},
    }

@bp.route("/")
def view_index():
    ctx = _base_ctx("index")
    return render_template("dashboard.html", **ctx)

@bp.route("/dashboard")
@_require_admin
def view():
    ctx = _base_ctx("dashboard")
    return render_template("dashboard.html", **ctx)

# ====================================================================================
# Window defaults / Symbols / Indicators / Strategies
# ====================================================================================

@bp.route("/update_window", methods=["POST"])
@_require_admin
def update_window():
    cfg = _cfg()
    cfg["window"] = {
        "start": request.form.get("start","08:00"),
        "end":   request.form.get("end","17:00"),
        "timezone": request.form.get("timezone", TIMEZONE),
    }
    cfg["live_tf"] = request.form.get("live_tf","M1")
    cfg["live_expiry"] = request.form.get("live_expiry","5m")
    _save_cfg(cfg)
    flash("Saved trading window & defaults", "ok")
    return redirect(url_for("dashboard.view"))

@bp.route("/update_symbols", methods=["POST"])
@_require_admin
def update_symbols():
    raw_text = (request.form.get("symbols_text") or "").replace(",", " ")
    from_multi_deriv = request.form.getlist("symbols_deriv_multi")
    from_multi_po    = request.form.getlist("symbols_po_multi")
    convert_po = bool(request.form.get("convert_po"))

    tokens = [t for t in (raw_text.split() + from_multi_deriv + from_multi_po) if t.strip()]
    tokens = list(dict.fromkeys([t.strip().upper() for t in tokens]))

    cfg = _cfg()
    cfg["symbols_raw"] = tokens[:]
    if convert_po:
        cfg["active_symbols"] = convert_po_to_deriv(tokens)
    else:
        cfg["active_symbols"] = tokens
    _save_cfg(cfg)
    flash("Saved symbols", "ok")
    return redirect(url_for("dashboard.view"))

@bp.route("/update_indicators", methods=["POST"])
@_require_admin
def update_indicators():
    cfg = _cfg()
    inds = {}
    for key, spec in INDICATOR_SPECS.items():
        enabled = bool(request.form.get(f"ind_{key}_enabled"))
        entry = {"enabled": enabled}
        # preserve params (strings accepted; convert where needed in compute layer)
        for p in (spec.get("params") or {}).keys():
            entry[p] = request.form.get(f"ind_{key}_{p}", spec["params"][p])
            # cast basic numbers if possible
            try:
                if isinstance(spec["params"][p], (int,float)):
                    entry[p] = float(entry[p]) if "." in str(entry[p]) else int(entry[p])
            except Exception:
                pass
        inds[key] = entry

    cfg["indicators"] = inds
    _save_cfg(cfg)
    flash("Saved indicators", "ok")
    return redirect(url_for("dashboard.view"))

@bp.route("/update_strategies", methods=["POST"])
@_require_admin
def update_strategies():
    cfg = _cfg()
    strategies = cfg.get("strategies") or DEFAULT_STRATEGIES.copy()
    # Turn everything OFF first, then enable the boxes you received
    for name in DEFAULT_STRATEGIES.keys():
        strategies[name] = strategies.get(name, {"enabled": False})
        strategies[name]["enabled"] = False

    for name in DEFAULT_STRATEGIES.keys():
        if request.form.get(f"s_{name}"):
            strategies[name]["enabled"] = True

    cfg["strategies"] = strategies
    # also defaults the user can change here
    cfg["live_tf"] = request.form.get("live_tf", cfg.get("live_tf","M1"))
    cfg["live_expiry"] = request.form.get("live_expiry", cfg.get("live_expiry","5m"))
    cfg["bt"] = cfg.get("bt", {})
    cfg["bt"]["tf"] = request.form.get("bt_tf", cfg["live_tf"])
    cfg["bt"]["expiry"] = request.form.get("bt_expiry", cfg["live_expiry"])
    _save_cfg(cfg)
    flash("Saved strategy toggles & defaults", "ok")
    return redirect(url_for("dashboard.view"))

# ====================================================================================
# Backtest / Files
# ====================================================================================

@bp.route("/backtest", methods=["POST"])
@_require_admin
def backtest():
    cfg = _cfg()
    symbols = (request.form.get("bt_symbols") or "").replace(",", " ").split()
    symbols = [s.strip() for s in symbols if s.strip()]
    if request.form.get("convert_po_bt"):
        symbols = convert_po_to_deriv(symbols)

    tf = request.form.get("bt_tf", cfg.get("live_tf","M1"))
    expiry = request.form.get("bt_expiry", cfg.get("live_expiry","5m"))
    strategy = request.form.get("bt_strategy","BASE").upper()
    gran = _granularity(tf)

    df = None
    # 1) Uploaded CSV wins
    if "bt_csv" in request.files and request.files["bt_csv"].filename:
        try:
            f = request.files["bt_csv"]
            df = load_csv(io.BytesIO(f.read()))
        except Exception as e:
            flash(f"Backtest error: CSV parse failed: {e}", "error")
            return redirect(url_for("dashboard.view"))
    # 2) Server fetch (Deriv)
    elif request.form.get("use_server"):
        attempts: List[str] = []
        try_syms = []
        for s in symbols:
            # try common casing variants
            try_syms += [s, s.upper(), s.lower(), s.title()]
            if s.upper().startswith("FRX") and s[3:].isupper():
                try_syms += [s.upper(), s.upper()]
        for sym in try_syms or ["frxEURUSD"]:
            try:
                df = fetch_deriv_history(sym, gran, days=5)
                break
            except Exception as e:
                attempts.append(str(e))
                df = None
        if df is None:
            flash(f"Backtest error: Deriv fetch failed. {'; '.join(attempts[-3:])}", "error")
            return redirect(url_for("dashboard.view"))

    else:
        flash("Backtest error: No CSV uploaded and server fetch unchecked.", "error")
        return redirect(url_for("dashboard.view"))

    # Build indicator config (only enabled pass through to compute layer)
    ind_cfg = {}
    ui = cfg.get("indicators") or {}
    for k, v in ui.items():
        if v.get("enabled"):
            # Pass a compact structure the compute layer understands
            if k in ("SMA","EMA","WMA","SMMA","TMA"):
                ind_cfg[k] = int(v.get("period", INDICATOR_SPECS[k]["params"]["period"]))
            elif k == "RSI":
                ind_cfg[k] = {"show": True, "period": int(v.get("period",14))}
            elif k == "STOCH":
                ind_cfg[k] = {"show": True, "k": int(v.get("k",14)), "d": int(v.get("d",3))}
            else:
                # keep params as-is (compute helpers ignore unknown keys gracefully)
                ind_cfg[k] = v

    # Compute + strategy signals
    indicators = compute_indicators(df, ind_cfg)
    signals, stats = backtest_run(df, strategy, ind_cfg, expiry)

    plot_name = plot_signals(df, signals, ind_cfg, strategy, tf, expiry)

    cfg["last_bt"] = {
        "tf": tf, "expiry": expiry, "strategy": strategy,
        "plot_name": plot_name,
        "summary": f"W:{stats['wins']} L:{stats['loss']} D:{stats['draw']}  WR:{stats['win_rate']:.1f}%",
        "warnings": [],
        "stats": stats,
    }
    _save_cfg(cfg)
    flash("Backtest complete", "ok")
    return redirect(url_for("dashboard.view"))

@bp.route("/plot/<name>")
@_require_admin
def plot_file(name: str):
    p = os.path.join("static","plots", name)
    return send_file(p, as_attachment=False)

@bp.route("/backtest/last.json")
@_require_admin
def backtest_last_json():
    bt = _cfg().get("last_bt") or {}
    return jsonify(bt)

@bp.route("/backtest/last.csv")
@_require_admin
def backtest_last_csv():
    # convenience: return a small CSV if you stored one (optional)
    return jsonify({"ok": True, "note":"Attach CSV storage if needed"})

# (Fix) Present in template: provide a safe no-op route so url_for exists
@bp.route("/deriv/fetch", methods=["POST"])
@_require_admin
def deriv_fetch():
    flash("Deriv CSV fetch is not wired here yet. Use Backtest server fetch option.", "error")
    return redirect(url_for("dashboard.view"))

# ====================================================================================
# Live engine + Telegram API used by the right column and the live page JS
# ====================================================================================

@bp.route("/live/status")
def live_status():
    return jsonify({"status": ENGINE.status()})

@bp.route("/live/tally")
def live_tally():
    s = ENGINE.status()
    t = s.get("tallies") or s.get("tally") or {}
    # normalize keys
    by = t.get("by_tier") or t
    all_total = by.get("all") or t.get("total") or (by.get("free",0)+by.get("basic",0)+by.get("pro",0)+by.get("vip",0))
    return jsonify({"tally": {
        "free": by.get("free",0), "basic": by.get("basic",0), "pro": by.get("pro",0),
        "vip": by.get("vip",0), "all": all_total
    }})

@bp.route("/live/start")
@_require_admin
def live_start():
    ENGINE.start()
    return jsonify({"ok":True, "status": ENGINE.status()})

@bp.route("/live/stop")
@_require_admin
def live_stop():
    ENGINE.stop()
    return jsonify({"ok":True, "status": ENGINE.status()})

@bp.route("/live/debug/on")
@_require_admin
def live_dbg_on():
    ENGINE.set_debug(True)
    return jsonify({"ok":True, "status": ENGINE.status()})

@bp.route("/live/debug/off")
@_require_admin
def live_dbg_off():
    ENGINE.set_debug(False)
    return jsonify({"ok":True, "status": ENGINE.status()})

# Compact API used by the new live page JS
@bp.route("/api/status")
def api_status():
    s = ENGINE.status()
    configured = {k: bool(v) for k, v in (TIER_TO_CHAT or {}).items()} if "TIER_TO_CHAT" in globals() else {}
    return jsonify({
        "running": s.get("running"),
        "debug": s.get("debug"),
        "loop_sleep": s.get("loop_sleep"),
        "day": s.get("day"),
        "tallies": s.get("tallies") or s.get("tally") or {},
        "caps": DAILY_CAPS,
        "configured_chats": {
            "free": configured.get("free", False),
            "basic": configured.get("basic", False),
            "pro": configured.get("pro", False),
            "vip": configured.get("vip", False),
        },
        "last_send_result": s.get("last_send_result") or {},
    })

@bp.route("/api/send", methods=["POST"])
@_require_admin
def api_send():
    data = request.get_json(force=True, silent=True) or {}
    tier = data.get("tier", "vip")
    text = data.get("text", "").strip()
    if not text:
        return jsonify({"ok": False, "error":"empty text"}), 400
    res = ENGINE.send_to_tier(tier, text)
    return jsonify({"ok":bool(res.get("ok")), "result":res, "status":ENGINE.status()})

@bp.route("/api/test/vip", methods=["POST"])
@_require_admin
def api_test_vip():
    data = request.get_json(force=True, silent=True) or {}
    text = data.get("text") or "🧪 VIP test"
    res = ENGINE.send_to_tier("vip", text)
    return jsonify({"ok":bool(res.get("ok")), "result":res, "status":ENGINE.status()})

@bp.route("/api/check_bot")
def api_check_bot():
    info = {
        "getMe": {"ok": bool(BOT_TOKEN), "result":{"username":"", "id": ""}},
        "configured_chats": TIER_TO_CHAT,
    }
    try:
        # lightweight: reuse live_engine diag if you have it; else echo env state
        info["ok"] = bool(BOT_TOKEN)
    except Exception as e:
        info["ok"] = False
        info["error"] = str(e)
    return jsonify(info)

# Manual test to all configured chats
@bp.route("/telegram/test", methods=["POST"])
@_require_admin
def telegram_test():
    ok, info = TG_TEST()
    return jsonify({"ok": ok, "info": info})

@bp.route("/telegram/diag")
def telegram_diag():
    return jsonify({
        "ok": bool(BOT_TOKEN),
        "token_present": bool(BOT_TOKEN),
        "configured_chats": TIER_TO_CHAT,
        "daily_caps": DAILY_CAPS
    })
