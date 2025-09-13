# routes.py — Flask Blueprint: dashboard UI, backtest, Telegram, live engine
import os, io, json
from datetime import datetime
from typing import Dict, Any

from flask import (
    Blueprint, render_template, request, redirect, url_for, flash,
    jsonify
)

from utils import (
    get_config, set_config, within_window,
    convert_po_to_deriv, fetch_deriv_history, load_csv,
    backtest_run, plot_signals, TZ
)
from live_engine import ENGINE  # rely only on ENGINE public API

bp = Blueprint("dashboard", __name__, template_folder="templates", static_folder="static")

# ---------------------------- Indicator specs (forms) -------------------------
# Keys must match utils.compute_indicators()
INDICATOR_SPECS: Dict[str, Dict[str, Any]] = {
    # Moving averages (overlays)
    "SMA":      {"title": "Simple MA",       "panel": "overlay", "fields": {"period": 50}},
    "EMA":      {"title": "Exponential MA",  "panel": "overlay", "fields": {"period": 20}},
    "WMA":      {"title": "Weighted MA",     "panel": "overlay", "fields": {"period": 20}},
    "SMMA":     {"title": "Smoothed MA",     "panel": "overlay", "fields": {"period": 20}},
    "TMA":      {"title": "Triangular MA",   "panel": "overlay", "fields": {"period": 20}},

    # Bands & channels (overlays)
    "BOLL":     {"title": "Bollinger Bands", "panel": "overlay", "fields": {"period": 20, "mult": 2}},
    "KELTNER":  {"title": "Keltner Channel", "panel": "overlay", "fields": {"period": 20, "mult": 2}},
    "DONCHIAN": {"title": "Donchian Channels","panel":"overlay", "fields": {"period": 20}},
    "ENVELOPES":{"title": "Envelopes",       "panel": "overlay", "fields": {"period": 20, "pct": 2.0}},

    # Trend overlays
    "ICHIMOKU": {"title": "Ichimoku Kinko Hyo", "panel":"overlay", "fields": {}},
    "PSAR":     {"title": "Parabolic SAR",      "panel":"overlay", "fields": {"step": 0.02, "max": 0.2}},
    "SUPERTREND":{"title":"Supertrend",         "panel":"overlay", "fields": {"period": 10, "mult": 3}},
    "ALLIGATOR":{"title":"Alligator",           "panel":"overlay",
                 "fields":{"jaw":13,"teeth":8,"lips":5,"jaw_shift":8,"teeth_shift":5,"lips_shift":3}},
    "FRACTAL":  {"title":"Fractal Chaos Bands", "panel":"overlay", "fields":{"lookback":2,"smooth":5}},

    # Oscillators (their own panels)
    "RSI":      {"title": "RSI",               "panel":"osc", "fields":{"period":14}},
    "STOCH":    {"title": "Stochastic",        "panel":"osc", "fields":{"k":14,"d":3}},
    "ATR":      {"title": "ATR",               "panel":"osc", "fields":{"period":14}},
    "ADX":      {"title": "ADX/+DI/-DI",       "panel":"osc", "fields":{"period":14}},
    "CCI":      {"title": "CCI",               "panel":"osc", "fields":{"period":20}},
    "MOMENTUM": {"title": "Momentum",          "panel":"osc", "fields":{"period":10}},
    "ROC":      {"title": "Rate of Change",    "panel":"osc", "fields":{"period":10}},
    "WILLR":    {"title": "Williams %R",       "panel":"osc", "fields":{"period":14}},
    "VORTEX":   {"title": "Vortex",            "panel":"osc", "fields":{"period":14}},
    "MACD":     {"title": "MACD",              "panel":"osc", "fields":{"fast":12,"slow":26,"signal":9}},
    "AO":       {"title": "Awesome Osc.",      "panel":"osc", "fields":{}},
    "AC":       {"title": "Accelerator Osc.",  "panel":"osc", "fields":{}},
    "BEARS":    {"title": "Bears Power",       "panel":"osc", "fields":{"period":13}},
    "BULLS":    {"title": "Bulls Power",       "panel":"osc", "fields":{"period":13}},
    "DEMARKER": {"title": "DeMarker",          "panel":"osc", "fields":{"period":14}},
    "OSMA":     {"title": "OsMA",              "panel":"osc", "fields":{}},
    "ZIGZAG":   {"title": "ZigZag",            "panel":"osc", "fields":{"pct":1.0}},
    "AROON":    {"title": "Aroon",             "panel":"osc", "fields":{"period":14}},
    "STC":      {"title": "Schaff Trend Cycle","panel":"osc", "fields":{"fast":23,"slow":50,"cycle":10}},
}

# ---------------------------- Helpers ----------------------------------------
def _load_form_config() -> Dict[str, Any]:
    """Merge saved config + form inputs, then persist."""
    cfg = get_config()

    # Window & general
    cfg["window_start"] = request.form.get("window_start", cfg.get("window_start", "08:00"))
    cfg["window_end"]   = request.form.get("window_end",   cfg.get("window_end",   "17:00"))
    cfg["use_deriv_fetch"] = bool(request.form.get("use_deriv_fetch", ""))
    cfg["deriv_count"]  = int(request.form.get("deriv_count", cfg.get("deriv_count", 300)))

    # Strategy toggles
    for key in ["BASE","TREND","CHOP","CUSTOM1","CUSTOM2","CUSTOM3"]:
        cfg[key] = bool(request.form.get(f"strategy_{key}", "1" if cfg.get(key) else ""))

    # Custom rule text boxes
    cfg["custom1_rules"] = request.form.get("custom1_rules", cfg.get("custom1_rules",""))
    cfg["custom2_rules"] = request.form.get("custom2_rules", cfg.get("custom2_rules",""))
    cfg["custom3_rules"] = request.form.get("custom3_rules", cfg.get("custom3_rules",""))

    # Indicators: toggles + params
    ind = cfg.get("indicators") or {}
    for key, spec in INDICATOR_SPECS.items():
        enabled = bool(request.form.get(f"ind_{key}", "1" if (ind.get(key,{}).get("enabled")) else ""))
        params = {}
        for fname, default in spec.get("fields", {}).items():
            v = request.form.get(f"ind_{key}_{fname}")
            if v is None or v == "":
                params[fname] = ind.get(key, {}).get(fname, default)
            else:
                try:
                    if isinstance(default, int):   params[fname] = int(float(v))
                    elif isinstance(default, float): params[fname] = float(v)
                    else:                           params[fname] = v
                except Exception:
                    params[fname] = default
        ind[key] = {"enabled": enabled, **params}
    cfg["indicators"] = ind

    set_config(cfg)
    return cfg

def _default_ctx() -> Dict[str, Any]:
    cfg = get_config()
    if "indicators" not in cfg:
        cfg["indicators"] = {k: {"enabled": False, **spec.get("fields", {})} for k, spec in INDICATOR_SPECS.items()}
        set_config(cfg)
    status = ENGINE.status()
    return {
        "cfg": cfg,
        "ind_specs": INDICATOR_SPECS,
        "engine": status,
        "now_local": datetime.now(TZ).strftime("%b %d, %Y %H:%M"),
        "window_ok": within_window(cfg),
    }

# ---------------------------- Pages ------------------------------------------
@bp.route("/")
def home():
    return render_template("index.html", **_default_ctx())

@bp.route("/dashboard", methods=["GET", "POST"])
def view():
    if request.method == "POST":
        _load_form_config()
        flash("Saved.", "success")
        return redirect(url_for("dashboard.view"))
    return render_template("dashboard.html", **_default_ctx())

# Keep a placeholder to match old template links and avoid 500s
@bp.route("/deriv_fetch", methods=["POST"])
def deriv_fetch():
    return jsonify({"ok": True})

# ---------------------------- Backtest & Screenshot --------------------------
@bp.route("/backtest", methods=["POST"])
def backtest():
    cfg = _load_form_config()
    ind_cfg = cfg.get("indicators") or {}

    symbols_raw = (request.form.get("symbols") or "").replace(";", ",")
    symbols = [s.strip() for s in symbols_raw.split(",") if s.strip()]
    if not symbols:
        return jsonify({"ok": False, "error": "No symbols provided."}), 400

    tf = request.form.get("tf", "M1").upper()
    expiry = request.form.get("expiry", "5m")
    use_deriv = bool(request.form.get("use_deriv_fetch", "1" if cfg.get("use_deriv_fetch") else ""))
    target = max(60, int(request.form.get("target", cfg.get("deriv_count", 300))))

    tf_map = {"M1":60,"M2":120,"M3":180,"M5":300,"M10":600,"M15":900,"M30":1800,"H1":3600,"H4":14400,"D1":86400}
    gran = tf_map.get(tf, 60)

    if bool(request.form.get("convert_po_deriv", "")):
        symbols = convert_po_to_deriv(symbols)

    df = None
    if "csv" in request.files and request.files["csv"].filename:
        try:
            df = load_csv(request.files["csv"])
        except Exception as e:
            return jsonify({"ok": False, "error": f"CSV error: {e}"}), 400
    else:
        errs = []
        for sym in symbols:
            try:
                if not use_deriv:
                    raise RuntimeError("Deriv fetch disabled; upload CSV or enable Deriv fetch.")
                df = fetch_deriv_history(sym, granularity_sec=gran, count=target)
                if df is not None and not df.empty:
                    break
            except Exception as e:
                errs.append(str(e))
        if df is None or df.empty:
            return jsonify({"ok": False, "error": "Deriv fetch failed. " + "; ".join(errs)[:800]}), 400

    strategy = request.form.get("strategy", "CUSTOM1").upper()
    signals, stats = backtest_run(df, strategy, ind_cfg, expiry)
    png_name = plot_signals(df, signals, ind_cfg, strategy=strategy, tf=tf, expiry=expiry)

    return jsonify({
        "ok": True,
        "plot": f"/static/plots/{png_name}",
        "signals": [{"ts": s["index"].isoformat(), "dir": s["direction"], "exp": s["expiry_idx"].isoformat()} for s in signals],
        "stats": stats
    })

# ---------------------------- Telegram ---------------------------------------
@bp.route("/telegram/send", methods=["POST"])
def telegram_send():
    data = request.form.to_dict() or {}
    text = (data.get("text") or "").strip()
    tier = (data.get("tier") or "all").lower()
    if not text:
        return jsonify({"ok": False, "error": "Text required"}), 400

    tiers = ["free","basic","pro","vip"] if tier == "all" else [tier]
    results = {t: ENGINE.send_to_tier(t, text) for t in tiers}
    return jsonify({"ok": True, "results": results, "status": ENGINE.status()})

@bp.route("/telegram/test", methods=["POST"])
def telegram_test():
    msg = request.form.get("text") or "🧪 Test from dashboard to all tiers"
    results = {t: ENGINE.send_to_tier(t, f"{msg} ({t.upper()})") for t in ["free","basic","pro","vip"]}
    return jsonify({"ok": True, "results": results})

# ---------------------------- Live Engine ------------------------------------
@bp.route("/live/start", methods=["POST"])
def live_start():
    ok, msg = ENGINE.start()
    return jsonify({"ok": ok, "msg": msg, "status": ENGINE.status()})

@bp.route("/live/stop", methods=["POST"])
def live_stop():
    ok, msg = ENGINE.stop()
    return jsonify({"ok": ok, "msg": msg, "status": ENGINE.status()})

@bp.route("/live/status")
def live_status():
    return jsonify(ENGINE.status())

@bp.route("/live/tally")
def live_tally():
    st = ENGINE.status()
    return jsonify(st.get("tally") or st.get("tallies") or {})

# ---------------------------- API echoes / debug -----------------------------
@bp.route("/api/indicators")
def api_indicators():
    return jsonify(INDICATOR_SPECS)

@bp.route("/api/status")
def api_status_dup():
    return jsonify(ENGINE.status())
