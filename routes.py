# routes.py — Dashboard, config, backtest, Telegram + Live APIs
from __future__ import annotations
import os, io, json, csv
from datetime import datetime, timezone
from typing import Dict, Any, List

from flask import (
    Blueprint, render_template, request, redirect, url_for,
    session, jsonify, flash, send_file
)

# utils you have
from utils import (
    TZ, get_config, set_config, within_window, convert_po_to_deriv,
    load_csv, fetch_deriv_history, compute_indicators,
    backtest_run, plot_signals
)

# -------- tolerate builds where tg_test may be missing ----------
try:
    from live_engine import ENGINE, tg_test, TIER_TO_CHAT, DAILY_CAPS
except Exception:
    from live_engine import ENGINE
    TIER_TO_CHAT = {"free": None, "basic": None, "pro": None, "vip": None}
    DAILY_CAPS = {}
    def tg_test():
        return False, {"ok": False, "error": "tg_test not available in this build"}
# -----------------------------------------------------------------

bp = Blueprint("dashboard", __name__)

ADMIN_PASS = os.getenv("ADMIN_PASSWORD", "admin")

# ---------------------------- Indicator specs (forms) -------------------------
# Keys must match names in utils.compute_indicators()
INDICATOR_SPECS: Dict[str, Dict[str, Any]] = {
    # MAs (overlays)
    "SMA":      {"title": "Simple MA",          "panel": "overlay", "fields": {"period": 50}},
    "EMA":      {"title": "Exponential MA",     "panel": "overlay", "fields": {"period": 20}},
    "WMA":      {"title": "Weighted MA",        "panel": "overlay", "fields": {"period": 20}},
    "SMMA":     {"title": "Smoothed MA",        "panel": "overlay", "fields": {"period": 20}},
    "TMA":      {"title": "Triangular MA",      "panel": "overlay", "fields": {"period": 20}},

    # Bands / Channels (overlays)
    "BOLL":     {"title": "Bollinger Bands",    "panel": "overlay", "fields": {"period": 20, "mult": 2}},
    "KELTNER":  {"title": "Keltner Channel",    "panel": "overlay", "fields": {"period": 20, "mult": 2}},
    "DONCHIAN": {"title": "Donchian Channels",  "panel": "overlay", "fields": {"period": 20}},
    "ENVELOPES":{"title": "Envelopes",          "panel": "overlay", "fields": {"period": 20, "pct": 2.0}},

    # Ichimoku / PSAR / Supertrend (overlays)
    "ICHIMOKU": {"title": "Ichimoku",           "panel":"overlay",  "fields": {}},
    "PSAR":     {"title": "Parabolic SAR",      "panel":"overlay",  "fields": {"step": 0.02, "max": 0.2}},
    "SUPERTREND":{"title":"Supertrend",         "panel":"overlay",  "fields": {"period": 10, "mult": 3}},

    # NEW overlays
    "ALLIGATOR":{"title":"Alligator",           "panel":"overlay",
                 "fields":{"jaw":13,"teeth":8,"lips":5,"jaw_shift":8,"teeth_shift":5,"lips_shift":3}},
    "FRACTAL":  {"title":"Fractal Chaos Bands", "panel":"overlay",  "fields":{"lookback":2,"smooth":5}},

    # Oscillators (separate panels)
    "RSI":      {"title":"RSI",                 "panel":"osc",      "fields":{"period":14}},
    "STOCH":    {"title":"Stochastic",          "panel":"osc",      "fields":{"k":14,"d":3}},
    "ATR":      {"title":"ATR",                 "panel":"osc",      "fields":{"period":14}},
    "ADX":      {"title":"ADX/+DI/-DI",         "panel":"osc",      "fields":{"period":14}},
    "CCI":      {"title":"CCI",                 "panel":"osc",      "fields":{"period":20}},
    "MOMENTUM": {"title":"Momentum",            "panel":"osc",      "fields":{"period":10}},
    "ROC":      {"title":"Rate of Change",      "panel":"osc",      "fields":{"period":10}},
    "WILLR":    {"title":"Williams %R",         "panel":"osc",      "fields":{"period":14}},
    "VORTEX":   {"title":"Vortex",              "panel":"osc",      "fields":{"period":14}},
    "MACD":     {"title":"MACD",                "panel":"osc",      "fields":{"fast":12,"slow":26,"signal":9}},
    "AO":       {"title":"Awesome Osc.",        "panel":"osc",      "fields":{}},
    "AC":       {"title":"Accelerator Osc.",    "panel":"osc",      "fields":{}},
    "BEARS":    {"title":"Bears Power",         "panel":"osc",      "fields":{"period":13}},
    "BULLS":    {"title":"Bulls Power",         "panel":"osc",      "fields":{"period":13}},
    "DEMARKER": {"title":"DeMarker",            "panel":"osc",      "fields":{"period":14}},
    "OSMA":     {"title":"OsMA",                "panel":"osc",      "fields":{}},
    "ZIGZAG":   {"title":"ZigZag",              "panel":"osc",      "fields":{"pct":1.0}},

    # NEW oscillators
    "AROON":    {"title":"Aroon",               "panel":"osc",      "fields":{"period":14}},
    "STC":      {"title":"Schaff Trend Cycle",  "panel":"osc",      "fields":{"fast":23,"slow":50,"cycle":10}},
}

TF_TO_GRAN = {"M1":60,"M2":120,"M3":180,"M5":300,"M10":600,"M15":900,"M30":1800,"H1":3600,"H4":14400,"D1":86400}

# ---------------- Helpers ----------------
def _ctx_base(view: str="dashboard") -> Dict[str, Any]:
    cfg = get_config() or {}
    now_local = datetime.now(TZ).strftime("%Y-%m-%d %H:%M:%S")
    return {
        "view": view,
        "tz": str(TZ),
        "now": now_local,
        "within": within_window(cfg),
        "session": {"admin": session.get("admin", False)},
    }

def _ensure_cfg_defaults(cfg: Dict[str, Any]) -> Dict[str, Any]:
    cfg = dict(cfg or {})
    # Window defaults (Mon–Fri enforced by within_window())
    cfg.setdefault("window", {"start":"08:00","end":"17:00","timezone":str(TZ)})
    # Live defaults
    cfg.setdefault("live_tf", "M1")
    cfg.setdefault("live_expiry", "5m")
    cfg.setdefault("deriv_count", 300)
    cfg.setdefault("use_deriv_fetch", True)
    # Indicator defaults: all off, but fields present
    if "indicators" not in cfg:
        cfg["indicators"] = {k: {"enabled": False, **spec["fields"]} for k, spec in INDICATOR_SPECS.items()}
    # Strategies
    cfg.setdefault("strategies", {
        "BASE":{"enabled": True},
        "TREND":{"enabled": False},
        "CHOP":{"enabled": False},
        "CUSTOM1":{"enabled": False},
        "CUSTOM2":{"enabled": False},
        "CUSTOM3":{"enabled": False},
    })
    # Custom rule text boxes
    cfg.setdefault("custom1_rules","")
    cfg.setdefault("custom2_rules","")
    cfg.setdefault("custom3_rules","")
    # Symbol list
    cfg.setdefault("symbols_raw", ["frxEURUSD","frxGBPUSD"])
    return cfg

def _admin_required(fn):
    def wrap(*a, **kw):
        if not session.get("admin"):
            return redirect(url_for("dashboard.login"))
        return fn(*a, **kw)
    wrap.__name__ = fn.__name__
    return wrap

def _default_ctx() -> Dict[str, Any]:
    cfg = _ensure_cfg_defaults(get_config())
    status = ENGINE.status()
    return {
        "cfg": cfg,
        "ind_specs": INDICATOR_SPECS,
        "engine": status,
        "now_local": datetime.now(TZ).strftime("%b %d, %Y %H:%M"),
        "window_ok": within_window(cfg),
        "tier_to_chat": TIER_TO_CHAT,
        "daily_caps": DAILY_CAPS,
    }

def _save_indicators_from_form(cfg: Dict[str, Any], form) -> None:
    ind = cfg.get("indicators") or {}
    for key, spec in INDICATOR_SPECS.items():
        enabled = bool(form.get(f"ind_{key}"))
        params = {}
        for f_name, default in spec.get("fields", {}).items():
            v = form.get(f"ind_{key}_{f_name}")
            if v is None or v == "":
                params[f_name] = ind.get(key, {}).get(f_name, default)
            else:
                try:
                    if isinstance(default, int):
                        params[f_name] = int(float(v))
                    elif isinstance(default, float):
                        params[f_name] = float(v)
                    else:
                        params[f_name] = v
                except Exception:
                    params[f_name] = default
        ind[key] = {"enabled": enabled, **params}
    cfg["indicators"] = ind

# ---------------- Basic pages ----------------
@bp.route("/")
def root():
    # Avoid TemplateNotFound for index.html: send to dashboard
    return redirect(url_for("dashboard.view"))

@bp.route("/login", methods=["GET","POST"])
def login():
    if request.method == "POST":
        pw = request.form.get("password","")
        if pw == ADMIN_PASS:
            session["admin"] = True
            return redirect(url_for("dashboard.view"))
        flash("Wrong password", "error")
    return render_template("dashboard.html", **_default_ctx() | {"view":"login"})

@bp.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("dashboard.root"))

@bp.route("/dashboard", methods=["GET","POST"])
@_admin_required
def view():
    if request.method == "POST":
        cfg = _ensure_cfg_defaults(get_config())

        # Window + live defaults
        w = cfg.get("window", {})
        w["start"] = request.form.get("window_start", w.get("start","08:00"))
        w["end"]   = request.form.get("window_end",   w.get("end","17:00"))
        w["timezone"] = request.form.get("window_tz", str(TZ))
        cfg["window"] = w
        cfg["live_tf"] = request.form.get("live_tf", cfg.get("live_tf","M1")).upper()
        cfg["live_expiry"] = request.form.get("live_expiry", cfg.get("live_expiry","5m"))

        # Symbols (textarea)
        syms_text = (request.form.get("symbols_text","") or "").replace(",", " ")
        syms = [x for x in syms_text.split() if x]
        if request.form.get("convert_po"):
            syms = convert_po_to_deriv(syms)
        if syms:
            cfg["symbols_raw"] = syms

        # Deriv fetch defaults
        cfg["use_deriv_fetch"] = bool(request.form.get("use_deriv_fetch"))
        try:
            cfg["deriv_count"] = int(request.form.get("deriv_count", cfg.get("deriv_count", 300)))
        except Exception:
            cfg["deriv_count"] = 300

        # Strategies
        st = cfg.get("strategies", {})
        for name in ["BASE","TREND","CHOP","CUSTOM1","CUSTOM2","CUSTOM3"]:
            st[name] = {"enabled": bool(request.form.get(f"s_{name}"))}
        cfg["strategies"] = st

        # Custom rule text boxes
        cfg["custom1_rules"] = request.form.get("custom1_rules", cfg.get("custom1_rules",""))
        cfg["custom2_rules"] = request.form.get("custom2_rules", cfg.get("custom2_rules",""))
        cfg["custom3_rules"] = request.form.get("custom3_rules", cfg.get("custom3_rules",""))

        # Indicators
        _save_indicators_from_form(cfg, request.form)

        set_config(cfg)
        flash("Saved.", "ok")
        return redirect(url_for("dashboard.view"))

    return render_template("dashboard.html", **_default_ctx())

# ---------------- Backtest ----------------
@bp.route("/backtest", methods=["POST"])
@_admin_required
def backtest():
    cfg = _ensure_cfg_defaults(get_config())
    ind_cfg = cfg.get("indicators") or {}

    # Symbols
    symbols_raw = (request.form.get("bt_symbols") or "").replace(";", ",")
    symbols = [s.strip() for s in symbols_raw.split(",") if s.strip()]
    if not symbols:
        # fallback to saved
        symbols = cfg.get("symbols_raw", [])
    if request.form.get("convert_po_bt"):
        symbols = convert_po_to_deriv(symbols)

    tf = request.form.get("bt_tf", cfg.get("live_tf","M1")).upper()
    expiry = request.form.get("bt_expiry", cfg.get("live_expiry","5m"))
    use_deriv = bool(request.form.get("use_deriv_fetch", "1" if cfg.get("use_deriv_fetch") else ""))
    target = max(60, int(request.form.get("bt_count", cfg.get("deriv_count", 300))))
    gran = TF_TO_GRAN.get(tf, 60)

    df = None
    # CSV overrides network
    if "bt_csv" in request.files and request.files["bt_csv"].filename:
        try:
            df = load_csv(io.BytesIO(request.files["bt_csv"].read()))
        except Exception as e:
            return jsonify({"ok": False, "error": f"CSV error: {e}"}), 400
    else:
        if not use_deriv:
            return jsonify({"ok": False, "error": "Deriv fetch disabled; upload CSV or enable Deriv fetch."}), 400
        errs = []
        for sym in symbols:
            try:
                df = fetch_deriv_history(sym, granularity_sec=gran, count=target)
                if df is not None and not df.empty:
                    break
            except Exception as e:
                errs.append(str(e))
        if df is None or df.empty:
            return jsonify({"ok": False, "error": "Deriv fetch failed. " + "; ".join(errs)[:900]}), 400

    # Run strategy & plot
    strategy = request.form.get("bt_strategy", "BASE").upper()
    sigs, stats = backtest_run(df, strategy, ind_cfg, expiry)
    png_name = plot_signals(df, sigs, ind_cfg, strategy=strategy, tf=tf, expiry=expiry)

    # Store brief state for page right-side
    session["bt_state"] = {
        "tf": tf, "expiry": expiry, "strategy": strategy,
        "plot_name": png_name,
        "summary": f"{stats.get('wins',0)}W / {stats.get('loss',0)}L / {stats.get('draw',0)}D — WR {stats.get('win_rate',0):.1f}%"
    }

    return jsonify({
        "ok": True,
        "plot": f"/plots/{png_name}",
        "signals": [{"ts": i["index"].isoformat(), "dir": i["direction"], "exp": i["expiry_idx"].isoformat()} for i in sigs],
        "stats": stats
    })

@bp.route("/backtest/last.json")
@_admin_required
def backtest_last_json():
    return jsonify(session.get("bt_state") or {})

@bp.route("/plots/<name>")
def plot_file(name: str):
    # static helper to serve generated PNGs
    return send_file(os.path.join("static","plots", name))

# ---------------- Telegram ----------------
@bp.route("/telegram/send", methods=["POST"])
@_admin_required
def telegram_send():
    data = request.form.to_dict() or {}
    text = (data.get("text") or "").strip()
    tier = (data.get("tier") or "all").lower()
    if not text:
        return jsonify({"ok": False, "error": "Text required"}), 400
    tiers = ["free","basic","pro","vip"] if tier == "all" else [tier]
    results = {}
    for t in tiers:
        results[t] = ENGINE.send_to_tier(t, text)
    return jsonify({"ok": True, "results": results, "status": ENGINE.status()})

@bp.route("/telegram/test", methods=["POST"])
@_admin_required
def telegram_test():
    ok, diag = tg_test()
    return jsonify({"ok": ok, "diag": diag})

@bp.route("/api/check_bot")
def api_check_bot():
    ok, diag = tg_test()
    # include configured chat presence
    diag["configured_chats"] = {k: bool(v) for k, v in TIER_TO_CHAT.items()}
    return jsonify(diag)

# ---------------- Live Engine controls ----------------
@bp.route("/live/start", methods=["POST"])
@_admin_required
def live_start():
    ok, msg = ENGINE.start()
    return jsonify({"ok": ok, "msg": msg, "status": ENGINE.status()})

@bp.route("/live/stop", methods=["POST"])
@_admin_required
def live_stop():
    ok, msg = ENGINE.stop()
    return jsonify({"ok": ok, "msg": msg, "status": ENGINE.status()})

@bp.route("/live/debug/on", methods=["POST"])
@_admin_required
def live_debug_on():
    ENGINE.set_debug(True); return jsonify({"ok": True})

@bp.route("/live/debug/off", methods=["POST"])
@_admin_required
def live_debug_off():
    ENGINE.set_debug(False); return jsonify({"ok": True})

@bp.route("/live/status")
def live_status():
    return jsonify(ENGINE.status())

@bp.route("/live/tally")
def live_tally():
    st = ENGINE.status()
    tally = st.get("tally") or st.get("tallies") or {}
    return jsonify(tally)

# ---------------- Quick forms (symbols / indicators split endpoints) ----------
@bp.route("/symbols/update", methods=["POST"])
@_admin_required
def update_symbols():
    cfg = _ensure_cfg_defaults(get_config())
    text = request.form.get("symbols_text","").replace(",", " ")
    syms = [x for x in text.split() if x]
    if request.form.get("convert_po"):
        syms = convert_po_to_deriv(syms)
    cfg["symbols_raw"] = syms or cfg.get("symbols_raw", [])
    set_config(cfg)
    flash("Symbols updated", "ok")
    return redirect(url_for("dashboard.view"))

@bp.route("/indicators/update", methods=["POST"])
@_admin_required
def update_indicators():
    cfg = _ensure_cfg_defaults(get_config())
    _save_indicators_from_form(cfg, request.form)
    set_config(cfg)
    flash("Indicators updated", "ok")
    return redirect(url_for("dashboard.view"))

@bp.route("/strategies/update", methods=["POST"])
@_admin_required
def update_strategies():
    cfg = _ensure_cfg_defaults(get_config())
    st = cfg.get("strategies", {})
    for name in ["BASE","TREND","CHOP","CUSTOM1","CUSTOM2","CUSTOM3"]:
        st[name] = {"enabled": bool(request.form.get(f"s_{name}"))}
    cfg["strategies"] = st
    set_config(cfg)
    flash("Strategies toggles updated", "ok")
    return redirect(url_for("dashboard.view"))

@bp.route("/custom/update", methods=["POST"])
@_admin_required
def update_custom():
    cfg = _ensure_cfg_defaults(get_config())
    cfg["custom1_rules"] = request.form.get("custom1_rules","")
    cfg["custom2_rules"] = request.form.get("custom2_rules","")
    cfg["custom3_rules"] = request.form.get("custom3_rules","")
    set_config(cfg)
    flash("Custom rules saved", "ok")
    return redirect(url_for("dashboard.view"))

# ---------------- API echoes / debug -----------------------------
@bp.route("/api/indicators")
def api_indicators():
    return jsonify(INDICATOR_SPECS)

@bp.route("/api/status")
def api_status_dup():
    return jsonify(ENGINE.status())

# ---------------- Health ----------------
@bp.route("/_up")
def up_check():
    return jsonify({"ok": True, "ts": datetime.now(timezone.utc).isoformat()+"Z"})
