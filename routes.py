# routes.py — Dashboard, config, backtest, Telegram + Live APIs (pairs selector + auth + /api/start|stop)
from __future__ import annotations
import os, io, json
from datetime import datetime, timezone
from typing import Dict, Any

from flask import (
    Blueprint, render_template, request, redirect, url_for, flash,
    jsonify, session, send_file
)

from utils import (
    get_config, set_config, within_window,
    convert_po_to_deriv, fetch_deriv_history, load_csv,
    backtest_run, plot_signals, TZ
)

# Rely only on ENGINE public API
try:
    from live_engine import ENGINE
except Exception:
    class _Dummy:
        def start(self): return False, "ENGINE missing"
        def stop(self): return False, "ENGINE missing"
        def set_debug(self, *_a, **_k): ...
        def status(self): return {"ok": False, "error": "ENGINE missing"}
        def send_to_tier(self, tier, text): return {"ok": False, "error": "ENGINE missing"}
    ENGINE = _Dummy()

bp = Blueprint("dashboard", __name__, template_folder="templates", static_folder="static")

ADMIN_PASS = os.getenv("ADMIN_PASSWORD", "admin")

# ---------------------------- Indicator specs (forms) -------------------------
# Keys must match the names used in utils.compute_indicators().
INDICATOR_SPECS: Dict[str, Dict[str, Any]] = {
    # MAs (overlays)
    "SMA":      {"title": "Simple MA",       "panel": "overlay",   "fields": {"period": 50}},
    "EMA":      {"title": "Exponential MA",  "panel": "overlay",   "fields": {"period": 20}},
    "WMA":      {"title": "Weighted MA",     "panel": "overlay",   "fields": {"period": 20}},
    "SMMA":     {"title": "Smoothed MA",     "panel": "overlay",   "fields": {"period": 20}},
    "TMA":      {"title": "Triangular MA",   "panel": "overlay",   "fields": {"period": 20}},

    # Bands / Channels (overlays)
    "BOLL":     {"title": "Bollinger Bands", "panel": "overlay",   "fields": {"period": 20, "mult": 2}},
    "KELTNER":  {"title": "Keltner Channel", "panel": "overlay",   "fields": {"period": 20, "mult": 2}},
    "DONCHIAN": {"title": "Donchian Channels","panel": "overlay",  "fields": {"period": 20}},
    "ENVELOPES":{"title": "Envelopes",       "panel": "overlay",   "fields": {"period": 20, "pct": 2.0}},

    # Ichimoku / PSAR / Supertrend (overlays)
    "ICHIMOKU": {"title": "Ichimoku Kinko Hyo","panel":"overlay",  "fields": {}},
    "PSAR":     {"title": "Parabolic SAR",   "panel": "overlay",   "fields": {"step": 0.02, "max": 0.2}},
    "SUPERTREND":{"title": "Supertrend",     "panel": "overlay",   "fields": {"period": 10, "mult": 3}},

    # New overlays
    "ALLIGATOR": {"title": "Alligator", "panel": "overlay",
                  "fields": {"jaw": 13, "teeth": 8, "lips": 5, "jaw_shift": 8, "teeth_shift": 5, "lips_shift": 3}},
    "FRACTAL":   {"title": "Fractal Chaos Bands", "panel": "overlay",
                  "fields": {"lookback": 2, "smooth": 5}},

    # Oscillators (separate panels)
    "RSI":      {"title": "RSI",            "panel": "osc",       "fields": {"period": 14}},
    "STOCH":    {"title": "Stochastic",     "panel": "osc",       "fields": {"k": 14, "d": 3}},
    "ATR":      {"title": "ATR",            "panel": "osc",       "fields": {"period": 14}},
    "ADX":      {"title": "ADX/+DI/-DI",    "panel": "osc",       "fields": {"period": 14}},
    "CCI":      {"title": "CCI",            "panel": "osc",       "fields": {"period": 20}},
    "MOMENTUM": {"title": "Momentum",       "panel": "osc",       "fields": {"period": 10}},
    "ROC":      {"title": "Rate of Change", "panel": "osc",       "fields": {"period": 10}},
    "WILLR":    {"title": "Williams %R",    "panel": "osc",       "fields": {"period": 14}},
    "VORTEX":   {"title": "Vortex",         "panel": "osc",       "fields": {"period": 14}},
    "MACD":     {"title": "MACD",           "panel": "osc",       "fields": {"fast": 12, "slow": 26, "signal": 9}},
    "AO":       {"title": "Awesome Osc.",   "panel": "osc",       "fields": {}},
    "AC":       {"title": "Accelerator Osc.", "panel":"osc",      "fields": {}},
    "BEARS":    {"title": "Bears Power",    "panel": "osc",       "fields": {"period": 13}},
    "BULLS":    {"title": "Bulls Power",    "panel": "osc",       "fields": {"period": 13}},
    "DEMARKER": {"title": "DeMarker",       "panel": "osc",       "fields": {"period": 14}},
    "OSMA":     {"title": "OsMA",           "panel": "osc",       "fields": {}},
    "ZIGZAG":   {"title": "ZigZag",         "panel": "osc",       "fields": {"pct": 1.0}},

    # New oscillators
    "AROON":    {"title": "Aroon",          "panel": "osc",       "fields": {"period": 14}},
    "STC":      {"title": "Schaff Trend Cycle", "panel": "osc",   "fields": {"fast": 23, "slow": 50, "cycle": 10}},
}

TF_TO_GRAN = {"M1":60,"M2":120,"M3":180,"M5":300,"M10":600,"M15":900,"M30":1800,"H1":3600,"H4":14400,"D1":86400}

# ---------------- Helpers ----------------
def _admin_required(fn):
    def wrap(*a, **kw):
        if not session.get("admin"):
            return redirect(url_for("dashboard.login"))
        return fn(*a, **kw)
    wrap.__name__ = fn.__name__
    return wrap

def _merge_indicator_form(cfg: dict, form) -> dict:
    ind = cfg.get("indicators") or {}
    for key, spec in INDICATOR_SPECS.items():
        enabled = bool(form.get(f"ind_{key}", "1" if (ind.get(key,{}).get("enabled")) else ""))
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
    return cfg

def _ensure_cfg_defaults(cfg: dict|None) -> dict:
    cfg = dict(cfg or {})
    # Window defaults; Mon–Fri enforced by utils.within_window
    cfg.setdefault("window", {"start":"08:00","end":"17:00","timezone":str(TZ)})
    # Live defaults
    cfg.setdefault("live_tf", "M1")
    cfg.setdefault("live_expiry", "5m")
    cfg.setdefault("deriv_count", 300)
    cfg.setdefault("use_deriv_fetch", True)
    # Symbols defaults
    cfg.setdefault("symbols_raw", ["frxEURUSD","frxGBPUSD"])
    # Indicators defaults
    if "indicators" not in cfg:
        cfg["indicators"] = {k: {"enabled": False, **spec["fields"]} for k, spec in INDICATOR_SPECS.items()}
    # Strategies toggles
    cfg.setdefault("strategies", {
        "BASE":{"enabled": True},
        "TREND":{"enabled": False},
        "CHOP":{"enabled": False},
        "CUSTOM1":{"enabled": False},
        "CUSTOM2":{"enabled": False},
        "CUSTOM3":{"enabled": False},
    })
    # Custom rules text boxes
    cfg.setdefault("custom1_rules","")
    cfg.setdefault("custom2_rules","")
    cfg.setdefault("custom3_rules","")
    return cfg

def _default_ctx() -> Dict[str, Any]:
    cfg = _ensure_cfg_defaults(get_config())
    status = ENGINE.status()
    return {
        "cfg": cfg,
        "ind_specs": INDICATOR_SPECS,
        "engine": status,
        "now_local": datetime.now(TZ).strftime("%b %d, %Y %H:%M"),
        "window_ok": within_window(cfg),
        "tz": str(TZ),
    }

# ---------------- Auth / Landing ----------------
@bp.route("/")
def root():
    # Send to dashboard (template expects login/logout endpoints)
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

# ---------------- Main dashboard ----------------
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

        # --------- Pairs selector + manual symbols ---------
        majors = ["EURUSD","GBPUSD","USDJPY","USDCHF","USDCAD","AUDUSD","NZDUSD","EURGBP","EURJPY","GBPJPY","EURAUD","AUDJPY","CADJPY","CHFJPY"]
        chosen_po = [po for po in majors if request.form.get(f"pair_{po}")]
        manual_text = (request.form.get("symbols_text","") or "").replace(",", " ")
        manual = [s for s in manual_text.split() if s]
        combined = chosen_po + manual
        if request.form.get("convert_po"):
            combined = convert_po_to_deriv(combined)
        # dedup preserve order
        seen=set(); final=[]
        for s in combined:
            if s not in seen:
                seen.add(s); final.append(s)
        if final:
            cfg["symbols_raw"] = final

        # Deriv fetch defaults
        cfg["use_deriv_fetch"] = bool(request.form.get("use_deriv_fetch"))
        try:
            cfg["deriv_count"] = int(request.form.get("deriv_count", cfg.get("deriv_count", 300)))
        except Exception:
            cfg["deriv_count"] = 300

        # Strategies toggles
        st = cfg.get("strategies", {})
        for name in ["BASE","TREND","CHOP","CUSTOM1","CUSTOM2","CUSTOM3"]:
            st[name] = {"enabled": bool(request.form.get(f"s_{name}"))}
        cfg["strategies"] = st

        # Custom rule text boxes
        cfg["custom1_rules"] = request.form.get("custom1_rules", cfg.get("custom1_rules",""))
        cfg["custom2_rules"] = request.form.get("custom2_rules", cfg.get("custom2_rules",""))
        cfg["custom3_rules"] = request.form.get("custom3_rules", cfg.get("custom3_rules",""))

        # Indicators
        cfg = _merge_indicator_form(cfg, request.form)

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

    # Symbols from input; fallback to saved cfg
    symbols_raw = (request.form.get("bt_symbols") or "").replace(";", ",")
    symbols = [s.strip() for s in symbols_raw.split(",") if s.strip()] or cfg.get("symbols_raw", [])
    if request.form.get("convert_po_bt"):
        symbols = convert_po_to_deriv(symbols)

    tf = (request.form.get("bt_tf") or cfg.get("live_tf","M1")).upper()
    expiry = request.form.get("bt_expiry", cfg.get("live_expiry","5m"))
    use_deriv = bool(request.form.get("use_deriv_fetch", "1" if cfg.get("use_deriv_fetch") else ""))
    try:
        target = max(60, int(request.form.get("bt_count", cfg.get("deriv_count", 300))))
    except Exception:
        target = cfg.get("deriv_count", 300)
    gran = TF_TO_GRAN.get(tf, 60)

    # CSV overrides network
    df = None
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

    strategy = (request.form.get("bt_strategy") or "BASE").upper()
    sigs, stats = backtest_run(df, strategy, ind_cfg, expiry)
    png_name = plot_signals(df, sigs, ind_cfg, strategy=strategy, tf=tf, expiry=expiry)

    return jsonify({
        "ok": True,
        "plot": f"/plots/{png_name}",
        "signals": [{"ts": i["index"].isoformat(), "dir": i["direction"], "exp": i["expiry_idx"].isoformat()} for i in sigs],
        "stats": stats
    })

@bp.route("/plots/<name>")
def plot_file(name: str):
    return send_file(os.path.join("static","plots", name))

# ---------------- Telegram ----------------
@bp.route("/telegram/send", methods=["POST"])
@_admin_required
def telegram_send():
    data = request.form.to_dict() or {}
    text = (data.get("text") or "").trim() if hasattr(str, "trim") else (data.get("text") or "").strip()
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
    # simple fanout to all tiers from the dashboard
    msg = request.form.get("text") or "🧪 Test from dashboard"
    results = {}
    for t in ["free","basic","pro","vip"]:
        results[t] = ENGINE.send_to_tier(t, f"{msg} ({t.upper()})")
    return jsonify({"ok": True, "results": results})

# ---------------- Live Engine (UI + API) ----------------
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

# API aliases used by the dashboard JS
@bp.route("/api/start", methods=["POST"])
@_admin_required
def api_start():
    ok, msg = ENGINE.start()
    return jsonify({"ok": ok, "msg": msg, "status": ENGINE.status()})

@bp.route("/api/stop", methods=["POST"])
@_admin_required
def api_stop():
    ok, msg = ENGINE.stop()
    return jsonify({"ok": ok, "msg": msg, "status": ENGINE.status()})

@bp.route("/api/status")
def api_status():
    return jsonify(ENGINE.status())

@bp.route("/live/status")
def live_status():
    return jsonify(ENGINE.status())

@bp.route("/live/tally")
def live_tally():
    st = ENGINE.status()
    tally = st.get("tally") or st.get("tallies") or {}
    return jsonify(tally)

# ---------------- Health ----------------
@bp.route("/_up")
def up_check():
    return jsonify({"ok": True, "ts": datetime.now(timezone.utc).isoformat()+"Z"})
