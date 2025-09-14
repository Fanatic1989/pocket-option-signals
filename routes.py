# routes.py — Dashboard, config, backtest, Telegram + Live APIs (everything wired)
from __future__ import annotations
import os, io
from datetime import datetime, timezone
from typing import Dict, Any

from flask import (
    Blueprint, render_template, request, redirect, url_for, flash,
    jsonify, session, send_file
)

from utils import (
    TZ, get_config, set_config, within_window,
    convert_po_to_deriv, fetch_deriv_history, load_csv,
    backtest_run, plot_signals, exec_sql
)

# ---------- Live engine (tolerant import) ----------
try:
    from live_engine import ENGINE, TIER_TO_CHAT, DAILY_CAPS
except Exception:
    class _Stub:
        def status(self): return {"running": False, "debug": False, "loop_sleep": 4,
                                  "tally": {"total": 0, "by_tier": {"free":0,"basic":0,"pro":0,"vip":0}},
                                  "tallies": {"all":0,"free":0,"basic":0,"pro":0,"vip":0}}
        def start(self):  return False, "ENGINE not wired"
        def stop(self):   return False, "ENGINE not wired"
        def set_debug(self, v): pass
        def send_to_tier(self, tier, text): return {"ok": False, "error": "ENGINE not wired"}
    ENGINE = _Stub()
    TIER_TO_CHAT = {"free": None, "basic": None, "pro": None, "vip": None}
    DAILY_CAPS = {"free":3, "basic":6, "pro":15, "vip": None}

bp = Blueprint("dashboard", __name__, template_folder="templates", static_folder="static")

ADMIN_PASS = os.getenv("ADMIN_PASSWORD", "admin")

# ---------------------------- Indicator specs (forms) -------------------------
# Keys must match utils.compute_indicators(); we’ll also accept a few aliases below.
INDICATOR_SPECS: Dict[str, Dict[str, Any]] = {
    # MAs (overlays)
    "SMA":      {"title":"Simple MA",       "panel":"overlay", "fields":{"period":50}},
    "EMA":      {"title":"Exponential MA",  "panel":"overlay", "fields":{"period":20}},
    "WMA":      {"title":"Weighted MA",     "panel":"overlay", "fields":{"period":20}},
    "SMMA":     {"title":"Smoothed MA",     "panel":"overlay", "fields":{"period":20}},
    "TMA":      {"title":"Triangular MA",   "panel":"overlay", "fields":{"period":20}},

    # Bands / Channels (overlays)
    "BOLL":     {"title":"Bollinger Bands", "panel":"overlay", "fields":{"period":20,"mult":2}},
    "KELTNER":  {"title":"Keltner Channel", "panel":"overlay", "fields":{"period":20,"mult":2}},
    "DONCHIAN": {"title":"Donchian Channels","panel":"overlay","fields":{"period":20}},
    "ENVELOPES":{"title":"Envelopes",       "panel":"overlay", "fields":{"period":20,"pct":2.0}},

    # Ichimoku / PSAR / Supertrend (overlays)
    "ICHIMOKU": {"title":"Ichimoku",        "panel":"overlay", "fields":{}},
    "PSAR":     {"title":"Parabolic SAR",   "panel":"overlay", "fields":{"step":0.02,"max":0.2}},
    "SUPERTREND":{"title":"Supertrend",     "panel":"overlay", "fields":{"period":10,"mult":3}},

    # New overlays
    "ALLIGATOR":{"title":"Alligator",       "panel":"overlay",
                 "fields":{"jaw":13,"teeth":8,"lips":5,"jaw_shift":8,"teeth_shift":5,"lips_shift":3}},
    "FRACTAL":  {"title":"Fractal Chaos Bands","panel":"overlay","fields":{"lookback":2,"smooth":5}},

    # Oscillators (separate panels)
    "RSI":      {"title":"RSI",             "panel":"osc",     "fields":{"period":14}},
    "STOCH":    {"title":"Stochastic",      "panel":"osc",     "fields":{"k":14,"d":3}},
    "ATR":      {"title":"ATR",             "panel":"osc",     "fields":{"period":14}},
    "ADX":      {"title":"ADX/+DI/-DI",     "panel":"osc",     "fields":{"period":14}},
    "CCI":      {"title":"CCI",             "panel":"osc",     "fields":{"period":20}},
    "MOMENTUM": {"title":"Momentum",        "panel":"osc",     "fields":{"period":10}},
    "ROC":      {"title":"Rate of Change",  "panel":"osc",     "fields":{"period":10}},
    "WILLR":    {"title":"Williams %R",     "panel":"osc",     "fields":{"period":14}},
    "VORTEX":   {"title":"Vortex",          "panel":"osc",     "fields":{"period":14}},
    "MACD":     {"title":"MACD",            "panel":"osc",     "fields":{"fast":12,"slow":26,"signal":9}},
    "AO":       {"title":"Awesome Osc.",    "panel":"osc",     "fields":{}},
    "AC":       {"title":"Accelerator Osc.","panel":"osc",     "fields":{}},
    "BEARS":    {"title":"Bears Power",     "panel":"osc",     "fields":{"period":13}},
    "BULLS":    {"title":"Bulls Power",     "panel":"osc",     "fields":{"period":13}},
    "DEMARKER": {"title":"DeMarker",        "panel":"osc",     "fields":{"period":14}},
    "OSMA":     {"title":"OsMA",            "panel":"osc",     "fields":{}},
    "ZIGZAG":   {"title":"ZigZag",          "panel":"osc",     "fields":{"pct":1.0}},

    # New oscillators
    "AROON":    {"title":"Aroon",           "panel":"osc",     "fields":{"period":14}},
    "STC":      {"title":"Schaff Trend Cycle","panel":"osc",   "fields":{"fast":23,"slow":50,"cycle":10}},
}

# Accept some friendly names coming from UI or external sources
INDICATOR_ALIASES = {
    "STOCHASTIC": "STOCH",
    "WILLIAMS%R": "WILLR",
    "WILLIAMS":   "WILLR",
    "SIMPLEMA":   "SMA",
    "EXPMA":      "EMA",
}

TF_TO_GRAN = {"M1":60,"M2":120,"M3":180,"M5":300,"M10":600,"M15":900,"M30":1800,"H1":3600,"H4":14400,"D1":86400}

# --------------------------- DB for Telegram users ---------------------------
exec_sql("""CREATE TABLE IF NOT EXISTS tg_users(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  tier   TEXT NOT NULL,
  chat_id TEXT NOT NULL UNIQUE,
  label  TEXT
)""")

# --------------------------- Helpers -----------------------------------------
def _ensure_cfg_defaults(cfg: dict | None) -> dict:
    cfg = dict(cfg or {})
    # Window
    cfg.setdefault("window", {"start":"08:00","end":"17:00","timezone":str(TZ)})
    # Live defaults
    cfg.setdefault("live_tf","M1")
    cfg.setdefault("live_expiry","5m")
    cfg.setdefault("deriv_count", 300)
    cfg.setdefault("use_deriv_fetch", True)
    # Indicators
    if "indicators" not in cfg or not isinstance(cfg["indicators"], dict):
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
    # Custom rules
    cfg.setdefault("custom1_rules","")
    cfg.setdefault("custom2_rules","")
    cfg.setdefault("custom3_rules","")
    # Symbols default
    cfg.setdefault("symbols_raw", ["frxEURUSD","frxGBPUSD"])
    set_config(cfg)
    return cfg

def _normalize_ind_key(k: str) -> str:
    k = (k or "").strip().upper()
    return INDICATOR_ALIASES.get(k, k)

def _save_indicators_from_form(cfg: dict, form) -> None:
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
                    if isinstance(default, int): params[f_name] = int(float(v))
                    elif isinstance(default, float): params[f_name] = float(v)
                    else: params[f_name] = v
                except Exception:
                    params[f_name] = default
        ind[key] = {"enabled": enabled, **params}
    cfg["indicators"] = ind

def _ctx_base(view: str="dashboard") -> Dict[str, Any]:
    cfg = _ensure_cfg_defaults(get_config())
    users = exec_sql("SELECT id,tier,chat_id,label FROM tg_users ORDER BY tier,id", fetch=True) or []
    return {
        "view": view,
        "cfg": cfg,
        "ind_specs": INDICATOR_SPECS,
        "engine": ENGINE.status(),
        "now_local": datetime.now(TZ).strftime("%b %d, %Y %H:%M"),
        "window_ok": within_window(cfg),
        "tz": str(TZ),
        "tier_to_chat": {k: bool(v) for k, v in (TIER_TO_CHAT or {}).items()},
        "daily_caps": DAILY_CAPS,
        "users": users,
    }

def _admin_required(fn):
    def wrap(*a, **kw):
        if not session.get("admin"):
            return redirect(url_for("dashboard.login"))
        return fn(*a, **kw)
    wrap.__name__ = fn.__name__
    return wrap

# ------------------------------ Routes: basic --------------------------------
@bp.route("/")
def root():
    return redirect(url_for("dashboard.view"))

@bp.route("/login", methods=["GET","POST"])
def login():
    if request.method == "POST":
        if request.form.get("password","") == ADMIN_PASS:
            session["admin"] = True
            return redirect(url_for("dashboard.view"))
        flash("Wrong password", "error")
    return render_template("dashboard.html", **_ctx_base("login"))

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

        # Symbols (pairs selector + free text)
        majors = ["EURUSD","GBPUSD","USDJPY","USDCHF","USDCAD","AUDUSD","NZDUSD",
                  "EURGBP","EURJPY","GBPJPY","EURAUD","AUDJPY","CADJPY","CHFJPY"]
        chosen = [m for m in majors if request.form.get(f"pair_{m}")]
        typed = (request.form.get("symbols_text","") or "").replace(",", " ").split()
        symbols = chosen + [t for t in typed if t]
        if request.form.get("convert_po"):
            symbols = convert_po_to_deriv(symbols)
        if symbols:
            # dedupe keep order
            cfg["symbols_raw"] = list(dict.fromkeys(symbols))

        # Deriv fetch settings
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

        # Custom rules text boxes
        cfg["custom1_rules"] = request.form.get("custom1_rules", cfg.get("custom1_rules",""))
        cfg["custom2_rules"] = request.form.get("custom2_rules", cfg.get("custom2_rules",""))
        cfg["custom3_rules"] = request.form.get("custom3_rules", cfg.get("custom3_rules",""))

        # Indicators
        _save_indicators_from_form(cfg, request.form)

        set_config(cfg)
        flash("Saved.", "ok")
        return redirect(url_for("dashboard.view"))

    return render_template("dashboard.html", **_ctx_base("dashboard"))

# ---------------- AJAX: live-wired indicator toggles ----------------
@bp.post("/api/indicators/toggle")
@_admin_required
def api_ind_toggle():
    data = request.get_json(silent=True) or {}
    key = _normalize_ind_key(data.get("key"))
    enabled = bool(data.get("enabled"))
    cfg = _ensure_cfg_defaults(get_config())
    if key not in INDICATOR_SPECS:
        return jsonify({"ok": False, "error": f"Unknown indicator '{data.get('key')}'"}), 400
    cur = cfg["indicators"].get(key, {"enabled": False, **INDICATOR_SPECS[key]["fields"]})
    cur["enabled"] = enabled
    cfg["indicators"][key] = cur
    set_config(cfg)
    return jsonify({"ok": True, "indicator": {key: cfg["indicators"][key]}})

@bp.post("/api/indicators/params")
@_admin_required
def api_ind_params():
    data = request.get_json(silent=True) or {}
    key = _normalize_ind_key(data.get("key"))
    params = data.get("params") or {}
    cfg = _ensure_cfg_defaults(get_config())
    if key not in INDICATOR_SPECS:
        return jsonify({"ok": False, "error": f"Unknown indicator '{data.get('key')}'"}), 400
    spec = INDICATOR_SPECS[key]
    cur = cfg["indicators"].get(key, {"enabled": False, **spec["fields"]})
    # cast by field type
    for fname, default in spec["fields"].items():
        if fname in params:
            val = params[fname]
            try:
                if isinstance(default, int):
                    cur[fname] = int(float(val))
                elif isinstance(default, float):
                    cur[fname] = float(val)
                else:
                    cur[fname] = val
            except Exception:
                cur[fname] = default
    cfg["indicators"][key] = cur
    set_config(cfg)
    return jsonify({"ok": True, "indicator": {key: cfg["indicators"][key]}})

@bp.get("/api/indicators")
@_admin_required
def api_indicators():
    cfg = _ensure_cfg_defaults(get_config())
    return jsonify({"specs": INDICATOR_SPECS, "state": cfg.get("indicators", {})})

# ------------------------------ Backtest -------------------------------------
@bp.post("/backtest")
@_admin_required
def backtest():
    cfg = _ensure_cfg_defaults(get_config())
    ind_cfg = cfg.get("indicators") or {}

    # Symbols: provided or from saved config
    symbols_raw = (request.form.get("bt_symbols") or "").replace(";", ",").replace(" ", ",")
    symbols = [s.strip() for s in symbols_raw.split(",") if s.strip()] or cfg.get("symbols_raw", [])
    if request.form.get("convert_po_bt"):
        symbols = convert_po_to_deriv(symbols)

    tf = request.form.get("bt_tf", cfg.get("live_tf","M1")).upper()
    expiry = request.form.get("bt_expiry", cfg.get("live_expiry","5m"))
    use_deriv = bool(request.form.get("use_deriv_fetch", "1" if cfg.get("use_deriv_fetch") else ""))
    try:
        target = max(60, int(request.form.get("bt_count", cfg.get("deriv_count", 300))))
    except Exception:
        target = cfg.get("deriv_count", 300)
    gran = TF_TO_GRAN.get(tf, 60)

    # CSV overrides fetch
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
            tip = "Tip: upload a CSV on Backtest, or try another symbol/timeframe."
            return jsonify({"ok": False, "error": "Deriv fetch failed. " + "; ".join(errs)[:900] + " " + tip}), 400

    strategy = request.form.get("bt_strategy", "BASE").upper()
    sigs, stats = backtest_run(df, strategy, ind_cfg, expiry)
    png_name = plot_signals(df, sigs, ind_cfg, strategy=strategy, tf=tf, expiry=expiry)

    return jsonify({
        "ok": True,
        "plot": f"/plots/{png_name}",
        "signals": [{"ts": i["index"].isoformat(), "dir": i["direction"], "exp": i["expiry_idx"].isoformat()} for i in sigs],
        "stats": stats
    })

@bp.get("/backtest/last.json")
@_admin_required
def backtest_last_json():
    return jsonify({"ok": True, **(session.get("bt_state") or {})})

@bp.get("/plots/<name>")
def plot_file(name: str):
    return send_file(os.path.join("static","plots", name))

# ------------------------------ Telegram -------------------------------------
@bp.post("/telegram/send")
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

@bp.post("/telegram/test")
@_admin_required
def telegram_test():
    msg = request.form.get("text") or "🧪 Broadcast test"
    results = {}
    for t in ["free","basic","pro","vip"]:
        results[t] = ENGINE.send_to_tier(t, f"{msg} ({t.upper()})")
    return jsonify({"ok": True, "results": results})

@bp.get("/api/check_bot")
def api_check_bot():
    token = (os.getenv("TELEGRAM_BOT_TOKEN") or "").strip()
    out = {
        "configured_chats": {k: bool(v) for k, v in (TIER_TO_CHAT or {}).items()},
        "caps": DAILY_CAPS,
        "running": ENGINE.status().get("running", False),
    }
    if token:
        try:
            import requests
            r = requests.get(f"https://api.telegram.org/bot{token}/getMe", timeout=10)
            out["getMe"] = r.json()
        except Exception as e:
            out["getMe"] = {"ok": False, "error": str(e)}
    else:
        out["getMe"] = {"ok": False, "error": "TELEGRAM_BOT_TOKEN not set"}
    return jsonify(out)

# ------------------------------ Live Engine ----------------------------------
@bp.post("/live/start")
@_admin_required
def live_start():
    ok, msg = ENGINE.start()
    return jsonify({"ok": ok, "msg": msg, "status": ENGINE.status()})

@bp.post("/live/stop")
@_admin_required
def live_stop():
    ok, msg = ENGINE.stop()
    return jsonify({"ok": ok, "msg": msg, "status": ENGINE.status()})

@bp.post("/live/debug/on")
@_admin_required
def live_debug_on():
    ENGINE.set_debug(True); return jsonify({"ok": True})

@bp.post("/live/debug/off")
@_admin_required
def live_debug_off():
    ENGINE.set_debug(False); return jsonify({"ok": True})

@bp.get("/live/status")
def live_status():
    return jsonify(ENGINE.status())

@bp.get("/live/tally")
def live_tally():
    st = ENGINE.status()
    tally = st.get("tally") or st.get("tallies") or {}
    return jsonify(tally)

# ------------------------------ Users (Telegram) -----------------------------
@bp.post("/users/add")
@_admin_required
def users_add():
    tier = (request.form.get("tier") or "").lower().strip()
    chat_id = (request.form.get("chat_id") or "").strip()
    label = (request.form.get("label") or "").strip() or None
    if tier not in ("free","basic","pro","vip"):
        flash("Invalid tier", "error"); return redirect(url_for("dashboard.view"))
    if not chat_id:
        flash("chat_id required", "error"); return redirect(url_for("dashboard.view"))
    try:
        exec_sql("INSERT OR IGNORE INTO tg_users(tier,chat_id,label) VALUES(?,?,?)",
                 (tier, chat_id, label))
        flash("User added.", "ok")
    except Exception as e:
        flash(f"DB error: {e}", "error")
    return redirect(url_for("dashboard.view"))

@bp.post("/users/delete")
@_admin_required
def users_delete():
    try:
        uid = int(request.form.get("id","0"))
        exec_sql("DELETE FROM tg_users WHERE id=?", (uid,))
        flash("User removed.", "ok")
    except Exception as e:
        flash(f"DB error: {e}", "error")
    return redirect(url_for("dashboard.view"))

@bp.get("/api/users")
@_admin_required
def api_users():
    rows = exec_sql("SELECT id,tier,chat_id,label FROM tg_users ORDER BY tier,id", fetch=True) or []
    return jsonify([{"id":r[0],"tier":r[1],"chat_id":r[2],"label":r[3]} for r in rows])

# ------------------------------ API echoes / health --------------------------
@bp.get("/api/indicators/specs")
def api_ind_specs():
    return jsonify(INDICATOR_SPECS)

@bp.get("/api/status")
def api_status_dup():
    s = ENGINE.status()
    out = {
        "caps": DAILY_CAPS,
        "configured_chats": {k: bool(v) for k, v in (TIER_TO_CHAT or {}).items()},
        "day": datetime.now(TZ).strftime("%Y-%m-%d"),
        "debug": s.get("debug", False),
        "last_error": s.get("last_error"),
        "last_send_result": s.get("last_send_result", {}),
        "loop_sleep": s.get("loop_sleep", 4),
        "running": s.get("running", False),
        "tallies": s.get("tallies", {"all":0,"basic":0,"free":0,"pro":0,"vip":0}),
        "tally": s.get("tally", {"date": datetime.now(TZ).strftime("%Y-%m-%d"),
                                  "total": 0, "by_tier": {"free":0,"basic":0,"pro":0,"vip":0}}),
    }
    return jsonify(out)

@bp.get("/_up")
def up_check():
    return jsonify({"ok": True, "ts": datetime.now(timezone.utc).isoformat()+"Z"})
