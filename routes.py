# routes.py — dashboard, config, indicators, customs, users, backtest, live controls, telegram
import os, io, json
from datetime import datetime, timezone
from typing import Dict, Any, List

from flask import (
    Blueprint, render_template, request, redirect, url_for, flash, session,
    send_from_directory, jsonify
)

from utils import (
    TZ, TIMEZONE, get_config, set_config, within_window, convert_po_to_deriv,
    load_csv, fetch_deriv_history, backtest_run, PO_PAIRS, DERIV_PAIRS
)
from live_engine import ENGINE  # start/stop/status + send_signal caps-respecting

bp = Blueprint("dashboard", __name__)

# ------------------------------- Auth ----------------------------------------
ADMIN_PW = os.getenv("ADMIN_PASSWORD", "admin123")

def require_admin(fn):
    def wrap(*a, **kw):
        if not session.get("admin"):
            return redirect(url_for("dashboard.login"))
        return fn(*a, **kw)
    wrap.__name__ = fn.__name__
    return wrap

# ----------------------------- Indicator specs -------------------------------
# Big dictionary that drives the UI (toggles + params) and maps 1:1 to utils.compute_indicators()
INDICATOR_SPECS: Dict[str, Dict[str, Any]] = {
    # Overlays
    "SMA": {"name":"Simple MA","kind":"overlay","params":{"period":50}},
    "EMA": {"name":"Exponential MA","kind":"overlay","params":{"period":20}},
    "WMA": {"name":"Weighted MA","kind":"overlay","params":{"period":20}},
    "SMMA":{"name":"Smoothed MA","kind":"overlay","params":{"period":20}},
    "TMA": {"name":"Triangular MA","kind":"overlay","params":{"period":20}},
    "BOLL":{"name":"Bollinger Bands","kind":"overlay","params":{"period":20,"mult":2}},
    "KELTNER":{"name":"Keltner Channel","kind":"overlay","params":{"period":20,"mult":2}},
    "DONCHIAN":{"name":"Donchian","kind":"overlay","params":{"period":20}},
    "ENVELOPES":{"name":"Envelopes","kind":"overlay","params":{"period":20,"pct":2}},
    "ICHIMOKU":{"name":"Ichimoku","kind":"overlay","params":{}},
    "PSAR":{"name":"Parabolic SAR","kind":"overlay","params":{"step":0.02,"max":0.2}},
    "SUPERTREND":{"name":"Supertrend","kind":"overlay","params":{"period":10,"mult":3}},

    # Oscillators / separate panels
    "RSI":{"name":"RSI","kind":"osc","params":{"period":14}},
    "STOCH":{"name":"Stochastic","kind":"osc","params":{"k":14,"d":3}},
    "ATR":{"name":"ATR","kind":"osc","params":{"period":14}},
    "ADX":{"name":"ADX","kind":"osc","params":{"period":14}},
    "CCI":{"name":"CCI","kind":"osc","params":{"period":20}},
    "MOMENTUM":{"name":"Momentum","kind":"osc","params":{"period":10}},
    "ROC":{"name":"ROC","kind":"osc","params":{"period":10}},
    "WILLR":{"name":"Williams %R","kind":"osc","params":{"period":14}},
    "VORTEX":{"name":"Vortex","kind":"osc","params":{"period":14}},
    "MACD":{"name":"MACD","kind":"osc","params":{"fast":12,"slow":26,"signal":9}},
    "AO":{"name":"Awesome Osc","kind":"osc","params":{}},
    "AC":{"name":"Accelerator Osc","kind":"osc","params":{}},
    "BEARS":{"name":"Bears Power","kind":"osc","params":{"period":13}},
    "BULLS":{"name":"Bulls Power","kind":"osc","params":{"period":13}},
    "DEMARKER":{"name":"DeMarker","kind":"osc","params":{"period":14}},
    "OSMA":{"name":"OSMA","kind":"osc","params":{}},
    "ZIGZAG":{"name":"ZigZag","kind":"osc","params":{"pct":1.0}},
}

# ------------------------------ Helpers --------------------------------------
def _ctx_base(view="index"):
    cfg = get_config() or {}
    window = cfg.get("window") or {}
    now = datetime.now(TZ).strftime("%Y-%m-%d %H:%M:%S")
    within = within_window({"window": window})
    live_tf = cfg.get("live_tf", "M1")
    live_expiry = cfg.get("live_expiry", "5m")
    active_symbols = cfg.get("active_symbols", [])
    symbols_raw = cfg.get("symbols_raw", [])
    indicators = cfg.get("indicators", {})
    strategies = cfg.get("strategies", {"BASE":{"enabled":True},"TREND":{"enabled":False},"CHOP":{"enabled":False}})
    customs = cfg.get("customs", [
        {"_idx":1,"enabled":False,"mode":"SIMPLE","lookback":3,"tol_pct":0.1},
        {"_idx":2,"enabled":False,"mode":"SIMPLE","lookback":3,"tol_pct":0.1},
        {"_idx":3,"enabled":False,"mode":"SIMPLE","lookback":3,"tol_pct":0.1},
    ])
    bt = cfg.get("bt_last")  # last backtest summary for re-display
    return {
        "view": view,
        "tz": TIMEZONE, "now": now, "within": within,
        "window": window,
        "live_tf": live_tf, "live_expiry": live_expiry,
        "active_symbols": active_symbols, "symbols_raw": symbols_raw,
        "available_groups": [
            {"label":"Deriv FRX", "items": DERIV_PAIRS},
            {"label":"PocketOption Majors", "items": PO_PAIRS},
        ],
        "indicators": indicators, "specs": INDICATOR_SPECS,
        "strategies": strategies,
        "strategies_all": {"BASE":{}, "TREND":{}, "CHOP":{}},
        "customs": [{"_idx":1, **customs[0]}, {"_idx":2, **customs[1]}, {"_idx":3, **customs[2]}] if customs else [],
        "bt": bt,
        "session": {"admin": bool(session.get("admin"))},
    }

def _save_cfg(mut: Dict[str, Any]):
    cfg = get_config() or {}
    cfg.update(mut)
    set_config(cfg)

# ------------------------------ Views ----------------------------------------
@bp.route("/")
def index():
    ctx = _ctx_base(view="index")
    return render_template("dashboard.html", **ctx)

@bp.route("/login", methods=["GET","POST"])
def login():
    if request.method == "POST":
        pw = request.form.get("password","")
        if pw == ADMIN_PW:
            session["admin"] = True
            return redirect(url_for("dashboard.view"))
        flash("Wrong password", "error")
    ctx = _ctx_base(view="login")
    return render_template("dashboard.html", **ctx)

@bp.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("dashboard.index"))

@bp.route("/dashboard")
@require_admin
def view():
    ctx = _ctx_base(view="dashboard")
    return render_template("dashboard.html", **ctx)

# --------------------------- Window defaults ---------------------------------
@bp.route("/update/window", methods=["POST"])
@require_admin
def update_window():
    start = request.form.get("start","08:00")
    end = request.form.get("end","17:00")
    tz = request.form.get("timezone", TIMEZONE)
    live_tf = request.form.get("live_tf","M1")
    live_expiry = request.form.get("live_expiry","5m")
    _save_cfg({"window":{"start":start,"end":end,"timezone":tz}, "live_tf":live_tf, "live_expiry":live_expiry})
    flash("Saved trading window & defaults.","ok")
    return redirect(url_for("dashboard.view"))

# ------------------------------ Symbols --------------------------------------
@bp.route("/update/symbols", methods=["POST"])
@require_admin
def update_symbols():
    cfg = get_config() or {}
    symbols_text = request.form.get("symbols_text","").strip()
    multi_deriv = request.form.getlist("symbols_deriv_multi")
    multi_po = request.form.getlist("symbols_po_multi")
    raw = []

    if multi_deriv: raw.extend(multi_deriv)
    if multi_po: raw.extend(multi_po)
    if symbols_text:
        # split on comma/space
        for tok in symbols_text.replace(",", " ").split():
            raw.append(tok.strip())

    convert_po = bool(request.form.get("convert_po"))
    active = convert_po_to_deriv(raw) if convert_po else raw
    _save_cfg({"symbols_raw": raw, "active_symbols": active})
    flash(f"Saved {len(active)} symbols.", "ok")
    return redirect(url_for("dashboard.view"))

# ------------------------------ Indicators -----------------------------------
@bp.route("/update/indicators", methods=["POST"])
@require_admin
def update_indicators():
    out = {}
    for key, spec in INDICATOR_SPECS.items():
        enabled = bool(request.form.get(f"ind_{key}_enabled"))
        params = {}
        for p_name in spec.get("params", {}).keys():
            v = request.form.get(f"ind_{key}_{p_name}")
            if v is not None and v != "":
                params[p_name] = v
        if enabled:
            out[key] = {"enabled": True, **params}
    _save_cfg({"indicators": out})
    flash("Indicators saved.", "ok")
    return redirect(url_for("dashboard.view"))

# ------------------------------ Strategies -----------------------------------
@bp.route("/update/strategies", methods=["POST"])
@require_admin
def update_strategies():
    strategies = {}
    for name in ["BASE","TREND","CHOP"]:
        strategies[name] = {"enabled": bool(request.form.get(f"s_{name}"))}
    live_tf = request.form.get("live_tf","M1")
    live_expiry = request.form.get("live_expiry","5m")
    bt_tf = request.form.get("bt_tf", live_tf)
    bt_expiry = request.form.get("bt_expiry", live_expiry)
    cfg = get_config() or {}
    cfg.update({"strategies": strategies, "live_tf":live_tf, "live_expiry":live_expiry})
    # Keep last BT defaults in config so form remembers
    cfg.setdefault("bt_defaults", {})["tf"] = bt_tf
    cfg["bt_defaults"]["expiry"] = bt_expiry
    set_config(cfg)
    flash("Strategies & defaults saved.", "ok")
    return redirect(url_for("dashboard.view"))

# ------------------------------- Custom rules --------------------------------
@bp.route("/update/custom", methods=["POST"])
@require_admin
def update_custom():
    cfg = get_config() or {}
    customs = cfg.get("customs") or [
        {"_idx":1},{"_idx":2},{"_idx":3}
    ]
    slot = int(request.form.get("slot","1"))
    idx = max(1, min(3, slot)) - 1
    cur = customs[idx] if idx < len(customs) else {"_idx":slot}

    cur["enabled"] = bool(request.form.get("enabled"))
    cur["mode"] = request.form.get("mode","SIMPLE")
    cur["lookback"] = int(request.form.get("lookback","3") or 3)
    try:
        cur["tol_pct"] = float(request.form.get("tol_pct","0.1") or 0.1)
    except Exception:
        cur["tol_pct"] = 0.1
    cur["simple_buy"] = request.form.get("simple_buy","")
    cur["simple_sell"] = request.form.get("simple_sell","")

    buy_json = request.form.get("buy_rule_json","").strip()
    sell_json = request.form.get("sell_rule_json","").strip()
    try:
        cur["buy_rule"] = json.loads(buy_json) if buy_json else None
    except Exception:
        flash(f"CUSTOM {slot}: invalid buy_rule JSON (kept previous if any).", "error")
    try:
        cur["sell_rule"] = json.loads(sell_json) if sell_json else None
    except Exception:
        flash(f"CUSTOM {slot}: invalid sell_rule JSON (kept previous if any).", "error")

    if len(customs) < 3:
        # pad
        while len(customs) < 3:
            customs.append({"_idx": len(customs)+1})

    customs[idx] = cur
    _save_cfg({"customs": customs})
    flash(f"Saved CUSTOM {slot}.", "ok")
    return redirect(url_for("dashboard.view"))

# -------------------------------- Users --------------------------------------
@bp.route("/users/add", methods=["POST"])
@require_admin
def users_add():
    # Simple inline store inside config (SQLite table if you want later)
    cfg = get_config() or {}
    users = cfg.get("users", [])
    tid = request.form.get("telegram_id","").strip()
    tier = request.form.get("tier","free").strip().lower()
    expires_at = request.form.get("expires_at","").strip()
    if not tid:
        flash("Telegram ID is required","error"); return redirect(url_for("dashboard.view"))
    # upsert
    found = False
    for u in users:
        if str(u.get("telegram_id")) == tid:
            u["tier"] = tier; u["expires_at"] = expires_at; found=True; break
    if not found:
        users.append({"telegram_id": tid, "tier": tier, "expires_at": expires_at})
    _save_cfg({"users": users})
    flash("User saved.","ok")
    return redirect(url_for("dashboard.view"))

@bp.route("/users/delete", methods=["POST"])
@require_admin
def users_delete():
    cfg = get_config() or {}
    users = cfg.get("users", [])
    tid = request.form.get("telegram_id","").strip()
    users = [u for u in users if str(u.get("telegram_id")) != tid]
    _save_cfg({"users": users})
    flash("User removed.","ok")
    return redirect(url_for("dashboard.view"))

# ------------------------------- Backtest ------------------------------------
@bp.route("/backtest", methods=["POST"])
@require_admin
def backtest():
    cfg = get_config() or {}
    use_server = bool(request.form.get("use_server"))
    tf = request.form.get("bt_tf", cfg.get("live_tf","M1")).upper()
    expiry = request.form.get("bt_expiry", cfg.get("live_expiry","5m")).lower()
    strategy = request.form.get("bt_strategy","BASE").upper()

    # seconds per candle map
    tf_map = {"M1":60,"M2":120,"M3":180,"M5":300,"M10":600,"M15":900,"M30":1800,"H1":3600,"H4":14400,"D1":86400}
    gran = tf_map.get(tf, 60)

    # symbols
    syms_text = request.form.get("bt_symbols","").strip()
    syms = [s for s in syms_text.replace(","," ").split() if s]
    if request.form.get("convert_po_bt"):
        syms = convert_po_to_deriv(syms)
    if not syms:
        active = cfg.get("active_symbols") or []
        syms = active[:1]  # just the first active if none typed

    # Load data
    df = None
    error = None
    file = request.files.get("bt_csv")
    try:
        if file and file.filename:
            df = load_csv(file)
        elif use_server:
            # Pull the first symbol only for single-chart preview
            sym = syms[0]
            df = fetch_deriv_history(sym, granularity_sec=gran, count=int(request.form.get("bt_count","300") or 300))
        else:
            error = "Please upload CSV or check 'Use Deriv server fetch'."
    except Exception as e:
        error = str(e)

    if df is None or df.empty:
        _save_cfg({"bt_last":{"error": f"Deriv fetch failed. {error or 'No data'}"}})
        flash(f"Backtest error: {error or 'No data'}","error")
        return redirect(url_for("dashboard.view"))

    try:
        indicators = cfg.get("indicators", {})
        signals, stats = backtest_run(df, strategy, indicators, expiry)
        from utils import plot_signals
        plot_name = plot_signals(df, signals, indicators, strategy, tf, expiry)
        summary = f"{stats['wins']}W / {stats['loss']}L / {stats['draw']}D • Win%={stats['win_rate']:.1f}"
        _save_cfg({"bt_last":{
            "tf": tf, "expiry": expiry, "strategy": strategy,
            "summary": summary, "plot_name": plot_name, "warnings": []
        }})
        flash("Backtest complete.","ok")
    except Exception as e:
        _save_cfg({"bt_last":{"error": str(e)}})
        flash(f"Backtest error: {e}","error")

    return redirect(url_for("dashboard.view"))

@bp.route("/plot/<name>")
def plot_file(name):
    return send_from_directory(os.path.join("static","plots"), name, as_attachment=False)

@bp.route("/backtest/last.json")
def backtest_last_json():
    cfg = get_config() or {}
    return jsonify(cfg.get("bt_last") or {})

@bp.route("/backtest/last.csv")
def backtest_last_csv():
    # If you want to persist CSVs, you can save them during backtest;
    # for now return 204 to keep the quick link alive.
    return ("", 204)

# Optional helper form in UI references this; keep it harmless
@bp.route("/deriv/fetch", methods=["POST"])
@require_admin
def deriv_fetch():
    flash("Server fetch helper is not implemented in this build. Use Backtest section above.", "error")
    return redirect(url_for("dashboard.view"))

# ------------------------------- Live engine ---------------------------------
@bp.route("/live/status")
def live_status():
    s = ENGINE.status()
    return jsonify({"status": s})

@bp.route("/live/tally")
def live_tally():
    t = ENGINE.tally()
    return jsonify({"tally": {
        "free": t["by_tier"]["free"],
        "basic": t["by_tier"]["basic"],
        "pro": t["by_tier"]["pro"],
        "vip": t["by_tier"]["vip"],
        "all": t["total"],
    }})

@bp.route("/live/start")
@require_admin
def live_start():
    ok, msg = ENGINE.start()
    return jsonify({"ok": ok, "msg": msg, "status": ENGINE.status()})

@bp.route("/live/stop")
@require_admin
def live_stop():
    ok, msg = ENGINE.stop()
    return jsonify({"ok": ok, "msg": msg, "status": ENGINE.status()})

@bp.route("/live/debug/on")
@require_admin
def live_debug_on():
    ENGINE.debug = True
    return jsonify({"ok": True, "status": ENGINE.status()})

@bp.route("/live/debug/off")
@require_admin
def live_debug_off():
    ENGINE.debug = False
    return jsonify({"ok": True, "status": ENGINE.status()})

# ------------------------------- API (UI AJAX) -------------------------------
@bp.route("/api/status")
def api_status():
    s = ENGINE.status()
    # Frontend expects flat tallies + caps + configured flags
    tallies = {
        "free": s["tally"]["by_tier"]["free"],
        "basic": s["tally"]["by_tier"]["basic"],
        "pro": s["tally"]["by_tier"]["pro"],
        "vip": s["tally"]["by_tier"]["vip"],
        "total": s["tally"]["total"],
    }
    # caps come from ENGINE.status() → not present; define per requirements
    caps = {"free":3,"basic":6,"pro":15,"vip":float("inf")}
    return jsonify({
        "running": s["running"], "debug": s["debug"],
        "loop_sleep": s["loop_sleep"], "tallies": tallies,
        "caps": caps,
        "configured_chats": s.get("configured_tiers", {}),
        "day": s["tally"]["date"]
    })

@bp.route("/api/send", methods=["POST"])
@require_admin
def api_send():
    data = request.get_json(force=True, silent=True) or {}
    tier = (data.get("tier") or "vip").lower()
    text = (data.get("text") or "").strip()
    if not text:
        return jsonify({"result":{"ok":False,"error":"Text is empty"}, "status": ENGINE.status()})
    ok, info = ENGINE.send_signal(tier, text)
    return jsonify({"result":{"ok":ok,"info":info}, "status": ENGINE.status()})

@bp.route("/api/broadcast", methods=["POST"])
@require_admin
def api_broadcast():
    data = request.get_json(force=True, silent=True) or {}
    text = (data.get("text") or "").strip()
    if not text:
        return jsonify({"ok": False, "error":"Text is empty", "status": ENGINE.status()})
    results = {}
    for tier in ("free","basic","pro","vip"):
        ok, info = ENGINE.send_signal(tier, text)
        results[tier] = {"ok":ok, "info":info}
    return jsonify({"ok": any(r["ok"] for r in results.values()), "results": results, "status": ENGINE.status()})

# ------------------------------ Uptime probe ---------------------------------
@bp.route("/_up")
def up_check():
    return jsonify({"ok": True, "ts": datetime.now(timezone.utc).isoformat()})
