# routes.py — dashboard, config, backtests, telegram tools, live engine wiring
from __future__ import annotations

import io
import os
import csv
import json
import time
from datetime import datetime
from typing import Dict, Any, List

from flask import (
    Blueprint, request, session, redirect, url_for, render_template,
    send_from_directory, jsonify, flash, make_response
)

from utils import (
    TZ, TIMEZONE, get_config, set_config, within_window,
    convert_po_to_deriv, load_csv, fetch_deriv_history,
    compute_indicators, backtest_run, plot_signals,
)

# Live engine exports (make sure your live_engine.py exports these)
from live_engine import ENGINE, tg_test_all, get_me_diag, TIER_TO_CHAT, DAILY_CAPS

bp = Blueprint("dashboard", __name__)

ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "admin")

# ------------------------------ Helpers --------------------------------------
def require_admin(fn):
    def wrap(*a, **kw):
        if not session.get("admin"):
            return redirect(url_for("dashboard.login"))
        return fn(*a, **kw)
    wrap.__name__ = fn.__name__
    return wrap

def _now_local() -> str:
    try:
        return datetime.now(TZ).strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")

def _indicator_specs() -> Dict[str, Dict[str, Any]]:
    """
    UI specs for indicators. Keep keys in sync with utils.compute_indicators().
    Each entry can define default params shown as inputs.
    """
    return {
        # Overlays
        "SMA":       {"name":"SMA","kind":"overlay","params":{"period":50}},
        "EMA":       {"name":"EMA","kind":"overlay","params":{"period":20}},
        "WMA":       {"name":"WMA","kind":"overlay","params":{"period":20}},
        "SMMA":      {"name":"SMMA","kind":"overlay","params":{"period":20}},
        "TMA":       {"name":"TMA","kind":"overlay","params":{"period":20}},
        "BOLL":      {"name":"Bollinger Bands","kind":"overlay","params":{"period":20,"mult":2}},
        "KELTNER":   {"name":"Keltner Channel","kind":"overlay","params":{"period":20,"mult":2}},
        "DONCHIAN":  {"name":"Donchian","kind":"overlay","params":{"period":20}},
        "ENVELOPES": {"name":"Envelopes","kind":"overlay","params":{"period":20,"pct":2}},
        "ICHIMOKU":  {"name":"Ichimoku","kind":"overlay","params":{}},
        "PSAR":      {"name":"Parabolic SAR","kind":"overlay","params":{"step":0.02,"max":0.2}},
        "SUPERTREND":{"name":"SuperTrend","kind":"overlay","params":{"period":10,"mult":3}},

        # Oscillators
        "RSI":       {"name":"RSI","kind":"osc","params":{"period":14}},
        "STOCH":     {"name":"Stochastic","kind":"osc","params":{"k":14,"d":3}},
        "ATR":       {"name":"ATR","kind":"osc","params":{"period":14}},
        "ADX":       {"name":"ADX","kind":"osc","params":{"period":14}},
        "CCI":       {"name":"CCI","kind":"osc","params":{"period":20}},
        "MOMENTUM":  {"name":"Momentum","kind":"osc","params":{"period":10}},
        "ROC":       {"name":"ROC","kind":"osc","params":{"period":10}},
        "WILLR":     {"name":"Williams %R","kind":"osc","params":{"period":14}},
        "VORTEX":    {"name":"Vortex","kind":"osc","params":{"period":14}},
        "MACD":      {"name":"MACD","kind":"osc","params":{"fast":12,"slow":26,"signal":9}},
        "AO":        {"name":"Awesome Osc","kind":"osc","params":{}},
        "AC":        {"name":"Acceleration","kind":"osc","params":{}},
        "BEARS":     {"name":"Bears Power","kind":"osc","params":{"period":13}},
        "BULLS":     {"name":"Bulls Power","kind":"osc","params":{"period":13}},
        "DEMARKER":  {"name":"DeMarker","kind":"osc","params":{"period":14}},
        "OSMA":      {"name":"OsMA","kind":"osc","params":{}},
        "ZIGZAG":    {"name":"ZigZag","kind":"osc","params":{"pct":1.0}},
    }

def _available_symbol_groups():
    # Group 0: Deriv (frx*)
    deriv = [
        "frxEURUSD","frxGBPUSD","frxUSDJPY","frxUSDCHF","frxUSDCAD","frxAUDUSD","frxNZDUSD",
        "frxEURGBP","frxEURJPY","frxGBPJPY","frxEURAUD","frxAUDJPY","frxCADJPY","frxCHFJPY"
    ]
    # Group 1: Pocket Option majors
    po = [
        "EURUSD","GBPUSD","USDJPY","USDCHF","USDCAD","AUDUSD","NZDUSD",
        "EURGBP","EURJPY","GBPJPY","EURAUD","AUDJPY","CADJPY","CHFJPY"
    ]
    return [
        {"label":"Deriv Majors (frx*)","items": deriv},
        {"label":"Pocket Option Majors","items": po},
    ]

def _granularity_map():
    # Common TFs to Deriv seconds
    return {
        "M1":60, "M2":120, "M3":180, "M5":300, "M10":600, "M15":900, "M30":1800,
        "H1":3600, "H4":14400, "D1":86400
    }

def _ctx_base(view="index"):
    cfg = get_config() or {}

    # Window defaults
    window = cfg.get("window", {"start":"08:00","end":"17:00","timezone": TIMEZONE})
    live_tf = cfg.get("live_tf", "M1")
    live_expiry = cfg.get("live_expiry", "5m")
    bt_def = cfg.get("bt_defaults", {"tf": live_tf, "expiry": live_expiry})

    # Strategies incl. CUSTOM1–3 (in toggles)
    strategies: Dict[str, Dict[str, Any]] = cfg.get("strategies") or {
        "BASE":{"enabled": True},
        "TREND":{"enabled": False},
        "CHOP":{"enabled": False},
        "CUSTOM1":{"enabled": False},
        "CUSTOM2":{"enabled": False},
        "CUSTOM3":{"enabled": False},
    }

    # Customs payload (3 slots)
    customs = cfg.get("customs") or [
        {"_idx":1,"enabled":False,"mode":"SIMPLE","lookback":3,"tol_pct":0.1,
         "simple_buy":"","simple_sell":"","buy_rule":None,"sell_rule":None},
        {"_idx":2,"enabled":False,"mode":"SIMPLE","lookback":3,"tol_pct":0.1,
         "simple_buy":"","simple_sell":"","buy_rule":None,"sell_rule":None},
        {"_idx":3,"enabled":False,"mode":"SIMPLE","lookback":3,"tol_pct":0.1,
         "simple_buy":"","simple_sell":"","buy_rule":None,"sell_rule":None},
    ]

    indicators = cfg.get("indicators", {})
    symbols_raw = cfg.get("symbols_raw", [])
    active_symbols = convert_po_to_deriv(symbols_raw)

    ctx = {
        "view": view,
        "tz": TIMEZONE,
        "now": _now_local(),
        "within": within_window({"window": window}),
        "window": window,

        "available_groups": _available_symbol_groups(),
        "symbols_raw": symbols_raw,
        "active_symbols": active_symbols,

        "indicators": indicators,
        "specs": _indicator_specs(),

        "strategies": strategies,
        "strategies_all": {   # used by the toggle list
            "BASE":{}, "TREND":{}, "CHOP":{},
            "CUSTOM1":{}, "CUSTOM2":{}, "CUSTOM3":{}
        },

        "bt": {"tf": bt_def.get("tf", live_tf), "expiry": bt_def.get("expiry", live_expiry)},
        "live_tf": live_tf,
        "live_expiry": live_expiry,

        "users": cfg.get("users", []),
    }
    return ctx

# ------------------------------ Views ----------------------------------------
@bp.route("/")
def index():
    ctx = _ctx_base(view="index")
    return render_template("dashboard.html", **ctx)

@bp.route("/login", methods=["GET","POST"])
def login():
    if request.method == "POST":
        pwd = request.form.get("password","")
        if pwd == ADMIN_PASSWORD:
            session["admin"] = True
            flash("Logged in.","ok")
            return redirect(url_for("dashboard.view"))
        flash("Invalid password.","error")
    ctx = _ctx_base(view="login")
    return render_template("dashboard.html", **ctx)

@bp.route("/logout")
def logout():
    session.clear()
    flash("Logged out.","ok")
    return redirect(url_for("dashboard.index"))

@bp.route("/dashboard")
@require_admin
def view():
    ctx = _ctx_base(view="dashboard")
    return render_template("dashboard.html", **ctx)

# ------------------------------ Update: Window --------------------------------
@bp.route("/update/window", methods=["POST"])
@require_admin
def update_window():
    cfg = get_config() or {}
    start = request.form.get("start","08:00")
    end   = request.form.get("end","17:00")
    tz    = request.form.get("timezone", TIMEZONE)
    live_tf = request.form.get("live_tf", cfg.get("live_tf","M1"))
    live_exp = request.form.get("live_expiry", cfg.get("live_expiry","5m"))
    cfg["window"] = {"start": start, "end": end, "timezone": tz}
    cfg["live_tf"] = live_tf
    cfg["live_expiry"] = live_exp
    set_config(cfg)
    flash("Window & defaults saved.", "ok")
    return redirect(url_for("dashboard.view"))

# ------------------------------ Update: Symbols -------------------------------
@bp.route("/update/symbols", methods=["POST"])
@require_admin
def update_symbols():
    cfg = get_config() or {}
    text = request.form.get("symbols_text","").strip()
    sel_deriv = request.form.getlist("symbols_deriv_multi")
    sel_po = request.form.getlist("symbols_po_multi")
    convert = bool(request.form.get("convert_po"))

    raw: List[str] = []
    for chunk in [text.replace(",", " "), " ".join(sel_deriv), " ".join(sel_po)]:
        for tok in chunk.split():
            t = tok.strip()
            if t and t not in raw:
                raw.append(t)

    if convert:
        raw = convert_po_to_deriv(raw)

    cfg["symbols_raw"] = raw
    set_config(cfg)
    flash("Symbols saved.", "ok")
    return redirect(url_for("dashboard.view"))

# ---------------------------- Update: Indicators ------------------------------
@bp.route("/update/indicators", methods=["POST"])
@require_admin
def update_indicators():
    cfg = get_config() or {}
    specs = _indicator_specs()
    out: Dict[str, Dict[str, Any]] = {}
    for key, spec in specs.items():
        enabled = bool(request.form.get(f"ind_{key}_enabled"))
        block: Dict[str, Any] = {"enabled": enabled}
        params = spec.get("params") or {}
        for p_name, default in params.items():
            val = request.form.get(f"ind_{key}_{p_name}", default)
            block[p_name] = val
        out[key] = block
    cfg["indicators"] = out
    set_config(cfg)
    flash("Indicators saved.", "ok")
    return redirect(url_for("dashboard.view"))

# ---------------------------- Update: Strategies ------------------------------
@bp.route("/update/strategies", methods=["POST"])
@require_admin
def update_strategies():
    strategies = {}
    for name in ["BASE","TREND","CHOP","CUSTOM1","CUSTOM2","CUSTOM3"]:
        strategies[name] = {"enabled": bool(request.form.get(f"s_{name}"))}

    live_tf = request.form.get("live_tf","M1")
    live_expiry = request.form.get("live_expiry","5m")
    bt_tf = request.form.get("bt_tf", live_tf)
    bt_expiry = request.form.get("bt_expiry", live_expiry)

    cfg = get_config() or {}
    cfg["strategies"] = strategies
    cfg["live_tf"] = live_tf
    cfg["live_expiry"] = live_expiry
    cfg["bt_defaults"] = {"tf": bt_tf, "expiry": bt_expiry}
    set_config(cfg)
    flash("Strategies & defaults saved.", "ok")
    return redirect(url_for("dashboard.view"))

# ---------------------------- Update: Custom Slots ----------------------------
@bp.route("/update/custom", methods=["POST"])
@require_admin
def update_custom():
    cfg = get_config() or {}
    customs = cfg.get("customs") or [
        {"_idx":1},{"_idx":2},{"_idx":3}
    ]
    slot = int(request.form.get("slot","1"))
    idx = max(1, min(3, slot)) - 1

    block = customs[idx] if idx < len(customs) else {"_idx": slot}
    block["enabled"] = bool(request.form.get("enabled"))
    block["mode"] = request.form.get("mode","SIMPLE")
    block["lookback"] = int(request.form.get("lookback","3") or 3)
    block["tol_pct"] = float(request.form.get("tol_pct","0.1") or 0.1)
    block["simple_buy"] = request.form.get("simple_buy","")
    block["simple_sell"] = request.form.get("simple_sell","")

    # Optional JSON rule fields
    def _parse_json_field(name):
        raw = request.form.get(name, "").strip()
        if not raw:
            return None
        try:
            return json.loads(raw)
        except Exception:
            return {"_raw": raw}  # store as-is if malformed to avoid losing it

    block["buy_rule"] = _parse_json_field("buy_rule_json")
    block["sell_rule"] = _parse_json_field("sell_rule_json")

    # Put back
    if idx >= len(customs):
        # pad
        while len(customs) < idx:
            customs.append({"_idx": len(customs)+1})
        customs.append(block)
    else:
        customs[idx] = block

    cfg["customs"] = customs
    set_config(cfg)
    flash(f"CUSTOM {slot} saved.", "ok")
    return redirect(url_for("dashboard.view"))

# ---------------------------- Users (in-config) -------------------------------
@bp.route("/users/add", methods=["POST"])
@require_admin
def users_add():
    cfg = get_config() or {}
    users = cfg.get("users", [])
    tg_id = (request.form.get("telegram_id") or "").strip()
    tier = (request.form.get("tier") or "free").lower()
    expires_at = request.form.get("expires_at") or ""
    if not tg_id:
        flash("Telegram ID required.","error")
        return redirect(url_for("dashboard.view"))
    # upsert on telegram_id
    updated = False
    for u in users:
        if str(u.get("telegram_id")) == tg_id:
            u["tier"] = tier; u["expires_at"] = expires_at
            updated = True; break
    if not updated:
        users.append({"telegram_id": tg_id, "tier": tier, "expires_at": expires_at})
    cfg["users"] = users
    set_config(cfg)
    flash("User saved.","ok")
    return redirect(url_for("dashboard.view"))

@bp.route("/users/delete", methods=["POST"])
@require_admin
def users_delete():
    cfg = get_config() or {}
    users = cfg.get("users", [])
    tg_id = (request.form.get("telegram_id") or "").strip()
    users = [u for u in users if str(u.get("telegram_id")) != tg_id]
    cfg["users"] = users
    set_config(cfg)
    flash("User deleted.","ok")
    return redirect(url_for("dashboard.view"))

# ----------------------------- Backtest + assets ------------------------------
_last_bt_json: Dict[str, Any] = {}
_last_bt_csv: str = ""

@bp.route("/backtest", methods=["POST"])
@require_admin
def backtest():
    cfg = get_config() or {}
    indicators = cfg.get("indicators", {})
    strategies = cfg.get("strategies", {})

    # Parse inputs
    symbols_text = request.form.get("bt_symbols","").strip()
    convert_bt = bool(request.form.get("convert_po_bt"))
    tf = request.form.get("bt_tf", cfg.get("bt_defaults",{}).get("tf", cfg.get("live_tf","M1")))
    expiry = request.form.get("bt_expiry", cfg.get("bt_defaults",{}).get("expiry", cfg.get("live_expiry","5m")))
    strategy = request.form.get("bt_strategy", "BASE").upper()

    # Strategy must be one of list incl CUSTOMs
    if strategy not in ["BASE","TREND","CHOP","CUSTOM1","CUSTOM2","CUSTOM3"]:
        strategy = "BASE"

    # Symbols
    symbols = []
    for tok in symbols_text.replace(",", " ").split():
        t = tok.strip()
        if t and t not in symbols:
            symbols.append(t)
    if not symbols:
        # fallback to configured active symbols if any
        symbols = cfg.get("symbols_raw", [])
    if convert_bt:
        symbols = convert_po_to_deriv(symbols)

    # Data source: CSV upload OR Deriv fetch
    use_server = bool(request.form.get("use_server"))
    file = request.files.get("bt_csv")
    df_all = None
    try:
        if (file is not None) and file.filename:
            df_all = load_csv(file)
        elif use_server:
            gmap = _granularity_map()
            gran = gmap.get(tf, 300)
            # Pick first symbol only for screenshot (consistent with previous flow)
            if not symbols:
                raise RuntimeError("No symbols to fetch.")
            sym = symbols[0]
            # Some env allow uppercase FRX*; convert to canonical 'frx*'
            if sym.upper().startswith("FRX") and not sym.startswith("frx"):
                sym = "frx" + sym[3:]
            df_all = fetch_deriv_history(sym, granularity_sec=gran, count=600)
        else:
            raise RuntimeError("Upload a CSV or check 'Use Deriv server fetch'.")
    except Exception as e:
        flash(f"Backtest error: {e}", "error")
        return redirect(url_for("dashboard.view"))

    try:
        # Compute signals & stats
        signals, stats = backtest_run(df_all, strategy, indicators, expiry)
        plot_name = plot_signals(df_all, signals, indicators, strategy, tf, expiry)

        # Prepare last JSON + CSV
        global _last_bt_json, _last_bt_csv
        _last_bt_json = {
            "tf": tf, "expiry": expiry, "strategy": strategy,
            "summary": stats, "count_rows": int(len(df_all)),
            "symbols": symbols[:],
            "warnings": [],
            "plot_name": plot_name,
        }
        # Minimal CSV dump (index + Close)
        out_io = io.StringIO()
        w = csv.writer(out_io)
        w.writerow(["time","open","high","low","close"])
        for ts, row in df_all.iterrows():
            w.writerow([ts.isoformat(), row["Open"], row["High"], row["Low"], row["Close"]])
        _last_bt_csv = out_io.getvalue()

        flash("Backtest completed.", "ok")
        ctx = _ctx_base(view="dashboard")
        ctx["bt"] = {
            "tf": tf, "expiry": expiry, "strategy": strategy,
            "summary": json.dumps(stats), "plot_name": plot_name,
            "warnings": []
        }
        return render_template("dashboard.html", **ctx)
    except Exception as e:
        flash(f"Backtest error: {e}", "error")
        return redirect(url_for("dashboard.view"))

@bp.route("/plot/<name>")
def plot_file(name):
    return send_from_directory(os.path.join("static","plots"), name, as_attachment=False)

@bp.route("/backtest/last.json")
def backtest_last_json():
    resp = make_response(json.dumps(_last_bt_json or {}, indent=2))
    resp.mimetype = "application/json"
    return resp

@bp.route("/backtest/last.csv")
def backtest_last_csv():
    resp = make_response(_last_bt_csv or "")
    resp.mimetype = "text/csv"
    return resp

# -------------------------- Live Engine (UI uses these) ----------------------
@bp.route("/live/start")
@require_admin
def live_start():
    ok, info = ENGINE.start()
    return jsonify({"ok": ok, "info": info, "status": ENGINE.status()})

@bp.route("/live/stop")
@require_admin
def live_stop():
    ok, info = ENGINE.stop()
    return jsonify({"ok": ok, "info": info, "status": ENGINE.status()})

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

@bp.route("/live/status")
def live_status():
    return jsonify({"status": ENGINE.status()})

@bp.route("/live/tally")
def live_tally():
    return jsonify({"tally": ENGINE.tally()})

# ------------------------------- API aliases ---------------------------------
@bp.route("/api/status")
def api_status():
    s = ENGINE.status()
    # adapt keys expected by alt UI (free/basic/pro/vip + total)
    t = s.get("tally", {})
    tallies = {
        "free": t.get("by_tier",{}).get("free",0),
        "basic": t.get("by_tier",{}).get("basic",0),
        "pro": t.get("by_tier",{}).get("pro",0),
        "vip": t.get("by_tier",{}).get("vip",0),
        "total": t.get("total",0)
    }
    return jsonify({
        "running": s.get("running"),
        "debug": s.get("debug"),
        "day": s.get("day"),
        "tallies": tallies,
        "caps": {
            "free": DAILY_CAPS.get("free"),
            "basic": DAILY_CAPS.get("basic"),
            "pro": DAILY_CAPS.get("pro"),
            "vip": float("inf") if DAILY_CAPS.get("vip") is None else DAILY_CAPS.get("vip")
        },
        "configured_chats": {k: bool(v) for k, v in TIER_TO_CHAT.items()},
        "last_send_result": s.get("last_send") or {},
    })

@bp.route("/api/check_bot")
def api_check_bot():
    return jsonify(get_me_diag())

@bp.route("/api/test/vip", methods=["POST"])
@require_admin
def api_test_vip():
    text = (request.json or {}).get("text") or "VIP test ping"
    ok, info = ENGINE.send_signal("vip", text)
    return jsonify({"ok": ok, "info": info, "status": ENGINE.status()})

@bp.route("/api/send", methods=["POST"])
@require_admin
def api_send():
    data = request.json or {}
    tier = (data.get("tier") or "vip").lower()
    text = (data.get("text") or "").strip()
    if not text:
        return jsonify({"ok": False, "result": {"ok": False, "error": "empty text"}}), 400
    ok, info = ENGINE.send_signal(tier, text)
    return jsonify({"ok": ok, "result": {"ok": ok, "info": info}, "status": ENGINE.status()})

# ----------------------------- Telegram tools --------------------------------
@bp.route("/telegram/test", methods=["POST"])
@require_admin
def telegram_test():
    diag = tg_test_all()
    return jsonify(diag)

@bp.route("/telegram/diag")
def telegram_diag():
    return jsonify(tg_test_all())

# ----------------------------- Deriv fetch stub ------------------------------
@bp.route("/deriv/fetch", methods=["POST"])
@require_admin
def deriv_fetch():
    """
    Optional helper endpoint referenced by the template.
    If you have a separate data_fetch.py with fetch_one_symbol/deriv_csv_path,
    wire it here. For now we just acknowledge and avoid template BuildError.
    """
    flash("Deriv fetch helper is a stub in this build. Use Backtest with 'Use Deriv server fetch' or CSV upload.", "ok")
    return redirect(url_for("dashboard.view"))

# ----------------------------- Uptime check ----------------------------------
@bp.route("/_up")
def up_check():
    return jsonify({"ok": True, "ts": int(time.time())})
