# routes.py
# Flask routes + dashboard for Pocket-Option Signals

from __future__ import annotations
import os
import io
import json
import time
from datetime import datetime
from typing import Dict, Any, List, Tuple, Optional

from flask import (
    Blueprint, render_template, request, redirect, url_for, flash, session,
    send_from_directory, jsonify, current_app
)

# ---- Local modules -----------------------------------------------------------
from utils import (
    TIMEZONE as UTZ, TZ, within_window,
    get_config, set_config,
    convert_po_to_deriv, load_csv, fetch_deriv_history,
    compute_indicators, simple_rule_engine, evaluate_signals_outcomes,
    plot_signals, backtest_run
)
from live_engine import ENGINE, tg_test, BOT_TOKEN, TIER_TO_CHAT, DAILY_CAPS

# -----------------------------------------------------------------------------
bp = Blueprint("dashboard", __name__)

ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "admin")

# ---------- helpers -----------------------------------------------------------
def _now_str() -> str:
    try:
        return datetime.now(TZ).strftime("%Y-%m-%d %H:%M")
    except Exception:
        return datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")

def admin_required(fn):
    def wrap(*a, **kw):
        if not session.get("admin"):
            flash("Login required", "error")
            return redirect(url_for("dashboard.login"))
        return fn(*a, **kw)
    wrap.__name__ = fn.__name__
    wrap.__doc__ = fn.__doc__
    return wrap

# ---------- basic / auth ------------------------------------------------------
@bp.get("/_up")
def up_check():
    return jsonify({"ok": True, "ts": int(time.time())})

@bp.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        pw = request.form.get("password", "")
        if pw == ADMIN_PASSWORD:
            session["admin"] = True
            return redirect(url_for("dashboard.view"))
        flash("Bad password", "error")
    ctx = _ctx_base()
    ctx["view"] = "login"
    return render_template("dashboard.html", **ctx)

@bp.get("/logout")
def logout():
    session.clear()
    return redirect(url_for("dashboard.index"))

@bp.get("/")
def index():
    cfg = get_config() or {}
    ctx = _ctx_base(cfg)
    ctx["view"] = "index"
    return render_template("dashboard.html", **ctx)

# ---------- main dashboard ----------------------------------------------------
@bp.get("/dashboard")
def view():
    cfg = get_config() or {}
    ctx = _ctx_base(cfg)
    ctx["view"] = "dashboard"
    return render_template("dashboard.html", **ctx)

def _ctx_base(cfg: Optional[dict] = None) -> dict:
    cfg = cfg or get_config() or {}

    # Indicator specs: if you keep specs elsewhere you can inject here
    specs = cfg.get("indicator_specs") or {
        # Minimal defaults; template iterates this structure
        "SMA":   {"name":"Simple MA",      "kind":"overlay",   "params":{"period":50}},
        "EMA":   {"name":"Exponential MA", "kind":"overlay",   "params":{"period":20}},
        "WMA":   {"name":"Weighted MA",    "kind":"overlay",   "params":{"period":20}},
        "SMMA":  {"name":"Smoothed MA",    "kind":"overlay",   "params":{"period":20}},
        "TMA":   {"name":"Triangular MA",  "kind":"overlay",   "params":{"period":20}},
        "RSI":   {"name":"RSI",            "kind":"oscillator","params":{"period":14, "show": True}},
        "STOCH": {"name":"Stochastic",     "kind":"oscillator","params":{"k":14,"d":3,"show": True}},
        "ATR":   {"name":"ATR",            "kind":"oscillator","params":{"period":14, "show": False}},
        "ADX":   {"name":"ADX/+DI/-DI",    "kind":"oscillator","params":{"period":14, "show": False}},
    }

    strategies_saved = cfg.get("strategies", {})  # SOURCE OF TRUTH
    strategies_all = {
        # Catalog only (labels); no defaults that flip boxes on
        "BASE":    {"label":"BASE"},
        "TREND":   {"label":"TREND"},
        "CHOP":    {"label":"CHOP"},
        "CUSTOM1": {"label":"CUSTOM1"},
        "CUSTOM2": {"label":"CUSTOM2"},
        "CUSTOM3": {"label":"CUSTOM3"},
    }

    available_groups = cfg.get("available_groups") or [
        {"title": "Deriv (frx*)", "items": [
            "frxEURUSD","frxGBPUSD","frxUSDJPY","frxUSDCHF","frxUSDCAD","frxAUDUSD","frxNZDUSD",
            "frxEURGBP","frxEURJPY","frxGBPJPY","frxEURAUD","frxAUDJPY","frxCADJPY","frxCHFJPY"
        ]},
        {"title": "PO majors", "items": [
            "EURUSD","GBPUSD","USDJPY","USDCHF","USDCAD","AUDUSD","NZDUSD",
            "EURGBP","EURJPY","GBPJPY","EURAUD","AUDJPY","CADJPY","CHFJPY"
        ]},
    ]

    ctx = {
        "tz": str(UTZ),
        "now": _now_str(),
        "within": within_window(cfg),
        "window": cfg.get("window", {}),
        "live_tf": cfg.get("live_tf", "M1"),
        "live_expiry": cfg.get("live_expiry", "5m"),
        "bt": cfg.get("last_bt", {}),
        "indicators": cfg.get("indicators", {}),
        "specs": specs,
        "strategies": strategies_saved,
        "strategies_all": strategies_all,
        "active_symbols": cfg.get("active_symbols", []),
        "symbols_raw": cfg.get("symbols_raw", []),
        "available_groups": available_groups,
        "customs": _inflate_customs(cfg.get("customs", [])),
        "users": cfg.get("users", []),
        "session": {"admin": bool(session.get("admin"))},
    }
    return ctx

def _inflate_customs(items: List[dict]) -> List[dict]:
    out = []
    for i in range(3):
        got = items[i] if i < len(items) else {}
        got = dict(got)
        got["_idx"] = i + 1
        out.append(got)
    return out

# ---------- window defaults ---------------------------------------------------
@bp.post("/window/save")
@admin_required
def update_window():
    cfg = get_config() or {}
    window = cfg.get("window", {})
    window["start"] = request.form.get("start") or window.get("start","08:00")
    window["end"] = request.form.get("end") or window.get("end","17:00")
    window["timezone"] = request.form.get("timezone") or window.get("timezone", str(UTZ))
    cfg["window"] = window

    cfg["live_tf"] = (request.form.get("live_tf") or cfg.get("live_tf","M1")).upper()
    cfg["live_expiry"] = (request.form.get("live_expiry") or cfg.get("live_expiry","5m")).lower()

    set_config(cfg)
    flash("Window/defaults saved", "ok")
    return redirect(url_for("dashboard.view"))

# ---------- symbols -----------------------------------------------------------
@bp.post("/symbols/save")
@admin_required
def update_symbols():
    cfg = get_config() or {}
    active = set(cfg.get("active_symbols", []))
    raw = set(cfg.get("symbols_raw", []))

    deriv_multi = request.form.getlist("symbols_deriv_multi")
    po_multi = request.form.getlist("symbols_po_multi")
    text = (request.form.get("symbols_text") or "").replace(",", " ").split()
    convert = bool(request.form.get("convert_po"))

    chosen = list(set(deriv_multi + po_multi + text))
    raw = chosen[:]  # keep what user typed/selected for display

    # Normalize to Deriv symbols if requested
    if convert:
        chosen = convert_po_to_deriv(chosen)

    cfg["active_symbols"] = chosen
    cfg["symbols_raw"] = raw
    set_config(cfg)
    flash(f"Saved {len(chosen)} symbol(s).", "ok")
    return redirect(url_for("dashboard.view"))

# ---------- indicators --------------------------------------------------------
@bp.post("/indicators/save")
@admin_required
def update_indicators():
    cfg = get_config() or {}
    specs = cfg.get("indicator_specs") or {}   # leave existing descriptions

    # Derive enabled + params from form
    result: Dict[str, dict] = {}
    def upd(key: str, params: List[str]):
        enabled = bool(request.form.get(f"ind_{key}_enabled"))
        entry = {"enabled": enabled}
        for p in params:
            form_key = f"ind_{key}_{p}"
            if form_key in request.form:
                val = request.form.get(form_key)
                # cast numerics when possible
                try:
                    if val is None: pass
                    elif val.strip() == "": pass
                    elif "." in val: val = float(val)
                    else: val = int(val)
                except Exception:
                    pass
                entry[p] = val
        result[key] = entry

    # Accept keys known in specs or from a default list
    keys = list((specs or {}).keys()) or ["SMA","EMA","WMA","SMMA","TMA","RSI","STOCH","ATR","ADX"]
    for k in keys:
        # collect param names if present in specs
        params = list(((specs.get(k) or {}).get("params") or {}).keys())
        # if specs absent, fall back to common fields
        if not params:
            if k == "RSI": params = ["period","show"]
            elif k == "STOCH": params = ["k","d","show"]
            elif k in ("ATR","ADX"): params = ["period","show"]
            else: params = ["period"]
        upd(k, params)

    cfg["indicators"] = result
    set_config(cfg)
    flash("Indicators saved.", "ok")
    return redirect(url_for("dashboard.view"))

# ---------- strategies (FIX: no default re-enable) ---------------------------
@bp.post("/strategies/save")
@admin_required
def update_strategies():
    cfg = get_config() or {}
    names = ["BASE","TREND","CHOP","CUSTOM1","CUSTOM2","CUSTOM3"]

    new_state = {}
    for n in names:
        new_state[n] = {"enabled": bool(request.form.get(f"s_{n}"))}
    cfg["strategies"] = new_state

    # Also allow defaults on this form
    if "bt_tf" in request.form:     cfg["bt_tf"] = (request.form.get("bt_tf") or cfg.get("bt_tf","M1")).upper()
    if "bt_expiry" in request.form: cfg["bt_expiry"] = (request.form.get("bt_expiry") or cfg.get("bt_expiry","5m")).lower()
    if "live_tf" in request.form:   cfg["live_tf"] = (request.form.get("live_tf") or cfg.get("live_tf","M1")).upper()
    if "live_expiry" in request.form: cfg["live_expiry"] = (request.form.get("live_expiry") or cfg.get("live_expiry","5m")).lower()

    set_config(cfg)
    flash("Strategy toggles saved.", "ok")
    return redirect(url_for("dashboard.view"))

# ---------- custom rule slots -------------------------------------------------
@bp.post("/custom/save")
@admin_required
def update_custom():
    slot = int(request.form.get("slot", "1"))
    cfg = get_config() or {}
    customs = cfg.get("customs", [])
    while len(customs) < 3:
        customs.append({})

    enabled = bool(request.form.get("enabled"))
    mode = request.form.get("mode","SIMPLE").upper()
    lookback = int(request.form.get("lookback") or 3)
    tol_pct = float(request.form.get("tol_pct") or 0.1)
    simple_buy = request.form.get("simple_buy","")
    simple_sell = request.form.get("simple_sell","")
    buy_rule_json = request.form.get("buy_rule_json","")
    sell_rule_json = request.form.get("sell_rule_json","")

    entry = {
        "enabled": enabled, "mode": mode, "lookback": lookback, "tol_pct": tol_pct,
        "simple_buy": simple_buy, "simple_sell": simple_sell
    }
    try:
        if buy_rule_json.strip():
            entry["buy_rule"] = json.loads(buy_rule_json)
        else:
            entry.pop("buy_rule", None)
    except Exception:
        flash(f"CUSTOM{slot} buy_rule JSON invalid; ignored", "error")
    try:
        if sell_rule_json.strip():
            entry["sell_rule"] = json.loads(sell_rule_json)
        else:
            entry.pop("sell_rule", None)
    except Exception:
        flash(f"CUSTOM{slot} sell_rule JSON invalid; ignored", "error")

    customs[slot-1] = entry
    cfg["customs"] = customs
    set_config(cfg)
    flash(f"CUSTOM {slot} saved.", "ok")
    return redirect(url_for("dashboard.view"))

# ---------- users -------------------------------------------------------------
@bp.post("/users/add")
@admin_required
def users_add():
    cfg = get_config() or {}
    users = cfg.get("users", [])
    tid = (request.form.get("telegram_id") or "").strip()
    tier = (request.form.get("tier") or "free").lower()
    exp = request.form.get("expires_at") or ""

    if not tid:
        flash("Telegram ID required", "error")
        return redirect(url_for("dashboard.view"))

    # upsert
    found = None
    for u in users:
        if str(u.get("telegram_id")) == str(tid):
            found = u; break
    if found:
        found["tier"] = tier; found["expires_at"] = exp
    else:
        users.append({"telegram_id": tid, "tier": tier, "expires_at": exp})
    cfg["users"] = users
    set_config(cfg)
    flash("User saved.", "ok")
    return redirect(url_for("dashboard.view"))

@bp.post("/users/delete")
@admin_required
def users_delete():
    cfg = get_config() or {}
    users = cfg.get("users", [])
    tid = (request.form.get("telegram_id") or "").strip()
    users = [u for u in users if str(u.get("telegram_id")) != str(tid)]
    cfg["users"] = users
    set_config(cfg)
    flash("User deleted.", "ok")
    return redirect(url_for("dashboard.view"))

# ---------- backtest ----------------------------------------------------------
@bp.post("/backtest")
@admin_required
def backtest():
    cfg = get_config() or {}
    use_server = bool(request.form.get("use_server"))
    tf = (request.form.get("bt_tf") or cfg.get("bt_tf") or cfg.get("live_tf","M1")).upper()
    expiry = (request.form.get("bt_expiry") or cfg.get("bt_expiry") or cfg.get("live_expiry","5m")).lower()
    strategy = (request.form.get("bt_strategy") or "BASE").upper()
    count = int(request.form.get("bt_count") or 300)

    syms_raw = (request.form.get("bt_symbols") or "frxEURUSD").replace(",", " ").split()
    if bool(request.form.get("convert_po_bt")):
        syms = convert_po_to_deriv(syms_raw)
    else:
        syms = syms_raw[:]

    # Only the first symbol is plotted per-run (simple dashboard UX)
    sym = syms[0] if syms else "frxEURUSD"

    # Map TF -> seconds
    tf_map = {
        "M1":60,"M2":120,"M3":180,"M5":300,"M10":600,"M15":900,"M30":1800,
        "H1":3600,"H4":14400,"D1":86400
    }
    gran = tf_map.get(tf, 60)

    try:
        # Load data
        df = None
        file = request.files.get("bt_csv")
        if file and file.filename:
            # CSV upload
            df = load_csv(io.BytesIO(file.read()))
        elif use_server:
            df = fetch_deriv_history(sym, granularity_sec=gran, days=5)
        else:
            raise RuntimeError("No data source selected. Upload CSV or tick 'Use Deriv server fetch'.")

        # Indicators config
        indicators = cfg.get("indicators", {})
        # Backtest engine
        signals, stats = backtest_run(df, strategy, indicators, expiry)

        # Plot
        plot_name = plot_signals(df, signals, indicators, strategy, tf, expiry)

        summary = f"W:{stats['wins']} L:{stats['loss']} D:{stats['draw']} | Win%={stats['win_rate']:.1f}%"
        result = {
            "tf": tf, "expiry": expiry, "strategy": strategy,
            "plot_name": plot_name, "summary": summary, "warnings": []
        }
        cfg["last_bt"] = result
        set_config(cfg)
        flash("Backtest complete.", "ok")
        return redirect(url_for("dashboard.view"))
    except Exception as e:
        cfg["last_bt"] = {"error": str(e), "tf": tf, "expiry": expiry, "strategy": strategy}
        set_config(cfg)
        flash(f"Backtest error: {e}", "error")
        return redirect(url_for("dashboard.view"))

@bp.get("/plot/<name>")
def plot_file(name: str):
    return send_from_directory("static/plots", name, as_attachment=False)

@bp.get("/backtest/last.json")
def backtest_last_json():
    cfg = get_config() or {}
    return jsonify(cfg.get("last_bt") or {})

@bp.get("/backtest/last.csv")
def backtest_last_csv():
    # stub: you can persist the last df if you want
    return jsonify({"ok": False, "error": "Not implemented"})

# ---------- Live engine (UI uses /live/*) ------------------------------------
@bp.get("/live/status")
def live_status():
    s = ENGINE.status()
    return jsonify({"status": {"running": s["running"], "debug": s["debug"], "loop_sleep": s["loop_sleep"]}})

@bp.get("/live/tally")
def live_tally():
    t = ENGINE.tally()
    by = t["by_tier"]
    return jsonify({"tally": {
        "free": by.get("free",0), "basic": by.get("basic",0),
        "pro": by.get("pro",0), "vip": by.get("vip",0),
        "all": t.get("total",0)
    }})

@bp.get("/live/start")
@admin_required
def live_start():
    ok, info = ENGINE.start()
    return jsonify({"ok": ok, "info": info})

@bp.get("/live/stop")
@admin_required
def live_stop():
    ok, info = ENGINE.stop()
    return jsonify({"ok": ok, "info": info})

@bp.get("/live/debug/on")
@admin_required
def live_debug_on():
    ENGINE.debug = True
    return jsonify({"ok": True, "debug": True})

@bp.get("/live/debug/off")
@admin_required
def live_debug_off():
    ENGINE.debug = False
    return jsonify({"ok": True, "debug": False})

# ---------- API used by alternative front-end (kept for compatibility) -------
@bp.get("/api/status")
def api_status():
    t = ENGINE.tally()
    by = t["by_tier"]
    s = ENGINE.status()
    return jsonify({
        "running": s["running"],
        "debug": s["debug"],
        "loop_sleep": s["loop_sleep"],
        "tallies": {
            "free": by.get("free",0), "basic": by.get("basic",0),
            "pro": by.get("pro",0), "vip": by.get("vip",0),
            "total": t.get("total",0)
        },
        "caps": {
            "free": DAILY_CAPS["free"],
            "basic": DAILY_CAPS["basic"],
            "pro": DAILY_CAPS["pro"],
            "vip": float("inf") if DAILY_CAPS["vip"] is None else DAILY_CAPS["vip"],
        },
        "configured_chats": {
            "free": bool(TIER_TO_CHAT.get("free")),
            "basic": bool(TIER_TO_CHAT.get("basic")),
            "pro": bool(TIER_TO_CHAT.get("pro")),
            "vip": bool(TIER_TO_CHAT.get("vip")),
        },
        "last_send_result": s.get("last_send_result", {}),
        "day": s.get("day","")
    })

@bp.post("/api/send")
@admin_required
def api_send():
    data = request.get_json(force=True, silent=True) or {}
    tier = (data.get("tier") or "vip").lower()
    text = (data.get("text") or "").strip()
    if not text:
        return jsonify({"ok": False, "error": "empty text"}), 400
    ok, info = ENGINE.send_signal(tier, text)
    return jsonify({"ok": ok, "result": {"ok": ok, "info": info}, "status": ENGINE.status()})

@bp.get("/api/check_bot")
def api_check_bot():
    # Minimal getMe/diag
    info = {"ok": bool(BOT_TOKEN), "configured_chats": {k: bool(v) for k,v in TIER_TO_CHAT.items()}}
    return jsonify(info)

@bp.post("/api/test/vip")
@admin_required
def api_test_vip():
    text = (request.get_json(force=True, silent=True) or {}).get("text") or "Test VIP"
    ok, info = ENGINE.send_signal("vip", text)
    return jsonify({"ok": ok, "result": {"ok": ok, "info": info}, "status": ENGINE.status()})

# ---------- Telegram diag / test-all -----------------------------------------
@bp.post("/telegram/test")
@admin_required
def telegram_test():
    ok, info_json = tg_test()   # sends a short message to each configured tier respecting their chat ids
    flash(f"Telegram test: {info_json}", "ok" if ok else "error")
    return redirect(url_for("dashboard.view"))

@bp.get("/telegram/diag")
def telegram_diag():
    return jsonify({
        "token_present": bool(BOT_TOKEN),
        "configured_chats": TIER_TO_CHAT,
        "caps": DAILY_CAPS
    })
