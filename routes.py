# routes.py — Flask routes for dashboard, config, backtests, Telegram + Live Engine
from __future__ import annotations
import io, os, json, time, csv
from datetime import datetime, timezone
from typing import Dict, Any, List

from flask import (
    Blueprint, render_template, request, redirect, url_for,
    session, send_file, jsonify, flash
)

from utils import (
    TZ, get_config, set_config, within_window, convert_po_to_deriv,
    load_csv, fetch_deriv_history,
    compute_indicators, backtest_run, plot_signals,
    PO_PAIRS, DERIV_PAIRS, EXPIRY_TO_BARS, trading_open_now
)

# Live engine bits
from live_engine import ENGINE, tg_test, BOT_TOKEN, TIER_TO_CHAT, DAILY_CAPS

bp = Blueprint("dashboard", __name__)

# ------------------------------- Helpers -------------------------------------
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "admin")

def _req_admin(fn):
    def wrap(*a, **kw):
        if not session.get("admin"):
            return redirect(url_for("dashboard.login"))
        return fn(*a, **kw)
    wrap.__name__ = fn.__name__
    return wrap

def _now_text():
    try:
        return datetime.now(TZ).strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")

# Specs for indicators (keys must match utils.compute_indicators)
SPECS: Dict[str, Dict[str, Any]] = {
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
    "SUPERTREND":{"name":"SuperTrend","kind":"overlay","params":{"period":10,"mult":3}},
    # Oscillators / panels
    "RSI":{"name":"RSI","kind":"oscillator","params":{"period":14}},
    "STOCH":{"name":"Stochastic","kind":"oscillator","params":{"k":14,"d":3}},
    "ATR":{"name":"ATR","kind":"oscillator","params":{"period":14}},
    "ADX":{"name":"ADX","kind":"oscillator","params":{"period":14}},
    "CCI":{"name":"CCI","kind":"oscillator","params":{"period":20}},
    "MOMENTUM":{"name":"Momentum","kind":"oscillator","params":{"period":10}},
    "ROC":{"name":"ROC","kind":"oscillator","params":{"period":10}},
    "WILLR":{"name":"Williams %R","kind":"oscillator","params":{"period":14}},
    "VORTEX":{"name":"Vortex","kind":"oscillator","params":{"period":14}},
    "MACD":{"name":"MACD","kind":"oscillator","params":{"fast":12,"slow":26,"signal":9}},
    "AO":{"name":"Awesome Osc","kind":"oscillator","params":{}},
    "AC":{"name":"Accel/Decel","kind":"oscillator","params":{}},
    "BEARS":{"name":"Bears Power","kind":"oscillator","params":{"period":13}},
    "BULLS":{"name":"Bulls Power","kind":"oscillator","params":{"period":13}},
    "DEMARKER":{"name":"DeMarker","kind":"oscillator","params":{"period":14}},
    "OSMA":{"name":"OSMA","kind":"oscillator","params":{}},
    "ZIGZAG":{"name":"ZigZag","kind":"overlay","params":{"pct":1.0}},
}

# All strategies including custom toggles
DEFAULT_STRATEGIES = {
    "BASE":{"enabled":True},
    "TREND":{"enabled":True},
    "CHOP":{"enabled":True},
    "CUSTOM1":{"enabled":False},
    "CUSTOM2":{"enabled":False},
    "CUSTOM3":{"enabled":False},
}

def _default_config() -> Dict[str, Any]:
    return {
        "window":{"start":"08:00","end":"17:00","timezone":str(TZ), "monday_friday_only": True},
        "live_tf":"M1","live_expiry":"5m",
        "bt":{"tf":"M1","expiry":"5m","strategy":"BASE"},
        "symbols_raw": ["frxEURUSD","frxGBPUSD"],
        "active_symbols": ["frxEURUSD","frxGBPUSD"],
        "indicators":{"RSI":{"enabled":True,"period":14},"STOCH":{"enabled":True,"k":14,"d":3},"SMA":{"enabled":True,"period":50}},
        "strategies": DEFAULT_STRATEGIES.copy(),
        "customs": [
            {"_idx":1, "enabled":False, "mode":"SIMPLE", "lookback":3, "tol_pct":0.1,
             "simple_buy":"RSI < 30 AND STOCH_K cross up",
             "simple_sell":"RSI > 70 AND STOCH_K cross down"},
            {"_idx":2, "enabled":False, "mode":"SIMPLE", "lookback":3, "tol_pct":0.1,
             "simple_buy":"", "simple_sell":""},
            {"_idx":3, "enabled":False, "mode":"SIMPLE", "lookback":3, "tol_pct":0.1,
             "simple_buy":"", "simple_sell":""},
        ],
    }

def _cfg() -> Dict[str, Any]:
    cfg = get_config() or {}
    if not cfg:
        cfg = _default_config()
        set_config(cfg)
    # Ensure required keys exist
    cfg.setdefault("strategies", DEFAULT_STRATEGIES.copy())
    cfg.setdefault("customs", _default_config()["customs"])
    cfg.setdefault("indicators", {})
    cfg.setdefault("window", _default_config()["window"])
    return cfg

# ------------------------------- Views ---------------------------------------
@bp.route("/")
def index():
    cfg = _cfg()
    return render_template("dashboard.html",
        view="index",
        tz=str(TZ),
        now=_now_text(),
        within=within_window(cfg),
    )

@bp.route("/login", methods=["GET","POST"])
def login():
    if request.method == "POST":
        if request.form.get("password") == ADMIN_PASSWORD:
            session["admin"] = True
            return redirect(url_for("dashboard.view"))
        flash("Wrong password", "error")
    return render_template("dashboard.html", view="login", tz=str(TZ), now=_now_text())

@bp.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("dashboard.index"))

@bp.route("/dashboard")
@_req_admin
def view():
    cfg = _cfg()
    ctx = {
        "view":"dashboard",
        "tz":str(TZ),
        "now":_now_text(),
        "within":within_window(cfg),
        "window":cfg.get("window"),
        "live_tf":cfg.get("live_tf","M1"),
        "live_expiry":cfg.get("live_expiry","5m"),
        "bt":cfg.get("bt"),
        "available_groups":[
            {"label":"Deriv","items":DERIV_PAIRS},
            {"label":"PO","items":PO_PAIRS},
        ],
        "active_symbols":cfg.get("active_symbols",[]),
        "symbols_raw":cfg.get("symbols_raw",[]),
        "indicators":cfg.get("indicators",{}),
        "specs":SPECS,
        "strategies_all":DEFAULT_STRATEGIES,
        "strategies":cfg.get("strategies",{}),
        "customs":cfg.get("customs",[]),
        "users": _load_users()
    }
    return render_template("dashboard.html", **ctx)

# ------------------------------- Config updates -------------------------------
@bp.route("/update/window", methods=["POST"])
@_req_admin
def update_window():
    cfg = _cfg()
    w = cfg.get("window", {})
    w["start"] = request.form.get("start","08:00")
    w["end"] = request.form.get("end","17:00")
    w["timezone"] = request.form.get("timezone", str(TZ))
    # enforce weekdays only
    w["monday_friday_only"] = True
    cfg["window"] = w
    cfg["live_tf"] = request.form.get("live_tf", cfg.get("live_tf","M1"))
    cfg["live_expiry"] = request.form.get("live_expiry", cfg.get("live_expiry","5m"))
    set_config(cfg)
    flash("Saved trading window & defaults", "ok")
    return redirect(url_for("dashboard.view"))

@bp.route("/update/symbols", methods=["POST"])
@_req_admin
def update_symbols():
    cfg = _cfg()
    raw_text = request.form.get("symbols_text","").replace(",", " ")
    parts = [p for p in raw_text.split() if p.strip()]
    if request.form.get("convert_po"):
        parts = convert_po_to_deriv(parts)
    # also include multi-selects
    for field in ("symbols_deriv_multi","symbols_po_multi"):
        items = request.form.getlist(field)
        if items:
            parts.extend(items)
    # de-dup, keep frx*
    seen = []
    for s in parts:
        if s not in seen: seen.append(s)
    cfg["symbols_raw"] = seen
    cfg["active_symbols"] = seen
    set_config(cfg)
    flash("Saved symbols", "ok")
    return redirect(url_for("dashboard.view"))

@bp.route("/update/indicators", methods=["POST"])
@_req_admin
def update_indicators():
    cfg = _cfg()
    new_ind = {}
    for key, spec in SPECS.items():
        enabled = request.form.get(f"ind_{key}_enabled") == "on"
        entry = {"enabled": enabled}
        for p_name, p_def in (spec.get("params") or {}).items():
            v = request.form.get(f"ind_{key}_{p_name}", p_def)
            entry[p_name] = v
        new_ind[key] = entry
    cfg["indicators"] = new_ind
    set_config(cfg)
    flash("Saved indicators", "ok")
    return redirect(url_for("dashboard.view"))

@bp.route("/update/strategies", methods=["POST"])
@_req_admin
def update_strategies():
    cfg = _cfg()
    st = {}
    for name in DEFAULT_STRATEGIES.keys():
        st[name] = {"enabled": (request.form.get(f"s_{name}") == "on")}
    cfg["strategies"] = st
    # Defaults for TF / expiry for both live and backtest areas
    cfg["bt"] = {
        "tf": request.form.get("bt_tf", cfg.get("bt",{}).get("tf","M1")),
        "expiry": request.form.get("bt_expiry", cfg.get("bt",{}).get("expiry","5m")),
        "strategy": cfg.get("bt",{}).get("strategy", "BASE")
    }
    cfg["live_tf"] = request.form.get("live_tf", cfg.get("live_tf","M1"))
    cfg["live_expiry"] = request.form.get("live_expiry", cfg.get("live_expiry","5m"))
    set_config(cfg)
    flash("Saved strategies & defaults", "ok")
    return redirect(url_for("dashboard.view"))

@bp.route("/update/custom", methods=["POST"])
@_req_admin
def update_custom():
    cfg = _cfg()
    slot = int(request.form.get("slot","1"))
    idx = max(1, min(3, slot)) - 1
    customs = cfg.get("customs", _default_config()["customs"])
    while len(customs) < 3:
        customs.append({"_idx": len(customs)+1})
    c = customs[idx]
    c["_idx"] = idx+1
    c["enabled"] = request.form.get("enabled") == "on"
    c["mode"] = request.form.get("mode","SIMPLE")
    c["lookback"] = int(request.form.get("lookback","3") or 3)
    c["tol_pct"] = float(request.form.get("tol_pct","0.1") or 0.1)
    c["simple_buy"] = request.form.get("simple_buy","")
    c["simple_sell"] = request.form.get("simple_sell","")
    # Optional JSON rules
    br = request.form.get("buy_rule_json","").strip()
    sr = request.form.get("sell_rule_json","").strip()
    c["buy_rule"] = json.loads(br) if br else None
    c["sell_rule"] = json.loads(sr) if sr else None
    customs[idx] = c
    cfg["customs"] = customs
    set_config(cfg)
    flash(f"Saved CUSTOM {idx+1}", "ok")
    return redirect(url_for("dashboard.view"))

# ------------------------------- Users ---------------------------------------
def _load_users():
    # Stored in app_config? Keep it simple — small in-memory list inside config.
    cfg = _cfg()
    return cfg.get("users", [])

@bp.route("/users/add", methods=["POST"])
@_req_admin
def users_add():
    cfg = _cfg()
    users = cfg.get("users", [])
    tid = (request.form.get("telegram_id") or "").strip()
    tier = request.form.get("tier","free")
    expires_at = request.form.get("expires_at") or None
    if not tid:
        flash("Telegram ID required","error")
        return redirect(url_for("dashboard.view"))
    # upsert
    found = False
    for u in users:
        if u.get("telegram_id") == tid:
            u["tier"] = tier
            u["expires_at"] = expires_at
            found = True
            break
    if not found:
        users.append({"telegram_id": tid, "tier": tier, "expires_at": expires_at})
    cfg["users"] = users
    set_config(cfg)
    flash("Saved user","ok")
    return redirect(url_for("dashboard.view"))

@bp.route("/users/delete", methods=["POST"])
@_req_admin
def users_delete():
    cfg = _cfg()
    users = cfg.get("users", [])
    tid = (request.form.get("telegram_id") or "").strip()
    users = [u for u in users if u.get("telegram_id") != tid]
    cfg["users"] = users
    set_config(cfg)
    flash("Deleted user","ok")
    return redirect(url_for("dashboard.view"))

# --------------------------- Backtest & plotting -----------------------------
_LAST_BT_JSON: Dict[str, Any] = {}
_LAST_BT_CSV = ""

@bp.route("/backtest", methods=["POST"])
@_req_admin
def backtest():
    cfg = _cfg()
    try:
        # symbols
        syms_text = request.form.get("bt_symbols","")
        parts = [p for p in syms_text.replace(",", " ").split() if p.strip()]
        if request.form.get("convert_po_bt"):
            parts = convert_po_to_deriv(parts)
        if not parts:
            raise RuntimeError("No symbols provided")

        tf = request.form.get("bt_tf", cfg.get("bt",{}).get("tf","M1"))
        expiry = request.form.get("bt_expiry", cfg.get("bt",{}).get("expiry","5m"))
        strategy = request.form.get("bt_strategy", cfg.get("bt",{}).get("strategy","BASE")).upper()
        use_server = request.form.get("use_server") == "on"
        count = int(request.form.get("bt_count","600") or 600)

        # data source
        if request.files.get("bt_csv") and not use_server:
            df = load_csv(request.files["bt_csv"])
        else:
            # only Deriv (no Yahoo)
            sym = parts[0]
            gmap = {"M1":60,"M2":120,"M3":180,"M5":300,"M10":600,"M15":900,"M30":1800,"H1":3600,"H4":14400,"D1":86400}
            df = fetch_deriv_history(sym, gmap.get(tf,60), count=count)

        # compute + signals + plot
        ind_cfg = cfg.get("indicators", {})
        # ensure RSI / STOCH panels show if enabled
        signals, stats = backtest_run(df, strategy, ind_cfg, expiry)
        plot_name = plot_signals(df, signals, ind_cfg, strategy, tf, expiry)

        # save JSON + CSV for quick links
        out = {"tf":tf, "expiry":expiry, "strategy":strategy,
               "summary": f"wins={stats['wins']} loss={stats['loss']} draw={stats['draw']} win_rate={stats['win_rate']:.1f}%",
               "signals": signals, "plot_name": plot_name}
        global _LAST_BT_JSON, _LAST_BT_CSV
        _LAST_BT_JSON = out
        # dump a tiny CSV of signals
        buf = io.StringIO()
        writer = csv.writer(buf)
        writer.writerow(["time","direction","expiry"])
        for s in signals:
            writer.writerow([s["index"], s["direction"], s["expiry_idx"]])
        _LAST_BT_CSV = buf.getvalue()

        flash("Backtest completed", "ok")
        return redirect(url_for("dashboard.view"))
    except Exception as e:
        flash(f"Backtest error: {e}", "error")
        return redirect(url_for("dashboard.view"))

@bp.route("/backtest/last.json")
@_req_admin
def backtest_last_json():
    return jsonify(_LAST_BT_JSON or {})

@bp.route("/backtest/last.csv")
@_req_admin
def backtest_last_csv():
    data = (_LAST_BT_CSV or "time,direction,expiry\n").encode("utf-8")
    return send_file(io.BytesIO(data), mimetype="text/csv", as_attachment=True, download_name="last_signals.csv")

@bp.route("/plot/<name>")
def plot_file(name: str):
    path = os.path.join("static","plots", name)
    if not os.path.exists(path):
        return "not found", 404
    return send_file(path, mimetype="image/png")

# --------------------------- Deriv helper ------------------------------------
@bp.route("/deriv/fetch", methods=["POST"])
@_req_admin
def deriv_fetch():
    # Stub helper to keep template happy; data is pulled via /backtest
    flash("Deriv fetch helper is a stub in this build. Use Backtest.", "ok")
    return redirect(url_for("dashboard.view"))

# --------------------------- Live Engine -------------------------------------
@bp.route("/live/start")
@_req_admin
def live_start():
    ok, msg = ENGINE.start()
    return jsonify({"ok": ok, "msg": msg})

@bp.route("/live/stop")
@_req_admin
def live_stop():
    ok, msg = ENGINE.stop()
    return jsonify({"ok": ok, "msg": msg})

@bp.route("/live/debug/on")
@_req_admin
def live_debug_on():
    ENGINE.debug = True
    return jsonify({"ok": True})

@bp.route("/live/debug/off")
@_req_admin
def live_debug_off():
    ENGINE.debug = False
    return jsonify({"ok": True})

@bp.route("/live/status")
def live_status():
    return jsonify({"status": ENGINE.status()})

@bp.route("/live/tally")
def live_tally():
    return jsonify({"tally": ENGINE.tally()})

# --------------------------- API for Telegram --------------------------------
@bp.route("/api/status")
@_req_admin
def api_status():
    s = ENGINE.status()
    # flatten for right panel
    tallies = {
        "free": s.get("tally",{}).get("by_tier",{}).get("free", 0),
        "basic": s.get("tally",{}).get("by_tier",{}).get("basic", 0),
        "pro": s.get("tally",{}).get("by_tier",{}).get("pro", 0),
        "vip": s.get("tally",{}).get("by_tier",{}).get("vip", 0),
        "total": s.get("tally",{}).get("total", 0),
    }
    return jsonify({
        "running": s.get("running"),
        "debug": s.get("debug"),
        "caps": DAILY_CAPS,
        "configured_chats": {k: bool(v) for k,v in TIER_TO_CHAT.items()},
        "tallies": tallies,
        "day": s.get("tally",{}).get("date"),
        "last_send_result": s.get("last_send_result", {}),
    })

@bp.route("/api/check_bot")
@_req_admin
def api_check_bot():
    if not BOT_TOKEN:
        return jsonify({"ok": False, "error": "Missing TELEGRAM_BOT_TOKEN"})
    # Reuse tg_test minimally to confirm token by attempting sends? Safer: call getMe via tg_test results
    ok, info = tg_test()
    return jsonify({"ok": ok, "getMe": {"raw": info}})

@bp.route("/api/test/vip", methods=["POST"])
@_req_admin
def api_test_vip():
    text = (request.json or {}).get("text") or "VIP test"
    ok, info = ENGINE.send_signal("vip", text)
    return jsonify({"ok": ok, "info": info, "status": ENGINE.status()})

@bp.route("/api/send", methods=["POST"])
@_req_admin
def api_send():
    body = request.json or {}
    tier = (body.get("tier") or "vip").lower()
    text = (body.get("text") or "").strip()
    if not text:
        return jsonify({"result":{"ok": False, "error":"empty text"}, "status": ENGINE.status()})
    # Block outside trading window (Mon–Fri 08:00–17:00)
    cfg = _cfg()
    if not trading_open_now(cfg):
        return jsonify({"result":{"ok": False, "error":"Trading window closed (Mon–Fri 08:00–17:00 only)"},
                        "status": ENGINE.status()})
    ok, info = ENGINE.send_signal(tier, text)
    res = {"ok": ok, "info": info}
    return jsonify({"result": res, "status": ENGINE.status()})

@bp.route("/telegram/diag")
@_req_admin
def telegram_diag():
    return jsonify({
        "BOT_TOKEN_present": bool(BOT_TOKEN),
        "tier_to_chat": TIER_TO_CHAT,
        "caps": DAILY_CAPS,
        "engine_status": ENGINE.status(),
    })

# --------------------------- Health ------------------------------------------
@bp.route("/_up")
def up_check():
    return jsonify({"ok": True, "ts": datetime.utcnow().isoformat()+"Z"})
