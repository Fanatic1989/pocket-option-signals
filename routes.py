# routes.py — Dashboard, config, backtest, Telegram + Live APIs
from __future__ import annotations
import os, io, json, csv
from datetime import datetime, timezone
from typing import Dict, Any, List, Tuple

from flask import (
    Blueprint, render_template, request, redirect, url_for,
    session, jsonify, flash, send_file
)

# utils you already have
from utils import (
    TZ, get_config, set_config, within_window, convert_po_to_deriv,
    load_csv, fetch_deriv_history, compute_indicators,
    backtest_run, plot_signals, PO_PAIRS, DERIV_PAIRS, EXPIRY_TO_BARS
)

# -------- tolerate builds where tg_test might be missing ----------
try:
    from live_engine import ENGINE, tg_test, BOT_TOKEN, TIER_TO_CHAT, DAILY_CAPS
except Exception:
    from live_engine import ENGINE, BOT_TOKEN, TIER_TO_CHAT, DAILY_CAPS
    def tg_test():
        return False, {"ok": False, "error": "tg_test not available in this build"}
# -----------------------------------------------------------------

bp = Blueprint("dashboard", __name__)

ADMIN_PASS = os.getenv("ADMIN_PASSWORD", "admin")

TF_TO_GRAN = {
    "M1":60, "M2":120, "M3":180, "M5":300, "M10":600, "M15":900, "M30":1800,
    "H1":3600, "H4":14400, "D1":86400
}

# ---------------- Helpers ----------------
def _ctx_base(view: str="index") -> Dict[str, Any]:
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
    cfg.setdefault("window", {"start":"08:00","end":"17:00","timezone":str(TZ)})
    cfg.setdefault("live_tf", "M1")
    cfg.setdefault("live_expiry", "5m")
    cfg.setdefault("symbols_raw", ["frxEURUSD","frxGBPUSD"])
    cfg.setdefault("indicators", {
        "SMA":{"enabled": True, "period":50},
        "RSI":{"enabled": True, "period":14},
        "STOCH":{"enabled": True, "k":14, "d":3},
        # add more here as needed, your utils supports many
    })
    cfg.setdefault("strategies", {
        "BASE":{"enabled": True},
        "TREND":{"enabled": False},
        "CHOP":{"enabled": False},
        "CUSTOM1":{"enabled": False},
        "CUSTOM2":{"enabled": False},
        "CUSTOM3":{"enabled": False},
    })
    cfg.setdefault("customs", [
        {"_idx":1, "enabled": False, "mode":"SIMPLE", "lookback":3, "tol_pct":0.1,
         "simple_buy":"", "simple_sell":"", "buy_rule":None, "sell_rule":None},
        {"_idx":2, "enabled": False, "mode":"SIMPLE", "lookback":3, "tol_pct":0.1,
         "simple_buy":"", "simple_sell":"", "buy_rule":None, "sell_rule":None},
        {"_idx":3, "enabled": False, "mode":"SIMPLE", "lookback":3, "tol_pct":0.1,
         "simple_buy":"", "simple_sell":"", "buy_rule":None, "sell_rule":None},
    ])
    return cfg

def _admin_required(fn):
    def wrap(*a, **kw):
        if not session.get("admin"):
            return redirect(url_for("dashboard.login"))
        return fn(*a, **kw)
    wrap.__name__ = fn.__name__
    return wrap

# ---------------- Basic pages ----------------
@bp.route("/")
def index():
    ctx = _ctx_base("index")
    return render_template("dashboard.html", **ctx)

@bp.route("/login", methods=["GET","POST"])
def login():
    if request.method == "POST":
        pw = request.form.get("password","")
        if pw == ADMIN_PASS:
            session["admin"] = True
            return redirect(url_for("dashboard.view"))
        flash("Wrong password", "error")
    ctx = _ctx_base("login")
    return render_template("dashboard.html", **ctx)

@bp.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("dashboard.index"))

@bp.route("/dashboard")
@_admin_required
def view():
    cfg = _ensure_cfg_defaults(get_config())
    ctx = _ctx_base("dashboard")
    ctx.update({
        "window": cfg.get("window"),
        "live_tf": cfg.get("live_tf","M1"),
        "live_expiry": cfg.get("live_expiry","5m"),
        "symbols_raw": cfg.get("symbols_raw", []),
        "active_symbols": convert_po_to_deriv(cfg.get("symbols_raw", [])),
        "available_groups": [
            {"label":"Deriv", "items": DERIV_PAIRS},
            {"label":"PocketOption majors", "items": ["EURUSD","GBPUSD","USDJPY","USDCHF","USDCAD","AUDUSD","NZDUSD","EURGBP","EURJPY","GBPJPY","EURAUD","AUDJPY","CADJPY","CHFJPY"]},
        ],
        "indicators": cfg.get("indicators", {}),
        "strategies": cfg.get("strategies", {}),
        "strategies_all": {
            "BASE":{}, "TREND":{}, "CHOP":{}, "CUSTOM1":{}, "CUSTOM2":{}, "CUSTOM3":{}
        },
        "customs": cfg.get("customs", []),
        "bt": session.get("bt_state"),
    })
    return render_template("dashboard.html", **ctx)

# ---------------- Update sections ----------------
@bp.route("/window/update", methods=["POST"])
@_admin_required
def update_window():
    cfg = _ensure_cfg_defaults(get_config())
    w = cfg.get("window", {})
    w["start"] = request.form.get("start","08:00")
    w["end"]   = request.form.get("end","17:00")
    w["timezone"] = request.form.get("timezone", str(TZ))
    cfg["window"] = w
    cfg["live_tf"] = request.form.get("live_tf", cfg.get("live_tf","M1"))
    cfg["live_expiry"] = request.form.get("live_expiry", cfg.get("live_expiry","5m"))
    set_config(cfg)
    flash("Window/defaults updated", "ok")
    return redirect(url_for("dashboard.view"))

@bp.route("/symbols/update", methods=["POST"])
@_admin_required
def update_symbols():
    cfg = _ensure_cfg_defaults(get_config())
    # multi-selects + free text
    d_multi = request.form.getlist("symbols_deriv_multi") or []
    p_multi = request.form.getlist("symbols_po_multi") or []
    text = request.form.get("symbols_text","").replace(",", " ")
    typed = [x for x in text.split() if x]
    raw = list(dict.fromkeys(typed + d_multi + p_multi))  # unique
    if request.form.get("convert_po"):
        raw = convert_po_to_deriv(raw)
    cfg["symbols_raw"] = raw
    set_config(cfg)
    flash("Symbols updated", "ok")
    return redirect(url_for("dashboard.view"))

@bp.route("/indicators/update", methods=["POST"])
@_admin_required
def update_indicators():
    cfg = _ensure_cfg_defaults(get_config())
    ind_cfg = cfg.get("indicators", {})

    def set_toggle(name: str, params: List[str]):
        block = ind_cfg.get(name, {})
        block["enabled"] = bool(request.form.get(f"ind_{name}_enabled"))
        # carry parameters
        for p in params:
            v = request.form.get(f"ind_{name}_{p}")
            if v is not None and v != "":
                block[p] = v
        ind_cfg[name] = block

    # Key indicators (you can extend freely; utils.py supports many)
    set_toggle("SMA", ["period"])
    set_toggle("EMA", ["period"])
    set_toggle("RSI", ["period"])
    set_toggle("STOCH", ["k","d"])
    set_toggle("BOLL", ["period","mult"])
    set_toggle("PSAR", ["step","max"])
    set_toggle("SUPERTREND", ["period","mult"])
    set_toggle("ATR", ["period"])
    set_toggle("ADX", ["period"])
    set_toggle("MACD", ["fast","slow","signal"])

    cfg["indicators"] = ind_cfg
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
    # defaults used by backtest/live panels
    cfg["live_tf"] = request.form.get("live_tf", cfg.get("live_tf","M1"))
    cfg["live_expiry"] = request.form.get("live_expiry", cfg.get("live_expiry","5m"))
    cfg["bt"] = {"tf": request.form.get("bt_tf", cfg["live_tf"]),
                 "expiry": request.form.get("bt_expiry", cfg["live_expiry"])}
    cfg["strategies"] = st
    set_config(cfg)
    flash("Strategies toggles updated", "ok")
    return redirect(url_for("dashboard.view"))

@bp.route("/custom/update", methods=["POST"])
@_admin_required
def update_custom():
    cfg = _ensure_cfg_defaults(get_config())
    customs = cfg.get("customs", [])
    try:
        slot = int(request.form.get("slot","1"))
    except Exception:
        slot = 1
    idx = max(1, min(3, slot)) - 1
    # build
    c = customs[idx] if idx < len(customs) else {"_idx": slot}
    c["_idx"] = slot
    c["enabled"] = bool(request.form.get("enabled"))
    c["mode"] = request.form.get("mode","SIMPLE")
    c["lookback"] = int(request.form.get("lookback", 3))
    try:
        c["tol_pct"] = float(request.form.get("tol_pct","0.1"))
    except Exception:
        c["tol_pct"] = 0.1
    c["simple_buy"] = request.form.get("simple_buy","")
    c["simple_sell"] = request.form.get("simple_sell","")
    # JSON fields (optional)
    br = request.form.get("buy_rule_json","").strip()
    sr = request.form.get("sell_rule_json","").strip()
    try:
        c["buy_rule"] = json.loads(br) if br else None
    except Exception:
        c["buy_rule"] = None
    try:
        c["sell_rule"] = json.loads(sr) if sr else None
    except Exception:
        c["sell_rule"] = None

    if idx < len(customs):
        customs[idx] = c
    else:
        customs.append(c)
    cfg["customs"] = customs
    set_config(cfg)
    flash(f"CUSTOM {slot} saved", "ok")
    return redirect(url_for("dashboard.view"))

# ---------------- Users (simple) ----------------
@bp.route("/users/add", methods=["POST"])
@_admin_required
def users_add():
    # Stubbed for now; if you later store in SQLite, wire here.
    flash("Users: demo handler (persist logic can be added to SQLite).", "ok")
    return redirect(url_for("dashboard.view"))

@bp.route("/users/delete", methods=["POST"])
@_admin_required
def users_delete():
    flash("Users: delete demo handler.", "ok")
    return redirect(url_for("dashboard.view"))

# ---------------- Backtest ----------------
@bp.route("/backtest", methods=["POST"])
@_admin_required
def backtest():
    cfg = _ensure_cfg_defaults(get_config())
    tf = request.form.get("bt_tf", cfg.get("live_tf","M1"))
    expiry = request.form.get("bt_expiry", cfg.get("live_expiry","5m"))
    strategy = request.form.get("bt_strategy","BASE").upper()
    use_server = bool(request.form.get("use_server"))
    count = int(request.form.get("bt_count","300"))
    symbols_text = (request.form.get("bt_symbols","") or "").replace(",", " ")
    symbols = [s for s in symbols_text.split() if s] or cfg.get("symbols_raw", [])
    if request.form.get("convert_po_bt"):
        symbols = convert_po_to_deriv(symbols)

    g = TF_TO_GRAN.get(tf, 300)
    last_plot = None
    last_rows = []
    warnings = []
    try:
        # If CSV uploaded, that overrides server fetch
        if "bt_csv" in request.files and request.files["bt_csv"].filename:
            f = request.files["bt_csv"]
            df = load_csv(io.BytesIO(f.read()))
        elif use_server:
            # Fetch first symbol only for quick validation plot
            if not symbols:
                raise RuntimeError("No symbol provided")
            sym = symbols[0]
            df = fetch_deriv_history(sym, g, count=count)
        else:
            raise RuntimeError("Upload a CSV or check 'Use Deriv server fetch'.")

        indicators = cfg.get("indicators", {})
        signals, stats = backtest_run(df, strategy, indicators, expiry)
        plot_name = plot_signals(df, signals, indicators, strategy, tf, expiry)
        last_plot = plot_name
        # Build CSV of signals
        for s in signals:
            last_rows.append([s["index"].isoformat(), s["direction"], s["expiry_idx"].isoformat()])

        bt_state = {
            "tf": tf, "expiry": expiry, "strategy": strategy,
            "plot_name": plot_name,
            "summary": f"{stats.get('wins',0)}W / {stats.get('loss',0)}L / {stats.get('draw',0)}D — WR {stats.get('win_rate',0):.1f}%",
            "warnings": warnings
        }
        session["bt_state"] = bt_state
        flash("Backtest completed", "ok")
    except Exception as e:
        session["bt_state"] = {"error": str(e), "tf": tf, "expiry": expiry, "strategy": strategy}
        flash(f"Backtest error: {type(e).__name__}: {e}", "error")
    return redirect(url_for("dashboard.view"))

@bp.route("/backtest/last.json")
@_admin_required
def backtest_last_json():
    bt = session.get("bt_state") or {}
    return jsonify(bt)

@bp.route("/backtest/last.csv")
@_admin_required
def backtest_last_csv():
    bt = session.get("bt_state") or {}
    rows = bt.get("rows") or []
    output = io.StringIO()
    w = csv.writer(output)
    w.writerow(["time","direction","expiry_time"])
    for r in rows: w.writerow(r)
    output.seek(0)
    return send_file(io.BytesIO(output.getvalue().encode("utf-8")),
                     mimetype="text/csv",
                     as_attachment=True,
                     download_name="last_signals.csv")

@bp.route("/plots/<name>")
def plot_file(name: str):
    # static helper
    return send_file(os.path.join("static","plots", name))

# ---------------- Telegram ----------------
@bp.route("/telegram/test", methods=["POST"])
@_admin_required
def telegram_test():
    ok, diag = tg_test()
    flash("VIP test sent" if ok else f"Telegram error: {diag}", "ok" if ok else "error")
    return redirect(url_for("dashboard.view"))

@bp.route("/telegram/diag")
def telegram_diag():
    ok, diag = tg_test()
    return jsonify(diag)

# ---------------- Live Engine controls ----------------
@bp.route("/live/start")
@_admin_required
def live_start():
    ok, msg = ENGINE.start(); return jsonify({"ok":ok,"msg":msg})

@bp.route("/live/stop")
@_admin_required
def live_stop():
    ok, msg = ENGINE.stop(); return jsonify({"ok":ok,"msg":msg})

@bp.route("/live/debug/on")
@_admin_required
def live_debug_on():
    ENGINE.set_debug(True); return jsonify({"ok":True})

@bp.route("/live/debug/off")
@_admin_required
def live_debug_off():
    ENGINE.set_debug(False); return jsonify({"ok":True})

@bp.route("/live/status")
def live_status():
    return jsonify({"status": ENGINE.status()})

@bp.route("/live/tally")
def live_tally():
    s = ENGINE.status()
    t = s.get("tally", {})
    out = {
        "tally": {
            "free": t.get("by_tier",{}).get("free",0),
            "basic": t.get("by_tier",{}).get("basic",0),
            "pro": t.get("by_tier",{}).get("pro",0),
            "vip": t.get("by_tier",{}).get("vip",0),
            "all": t.get("total",0),
        }
    }
    return jsonify(out)

# REST for dashboard.js (right side)
@bp.route("/api/status")
def api_status():
    return jsonify(ENGINE.status())

@bp.route("/api/check_bot")
def api_check_bot():
    ok, diag = tg_test()
    return jsonify(diag)

@bp.route("/api/test/vip", methods=["POST"])
@_admin_required
def api_test_vip():
    text = (request.json or {}).get("text") or "VIP test"
    res = ENGINE.send_signal("vip", text)
    return jsonify({"result": res, "status": ENGINE.status()})

@bp.route("/api/send", methods=["POST"])
@_admin_required
def api_send():
    data = request.json or {}
    tier = (data.get("tier") or "vip").lower()
    text = data.get("text") or ""
    res = ENGINE.send_signal(tier, text)
    return jsonify({"result": res, "status": ENGINE.status()})

@bp.route("/api/send_all", methods=["POST"])
@_admin_required
def api_send_all():
    data = request.json or {}
    text = data.get("text") or ""
    results = {}
    for t in ("free","basic","pro","vip"):
        results[t] = ENGINE.send_signal(t, text)
    return jsonify({"results": results, "status": ENGINE.status()})

# --------- Optional helper to avoid BuildError from template link ------------
@bp.route("/deriv/fetch", methods=["POST"])
@_admin_required
def deriv_fetch():
    flash("Deriv Fetch helper not wired on this build. Use CSV upload or 'Use Deriv server fetch' in Backtest.", "ok")
    return redirect(url_for("dashboard.view"))

# ---------------- Health ----------------
@bp.route("/_up")
def up_check():
    return jsonify({"ok": True, "ts": datetime.now(timezone.utc).isoformat()+"Z"})
