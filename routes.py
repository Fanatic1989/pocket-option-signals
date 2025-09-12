# routes.py — Dashboard, backtest, users, Telegram test/diag, live engine endpoints
import os, io, json, re, csv
from datetime import datetime
import pandas as pd
from flask import (
    Blueprint, render_template, request, redirect, url_for, session, flash, jsonify,
    send_from_directory, make_response
)
from utils import (
    exec_sql, get_config, set_config, within_window, TZ,
    load_csv, convert_po_to_deriv, fetch_deriv_history,
    backtest_run, plot_signals
)
from live_engine import ENGINE, tg_test_all

bp = Blueprint("dashboard", __name__, template_folder="templates", static_folder="static")

def _cfg() -> dict:
    c = get_config() or {}
    for k in ("window","strategies","indicators","symbols","symbols_raw","live_tf","live_expiry","bt_tf","bt_expiry"):
        c.setdefault(k, {} if k in ("window","strategies","indicators") else None)
    return c

def _to_list(text: str) -> list:
    if not text: return []
    parts = re.split(r"[\s,;]+", text.strip())
    return [p for p in parts if p]

def require_login(fn):
    from functools import wraps
    @wraps(fn)
    def wrap(*a, **kw):
        if not session.get("admin"): return redirect(url_for("dashboard.login", next=request.path))
        return fn(*a, **kw)
    return wrap

@bp.route("/")
def index():
    cfg = _cfg()
    return render_template("dashboard.html",
        view="index",
        within=within_window(cfg),
        now=datetime.now(TZ).strftime("%Y-%m-%d %H:%M:%S"),
        tz=getattr(TZ, "zone", "UTC"),
        window=cfg.get("window", {})
    )

@bp.route("/login", methods=["GET","POST"])
def login():
    if request.method == "POST":
        if request.form.get("password") == os.getenv("ADMIN_PASSWORD","admin123"):
            session["admin"] = True
            return redirect(request.args.get("next") or url_for("dashboard.dashboard"))
        flash("Invalid password", "error")
    return render_template("dashboard.html", view="login", tz=getattr(TZ,"zone","UTC"))

@bp.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("dashboard.index"))

@bp.route("/_up")
def up_check(): return "OK", 200

PO_MAJOR = ["EURUSD","GBPUSD","USDJPY","USDCHF","USDCAD","AUDUSD","NZDUSD",
            "EURGBP","EURJPY","GBPJPY","EURAUD","AUDJPY","CADJPY","CHFJPY"]
DERIV_FRX = ["frxEURUSD","frxGBPUSD","frxUSDJPY","frxUSDCHF","frxUSDCAD","frxAUDUSD","frxNZDUSD",
             "frxEURGBP","frxEURJPY","frxGBPJPY","frxEURAUD","frxAUDJPY","frxCADJPY","frxCHFJPY"]
AVAILABLE_GROUPS = [
    {"label":"Deriv (frx*)","items":DERIV_FRX},
    {"label":"Pocket Option majors","items":PO_MAJOR},
]

@bp.route("/dashboard")
@require_login
def dashboard():
    exec_sql("""CREATE TABLE IF NOT EXISTS users(telegram_id TEXT PRIMARY KEY, tier TEXT, expires_at TEXT)""")
    users = [{"telegram_id":r[0], "tier":r[1], "expires_at":r[2] or None}
             for r in (exec_sql("SELECT telegram_id,tier,COALESCE(expires_at,'') FROM users", fetch=True) or [])]
    cfg = _cfg()
    customs = [dict(cfg.get("custom1",{}), _idx=1),
               dict(cfg.get("custom2",{}), _idx=2),
               dict(cfg.get("custom3",{}), _idx=3)]
    strategies_core = cfg.get("strategies") or {"BASE":{"enabled":True},"TREND":{"enabled":False},"CHOP":{"enabled":False}}
    strategies_all = dict(strategies_core)
    for i, c in enumerate(customs, 1): strategies_all[f"CUSTOM{i}"] = {"enabled": bool(c.get("enabled"))}
    bt = session.get("bt", {})

    # --- INDICATOR SPECS (UI)
    specs = {
        # MAs
        "SMA":{"name":"SMA","kind":"Moving Average","params":{"period":20}},
        "EMA":{"name":"EMA","kind":"Moving Average","params":{"period":20}},
        "WMA":{"name":"WMA","kind":"Weighted MA","params":{"period":20}},
        "SMMA":{"name":"SMMA","kind":"Smoothed MA","params":{"period":20}},
        "TMA":{"name":"TMA","kind":"Triangular MA","params":{"period":20}},
        # Oscillators / trend
        "RSI":{"name":"RSI","kind":"Oscillator","params":{"period":14}},
        "STOCH":{"name":"Stochastic Oscillator","kind":"Oscillator","params":{"k":14,"d":3}},
        "ATR":{"name":"Average True Range","kind":"Volatility","params":{"period":14}},
        "ADX":{"name":"ADX (+DI/-DI)","kind":"Trend","params":{"period":14}},
        "BBANDS":{"name":"Bollinger Bands","kind":"Volatility","params":{"period":20,"mult":2.0}},
        "BBWIDTH":{"name":"Bollinger Bands Width","kind":"Volatility","params":{"period":20,"mult":2.0}},  # width derived from BBANDS but toggle is ok
        "KELTNER":{"name":"Keltner Channel","kind":"Volatility","params":{"period":20,"mult":2.0}},
        "DONCHIAN":{"name":"Donchian Channels","kind":"Volatility","params":{"period":20}},
        "ENVELOPES":{"name":"Envelopes","kind":"Volatility","params":{"period":20,"pct":0.02}},
        "MACD":{"name":"MACD","kind":"Momentum","params":{"fast":12,"slow":26,"signal":9}},
        "OSMA":{"name":"OsMA (MACD Hist)","kind":"Momentum","params":{"fast":12,"slow":26,"signal":9}},
        "MOMENTUM":{"name":"Momentum","kind":"Momentum","params":{"period":10}},
        "ROC":{"name":"Rate of Change","kind":"Momentum","params":{"period":10}},
        "WILLR":{"name":"Williams %R","kind":"Momentum","params":{"period":14}},
        "CCI":{"name":"CCI","kind":"Momentum","params":{"period":20}},
        "AROON":{"name":"Aroon","kind":"Trend","params":{"period":14}},
        "VORTEX":{"name":"Vortex","kind":"Trend","params":{"period":14}},
        "AO":{"name":"Awesome Oscillator","kind":"Momentum","params":{}},
        "AC":{"name":"Accelerator Oscillator","kind":"Momentum","params":{}},
        "ICHIMOKU":{"name":"Ichimoku Kinko Hyo","kind":"Trend","params":{"conversion":9,"base":26,"spanb":52}},
        "PSAR":{"name":"Parabolic SAR","kind":"Trend","params":{"af":0.02,"af_max":0.2}},
        "SUPERtrend":{"name":"SuperTrend","kind":"Trend","params":{"period":10,"mult":3.0}},
        "FRACTAL":{"name":"Fractal","kind":"Bill Williams","params":{}},
        "FRACTAL_BANDS":{"name":"Fractal Chaos Bands","kind":"Volatility","params":{"period":20}},
        "BEARS":{"name":"Bears Power","kind":"Bill Williams","params":{"period":13}},
        "BULLS":{"name":"Bulls Power","kind":"Bill Williams","params":{"period":13}},
        "SCHAFF":{"name":"Schaff Trend Cycle","kind":"Momentum","params":{"fast":23,"slow":50,"cycle":10}},
        "ZIGZAG":{"name":"ZigZag","kind":"Price pattern","params":{"pct":5.0}},
    }

    return render_template("dashboard.html",
        view="dashboard",
        window=cfg.get("window", {}),
        strategies_all=strategies_all,
        strategies=strategies_core,
        indicators=cfg.get("indicators") or {},
        specs=specs,
        customs=customs,
        active_symbols=cfg.get("symbols") or [],
        symbols_raw=cfg.get("symbols_raw") or [],
        available_groups=AVAILABLE_GROUPS,
        users=users,
        bt=bt,
        live_tf=cfg.get("live_tf","M5"),
        live_expiry=cfg.get("live_expiry","5m"),
        now=datetime.now(TZ).strftime("%Y-%m-%d %H:%M:%S"),
        tz=getattr(TZ,"zone","UTC"),
        within=within_window(cfg)
    )

# --- the remainder of routes.py is exactly the same as the version I gave you previously ---
@bp.route("/update_window", methods=["POST"])
@require_login
def update_window():
    cfg = _cfg()
    cfg.setdefault("window", {"start":"08:00","end":"17:00","timezone":getattr(TZ,"zone","UTC")})
    for k in ("start","end","timezone"):
        if request.form.get(k): cfg["window"][k] = request.form[k]
    if request.form.get("live_tf"): cfg["live_tf"] = request.form["live_tf"].upper()
    if request.form.get("live_expiry"): cfg["live_expiry"] = request.form["live_expiry"]
    set_config(cfg); flash("Trading window & live defaults updated.")
    return redirect(url_for("dashboard.dashboard"))

@bp.route("/update_symbols", methods=["POST"])
@require_login
def update_symbols():
    cfg = _cfg()
    def _to_list(text: str) -> list:
        if not text: return []
        parts = re.split(r"[\s,;]+", text.strip())
        return [p for p in parts if p]
    sym_raw = _to_list(request.form.get("symbols_text",""))
    multi_deriv = request.form.getlist("symbols_deriv_multi")
    multi_po    = request.form.getlist("symbols_po_multi")
    sym_raw = list(dict.fromkeys(sym_raw + multi_deriv + multi_po))
    if request.form.get("convert_po"):
        cfg["symbols"] = convert_po_to_deriv(sym_raw)
    else:
        cfg["symbols"] = sym_raw
    cfg["symbols_raw"] = sym_raw
    set_config(cfg); flash("Symbols updated.")
    return redirect(url_for("dashboard.dashboard"))

@bp.route("/update_indicators", methods=["POST"])
@require_login
def update_indicators():
    cfg = _cfg(); cfg.setdefault("indicators", {})
    incoming = {}
    for key in request.form.keys():
        # keys look like ind_RSI_enabled, ind_MACD_fast, etc. We rebuild dict by known spec prefixes.
        if not key.startswith("ind_"): continue
    # Build from specs to keep defaults
    # pull specs from dashboard() again:
    specs = {
        "SMA":{}, "EMA":{}, "WMA":{}, "SMMA":{}, "TMA":{},
        "RSI":{}, "STOCH":{}, "ATR":{}, "ADX":{}, "BBANDS":{}, "BBWIDTH":{},
        "KELTNER":{}, "DONCHIAN":{}, "ENVELOPES":{}, "MACD":{}, "OSMA":{},
        "MOMENTUM":{}, "ROC":{}, "WILLR":{}, "CCI":{}, "AROON":{}, "VORTEX":{},
        "AO":{}, "AC":{}, "ICHIMOKU":{}, "PSAR":{}, "SUPERtrend":{}, "FRACTAL":{},
        "FRACTAL_BANDS":{}, "BEARS":{}, "BULLS":{}, "SCHAFF":{}, "ZIGZAG":{}
    }
    for k in specs.keys():
        box = bool(request.form.get(f"ind_{k}_enabled"))
        params = {}
        for p in ("period","k","d","mult","pct","fast","slow","signal","conversion","base","spanb","af","af_max","cycle"):
            v = request.form.get(f"ind_{k}_{p}")
            if v is not None and v != "":
                try: params[p] = float(v) if ('.' in str(v)) else int(v)
                except: params[p] = v
        incoming[k] = {"enabled": box, **params}
    cfg["indicators"] = incoming
    set_config(cfg); flash("Indicators saved.")
    return redirect(url_for("dashboard.dashboard"))

@bp.route("/update_strategies", methods=["POST"])
@require_login
def update_strategies():
    cfg = _cfg()
    base = {"BASE":{"enabled":False},"TREND":{"enabled":False},"CHOP":{"enabled":False}}
    for k in list(base.keys()):
        base[k]["enabled"] = bool(request.form.get(f"s_{k}"))
    for i in (1,2,3):
        base[f"CUSTOM{i}"] = {"enabled": bool(request.form.get(f"s_CUSTOM{i}"))}
        cfg.setdefault(f"custom{i}", {})["enabled"] = base[f"CUSTOM{i}"]["enabled"]
    cfg["strategies"] = base
    if request.form.get("bt_tf"): cfg["bt_tf"] = request.form["bt_tf"].upper()
    if request.form.get("bt_expiry"): cfg["bt_expiry"] = request.form["bt_expiry"]
    if request.form.get("live_tf"): cfg["live_tf"] = request.form["live_tf"].upper()
    if request.form.get("live_expiry"): cfg["live_expiry"] = request.form["live_expiry"]
    set_config(cfg); flash("Strategies & defaults saved.")
    return redirect(url_for("dashboard.dashboard"))

@bp.route("/update_custom", methods=["POST"])
@require_login
def update_custom():
    cfg = _cfg()
    slot = int(request.form.get("slot","1"))
    key = f"custom{slot}"
    cfg.setdefault(key, {})
    c = cfg[key]
    c["enabled"] = bool(request.form.get("enabled"))
    c["mode"] = request.form.get("mode","SIMPLE")
    for k in ("lookback","tol_pct"):
        v = request.form.get(k)
        if v is not None and v != "":
            try: c[k] = float(v)
            except: c[k] = v
    c["simple_buy"]  = request.form.get("simple_buy","")
    c["simple_sell"] = request.form.get("simple_sell","")
    import json as _json
    br = request.form.get("buy_rule_json") or ""
    sr = request.form.get("sell_rule_json") or ""
    try: c["buy_rule"]  = _json.loads(br) if br.strip() else None
    except: c["buy_rule"] = None
    try: c["sell_rule"] = _json.loads(sr) if sr.strip() else None
    except: c["sell_rule"] = None
    set_config(cfg); flash(f"CUSTOM {slot} saved.")
    return redirect(url_for("dashboard.dashboard"))

@bp.route("/users/add", methods=["POST"])
@require_login
def users_add():
    telegram_id = (request.form.get("telegram_id") or "").strip()
    tier = (request.form.get("tier") or "free").lower()
    exp  = request.form.get("expires_at") or None
    if not telegram_id:
        flash("Missing Telegram ID","error"); return redirect(url_for("dashboard.dashboard"))
    exec_sql("""CREATE TABLE IF NOT EXISTS users(telegram_id TEXT PRIMARY KEY, tier TEXT, expires_at TEXT)""")
    exec_sql("INSERT INTO users(telegram_id,tier,expires_at) VALUES(?,?,?) "
             "ON CONFLICT(telegram_id) DO UPDATE SET tier=excluded.tier, expires_at=excluded.expires_at",
             (telegram_id, tier, exp))
    flash("User saved.")
    return redirect(url_for("dashboard.dashboard"))

@bp.route("/users/delete", methods=["POST"])
@require_login
def users_delete():
    telegram_id = request.form.get("telegram_id")
    if telegram_id:
        exec_sql("DELETE FROM users WHERE telegram_id=?", (telegram_id,))
        flash("User deleted.")
    return redirect(url_for("dashboard.dashboard"))

@bp.route("/backtest", methods=["POST"])
@require_login
def backtest():
    cfg = _cfg()
    raw = _to_list(request.form.get("bt_symbols",""))
    if request.form.get("convert_po_bt"):
        symbols = convert_po_to_deriv(raw)
    else:
        symbols = raw
    tf = (request.form.get("bt_tf") or cfg.get("bt_tf") or "M5").upper()
    expiry = request.form.get("bt_expiry") or cfg.get("bt_expiry") or cfg.get("live_expiry") or "5m"
    strat = (request.form.get("bt_strategy") or "BASE").upper()
    up = request.files.get("bt_csv")
    if up and up.filename:
        df = load_csv(up)
    else:
        gmap = {"M1":60,"M2":120,"M3":180,"M5":300,"M10":600,"M15":900,"M30":1800,"H1":3600,"H4":14400,"D1":86400}
        sym = (symbols[0] if symbols else "frxEURUSD")
        df = fetch_deriv_history(sym, gmap.get(tf, 300), days=5)

    signals, stats = backtest_run(df, strat, cfg.get("indicators") or {}, expiry)
    plot_name = plot_signals(df, signals, cfg.get("indicators") or {}, strat, tf, expiry)
    bt = {
        "tf": tf, "expiry": expiry, "strategy": strat,
        "summary": f"{stats['wins']}W {stats['loss']}L {stats['draw']}D • WR={stats['win_rate']:.1f}%",
        "plot_name": plot_name
    }
    session["bt"] = bt
    flash("Backtest complete.")
    return redirect(url_for("dashboard.dashboard"))

@bp.route("/backtest/last.json")
def backtest_last_json(): return jsonify(session.get("bt", {}))

@bp.route("/backtest/last.csv")
def backtest_last_csv():
    bt = session.get("bt", {})
    s = io.StringIO(); w = csv.writer(s); w.writerow(["key","value"])
    for k,v in bt.items(): w.writerow([k,v])
    resp = make_response(s.getvalue()); resp.headers["Content-Type"]="text/csv"; return resp

@bp.route("/plots/<name>")
def plot_file(name):
    resp = send_from_directory("static/plots", name)
    resp.headers["Cache-Control"] = "no-store"; return resp

@bp.route("/telegram/test", methods=["POST"])
@require_login
def telegram_test():
    ok, info = tg_test_all()
    flash("Test sent to configured tiers." if ok else f"Telegram test error: {info}", "error" if not ok else "message")
    return redirect(url_for("dashboard.dashboard"))

@bp.route("/telegram/diag")
def telegram_diag():
    from live_engine import BOT_TOKEN
    diag = {"configured_chats": {
                "free":bool(os.getenv("TELEGRAM_CHAT_FREE") or os.getenv("TELEGRAM_CHAT_ID_FREE")),
                "basic":bool(os.getenv("TELEGRAM_CHAT_BASIC") or os.getenv("TELEGRAM_CHAT_ID_BASIC")),
                "pro":bool(os.getenv("TELEGRAM_CHAT_PRO") or os.getenv("TELEGRAM_CHAT_ID_PRO")),
                "vip":bool(os.getenv("TELEGRAM_CHAT_VIP") or os.getenv("TELEGRAM_CHAT_ID_VIP")),
            },
            "getMe": {}, "ok": False, "send_result":"n/a",
            "token_masked": (BOT_TOKEN[:9]+"..."+BOT_TOKEN[-6:]) if BOT_TOKEN else ""}
    try:
        import requests
        if BOT_TOKEN:
            r = requests.get(f"https://api.telegram.org/bot{BOT_TOKEN}/getMe", timeout=10)
            diag["getMe"] = r.json(); diag["ok"] = bool(diag["getMe"].get("ok"))
    except Exception as e:
        diag["getMe"] = {"ok": False, "error": str(e)}
    return jsonify(diag)

@bp.route("/live/start")
def live_start(): ok,msg = ENGINE.start(); return jsonify({"ok":ok,"msg":msg,"status":ENGINE.status()})
@bp.route("/live/stop")
def live_stop(): ok,msg = ENGINE.stop(); return jsonify({"ok":ok,"msg":msg,"status":ENGINE.status()})
@bp.route("/live/debug/on")
def live_dbg_on(): ENGINE.set_debug(True); return jsonify({"ok":True,"status":ENGINE.status()})
@bp.route("/live/debug/off")
def live_dbg_off(): ENGINE.set_debug(False); return jsonify({"ok":True,"status":ENGINE.status()})
@bp.route("/live/status")
def live_status(): return jsonify({"status":ENGINE.status()})
@bp.route("/live/tally")
def live_tally(): s=ENGINE.status(); return jsonify({"tally": s.get("tally",{}), "caps": s.get("caps",{})})
