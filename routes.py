# routes.py — Dashboard blueprint: config, indicators, strategies, users, backtest, Telegram, Live engine
from __future__ import annotations

import os
import io
import csv
import json
from functools import wraps
from typing import Dict, Any, List, Tuple

from flask import (
    Blueprint, render_template, request, redirect, url_for, flash,
    session, send_from_directory, jsonify, Response
)

from datetime import datetime, timezone

# App deps
from utils import (
    TZ, TIMEZONE, get_config, set_config, within_window,
    convert_po_to_deriv, load_csv, fetch_deriv_history,
    backtest_run, plot_signals, exec_sql
)
from live_engine import ENGINE, DAILY_CAPS, TIER_TO_CHAT, tg_test_all

bp = Blueprint("dashboard", __name__)

# ---------------------------- Auth helpers -----------------------------------
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "admin123")

def require_login(fn):
    @wraps(fn)
    def wrap(*a, **kw):
        if not session.get("admin"):
            flash("Please login.", "error")
            return redirect(url_for("dashboard.login"))
        return fn(*a, **kw)
    return wrap

# ---------------------------- Users storage ----------------------------------
# Create a simple table for users (telegram_id, tier, expires_at)
exec_sql("""CREATE TABLE IF NOT EXISTS users(
  telegram_id TEXT PRIMARY KEY,
  tier TEXT,
  expires_at TEXT
)""")

def list_users() -> List[Dict[str, Any]]:
    rows = exec_sql("SELECT telegram_id, tier, expires_at FROM users ORDER BY telegram_id", fetch=True) or []
    return [{"telegram_id": r[0], "tier": r[1], "expires_at": r[2]} for r in rows]

def upsert_user(telegram_id: str, tier: str, expires_at: str | None):
    exec_sql("""INSERT INTO users(telegram_id, tier, expires_at) VALUES(?,?,?)
                ON CONFLICT(telegram_id) DO UPDATE SET tier=excluded.tier, expires_at=excluded.expires_at""",
             (telegram_id, tier, expires_at or None))

def delete_user(telegram_id: str):
    exec_sql("DELETE FROM users WHERE telegram_id=?", (telegram_id,))

# ---------------------------- Indicator specs --------------------------------
# Keys must match compute_indicators() in utils.py
INDICATOR_SPECS: Dict[str, Dict[str, Any]] = {
    "SMA": {"name":"Simple MA","kind":"overlay","params":{"period":20}},
    "EMA": {"name":"Exponential MA","kind":"overlay","params":{"period":20}},
    "WMA": {"name":"Weighted MA","kind":"overlay","params":{"period":20}},
    "SMMA":{"name":"Smoothed MA","kind":"overlay","params":{"period":20}},
    "TMA": {"name":"Triangular MA","kind":"overlay","params":{"period":20}},
    "RSI": {"name":"RSI","kind":"oscillator","params":{"period":14}},
    "STOCH":{"name":"Stochastic","kind":"oscillator","params":{"k":14,"d":3}},
    "ATR": {"name":"ATR","kind":"oscillator","params":{"period":14}},
    "ADX": {"name":"ADX/+DI/-DI","kind":"oscillator","params":{"period":14}},
    "BBANDS":{"name":"Bollinger Bands","kind":"overlay","params":{"period":20,"mult":2.0}},
    "KELTNER":{"name":"Keltner Channel","kind":"overlay","params":{"period":20,"mult":2.0}},
    "DONCHIAN":{"name":"Donchian","kind":"overlay","params":{"period":20}},
    "ENVELOPES":{"name":"Envelopes","kind":"overlay","params":{"period":20,"pct":0.02}},
    "MACD":{"name":"MACD/Signal/Hist","kind":"oscillator","params":{"fast":12,"slow":26,"signal":9}},
    "MOMENTUM":{"name":"Momentum","kind":"oscillator","params":{"period":10}},
    "ROC":{"name":"Rate of Change","kind":"oscillator","params":{"period":10}},
    "WILLR":{"name":"Williams %R","kind":"oscillator","params":{"period":14}},
    "CCI":{"name":"CCI","kind":"oscillator","params":{"period":20}},
    "AROON":{"name":"Aroon Up/Down","kind":"oscillator","params":{"period":14}},
    "VORTEX":{"name":"Vortex +VI/-VI","kind":"oscillator","params":{"period":14}},
    "AO":{"name":"Awesome Osc","kind":"oscillator","params":{}},
    "AC":{"name":"Accelerator Osc","kind":"oscillator","params":{}},
    "ICHIMOKU":{"name":"Ichimoku (basic)","kind":"overlay","params":{"conversion":9,"base":26,"spanb":52}},
    "PSAR":{"name":"Parabolic SAR","kind":"overlay","params":{"af":0.02,"af_max":0.2}},
    "SUPERtrend":{"name":"SuperTrend","kind":"overlay","params":{"period":10,"mult":3.0}},
    "FRACTAL":{"name":"Fractals (markers)","kind":"overlay","params":{}},
    "FRACTAL_BANDS":{"name":"Fractal Bands","kind":"overlay","params":{"period":20}},
    "BEARS":{"name":"Bears Power","kind":"oscillator","params":{"period":13}},
    "BULLS":{"name":"Bulls Power","kind":"oscillator","params":{"period":13}},
    "SCHAFF":{"name":"Schaff Trend Cycle","kind":"oscillator","params":{"fast":23,"slow":50,"cycle":10}},
    "ZIGZAG":{"name":"ZigZag (pct)","kind":"overlay","params":{"pct":5.0}},
}

# Strategy Defaults
ALL_STRATEGIES = {
    "BASE": {"enabled": True},
    "TREND": {"enabled": True},
    "CHOP": {"enabled": False},
    "CUSTOM1": {"enabled": False},
    "CUSTOM2": {"enabled": False},
    "CUSTOM3": {"enabled": False},
}

# 3 custom slots default
def _default_customs():
    return [
        {"_idx": 1, "enabled": False, "mode": "SIMPLE", "lookback": 3, "tol_pct": 0.1},
        {"_idx": 2, "enabled": False, "mode": "SIMPLE", "lookback": 3, "tol_pct": 0.1},
        {"_idx": 3, "enabled": False, "mode": "SIMPLE", "lookback": 3, "tol_pct": 0.1},
    ]

# ---------------------------- Utils ------------------------------------------
def _cfg() -> Dict[str, Any]:
    cfg = get_config() or {}
    # defaults
    cfg.setdefault("window", {"start":"08:00","end":"17:00","timezone":TIMEZONE})
    cfg.setdefault("live_tf", "M5")
    cfg.setdefault("live_expiry", "5m")
    cfg.setdefault("indicators", {})
    cfg.setdefault("strategies", ALL_STRATEGIES.copy())
    cfg.setdefault("customs", _default_customs())
    cfg.setdefault("symbols_raw", ["frxEURUSD","frxGBPUSD"])
    cfg.setdefault("active_symbols", cfg.get("symbols_raw"))
    cfg.setdefault("bt", {})  # last backtest meta
    return cfg

def _save_cfg(cfg: Dict[str, Any]):
    set_config(cfg)

def _to_list(s: str) -> List[str]:
    if not s: return []
    parts = []
    for p in s.replace(",", " ").split():
        q = p.strip()
        if q: parts.append(q)
    return parts

# ---------------------------- Public pages -----------------------------------
@bp.route("/")
def index():
    cfg = _cfg()
    now = datetime.now(TZ).strftime("%Y-%m-%d %H:%M:%S")
    return render_template("dashboard.html",
        view="index",
        tz=TIMEZONE,
        now=now,
        within=within_window(cfg),
    )

@bp.route("/dashboard")
def dashboard():
    cfg = _cfg()
    now = datetime.now(TZ).strftime("%Y-%m-%d %H:%M:%S")

    # Build available groups for symbols pickers (Deriv + PO)
    available_groups = [
        {"name":"Deriv FRX", "items":[
            "frxEURUSD","frxGBPUSD","frxUSDJPY","frxUSDCHF","frxUSDCAD","frxAUDUSD","frxNZDUSD",
            "frxEURGBP","frxEURJPY","frxGBPJPY","frxEURAUD","frxAUDJPY","frxCADJPY","frxCHFJPY"
        ]},
        {"name":"PO majors", "items":[
            "EURUSD","GBPUSD","USDJPY","USDCHF","USDCAD","AUDUSD","NZDUSD",
            "EURGBP","EURJPY","GBPJPY","EURAUD","AUDJPY","CADJPY","CHFJPY"
        ]},
    ]

    return render_template("dashboard.html",
        view="dashboard",
        tz=TIMEZONE,
        now=now,
        within=within_window(cfg),
        window=cfg.get("window"),
        live_tf=cfg.get("live_tf","M5"),
        live_expiry=cfg.get("live_expiry","5m"),
        symbols_raw=cfg.get("symbols_raw"),
        active_symbols=cfg.get("active_symbols"),
        available_groups=available_groups,
        indicators=cfg.get("indicators"),
        specs=INDICATOR_SPECS,
        strategies=cfg.get("strategies"),
        strategies_all=ALL_STRATEGIES,
        customs=cfg.get("customs") or _default_customs(),
        users=list_users(),
        bt=cfg.get("bt") or {},
        session=session,
    )

# ---------------------------- Auth -------------------------------------------
@bp.route("/login", methods=["GET","POST"])
def login():
    if request.method == "POST":
        if (request.form.get("password") or "") == ADMIN_PASSWORD:
            session["admin"] = True
            flash("Welcome.", "ok")
            return redirect(url_for("dashboard.dashboard"))
        flash("Invalid password.", "error")
    return render_template("dashboard.html", view="login", session=session)

@bp.route("/logout")
def logout():
    session.clear()
    flash("Logged out.")
    return redirect(url_for("dashboard.index"))

# ---------------------------- Window defaults --------------------------------
@bp.route("/update/window", methods=["POST"])
@require_login
def update_window():
    cfg = _cfg()
    w = cfg.get("window") or {}
    w["start"] = request.form.get("start") or w.get("start") or "08:00"
    w["end"] = request.form.get("end") or w.get("end") or "17:00"
    w["timezone"] = request.form.get("timezone") or w.get("timezone") or TIMEZONE
    cfg["window"] = w
    cfg["live_tf"] = request.form.get("live_tf") or cfg.get("live_tf","M5")
    cfg["live_expiry"] = request.form.get("live_expiry") or cfg.get("live_expiry","5m")
    _save_cfg(cfg)
    flash("Window/defaults saved.")
    return redirect(url_for("dashboard.dashboard"))

# ---------------------------- Symbols ----------------------------------------
@bp.route("/update/symbols", methods=["POST"])
@require_login
def update_symbols():
    cfg = _cfg()
    # multi-selects
    deriv_sel = request.form.getlist("symbols_deriv_multi")
    po_sel = request.form.getlist("symbols_po_multi")
    text = _to_list(request.form.get("symbols_text",""))
    combined = list({*deriv_sel, *po_sel, *text})
    cfg["symbols_raw"] = combined
    if request.form.get("convert_po"):
        cfg["active_symbols"] = convert_po_to_deriv(combined)
    else:
        cfg["active_symbols"] = combined
    _save_cfg(cfg)
    flash("Symbols updated.")
    return redirect(url_for("dashboard.dashboard"))

# ---------------------------- Indicators -------------------------------------
@bp.route("/update/indicators", methods=["POST"])
@require_login
def update_indicators():
    cfg = _cfg()
    out = {}
    for key, spec in INDICATOR_SPECS.items():
        enabled = bool(request.form.get(f"ind_{key}_enabled"))
        entry: Dict[str, Any] = {"enabled": enabled}
        for p_name, p_def in spec.get("params", {}).items():
            val = request.form.get(f"ind_{key}_{p_name}", p_def)
            # cast numbers if possible
            try:
                if isinstance(p_def, int): val = int(val)
                elif isinstance(p_def, float): val = float(val)
            except Exception:
                pass
            entry[p_name] = val
        out[key] = entry
    cfg["indicators"] = out
    _save_cfg(cfg)
    flash("Indicators saved.")
    return redirect(url_for("dashboard.dashboard"))

# ---------------------------- Strategies & defaults --------------------------
@bp.route("/update/strategies", methods=["POST"])
@require_login
def update_strategies():
    cfg = _cfg()
    # toggles
    st = {}
    for name in ALL_STRATEGIES.keys():
        st[name] = {"enabled": bool(request.form.get(f"s_{name}"))}
    cfg["strategies"] = st
    # defaults for backtest/live
    cfg["bt"] = cfg.get("bt") or {}
    cfg["bt"]["tf"] = request.form.get("bt_tf") or cfg.get("bt",{}).get("tf") or cfg.get("live_tf","M5")
    cfg["bt"]["expiry"] = request.form.get("bt_expiry") or cfg.get("bt",{}).get("expiry") or cfg.get("live_expiry","5m")
    cfg["live_tf"] = request.form.get("live_tf") or cfg.get("live_tf","M5")
    cfg["live_expiry"] = request.form.get("live_expiry") or cfg.get("live_expiry","5m")
    _save_cfg(cfg)
    flash("Strategy toggles/defaults saved.")
    return redirect(url_for("dashboard.dashboard"))

# ---------------------------- Custom rules -----------------------------------
@bp.route("/update/custom", methods=["POST"])
@require_login
def update_custom():
    cfg = _cfg()
    customs = cfg.get("customs") or _default_customs()
    try:
        slot = int(request.form.get("slot") or 1)
    except Exception:
        slot = 1
    idx = max(1, min(3, slot)) - 1

    entry = customs[idx] if idx < len(customs) else {"_idx": idx+1}
    entry["enabled"] = bool(request.form.get("enabled"))
    entry["mode"] = request.form.get("mode") or entry.get("mode","SIMPLE")
    entry["lookback"] = int(request.form.get("lookback") or entry.get("lookback",3))
    try:
        entry["tol_pct"] = float(request.form.get("tol_pct") or entry.get("tol_pct",0.1))
    except Exception:
        entry["tol_pct"] = entry.get("tol_pct", 0.1)

    sb = (request.form.get("simple_buy") or "").strip()
    ss = (request.form.get("simple_sell") or "").strip()
    if sb: entry["simple_buy"] = sb
    if ss: entry["simple_sell"] = ss

    # JSON rules if provided
    bj = (request.form.get("buy_rule_json") or "").strip()
    sj = (request.form.get("sell_rule_json") or "").strip()
    try:
        entry["buy_rule"] = json.loads(bj) if bj else entry.get("buy_rule")
    except Exception:
        flash(f"CUSTOM {idx+1}: invalid buy_rule JSON ignored", "error")
    try:
        entry["sell_rule"] = json.loads(sj) if sj else entry.get("sell_rule")
    except Exception:
        flash(f"CUSTOM {idx+1}: invalid sell_rule JSON ignored", "error")

    if idx >= len(customs):
        customs.append(entry)
    else:
        customs[idx] = entry

    cfg["customs"] = customs
    _save_cfg(cfg)
    flash(f"CUSTOM {idx+1} saved.")
    return redirect(url_for("dashboard.dashboard"))

# ---------------------------- Users ------------------------------------------
@bp.route("/users/add", methods=["POST"])
@require_login
def users_add():
    telegram_id = (request.form.get("telegram_id") or "").strip()
    tier = (request.form.get("tier") or "free").strip().lower()
    expires_at = (request.form.get("expires_at") or "").strip() or None
    if not telegram_id:
        flash("Telegram ID required.", "error")
        return redirect(url_for("dashboard.dashboard"))
    upsert_user(telegram_id, tier, expires_at)
    flash("User saved.")
    return redirect(url_for("dashboard.dashboard"))

@bp.route("/users/delete", methods=["POST"])
@require_login
def users_delete():
    telegram_id = (request.form.get("telegram_id") or "").strip()
    if telegram_id:
        delete_user(telegram_id)
        flash("User deleted.")
    return redirect(url_for("dashboard.dashboard"))

# ---------------------------- Backtest & screenshot --------------------------
_LAST_JSON = "static/plots/last.json"
_LAST_CSV  = "static/plots/last.csv"

@bp.route("/backtest", methods=["POST"])
@require_login
def backtest():
    cfg = _cfg()

    # Symbols (with optional PO -> Deriv conversion)
    raw = _to_list(request.form.get("bt_symbols",""))
    symbols = convert_po_to_deriv(raw) if request.form.get("convert_po_bt") else raw
    sym = (symbols[0] if symbols else (cfg.get("active_symbols") or ["frxEURUSD"])[0])

    # TF / expiry / strategy
    tf = (request.form.get("bt_tf") or cfg.get("bt",{}).get("tf") or cfg.get("live_tf") or "M5").upper()
    expiry = request.form.get("bt_expiry") or cfg.get("bt",{}).get("expiry") or cfg.get("live_expiry") or "5m"
    strat = (request.form.get("bt_strategy") or "BASE").upper()

    # Inputs
    up = request.files.get("bt_csv")
    use_server = bool(request.form.get("use_server"))
    count = int(request.form.get("bt_count") or 300)

    df = None
    warnings: List[str] = []

    try:
        if up and up.filename:
            df = load_csv(up)
        else:
            if not use_server:
                raise RuntimeError("No CSV uploaded. Tick “Use Deriv server fetch” to fetch candles automatically.")
            # Map TF → seconds
            gmap = {"M1":60,"M2":120,"M3":180,"M5":300,"M10":600,"M15":900,"M30":1800,"H1":3600,"H4":14400,"D1":86400}
            gran = gmap.get(tf, 300)
            # Rough days estimate from requested count so charts look good
            est_days = max(1, min(10, int((count * gran) / 86400) + 1))
            df = fetch_deriv_history(sym, granularity_sec=gran, days=est_days)
            if df is None or df.empty:
                raise RuntimeError("Empty candle response from Deriv.")

        # Backtest run (indicators are taken from cfg["indicators"])
        signals, stats = backtest_run(df, strat, cfg.get("indicators") or {}, expiry)
        plot_name = plot_signals(df, signals, cfg.get("indicators") or {}, strat, tf, expiry)

        # Persist “last.*” for download links
        os.makedirs("static/plots", exist_ok=True)
        with open(_LAST_JSON, "w") as f:
            json.dump({
                "symbol": sym, "tf": tf, "expiry": expiry, "strategy": strat,
                "stats": stats, "signals": [
                    {"time": s["index"].isoformat(), "dir": s["direction"], "expiry": s["expiry_idx"].isoformat()}
                    for s in signals
                ]
            }, f, indent=2)
        # CSV: dump the dataframe used
        df.reset_index().rename(columns={"index":"time"}).to_csv(_LAST_CSV, index=False)

        cfg["bt"] = {
            "tf": tf, "expiry": expiry, "strategy": strat, "plot_name": plot_name,
            "summary": f"{stats['wins']}W {stats['loss']}L {stats['draw']}D • WR={stats['win_rate']:.1f}%",
            "warnings": warnings
        }
        _save_cfg(cfg)
        flash("Backtest complete.")
    except Exception as e:
        cfg["bt"] = {"error": str(e)}
        _save_cfg(cfg)
        flash(f"Backtest error: {e}", "error")

    return redirect(url_for("dashboard.dashboard"))

@bp.route("/backtest/last.json")
def backtest_last_json():
    if not os.path.exists(_LAST_JSON):
        return jsonify({"ok": False, "error": "No backtest yet"}), 404
    with open(_LAST_JSON, "r") as f:
        js = json.load(f)
    return jsonify(js)

@bp.route("/backtest/last.csv")
def backtest_last_csv():
    if not os.path.exists(_LAST_CSV):
        return Response("No backtest yet", status=404)
    return send_from_directory("static/plots", "last.csv", as_attachment=True, download_name="backtest_last.csv")

@bp.route("/plot/<path:name>")
def plot_file(name: str):
    return send_from_directory("static/plots", name)

# ---------------------------- Telegram ---------------------------------------
@bp.route("/telegram/test", methods=["POST"])
@require_login
def telegram_test():
    # Optional: send custom text to all configured groups
    text = (request.form.get("text") or "🧪 Test message from dashboard").strip()
    ok, info = tg_test_all()
    if ok:
        flash("Sent test messages (see last send status on live panel).")
    else:
        flash(f"Telegram test error: {info}", "error")
    return redirect(url_for("dashboard.dashboard"))

@bp.route("/telegram/diag")
def telegram_diag():
    import requests
    token = (os.getenv("TELEGRAM_BOT_TOKEN") or "").strip()
    out = {
        "ok": False,
        "configured_chats": TIER_TO_CHAT,
        "caps": DAILY_CAPS,
    }
    if token:
        try:
            r = requests.get(f"https://api.telegram.org/bot{token}/getMe", timeout=10)
            out["getMe"] = r.json()
            out["ok"] = bool(out["getMe"].get("ok"))
        except Exception as e:
            out["getMe"] = {"ok": False, "error": str(e)}
    return jsonify(out)

# ---------------------------- Live Engine ------------------------------------
@bp.route("/live/start")
@require_login
def live_start():
    ok, msg = ENGINE.start()
    flash(f"Live engine: {msg}")
    return jsonify({"ok": ok, "status": ENGINE.status()})

@bp.route("/live/stop")
@require_login
def live_stop():
    ok, msg = ENGINE.stop()
    flash(f"Live engine: {msg}")
    return jsonify({"ok": ok, "status": ENGINE.status()})

@bp.route("/live/debug/on")
@require_login
def live_dbg_on():
    ENGINE.set_debug(True)
    return jsonify({"ok": True, "status": ENGINE.status()})

@bp.route("/live/debug/off")
@require_login
def live_dbg_off():
    ENGINE.set_debug(False)
    return jsonify({"ok": True, "status": ENGINE.status()})

@bp.route("/live/status")
def live_status():
    return jsonify({"status": ENGINE.status()})

@bp.route("/live/tally")
def live_tally():
    s = ENGINE.status()
    # Match the right-column card IDs in your template
    tallies = s.get("tallies") or s.get("tally", {}).get("by_tier") or {}
    out = {
        "tally": {
            "free": tallies.get("free", 0),
            "basic": tallies.get("basic", 0),
            "pro": tallies.get("pro", 0),
            "vip": tallies.get("vip", 0),
            "all": tallies.get("all", tallies.get("total", 0)),
        }
    }
    return jsonify(out)

# ---------------------------- Misc -------------------------------------------
@bp.route("/_up")
def up_check():
    return jsonify({"ok": True, "time": datetime.utcnow().isoformat()+"Z"})

# ---------------------------- Static helpers for template --------------------
# (the template uses url_for('dashboard.plot_file', name=...))
