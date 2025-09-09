import os, json, re
from io import StringIO
from datetime import datetime
import pandas as pd
import requests
from flask import Blueprint, render_template, request, redirect, url_for, session, flash, jsonify

from utils import exec_sql, get_config, set_config, within_window, TZ, TIMEZONE
from indicators import INDICATOR_SPECS
from strategies import run_backtest_core_binary
from rules import parse_natural_rule
from data_fetch import deriv_csv_path as _deriv_csv_path, fetch_one_symbol as _fetch_one_symbol
from live_engine import ENGINE, tg_test

bp = Blueprint('dashboard', __name__)

# ---------- Curated symbol lists ----------
DERIV_FRX = [
    "frxEURUSD","frxGBPUSD","frxUSDJPY","frxUSDCHF","frxUSDCAD","frxAUDUSD","frxNZDUSD",
    "frxEURGBP","frxEURJPY","frxEURCHF","frxEURAUD","frxGBPAUD","frxGBPJPY","frxGBPNZD",
    "frxAUDJPY","frxAUDCAD","frxAUDCHF","frxCADJPY","frxCADCHF","frxCHFJPY","frxNZDJPY",
    "frxEURNZD","frxEURCAD","frxGBPCAD","frxGBPCHF","frxNZDCHF","frxNZDCAD"
]
PO_MAJOR = [
    "EURUSD","GBPUSD","USDJPY","USDCHF","USDCAD","AUDUSD","NZDUSD","EURGBP","EURJPY","GBPJPY"
]
AVAILABLE_GROUPS = [
    {"label":"Deriv (frx*)", "items": DERIV_FRX},
    {"label":"Pocket Option majors", "items": PO_MAJOR},
]

# ---------- Helpers ----------
def _is_po_symbol(sym: str) -> bool:
    s = (sym or "").upper()
    return s in PO_MAJOR or bool(re.fullmatch(r"[A-Z]{6}", s))

def _to_deriv(sym: str) -> str:
    if not sym:
        return sym
    s = sym.strip()
    if s.startswith("frx"):
        return s
    sU = s.upper().replace("/", "")
    if _is_po_symbol(sU):
        return "frx" + sU
    return s

def _merge_unique(seq):
    seen, out = set(), []
    for x in seq:
        if not x:
            continue
        if x not in seen:
            out.append(x); seen.add(x)
    return out

def _cfg_dict(x):
    """Ensure config is a dict even if storage returned JSON string or None."""
    if isinstance(x, dict):
        return x
    if isinstance(x, str):
        try:
            j = json.loads(x)
            return j if isinstance(j, dict) else {}
        except Exception:
            return {}
    return {}

# ---------------- Auth ----------------
def require_login(func):
    from functools import wraps
    @wraps(func)
    def wrapper(*args, **kwargs):
        if not session.get("admin"):
            return redirect(url_for('dashboard.login', next=request.path))
        return func(*args, **kwargs)
    return wrapper

@bp.route('/_up', methods=['GET','HEAD'])
def up_check():
    return "OK", 200

@bp.route('/login', methods=['GET','POST'])
def login():
    if request.method == 'POST':
        if request.form.get('password') == os.getenv('ADMIN_PASSWORD','admin123'):
            session['admin'] = True
            flash("Logged in.")
            return redirect(request.args.get('next') or url_for('dashboard.dashboard'))
        flash("Invalid password")
    cfg = _cfg_dict(get_config())
    return render_template('dashboard.html', view='login', window=cfg.get('window', {}), tz=TIMEZONE)

@bp.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('dashboard.index'))

@bp.route('/')
def index():
    cfg = _cfg_dict(get_config())
    return render_template('dashboard.html', view='index', within=within_window(cfg),
                           now=datetime.now(TZ).strftime('%Y-%m-%d %H:%M:%S'),
                           tz=TIMEZONE, window=cfg.get('window', {}))

# ---------------- Dashboard ----------------
@bp.route('/dashboard')
@require_login
def dashboard():
    cfg = _cfg_dict(get_config())
    rows = exec_sql("SELECT telegram_id, tier, COALESCE(expires_at,'') FROM users", fetch=True) or []
    users = [{"telegram_id": r[0], "tier": r[1], "expires_at": r[2] or None} for r in rows]

    customs = [
        dict(_cfg_dict(cfg.get('custom1')), _idx=1),
        dict(_cfg_dict(cfg.get('custom2')), _idx=2),
        dict(_cfg_dict(cfg.get('custom3')), _idx=3),
    ]

    strategies_core = _cfg_dict(cfg.get('strategies')) or {
        "BASE":{"enabled": True},
        "TREND":{"enabled": False},
        "CHOP":{"enabled": False},
    }
    strategies_all = dict(strategies_core)
    for i, c in enumerate(customs, start=1):
        strategies_all[f"CUSTOM{i}"] = {"enabled": bool(c.get("enabled"))}

    bt = session.get("bt", {})
    return render_template('dashboard.html', view='dashboard',
                           window=cfg.get('window', {}),
                           strategies_all=strategies_all,
                           strategies=strategies_core,
                           indicators=_cfg_dict(cfg.get('indicators')),
                           specs=INDICATOR_SPECS,
                           customs=customs,
                           active_symbols=cfg.get("symbols") or [],
                           symbols_raw=cfg.get("symbols_raw") or [],
                           available_groups=AVAILABLE_GROUPS,
                           users=users,
                           bt=bt,
                           live_tf=cfg.get("live_tf","M5"),
                           live_expiry=cfg.get("live_expiry","5m"),
                           now=datetime.now(TZ).strftime('%Y-%m-%d %H:%M:%S'),
                           tz=TIMEZONE)

# ---------------- Config: Window / Strategies / Symbols ----------------
@bp.route('/update_window', methods=['POST'])
@require_login
def update_window():
    cfg = _cfg_dict(get_config())
    cfg.setdefault('window', {"start":"08:00","end":"17:00","timezone":TIMEZONE})
    cfg['window']['start'] = request.form.get('start', cfg['window']['start'])
    cfg['window']['end']   = request.form.get('end',   cfg['window']['end'])
    cfg['window']['timezone'] = request.form.get('timezone', cfg['window']['timezone'])
    if request.form.get('live_tf'): cfg['live_tf'] = request.form.get('live_tf').upper()
    if request.form.get('live_expiry'): cfg['live_expiry'] = request.form.get('live_expiry')
    set_config(cfg); flash("Trading window & live defaults updated.")
    return redirect(url_for('dashboard.dashboard'))

@bp.route('/update_strategies', methods=['POST'])
@require_login
def update_strategies():
    cfg = _cfg_dict(get_config())
    cfg.setdefault('strategies', {"BASE":{"enabled":True},"TREND":{"enabled":False},"CHOP":{"enabled":False}})
    for name in list(cfg['strategies'].keys()):
        cfg['strategies'][name]['enabled'] = bool(request.form.get(f's_{name}'))
    for i in (1,2,3):
        box = bool(request.form.get(f's_CUSTOM{i}'))
        key = f'custom{i}'
        cfg.setdefault(key, {})
        cfg[key]['enabled'] = box
        cfg['strategies'][f"CUSTOM{i}"] = {"enabled": box}
    set_config(cfg)
    flash("Strategies (including CUSTOM) updated.")
    return redirect(url_for('dashboard.dashboard'))

@bp.route('/update_symbols', methods=['POST'])
@require_login
def update_symbols():
    cfg = _cfg_dict(get_config())
    sel_po     = request.form.getlist('symbols_po_multi')
    sel_deriv  = request.form.getlist('symbols_deriv_multi')
    raw = (request.form.get('symbols_text') or "").strip()
    text_syms = [s.strip() for s in re.split(r"[,\s]+", raw) if s.strip()]
    convert_po = bool(request.form.get('convert_po'))

    merged = _merge_unique(sel_po + sel_deriv + text_syms)
    normalized = [_to_deriv(s) if convert_po else s for s in merged]

    cfg['symbols'] = normalized
    cfg['symbols_raw'] = merged
    set_config(cfg)

    shown = ", ".join(normalized) if normalized else "(none)"
    flash(("Active symbols saved (PO→Deriv conversion ON): " if convert_po else "Active symbols saved: ") + shown)
    return redirect(url_for('dashboard.dashboard'))

# ---------------- Indicators ----------------
@bp.route('/update_indicators', methods=['POST'])
@require_login
def update_indicators():
    cfg = _cfg_dict(get_config())
    inds = _cfg_dict(cfg.get("indicators"))
    for key, spec in INDICATOR_SPECS.items():
        enabled = bool(request.form.get(f'ind_{key}_enabled'))
        inds.setdefault(key, {})
        inds[key]['enabled'] = enabled
        for p in spec.get("params", {}).keys():
            form_key = f'ind_{key}_{p}'
            if form_key in request.form and request.form.get(form_key) != "":
                val = request.form.get(form_key)
                try:
                    inds[key][p] = int(val) if re.fullmatch(r"-?\d+", val) else float(val)
                except Exception:
                    inds[key][p] = val
        if key == "sma" and 'period' not in inds[key]:
            inds[key]['period'] = spec['params']['period']
    cfg['indicators'] = inds
    set_config(cfg)
    flash("Indicators updated.")
    return redirect(url_for('dashboard.dashboard'))

# ---------------- Custom strategies ----------------
@bp.route('/update_custom', methods=['POST'])
@require_login
def update_custom():
    slot = (request.form.get('slot') or '1').strip()
    field_prefix = f'custom{slot}'
    cfg = _cfg_dict(get_config())
    cfg.setdefault(field_prefix, {})

    mode = (request.form.get('mode', 'SIMPLE') or 'SIMPLE').upper()
    cfg[field_prefix]['enabled'] = bool(request.form.get('enabled'))
    cfg[field_prefix]['mode'] = mode

    if mode == "SIMPLE":
        cfg[field_prefix]['simple_buy']  = request.form.get('simple_buy', '')
        cfg[field_prefix]['simple_sell'] = request.form.get('simple_sell', '')
        cfg[field_prefix]['buy_rule']  = parse_natural_rule(cfg[field_prefix]['simple_buy'])
        cfg[field_prefix]['sell_rule'] = parse_natural_rule(cfg[field_prefix]['simple_sell'])
    else:
        try:
            cfg[field_prefix]['buy_rule']  = json.loads(request.form.get('buy_rule_json','{}'))
        except Exception: cfg[field_prefix]['buy_rule'] = {}
        try:
            cfg[field_prefix]['sell_rule'] = json.loads(request.form.get('sell_rule_json','{}'))
        except Exception: cfg[field_prefix]['sell_rule'] = {}

    try: cfg[field_prefix]['tol_pct'] = float(request.form.get('tol_pct', cfg[field_prefix].get('tol_pct', 0.1)))
    except Exception: pass
    try: cfg[field_prefix]['lookback'] = int(request.form.get('lookback', cfg[field_prefix].get('lookback', 3)))
    except Exception: pass

    cfg.setdefault('strategies', {})
    cfg['strategies'][f"CUSTOM{slot}"] = {"enabled": bool(cfg[field_prefix]['enabled'])}

    set_config(cfg)
    flash(f"Custom #{slot} saved.")
    return redirect(url_for('dashboard.dashboard'))

# ---------------- Deriv pull (multi-pair) ----------------
@bp.route('/deriv_fetch', methods=['POST'])
@require_login
def deriv_fetch():
    app_id = os.getenv("DERV_APP_ID", None) or os.getenv("DERIV_APP_ID", "1089")
    symbols = (request.form.get('fetch_symbols') or "").strip()
    tf = (request.form.get('fetch_tf') or "M5").upper()
    count = int(request.form.get('fetch_count') or "300")
    convert_po = bool(request.form.get('convert_po_fetch'))

    gran_map = {"M1":60,"M2":120,"M3":180,"M5":300,"M10":600,"M15":900,"M30":1800,"H1":3600,"H4":14400,"D1":86400}
    gran = gran_map.get(tf, 300)

    pairs_in = [s.strip() for s in re.split(r"[,\s]+", symbols) if s.strip()]
    pairs = [_to_deriv(s) if convert_po else s for s in pairs_in]

    ok = 0; fail = []
    for sym in pairs:
        try:
            _fetch_one_symbol(app_id, sym, gran, count)
            ok += 1
        except Exception as e:
            fail.append(f"{sym}: {e}")

    if ok: flash(f"Deriv: saved {ok} symbol(s) @ {tf}")
    if fail: flash("Errors: " + "; ".join(fail))
    return redirect(url_for('dashboard.dashboard'))

# ---------------- Backtest (POST only) ----------------
@bp.route('/backtest', methods=['POST'])
@require_login
def backtest():
    cfg = _cfg_dict(get_config())

    tf = (request.form.get('bt_tf') or 'M5').upper()
    expiry = request.form.get('bt_expiry') or '5m'
    strategy = (request.form.get('bt_strategy') or 'BASE').upper()
    use_server = bool(request.form.get('use_server'))
    app_id = os.getenv("DERV_APP_ID", None) or os.getenv("DERIV_APP_ID", "1089")
    count = int(request.form.get('bt_count') or "300")
    convert_po = bool(request.form.get('convert_po_bt'))

    gran_map = {"M1":60,"M2":120,"M3":180,"M5":300,"M10":600,"M15":900,"M30":1800,"H1":3600,"H4":14400,"D1":86400}
    gran = gran_map.get(tf, 300)

    raw_syms = request.form.get('bt_symbols') or " ".join(cfg.get('symbols') or [])
    from_multi = request.form.getlist('bt_symbols_multi')
    symbols_in = from_multi if from_multi else [s.strip() for s in re.split(r"[,\s]+", raw_syms) if s.strip()]
    symbols = [_to_deriv(s) if convert_po else s for s in symbols_in]

    uploaded = request.files.get('bt_csv')
    results = []
    summary = {"trades":0,"wins":0,"losses":0,"draws":0,"winrate":0.0}

    def run_one(sym, df):
        # build cfg_run safely
        if strategy in ("CUSTOM1","CUSTOM2","CUSTOM3"):
            sid = strategy[-1]
            cfg_run = dict(cfg)
            cfg_run["custom"] = _cfg_dict(cfg.get(f"custom{sid}"))
            core = "CUSTOM"
        else:
            cfg_run = dict(cfg)
            core = strategy
        bt = run_backtest_core_binary(df, core, cfg_run, tf, expiry)
        results.append({"symbol": sym, "trades": bt.trades, "wins": bt.wins, "losses": bt.losses, "draws": bt.draws, "winrate": round(bt.winrate*100,2), "rows": bt.rows})
        summary["trades"] += bt.trades
        summary["wins"]   += bt.wins
        summary["losses"] += bt.losses
        summary["draws"]  += bt.draws

    try:
        if uploaded and uploaded.filename:
            data = uploaded.read().decode("utf-8", errors="ignore")
            df = pd.read_csv(StringIO(data))
            df.columns = [c.strip().lower() for c in df.columns]
            if "timestamp" in df.columns: df["timestamp"] = pd.to_datetime(df["timestamp"])
            for c in ("open","high","low","close"):
                df[c] = pd.to_numeric(df[c], errors="coerce")
            df = df.dropna(subset=["close"]).sort_values("timestamp").reset_index(drop=True)
            run_one(symbols[0] if symbols else "CSV", df)
        else:
            for sym in symbols:
                if use_server:
                    _fetch_one_symbol(app_id, sym, gran, count)
                path = _deriv_csv_path(sym, gran)
                if not os.path.exists(path):
                    raise RuntimeError(f"No server data for {sym} @ {tf}. Pull from Deriv first or upload CSV.")
                df = pd.read_csv(path)
                df.columns = [c.strip().lower() for c in df.columns]
                if "timestamp" in df.columns: df["timestamp"] = pd.to_datetime(df["timestamp"])
                for c in ("open","high","low","close"):
                    df[c] = pd.to_numeric(df[c], errors="coerce")
                df = df.dropna(subset=["close"]).sort_values("timestamp").reset_index(drop=True)
                run_one(sym, df)

        summary["winrate"] = round((summary["wins"]/summary["trades"])*100,2) if summary["trades"] else 0.0
        session["bt"] = {"summary": summary, "results": results, "tf": tf, "expiry": expiry, "strategy": strategy}
        flash("Backtest complete.")
    except Exception as e:
        session["bt"] = {"error": str(e)}
        flash(f"Backtest error: {e}")
    return redirect(url_for('dashboard.dashboard'))

# ---------------- Live Engine endpoints ----------------
@bp.route('/live/status')
def live_status():
    return jsonify({"ok": True, "status": ENGINE.status()})

@bp.route('/live/start', methods=['POST','GET'])
def live_start():
    ok, msg = ENGINE.start()
    if request.method == 'GET':
        return jsonify({"ok": ok, "msg": msg})
    flash(f"Live: {msg}")
    return redirect(url_for('dashboard.dashboard'))

@bp.route('/live/stop', methods=['POST','GET'])
def live_stop():
    ok, msg = ENGINE.stop()
    if request.method == 'GET':
        return jsonify({"ok": ok, "msg": msg})
    flash(f"Live: {msg}")
    return redirect(url_for('dashboard.dashboard'))

@bp.route('/telegram/test', methods=['POST','GET'])
def telegram_test():
    ok, msg = tg_test()
    if request.method == 'GET':
        return jsonify({"ok": ok, "msg": msg})
    flash("Telegram OK" if ok else f"Telegram error: {msg}")
    return redirect(url_for('dashboard.dashboard'))

@bp.route('/telegram/diag')
def telegram_diag():
    from live_engine import _send_telegram, TELEGRAM_CHAT_KEYS
    token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    if not token:
        return jsonify({"ok": False, "error": "Missing TELEGRAM_BOT_TOKEN"}), 200
    try:
        r = requests.get(f"https://api.telegram.org/bot{token}/getMe", timeout=8)
        getme = r.json()
    except Exception as e:
        getme = {"ok": False, "error": str(e)}
    configured = {k: os.getenv(k, "").strip() for k in TELEGRAM_CHAT_KEYS if os.getenv(k, "").strip()}
    ok, info = _send_telegram("🧪 Telegram DIAG: test message from Pocket Option Signals.")
    masked = token[:9] + "..." + token[-6:] if len(token) > 18 else "***"
    return jsonify({"ok": ok, "token_masked": masked, "getMe": getme, "configured_chats": configured, "send_result": info})
