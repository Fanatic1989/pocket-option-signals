import os, json, re
from io import StringIO
from datetime import datetime, timedelta
import pandas as pd
from flask import Blueprint, render_template, request, redirect, url_for, session, flash, jsonify

from utils import exec_sql, get_config, set_config, within_window, TZ, TIMEZONE, log
from indicators import INDICATOR_SPECS
from strategies import run_backtest_core_binary
from rules import parse_natural_rule, parse_natural_pair

bp = Blueprint('dashboard', __name__)

# -------------------- Auth --------------------
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
    cfg = get_config()
    return render_template('dashboard.html', view='login', window=cfg['window'], tz=TIMEZONE)

@bp.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('dashboard.index'))

@bp.route('/')
def index():
    cfg = get_config()
    return render_template('dashboard.html', view='index', within=within_window(cfg),
                           now=datetime.now(TZ).strftime('%Y-%m-%d %H:%M:%S'),
                           tz=TIMEZONE, window=cfg['window'])

# -------------------- Dashboard --------------------
@bp.route('/dashboard')
@require_login
def dashboard():
    cfg = get_config()
    rows = exec_sql("SELECT telegram_id, tier, COALESCE(expires_at,'') FROM users", fetch=True) or []
    users = [{"telegram_id": r[0], "tier": r[1], "expires_at": r[2] or None} for r in rows]
    active_symbols = (cfg.get("symbols") or [])
    for i in (1,2,3):
        cfg.setdefault(f'custom{i}', {
            "enabled": False, "mode": "SIMPLE",
            "simple_buy": "", "simple_sell": "",
            "buy_rule": "", "sell_rule": "",
            "tol_pct": 0.1, "lookback": 3
        })
    return render_template('dashboard.html', view='dashboard', window=cfg['window'],
                           strategies=cfg['strategies'], indicators=cfg['indicators'],
                           custom1=cfg['custom1'], custom2=cfg['custom2'], custom3=cfg['custom3'],
                           users=users, specs=INDICATOR_SPECS, tz=TIMEZONE,
                           now=datetime.now(TZ).strftime('%Y-%m-%d %H:%M:%S'),
                           active_symbols=active_symbols)

# -------------------- Config Updates --------------------
@bp.route('/update_window', methods=['POST'])
@require_login
def update_window():
    cfg = get_config()
    cfg['window']['start'] = request.form.get('start', cfg['window']['start'])
    cfg['window']['end'] = request.form.get('end', cfg['window']['end'])
    cfg['window']['timezone'] = request.form.get('timezone', cfg['window']['timezone'])
    set_config(cfg); flash("Window updated.")
    return redirect(url_for('dashboard.dashboard'))

@bp.route('/update_strategies', methods=['POST'])
@require_login
def update_strategies():
    cfg = get_config()
    for name in list(cfg['strategies'].keys()):
        cfg['strategies'][name]['enabled'] = bool(request.form.get(f's_{name}'))
    set_config(cfg); flash("Strategies updated.")
    return redirect(url_for('dashboard.dashboard'))

@bp.route('/update_symbols', methods=['POST'])
@require_login
def update_symbols():
    cfg = get_config()
    raw = (request.form.get('symbols') or "").strip()
    symbols = [s.strip() for s in re.split(r"[,\s]+", raw) if s.strip()]
    cfg['symbols'] = symbols
    set_config(cfg)
    flash(f"Active symbols updated: {', '.join(symbols) if symbols else '(none)'}")
    return redirect(url_for('dashboard.dashboard'))

@bp.route('/update_custom', methods=['POST'])
@require_login
def update_custom():
    slot = (request.form.get('slot') or '1').strip()
    return redirect(url_for('dashboard.dashboard'))  # simplified for brevity

# -------------------- Live Engine --------------------
from live_engine import ENGINE, tg_test

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
