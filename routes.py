# routes.py
import os, json
from io import StringIO
from datetime import datetime, timedelta
import pandas as pd
import pytz
from flask import Blueprint, render_template, request, redirect, url_for, session, flash

from utils import exec_sql, get_config, set_config, within_window, TZ, TIMEZONE, log
from indicators import INDICATOR_SPECS
from strategies import run_backtest_core_binary
from rules import parse_natural_rule

bp = Blueprint('dashboard', __name__)

def require_login(func):
    from functools import wraps
    @wraps(func)
    def wrapper(*args, **kwargs):
        if not session.get("admin"):
            return redirect(url_for('dashboard.login', next=request.path))
        return func(*args, **kwargs)
    return wrapper

@bp.route('/login', methods=['GET','POST'])
def login():
    if request.method == 'POST':
        if request.form.get('password') == os.getenv('ADMIN_PASSWORD','admin123'):
            session['admin'] = True; flash("Logged in.")
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

@bp.route('/dashboard')
@require_login
def dashboard():
    cfg = get_config()
    rows = exec_sql("SELECT telegram_id, tier, COALESCE(expires_at,'') FROM users", fetch=True) or []
    users = [{"telegram_id": r[0], "tier": r[1], "expires_at": r[2] or None} for r in rows]
    return render_template('dashboard.html', view='dashboard', window=cfg['window'],
                           strategies=cfg['strategies'], indicators=cfg['indicators'], custom=cfg['custom'],
                           users=users, specs=INDICATOR_SPECS, tz=TIMEZONE, 
                           now=datetime.now(TZ).strftime('%Y-%m-%d %H:%M:%S'))

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

@bp.route('/update_indicators', methods=['POST'])
@require_login
def update_indicators():
    cfg = get_config(); ind = cfg['indicators']
    for key, spec in INDICATOR_SPECS.items():
        ind.setdefault(key, {"enabled": False, **spec["params"]})
        ind[key]["enabled"] = bool(request.form.get(f"{key}_enabled"))
        for pkey, default in spec["params"].items():
            raw = request.form.get(f"{key}_{pkey}", default)
            try:
                ind[key][pkey] = float(raw) if "." in str(raw) else int(raw)
            except:
                ind[key][pkey] = default
    cfg['indicators'] = ind
    set_config(cfg); flash("Indicators saved.")
    return redirect(url_for('dashboard.dashboard'))

@bp.route('/update_custom', methods=['POST'])
@require_login
def update_custom():
    cfg = get_config()
    c = cfg.get('custom', {})
    c['enabled'] = bool(request.form.get('custom_enabled'))
    c['mode'] = (request.form.get('mode') or 'SIMPLE').upper()
    try: c['tol_pct'] = float(request.form.get('tol_pct', c.get('tol_pct',0.1)))
    except: pass
    try: c['lookback'] = int(request.form.get('lookback', c.get('lookback',3)))
    except: pass

    if c['mode'] == 'SIMPLE':
        c['simple_buy']  = (request.form.get('simple_buy') or '').strip()
        c['simple_sell'] = (request.form.get('simple_sell') or '').strip()
        c['buy_rule']  = parse_natural_rule(c['simple_buy'])
        c['sell_rule'] = parse_natural_rule(c['simple_sell'])
        if not c['buy_rule'] and not c['sell_rule']:
            flash("Simple rules could not be parsed; refine wording or switch to Expert.")
    else:
        c['buy_rule']  = (request.form.get('buy_rule') or '').strip()
        c['sell_rule'] = (request.form.get('sell_rule') or '').strip()

    cfg['custom'] = c
    set_config(cfg); flash("Custom rules saved.")
    return redirect(url_for('dashboard.dashboard'))

def filter_df_by_month_and_weekdays(df: pd.DataFrame, month_str: str, weekdays):
    from calendar import monthrange
    if "timestamp" not in df.columns: return df
    try:
        year, month = map(int, month_str.split("-"))
        start = datetime(year, month, 1, 0, 0, 0)
        last_day = monthrange(year, month)[1]
        end = datetime(year, month, last_day, 23, 59, 59)
    except Exception:
        return df
    wkmap = {"Mon":0,"Tue":1,"Wed":2,"Thu":3,"Fri":4,"Sat":5,"Sun":6}
    allowed = {wkmap[n] for n in (weekdays or []) if n in wkmap}
    _df = df.copy()
    _df["ts"] = pd.to_datetime(_df["timestamp"])
    _df = _df[(_df["ts"] >= start) & (_df["ts"] <= end)]
    if allowed:
        _df = _df[_df["ts"].dt.weekday.isin(allowed)]
    _df = _df.drop(columns=["ts"])
    return _df

def filter_df_last_n_days(df: pd.DataFrame, n_days: int):
    if "timestamp" not in df.columns or n_days <= 0: return df
    end = datetime.utcnow()
    start = end - timedelta(days=n_days)
    _df = df.copy()
    _df["ts"] = pd.to_datetime(_df["timestamp"])
    _df = _df[(_df["ts"] >= start) & (_df["ts"] <= end)].drop(columns=["ts"])
    return _df

@bp.route('/backtest', methods=['POST'])
@require_login
def backtest():
    cfg = get_config()
    use_server = bool(request.form.get('use_server')); df = None

    if not use_server:
        file = request.files.get('file')
        if file and file.filename:
            try:
                raw = file.read()
                try: df = pd.read_csv(StringIO(raw.decode('utf-8')))
                except Exception: df = pd.read_csv(StringIO(raw.decode('latin-1')))
            except Exception as e:
                flash(f"CSV parse error: {e}"); return redirect(url_for('dashboard.dashboard'))
        else:
            flash("Please choose a CSV file or tick 'Use server data'."); return redirect(url_for('dashboard.dashboard'))
    else:
        path = os.getenv('DERIV_SAVE_PATH','/tmp/deriv_last_month.csv')
        if not os.path.exists(path):
            flash("No server data found. Click 'Pull from Deriv' first."); return redirect(url_for('dashboard.dashboard'))
        try: df = pd.read_csv(path)
        except Exception as e:
            flash(f"Server CSV parse error: {e}"); return redirect(url_for('dashboard.dashboard'))

    df.columns = [c.strip().lower() for c in df.columns]
    if 'timestamp' in df.columns:
        try: df['timestamp'] = pd.to_datetime(df['timestamp'])
        except: df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    for c in ('open','high','low','close'):
        if c in df.columns: df[c] = pd.to_numeric(df[c], errors='coerce')
    if 'close' in df.columns: df = df.dropna(subset=['close'])
    if 'timestamp' in df.columns: df = df.sort_values('timestamp').reset_index(drop=True)

    last_n_raw = (request.form.get('last_n') or '').strip()
    month_str = (request.form.get('month') or '').strip()
    weekdays = request.form.getlist('wd')

    if last_n_raw:
        try: df = filter_df_last_n_days(df, int(last_n_raw))
        except: pass
    elif month_str:
        df = filter_df_by_month_and_weekdays(df, month_str, weekdays or ['Mon','Tue','Wed','Thu','Fri'])

    if df.empty:
        flash('No candles after applying date filters.'); return redirect(url_for('dashboard.dashboard'))

    strategy = request.form.get('strategy','BASE')
    tf = request.form.get('tf','M5')
    expiry_label = request.form.get('expiry','5m')

    bt = run_backtest_core_binary(df, strategy, cfg, tf, expiry_label)

    rows = exec_sql("SELECT telegram_id, tier, COALESCE(expires_at,'') FROM users", fetch=True) or []
    users = [{"telegram_id": r[0], "tier": r[1], "expires_at": r[2] or None} for r in rows]

    return render_template('dashboard.html', view='dashboard', window=cfg['window'],
        strategies=cfg['strategies'], indicators=cfg['indicators'], custom=cfg['custom'],
        users=users, specs=INDICATOR_SPECS, tz=TIMEZONE, now=datetime.now(TZ).strftime('%Y-%m-%d %H:%M:%S'),
        bt=bt, sel_inds=[])

# ---- Deriv fetch via websocket ----
from websocket import WebSocketApp
DERIV_SAVE_PATH = os.getenv('DERIV_SAVE_PATH','/tmp/deriv_last_month.csv')

def fetch_deriv_candles(app_id: str, symbol: str, granularity_sec: int, count: int = 1440) -> str:
    result = {'done': False, 'error': None}
    def on_message(ws, message):
        try:
            data = json.loads(message)
            if 'candles' in data:
                df = pd.DataFrame(data['candles'])
                if df.empty: result['error'] = 'No candles returned'
                else:
                    df.rename(columns={'epoch':'timestamp'}, inplace=True)
                    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
                    df = df[['timestamp','open','high','low','close']]
                    df.to_csv(DERIV_SAVE_PATH, index=False)
            elif 'error' in data:
                result['error'] = data['error'].get('message','Deriv API error')
        except Exception as e:
            result['error'] = f'Parse error: {e}'
        finally:
            result['done'] = True; ws.close()
    def on_open(ws):
        req = {'candles': symbol, 'count': int(count), 'granularity': int(granularity_sec), 'end': 'latest'}
        ws.send(json.dumps(req))
    url = f"wss://ws.derivws.com/websockets/v3?app_id={app_id or '1089'}"
    ws = WebSocketApp(url, on_open=on_open, on_message=on_message)
    import threading, time
    th = threading.Thread(target=ws.run_forever, daemon=True); th.start()
    for _ in range(120):
        if result['done']: break
        time.sleep(0.1)
    if not result['done']: raise RuntimeError('Timeout connecting to Deriv')
    if result['error']: raise RuntimeError(result['error'])
    return DERIV_SAVE_PATH

@bp.route('/deriv_fetch', methods=['POST'])
@require_login
def deriv_fetch():
    app_id = (request.form.get('app_id') or '').strip() or '1089'
    symbol = request.form.get('symbol', '').strip()
    gran = int(request.form.get('granularity', '300'))
    count = int(request.form.get('count', '1440'))
    if not symbol: flash('Symbol is required.'); return redirect(url_for('dashboard.dashboard'))
    try:
        path = fetch_deriv_candles(app_id, symbol, granularity_sec=gran, count=count)
        flash(f'Pulled candles for {symbol}. Saved {path}.')
    except Exception as e:
        log('ERROR', f'Deriv fetch failed: {e}')
        flash(f'Deriv fetch failed: {e}')
    return redirect(url_for('dashboard.dashboard'))

# ---- Tally compute/broadcast (manual trigger from UI) ----
from utils import compute_tally, send_telegram_message

@bp.route('/send_tally', methods=['POST'])
@require_login
def send_tally():
    kind = (request.form.get('kind') or 'day').lower()
    if kind not in ('day','week'): kind = 'day'
    t = compute_tally(kind)
    if request.form.get('broadcast'):
        send_telegram_message(t["text"])
        flash(f"{kind.title()} tally computed and broadcast to Telegram.")
    else:
        flash(f"{kind.title()} tally computed (not broadcast).")
    session['last_tally'] = t
    return redirect(url_for('dashboard.dashboard'))
