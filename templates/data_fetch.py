import os, json
import pandas as pd
from websocket import WebSocketApp

def deriv_csv_path(symbol: str, granularity: int) -> str:
    """
    Returns the canonical CSV path for a Deriv symbol+granularity.
    Example: /tmp/deriv_frxEURUSD_300.csv
    """
    base = os.getenv('DERIV_DIR','/tmp')
    os.makedirs(base, exist_ok=True)
    safe_sym = str(symbol).replace("/", "_")
    return os.path.join(base, f"deriv_{safe_sym}_{int(granularity)}.csv")

def fetch_one_symbol(app_id: str, symbol: str, granularity_sec: int, count: int) -> str:
    """
    Pulls historical candles via Deriv WS and saves to CSV at deriv_csv_path(...).
    Raises RuntimeError on failure/timeouts. Returns the saved path on success.
    """
    save_path = deriv_csv_path(symbol, granularity_sec)
    result = {'done': False, 'error': None, 'saved': False}

    def on_message(ws, message):
        try:
            data = json.loads(message)
            if 'error' in data:
                result['error'] = data['error'].get('message','Deriv API error')
                result['done'] = True; ws.close(); return

            candles = None
            if 'candles' in data and isinstance(data['candles'], list):
                candles = data['candles']
            elif 'history' in data and isinstance(data['history'], dict) and 'candles' in data['history']:
                candles = data['history']['candles']

            if candles is None:
                return

            df = pd.DataFrame(candles)
            if df.empty:
                result['error'] = 'No candles returned'
            else:
                if 'epoch' not in df.columns:
                    result['error'] = 'Response missing epoch'
                else:
                    df.rename(columns={'epoch':'timestamp'}, inplace=True)
                    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
                    for c in ('open','high','low','close'):
                        if c in df.columns:
                            df[c] = pd.to_numeric(df[c], errors='coerce')
                    df = df[['timestamp','open','high','low','close']].sort_values('timestamp')
                    root = os.getenv('DERIV_DIR','/tmp'); os.makedirs(root, exist_ok=True)
                    df.to_csv(save_path, index=False)
                    result['saved'] = True
            result['done'] = True; ws.close()
        except Exception as e:
            result['error'] = f'Parse error: {e}'
            result['done'] = True; ws.close()

    def on_open(ws):
        req = {"ticks_history": str(symbol), "count": int(count), "end": "latest",
               "granularity": int(granularity_sec), "style": "candles", "adjust_start_time": 1}
        ws.send(json.dumps(req))

    url = f"wss://ws.derivws.com/websockets/v3?app_id={app_id or '1089'}"
    ws = WebSocketApp(url, on_open=on_open, on_message=on_message)

    import threading, time
    th = threading.Thread(target=ws.run_forever, daemon=True); th.start()
    for _ in range(300):
        if result['done']: break
        time.sleep(0.1)

    if not result['done']:
        raise RuntimeError(f'{symbol}: timeout (no response)')
    if result['error']:
        raise RuntimeError(f'{symbol}: {result["error"]}')
    if not result['saved']:
        raise RuntimeError(f'{symbol}: nothing saved')
    return save_path
