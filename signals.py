import os, time, csv, json
from pathlib import Path
from datetime import datetime, timezone, timedelta

import numpy as np
import pandas as pd
import yfinance as yf
import yaml
import requests
import urllib.parse


# ===== Config (env-overridable) =====
INTERVAL_DEFAULT = os.getenv("INTERVAL", "5m")
LOOKBACK_DEFAULT = os.getenv("LOOKBACK", "15d")
EXPIRY_MIN = int(os.getenv("EXPIRY_MIN", "10"))
RSI_BUY    = int(os.getenv("RSI_BUY", "30"))
RSI_SELL   = int(os.getenv("RSI_SELL", "70"))
MIN_SCORE  = int(os.getenv("MIN_SCORE", "2"))
MUST_TRADE = int(os.getenv("MUST_TRADE", "0"))  # default off for strict regular-hours
SUPPRESS_EMPTY = int(os.getenv("SUPPRESS_EMPTY", "1"))  # don't post if nothing to trade

# Regular-hours windows (UTC) per group — tweak via env if you like
FX_HOURS_UTC        = os.getenv("FX_HOURS_UTC",        "0000-2100")  # from PO schedule example (EUR/USD 02:00–22:45 UTC+2) -> ~00:00–20:45 UTC
COMMODITY_HOURS_UTC = os.getenv("COMMODITY_HOURS_UTC", "0000-2100")
INDEX_HOURS_UTC     = os.getenv("INDEX_HOURS_UTC",     "1330-2000")
STOCK_HOURS_UTC     = os.getenv("STOCK_HOURS_UTC",     "1330-2000")

# Which weekdays to allow (1=Mon ... 7=Sun). OTC-free default: weekdays only.
WEEKDAYS_FX         = os.getenv("WEEKDAYS_FX",         "12345")
WEEKDAYS_COMMODITY  = os.getenv("WEEKDAYS_COMMODITY",  "12345")
WEEKDAYS_INDEX      = os.getenv("WEEKDAYS_INDEX",      "12345")
WEEKDAYS_STOCK      = os.getenv("WEEKDAYS_STOCK",      "12345")

DATA_DIR    = Path("data"); DATA_DIR.mkdir(parents=True, exist_ok=True)
SIGNALS_CSV = DATA_DIR / "signals.csv"
SYMBOLS_YML = Path("symbols.yaml")

BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
CHAT_ID   = os.getenv("TELEGRAM_CHAT_ID")

COUNTDOWN_BASE = os.getenv("COUNTDOWN_BASE")  # e.g. https://Fanatic1989.github.io/pocket-option-signals

def build_countdown_url(pair: str, expiry_dt, start_dt, bar_label: str):
    if not COUNTDOWN_BASE:
        return None
    try:
        exp = int(expiry_dt.timestamp()); start = int(start_dt.timestamp())
    except Exception:
        return None
    q = {
        "pair": pair, "expiry": str(exp), "start": str(start)
    }
    if bar_label: q["bar"] = bar_label
    return f"{COUNTDOWN_BASE.rstrip('/')}/countdown.html?"+urllib.parse.urlencode(q)

# ---- load adaptive tuning if present ----
try:
    import json
    _tj = json.load(open("tuning.json"))
    RSI_BUY  = int(_tj.get("RSI_BUY", RSI_BUY))
    RSI_SELL = int(_tj.get("RSI_SELL", RSI_SELL))
    MIN_SCORE= int(_tj.get("MIN_SCORE", MIN_SCORE))
except Exception:
    pass
# ---- load adaptive tuning if present ----
try:
    import json
    _tj = json.load(open("tuning.json"))
    RSI_BUY  = int(_tj.get("RSI_BUY", RSI_BUY))
    RSI_SELL = int(_tj.get("RSI_SELL", RSI_SELL))
    MIN_SCORE= int(_tj.get("MIN_SCORE", MIN_SCORE))
except Exception:
    pass


UA = {"User-Agent":"Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome Safari"}

# ===== Crypto multi-provider (robust) =====
BINANCE_MAP  = {"BTC-USD":"BTCUSDT", "ETH-USD":"ETHUSDT"}
COINBASE_MAP = {"BTC-USD":"BTC-USD",  "ETH-USD":"ETH-USD"}
KRAKEN_MAP   = {"BTC-USD":"XBTUSD",   "ETH-USD":"ETHUSD"}

def ema(s: pd.Series, length:int) -> pd.Series: return s.ewm(span=length, adjust=False).mean()
def rsi(s: pd.Series, length:int=14) -> pd.Series:
    d = s.diff(); up=d.clip(lower=0); dn=-d.clip(upper=0)
    ag=up.ewm(alpha=1/length, adjust=False).mean()
    al=dn.ewm(alpha=1/length, adjust=False).mean().replace(0,np.nan)
    return (100 - 100/(1+(ag/al))).fillna(50)
def macd(s: pd.Series, fast=12, slow=26, signal=9):
    m=ema(s,fast)-ema(s,slow); sig=ema(m,signal); return m, sig, m-sig

def _hhmm(now: datetime) -> int: return now.hour*100 + now.minute
def _in_window(now: datetime, hhmm_range: str, weekdays: str) -> bool:
    try: start,end = map(int, hhmm_range.split("-"))
    except: start,end = (0,2359)
    return (str(now.isoweekday()) in set(weekdays)) and (start <= _hhmm(now) <= end)

def group_in_session(now: datetime, group: str) -> bool:
    g = group.upper()
    if g == "CRYPTO":  # crypto 24/7 unless you gate it by editing here or adding an env
        return True
    if g == "FX":
        return _in_window(now, FX_HOURS_UTC, WEEKDAYS_FX)
    if g == "COMMODITY":
        return _in_window(now, COMMODITY_HOURS_UTC, WEEKDAYS_COMMODITY)
    if g in {"INDEX","INDICES"}:
        return _in_window(now, INDEX_HOURS_UTC, WEEKDAYS_INDEX)
    if g == "STOCK" or g == "STOCKS":
        return _in_window(now, STOCK_HOURS_UTC, WEEKDAYS_STOCK)
    # default: treat as FX rules
    return _in_window(now, FX_HOURS_UTC, WEEKDAYS_FX)

# ---------- data sources ----------
def df_from_klines(rows):
    out=[]; 
    for k in rows:
        ts=pd.to_datetime(k[0], unit="ms", utc=True)
        out.append({"datetime":ts,"open":float(k[1]),"high":float(k[2]),"low":float(k[3]),"close":float(k[4]),"volume":float(k[5])})
    return pd.DataFrame(out).set_index("datetime")

def fetch_binance(host, symbol, interval, limit=1000):
    url=f"https://{host}/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}"
    r=requests.get(url, timeout=15, headers=UA)
    if r.status_code!=200: return None
    arr=r.json()
    if not isinstance(arr,list) or not arr: return None
    return df_from_klines(arr)

def fetch_coinbase(pid, gran=300, limit=300):
    url=f"https://api.exchange.coinbase.com/products/{pid}/candles?granularity={gran}&limit={limit}"
    r=requests.get(url, timeout=15, headers=UA)
    if r.status_code!=200: return None
    arr=r.json()
    if not isinstance(arr,list) or not arr: return None
    rows=[]
    for t,low,high,open_,close,vol in arr:
        ts=pd.to_datetime(t, unit="s", utc=True)
        rows.append({"datetime":ts,"open":open_,"high":high,"low":low,"close":close,"volume":vol})
    return pd.DataFrame(rows).set_index("datetime").sort_index()

def fetch_kraken(pair, interval=5):
    url=f"https://api.kraken.com/0/public/OHLC?pair={pair}&interval={interval}"
    r=requests.get(url, timeout=15, headers=UA)
    if r.status_code!=200: return None
    js=r.json(); 
    if "result" not in js: return None
    key=next((k for k in js["result"].keys() if k!="last"), None)
    if not key: return None
    rows=[]
    for t,o,h,l,c,vwap,vol,cnt in js["result"][key]:
        ts=pd.to_datetime(int(t), unit="s", utc=True)
        rows.append({"datetime":ts,"open":float(o),"high":float(h),"low":float(l),"close":float(c),"volume":float(vol)})
    return pd.DataFrame(rows).set_index("datetime").sort_index()

def fetch_crypto(yf_symbol: str, interval: str) -> pd.DataFrame | None:
    if interval not in {"1m","5m"}: interval="5m"
    bi_int={"1m":"1m","5m":"5m"}[interval]
    # Binance → Coinbase → Kraken
    if yf_symbol in BINANCE_MAP:
        sym=BINANCE_MAP[yf_symbol]
        for host in ("api.binance.com","data.binance.com","api1.binance.com"):
            try:
                df=fetch_binance(host, sym, bi_int, 1000)
                if df is not None and len(df)>0: return df
            except: pass
    if yf_symbol in COINBASE_MAP:
        pid=COINBASE_MAP[yf_symbol]; gran=60 if interval=="1m" else 300
        try:
            df=fetch_coinbase(pid, gran, 300)
            if df is not None and len(df)>0: return df
        except: pass
    if yf_symbol in KRAKEN_MAP:
        pair=KRAKEN_MAP[yf_symbol]; iv=1 if interval=="1m" else 5
        try:
            df=fetch_kraken(pair, iv)
            if df is not None and len(df)>0: return df
        except: pass
    return None

def fetch_yf(yf_symbol: str, interval: str, period: str) -> pd.DataFrame | None:
    df=yf.download(yf_symbol, interval=interval, period=period, progress=False, auto_adjust=False, threads=False)
    if df is None or df.empty: return None
    return df.dropna().rename(columns=str.lower)

def robust_fetch(yf_symbol: str, group: str) -> tuple[pd.DataFrame|None, str]:
    tried=[]
    if group.upper()=="CRYPTO":
        for interval in ("5m","1m"):
            try:
                df=fetch_crypto(yf_symbol, interval)
                if df is None or df.empty: raise ValueError("empty")
                if len(df) < (210 if interval=="1m" else 220): raise ValueError(f"short {len(df)}")
                df.attrs["interval"]=interval; df.attrs["source"]="CRYPTO_MULTI"
                return df, f"{interval}/CRYPTO_MULTI"
            except Exception as e:
                tried.append(f"{interval}/CRYPTO_MULTI: {e}")
        return None, "; ".join(tried)
    else:
        for (interval, period) in (("5m","15d"),):
            try:
                df=fetch_yf(yf_symbol, interval, period)
                if df is None or df.empty: raise ValueError("empty")
                if len(df) < 220: raise ValueError(f"short {len(df)}")
                df.attrs["interval"]=interval; df.attrs["source"]="YF"
                return df, f"{interval}/{period}"
            except Exception as e:
                tried.append(f"{interval}/{period}: {e}")
        return None, "; ".join(tried)

# ---------- indicators & signal ----------
def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    c=df["close"]
    df["ema50"]=ema(c,50); df["ema200"]=ema(c,200)
    m,sg,_=macd(c,12,26,9)
    df["macd"]=m; df["macds"]=sg
    df["rsi14"]=rsi(c,14)
    return df.dropna()

def confluence(prev, cur):
    score=0; why=[]
    up  = cur["ema50"]>cur["ema200"]; why.append("EMA50>EMA200" if up else "EMA50<EMA200")
    upX = (prev["macd"]<=prev["macds"]) and (cur["macd"]>cur["macds"])
    dnX = (prev["macd"]>=prev["macds"]) and (cur["macd"]<cur["macds"])
    if upX: why.append("MACD↑"); score+=1
    if dnX: why.append("MACD↓"); score+=1
    if cur["rsi14"]<=RSI_BUY:  why.append(f"RSI≤{RSI_BUY}");  score+=1
    if cur["rsi14"]>=RSI_SELL: why.append(f"RSI≥{RSI_SELL}"); score+=1
    return score, ", ".join(why), up, upX, dnX

def classify(prev, cur):
    score, why, up, upX, dnX = confluence(prev, cur)
    sig=None
    if up and upX: sig="BUY"
    if (not up) and dnX: sig="SELL"
    if sig is None:
        if cur["rsi14"]<=RSI_BUY: sig="BUY"
        elif cur["rsi14"]>=RSI_SELL: sig="SELL"
    return sig, score, why

def send_telegram(text: str):
    if not BOT_TOKEN or not CHAT_ID: return
    url=f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    r=requests.post(url, data={"chat_id":CHAT_ID,"text":text,"parse_mode":"Markdown"}, timeout=30)
    r.raise_for_status()

def append_signal(row: dict):
    header=["ts_utc","symbol_yf","symbol_pretty","signal","price","expiry_min","evaluate_at_utc","status","score","why"]
    exists=SIGNALS_CSV.exists()
    with open(SIGNALS_CSV,"a",newline="") as f:
        w=csv.DictWriter(f, fieldnames=header)
        if not exists: w.writeheader()
        w.writerow(row)

# ---------- main ----------
def main():
    t0=time.time()
    now=datetime.now(timezone.utc)

    with open(SYMBOLS_YML,"r") as f:
        cfg=yaml.safe_load(f)
    symbols=[(s["yf"], s["name"], str(s.get("group","FX")).upper()) for s in cfg["symbols"] if s.get("enabled", True)]

    posted=0
    lines=[]  # keep for debug if you want

    for yf_sym, pretty, group in symbols:
        if not group_in_session(now, group):
            continue  # silently skip outside regular hours

        df, debug=robust_fetch(yf_sym, group)
        if df is None:
            continue  # silently skip if no data

        df=add_indicators(df)
        if len(df)<2:
            continue

        prev, cur=df.iloc[-2], df.iloc[-1]
        px=float(cur["close"])
        sig, score, why=classify(prev, cur)

        if sig and score>=MIN_SCORE:
            arrow="🟢 BUY" if sig=="BUY" else "🔴 SELL"
            msg=f"📡 *Pocket Option Signal* — {now:%Y-%m-%d %H:%M UTC}\n{arrow} *{pretty}* @ `{px:.5f}`\n• {why} (score {score})\nExpiry: {EXPIRY_MIN}m"
            send_telegram(msg)
            evaluate_at=now+timedelta(minutes=EXPIRY_MIN)
            append_signal({
                "ts_utc": now.strftime("%Y-%m-%d %H:%M:%S"),
                "symbol_yf": yf_sym,
                "symbol_pretty": pretty,
                "signal": sig,
                "price": f"{px:.8f}",
                "expiry_min": EXPIRY_MIN,
                "evaluate_at_utc": evaluate_at.strftime("%Y-%m-%d %H:%M:%S"),
                "status": "open",
                "score": score,
                "why": why
            })
            posted+=1

    # If nothing posted:
    if posted==0 and not SUPPRESS_EMPTY:
        send_telegram(f"📡 *Pocket Option Signals* — {now:%Y-%m-%d %H:%M UTC}\n(No valid setups this run)")
    # else: stay silent

if __name__ == "__main__":
    main()
