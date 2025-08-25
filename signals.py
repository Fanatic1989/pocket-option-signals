import os, time, csv, json
from pathlib import Path
from datetime import datetime, timezone, timedelta

import numpy as np
import pandas as pd
import yfinance as yf
import yaml
import requests

# ===== Config (env-overridable) =====
INTERVAL_DEFAULT = os.getenv("INTERVAL", "5m")
LOOKBACK_DEFAULT = os.getenv("LOOKBACK", "15d")
EXPIRY_MIN = int(os.getenv("EXPIRY_MIN", "10"))
RSI_BUY    = int(os.getenv("RSI_BUY", "30"))
RSI_SELL   = int(os.getenv("RSI_SELL", "70"))
MIN_SCORE  = int(os.getenv("MIN_SCORE", "2"))
MUST_TRADE = int(os.getenv("MUST_TRADE", "1"))  # post at least one pick if possible
SESSION_UTC = os.getenv("SESSION_UTC", "0700-1700")
WEEKDAYS    = os.getenv("WEEKDAYS", "12345")
GATED_GROUPS = set(os.getenv("GATED_GROUPS", "FX,COMMODITY,INDEX").split(","))

# Group-specific fetch parameters
GROUP_PARAMS = {
    "FX":        {"min_rows": 220, "candidates": [("5m","15d")]},
    "COMMODITY": {"min_rows": 220, "candidates": [("5m","15d")]},
    "INDEX":     {"min_rows": 220, "candidates": [("5m","15d")]},
    # Crypto: prefer Binance (1000 candles). yfinance as a last resort.
    "CRYPTO":    {"min_rows": 210, "candidates": [("5m","BINANCE"), ("1m","BINANCE"), ("5m","15d"), ("1m","7d")]},
}

RETRIES     = 2
RETRY_SLEEP = 1.0

DATA_DIR    = Path("data"); DATA_DIR.mkdir(parents=True, exist_ok=True)
SIGNALS_CSV = DATA_DIR / "signals.csv"
SYMBOLS_YML = Path("symbols.yaml")

BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
CHAT_ID   = os.getenv("TELEGRAM_CHAT_ID")

# ===== Binance map for crypto =====
CRYPTO_BINANCE = {
    "BTC-USD": "BTCUSDT",
    "ETH-USD": "ETHUSDT",
}

BINANCE_INTERVALS = {"1m":"1m","5m":"5m"}

# ===== Indicators =====
def ema(s: pd.Series, length:int) -> pd.Series:
    return s.ewm(span=length, adjust=False).mean()

def rsi(s: pd.Series, length:int=14) -> pd.Series:
    d = s.diff()
    up = d.clip(lower=0); dn = -d.clip(upper=0)
    avg_up = up.ewm(alpha=1/length, adjust=False).mean()
    avg_dn = dn.ewm(alpha=1/length, adjust=False).mean().replace(0,np.nan)
    rs = avg_up / avg_dn
    return (100 - 100/(1+rs)).fillna(50)

def macd(s: pd.Series, fast=12, slow=26, signal=9):
    m = ema(s, fast) - ema(s, slow)
    sig = ema(m, signal)
    return m, sig, m - sig

# ===== Session gate =====
def _parse_session(s: str):
    try:
        a,b = s.split("-"); return int(a), int(b)
    except: return 0, 2359

def in_session_utc(now: datetime, group: str) -> bool:
    if group.upper() not in GATED_GROUPS:
        return True
    if str(now.isoweekday()) not in set(WEEKDAYS):  return False
    start, end = _parse_session(SESSION_UTC)
    hhmm = now.hour*100 + now.minute
    return start <= hhmm <= end

# ===== Data sources =====
def fetch_binance(symbol_usdt: str, interval: str, limit: int = 1000) -> pd.DataFrame | None:
    # Public endpoint, no key.
    url = f"https://api.binance.com/api/v3/klines?symbol={symbol_usdt}&interval={interval}&limit={limit}"
    r = requests.get(url, timeout=15)
    if r.status_code != 200:
        return None
    arr = r.json()
    if not isinstance(arr, list) or len(arr) == 0:
        return None
    # columns: open time, open, high, low, close, volume, close time, ...
    rows = []
    for k in arr:
        ts = pd.to_datetime(k[0], unit="ms", utc=True)
        rows.append({
            "datetime": ts,
            "open": float(k[1]),
            "high": float(k[2]),
            "low":  float(k[3]),
            "close":float(k[4]),
            "volume": float(k[5]),
        })
    df = pd.DataFrame(rows).set_index("datetime")
    return df

def fetch_yf(yf_symbol: str, interval: str, period: str) -> pd.DataFrame | None:
    df = yf.download(yf_symbol, interval=interval, period=period, progress=False, auto_adjust=False, threads=False)
    if df is None or df.empty:
        return None
    df = df.dropna().rename(columns=str.lower)
    return df

def robust_fetch(yf_symbol: str, group: str) -> tuple[pd.DataFrame|None, str]:
    params = GROUP_PARAMS.get(group, {"min_rows":220, "candidates":[(INTERVAL_DEFAULT, LOOKBACK_DEFAULT)]})
    min_rows = params["min_rows"]
    tried = []
    for (interval, source) in params["candidates"]:
        for attempt in range(1, RETRIES+1):
            try:
                if source == "BINANCE" and group == "CRYPTO":
                    bsym = CRYPTO_BINANCE.get(yf_symbol)
                    if not bsym or interval not in BINANCE_INTERVALS:
                        raise ValueError("binance mapping/interval missing")
                    df = fetch_binance(bsym, BINANCE_INTERVALS[interval], limit=1000)
                else:
                    df = fetch_yf(yf_symbol, interval=interval, period=source)
                if df is None or df.empty:
                    raise ValueError("empty")
                if len(df) < min_rows:
                    raise ValueError(f"short {len(df)} < {min_rows}")
                df.attrs["interval"] = interval
                df.attrs["source"]   = source
                return df, f"{interval}/{source}"
            except Exception as e:
                tried.append(f"{interval}/{source} (try {attempt}): {e}")
                if attempt == RETRIES: break
                time.sleep(RETRY_SLEEP)
    return None, "; ".join(tried)

# ===== Core helpers =====
def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    c = df["close"]
    df["ema50"]  = ema(c,50); df["ema200"]=ema(c,200)
    m,sig,_ = macd(c,12,26,9)
    df["macd"]=m; df["macds"]=sig
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

# ===== Main =====
def main():
    t0=time.time()
    now = datetime.now(timezone.utc)
    with open(SYMBOLS_YML,"r") as f:
        cfg = yaml.safe_load(f)
    symbols=[]
    for s in cfg["symbols"]:
        if not s.get("enabled", True): continue
        symbols.append( (s["yf"], s["name"], str(s.get("group","FX")).upper()) )

    lines=[f"📡 *Pocket Option Signals* — {now:%Y-%m-%d %H:%M UTC}\nInterval: {INTERVAL_DEFAULT} | Expiry: {EXPIRY_MIN}m\nSession: {SESSION_UTC} UTC, Weekdays {WEEKDAYS}\n"]

    posted = 0
    best_choice = None   # (preference_tuple, text, rowdict)

    for yf_sym, pretty, group in symbols:
        # session gate for non-crypto groups
        if not in_session_utc(now, group):
            lines.append(f"⚪ *{pretty}* — skipped (outside session for {group})")
            continue

        df, debug = robust_fetch(yf_sym, group)
        if df is None:
            lines.append(f"⚪ *{pretty}* — skipped (no/short data; tried {debug})")
            continue

        try:
            df = add_indicators(df)
            if len(df) < 2:
                lines.append(f"⚪ *{pretty}* — skipped (insufficient after indicators)")
                continue

            prev, cur = df.iloc[-2], df.iloc[-1]
            px = float(cur["close"])
            sig, score, why = classify(prev, cur)

            if sig and score >= MIN_SCORE:
                arrow = "🟢 BUY" if sig=="BUY" else "🔴 SELL"
                msg = f"{arrow} *{pretty}* @ `{px:.5f}`\n• {why} (score {score})"
                lines.append(msg)
                evaluate_at = now + timedelta(minutes=EXPIRY_MIN)
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
                posted += 1
            else:
                lines.append(f"⚪ *{pretty}* — No setup (score {score if sig else 0})")
                if sig:
                    cand_row = {
                        "ts_utc": now.strftime("%Y-%m-%d %H:%M:%S"),
                        "symbol_yf": yf_sym,
                        "symbol_pretty": pretty,
                        "signal": sig,
                        "price": f"{px:.8f}",
                        "expiry_min": EXPIRY_MIN,
                        "evaluate_at_utc": (now+timedelta(minutes=EXPIRY_MIN)).strftime("%Y-%m-%d %H:%M:%S"),
                        "status": "open",
                        "score": score,
                        "why": why
                    }
                    cand_text = f"🟨 *MustTrade* {sig} {pretty} @ `{px:.5f}`\n• {why} (score {score})"
                    # prefer higher score, and prefer CRYPTO
                    pref = ( (group=="CRYPTO"), score )
                    if (best_choice is None) or (pref > best_choice[0]):
                        best_choice = (pref, cand_text, cand_row)
        except Exception as e:
            lines.append(f"⚠️ *{pretty}* error: {e}")

    # Must-trade fallback
    if posted == 0 and MUST_TRADE and best_choice:
        _, text, row = best_choice
        lines.append("\n🔥 No standard setups — posting best available:\n" + text)
        append_signal(row)
        posted = 1

    build_time = time.time() - t0
    lines.append(f"\n⏱ Build time: `{build_time:.1f}s`")
    send_telegram("\n\n".join(lines))

if __name__ == "__main__":
    main()
