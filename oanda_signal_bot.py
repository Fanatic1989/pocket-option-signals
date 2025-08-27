import os, sys, requests, pandas as pd
from datetime import datetime, timezone, timedelta
from pathlib import Path
import pandas_ta as ta

# ========= Config via ENV =========
OANDA_API_KEY = os.environ.get("OANDA_API_KEY")
OANDA_ENV     = os.environ.get("OANDA_ENV", "practice")  # practice|live

INTERVAL      = os.environ.get("INTERVAL", "5m")        # 1m/5m/15m/30m/60m
EXPIRY_MIN    = int(os.environ.get("EXPIRY_MIN", "10"))
MIN_SCORE     = int(os.environ.get("MIN_SCORE", "2"))   # 1..3
RSI_BUY       = int(os.environ.get("RSI_BUY", "30"))
RSI_SELL      = int(os.environ.get("RSI_SELL", "70"))
MUST_TRADE    = os.environ.get("MUST_TRADE", "0") == "1"
USE_TREND     = os.environ.get("USE_TREND", "1") == "1"

# Telegram
TG_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
TG_CHATS = {
    "FREE":  os.environ.get("TELEGRAM_CHAT_FREE"),
    "BASIC": os.environ.get("TELEGRAM_CHAT_BASIC"),
    "PRO":   os.environ.get("TELEGRAM_CHAT_PRO"),
    "VIP":   os.environ.get("TELEGRAM_CHAT_VIP"),
}

DATA_DIR = Path("data"); DATA_DIR.mkdir(exist_ok=True)
SIGNALS_CSV = DATA_DIR / "signals.csv"

# ===== Symbol maps by Tier =====
FX_ALL = [
    ("EURUSD=X","EUR/USD"),("GBPUSD=X","GBP/USD"),("USDJPY=X","USD/JPY"),("USDCHF=X","USD/CHF"),
    ("USDCAD=X","USD/CAD"),("AUDUSD=X","AUD/USD"),("NZDUSD=X","NZD/USD"),
    ("EURJPY=X","EUR/JPY"),("GBPJPY=X","GBP/JPY")
]
METALS = [("XAUUSD=X","Gold"),("XAGUSD=X","Silver")]
CRYPTO_TOP = [
    ("CRYPTO:BTCUSDT","BTC/USDT"),("CRYPTO:ETHUSDT","ETH/USDT"),("CRYPTO:SOLUSDT","SOL/USDT"),
    ("CRYPTO:DOGEUSDT","DOGE/USDT"),("CRYPTO:LTCUSDT","LTC/USDT"),("CRYPTO:XRPUSDT","XRP/USDT"),("CRYPTO:BCHUSDT","BCH/USDT")
]

TIERS = {
    "FREE":  [("EURUSD=X","EUR/USD"), ("CRYPTO:BTCUSDT","BTC/USDT")],
    "BASIC": FX_ALL[:6] + [("XAUUSD=X","Gold")] + [("CRYPTO:BTCUSDT","BTC/USDT")],
    "PRO":   FX_ALL + METALS + CRYPTO_TOP,
    "VIP":   FX_ALL + METALS + CRYPTO_TOP
}

# ========= OANDA helpers (FX/metals) =========
def oanda_symbol(yf_sym: str):
    t = yf_sym.upper()
    if t.endswith("=X") and len(t[:-2]) == 6:
        return t[:3] + "_" + t[3:6]
    if t in ("XAUUSD=X","XAGUSD=X"):
        return t[:3] + "_" + t[3:6]
    return None

def oanda_base_url():
    return "https://api-fxtrade.oanda.com" if OANDA_ENV == "live" else "https://api-fxpractice.oanda.com"

def oanda_granularity(interval: str):
    iv = "".join(ch for ch in interval if ch.isdigit())
    try: iv = int(iv)
    except: iv = 5
    return {1:"M1",5:"M5",15:"M15",30:"M30",60:"H1"}.get(iv, "M5")

def fetch_oanda_df(yf_sym: str, interval: str, count: int = 500) -> pd.DataFrame:
    if not OANDA_API_KEY:
        raise RuntimeError("OANDA_API_KEY is not set")
    inst = oanda_symbol(yf_sym)
    if not inst:
        raise RuntimeError(f"no OANDA mapping for {yf_sym}")
    url = f"{oanda_base_url()}/v3/instruments/{inst}/candles"
    headers = {"Authorization": f"Bearer {OANDA_API_KEY}"}
    params  = {"granularity": oanda_granularity(interval), "count": str(count), "price":"M"}
    r = requests.get(url, headers=headers, params=params, timeout=20)
    r.raise_for_status()
    rows=[]
    for c in r.json().get("candles", []):
        if not c.get("complete"): continue
        mid = c.get("mid") or {}
        rows.append({
            "time": c["time"],
            "open": float(mid.get("o","nan")),
            "high": float(mid.get("h","nan")),
            "low" : float(mid.get("l","nan")),
            "close":float(mid.get("c","nan")),
            "volume": int(c.get("volume",0))
        })
    df = pd.DataFrame(rows)
    if df.empty: raise RuntimeError("oanda returned empty df")
    df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
    df = df.dropna(subset=["time"]).set_index("time").sort_index()
    return df[["open","high","low","close","volume"]].dropna()

# ========= Binance helpers (crypto) =========
def binance_granularity(interval: str):
    iv = "".join(ch for ch in interval if ch.isdigit())
    try: iv = int(iv)
    except: iv = 1
    return {1:"1m",5:"5m",15:"15m",30:"30m",60:"1h"}.get(iv, "5m")

def is_crypto_sym(yf_sym: str) -> bool:
    return str(yf_sym).upper().startswith("CRYPTO:")

def binance_symbol_from_crypto(yf_sym: str) -> str:
    return "".join(ch for ch in yf_sym.split(":",1)[1].upper() if ch.isalnum())

def fetch_binance_df(yf_sym: str, interval: str, limit: int = 500) -> pd.DataFrame:
    s = binance_symbol_from_crypto(yf_sym)
    url = "https://api.binance.com/api/v3/klines"
    params = {"symbol": s, "interval": binance_granularity(interval), "limit": str(limit)}
    r = requests.get(url, params=params, timeout=20)
    r.raise_for_status()
    rows=[]
    for k in r.json():
        rows.append({
            "time": pd.to_datetime(int(k[6]), unit="ms", utc=True), # closeTime
            "open": float(k[1]),
            "high": float(k[2]),
            "low" : float(k[3]),
            "close":float(k[4]),
            "volume": float(k[5]),
        })
    df = pd.DataFrame(rows)
    if df.empty: raise RuntimeError(f"binance returned empty for {s}")
    df = df.dropna(subset=["time"]).set_index("time").sort_index()
    return df[["open","high","low","close","volume"]].dropna()

# ========= Router =========
def fetch_df_router(yf_sym: str, interval: str) -> pd.DataFrame:
    if is_crypto_sym(yf_sym):
        return fetch_binance_df(yf_sym, interval)
    return fetch_oanda_df(yf_sym, interval)

# ========= Indicators & Scoring =========
def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df["ema50"]  = ta.ema(df["close"], length=50)
    df["ema200"] = ta.ema(df["close"], length=200)
    macd = ta.macd(df["close"], fast=12, slow=26, signal=9)
    df["macd"]  = macd["MACD_12_26_9"]
    df["macds"] = macd["MACDs_12_26_9"]
    df["rsi14"] = ta.rsi(df["close"], length=14)
    return df.dropna()

def score_and_signal(prev, cur):
    score = 0
    uptrend   = cur["ema50"] > cur["ema200"]
    macd_up   = (prev["macd"] <= prev["macds"]) and (cur["macd"] > cur["macds"])
    macd_down = (prev["macd"] >= prev["macds"]) and (cur["macd"] < cur["macds"])

    if USE_TREND:
        score += 1
    if macd_up or macd_down:
        score += 1

    rsi = cur["rsi14"]
    rsi_bull = rsi <= RSI_BUY
    rsi_bear = rsi >= RSI_SELL
    if rsi_bull or rsi_bear:
        score += 1

    side = None
    if uptrend and macd_up:
        side = "BUY"
    elif (not uptrend) and macd_down:
        side = "SELL"
    else:
        if rsi_bull: side = "BUY"
        elif rsi_bear: side = "SELL"

    return side, score, {"rsi": float(rsi)}

# ========= Logging & Telegram =========
def send_telegram(text: str, chat_id: str):
    if not TG_TOKEN or not chat_id:
        return
    try:
        url = f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage"
        requests.post(url, data={"chat_id": chat_id, "text": text, "parse_mode":"Markdown"}, timeout=20).raise_for_status()
    except Exception as e:
        print("Telegram error:", e, file=sys.stderr)

def log_row(ts, name, side, score, price, interval, expiry_min, expires_utc, rsi):
    import csv
    hdr = ["ts_utc","pair","side","score","price","interval","expiry_min","expires_utc","rsi"]
    exists = SIGNALS_CSV.exists()
    with open(SIGNALS_CSV, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=hdr)
        if not exists: w.writeheader()
        w.writerow({
            "ts_utc": ts, "pair": name, "side": side or "", "score": score,
            "price": f"{price:.8f}" if price is not None else "",
            "interval": interval, "expiry_min": expiry_min,
            "expires_utc": expires_utc or "", "rsi": f"{rsi:.2f}" if rsi is not None else ""
        })

# ========= Main per-tier =========
def run_for_tier(tier_name: str, pairs: list, chat_id: str):
    if not chat_id:
        return

    now = datetime.now(timezone.utc)
    header = f"📡 {tier_name} — {now:%Y-%m-%d %H:%M UTC}  (tf {INTERVAL}, expiry {EXPIRY_MIN}m, RSI≤{RSI_BUY}/≥{RSI_SELL}, MIN_SCORE={MIN_SCORE})"
    print(header)
    lines=[]
    ts = now.strftime("%Y-%m-%d %H:%M:%S")
    best = None

    for yf_sym, name in pairs:
        try:
            df  = add_indicators(fetch_df_router(yf_sym, INTERVAL))
            prev, cur = df.iloc[-2], df.iloc[-1]
            px = float(cur["close"])
            side, score, info = score_and_signal(prev, cur)
            rsi = info["rsi"]

            if side and score >= MIN_SCORE:
                evaluate_at = now + timedelta(minutes=EXPIRY_MIN)
                line = f"✅ {side} *{name}* @ `{px:.5f}`  (score {score})  expires {evaluate_at:%H:%M UTC}"
                print(line); lines.append(line)
                log_row(ts, name, side, score, px, INTERVAL, EXPIRY_MIN, evaluate_at.strftime("%Y-%m-%d %H:%M:%S"), rsi)
            else:
                dist = min(abs(rsi - RSI_BUY), abs(rsi - RSI_SELL))
                rtext = f"🤖 Fallback {('BUY' if rsi < 50 else 'SELL')} *{name}* @ `{px:.5f}` (score {score}) [RSI {rsi:.1f}]"
                cand = (score, -dist, rtext, name, px, rsi)
                if (best is None) or (cand > best): best = cand

                line = f"⚪ {name} — no setup (score {score}) [RSI {rsi:.1f}]"
                print(line); lines.append(line)
                log_row(ts, name, None, score, px, INTERVAL, EXPIRY_MIN, None, rsi)

        except Exception as e:
            msg = f"⚠️ {name} error: {e}"
            print(msg); lines.append(msg)
            log_row(ts, name, None, -1, None, INTERVAL, EXPIRY_MIN, None, None)

    if MUST_TRADE and not any(l.startswith("✅") for l in lines) and best:
        _, _, text, name, px, rsi = best
        print(text); lines.append(text)
        side = "BUY" if "Fallback BUY" in text else "SELL"
        log_row(ts, name, side, 0, px, INTERVAL, EXPIRY_MIN, None, rsi)

    if lines:
        send_telegram(header + "\n" + "\n".join(lines), chat_id)

def main():
    # Run once per available tier chat
    for tier, chat in TG_CHATS.items():
        if chat:
            pairs = TIERS.get(tier, [])
            run_for_tier(tier, pairs, chat)

if __name__ == "__main__":
    main()
