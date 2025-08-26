import os, sys, json, requests, pandas as pd
from datetime import datetime, timezone, timedelta
import pandas_ta as ta

# ---- Config via ENV ----
OANDA_API_KEY = os.environ.get("OANDA_API_KEY")
OANDA_ENV     = os.environ.get("OANDA_ENV", "practice")           # practice|live
INTERVAL      = os.environ.get("INTERVAL", "5m")                  # 1m/5m/15m/30m/60m
EXPIRY_MIN    = int(os.environ.get("EXPIRY_MIN", "10"))
MIN_SCORE     = int(os.environ.get("MIN_SCORE", "2"))             # 1..3
RSI_BUY       = int(os.environ.get("RSI_BUY", "30"))              # e.g. 40 for more signals
RSI_SELL      = int(os.environ.get("RSI_SELL", "70"))             # e.g. 60 for more signals
MUST_TRADE    = os.environ.get("MUST_TRADE", "0") == "1"          # always output 1 pick
USE_TREND     = os.environ.get("USE_TREND", "1") == "1"           # include EMA trend in score

# Telegram (optional)
TG_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
TG_CHAT  = os.environ.get("TELEGRAM_CHAT_ID")  # @channelusername or numeric id

# Symbols (you can change here)
from pathlib import Path
DATA_DIR=Path('data'); DATA_DIR.mkdir(exist_ok=True)
SIGNALS_CSV=DATA_DIR/'signals.csv'

SYMBOLS = [
    ("EURUSD=X", "EUR/USD"),
    ("GBPUSD=X", "GBP/USD"),
    ("USDJPY=X", "USD/JPY"),
    ("XAUUSD=X", "Gold"),
    # add crypto only if your OANDA practice exposes it:
    # ("BTCUSD=X", "BTC/USD"),
]

# ---- OANDA helpers ----
def oanda_symbol(yf_sym: str):
    t = yf_sym.upper()
    if t.endswith("=X") and len(t[:-2]) == 6:
        return t[:3] + "_" + t[3:6]
    if t in ("XAUUSD=X", "XAGUSD=X"):
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

# ---- Indicators & signal ----
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

    # scoring: trend contributes 1 if enabled
    if USE_TREND:
        score += 1 if uptrend else 1  # trend counts toward either direction; we decide side below

    # MACD cross adds 1
    if macd_up or macd_down:
        score += 1

    # RSI extreme adds 1 if outside thresholds
    rsi = cur["rsi14"]
    rsi_bull = rsi <= RSI_BUY
    rsi_bear = rsi >= RSI_SELL
    if rsi_bull or rsi_bear:
        score += 1

    # choose side with hierarchy
    side = None
    if uptrend and macd_up:
        side = "BUY"
    elif (not uptrend) and macd_down:
        side = "SELL"
    else:
        # fallback on RSI
        if rsi_bull: side = "BUY"
        elif rsi_bear: side = "SELL"

    return side, score, {"uptrend":uptrend, "macd_up":macd_up, "macd_down":macd_down, "rsi":float(rsi)}

def send_telegram(text: str):
    if not TG_TOKEN or not TG_CHAT: 
        return
    url = f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage"
    r = requests.post(url, data={"chat_id": TG_CHAT, "text": text, "parse_mode":"Markdown"}, timeout=20)
    try: r.raise_for_status()
    except Exception as e:
        print("Telegram error:", e, file=sys.stderr)

def main():
    now = datetime.now(timezone.utc)
    header = f"📡 OANDA bot — {now:%Y-%m-%d %H:%M UTC}  (tf {INTERVAL}, expiry {EXPIRY_MIN}m, RSI≤{RSI_BUY}/≥{RSI_SELL}, MIN_SCORE={MIN_SCORE})"
    print(header)

    lines=[]
    best = None  # (score, abs_rsi_dist, text)
    for yf_sym, name in SYMBOLS:
        try:
            df  = add_indicators(fetch_oanda_df(yf_sym, INTERVAL))
            prev, cur = df.iloc[-2], df.iloc[-1]
            px = float(cur["close"])
            side, score, info = score_and_signal(prev, cur)

            if side and score >= MIN_SCORE:
                evaluate_at = now + timedelta(minutes=EXPIRY_MIN)
                line = f"✅ {side} *{name}* @ `{px:.5f}`  (score {score})  expires {evaluate_at:%H:%M UTC}"
                print(line); lines.append(line)
            else:
                rsi = info["rsi"]; dist = min(abs(rsi - RSI_BUY), abs(rsi - RSI_SELL))
                # track best fallback candidate
                cand = (score, -dist, f"🤖 Fallback {('BUY' if rsi < 50 else 'SELL')} *{name}* @ `{px:.5f}` (score {score}) [RSI {rsi:.1f}]")
                if (best is None) or (cand > best):
                    best = cand
                line = f"⚪ {name} — no setup (score {score}) [RSI {rsi:.1f}]"
                print(line); lines.append(line)

        except Exception as e:
            msg = f"⚠️ {name} error: {e}"
            print(msg); lines.append(msg)

    # MUST_TRADE fallback: always send 1 pick if nothing qualified
    if MUST_TRADE and not any(l.startswith("✅") for l in lines) and best:
        print(best[2]); lines.append(best[2])

    # Optional Telegram mirror
    if lines:
        send_telegram(header + "\n" + "\n".join(lines))

if __name__ == "__main__":
    main()


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
