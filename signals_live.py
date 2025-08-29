import os, time, csv, json, requests
from pathlib import Path
from datetime import datetime, timezone, timedelta

import pandas as pd
import yfinance as yf
import yaml

from scalper import add_indicators, classify
from telegram_send import send_to_tiers

# ===== Env / Config =====
INTERVAL   = os.getenv("INTERVAL", "1m")   # 1m scalping
EXPIRY_MIN = int(os.getenv("EXPIRY_MIN", "5"))
RSI_BUY    = int(os.getenv("RSI_BUY", "50"))
RSI_SELL   = int(os.getenv("RSI_SELL", "50"))
MIN_SCORE  = int(os.getenv("MIN_SCORE", "1"))
MUST_TRADE = int(os.getenv("MUST_TRADE", "0"))
SUPPRESS_EMPTY = int(os.getenv("SUPPRESS_EMPTY", "1"))

DATA_DIR    = Path("data"); DATA_DIR.mkdir(parents=True, exist_ok=True)
SIGNALS_CSV = DATA_DIR / "signals.csv"
SYMBOLS_YML = Path("symbols.yaml")

OANDA_API_KEY = os.getenv("OANDA_API_KEY","")
OANDA_ENV     = os.getenv("OANDA_ENV","practice").lower()
OANDA_HOST    = "https://api-fxpractice.oanda.com" if OANDA_ENV!="live" else "https://api-fxtrade.oanda.com"

def load_symbols():
    cfg = yaml.safe_load(SYMBOLS_YML.read_text())
    return [(s["yf"], s.get("name", s["yf"]), s.get("group","FX")) for s in cfg.get("symbols",[]) if s.get("enabled",True)]

def yf_fetch(yf_symbol: str, interval: str, bars: int = 200) -> pd.DataFrame:
    # yfinance: get ~bars worth; period must match interval
    period = "7d" if interval=="1m" else "15d"
    df = yf.download(yf_symbol, interval=interval, period=period, progress=False)
    df = df.dropna()
    df = df.rename(columns=str.lower)
    return df.tail(bars)

def oanda_fetch_m1(instrument: str, count: int = 200) -> pd.DataFrame:
    if not OANDA_API_KEY:
        raise RuntimeError("OANDA_API_KEY not set")
    url = f"{OANDA_HOST}/v3/instruments/{instrument}/candles"
    r = requests.get(url, params={"granularity":"M1","count":count,"price":"M"},
                     headers={"Authorization":f"Bearer {OANDA_API_KEY}"}, timeout=20)
    r.raise_for_status()
    js = r.json()
    recs = []
    for c in js.get("candles",[]):
        if not c.get("complete"): continue
        ts = pd.to_datetime(c["time"], utc=True)
        o = float(c["mid"]["o"]); h = float(c["mid"]["h"]); l = float(c["mid"]["l"]); cl = float(c["mid"]["c"])
        recs.append((ts,o,h,l,cl))
    if not recs: return pd.DataFrame()
    df = pd.DataFrame(recs, columns=["datetime","open","high","low","close"]).set_index("datetime")
    return df

def map_to_oanda(yf_symbol: str) -> str | None:
    # EURUSD=X -> EUR_USD ; XAUUSD=X -> XAU_USD ; others None
    if yf_symbol.endswith("=X"):
        base = yf_symbol.replace("=X","")
        if len(base) in (6,7):  # XAUUSD is 6, sometimes others
            if base[:3] and base[3:]:
                return base[:3] + "_" + base[3:]
    return None

def append_signal(row: dict):
    header = ["ts_utc","symbol_yf","symbol_pretty","signal","price","expiry_min","evaluate_at_utc","status","score","why"]
    exists = SIGNALS_CSV.exists()
    with open(SIGNALS_CSV, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        if not exists: w.writeheader()
        w.writerow(row)

def main():
    t0 = time.time()
    now = datetime.now(timezone.utc)

    lines = [f"📡 Pocket Option Signals — {now:%Y-%m-%d %H:%M UTC}",
             f"Candle: {INTERVAL} | Expiry: {EXPIRY_MIN}m",
             ""]
    picked = 0

    for yf_sym, pretty, group in load_symbols():
        try:
            o_inst = map_to_oanda(yf_sym)
            if o_inst:  # FX / Metals via OANDA
                df = oanda_fetch_m1(o_inst, count=220)
            else:       # Crypto / indices via Yahoo
                df = yf_fetch(yf_sym, INTERVAL, bars=220)

            if df.empty or len(df) < 60:
                lines.append(f"⚪ {pretty} — no/short data")
                continue

            df = add_indicators(df)
            prev, cur = df.iloc[-2], df.iloc[-1]
            side, score, why = classify(prev, cur, rsi_buy=RSI_BUY, rsi_sell=RSI_SELL, min_score=MIN_SCORE)
            px = float(cur["close"])

            if side and score >= MIN_SCORE:
                emoji = "🟢" if side=="BUY" else "🔴"
                lines.append(f"{emoji} {pretty} — {side} @ {px:.5f}")
                lines.append(f"• {why} (score {score})")
                evaluate_at = now + timedelta(minutes=EXPIRY_MIN)
                append_signal({
                    "ts_utc": now.strftime("%Y-%m-%d %H:%M:%S"),
                    "symbol_yf": yf_sym,
                    "symbol_pretty": pretty,
                    "signal": side,
                    "price": f"{px:.8f}",
                    "expiry_min": EXPIRY_MIN,
                    "evaluate_at_utc": evaluate_at.strftime("%Y-%m-%d %H:%M:%S"),
                    "status": "open",
                    "score": score,
                    "why": why
                })
                picked += 1
            else:
                lines.append(f"⚪ {pretty} — no setup")
        except Exception as e:
            lines.append(f"⚠️ {pretty} — error: {e}")

    if picked == 0 and SUPPRESS_EMPTY:
        return None  # silent run if nothing good

    lines.append("")
    lines.append(f"⏱ Build time: `{time.time()-t0:.1f}s`")
    return lines

if __name__ == "__main__":
    out = main()
    if out:
        send_to_tiers("\n\n".join(out))
        print("✅ Sent to all tiers.")
    else:
        print("ℹ️ No signals to send (SUPPRESS_EMPTY=1).")
