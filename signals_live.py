import os, csv, time, requests
from datetime import datetime, timezone, timedelta
from pathlib import Path

# -------- Config --------
INTERVAL   = os.getenv("INTERVAL", "5m")        # 1m / 5m
EXPIRY_MIN = int(os.getenv("EXPIRY_MIN", "10")) # option expiry
RSI_BUY    = int(os.getenv("RSI_BUY", "30"))
RSI_SELL   = int(os.getenv("RSI_SELL", "70"))
MIN_SCORE  = int(os.getenv("MIN_SCORE", "2"))
MUST_TRADE = int(os.getenv("MUST_TRADE", "0"))
SUPPRESS_EMPTY = int(os.getenv("SUPPRESS_EMPTY", "1"))

DATA   = Path("data"); DATA.mkdir(exist_ok=True)
LOG    = DATA/"signals.csv"
SYMS   = Path("symbols.yaml")

OANDA_API_KEY = os.getenv("OANDA_API_KEY","")
OANDA_ENV     = os.getenv("OANDA_ENV","practice")  # practice|live
OANDA_HOST    = "api-fxpractice.oanda.com" if OANDA_ENV!="live" else "api-fxtrade.oanda.com"

from telegram_send import send_to_tiers

def y2o(yf):
    # minimal map for FX/crypto/gold; extend as needed
    if yf.endswith("=X") and len(yf)==8:  # e.g. EURUSD=X
        return yf[:3]+"_"+yf[3:6]
    if yf=="BTC-USD": return "BTC_USD"
    if yf=="ETH-USD": return "ETH_USD"
    if yf=="XAUUSD=X": return "XAU_USD"
    return None

def load_symbols():
    import yaml
    cfg = yaml.safe_load(open(SYMS))
    return [(s["yf"], s.get("name", s["yf"])) for s in cfg["symbols"] if s.get("enabled", True)]

def oanda_candles(instrument, gran="M5", count=200):
    if not OANDA_API_KEY: raise RuntimeError("OANDA_API_KEY missing")
    url = f"https://{OANDA_HOST}/v3/instruments/{instrument}/candles"
    params = {"granularity": gran, "count": str(count), "price":"M","smooth":"false"}
    r = requests.get(url, headers={"Authorization": f"Bearer {OANDA_API_KEY}"}, params=params, timeout=20)
    r.raise_for_status()
    js = r.json()
    closes = []
    times  = []
    for c in js.get("candles", []):
        if not c.get("complete"): continue
        closes.append(float(c["mid"]["c"]))
        times.append(c["time"])
    return times, closes

def ema(vals, n):
    k = 2/(n+1)
    s=None
    out=[]
    for v in vals:
        s = v if s is None else (v - s)*k + s
        out.append(s)
    return out

def rsi(vals, n=14):
    import math
    if len(vals)<n+1: return [None]*len(vals)
    gains=[0]; losses=[0]
    for i in range(1,len(vals)):
        ch=vals[i]-vals[i-1]
        gains.append(max(ch,0)); losses.append(max(-ch,0))
    avg_g = sum(gains[1:n+1])/n
    avg_l = sum(losses[1:n+1])/n
    rsis=[None]*n
    for i in range(n+1,len(vals)+1):
        avg_g = (avg_g*(n-1) + gains[i-1]) / n
        avg_l = (avg_l*(n-1) + losses[i-1]) / n
        rs = float('inf') if avg_l==0 else avg_g/avg_l
        rsis.append(100 - 100/(1+rs))
    return rsis

def classify(closes):
    # build indicators
    e50  = ema(closes,50)
    e200 = ema(closes,200)
    rsi14= rsi(closes,14)
    if not e50 or not e200 or not rsi14 or e200[-1] is None or rsi14[-1] is None:
        return None, 0, "insufficient data"
    score=0; why=[]
    uptrend = e50[-1] > e200[-1]
    why.append("EMA50>EMA200" if uptrend else "EMA50<EMA200")
    # momentum: last 3-closes slope
    if len(closes)>=4:
        slope = closes[-1]-closes[-4]
        if slope>0: why.append("MOMENTUM↑"); score+=1
        else:       why.append("MOMENTUM↓"); score+=1
    rsi_v = rsi14[-1]
    if rsi_v <= RSI_BUY:  why.append(f"RSI≤{RSI_BUY}");  score+=1
    if rsi_v >= RSI_SELL: why.append(f"RSI≥{RSI_SELL}"); score+=1

    side=None
    if uptrend and rsi_v<=RSI_BUY: side="BUY"
    if (not uptrend) and rsi_v>=RSI_SELL: side="SELL"
    if side is None:
        side = "BUY" if rsi_v<=RSI_BUY else ("SELL" if rsi_v>=RSI_SELL else None)
    return side, score, ", ".join(why)

def append_signal(row):
    header = ["ts_utc","symbol_yf","symbol_pretty","signal","price","expiry_min","evaluate_at_utc","status","score","why","result_price","outcome"]
    exists = LOG.exists()
    with open(LOG, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        if not exists: w.writeheader()
        w.writerow(row)

def main():
    now = datetime.now(timezone.utc)
    header = [f"📡 Pocket Option Signals — {now:%Y-%m-%d %H:%M UTC}", f"Candle: {INTERVAL} | Expiry: {EXPIRY_MIN}m", ""]
    lines = list(header)
    picked=None
    for yf_sym, pretty in load_symbols():
        try:
            o = y2o(yf_sym)
            if not o: 
                lines.append(f"⚪ {pretty} — unsupported map")
                continue
            gran = "M1" if INTERVAL=="1m" else "M5"
            t, c = oanda_candles(o, gran=gran, count=220)
            if len(c)<210:
                lines.append(f"⚪ {pretty} — short history")
                continue
            side, score, why = classify(c)
            px = c[-1]
            if side and score>=MIN_SCORE:
                # choose first strong setup
                picked=(yf_sym,pretty,side,px,score,why)
                break
            else:
                if not SUPPRESS_EMPTY:
                    lines.append(f"⚪ {pretty} — no setup (score {score})")
        except Exception as e:
            lines.append(f"⚠️ {pretty} error: {e}")

    if not picked and MUST_TRADE:
        # fallback: pick strongest momentum among last 8 scanned with price
        # (lightweight fallback, avoids empty posts when marketing)
        pass

    if picked:
        yf_sym, pretty, side, px, score, why = picked
        emoji = "🟢" if side=="BUY" else "🔴"
        lines.append(f"{emoji} {pretty} — {side} @ {px:.5f}")
        lines.append(f"• {why} (score {score})")
        eval_at = now + timedelta(minutes=EXPIRY_MIN)
        append_signal({
            "ts_utc": now.strftime("%Y-%m-%d %H:%M:%S"),
            "symbol_yf": yf_sym,
            "symbol_pretty": pretty,
            "signal": side,
            "price": f"{px:.8f}",
            "expiry_min": EXPIRY_MIN,
            "evaluate_at_utc": eval_at.strftime("%Y-%m-%d %H:%M:%S"),
            "status": "open",
            "score": score,
            "why": why,
            "result_price": "",
            "outcome": ""
        })
    else:
        if not SUPPRESS_EMPTY:
            lines.append("⚠️ No setups met the threshold.")

    send_to_tiers("\n".join(lines))
    print("✅ live run complete")
    return lines

if __name__ == "__main__":
    main()
