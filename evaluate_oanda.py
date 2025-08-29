import os, csv, requests, yfinance as yf
from pathlib import Path
from datetime import datetime, timezone
import pandas as pd

DATA = Path("data"); DATA.mkdir(exist_ok=True)
SIGNALS = DATA / "signals.csv"

OANDA_API_KEY = os.getenv("OANDA_API_KEY","")
OANDA_ENV     = os.getenv("OANDA_ENV","practice").lower()
OANDA_HOST    = "https://api-fxpractice.oanda.com" if OANDA_ENV!="live" else "https://api-fxtrade.oanda.com"

def map_to_oanda(yf_symbol:str)->str|None:
    # EURUSD=X -> EUR_USD ; XAUUSD=X -> XAU_USD
    if yf_symbol.endswith("=X"):
        base=yf_symbol[:-2]
        if len(base)>=6: return base[:3]+"_"+base[3:]
    return None

def oanda_first_close_at_or_after(inst:str, when_utc:datetime)->float|None:
    if not OANDA_API_KEY: return None
    # Grab last ~120 minutes; pick first candle at/after evaluate_at
    url=f"{OANDA_HOST}/v3/instruments/{inst}/candles"
    r=requests.get(url,params={"granularity":"M1","count":120,"price":"M"},
                   headers={"Authorization":f"Bearer {OANDA_API_KEY}"}, timeout=20)
    r.raise_for_status()
    js=r.json()
    rows=[]
    for c in js.get("candles",[]):
        if not c.get("complete"): continue
        ts=pd.to_datetime(c["time"], utc=True)
        close=float(c["mid"]["c"])
        rows.append((ts,close))
    if not rows: return None
    for ts,cl in rows:
        if ts>=when_utc: return cl
    return rows[-1][1]  # fallback: last

def yf_first_close_at_or_after(yf_symbol:str, when_utc:datetime)->float|None:
    df=yf.download(yf_symbol, interval="1m", period="2d", progress=False)
    if df is None or df.empty: return None
    df.index = df.index.tz_convert("UTC")
    later=df[df.index>=when_utc]
    if later.empty: return float(df["Close"].iloc[-1])
    return float(later["Close"].iloc[0])

def load_rows():
    if not SIGNALS.exists(): return []
    with open(SIGNALS,newline="") as f:
        return list(csv.DictReader(f))

def write_rows(rows):
    header=["ts_utc","symbol_yf","symbol_pretty","signal","price","expiry_min","evaluate_at_utc","status","score","why","result_price","outcome"]
    with open(SIGNALS,"w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=header)
        w.writeheader()
        for r in rows: w.writerow(r)

def main():
    rows=load_rows()
    if not rows: 
        print("No signals file.")
        return
    now=datetime.now(timezone.utc)
    changed=False
    for r in rows:
        if r.get("status")!="open": continue
        if not r.get("evaluate_at_utc"): continue
        try:
            eval_at=datetime.strptime(r["evaluate_at_utc"], "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
        except: 
            continue
        if now<eval_at: continue

        yf_sym=r["symbol_yf"]
        entry=float(r["price"])
        side=r["signal"]
        # choose price source
        inst=map_to_oanda(yf_sym)
        if inst:
            px=oanda_first_close_at_or_after(inst, eval_at)
        else:
            px=yf_first_close_at_or_after(yf_sym, eval_at)
        if px is None: 
            continue

        eps=1e-8
        if abs(px-entry)<=eps:
            outcome="DRAW"
        else:
            win = (px>entry) if side=="BUY" else (px<entry)
            outcome="WIN" if win else "LOSS"

        r["result_price"]=f"{px:.8f}"
        r["outcome"]=outcome
        r["status"]="closed"
        changed=True

    if changed:
        write_rows(rows)
        # Optional: post a tiny per-run tally to VIP only
        wins=loss=draw=0
        for r in rows:
            if r.get("status")=="closed" and r.get("outcome"):
                if r["outcome"]=="WIN": wins+=1
                elif r["outcome"]=="LOSS": loss+=1
                elif r["outcome"]=="DRAW": draw+=1
        try:
            from telegram_send import send_telegram
            tok=os.getenv("TELEGRAM_BOT_TOKEN"); vip=os.getenv("TELEGRAM_CHAT_VIP")
            if tok and vip:
                acc = round(100*wins/max(1,wins+loss),1)
                msg = f"🧮 Eval update: W {wins} | L {loss} | D {draw} (acc {acc}%)"
                send_telegram(msg, vip)
        except Exception as e:
            print("Note: telegram summary skipped:", e)
        print("Updated outcomes.")
    else:
        print("No signals due for evaluation.")
if __name__=="__main__":
    main()
