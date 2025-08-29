import csv, requests, os
from datetime import datetime, timezone
from pathlib import Path

DATA   = Path("data"); DATA.mkdir(exist_ok=True)
LOG    = DATA/"signals.csv"
PERF   = DATA/"perf.csv"

OANDA_API_KEY = os.getenv("OANDA_API_KEY","")
OANDA_ENV     = os.getenv("OANDA_ENV","practice")
OANDA_HOST    = "api-fxpractice.oanda.com" if OANDA_ENV!="live" else "api-fxtrade.oanda.com"

def y2o(yf):
    if yf.endswith("=X") and len(yf)==8: return yf[:3]+"_"+yf[3:6]
    if yf=="BTC-USD": return "BTC_USD"
    if yf=="ETH-USD": return "ETH_USD"
    if yf=="XAUUSD=X": return "XAU_USD"
    return None

def oanda_last_after(instrument, when_utc):
    if not OANDA_API_KEY: raise RuntimeError("OANDA_API_KEY missing")
    url = f"https://{OANDA_HOST}/v3/instruments/{instrument}/candles"
    params = {"granularity":"M1","count":"10","price":"M","smooth":"false"}
    r = requests.get(url, headers={"Authorization": f"Bearer {OANDA_API_KEY}"}, params=params, timeout=20)
    r.raise_for_status()
    js = r.json()
    best=None
    for c in js.get("candles", []):
        if not c.get("complete"): continue
        t = datetime.fromisoformat(c["time"].replace("Z","+00:00"))
        if t >= when_utc:
            best = float(c["mid"]["c"]); break
    if best is None and js.get("candles"):
        best = float(js["candles"][-1]["mid"]["c"])
    return best

def read_rows():
    if not LOG.exists(): return []
    with open(LOG, newline="") as f: return list(csv.DictReader(f))

def write_rows(rows):
    header = ["ts_utc","symbol_yf","symbol_pretty","signal","price","expiry_min","evaluate_at_utc","status","score","why","result_price","outcome"]
    with open(LOG,"w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=header); w.writeheader(); [w.writerow(r) for r in rows]

def append_perf(acc1d, acc7d, count, wins, losses):
    exists = PERF.exists()
    with open(PERF,"a",newline="") as f:
        w=csv.DictWriter(f, fieldnames=["ts_utc","count_total","acc_1d","acc_7d","wins","losses"])
        if not exists: w.writeheader()
        w.writerow({
            "ts_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
            "count_total": count, "acc_1d": acc1d, "acc_7d": acc7d, "wins": wins, "losses": losses
        })

def main():
    rows = read_rows()
    if not rows: 
        print("No rows"); return
    now = datetime.now(timezone.utc)
    changed=False
    for r in rows:
        if r.get("status")!="open": continue
        eval_at = datetime.strptime(r["evaluate_at_utc"], "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
        if now < eval_at: continue
        inst = y2o(r["symbol_yf"])
        px_res = oanda_last_after(inst, eval_at)
        entry = float(r["price"]); side = r["signal"]
        if abs(px_res - entry) <= 1e-10:
            outcome="DRAW"
        else:
            outcome = "WIN" if (px_res>entry) == (side=="BUY") else "LOSS"
        r["result_price"]=f"{px_res:.8f}"; r["outcome"]=outcome; r["status"]="closed"; changed=True
    if changed:
        write_rows(rows)
        # quick accuracy calc (full rolling in your existing scripts if you prefer)
        closed=[r for r in rows if r.get("status")=="closed"]
        if closed:
            wins=sum(1 for r in closed if r.get("outcome")=="WIN")
            losses=sum(1 for r in closed if r.get("outcome")=="LOSS")
            acc = round(100*wins/max(1,wins+losses),1)
            append_perf(acc, acc, len(closed), wins, losses)
            print(f"Updated {len(closed)} closed; acc={acc}%")
    else:
        print("No signals due.")
if __name__=="__main__": main()
