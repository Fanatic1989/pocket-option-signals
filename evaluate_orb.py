import os, csv, time, requests
from pathlib import Path
from datetime import datetime, timezone, timedelta

DATA = Path("data"); DATA.mkdir(exist_ok=True)
LOG  = DATA/"signals.csv"
PERF = DATA/"perf.csv"

KEY  = os.getenv("OANDA_API_KEY","").strip()
ENV  = os.getenv("OANDA_ENV","practice").lower().strip()
HOST = "https://api-fxtrade.oanda.com" if ENV=="live" else "https://api-fxpractice.oanda.com"

def latest_mid(oanda_code):
    if not KEY: raise RuntimeError("OANDA_API_KEY missing")
    url = f"{HOST}/v3/instruments/{oanda_code}/candles"
    r = requests.get(url, params={"granularity":"M1","count":2,"price":"M"},
                     headers={"Authorization": f"Bearer {KEY}"}, timeout=20)
    r.raise_for_status()
    js = r.json().get("candles",[])
    if not js: raise RuntimeError("no candles")
    m = js[-1]["mid"]   # {'o','h','l','c'}
    return float(m["c"])

def read_rows():
    if not LOG.exists(): return []
    with open(LOG, newline="") as f: return list(csv.DictReader(f))

def write_rows(rows):
    header = ["ts_utc","symbol_oanda","symbol_pretty","signal","price","expiry_min",
              "evaluate_at_utc","status","score","why","result_price","outcome"]
    with open(LOG,"w",newline="") as f:
        w=csv.DictWriter(f, fieldnames=header); w.writeheader()
        for r in rows: w.writerow(r)

def summarize(rows):
    closed=[r for r in rows if r.get("status")=="closed"]
    if not closed: return None
    wins  = sum(1 for r in closed if r.get("outcome")=="WIN")
    loss  = sum(1 for r in closed if r.get("outcome")=="LOSS")
    draw  = sum(1 for r in closed if r.get("outcome")=="DRAW")
    acc   = round(100* wins / max(1, (wins+loss)), 1)
    return {"ts_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
            "count_total": len(closed), "acc_7d": acc, "wins": wins, "losses": loss}

def main():
    rows = read_rows()
    if not rows: 
        print("No signals.csv yet."); return

    now = datetime.now(timezone.utc)
    changed=False
    for r in rows:
        if r.get("status")!="open": continue
        eval_at = datetime.strptime(r["evaluate_at_utc"], "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
        if now >= eval_at:
            try:
                px_res = latest_mid(r["symbol_oanda"])
            except Exception as e:
                print("Fetch result price failed:", e)
                continue
            entry = float(r["price"])
            side  = r["signal"]
            eps   = 1e-9
            if abs(px_res - entry) <= eps:
                outcome="DRAW"
            else:
                win = (px_res > entry) if side=="BUY" else (px_res < entry)
                outcome="WIN" if win else "LOSS"
            r["result_price"]=f"{px_res:.8f}"
            r["outcome"]=outcome
            r["status"]="closed"
            changed=True

    if changed:
        write_rows(rows)
        s = summarize(rows)
        if s:
            exists = PERF.exists()
            with open(PERF,"a",newline="") as f:
                w=csv.DictWriter(f, fieldnames=["ts_utc","count_total","acc_7d","wins","losses"])
                if not exists: w.writeheader()
                w.writerow(s)
        print("Updated outcomes.")
    else:
        print("No signals due for evaluation.")

if __name__=="__main__":
    main()
