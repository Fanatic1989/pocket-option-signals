import csv
from datetime import datetime, timezone
from pathlib import Path

import yfinance as yf
import pandas as pd

DATA_DIR = Path("data"); DATA_DIR.mkdir(exist_ok=True)
SIGNALS_CSV = DATA_DIR / "signals.csv"
PERF_CSV    = DATA_DIR / "perf.csv"

def read_signals():
    if not SIGNALS_CSV.exists(): return []
    with open(SIGNALS_CSV, newline="") as f:
        return list(csv.DictReader(f))

def write_signals(rows):
    header = ["ts_utc","symbol_yf","symbol_pretty","signal","price","expiry_min","evaluate_at_utc","status","score","why","result_price","outcome"]
    with open(SIGNALS_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        for r in rows:
            w.writerow(r)

def evaluate_due(rows):
    now = datetime.now(timezone.utc)
    changed = False
    for r in rows:
        if r.get("status","open") != "open": continue
        eval_at = datetime.strptime(r["evaluate_at_utc"], "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
        if now >= eval_at:
            yf_sym = r["symbol_yf"]
            hist = yf.download(tickers=yf_sym, interval="1m", period="1d", progress=False).dropna()
            if hist.empty:
                continue
            idx = hist.index.tz_convert("UTC")
            later = hist[idx >= eval_at]
            if later.empty:
                result_px = float(hist["Close"].iloc[-1])
            else:
                result_px = float(later["Close"].iloc[0])
            entry_px = float(r["price"])
            side = r["signal"]
            eps=1e-8
            if abs(result_px-entry_px)<=eps:
                win=None  # DRAW
            else:
                win = (result_px > entry_px) if side == "BUY" else (result_px < entry_px)
            r["result_price"] = f"{result_px:.8f}"
            r["outcome"] = "DRAW" if win is None else ("WIN" if win else "LOSS")
            r["status"] = "closed"
            changed = True
    return changed

def recompute_perf(rows):
    df = pd.DataFrame(rows)
    if df.empty: return None
    df = df[df["status"]=="closed"]
    if df.empty: return None
    df["ts_utc"] = pd.to_datetime(df["ts_utc"], utc=True)
    df["is_win"] = (df["outcome"]=="WIN").astype(int)

    now = pd.Timestamp.utcnow()
    d1 = df[df["ts_utc"] >= now - pd.Timedelta(days=1)]
    d7 = df[df["ts_utc"] >= now - pd.Timedelta(days=7)]

    def acc(x): 
        return round(100 * (x["is_win"].mean() if len(x)>0 else 0), 1)

    perf = {
        "ts_utc": now.strftime("%Y-%m-%d %H:%M:%S"),
        "count_total": int(len(df)),
        "acc_1d": acc(d1),
        "acc_7d": acc(d7),
        "wins": int(df["is_win"].sum()),
        "losses": int((1-df["is_win"]).sum()),
    }
    header = ["ts_utc","count_total","acc_1d","acc_7d","wins","losses"]
    exists = PERF_CSV.exists()
    with open(PERF_CSV,"a",newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        if not exists: w.writeheader()
        w.writerow(perf)
    return perf

def main():
    rows = read_signals()
    changed = evaluate_due(rows)
    if changed:
        write_signals(rows)
        perf = recompute_perf(rows)
        if perf: print("Updated accuracy:", perf)
    else:
        print("No signals due for evaluation.")

if __name__ == "__main__":
    main()
