import csv, os
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd

from oanda_utils import oanda_price_at

DATA = Path("data"); DATA.mkdir(exist_ok=True)
SIGNALS_CSV = DATA/"signals.csv"
PERF_CSV    = DATA/"perf.csv"

# Map Yahoo symbols to OANDA instruments
YF_TO_OANDA = {
    "EURUSD=X":"EUR_USD", "GBPUSD=X":"GBP_USD", "USDJPY=X":"USD_JPY",
    "USDCHF=X":"USD_CHF", "USDCAD=X":"USD_CAD", "AUDUSD=X":"AUD_USD",
    "NZDUSD=X":"NZD_USD", "EURJPY=X":"EUR_JPY", "XAUUSD=X":"XAU_USD",
    "BTC-USD":"BTC_USD", "ETH-USD":"ETH_USD"
}

def tolerance(symbol_yf: str) -> float:
    s = symbol_yf.upper()
    if s.endswith("JPY=X"): return 1e-3    # ~0.1 pip in JPY pairs
    if s == "XAUUSD=X":     return 0.1     # 10c in gold
    if s in ("BTC-USD","ETH-USD"): return 1.0
    return 1e-5                              # ~0.1 pip

def read_signals():
    if not SIGNALS_CSV.exists(): return []
    with open(SIGNALS_CSV, newline="") as f:
        return list(csv.DictReader(f))

def write_signals(rows):
    header = ["ts_utc","symbol_yf","symbol_pretty","signal","price","expiry_min",
              "evaluate_at_utc","status","score","why","result_price","outcome"]
    with open(SIGNALS_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header); w.writeheader()
        for r in rows: w.writerow(r)

def evaluate_due(rows):
    now = datetime.now(timezone.utc)
    changed = False
    for r in rows:
        if r.get("status","open") != "open": continue
        if not r.get("evaluate_at_utc"):     continue
        try:
            eval_at = datetime.strptime(r["evaluate_at_utc"], "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
        except Exception:
            continue
        if now < eval_at: continue

        yf_symbol = r.get("symbol_yf","")
        inst = YF_TO_OANDA.get(yf_symbol)
        if not inst:
            # skip if we don't have a clean mapping
            continue

        # fetch result price at/after evaluate time
        try:
            result_px = float(oanda_price_at(inst, eval_at.isoformat().replace("+00:00","Z"), granularity="M1"))
        except Exception as e:
            # leave open if we truly can't price
            print(f"⚠️ pricing error for {yf_symbol} at {eval_at}: {e}")
            continue

        entry_px = float(r.get("price","0") or "0")
        side = r.get("signal")
        tol  = tolerance(yf_symbol)

        out=None
        if abs(result_px - entry_px) <= tol:
            out = "DRAW"
        else:
            win = (result_px > entry_px) if side == "BUY" else (result_px < entry_px)
            out = "WIN" if win else "LOSS"

        r["result_price"] = f"{result_px:.8f}"
        r["outcome"] = out
        r["status"] = "closed"
        changed = True
    return changed

def recompute_perf(rows):
    df = pd.DataFrame(rows)
    if df.empty: return None
    df = df[df["status"]=="closed"].copy()
    if df.empty: return None
    df["ts_utc"] = pd.to_datetime(df["ts_utc"], utc=True, errors="coerce")
    df = df.dropna(subset=["ts_utc"])
    now = pd.Timestamp.utcnow()
    d1 = df[df["ts_utc"] >= now - pd.Timedelta(days=1)]
    d7 = df[df["ts_utc"] >= now - pd.Timedelta(days=7)]

    def acc(x):
        if x.empty: return 0.0
        y = x[x["outcome"]!="DRAW"]
        return round(100 * (y["outcome"].eq("WIN").mean() if not y.empty else 0), 1)

    perf = {
        "ts_utc": now.strftime("%Y-%m-%d %H:%M:%S"),
        "count_total": int(len(df)),
        "acc_1d": acc(d1),
        "acc_7d": acc(d7),
        "wins": int((df["outcome"]=="WIN").sum()),
        "losses": int((df["outcome"]=="LOSS").sum()),
        "draws": int((df["outcome"]=="DRAW").sum()),
    }
    header = ["ts_utc","count_total","acc_1d","acc_7d","wins","losses","draws"]
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
