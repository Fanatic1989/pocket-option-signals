import csv, requests
from pathlib import Path
from datetime import datetime, timezone, timedelta
import pandas as pd
from oanda_utils import oanda_get_candle
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass


DATA = Path("data"); DATA.mkdir(exist_ok=True)
SIGNALS_CSV = DATA / "signals.csv"
PERF_CSV    = DATA / "perf.csv"

def read_signals():
    if not SIGNALS_CSV.exists(): return []
    with open(SIGNALS_CSV, newline="") as f:
        return list(csv.DictReader(f))

def write_signals(rows):
    header = ["ts_utc","symbol_yf","symbol_pretty","signal","price","expiry_min",
              "evaluate_at_utc","status","score","why","result_price","outcome"]
    with open(SIGNALS_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        for r in rows: w.writerow(r)

def fetch_1m_close(symbol_yf: str, when_utc: datetime) -> float:
    sym = (symbol_yf or "").upper()

    # --- Crypto via Binance (USDT)
    if sym in ("BTC-USD","BTCUSD","BTCUSD=X"):
        pair = "BTCUSDT"
    elif sym in ("ETH-USD","ETHUSD","ETHUSD=X"):
        pair = "ETHUSDT"
    else:
        pair = None

    if pair:
        url = f"https://api.binance.com/api/v3/klines?symbol={pair}&interval=1m&limit=6"
        js = requests.get(url, timeout=10).json()
        # kline: [open_time, open, high, low, close, volume, close_time, ...]
        target = int(when_utc.timestamp())
        for k in js:
            ts = int(k[0]) // 1000
            if ts >= target:
                return float(k[4])
        return float(js[-1][4])

    # --- FX via Stooq (eurusd -> /?s=eurusd&i=1)
    if sym.endswith("=X") and len(sym) == 8:
        base = sym[:3].lower() + sym[3:6].lower()   # EURUSD=X -> eurusd
        url = f"https://stooq.com/q/d/l/?s={base}&i=1"
        txt = requests.get(url, timeout=10).text.strip().splitlines()
        # header: Date,Time,Open,High,Low,Close,Volume
        for row in txt[1:]:
            d,t,o,h,l,c,v = row.split(",")
            ts = datetime.strptime(d + " " + t, "%Y-%m-%d %H:%M").replace(tzinfo=timezone.utc)
            if ts >= when_utc:
                return float(c)
        # if none >= when_utc, return last close
        return float(txt[-1].split(",")[5])

    # If we get here, no supported feed
    raise RuntimeError(f"No feed for {symbol_yf}")

def evaluate_due(rows):
    now = datetime.now(timezone.utc)
    changed = False
    for r in rows:
        if r.get("status","open") != "open": 
            continue
        dt = r.get("evaluate_at_utc") or r.get("expires_utc")
        if not dt: 
            continue
        # tolerant parse (supports ISO with Z/offsets)
        try:
            eval_at = datetime.fromisoformat(dt.replace("Z","+00:00")).astimezone(timezone.utc)
        except Exception:
            eval_at = datetime.strptime(dt, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)

        if now >= eval_at:
            try:
                result_px = fetch_1m_close(r["symbol_yf"], eval_at)
            except Exception:
                # skip if no data
                continue

            entry_px = float(r["price"])
            side = r["signal"]
            eps = 1e-5
            if abs(result_px - entry_px) <= eps:
                outcome = "DRAW"
            else:
                if side == "BUY":
                    outcome = "WIN" if result_px > entry_px else "LOSS"
                else:
                    outcome = "WIN" if result_px < entry_px else "LOSS"

            r["result_price"] = f"{result_px:.8f}"
            r["outcome"] = outcome
            r["status"] = "closed"
            changed = True
    return changed

def recompute_perf(rows):
    df = pd.DataFrame(rows)
    if df.empty: return None
    df = df[df["status"]=="closed"]
    if df.empty: return None
    df["ts_utc"] = pd.to_datetime(df["ts_utc"], utc=True, errors="coerce")
    df = df.dropna(subset=["ts_utc"])
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