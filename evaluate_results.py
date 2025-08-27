import os, requests, pandas as pd
from pathlib import Path
from datetime import datetime, timezone

DATA = Path("data"); DATA.mkdir(exist_ok=True)
CSV  = DATA/"signals.csv"

OANDA_API_KEY = os.getenv("OANDA_API_KEY")
OANDA_ENV     = os.getenv("OANDA_ENV","practice")

def oanda_base():
    return "https://api-fxtrade.oanda.com" if OANDA_ENV=="live" else "https://api-fxpractice.oanda.com"

def pair_to_oanda(pair:str):
    p = pair.upper().replace(" ","")
    if p=="GOLD": return "XAU_USD"
    if p=="SILVER": return "XAG_USD"
    if "/" in p and len(p)==7:
        a,b = p.split("/")
        return f"{a}_{b}"
    return None

def fetch_oanda_close_at(instrument:str, ts_utc: pd.Timestamp):
    url = f"{oanda_base()}/v3/instruments/{instrument}/candles"
    headers={"Authorization": f"Bearer {OANDA_API_KEY}"}
    params = {"granularity":"M1","count":"200","price":"M"}
    r = requests.get(url, headers=headers, params=params, timeout=20); r.raise_for_status()
    candles = r.json().get("candles",[])
    best = None
    for c in candles:
        if not c.get("complete"): continue
        ct = pd.to_datetime(c["time"], utc=True)
        if ct >= ts_utc and best is None:
            best = float(c["mid"]["c"])
            break
    if best is None and candles:
        best = float(candles[-1]["mid"]["c"])
    return best

def crypto_to_binance(pair:str):
    # e.g. BTC/USDT -> BTCUSDT
    p = pair.upper().replace("/","")
    return p if p.endswith("USDT") else None

def fetch_binance_close_at(symbol:str, ts_utc: pd.Timestamp):
    url="https://api.binance.com/api/v3/klines"
    params={"symbol":symbol,"interval":"1m","limit":"500"}
    r=requests.get(url, params=params, timeout=20); r.raise_for_status()
    rows=r.json()
    target=None
    for k in rows:
        closeTime = pd.to_datetime(int(k[6]), unit="ms", utc=True)
        if closeTime >= ts_utc:
            target=float(k[4]); break
    if target is None and rows:
        target=float(rows[-1][4])
    return target

def ensure_columns(df: pd.DataFrame):
    for col in ["status","result_price","outcome"]:
        if col not in df.columns:
            df[col] = ""
    # Mark open where we have an expiry time and no outcome
    df.loc[(df["expires_utc"].notna()) & (df["expires_utc"]!="") & (df["outcome"]==""), "status"] = \
        df["status"].mask(df["status"]=="", "open")
    return df

def main():
    if not CSV.exists():
        print("No signals.csv yet."); return
    df = pd.read_csv(CSV)
    if df.empty:
        print("signals.csv empty"); return

    # normalize timestamps
    df = ensure_columns(df)
    df["expires_utc"] = pd.to_datetime(df["expires_utc"], utc=True, errors="coerce")
    df["ts_utc"]      = pd.to_datetime(df["ts_utc"], utc=True, errors="coerce")

    now = pd.Timestamp.now(tz=timezone.utc)
    changed = False
    for i, r in df.iterrows():
        if r["status"] != "open": 
            continue
        exp = r["expires_utc"]
        if pd.isna(exp) or now < exp:
            continue

        pair = str(r["pair"])
        side = str(r["side"]).upper()
        entry = float(r["price"]) if r["price"]!="" else None

        # choose source
        result_px = None
        try:
            if pair.endswith("/USDT"):  # crypto on Binance
                sym = crypto_to_binance(pair)
                if sym: result_px = fetch_binance_close_at(sym, exp)
            else:                        # FX/metals on OANDA
                inst = pair_to_oanda(pair)
                if inst and OANDA_API_KEY:
                    result_px = fetch_oanda_close_at(inst, exp)
        except Exception as e:
            print(f"Fetch error for {pair}: {e}")
            continue

        if result_px is None or entry is None:
            continue

        eps = 1e-8
        if abs(result_px - entry) <= eps:
            outcome = "DRAW"
        else:
            win = (result_px > entry) if side=="BUY" else (result_px < entry)
            outcome = "WIN" if win else "LOSS"

        df.at[i, "result_price"] = f"{result_px:.8f}"
        df.at[i, "outcome"] = outcome
        df.at[i, "status"] = "closed"
        changed = True
        print(f"Closed: {pair} {side} -> {outcome}  entry {entry}  result {result_px}")

    if changed:
        df.to_csv(CSV, index=False)
        print("Updated signals.csv")
    else:
        print("No trades due for evaluation.")

if __name__ == "__main__":
    main()
