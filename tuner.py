import json, math
from pathlib import Path
import pandas as pd
from datetime import datetime, timezone, timedelta

DATA = Path("data"); DATA.mkdir(exist_ok=True)
SIGNALS = DATA/"signals.csv"
TUNING = Path("tuning.json")

# Defaults & guardrails
CFG = {
  "RSI_BUY": 30, "RSI_SELL": 70, "MIN_SCORE": 2,
  "BOUNDS": {"RSI_BUY": [20, 40], "RSI_SELL": [60, 80], "MIN_SCORE": [2, 3]},
  "STEP": {"RSI": 2, "SCORE": 1},
  "TARGET_ACC": 55.0,             # % target over last 7d
  "MIN_TRADES_7D": 30,            # only tune when we have some data
  "COOLDOWN_HOURS": 12            # tune at most twice a day
}

def clamp(v, lo, hi): return max(lo, min(hi, v))

def load_signals():
  if not SIGNALS.exists(): return pd.DataFrame()
  df = pd.read_csv(SIGNALS)
  if df.empty: return df
  # closed outcomes only
  df = df[df["status"].eq("closed").fillna(False)].copy()
  # parse timestamps
  df["ts_utc"] = pd.to_datetime(df["ts_utc"], utc=True, errors="coerce")
  df = df.dropna(subset=["ts_utc"])
  return df

def win_rate(x):
  if len(x)==0: return 0.0
  wins = (x["outcome"]=="WIN").sum()
  return round(100.0 * wins / len(x), 1)

def main():
  now = datetime.now(timezone.utc)
  df = load_signals()
  if df.empty:
    # first run: write defaults if not present
    if not TUNING.exists():
      json.dump({"RSI_BUY":CFG["RSI_BUY"],"RSI_SELL":CFG["RSI_SELL"],"MIN_SCORE":CFG["MIN_SCORE"],"updated":now.isoformat()}, open(TUNING,"w"))
    print("No closed trades yet; keeping defaults.")
    return

  last7 = df[df["ts_utc"] >= (now - timedelta(days=7))].copy()
  if len(last7) < CFG["MIN_TRADES_7D"]:
    # not enough data to tune
    if not TUNING.exists():
      json.dump({"RSI_BUY":CFG["RSI_BUY"],"RSI_SELL":CFG["RSI_SELL"],"MIN_SCORE":CFG["MIN_SCORE"],"updated":now.isoformat()}, open(TUNING,"w"))
    print(f"Only {len(last7)} trades in 7d; skipping tune.")
    return

  # read current
  cur = {"RSI_BUY":CFG["RSI_BUY"],"RSI_SELL":CFG["RSI_SELL"],"MIN_SCORE":CFG["MIN_SCORE"],"updated":(now - timedelta(days=1)).isoformat()}
  if TUNING.exists():
    try: cur.update(json.load(open(TUNING)))
    except: pass

  # cooldown
  try:
    last_upd = datetime.fromisoformat(cur.get("updated")).replace(tzinfo=timezone.utc)
  except: 
    last_upd = now - timedelta(days=1)
  if (now - last_upd).total_seconds() < CFG["COOLDOWN_HOURS"]*3600:
    print("Cooldown active; no tune.")
    return

  # compute recent performance
  acc7 = win_rate(last7)
  buys = last7[last7["signal"]=="BUY"]
  sells= last7[last7["signal"]=="SELL"]
  accB = win_rate(buys)
  accS = win_rate(sells)

  new = dict(cur)

  # Adjust MIN_SCORE first: if under target and enough trades → tighten to 3, else 2
  if acc7 < CFG["TARGET_ACC"]:
    new["MIN_SCORE"] = clamp(cur["MIN_SCORE"] + CFG["STEP"]["SCORE"], *CFG["BOUNDS"]["MIN_SCORE"])
  else:
    # if too few signals overall (e.g., < 200 in 7d), relax back
    if len(last7) < 200:
      new["MIN_SCORE"] = clamp(cur["MIN_SCORE"] - CFG["STEP"]["SCORE"], *CFG["BOUNDS"]["MIN_SCORE"])

  # Balance BUY vs SELL quality with small RSI nudges
  # If BUY perf < SELL, make buys pickier (lower RSI_BUY -> need deeper oversold)
  if accB < accS - 2:
    new["RSI_BUY"]  = clamp(cur["RSI_BUY"] - CFG["STEP"]["RSI"], *CFG["BOUNDS"]["RSI_BUY"])
  elif accB > accS + 2:
    new["RSI_BUY"]  = clamp(cur["RSI_BUY"] + CFG["STEP"]["RSI"], *CFG["BOUNDS"]["RSI_BUY"])

  # If SELL perf < BUY, make sells pickier (raise RSI_SELL -> need higher overbought)
  if accS < accB - 2:
    new["RSI_SELL"] = clamp(cur["RSI_SELL"] + CFG["STEP"]["RSI"], *CFG["BOUNDS"]["RSI_SELL"])
  elif accS > accB + 2:
    new["RSI_SELL"] = clamp(cur["RSI_SELL"] - CFG["STEP"]["RSI"], *CFG["BOUNDS"]["RSI_SELL"])

  new["updated"] = now.isoformat()
  json.dump(new, open(TUNING,"w"))
  print("Tuned:", new)

if __name__ == "__main__":
  main()
