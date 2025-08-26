import json, math
from pathlib import Path
import pandas as pd
from datetime import datetime, timezone, timedelta

DATA = Path("data"); DATA.mkdir(exist_ok=True)
SIGNALS = DATA/"signals.csv"
TUNING = Path("tuning.json")

CFG = {
  "RSI_BUY": 30, "RSI_SELL": 70, "MIN_SCORE": 2,
  "BOUNDS": {"RSI_BUY": [20, 40], "RSI_SELL": [60, 80], "MIN_SCORE": [2, 3]},
  "STEP": {"RSI": 2, "SCORE": 1},
  "TARGET_ACC": 55.0,
  "MIN_TRADES_7D": 30,
  "COOLDOWN_HOURS": 12
}

def clamp(v, lo, hi): return max(lo, min(hi, v))

def load_signals():
  if not SIGNALS.exists(): return pd.DataFrame()
  df = pd.read_csv(SIGNALS)
  if df.empty: return df
  df = df[df["status"].eq("closed").fillna(False)].copy()
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
    if not TUNING.exists():
      json.dump({"RSI_BUY":CFG["RSI_BUY"],"RSI_SELL":CFG["RSI_SELL"],"MIN_SCORE":CFG["MIN_SCORE"],"updated":now.isoformat()}, open(TUNING,"w"))
    print("No closed trades yet; keeping defaults."); return

  last7 = df[df["ts_utc"] >= (now - timedelta(days=7))].copy()
  if len(last7) < CFG["MIN_TRADES_7D"]:
    if not TUNING.exists():
      json.dump({"RSI_BUY":CFG["RSI_BUY"],"RSI_SELL":CFG["RSI_SELL"],"MIN_SCORE":CFG["MIN_SCORE"],"updated":now.isoformat()}, open(TUNING,"w"))
    print(f"Only {len(last7)} trades in 7d; skipping tune."); return

  cur = {"RSI_BUY":CFG["RSI_BUY"],"RSI_SELL":CFG["RSI_SELL"],"MIN_SCORE":CFG["MIN_SCORE"],"updated":(now - timedelta(days=1)).isoformat()}
  if TUNING.exists():
    try: cur.update(json.load(open(TUNING)))
    except: pass

  try:
    last_upd = datetime.fromisoformat(cur.get("updated")).replace(tzinfo=timezone.utc)
  except:
    last_upd = now - timedelta(days=1)
  if (now - last_upd).total_seconds() < CFG["COOLDOWN_HOURS"]*3600:
    print("Cooldown active; no tune."); return

  acc7 = win_rate(last7)
  buys = last7[last7["signal"]=="BUY"]
  sells= last7[last7["signal"]=="SELL"]
  accB, accS = win_rate(buys), win_rate(sells)

  new = dict(cur)
  if acc7 < CFG["TARGET_ACC"]:
    new["MIN_SCORE"] = clamp(cur["MIN_SCORE"] + CFG["STEP"]["SCORE"], *CFG["BOUNDS"]["MIN_SCORE"])
  else:
    if len(last7) < 200:
      new["MIN_SCORE"] = clamp(cur["MIN_SCORE"] - CFG["STEP"]["SCORE"], *CFG["BOUNDS"]["MIN_SCORE"])

  if accB < accS - 2:
    new["RSI_BUY"]  = clamp(cur["RSI_BUY"] - CFG["STEP"]["RSI"], *CFG["BOUNDS"]["RSI_BUY"])
  elif accB > accS + 2:
    new["RSI_BUY"]  = clamp(cur["RSI_BUY"] + CFG["STEP"]["RSI"], *CFG["BOUNDS"]["RSI_BUY"])

  if accS < accB - 2:
    new["RSI_SELL"] = clamp(cur["RSI_SELL"] + CFG["STEP"]["RSI"], *CFG["BOUNDS"]["RSI_SELL"])
  elif accS > accB + 2:
    new["RSI_SELL"] = clamp(cur["RSI_SELL"] - CFG["STEP"]["RSI"], *CFG["BOUNDS"]["RSI_SELL"])

  new["updated"] = now.isoformat()
  json.dump(new, open(TUNING,"w"))
  print("Tuned:", new)

if __name__ == "__main__":
  main()
