try:
    import orb
except ModuleNotFoundError:
    print("⚠️ ORB module not found, skipping ORB strategy.")
    orb=None

try:
    import orb
except ModuleNotFoundError:
    print("⚠️ ORB module not found, skipping ORB strategy.")
    orb=None

import os, csv, yaml, time
from pathlib import Path
from datetime import datetime, timezone, timedelta
from telegram_send import send_to_tiers
try: import orb
except ModuleNotFoundError: print("⚠️ ORB module not found, skipping ORB strategy."); orb=None

INTERVAL   = os.getenv("INTERVAL","1m")         # 1m candles
EXPIRY_MIN = int(os.getenv("EXPIRY_MIN","5"))   # 5m expiry
SHARDS     = int(os.getenv("SHARDS","5"))       # spread load across minutes
DATA       = Path("data"); DATA.mkdir(exist_ok=True)
LOG        = DATA/"signals.csv"

def load_symbols():
    cfg = yaml.safe_load(open("symbols.yaml"))
    out=[]
    for s in cfg.get("symbols",[]):
        if not s.get("enabled",True): continue
        oanda = s.get("oanda")
        if not oanda:  # derive from yf if needed
            yf = s.get("yf","")
            if yf.endswith("=X"):
                bq=yf[:-2]; oanda=f"{bq[:3]}_{bq[3:]}"
        pretty = s.get("name", oanda or "Unknown")
        group  = s.get("group","FX")
        if oanda: out.append((oanda, pretty, group))
    return out

def shard(symbols):
    now = datetime.utcnow()
    i = now.minute % SHARDS
    return [s for idx,s in enumerate(symbols) if idx % SHARDS == i]

def append_row(row: dict):
    header = ["ts_utc","symbol_oanda","symbol_pretty","signal","price","expiry_min",
              "evaluate_at_utc","status","score","why","result_price","outcome"]
    exists = LOG.exists()
    with open(LOG,"a",newline="") as f:
        w=csv.DictWriter(f, fieldnames=header)
        if not exists: w.writeheader()
        w.writerow(row)

def main():
    t0 = time.time()
    now = datetime.now(timezone.utc)

    symbols = load_symbols()
    symbols = shard(symbols)  # load-balanced slice this minute

    lines = [
        f"📡 Pocket Option Signals — {now:%Y-%m-%d %H:%M UTC}",
        f"Candle: 1m | Expiry: {EXPIRY_MIN}m",
        ""
    ]

    any_signal=False
    for oanda_code, pretty, group in symbols:
        try:
            sig = orb.orb_signal(oanda_code, pretty, expiry_min=EXPIRY_MIN)
            # Expect either (side, score, why, price) or (None, reason)
            if isinstance(sig, tuple) and sig and sig[0] in ("BUY","SELL"):
                side, score, why, price = sig[:4]
                emoji = "🟢" if side=="BUY" else "🔴"
                lines.append(f"{emoji} {pretty} — {side} @ {float(price):.5f}")
                if why: lines.append(f"• {why} (score {score})")
                evaluate_at = now + timedelta(minutes=EXPIRY_MIN)
                append_row({
                    "ts_utc": now.strftime("%Y-%m-%d %H:%M:%S"),
                    "symbol_oanda": oanda_code,
                    "symbol_pretty": pretty,
                    "signal": side,
                    "price": f"{float(price):.8f}",
                    "expiry_min": EXPIRY_MIN,
                    "evaluate_at_utc": evaluate_at.strftime("%Y-%m-%d %H:%M:%S"),
                    "status": "open",
                    "score": score,
                    "why": why,
                    "result_price": "",
                    "outcome": ""
                })
                any_signal=True
            else:
                # quiet unless you want per-pair "no setup"
                pass
        except Exception as e:
            lines.append(f"⚠️ {pretty} — error: {e}")

    if not any_signal:
        lines.append("⚪ No ORB setups this run.")

    lines.append(f"\n⏱ Build time: `{time.time()-t0:.1f}s`")
    send_to_tiers("\n".join(lines))
    print("✅ Signals sent.")
    return lines

if __name__=="__main__":
    main()
