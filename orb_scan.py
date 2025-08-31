import os, csv, yaml
from datetime import datetime, timezone, timedelta
from pathlib import Path

from orb import orb_signal
from telegram_send import send_to_tiers  # uses your TELEGRAM_* envs

DATA = Path("data"); DATA.mkdir(exist_ok=True)
SIGNALS_CSV = DATA/"signals.csv"
EXPIRY_MIN = int(os.getenv("EXPIRY_MIN","5"))  # for ORB we want 5 by default
INTERVAL   = os.getenv("INTERVAL","1m")

def load_symbols():
    cfg = yaml.safe_load(open("symbols.yaml"))
    return [(s["yf"], s.get("name", s["yf"])) for s in cfg["symbols"] if s.get("enabled", True)]

def yf_to_oanda(yf_sym: str) -> str:
    s = (yf_sym or "").upper().strip()
    # EURUSD=X, GBPJPY=X, XAUUSD=X  ->  EUR_USD, GBP_JPY, XAU_USD
    if s.endswith("=X") and len(s) >= 8:
        core = s[:-2]          # drop '=X'
        base, quote = core[:3], core[3:6]
        return f"{base}_{quote}"
    # BTC-USD, ETH-USD -> BTC_USD, ETH_USD
    if "-" in s:
        base, quote = s.split("-", 1)
        return f"{base}_{quote}"
    # Fallback: replace slash if given like EUR/USD
    return s.replace("/", "_")
    if "-" in yf_sym:
        a,b = yf_sym.split("-",1)
        return f"{a}_{b}"
    return yf_sym.replace("/","_").upper()

def append_signal(row: dict):
    header = ["ts_utc","symbol_yf","symbol_pretty","signal","price","expiry_min","evaluate_at_utc","status","score","why","result_price","outcome"]
    exists = SIGNALS_CSV.exists() and SIGNALS_CSV.read_text().strip()!=""
    with open(SIGNALS_CSV,"a",newline="") as f:
        w=csv.DictWriter(f, fieldnames=header)
        if not exists: w.writeheader()
        w.writerow(row)

def main():
    now = datetime.now(timezone.utc)
    lines=[f"📡 Pocket Option Signals — {now:%Y-%m-%d %H:%M UTC}",
           f"Candle: {INTERVAL} | Expiry: {EXPIRY_MIN}m", ""]
    fired=0
    for yf_sym, pretty in load_symbols():
        oanda = yf_to_oanda(yf_sym)
        try:
            res, note = orb_signal(oanda, pretty, expiry_min=EXPIRY_MIN)
        except Exception as e:
            lines.append(f"⚠️ {pretty} — error: {e}")
            continue
        if not res:
            continue
        side = res["signal"]; px = float(res["price"]); score=int(res.get("score",2)); why=res.get("why","ORB")
        emoji = "🟢" if side=="BUY" else "🔴"
        lines.append(f"{emoji} {pretty} — {side} @ {px:.5f}\n• {why} (score {score})")
        evaluate_at = now + timedelta(minutes=EXPIRY_MIN)
        append_signal({
            "ts_utc": now.strftime("%Y-%m-%d %H:%M:%S"),
            "symbol_yf": yf_sym,
            "symbol_pretty": pretty,
            "signal": side,
            "price": f"{px:.8f}",
            "expiry_min": EXPIRY_MIN,
            "evaluate_at_utc": evaluate_at.strftime("%Y-%m-%d %H:%M:%S"),
            "status": "open",
            "score": score,
            "why": why
        })
        fired += 1

    if fired==0:
        # keep quiet if you prefer; here we post a short heartbeat
        lines.append("⚪ No ORB setups this run.")
    send_to_tiers("\n".join(lines))
    print(f"Done. Signals fired: {fired}")

if __name__ == "__main__":
    main()
