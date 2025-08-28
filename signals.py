import os, csv
from datetime import datetime, timezone, timedelta
from pathlib import Path
import requests

# ---------- ENV ----------
INTERVAL   = os.getenv("INTERVAL", "5m")
EXPIRY_MIN = int(os.getenv("EXPIRY_MIN", "10"))
MUST_TRADE = int(os.getenv("MUST_TRADE", "0"))
SUPPRESS_EMPTY = int(os.getenv("SUPPRESS_EMPTY", "1"))

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TG_IDS = {
    "FREE":  os.getenv("TELEGRAM_CHAT_FREE",  ""),
    "BASIC": os.getenv("TELEGRAM_CHAT_BASIC",""),
    "PRO":   os.getenv("TELEGRAM_CHAT_PRO",  ""),
    "VIP":   os.getenv("TELEGRAM_CHAT_VIP",  ""),
}

DATA = Path("data"); DATA.mkdir(exist_ok=True)
SIGNALS_CSV = DATA/"signals.csv"

# ---------- TELEGRAM ----------
def send_telegram(text: str, chat_id: str) -> bool:
    if not TELEGRAM_BOT_TOKEN or not chat_id:
        return False
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    r = requests.post(url, data={"chat_id": chat_id, "text": text, "parse_mode":"Markdown"})
    try:
        r.raise_for_status()
        ok = r.json().get("ok", False)
        print(("✅" if ok else "⚠️"), "sent →", chat_id)
        return ok
    except Exception as e:
        print("⚠️ telegram error:", e)
        return False

def send_to_all_tiers(text: str):
    for k, cid in TG_IDS.items():
        if cid:
            send_telegram(text, cid)

# ---------- CSV LOG ----------
HEAD = ["ts_utc","symbol_yf","symbol_pretty","signal","price","expiry_min",
        "evaluate_at_utc","status","score","why","result_price","outcome"]

def append_signal(row: dict):
    exists = SIGNALS_CSV.exists() and SIGNALS_CSV.read_text().strip() != ""
    with open(SIGNALS_CSV, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=HEAD)
        if not exists:
            w.writeheader()
        w.writerow(row)

# ---------- MAIN ----------
def main():
    now = datetime.now(timezone.utc)
    header = f"📡 Pocket Option Signals — {now:%Y-%m-%d %H:%M UTC}\nCandle: 5m | Expiry: {EXPIRY_MIN}m\n"
    lines = [header]

    # NOTE: This minimal version does not fetch markets.
    # It only posts a test pick if MUST_TRADE=1.
    picked = False
    if MUST_TRADE:
        symbol = "EUR/USD"
        side = "BUY"
        entry = "1.10000"
        emoji = "🟢" if side == "BUY" else "🔴"
        lines.append(f"{emoji} {symbol} — {side} @ `{entry}`")
        picked = True

        # Log a proper, evaluatable row
        eval_at = now + timedelta(minutes=EXPIRY_MIN)
        append_signal({
            "ts_utc": now.strftime("%Y-%m-%d %H:%M:%S"),
            "symbol_yf": "EURUSD=X",
            "symbol_pretty": symbol,
            "signal": side,
            "price": entry,
            "expiry_min": EXPIRY_MIN,
            "evaluate_at_utc": eval_at.strftime("%Y-%m-%d %H:%M:%S"),
            "status": "open",
            "score": 1,
            "why": "FORCED-TEST",
            "result_price": "",
            "outcome": ""
        })

    if not picked:
        if SUPPRESS_EMPTY:
            return []  # send nothing
        lines.append("⚪ No setups.")

    lines.append("\n⏱ Build time: `0.0s`")
    return lines

if __name__ == "__main__":
    try:
        out = main() or []
        if out:
            msg = "\n\n".join(out)
            send_to_all_tiers(msg)
        print("✅ Signals run completed.")
    except Exception as e:
        import traceback
        print("❌ Fatal error in signals.py:", e)
        traceback.print_exc()
