import os
import traceback
from datetime import datetime, timezone
from telegram_send import send_to_tiers
from scalper import add_indicators, classify
from oanda_utils import oanda_fetch_m1

INTERVAL = os.getenv("INTERVAL", "1m")
EXPIRY_MIN = int(os.getenv("EXPIRY_MIN", "5"))
MIN_SCORE = int(os.getenv("MIN_SCORE", "1"))
SUPPRESS_EMPTY = os.getenv("SUPPRESS_EMPTY", "1") == "1"

def analyze_symbol(symbol: str, pretty: str):
    """Fetch data, run scalper, return a trade signal or None."""
    try:
        df = oanda_fetch_m1(symbol, count=220)
        df = add_indicators(df)
        prev, cur = df.iloc[-2], df.iloc[-1]
        side, score, why = classify(prev, cur, min_score=MIN_SCORE)
        if side and score >= MIN_SCORE:
            price = f"{cur['close']:.5f}"
            emoji = "🟢" if side == "BUY" else "🔴"
            return f"{emoji} {pretty} — {side} @ {price}\n• {why} (score {score})"
    except Exception as e:
        return f"⚠️ {pretty} — error: {e}"
    return None

def main():
    from symbols import SYMBOLS  # ensure SYMBOLS is defined as a list of dicts
    now = datetime.now(timezone.utc)
    header = f"📡 Pocket Option Signals — {now:%Y-%m-%d %H:%M UTC}\nCandle: {INTERVAL} | Expiry: {EXPIRY_MIN}m\n"
    signals = [analyze_symbol(s['oanda'], s['name']) for s in SYMBOLS]
    signals = [s for s in signals if s]  # drop None
    body = "\n\n".join(signals)
    if not signals and SUPPRESS_EMPTY:
        return []
    return [header, body]

if __name__ == "__main__":
    try:
        lines = main() or ["⚠️ No signals generated."]
        msg = "\n\n".join(lines)
        send_to_tiers(msg)
        print("✅ Signals run completed.")
    except Exception as e:
        print("❌ Fatal error:", e)
        traceback.print_exc()
