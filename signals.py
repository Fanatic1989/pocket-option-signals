try:
    import orb
except ModuleNotFoundError:
    print("⚠️ ORB module not found, skipping ORB strategy.")
    orb=None

import os
from datetime import datetime, timezone
from telegram_send import send_to_tiers

INTERVAL   = os.getenv("INTERVAL", "5m")
EXPIRY_MIN = int(os.getenv("EXPIRY_MIN", "10"))

def main():
    now = datetime.now(timezone.utc)
    lines = [
        f"📡 Pocket Option Signals — {now:%Y-%m-%d %H:%M UTC}",
        f"Candle: {INTERVAL} | Expiry: {EXPIRY_MIN}m",
        "",
        "🧪 TEST — pipeline sanity message (plain text)."
    ]
    return lines

if __name__ == "__main__":
    lines = main() or ["⚠️ No signals generated."]
    msg = "\n".join(lines)
    send_to_tiers(msg)
    print("✅ Signals run completed.")
