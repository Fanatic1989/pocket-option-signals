import os, time, requests
from datetime import datetime, timezone
import signals

now = datetime.now(timezone.utc)
msg = f"🧪 DEBUG PING — {now:%Y-%m-%d %H:%M:%S} UTC\nThis proves runner & Telegram wiring."
print("Sending:\n", msg)

try:
    signals.send_to_tiers(msg)
    print("✅ Sent to tiers (VIP/PRO/BASIC/FREE if set).")
except Exception as e:
    print("send_to_tiers failed:", e)
    try:
        signals.send_telegram(msg)
        print("✅ Fallback send_telegram worked.")
    except Exception as e2:
        print("❌ Could not send:", e2)
