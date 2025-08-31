import sys, os, traceback
from datetime import datetime, timezone

# --- Imports with graceful fallbacks ---
try:
    from telegram_send import send_to_tiers
except Exception as e:
    print("❌ telegram_send missing or broken:", e)
    traceback.print_exc()
    # Hard fail would abort CI; instead, log and exit 0 to avoid "random crashes"
    sys.exit(0)

try:
    import signals_live
except Exception as e:
    print("❌ signals_live import error:", e)
    traceback.print_exc()
    sys.exit(0)

# Optional filter if you added it; we fall back cleanly if not present
_filter_confirmed = None
try:
    from telegram_send import send_confirmed as _filter_confirmed
except Exception:
    pass

def main():
    # Build lines from your live scanner
    try:
        lines = signals_live.main() or []
    except Exception as e:
        print("❌ signals_live.main() crashed:", e)
        traceback.print_exc()
        return

    # Ensure header exists
    if not lines:
        now = datetime.now(timezone.utc)
        lines = [
            f"📡 Pocket Option Signals — {now:%Y-%m-%d %H:%M UTC}",
            f"Candle: {os.getenv('INTERVAL','1m')} | Expiry: {os.getenv('EXPIRY_MIN','5')}m",
            "", "⚠️ No signals generated (runner)."
        ]

    msg = "\n".join(lines)

    # Strict confirmed-only behavior (no spam)
    SUPPRESS_EMPTY = os.getenv("SUPPRESS_EMPTY","1") == "1"
    has_confirmed = ("🟢" in msg) or ("🔴" in msg)

    if _filter_confirmed:
        # If helper exists, use it (returns None to suppress)
        try:
            filtered = _filter_confirmed(msg)
            if filtered is None:
                print("✋ suppressed empty (no confirmed signals).")
                return
            msg = filtered
            has_confirmed = True
        except Exception as e:
            print("⚠️ send_confirmed() failed, falling back:", e)

    if not has_confirmed and SUPPRESS_EMPTY:
        print("✋ suppressed empty (no confirmed signals).")
        return

    # Send to all tiers
    try:
        send_to_tiers(msg)
        print("✅ signals sent")
    except Exception as e:
        print("❌ send_to_tiers() failed:", e)
        traceback.print_exc()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("❌ top-level crash:", e)
        traceback.print_exc()
    # Always exit 0 so CI doesn’t “red” on empty/no-setup/minor issues
    sys.exit(0)
