#!/usr/bin/env python3
"""
run_and_send.py
- Builds the message via signals_live.main()
- Keeps only confirmed trades (🟢/🔴 + their reason line)
- Suppresses empty posts when SUPPRESS_EMPTY=1
- Sends to all tiers
"""
import os
import traceback
from typing import List
from datetime import datetime, timezone

from signals_live import main as build_message
from telegram_send import send_to_tiers

def send_confirmed(text: str) -> str | None:
    """
    Keep header (first 2–3 lines) and only confirmed trades:
    lines starting with 🟢/🔴 plus the following bullet (• ...).
    Return None if nothing confirmed and SUPPRESS_EMPTY=1.
    """
    if not isinstance(text, str):
        return None
    lines: List[str] = text.splitlines()
    keep: List[str] = []
    next_is_reason = False

    # keep first 3 lines as header if present
    header_n = 3 if len(lines) >= 3 else len(lines)
    for i, ln in enumerate(lines):
        if i < header_n:
            keep.append(ln)
            continue
        s = ln.lstrip()
        if s.startswith("🟢") or s.startswith("🔴"):
            keep.append(ln)
            next_is_reason = True
            continue
        if next_is_reason and s.startswith("•"):
            keep.append(ln)
            keep.append("")  # spacer
            next_is_reason = False
            continue

    has_confirmed = any(l.lstrip().startswith(("🟢","🔴")) for l in keep[header_n:])
    suppress = os.getenv("SUPPRESS_EMPTY", "1") == "1"
    if not has_confirmed and suppress:
        return None
    return "\n".join(keep).strip()

def main():
    try:
        lines = build_message() or []
        msg = "\n".join(lines).strip()
        filtered = send_confirmed(msg)
        if not filtered:
            print("✋ suppressed empty (no confirmed signals).")
            return
        send_to_tiers(filtered)
        print("✅ sent.")
    except Exception as e:
        print("❌ run_and_send.py fatal:", e)
        traceback.print_exc()

if __name__ == "__main__":
    main()
