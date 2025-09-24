# pocket_worker_send.py
import os, requests

CORE_SEND_URL = os.getenv("CORE_SEND_URL", "http://127.0.0.1:8000/api/core/send")
CORE_SEND_KEY = os.getenv("CORE_SEND_KEY", "")  # must match app.py env if set

def send_signal_api(tier: str, text: str):
    payload = {"tier": tier.lower(), "text": text}
    headers = {"Content-Type": "application/json"}
    if CORE_SEND_KEY:
        headers["X-API-Key"] = CORE_SEND_KEY
    r = requests.post(CORE_SEND_URL, json=payload, headers=headers, timeout=15)
    try:
        js = r.json()
    except Exception:
        js = {"ok": False, "error": f"HTTP {r.status_code}", "text": r.text[:300]}
    print("PO->API SEND:", js)
    return js

# Example:
# send_signal_api("pro", "EURUSD PUT 1m @ 12:34:56 UTC")
