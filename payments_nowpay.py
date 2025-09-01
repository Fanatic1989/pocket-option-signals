import os, requests, time

API = "https://api.nowpayments.io/v1"
KEY = os.getenv("NOWPAY_API_KEY")
HDR = {"x-api-key": KEY, "Content-Type": "application/json"}

TIER_PRICE = {"basic": 29.0, "pro": 49.0, "vip": 79.0}
DAYS = {"basic": 30, "pro": 30, "vip": 30}

def _check_key():
    if not KEY:
        raise RuntimeError("NOWPAY_API_KEY missing (set it in .env or GitHub Secrets).")

def create_invoice(user_id: int, tier: str) -> dict:
    _check_key()
    assert tier in TIER_PRICE, "unknown tier"
    body = {
        "price_amount": TIER_PRICE[tier],
        "price_currency": "usd",
        "order_id": f"{tier}:{user_id}:{int(time.time())}",
        "order_description": f"Pocket Signals {tier.upper()} {DAYS[tier]} days",
        "success_url": "https://t.me/your_bot?start=success",
        "cancel_url": "https://t.me/your_bot?start=cancel",
    }
    r = requests.post(f"{API}/invoice", json=body, headers=HDR, timeout=30)
    r.raise_for_status()
    return r.json()  # includes: id, invoice_url

def get_payment_status(invoice_id: str) -> dict:
    _check_key()
    r = requests.get(f"{API}/invoice/{invoice_id}", headers=HDR, timeout=30)
    r.raise_for_status()
    return r.json()   # has payment_status
