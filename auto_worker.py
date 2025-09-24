# auto_worker.py
# Fully automated loop:
# - pulls/generates signals
# - respects daily caps via /api/core/status
# - rate-limits, backoff, retries
# Env:
#   CORE_SEND_URL, CORE_SEND_KEY
#   WORKER_TIER=vip|pro|basic|free (default: vip)
#   WORKER_MIN_SEC_BETWEEN_SENDS=5
#   WORKER_MAX_PER_MIN=20
#   WORKER_IDLE_SEC=10 (sleep when nothing to send)

import os, time, json, math, random, requests
from datetime import datetime, timezone

CORE_SEND_URL  = os.getenv("CORE_SEND_URL",  "http://127.0.0.1:8000/api/core/send").strip()
CORE_STATUS_URL= CORE_SEND_URL.replace("/send", "/status")
CORE_SEND_KEY  = os.getenv("CORE_SEND_KEY", "").strip()

WORKER_TIER    = (os.getenv("WORKER_TIER", "vip") or "vip").lower()
MIN_GAP_SEC    = int(os.getenv("WORKER_MIN_SEC_BETWEEN_SENDS", "5"))
MAX_PER_MIN    = int(os.getenv("WORKER_MAX_PER_MIN", "20"))
IDLE_SEC       = int(os.getenv("WORKER_IDLE_SEC", "10"))

HEADERS = {"Content-Type":"application/json"}
if CORE_SEND_KEY:
    HEADERS["X-API-Key"] = CORE_SEND_KEY

# ----------------- your signal generator hook -----------------
# Replace this with your real discovery logic (PO, indicators, etc.)
def discover_signals():
    """
    Return a list of (tier, text) to send right now.
    Example integrates your Pocket Options logic; for now we emit a sample
    only once per minute for demo.
    """
    utc = datetime.now(timezone.utc)
    if utc.second < 3:  # e.g., one signal near top of minute
        return [(WORKER_TIER, f"[AUTO] Heartbeat {utc.isoformat(timespec='seconds')}Z")]
    return []

# ----------------- helpers -----------------
def get_status():
    try:
        r = requests.get(CORE_STATUS_URL, timeout=10)
        return r.json()
    except Exception as e:
        return {"ok": False, "error": str(e)}

def can_send_to_tier(status, tier):
    caps   = status.get("caps", {})
    tally  = (status.get("tally") or {}).get("by_tier", {})
    cap    = caps.get(tier)  # None => unlimited
    used   = tally.get(tier, 0)
    if cap is None:
        return True
    return used < cap

def post_signal(tier, text):
    payload = {"tier": tier, "text": text}
    try:
        r = requests.post(CORE_SEND_URL, json=payload, headers=HEADERS, timeout=15)
        try:
            return r.json()
        except Exception:
            return {"ok": False, "error": f"HTTP {r.status_code}", "text": r.text[:300]}
    except Exception as e:
        return {"ok": False, "error": str(e)}

# Simple send-rate guard
_last_sends = []

def throttle_ok():
    # evict old timestamps >60s
    now = time.time()
    while _last_sends and now - _last_sends[0] > 60:
        _last_sends.pop(0)
    # limit per minute
    if len(_last_sends) >= MAX_PER_MIN:
        return False
    # min gap
    if _last_sends and (now - _last_sends[-1] < MIN_GAP_SEC):
        return False
    return True

def mark_sent():
    _last_sends.append(time.time())

# ----------------- main loop -----------------
def main():
    print(f"[WORKER] starting | tier={WORKER_TIER} url={CORE_SEND_URL}")
    backoff = 2
    while True:
        status = get_status()
        if not status or status.get("running") is not True:
            print("[WORKER] engine not running or status error; retry soon:", status)
            time.sleep(backoff)
            backoff = min(backoff * 2, 30)
            continue
        backoff = 2

        # Respect tier caps
        if not can_send_to_tier(status, WORKER_TIER):
            # hit cap; idle to next minute (engine will auto reset next day)
            print(f"[WORKER] cap reached for {WORKER_TIER}; sleeping 60s")
            time.sleep(60)
            continue

        signals = discover_signals()

        sent_this_cycle = 0
        for (tier, text) in signals:
            tier = (tier or WORKER_TIER).lower()
            if not can_send_to_tier(status, tier):
                print(f"[WORKER] cap reached for {tier}; skipping signal")
                continue
            if not throttle_ok():
                # wait briefly, then retry this signal once
                wait = max(0.5, MIN_GAP_SEC - (time.time() - (_last_sends[-1] if _last_sends else 0)))
                print(f"[WORKER] throttling {wait:.1f}s")
                time.sleep(wait)
                if not throttle_ok():
                    print("[WORKER] still throttled; skipping")
                    continue

            res = post_signal(tier, text)
            print("[WORKER] SEND =>", json.dumps(res))
            if res.get("ok") or (isinstance(res.get("results"), dict) and list(res["results"].values())[0].get("ok")):
                mark_sent()
                sent_this_cycle += 1
                # refresh status to update tallies
                status = get_status()
            else:
                # If Telegram/engine rejected, log and continue
                time.sleep(1)

        # Idle if nothing to send
        if sent_this_cycle == 0 and not signals:
            time.sleep(IDLE_SEC)
        else:
            # small jitter to avoid schedule collisions
            time.sleep(1 + random.random()*2)

if __name__ == "__main__":
    main()
