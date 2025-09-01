#!/usr/bin/env python3
import os, time, json, sqlite3, requests, argparse
from datetime import datetime, timezone, timedelta

# --- ENV ---
BOT   = os.getenv("TELEGRAM_BOT_TOKEN","")
FREE  = os.getenv("TELEGRAM_CHAT_FREE","")
BASIC = os.getenv("TELEGRAM_CHAT_BASIC","")
PRO   = os.getenv("TELEGRAM_CHAT_PRO","")
VIP   = os.getenv("TELEGRAM_CHAT_VIP","")

NOWPAY_API_KEY = os.getenv("NOWPAY_API_KEY","")
NOWPAY_API = "https://api.nowpayments.io/v1"

# Pricing / durations
TIER_PRICE = {"basic": 29.0, "pro": 49.0, "vip": 79.0}
TIER_DAYS  = {"basic": 30,   "pro": 30,   "vip": 30}

CHAT_BY_TIER = {"basic": BASIC, "pro": PRO, "vip": VIP}

DB = "members.db"

def db():
    cx = sqlite3.connect(DB)
    cx.row_factory = sqlite3.Row
    return cx

def ensure_schema():
    with db() as cx:
        cx.executescript("""
        CREATE TABLE IF NOT EXISTS members(
          user_id    INTEGER PRIMARY KEY,
          username   TEXT,
          tier       TEXT,
          status     TEXT,           -- 'active' | 'expired' | 'pending'
          start_ts   INTEGER,
          expiry_ts  INTEGER,
          last_check INTEGER
        );
        CREATE TABLE IF NOT EXISTS invoices(
          invoice_id TEXT PRIMARY KEY,
          user_id    INTEGER,
          tier       TEXT,
          status     TEXT,           -- nowpayments status
          invoice_url TEXT,
          created_ts INTEGER
        );
        """)
    print("✅ DB ready")

# --- Telegram helpers (HTTP, no PTB dependency) ---
def tg(method, **data):
    if not BOT:
        raise RuntimeError("TELEGRAM_BOT_TOKEN missing")
    url = f"https://api.telegram.org/bot{BOT}/{method}"
    r = requests.post(url, data=data, timeout=30)
    try:
        js = r.json()
    except Exception:
        js = {"ok": False, "error": f"HTTP {r.status_code}"}
    if not js.get("ok"):
        raise RuntimeError(f"Telegram {method} -> {js}")
    return js["result"]

def dm(user_id:int, text:str):
    try:
        tg("sendMessage", chat_id=user_id, text=text)
        return True
    except Exception as e:
        print(f"⚠️ DM to {user_id} failed: {e}")
        return False

def make_invite(chat_id:str, expire_seconds:int=600, creates_join_request:bool=False):
    # create single-use invite link, ~10 minutes validity
    return tg("createChatInviteLink",
              chat_id=chat_id,
              expire_date=int(time.time())+expire_seconds,
              member_limit=1,
              creates_join_request="true" if creates_join_request else "false")

def kick(chat_id:str, user_id:int):
    try:
        tg("banChatMember", chat_id=chat_id, user_id=user_id)
        tg("unbanChatMember", chat_id=chat_id, user_id=user_id)  # optional: allow future rejoin
        print(f"⛔ Removed user {user_id} from {chat_id}")
        return True
    except Exception as e:
        print(f"⚠️ Could not remove {user_id} from {chat_id}: {e}")
        return False

# --- NOWPayments helpers ---
def np_headers():
    if not NOWPAY_API_KEY:
        raise RuntimeError("NOWPAY_API_KEY missing")
    return {"x-api-key": NOWPAY_API_KEY, "Content-Type": "application/json"}

def create_invoice_record(user_id:int, tier:str) -> dict:
    assert tier in TIER_PRICE, "unknown tier"
    body = {
        "price_amount": TIER_PRICE[tier],
        "price_currency": "usd",
        "order_id": f"{tier}:{user_id}:{int(time.time())}",
        "order_description": f"Pocket Signals {tier.upper()} {TIER_DAYS[tier]} days",
        # we won't use webhooks – we poll:
        "success_url": "https://t.me/your_bot?start=success",
        "cancel_url":  "https://t.me/your_bot?start=cancel",
    }
    r = requests.post(f"{NOWPAY_API}/invoice", headers=np_headers(), data=json.dumps(body), timeout=30)
    r.raise_for_status()
    js = r.json()  # has id, invoice_url
    inv_id, url = js["id"], js["invoice_url"]
    with db() as cx:
        cx.execute("INSERT OR REPLACE INTO invoices(invoice_id,user_id,tier,status,invoice_url,created_ts) VALUES(?,?,?,?,?,?)",
                   (inv_id, user_id, tier, "waiting", url, int(time.time())))
        # mark member pending so we know they’re in the funnel
        cx.execute("INSERT OR IGNORE INTO members(user_id, username, tier, status, start_ts, expiry_ts, last_check) VALUES(?,?,?,?,?,?,?)",
                   (user_id, None, tier, "pending", None, None, int(time.time())))
    return js

def check_invoice(invoice_id:str) -> dict:
    r = requests.get(f"{NOWPAY_API}/invoice/{invoice_id}", headers=np_headers(), timeout=30)
    r.raise_for_status()
    return r.json()

# --- Activation / expiry ---
def activate_member(user_id:int, tier:str):
    days = TIER_DAYS[tier]
    now = int(time.time())
    exp  = now + days*86400
    with db() as cx:
        cx.execute("""INSERT INTO members(user_id,username,tier,status,start_ts,expiry_ts,last_check)
                      VALUES(?,?,?,?,?,?,?)
                      ON CONFLICT(user_id) DO UPDATE SET
                        tier=?,
                        status='active',
                        start_ts=COALESCE(members.start_ts, ?),
                        expiry_ts=?,
                        last_check=?""",
                   (user_id,None,tier,'active',now,exp,now,
                    tier, now, exp, now))
    chat_id = CHAT_BY_TIER.get(tier)
    if not chat_id:
        print(f"⚠️ No chat configured for tier {tier}")
        return
    link = make_invite(chat_id, expire_seconds=600)
    dm(user_id, f"🎉 Payment confirmed!\nHere’s your {tier.upper()} invite (valid ~10 min, one use):\n{link['invite_link']}\n\nTap to join. If it expires, ask me again.")
    print(f"✅ Activated {user_id} -> {tier}")

def remove_expired():
    now = int(time.time())
    with db() as cx:
        rows = cx.execute("SELECT user_id, tier, expiry_ts FROM members WHERE status='active'").fetchall()
    cnt = 0
    for r in rows:
        if r["expiry_ts"] and r["expiry_ts"] < now:
            chat_id = CHAT_BY_TIER.get(r["tier"])
            if chat_id:
                kick(chat_id, r["user_id"])
            with db() as cx:
                cx.execute("UPDATE members SET status='expired', last_check=? WHERE user_id=?", (now, r["user_id"]))
            cnt += 1
    if cnt:
        print(f"🔻 Expired removed: {cnt}")

def poll_invoices():
    with db() as cx:
        rows = cx.execute("SELECT invoice_id, user_id, tier, status FROM invoices WHERE status NOT IN ('finished','confirmed','failed','expired')").fetchall()
    if not rows:
        return
    for r in rows:
        try:
            js = check_invoice(r["invoice_id"])
            st = js.get("payment_status","")
            if st and st != r["status"]:
                with db() as cx:
                    cx.execute("UPDATE invoices SET status=? WHERE invoice_id=?", (st, r["invoice_id"]))
                print(f"ℹ️ invoice {r['invoice_id']} -> {st}")
            if st in ("finished","confirmed"):
                activate_member(r["user_id"], r["tier"])
        except Exception as e:
            print(f"⚠️ invoice {r['invoice_id']} error: {e}")

# --- CLI ---
def main():
    ensure_schema()
    ap = argparse.ArgumentParser()
    ap.add_argument("--create-invoice", action="store_true", help="Create invoice for a user/tier")
    ap.add_argument("--user", type=int, help="Telegram user id")
    ap.add_argument("--tier", choices=("basic","pro","vip"), help="Tier")
    ap.add_argument("--once", action="store_true", help="Run one maintenance pass and exit")
    args = ap.parse_args()

    if args.create_invoice:
        if not (args.user and args.tier):
            raise SystemExit("--create-invoice needs --user and --tier")
        inv = create_invoice_record(args.user, args.tier)
        url = inv["invoice_url"]
        # try to DM the user the link (works if user started the bot)
        dm(args.user, f"💳 Pay for {args.tier.upper()} here:\n{url}\n\nI’ll auto-activate you when payment confirms.")
        print(f"✅ Invoice created: {inv['id']} -> {url}")
        return

    # Maintenance pass (poll payments + expire members)
    poll_invoices()
    remove_expired()

    if args.once:
        return

    # If not --once, loop forever every 60s (local/service mode)
    while True:
        time.sleep(60)
        poll_invoices()
        remove_expired()

if __name__ == "__main__":
    main()
