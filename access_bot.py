#!/usr/bin/env python3
"""
Telegram access bot:
- /buy_basic, /buy_pro, /buy_vip -> creates NOWPayments invoice and DM’s link
- Polls unpaid invoices every 60s; activates on paid (confirmed/finished)
- /claim <tier> -> fresh single-use invite for paid users
- Auto-removes expired members every 60s
Deploy as a background worker (Render, etc.):  python access_bot.py
"""
import os, time, sqlite3, json, requests, asyncio
from datetime import datetime, timedelta, timezone
from typing import Optional, Tuple

from telegram import Update, ChatInviteLink
from telegram.ext import Application, CommandHandler, ContextTypes

# --- ENV ---
BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
FREE  = os.getenv("TELEGRAM_CHAT_FREE",  "")
BASIC = os.getenv("TELEGRAM_CHAT_BASIC", "")
PRO   = os.getenv("TELEGRAM_CHAT_PRO",   "")
VIP   = os.getenv("TELEGRAM_CHAT_VIP",   "")
NOWPAY_API_KEY = os.getenv("NOWPAY_API_KEY", "")

# Render-friendly: tighten I/O
os.environ.setdefault("PYTHONUNBUFFERED","1")

# --- TIERS / PRICING ---
PRICES = {"basic": 29.0, "pro": 49.0, "vip": 79.0}   # USD
DUR_DAYS = {"basic": 30, "pro": 30, "vip": 30}
TIER_CHAT = {"basic": BASIC, "pro": PRO, "vip": VIP}

# --- NOWPayments REST ---
NP_API = "https://api.nowpayments.io/v1"
NP_HDR = {"x-api-key": NOWPAY_API_KEY, "Content-Type": "application/json"}

# --- DB ---
DB_PATH = os.getenv("MEMBER_DB", "members.db")

def db():
    return sqlite3.connect(DB_PATH)

def init_db():
    with db() as cx:
        cx.execute("""CREATE TABLE IF NOT EXISTS members(
            user_id INTEGER PRIMARY KEY,
            username TEXT,
            tier TEXT,
            status TEXT,            -- 'active' | 'expired' | 'pending'
            start_ts INTEGER,
            expiry_ts INTEGER,
            last_check INTEGER
        );""")
        cx.execute("""CREATE TABLE IF NOT EXISTS invoices(
            id TEXT PRIMARY KEY,    -- invoice_id from NOWPayments
            user_id INTEGER,
            tier TEXT,
            status TEXT,            -- waiting/confirming/confirmed/finished/expired/failed
            created_ts INTEGER
        );""")

def upsert_member(user_id:int, username:str, tier:str, status:str, days:int):
    now = int(time.time())
    exp = now + days*86400 if days else None
    with db() as cx:
        cx.execute("""INSERT INTO members(user_id,username,tier,status,start_ts,expiry_ts,last_check)
                      VALUES(?,?,?,?,?,?,?)
                      ON CONFLICT(user_id) DO UPDATE SET
                        username=excluded.username,
                        tier=excluded.tier,
                        status=excluded.status,
                        start_ts=COALESCE(members.start_ts,excluded.start_ts),
                        expiry_ts=excluded.expiry_ts,
                        last_check=excluded.last_check""",
                    (user_id, username, tier, status, now, exp, now))

def set_member_expired(user_id:int):
    with db() as cx:
        cx.execute("UPDATE members SET status='expired' WHERE user_id=?", (user_id,))

def get_member(user_id:int)->Optional[Tuple]:
    with db() as cx:
        cur = cx.execute("SELECT user_id, username, tier, status, start_ts, expiry_ts FROM members WHERE user_id=?", (user_id,))
        return cur.fetchone()

def save_invoice(invoice_id:str, user_id:int, tier:str, status:str):
    with db() as cx:
        cx.execute("""INSERT INTO invoices(id,user_id,tier,status,created_ts)
                      VALUES(?,?,?,?,?)
                      ON CONFLICT(id) DO UPDATE SET status=excluded.status""",
                   (invoice_id, user_id, tier, status, int(time.time())))

def unpaid_invoices()->list[Tuple[str,int,str]]:
    with db() as cx:
        cur = cx.execute("SELECT id,user_id,tier FROM invoices WHERE status IN ('waiting','confirming')")
        return cur.fetchall()

def mark_invoice_status(invoice_id:str, status:str):
    with db() as cx:
        cx.execute("UPDATE invoices SET status=? WHERE id=?", (status, invoice_id))

# --- NOWPayments helpers (polling) ---
def create_invoice(user_id:int, tier:str) -> dict:
    assert tier in PRICES, "unknown tier"
    body = {
        "price_amount": PRICES[tier],
        "price_currency": "usd",
        "order_id": f"{tier}:{user_id}:{int(time.time())}",
        "order_description": f"Pocket Signals {tier.upper()} {DUR_DAYS[tier]} days",
        # These URLs are optional (no webhook). User pays on NOWPayments page and returns to Telegram.
        "success_url": "https://t.me/",
        "cancel_url": "https://t.me/"
    }
    r = requests.post(f"{NP_API}/invoice", json=body, headers=NP_HDR, timeout=30)
    r.raise_for_status()
    return r.json()

def get_invoice(invoice_id:str) -> dict:
    r = requests.get(f"{NP_API}/invoice/{invoice_id}", headers=NP_HDR, timeout=30)
    r.raise_for_status()
    return r.json()

# --- Telegram helpers ---
async def send_dm(update:Update, text:str):
    try:
        await update.effective_chat.send_message(text)
    except Exception:
        # If command was issued in a group, fallback to replying there
        await update.message.reply_text(text)

async def make_one_use_link(context:ContextTypes.DEFAULT_TYPE, chat_id:str, expire_sec:int=600) -> ChatInviteLink:
    # requires the bot to be admin in that group/channel
    return await context.bot.create_chat_invite_link(
        chat_id=chat_id,
        expire_date=int(time.time()) + expire_sec,
        member_limit=1,
        creates_join_request=False,
        name=f"auto-{int(time.time())}"
    )

async def kick_user(context:ContextTypes.DEFAULT_TYPE, chat_id:str, user_id:int):
    try:
        await context.bot.ban_chat_member(chat_id, user_id)
        await context.bot.unban_chat_member(chat_id, user_id)  # allow rejoin later when paid
        print(f"⛔ removed {user_id} from {chat_id}")
    except Exception as e:
        print(f"⚠️ remove failed {user_id} from {chat_id}: {e}")

# --- Commands ---
async def start_cmd(update:Update, context:ContextTypes.DEFAULT_TYPE):
    text = (
        "Hey! 👋\n"
        "Choose a tier to subscribe:\n"
        f"• /buy_basic  — ${PRICES['basic']:.0f} / {DUR_DAYS['basic']} days\n"
        f"• /buy_pro    — ${PRICES['pro']:.0f} / {DUR_DAYS['pro']} days\n"
        f"• /buy_vip    — ${PRICES['vip']:.0f} / {DUR_DAYS['vip']} days\n\n"
        "Already paid? Use /claim <tier> to get your invite link (e.g. /claim vip)."
    )
    await send_dm(update, text)

async def _buy(update:Update, context:ContextTypes.DEFAULT_TYPE, tier:str):
    u = update.effective_user
    if not NOWPAY_API_KEY:
        await send_dm(update, "⚠️ Payments disabled (NOWPAY_API_KEY missing).")
        return
    try:
        inv = create_invoice(u.id, tier)
        invoice_id = inv.get("id")
        url = inv.get("invoice_url")
        if not invoice_id or not url:
            raise RuntimeError(f"bad invoice response: {inv}")
        save_invoice(invoice_id, u.id, tier, "waiting")
        upsert_member(u.id, u.username or "", tier, "pending", 0)
        await send_dm(update, f"💳 {tier.upper()} invoice created:\n{url}\n\n"
                              f"After paying, return and run /claim {tier}.\n"
                              f"(We also auto-check payments every 60s.)")
    except Exception as e:
        await send_dm(update, f"❌ Could not create invoice: {e}")

async def buy_basic(update:Update, ctx): return await _buy(update, ctx, "basic")
async def buy_pro(update:Update, ctx):   return await _buy(update, ctx, "pro")
async def buy_vip(update:Update, ctx):   return await _buy(update, ctx, "vip")

async def claim(update:Update, context:ContextTypes.DEFAULT_TYPE):
    if len(context.args) != 1 or context.args[0].lower() not in TIER_CHAT:
        await send_dm(update, "Usage: /claim <basic|pro|vip>")
        return
    tier = context.args[0].lower()
    chat_id = TIER_CHAT[tier]
    if not chat_id:
        await send_dm(update, f"Tier {tier} not configured.")
        return

    u = update.effective_user
    m = get_member(u.id)

    # If not yet activated, let the user know we’re still waiting
    if not m or m[3] != "active":
        await send_dm(update, f"⏳ Payment not confirmed yet for {tier.upper()}.\n"
                              f"If you just paid, give it up to a minute, then /claim {tier} again.")
        return

    try:
        link = await make_one_use_link(context, chat_id)
        await send_dm(update, f"✅ Here’s your {tier.upper()} invite (valid ~10 min, one-use):\n{link.invite_link}\n\n"
                              f"Tap to join. If it expires, run /claim {tier} again.")
    except Exception as e:
        await send_dm(update, f"⚠️ Could not create invite link: {e}")

# --- Background jobs ---
async def job_poll_invoices(context:ContextTypes.DEFAULT_TYPE):
    """Poll NOWPayments invoices; activate members on paid."""
    rows = unpaid_invoices()
    if not rows:
        return
    for invoice_id, user_id, tier in rows:
        try:
            data = get_invoice(invoice_id)
            status = (data.get("payment_status") or data.get("invoice_status") or "").lower()
            if not status:
                continue
            mark_invoice_status(invoice_id, status)
            if status in ("confirmed", "finished"):
                # Activate membership
                upsert_member(user_id, "", tier, "active", DUR_DAYS[tier])
                print(f"✅ activated user {user_id} -> {tier}")
        except Exception as e:
            print(f"poll invoice {invoice_id} failed: {e}")

async def job_expire_and_kick(context:ContextTypes.DEFAULT_TYPE):
    """Expire overdue members and remove them from their tier chat."""
    now = int(time.time())
    with db() as cx:
        cur = cx.execute("SELECT user_id, tier, expiry_ts FROM members WHERE status='active'")
        rows = cur.fetchall()
    for user_id, tier, expiry_ts in rows:
        if expiry_ts and expiry_ts < now:
            chat_id = TIER_CHAT.get(tier)
            if chat_id:
                await kick_user(context, chat_id, user_id)
            set_member_expired(user_id)

# --- main ---
def main():
    if not BOT_TOKEN:
        raise SystemExit("TELEGRAM_BOT_TOKEN not set.")
    init_db()

    app = Application.builder().token(BOT_TOKEN).build()

    app.add_handler(CommandHandler("start", start_cmd))
    app.add_handler(CommandHandler("buy_basic", buy_basic))
    app.add_handler(CommandHandler("buy_pro", buy_pro))
    app.add_handler(CommandHandler("buy_vip", buy_vip))
    app.add_handler(CommandHandler("claim", claim))

    # every 60s: check payments + kick expired
    app.job_queue.run_repeating(job_poll_invoices, interval=60, first=10)
    app.job_queue.run_repeating(job_expire_and_kick, interval=60, first=20)

    print("✅ access bot is running.")
    app.run_polling(close_loop=False)

if __name__ == "__main__":
    main()
