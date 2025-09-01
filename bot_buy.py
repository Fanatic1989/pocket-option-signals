#!/usr/bin/env python3
import os, subprocess
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes

BOT = os.getenv("TELEGRAM_BOT_TOKEN","")
TIERS = {"buy_basic":"basic","buy_pro":"pro","buy_vip":"vip"}

async def start(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    txt = (
        "Hey! 👋\n"
        "Choose a tier to subscribe:\n"
        "• /buy_basic — $29 / 30 days\n"
        "• /buy_pro   — $49 / 30 days\n"
        "• /buy_vip   — $79 / 30 days\n\n"
        "After paying, you’ll be added automatically."
    )
    await update.message.reply_text(txt)

async def buy(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    cmd = update.message.text.strip().lstrip("/")
    tier = TIERS.get(cmd)
    if not tier:
        return
    user = update.effective_user.id
    # call manage_members.py to create & save invoice (returns invoice_url on stdout via print)
    try:
        out = subprocess.check_output(
            ["python","manage_members.py","--create-invoice","--user",str(user),"--tier",tier],
            stderr=subprocess.STDOUT, text=True
        )
        # manage_members.py already DMs the link; echo a confirmation:
        await update.message.reply_text("✅ Invoice created — check your DM for the payment link.")
    except subprocess.CalledProcessError as e:
        await update.message.reply_text(f"⚠️ Couldn’t create invoice: {e.output[-300:]}")

def main():
    if not BOT:
        raise SystemExit("TELEGRAM_BOT_TOKEN is missing in env.")
    app = Application.builder().token(BOT).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler(["buy_basic","buy_pro","buy_vip"], buy))
    app.run_polling(close_loop=False)

if __name__ == "__main__":
    main()
