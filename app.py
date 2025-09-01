# app.py
import os
import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI, Response
from fastapi.responses import PlainTextResponse
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes
from telegram.error import Conflict

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()

telegram_app: Application | None = None
bot_started = False               # <— start guard
bot_polling_task: asyncio.Task | None = None
bot_lock = asyncio.Lock()         # <— prevents concurrent starters

# === Handlers ===
async def start_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("✅ Pocket Option Signals bot is running.")

async def ping_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("pong")

def build_telegram_app() -> Application:
    app = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", start_cmd))
    app.add_handler(CommandHandler("ping", ping_cmd))
    return app

async def _start_bot_polling() -> None:
    """
    Delete any webhook, then start polling in background.
    Using PTB 21.x non-blocking sequence.
    """
    assert telegram_app is not None
    # Ensure no webhook is set (webhook and polling are mutually exclusive)
    try:
        await telegram_app.bot.delete_webhook(drop_pending_updates=True)
        print("🧹 Deleted Telegram webhook (drop_pending_updates=True)")
    except Exception as e:
        print(f"Webhook delete warning: {e}")

    await telegram_app.initialize()
    await telegram_app.start()
    try:
        # Start polling loop in background
        await telegram_app.updater.start_polling(drop_pending_updates=True)
        print("🤖 Telegram bot: polling started")
    except Conflict as e:
        # Another poller somewhere else — log and retry a moment later
        print(f"⚠️ Conflict starting polling: {e}. Retrying in 5s…")
        await asyncio.sleep(5)
        # Try once more after forcing webhook deletion again
        try:
            await telegram_app.bot.delete_webhook(drop_pending_updates=True)
        except Exception:
            pass
        await telegram_app.updater.start_polling(drop_pending_updates=True)
        print("🤖 Telegram bot: polling started after retry")

async def _stop_bot_polling() -> None:
    if telegram_app is None:
        return
    try:
        await telegram_app.updater.stop()
        await telegram_app.stop()
        await telegram_app.shutdown()
        print("🛑 Telegram bot: polling stopped")
    except Exception as e:
        print(f"Bot shutdown error: {e}")

@asynccontextmanager
async def lifespan(app: FastAPI):
    global telegram_app, bot_started, bot_polling_task
    if not TELEGRAM_BOT_TOKEN:
        print("⚠️ TELEGRAM_BOT_TOKEN is empty; bot will NOT start.")
        yield
        return

    async with bot_lock:
        if not bot_started:
            telegram_app = build_telegram_app()
            # Launch the non-blocking starter as a background task
            bot_polling_task = asyncio.create_task(_start_bot_polling())
            bot_started = True
        else:
            print("ℹ️ Bot already started; skipping duplicate start")

    yield  # ---- app running ----

    # Teardown
    try:
        if bot_started and telegram_app is not None:
            await _stop_bot_polling()
            bot_started = False
        if bot_polling_task and not bot_polling_task.done():
            bot_polling_task.cancel()
    except Exception as e:
        print(f"Lifespan shutdown error: {e}")

app = FastAPI(lifespan=lifespan)

@app.get("/", response_class=PlainTextResponse)
async def root():
    return "Pocket Option Signals — OK"

@app.get("/healthz", response_class=PlainTextResponse)
async def healthz_get():
    return "ok"

@app.head("/healthz")
async def healthz_head():
    return Response(status_code=200)
