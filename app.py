# app.py
import os
import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI, Response
from fastapi.responses import PlainTextResponse

# --- Telegram bot (python-telegram-bot v21.x) ---
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()

# Build the bot application once (so we can start/stop inside FastAPI lifespan)
telegram_app: Application | None = None
bot_task: asyncio.Task | None = None

# === Handlers ===
async def start_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("✅ Pocket Option Signals bot is running.")

async def ping_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("pong")

def build_telegram_app() -> Application:
    app = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", start_cmd))
    app.add_handler(CommandHandler("ping", ping_cmd))
    # Add your other handlers here
    return app

async def _start_bot_polling() -> None:
    """
    Initialize + start polling without blocking the FastAPI event loop.
    """
    assert telegram_app is not None, "telegram_app not built"
    await telegram_app.initialize()
    await telegram_app.start()
    # Start polling: this runs internal polling loop in background
    await telegram_app.updater.start_polling(drop_pending_updates=True)

async def _stop_bot_polling() -> None:
    if telegram_app is None:
        return
    # Stop polling first, then stop app
    await telegram_app.updater.stop()
    await telegram_app.stop()
    await telegram_app.shutdown()

# --- FastAPI app with lifespan for startup/shutdown wiring ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    global telegram_app, bot_task

    # Basic guard for missing token
    if not TELEGRAM_BOT_TOKEN:
        print("⚠️ TELEGRAM_BOT_TOKEN is empty; bot will NOT start.")

    # Build once (even if token missing, we still serve web)
    if TELEGRAM_BOT_TOKEN:
        telegram_app = build_telegram_app()
        # Launch bot polling as a background task
        bot_task = asyncio.create_task(_start_bot_polling())
        print("🤖 Telegram bot: starting polling...")

    yield  # ---- Application is running ----

    # Teardown
    try:
        if TELEGRAM_BOT_TOKEN and telegram_app is not None:
            print("🛑 Telegram bot: stopping polling...")
            await _stop_bot_polling()
        if bot_task and not bot_task.done():
            bot_task.cancel()
    except Exception as e:
        print(f"Bot shutdown error: {e}")

app = FastAPI(lifespan=lifespan)

# --- Health + root routes ---
@app.get("/", response_class=PlainTextResponse)
async def root():
    return "Pocket Option Signals — OK"

@app.get("/healthz", response_class=PlainTextResponse)
async def healthz_get():
    """
    GET health for browsers & Render health checks.
    """
    return "ok"

@app.head("/healthz")
async def healthz_head():
    """
    HEAD health for UptimeRobot (returns 200 with empty body).
    """
    return Response(status_code=200)
