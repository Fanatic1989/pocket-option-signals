import os
import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Response
from fastapi.responses import PlainTextResponse

from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes
from telegram.error import Conflict

# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()

# Logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("pocket-option-signals")

# Single-process globals
telegram_app: Application | None = None
bot_started: bool = False
bot_polling_task: asyncio.Task | None = None
bot_lock = asyncio.Lock()


# -----------------------------------------------------------------------------
# Telegram Handlers
# -----------------------------------------------------------------------------
async def start_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("✅ Pocket Option Signals bot is running.")

async def ping_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("pong")


def build_telegram_app() -> Application:
    app = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", start_cmd))
    app.add_handler(CommandHandler("ping", ping_cmd))
    # TODO: add your signal commands/handlers here
    return app


# -----------------------------------------------------------------------------
# Bot lifecycle (polling)
# -----------------------------------------------------------------------------
async def _start_bot_polling() -> None:
    """
    Ensure webhook is disabled, then start polling in the background.
    Includes conflict backoff to avoid log storms.
    """
    assert telegram_app is not None

    # Make absolutely sure we are not in webhook mode
    try:
        await telegram_app.bot.delete_webhook(drop_pending_updates=True)
        log.info("🧹 Deleted Telegram webhook (drop_pending_updates=True)")
    except Exception as e:
        log.warning(f"Webhook delete warning: {e}")

    await telegram_app.initialize()
    await telegram_app.start()

    max_attempts = 6  # ~1 minute worst case with backoff
    delay = 2
    for attempt in range(1, max_attempts + 1):
        try:
            await telegram_app.updater.start_polling(drop_pending_updates=True)
            log.info("🤖 Telegram bot: polling started")
            return
        except Conflict as e:
            log.warning(f"⚠️ Conflict starting polling (attempt {attempt}/{max_attempts}): {e}")
            await asyncio.sleep(delay)
            delay = min(delay * 2, 20)

    log.error(
        "🛑 Gave up starting polling due to persistent Conflict. "
        "Another instance is almost certainly running with this token."
    )


async def _stop_bot_polling() -> None:
    if telegram_app is None:
        return
    try:
        await telegram_app.updater.stop()
        await telegram_app.stop()
        await telegram_app.shutdown()
        log.info("🛑 Telegram bot: polling stopped")
    except Exception as e:
        log.error(f"Bot shutdown error: {e}")


# -----------------------------------------------------------------------------
# FastAPI app with lifespan—starts/stops bot inside same process
# -----------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    global telegram_app, bot_started, bot_polling_task

    if not TELEGRAM_BOT_TOKEN:
        log.warning("⚠️ TELEGRAM_BOT_TOKEN is empty; bot will NOT start.")
        yield
        return

    async with bot_lock:
        if not bot_started:
            telegram_app = build_telegram_app()
            bot_polling_task = asyncio.create_task(_start_bot_polling())
            bot_started = True
            log.info("🔌 Bot launch scheduled")
        else:
            log.info("ℹ️ Bot already started; skipping duplicate start")

    yield  # ---- app running ----

    try:
        if bot_started and telegram_app is not None:
            await _stop_bot_polling()
            bot_started = False
        if bot_polling_task and not bot_polling_task.done():
            bot_polling_task.cancel()
    except Exception as e:
        log.error(f"Lifespan shutdown error: {e}")


app = FastAPI(lifespan=lifespan)


# -----------------------------------------------------------------------------
# HTTP Routes (with HEAD for UptimeRobot)
# -----------------------------------------------------------------------------
@app.get("/", response_class=PlainTextResponse)
async def root_get():
    return "Pocket Option Signals — OK"

@app.head("/")
async def root_head():
    return Response(status_code=200)

@app.get("/healthz", response_class=PlainTextResponse)
async def healthz_get():
    return "ok"

@app.head("/healthz")
async def healthz_head():
    return Response(status_code=200)
