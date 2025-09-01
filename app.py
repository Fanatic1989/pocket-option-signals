import os
import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Response, Request, Header, HTTPException
from fastapi.responses import PlainTextResponse

from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes

# =============================================================================
# Config (set these in Render → Environment)
# =============================================================================
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
BASE_URL = os.getenv("BASE_URL", "").rstrip("/")  # e.g. https://pocket-option-signals-hup9.onrender.com
WEBHOOK_PATH = os.getenv("WEBHOOK_PATH", "/telegram/webhook").strip()  # route path (no trailing slash)
WEBHOOK_SECRET = os.getenv("WEBHOOK_SECRET", "change-me-secret").strip()  # any random string
ENABLE_BOT_POLLING = os.getenv("ENABLE_BOT_POLLING", "0").strip()  # must be "0" for webhook mode

# Logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("pocket-option-signals")

# Globals
telegram_app: Application | None = None

# =============================================================================
# Telegram Handlers
# =============================================================================
async def start_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("✅ Pocket Option Signals bot (webhook mode) is running.")

async def ping_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("pong")

def build_telegram_app() -> Application:
    app = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", start_cmd))
    app.add_handler(CommandHandler("ping", ping_cmd))
    # TODO: add your command/handlers here
    return app

# =============================================================================
# Webhook lifecycle
# =============================================================================
async def _set_webhook():
    assert telegram_app is not None
    if not BASE_URL:
        raise RuntimeError("BASE_URL env var is required for webhook mode (your public Render URL).")

    url = f"{BASE_URL}{WEBHOOK_PATH}"
    # Ensure fresh start
    try:
        await telegram_app.bot.delete_webhook(drop_pending_updates=True)
        log.info("🧹 Deleted existing webhook (drop_pending_updates=True)")
    except Exception as e:
        log.warning(f"Webhook delete warning: {e}")

    await telegram_app.initialize()
    await telegram_app.start()
    # Set webhook
    await telegram_app.bot.set_webhook(
        url=url,
        secret_token=WEBHOOK_SECRET,
        drop_pending_updates=True,
        allowed_updates=None,  # receive all update types
    )
    log.info(f"🔗 Webhook set to: {url}")

async def _unset_webhook():
    if telegram_app is None:
        return
    try:
        await telegram_app.bot.delete_webhook(drop_pending_updates=False)
        log.info("🧹 Webhook removed")
    except Exception as e:
        log.warning(f"Webhook removal warning: {e}")
    try:
        await telegram_app.stop()
        await telegram_app.shutdown()
        log.info("🛑 Telegram app stopped")
    except Exception as e:
        log.error(f"Shutdown error: {e}")

# =============================================================================
# FastAPI (with lifespan)
# =============================================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    global telegram_app

    if not TELEGRAM_BOT_TOKEN:
        log.error("❌ TELEGRAM_BOT_TOKEN is missing. Bot will not start.")
        yield
        return

    if ENABLE_BOT_POLLING == "1":
        log.warning("⚠️ ENABLE_BOT_POLLING='1' but webhook mode is configured. For webhook mode set it to '0'.")
    telegram_app = build_telegram_app()
    await _set_webhook()

    yield  # App is running

    await _unset_webhook()

app = FastAPI(lifespan=lifespan)

# =============================================================================
# HTTP Routes (health + webhook + uptime HEADs)
# =============================================================================
@app.get("/", response_class=PlainTextResponse)
async def root_get():
    return "Pocket Option Signals — OK (webhook mode)"

@app.head("/")
async def root_head():
    return Response(status_code=200)

@app.get("/healthz", response_class=PlainTextResponse)
async def healthz_get():
    return "ok"

@app.head("/healthz")
async def healthz_head():
    return Response(status_code=200)

# Telegram webhook receiver
@app.post(WEBHOOK_PATH)
async def telegram_webhook(
    request: Request,
    x_telegram_bot_api_secret_token: str = Header(None),
):
    # Verify Telegram's secret header matches ours (prevents random posts)
    if WEBHOOK_SECRET and x_telegram_bot_api_secret_token != WEBHOOK_SECRET:
        raise HTTPException(status_code=403, detail="Forbidden")

    if telegram_app is None:
        raise HTTPException(status_code=503, detail="Bot not initialized")

    data = await request.json()
    update = Update.de_json(data, telegram_app.bot)
    await telegram_app.process_update(update)
    return {"ok": True}
