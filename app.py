import os
import asyncio
import logging
from datetime import datetime, timedelta, timezone
from contextlib import asynccontextmanager
from typing import Dict, List, Any, Optional

import pandas as pd
import pandas_ta as ta
import httpx

from fastapi import FastAPI, Response, Request, Header, HTTPException
from fastapi.responses import PlainTextResponse, JSONResponse

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes

# =============================================================================
# ENV
# =============================================================================
# Telegram / Webhook
TELEGRAM_BOT_TOKEN   = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
TELEGRAM_GROUP_ID    = os.getenv("TELEGRAM_GROUP_ID", "").strip()
BASE_URL             = os.getenv("BASE_URL", "").rstrip("/")
WEBHOOK_PATH         = os.getenv("WEBHOOK_PATH", "/telegram/webhook").strip()
WEBHOOK_SECRET       = os.getenv("WEBHOOK_SECRET", "change-me").strip()
ENABLE_BOT_POLLING   = os.getenv("ENABLE_BOT_POLLING", "0").strip()  # keep 0 for webhook mode

# OANDA
OANDA_API_KEY        = os.getenv("OANDA_API_KEY", "").strip()
OANDA_ACCOUNT_ID     = os.getenv("OANDA_ACCOUNT_ID", "").strip()
OANDA_ENV            = os.getenv("OANDA_ENV", "practice").strip().lower()  # practice|live
OANDA_INSTRUMENTS    = [s.strip() for s in os.getenv("OANDA_INSTRUMENTS", "EUR_USD,GBP_USD,USD_JPY").split(",") if s.strip()]
OANDA_GRANULARITY    = os.getenv("OANDA_GRANULARITY", "M1").strip().upper()  # M1/M3/M5/M15 etc.

# Pocket Option guidance
PO_EXPIRY_MIN        = int(os.getenv("PO_EXPIRY_MIN", "3"))      # suggested expiry minutes per signal
PO_ENTRY_DELAY_SEC   = int(os.getenv("PO_ENTRY_DELAY_SEC", "0")) # delay before entry (secs) if you want next-candle entry
ALERT_COOLDOWN_MIN   = int(os.getenv("ALERT_COOLDOWN_MIN", "10")) # suppress repeats per pair

# Signal tuning
RSI_MIN_BUY          = float(os.getenv("RSI_MIN_BUY", "50"))
RSI_MAX_SELL         = float(os.getenv("RSI_MAX_SELL", "50"))
EMA_FAST             = int(os.getenv("EMA_FAST", "9"))
EMA_SLOW             = int(os.getenv("EMA_SLOW", "21"))
ATR_FILTER_MULT      = float(os.getenv("ATR_FILTER_MULT", "0.0"))  # set >0 to filter low vol (e.g., 0.5)

# Scheduler time zone = UTC to match OANDA times
SCHED_TZ             = timezone.utc

# Logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("pocket-option-signals")

# Globals
telegram_app: Application | None = None
scheduler: AsyncIOScheduler | None = None
_last_alert_at: Dict[str, datetime] = {}   # debounce per instrument (UTC time)
_last_candle_time: Dict[str, datetime] = {}  # to only fire once per completed candle

# =============================================================================
# Telegram Handlers
# =============================================================================
async def start_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("✅ Pocket Option Signals (OANDA candles) — live.")

async def ping_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("pong")

def build_telegram_app() -> Application:
    app = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", start_cmd))
    app.add_handler(CommandHandler("ping",  ping_cmd))
    return app

# =============================================================================
# OANDA Helpers
# =============================================================================
def oanda_base_url() -> str:
    if OANDA_ENV == "live":
        return "https://api-fxtrade.oanda.com"
    return "https://api-fxpractice.oanda.com"

def _auth_headers() -> Dict[str, str]:
    return {
        "Authorization": f"Bearer {OANDA_API_KEY}",
        "Content-Type": "application/json",
    }

async def fetch_oanda_candles(instrument: str, granularity: str, count: int = 300) -> pd.DataFrame:
    """
    Return ascending DataFrame with columns: time,o,h,l,c,volume,complete (last row may be incomplete).
    """
    url = f"{oanda_base_url()}/v3/instruments/{instrument}/candles"
    params = {"count": str(count), "granularity": granularity, "price": "M"}
    async with httpx.AsyncClient(timeout=25) as client:
        r = await client.get(url, params=params, headers=_auth_headers())
        r.raise_for_status()
        data = r.json()

    rows = []
    for c in data.get("candles", []):
        mid = c.get("mid", {})
        rows.append({
            "time":   pd.to_datetime(c["time"]),
            "o":      float(mid["o"]),
            "h":      float(mid["h"]),
            "l":      float(mid["l"]),
            "c":      float(mid["c"]),
            "volume": int(c.get("volume", 0)),
            "complete": bool(c.get("complete", False)),
        })
    df = pd.DataFrame(rows).sort_values("time").reset_index(drop=True)
    return df

# =============================================================================
# Signal logic (Binary Options friendly)
# =============================================================================
def last_completed(df: pd.DataFrame) -> Optional[int]:
    """Return index of last *completed* candle, or None."""
    if df.empty:
        return None
    # some feeds mark last as not complete
    if df.iloc[-1]["complete"]:
        return df.index[-1]
    if len(df) >= 2 and df.iloc[-2]["complete"]:
        return df.index[-2]
    return None

def pocket_asset_name(instrument: str) -> str:
    """
    Map OANDA instrument -> Pocket Option asset label (best-effort).
    You can expand this mapping as needed.
    """
    m = {
        "EUR_USD": "EUR/USD",
        "GBP_USD": "GBP/USD",
        "USD_JPY": "USD/JPY",
        "AUD_USD": "AUD/USD",
        "USD_CAD": "USD/CAD",
        "USD_CHF": "USD/CHF",
        "NZD_USD": "NZD/USD",
        "XAU_USD": "GOLD",
    }
    return m.get(instrument, instrument.replace("_", "/"))

def compute_signal_row(df: pd.DataFrame, idx: int) -> Dict[str, Any]:
    """
    Compute CALL/PUT recommendation using EMA(EMA_FAST/EMA_SLOW) + RSI(14),
    optional ATR filter; evaluate using candle at `idx` and previous candle (idx-1).
    """
    close = df["c"].astype(float)
    ema_fast = ta.ema(close, length=EMA_FAST)
    ema_slow = ta.ema(close, length=EMA_SLOW)
    rsi14    = ta.rsi(close, length=14)
    atr      = ta.atr(df["h"], df["l"], close, length=14)

    df = df.copy()
    df["ema_fast"] = ema_fast
    df["ema_slow"] = ema_slow
    df["rsi14"]    = rsi14
    df["atr14"]    = atr

    if idx < 1:
        return {"signal": "HOLD", "reason": "Not enough candles"}

    cur  = df.iloc[idx]
    prev = df.iloc[idx - 1]

    # Optional ATR vol filter
    if ATR_FILTER_MULT > 0:
        # tiny moves: skip
        if (cur["atr14"] or 0) < ATR_FILTER_MULT * (cur["c"] * 1e-5):
            return {"signal": "HOLD", "reason": "Low volatility filter (ATR)"}

    signal = "HOLD"
    reason = "No cross"
    direction = None

    # BUY/CALL if fast crosses above slow and RSI >= RSI_MIN_BUY
    if (prev["ema_fast"] <= prev["ema_slow"]) and (cur["ema_fast"] > cur["ema_slow"]) and (cur["rsi14"] >= RSI_MIN_BUY):
        signal = "CALL"
        reason = f"EMA{EMA_FAST}↑EMA{EMA_SLOW} & RSI≥{RSI_MIN_BUY}"
        direction = "UP"
    # SELL/PUT if fast crosses below slow and RSI <= RSI_MAX_SELL
    elif (prev["ema_fast"] >= prev["ema_slow"]) and (cur["ema_fast"] < cur["ema_slow"]) and (cur["rsi14"] <= RSI_MAX_SELL):
        signal = "PUT"
        reason = f"EMA{EMA_FAST}↓EMA{EMA_SLOW} & RSI≤{RSI_MAX_SELL}"
        direction = "DOWN"

    return {
        "signal": signal,
        "reason": reason,
        "direction": direction,
        "price": float(cur["c"]),
        "time": cur["time"].to_pydatetime().replace(tzinfo=timezone.utc),
        "ema_fast": float(cur["ema_fast"]),
        "ema_slow": float(cur["ema_slow"]),
        "rsi14": float(cur["rsi14"]),
    }

def cooldown_ok(instrument: str) -> bool:
    now = datetime.now(timezone.utc)
    last = _last_alert_at.get(instrument)
    return True if (not last) else (now - last) >= timedelta(minutes=ALERT_COOLDOWN_MIN)

def mark_alert(instrument: str) -> None:
    _last_alert_at[instrument] = datetime.now(timezone.utc)

async def send_signal_message(inst: str, res: Dict[str, Any]) -> None:
    if not TELEGRAM_GROUP_ID:
        log.warning("TELEGRAM_GROUP_ID not set; skipping Telegram send.")
        return
    asset = pocket_asset_name(inst)
    entry_at = datetime.now(timezone.utc) + timedelta(seconds=PO_ENTRY_DELAY_SEC)
    expiry_at = entry_at + timedelta(minutes=PO_EXPIRY_MIN)

    msg = (
        f"⚡️ *PO Signal* — {asset}\n"
        f"• Action: *{res['signal']}* ({res['direction']})\n"
        f"• Price: {res['price']}\n"
        f"• TF: {OANDA_GRANULARITY}\n"
        f"• Entry: {entry_at.strftime('%H:%M:%S UTC')} (in {PO_ENTRY_DELAY_SEC}s)\n"
        f"• Expiry: {PO_EXPIRY_MIN} min → {expiry_at.strftime('%H:%M UTC')}\n"
        f"• EMA{EMA_FAST}/{EMA_SLOW}: {res['ema_fast']:.5f}/{res['ema_slow']:.5f}\n"
        f"• RSI14: {res['rsi14']:.2f}\n"
        f"• Reason: {res['reason']}\n"
        f"• Candle Time: {res['time'].strftime('%Y-%m-%d %H:%M UTC')}\n"
        f"• Mode: OANDA candles → Pocket Option"
    )
    try:
        await telegram_app.bot.send_message(
            chat_id=TELEGRAM_GROUP_ID,
            text=msg,
            parse_mode=None,  # keep plain text for compatibility
            disable_web_page_preview=True,
        )
        log.info(f"📤 Sent PO {res['signal']} for {inst}")
    except Exception as e:
        log.error(f"Telegram send error: {e}")

# =============================================================================
# Engine
# =============================================================================
async def process_instrument(inst: str) -> None:
    try:
        df = await fetch_oanda_candles(inst, OANDA_GRANULARITY, count=300)
        idx = last_completed(df)
        if idx is None:
            log.info(f"{inst}: no completed candle yet")
            return

        # Fire once per completed candle
        last_time = df.iloc[idx]["time"].to_pydatetime().replace(tzinfo=timezone.utc)
        if _last_candle_time.get(inst) == last_time:
            return  # already processed
        _last_candle_time[inst] = last_time

        res = compute_signal_row(df, idx)
        sig = res.get("signal", "HOLD")
        if sig in ("CALL", "PUT") and cooldown_ok(inst):
            await send_signal_message(inst, res)
            mark_alert(inst)
        else:
            log.info(f"{inst}: {sig} ({res.get('reason')})")
    except Exception as e:
        log.error(f"{inst} error: {e}")

async def run_engine():
    if not OANDA_API_KEY or not OANDA_ACCOUNT_ID:
        log.warning("OANDA credentials missing; engine skipped.")
        return
    await asyncio.gather(*(process_instrument(i) for i in OANDA_INSTRUMENTS))

# =============================================================================
# Webhook lifecycle
# =============================================================================
async def set_webhook():
    assert telegram_app is not None
    if not BASE_URL:
        raise RuntimeError("BASE_URL is required for webhook mode.")
    url = f"{BASE_URL}{WEBHOOK_PATH}"
    try:
        await telegram_app.bot.delete_webhook(drop_pending_updates=True)
        log.info("🧹 Deleted existing webhook (drop_pending_updates=True)")
    except Exception as e:
        log.warning(f"Webhook delete warning: {e}")
    await telegram_app.initialize()
    await telegram_app.start()
    await telegram_app.bot.set_webhook(url=url, secret_token=WEBHOOK_SECRET, drop_pending_updates=True)
    log.info(f"🔗 Webhook set to: {url}")

async def unset_webhook():
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
        log.info("🛑 Telegram stopped")
    except Exception as e:
        log.error(f"Shutdown error: {e}")

# =============================================================================
# FastAPI App (lifespan: webhook + scheduler aligned to candle close)
# =============================================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    global telegram_app, scheduler
    if not TELEGRAM_BOT_TOKEN:
        log.error("❌ TELEGRAM_BOT_TOKEN missing.")
        yield
        return
    if ENABLE_BOT_POLLING == "1":
        log.warning("⚠️ Polling set but webhook mode is used. Set ENABLE_BOT_POLLING=0.")

    telegram_app = build_telegram_app()
    await set_webhook()

    # Schedule at :00 of each minute (or align to chosen granularity)
    # For M1/M5/M15 etc., triggering each minute is fine; we gate on last completed candle.
    scheduler = AsyncIOScheduler(timezone=SCHED_TZ)
    scheduler.add_job(run_engine, CronTrigger(second=5))  # run at HH:MM:05 UTC every minute
    scheduler.start()
    log.info(f"⏱️ Engine scheduled every minute (evaluates completed {OANDA_GRANULARITY} candles)")

    yield
    try:
        if scheduler:
            scheduler.shutdown(wait=False)
        await unset_webhook()
    except Exception as e:
        log.error(f"Lifespan shutdown error: {e}")

app = FastAPI(lifespan=lifespan)

# =============================================================================
# Routes
# =============================================================================
@app.get("/", response_class=PlainTextResponse)
async def root_get():
    return "Pocket Option Signals — OANDA → PO (Webhook Mode)"

@app.head("/")
async def root_head():
    return Response(status_code=200)

@app.get("/healthz", response_class=PlainTextResponse)
async def healthz_get():
    return "ok"

@app.head("/healthz")
async def healthz_head():
    return Response(status_code=200)

@app.post(WEBHOOK_PATH)
async def telegram_webhook(
    request: Request,
    x_telegram_bot_api_secret_token: str = Header(None),
):
    if WEBHOOK_SECRET and x_telegram_bot_api_secret_token != WEBHOOK_SECRET:
        raise HTTPException(status_code=403, detail="Forbidden")
    if telegram_app is None:
        raise HTTPException(status_code=503, detail="Bot not initialized")
    data = await request.json()
    update = Update.de_json(data, telegram_app.bot)
    await telegram_app.process_update(update)
    return {"ok": True}

@app.get("/status")
async def status():
    return JSONResponse({
        "ok": True,
        "mode": "webhook",
        "oanda_env": OANDA_ENV,
        "instruments": OANDA_INSTRUMENTS,
        "granularity": OANDA_GRANULARITY,
        "po_expiry_min": PO_EXPIRY_MIN,
        "po_entry_delay_sec": PO_ENTRY_DELAY_SEC,
        "cooldown_min": ALERT_COOLDOWN_MIN,
        "last_alert_at": {k: v.isoformat() for k, v in _last_alert_at.items()},
        "last_candle_seen": {k: v.isoformat() for k, v in _last_candle_time.items()},
    })
