# app.py
# Pocket Option Signals — OANDA→PO (Webhook, OTC-aware, Tiers, W/L/D, NOWPayments paywall)
# Single-process Render service: web endpoints + Telegram bot + schedulers
# ------------------------------------------------------------------------------------------

import os
import json
import hmac
import hashlib
import asyncio
import logging
from datetime import datetime, timedelta, timezone
from contextlib import asynccontextmanager
from typing import Dict, List, Any, Optional, Tuple, Set

import pandas as pd
import pandas_ta as ta
import httpx

from fastapi import FastAPI, Response, Request, Header, HTTPException, Query
from fastapi.responses import PlainTextResponse, JSONResponse

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes

# ==========================================================================================
# ENV
# ==========================================================================================
# Webhook / Telegram
TELEGRAM_BOT_TOKEN   = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
BASE_URL             = os.getenv("BASE_URL", "").rstrip("/")
WEBHOOK_PATH         = os.getenv("WEBHOOK_PATH", "/telegram/webhook").strip()
WEBHOOK_SECRET       = os.getenv("WEBHOOK_SECRET", "change-me").strip()
ENABLE_BOT_POLLING   = os.getenv("ENABLE_BOT_POLLING", "0").strip()  # keep 0 for webhook mode

# Legacy single group (treated as VIP/unlimited). Optional.
TELEGRAM_GROUP_ID    = os.getenv("TELEGRAM_GROUP_ID", "").strip()

# Tiered chat IDs (optional; set the ones you use)
TELEGRAM_CHAT_FREE   = os.getenv("TELEGRAM_CHAT_FREE", "").strip()    # 3/day
TELEGRAM_CHAT_BASIC  = os.getenv("TELEGRAM_CHAT_BASIC", "").strip()   # 6/day
TELEGRAM_CHAT_PRO    = os.getenv("TELEGRAM_CHAT_PRO", "").strip()     # 15/day
TELEGRAM_CHAT_VIP    = os.getenv("TELEGRAM_CHAT_VIP", "").strip()     # unlimited

# Per-tier caps (override via env if needed)
LIMIT_FREE           = int(os.getenv("LIMIT_FREE", "3"))
LIMIT_BASIC          = int(os.getenv("LIMIT_BASIC", "6"))
LIMIT_PRO            = int(os.getenv("LIMIT_PRO", "15"))
# VIP = unlimited

# Invite links for paid groups (bot must be admin or use static links you generate)
TELEGRAM_LINK_BASIC  = os.getenv("TELEGRAM_LINK_BASIC", "").strip()
TELEGRAM_LINK_PRO    = os.getenv("TELEGRAM_LINK_PRO", "").strip()
TELEGRAM_LINK_VIP    = os.getenv("TELEGRAM_LINK_VIP", "").strip()

# OANDA
OANDA_API_KEY        = os.getenv("OANDA_API_KEY", "").strip()
OANDA_ACCOUNT_ID     = os.getenv("OANDA_ACCOUNT_ID", "").strip()
OANDA_ENV            = os.getenv("OANDA_ENV", "practice").strip().lower()  # practice|live
OANDA_INSTRUMENTS    = [s.strip() for s in os.getenv("OANDA_INSTRUMENTS", "EUR_USD,GBP_USD,USD_JPY").split(",") if s.strip()]
OANDA_GRANULARITY    = os.getenv("OANDA_GRANULARITY", "M1").strip().upper()  # M1/M5/M15 etc.

# Pocket Option signal settings
PO_EXPIRY_MIN        = int(os.getenv("PO_EXPIRY_MIN", "5"))      # suggested expiry (min)
PO_ENTRY_DELAY_SEC   = int(os.getenv("PO_ENTRY_DELAY_SEC", "2")) # seconds after candle close
ALERT_COOLDOWN_MIN   = int(os.getenv("ALERT_COOLDOWN_MIN", "10"))

# Signal tuning
RSI_MIN_BUY          = float(os.getenv("RSI_MIN_BUY", "50"))
RSI_MAX_SELL         = float(os.getenv("RSI_MAX_SELL", "50"))
EMA_FAST             = int(os.getenv("EMA_FAST", "9"))
EMA_SLOW             = int(os.getenv("EMA_SLOW", "21"))
ATR_FILTER_MULT      = float(os.getenv("ATR_FILTER_MULT", "0.0"))

# OTC control (HARD-CODED PO LIVE WINDOW; ignore PO_OTC_UTC_WINDOWS)
# PO Forex LIVE Mon–Fri 02:00–22:45 UTC → signals ON; otherwise OTC → OFF
PO_BLOCK_DURING_OTC  = int(os.getenv("PO_BLOCK_DURING_OTC", "1"))

# Scheduler TZ (OANDA timestamps are UTC)
SCHED_TZ             = timezone.utc

# Files (ephemeral on Render; OK for counters)
DATA_DIR             = os.getenv("DATA_DIR", "./data").strip()
STATS_FILE           = os.path.join(DATA_DIR, "stats.json")
OPEN_TRADES_FILE     = os.path.join(DATA_DIR, "open_trades.json")
QUOTA_FILE           = os.path.join(DATA_DIR, "quota.json")        # per-chat daily counters
UNSUPPORTED_FILE     = os.path.join(DATA_DIR, "unsupported.json")  # instruments to skip
ORDERS_FILE          = os.path.join(DATA_DIR, "orders.json")       # invoice order_id -> context

# NOWPayments (Paywall)
NOWPAY_API_KEY       = os.getenv("NOWPAY_API_KEY", "").strip()            # header x-api-key
NOWPAY_IPN_SECRET    = os.getenv("NOWPAY_IPN_SECRET", "").strip()         # for x-nowpayments-sig verification (HMAC-SHA512)
NOWPAY_PRICE_CCY     = os.getenv("NOWPAY_PRICE_CCY", "USD").strip().lower()
PRICE_BASIC          = float(os.getenv("PRICE_BASIC", "19.0"))
PRICE_PRO            = float(os.getenv("PRICE_PRO", "39.0"))
PRICE_VIP            = float(os.getenv("PRICE_VIP", "99.0"))
NOWPAY_PAY_CCY       = os.getenv("NOWPAY_PAY_CCY", "").strip().lower()    # optional (let user choose if empty)

# Logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("pocket-option-signals")

# Globals
telegram_app: Application | None = None
scheduler: AsyncIOScheduler | None = None

_last_candle_time: Dict[str, datetime] = {}  # instrument -> last processed completed candle utc

# In-memory stores, persisted to JSON
_stats_lock = asyncio.Lock()
_trades_lock = asyncio.Lock()
_quota_lock = asyncio.Lock()
_unsupported_lock = asyncio.Lock()
_orders_lock = asyncio.Lock()

_stats: Dict[str, Any] = {}         # {'daily': {'YYYY-MM-DD': {'win':..,'loss':..,'draw':..}}, 'weekly': {...}}
_open_trades: List[Dict[str, Any]] = []
_quota: Dict[str, Dict[str, int]] = {}      # {chat_id: { 'YYYY-MM-DD': count }}
_unsupported: Set[str] = set()               # instruments that produced 400/invalid; we skip them
_orders: Dict[str, Dict[str, Any]] = {}      # order_id -> {'tier': 'BASIC', 'telegram_id': 12345, ...}

# ==========================================================================================
# Utilities: FS
# ==========================================================================================
def _ensure_dirs():
    os.makedirs(DATA_DIR, exist_ok=True)

def _week_key(dt: datetime) -> str:
    iso = dt.isocalendar()
    return f"{iso.year}-W{iso.week:02d}"

async def load_state():
    _ensure_dirs()
    global _stats, _open_trades, _quota, _unsupported, _orders
    try:
        _stats = json.load(open(STATS_FILE)) if os.path.exists(STATS_FILE) else {"daily": {}, "weekly": {}}
    except Exception as e:
        log.error(f"Failed to load stats: {e}")
        _stats = {"daily": {}, "weekly": {}}
    try:
        _open_trades = json.load(open(OPEN_TRADES_FILE)) if os.path.exists(OPEN_TRADES_FILE) else []
    except Exception as e:
        log.error(f"Failed to load open_trades: {e}")
        _open_trades = []
    try:
        _quota = json.load(open(QUOTA_FILE)) if os.path.exists(QUOTA_FILE) else {}
    except Exception as e:
        log.error(f"Failed to load quota: {e}")
        _quota = {}
    try:
        _unsupported = set(json.load(open(UNSUPPORTED_FILE))) if os.path.exists(UNSUPPORTED_FILE) else set()
    except Exception as e:
        log.error(f"Failed to load unsupported: {e}")
        _unsupported = set()
    try:
        _orders = json.load(open(ORDERS_FILE)) if os.path.exists(ORDERS_FILE) else {}
    except Exception as e:
        log.error(f"Failed to load orders: {e}")
        _orders = {}

async def _save_json(path: str, data: Any):
    _ensure_dirs()
    with open(path, "w") as f:
        json.dump(data, f)

async def save_stats():        await _save_json(STATS_FILE, _stats)
async def save_open_trades():  await _save_json(OPEN_TRADES_FILE, _open_trades)
async def save_quota():        await _save_json(QUOTA_FILE, _quota)
async def save_unsupported():  await _save_json(UNSUPPORTED_FILE, sorted(list(_unsupported)))
async def save_orders():       await _save_json(ORDERS_FILE, _orders)

# ==========================================================================================
# Telegram Handlers (commands for onboarding + paywall)
# ==========================================================================================
HELP_TEXT = (
    "Welcome to Pocket Option Signals 🤖\n"
    "• 3/6/15/Unlimited signals by tier (FREE/BASIC/PRO/VIP)\n"
    "• Live-only hours (OTC paused)\n"
    "• W/L/D stats daily & weekly\n\n"
    "Commands:\n"
    "/start — intro\n"
    "/ping — health check\n"
    "/plans — show prices & how to upgrade\n"
    "/upgrade <BASIC|PRO|VIP> — create a crypto invoice\n"
)

async def start_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(HELP_TEXT)

async def ping_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("pong")

def _plan_price(tier: str) -> Optional[float]:
    t = tier.upper()
    return {"BASIC": PRICE_BASIC, "PRO": PRICE_PRO, "VIP": PRICE_VIP}.get(t)

def _plan_link(tier: str) -> str:
    t = tier.upper()
    return {"BASIC": TELEGRAM_LINK_BASIC, "PRO": TELEGRAM_LINK_PRO, "VIP": TELEGRAM_LINK_VIP}.get(t, "")

async def plans_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = (
        f"Plans (paid monthly, pay in crypto via NOWPayments):\n"
        f"• BASIC — {PRICE_BASIC:.2f} {NOWPAY_PRICE_CCY.upper()} — 6 signals/day\n"
        f"• PRO   — {PRICE_PRO:.2f} {NOWPAY_PRICE_CCY.upper()} — 15 signals/day\n"
        f"• VIP   — {PRICE_VIP:.2f} {NOWPAY_PRICE_CCY.upper()} — Unlimited\n\n"
        f"Use /upgrade BASIC (or PRO/VIP) to get your payment link."
    )
    await update.message.reply_text(msg)

async def upgrade_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # /upgrade TIER
    if not NOWPAY_API_KEY:
        await update.message.reply_text("Payments are temporarily unavailable. Try later.")
        return
    args = context.args or []
    if len(args) < 1:
        await update.message.reply_text("Usage: /upgrade BASIC|PRO|VIP")
        return
    tier = args[0].upper()
    price = _plan_price(tier)
    if price is None:
        await update.message.reply_text("Tier not found. Use BASIC, PRO, or VIP.")
        return

    user = update.effective_user
    user_id = user.id if user else 0
    order_id = f"tier:{tier}:uid:{user_id}:{int(datetime.now(timezone.utc).timestamp())}"
    description = f"Pocket Option Signals — {tier} plan for Telegram user {user_id}"

    # Create NOWPayments invoice
    try:
        invoice = await nowpay_create_invoice(
            price_amount=price,
            price_currency=NOWPAY_PRICE_CCY,
            order_id=order_id,
            order_description=description,
            pay_currency=NOWPAY_PAY_CCY or None,
            success_url=f"{BASE_URL}/pay/success?tier={tier}",
            cancel_url=f"{BASE_URL}/pay/cancel?tier={tier}",
            ipn_callback_url=f"{BASE_URL}/pay/ipn"
        )
    except Exception as e:
        log.error(f"Invoice error: {e}")
        await update.message.reply_text("Could not create invoice. Please try again later.")
        return

    invoice_url = invoice.get("invoice_url") or invoice.get("url") or ""
    if not invoice_url:
        await update.message.reply_text("Invoice link unavailable. Please try later.")
        return

    # Save minimal order context for later IPN use
    async with _orders_lock:
        _orders[order_id] = {"tier": tier, "telegram_id": user_id, "created_at": datetime.now(timezone.utc).isoformat()}
        await save_orders()

    await update.message.reply_text(
        f"✅ Invoice created for {tier} ({price:.2f} {NOWPAY_PRICE_CCY.upper()}).\n"
        f"Pay here: {invoice_url}\n\n"
        f"Once payment is confirmed, you’ll receive your private group invite link."
    )

def build_telegram_app() -> Application:
    app = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", start_cmd))
    app.add_handler(CommandHandler("ping",  ping_cmd))
    app.add_handler(CommandHandler("plans", plans_cmd))
    app.add_handler(CommandHandler("upgrade", upgrade_cmd))
    return app

# ==========================================================================================
# OANDA Helpers
# ==========================================================================================
def oanda_base_url() -> str:
    return "https://api-fxtrade.oanda.com" if OANDA_ENV == "live" else "https://api-fxpractice.oanda.com"

def _auth_headers() -> Dict[str, str]:
    return {"Authorization": f"Bearer {OANDA_API_KEY}", "Content-Type": "application/json"}

async def fetch_oanda_candles(instrument: str, granularity: str, count: int = 300) -> pd.DataFrame:
    url = f"{oanda_base_url()}/v3/instruments/{instrument}/candles"
    params = {"count": str(count), "granularity": granularity, "price": "M"}
    async with httpx.AsyncClient(timeout=25) as client:
        r = await client.get(url, params=params, headers=_auth_headers())
        # If instrument invalid/disabled → 400; mark as unsupported
        if r.status_code == 400:
            async with _unsupported_lock:
                if instrument not in _unsupported:
                    _unsupported.add(instrument)
                    log.error(f"{instrument} marked unsupported (400).")
                    await save_unsupported()
            r.raise_for_status()
        r.raise_for_status()
        data = r.json()

    rows = []
    for c in data.get("candles", []):
        mid = c.get("mid", {})
        if "time" not in c or not mid:
            continue
        try:
            rows.append({
                "time":   pd.to_datetime(c["time"]),
                "o":      float(mid.get("o", "nan")),
                "h":      float(mid.get("h", "nan")),
                "l":      float(mid.get("l", "nan")),
                "c":      float(mid.get("c", "nan")),
                "volume": int(c.get("volume", 0)),
                "complete": bool(c.get("complete", False)),
            })
        except Exception:
            continue
    df = pd.DataFrame(rows).dropna().sort_values("time").reset_index(drop=True)
    return df

def last_completed(df: pd.DataFrame) -> Optional[int]:
    if df.empty:
        return None
    if bool(df.iloc[-1].get("complete", False)):
        return int(df.index[-1])
    if len(df) >= 2 and bool(df.iloc[-2].get("complete", False)):
        return int(df.index[-2])
    return None

# ==========================================================================================
# PO mapping & signals
# ==========================================================================================
def pocket_asset_name(instrument: str) -> str:
    return instrument.replace("_", "/")

def compute_signal_row(df: pd.DataFrame, idx: int) -> Dict[str, Any]:
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

    if ATR_FILTER_MULT > 0 and (cur["atr14"] or 0) < ATR_FILTER_MULT * (cur["c"] * 1e-5):
        return {"signal": "HOLD", "reason": "Low volatility (ATR filter)"}

    signal = "HOLD"; reason = "No cross"; direction = None
    if (prev["ema_fast"] <= prev["ema_slow"]) and (cur["ema_fast"] > cur["ema_slow"]) and (cur["rsi14"] >= RSI_MIN_BUY):
        signal, reason, direction = "CALL", f"EMA{EMA_FAST}↑EMA{EMA_SLOW} & RSI≥{RSI_MIN_BUY}", "UP"
    elif (prev["ema_fast"] >= prev["ema_slow"]) and (cur["ema_fast"] < cur["ema_slow"]) and (cur["rsi14"] <= RSI_MAX_SELL):
        signal, reason, direction = "PUT",  f"EMA{EMA_FAST}↓EMA{EMA_SLOW} & RSI≤{RSI_MAX_SELL}", "DOWN"

    return {
        "signal": signal, "reason": reason, "direction": direction,
        "price": float(cur["c"]),
        "time": cur["time"].to_pydatetime().replace(tzinfo=timezone.utc),
        "ema_fast": float(cur["ema_fast"]), "ema_slow": float(cur["ema_slow"]), "rsi14": float(cur["rsi14"]),
    }

# ==========================================================================================
# OTC detection (HARD rule for PO: Live Mon–Fri 02:00–22:45 UTC)
# ==========================================================================================
def is_otc_now(now_utc: Optional[datetime] = None) -> bool:
    if PO_BLOCK_DURING_OTC != 1:
        return False
    now = (now_utc or datetime.now(timezone.utc))
    wd = now.weekday()  # Monday=0, Sunday=6
    if wd in (5, 6):  # Sat, Sun
        return True
    live_start = now.replace(hour=2, minute=0, second=0, microsecond=0)
    live_end   = now.replace(hour=22, minute=45, second=0, microsecond=0)
    if live_start <= now <= live_end:
        return False  # live hours → signals ON
    return True        # outside live → OTC → signals OFF

# ==========================================================================================
# QUOTA / TIERS
# ==========================================================================================
def _tiers() -> List[Tuple[str, Optional[int], str]]:
    """
    Returns list of (chat_id, cap or None, label)
    VIP and TELEGRAM_GROUP_ID (legacy) are unlimited (cap=None).
    """
    out: List[Tuple[str, Optional[int], str]] = []
    if TELEGRAM_CHAT_FREE:  out.append((TELEGRAM_CHAT_FREE,  LIMIT_FREE,  "FREE"))
    if TELEGRAM_CHAT_BASIC: out.append((TELEGRAM_CHAT_BASIC, LIMIT_BASIC, "BASIC"))
    if TELEGRAM_CHAT_PRO:   out.append((TELEGRAM_CHAT_PRO,   LIMIT_PRO,   "PRO"))
    if TELEGRAM_CHAT_VIP:   out.append((TELEGRAM_CHAT_VIP,   None,        "VIP"))
    if TELEGRAM_GROUP_ID:   out.append((TELEGRAM_GROUP_ID,   None,        "VIP"))  # legacy treated as VIP
    return out

def _today_key() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")

async def _quota_get(chat_id: str) -> int:
    async with _quota_lock:
        day = _today_key()
        return int(_quota.get(chat_id, {}).get(day, 0))

async def _quota_inc(chat_id: str) -> int:
    async with _quota_lock:
        day = _today_key()
        _quota.setdefault(chat_id, {})
        _quota[chat_id][day] = int(_quota[chat_id].get(day, 0)) + 1
        await save_quota()
        return _quota[chat_id][day]

async def send_to_tiers(text: str):
    targets = _tiers()
    if not targets:
        log.warning("No Telegram chats configured; skipping send.")
        return
    for chat_id, cap, label in targets:
        can_send = True
        if cap is not None:
            used = await _quota_get(chat_id)
            can_send = used < cap
        if not can_send:
            log.info(f"Quota reached for {label} ({chat_id}). Skipping.")
            continue
        try:
            suffix = "" if cap is None else f" [{label} {await _quota_get(chat_id)+1}/{cap}]"
            await telegram_app.bot.send_message(chat_id=chat_id, text=text + suffix, disable_web_page_preview=True)
            await _quota_inc(chat_id)
        except Exception as e:
            log.error(f"Telegram send error ({label}/{chat_id}): {e}")

# ==========================================================================================
# W/L/D settlement & stats
# ==========================================================================================
async def price_at_or_after(instrument: str, when_utc: datetime) -> Optional[float]:
    df = await fetch_oanda_candles(instrument, "M1", count=400)
    if df.empty or "time" not in df.columns:
        return None
    df_times = df["time"].dt.tz_convert("UTC")
    after = df[df_times >= pd.Timestamp(when_utc)]
    if not after.empty:
        return float(after.iloc[0]["c"])
    before = df[df_times <= pd.Timestamp(when_utc)]
    if not before.empty:
        return float(before.iloc[-1]["c"])
    return None

async def record_open_trade(inst: str, direction: str, entry_price: float, entry_time: datetime, expiry_min: int):
    expiry_time = entry_time + timedelta(minutes=expiry_min)
    trade = {
        "instrument": inst,
        "direction": direction,  # CALL/PUT
        "entry_price": entry_price,
        "entry_time": entry_time.isoformat(),
        "expiry_time": expiry_time.isoformat(),
        "settled": False,
        "result": None,
    }
    async with _trades_lock:
        _open_trades.append(trade)
        await save_open_trades()

def _bump_stats(dt: datetime, result: str):
    dkey = dt.strftime("%Y-%m-%d")
    wkey = _week_key(dt)
    for bucket, key in (("daily", dkey), ("weekly", wkey)):
        _stats.setdefault(bucket, {})
        _stats[bucket].setdefault(key, {"win": 0, "loss": 0, "draw": 0})
        _stats[bucket][key][result.lower()] += 1

async def settle_due_trades():
    now = datetime.now(timezone.utc)
    changed = False
    async with _trades_lock:
        for t in _open_trades:
            if t["settled"]:
                continue
            expiry = datetime.fromisoformat(t["expiry_time"]).astimezone(timezone.utc)
            if now < expiry + timedelta(seconds=5):
                continue
            px = await price_at_or_after(t["instrument"], expiry)
            if px is None:
                continue
            entry = float(t["entry_price"])
            if px > entry:
                result = "WIN"  if t["direction"] == "CALL" else "LOSS"
            elif px < entry:
                result = "LOSS" if t["direction"] == "CALL" else "WIN"
            else:
                result = "DRAW"
            t["settled"] = True
            t["result"] = result
            changed = True
            async with _stats_lock:
                _bump_stats(expiry, result)
        if changed:
            await save_open_trades()
            await save_stats()

async def send_daily_weekly_reports():
    now = datetime.now(timezone.utc)
    dkey = now.strftime("%Y-%m-%d")
    wkey = _week_key(now)
    async with _stats_lock:
        d = _stats.get("daily", {}).get(dkey, {"win": 0, "loss": 0, "draw": 0})
        w = _stats.get("weekly", {}).get(wkey, {"win": 0, "loss": 0, "draw": 0})
    msg = (
        "📈 Pocket Option Results\n"
        f"Daily ({dkey}) — W:{d['win']} L:{d['loss']} D:{d['draw']}\n"
        f"Weekly ({wkey}) — W:{w['win']} L:{w['loss']} D:{w['draw']}"
    )
    await send_to_tiers(msg)

# ==========================================================================================
# Engine
# ==========================================================================================
def is_supported(instrument: str) -> bool:
    return instrument not in _unsupported

async def process_instrument(inst: str) -> None:
    if not is_supported(inst):
        return
    try:
        df = await fetch_oanda_candles(inst, OANDA_GRANULARITY, count=300)
        idx = last_completed(df)
        if idx is None:
            return
        last_time = df.iloc[idx]["time"].to_pydatetime().replace(tzinfo=timezone.utc)
        if _last_candle_time.get(inst) == last_time:
            return  # already processed this completed candle
        _last_candle_time[inst] = last_time

        res = compute_signal_row(df, idx)
        sig = res.get("signal", "HOLD")
        if sig in ("CALL", "PUT"):
            if is_otc_now():
                return
            entry_time = datetime.now(timezone.utc) + timedelta(seconds=PO_ENTRY_DELAY_SEC)
            asset = pocket_asset_name(inst)
            msg = (
                f"⚡️ PO Signal — {asset}\n"
                f"• Action: {sig} ({res['direction']})\n"
                f"• Price: {res['price']}\n"
                f"• TF: {OANDA_GRANULARITY}\n"
                f"• Entry: {entry_time.strftime('%H:%M:%S UTC')} (+{PO_ENTRY_DELAY_SEC}s)\n"
                f"• Expiry: {PO_EXPIRY_MIN} min\n"
                f"• EMA{EMA_FAST}/{EMA_SLOW}: {res['ema_fast']:.5f}/{res['ema_slow']:.5f}\n"
                f"• RSI14: {res['rsi14']:.2f}\n"
                f"• Reason: {res['reason']}\n"
                f"• Candle: {res['time'].strftime('%Y-%m-%d %H:%M UTC')}\n"
                f"• Source: OANDA→PO"
            )
            await send_to_tiers(msg)
            await record_open_trade(inst, sig, float(res["price"]), entry_time, PO_EXPIRY_MIN)
        else:
            log.info(f"{inst}: {sig} ({res.get('reason')})")
    except httpx.HTTPStatusError as e:
        if e.response is not None and e.response.status_code == 400:
            async with _unsupported_lock:
                if inst not in _unsupported:
                    _unsupported.add(inst)
                    await save_unsupported()
            log.error(f"{inst} marked unsupported (400): {e}")
        else:
            log.error(f"{inst} HTTP error: {e}")
    except Exception as e:
        log.error(f"{inst} error: {e}")

async def run_engine():
    if not OANDA_API_KEY or not OANDA_ACCOUNT_ID:
        log.warning("OANDA credentials missing; engine skipped.")
        return
    if is_otc_now():
        log.info("OTC detected — signals paused.")
        return
    await asyncio.gather(*(process_instrument(i) for i in OANDA_INSTRUMENTS))

# ==========================================================================================
# NOWPayments helper
# ==========================================================================================
NOWPAY_BASE = "https://api.nowpayments.io/v1"

async def nowpay_create_invoice(
    *,
    price_amount: float,
    price_currency: str,
    order_id: str,
    order_description: str,
    pay_currency: Optional[str],
    success_url: str,
    cancel_url: str,
    ipn_callback_url: str,
) -> Dict[str, Any]:
    """
    Creates a NOWPayments invoice and returns JSON (expects 'invoice_url').
    Headers: x-api-key
    Endpoint: POST /v1/invoice
    """
    payload = {
        "price_amount": float(price_amount),
        "price_currency": price_currency.lower(),
        "order_id": order_id,
        "order_description": order_description,
        "success_url": success_url,
        "cancel_url": cancel_url,
        "ipn_callback_url": ipn_callback_url,
    }
    if pay_currency:
        payload["pay_currency"] = pay_currency.lower()

    headers = {"x-api-key": NOWPAY_API_KEY, "Content-Type": "application/json"}
    async with httpx.AsyncClient(timeout=25) as client:
        r = await client.post(f"{NOWPAY_BASE}/invoice", headers=headers, json=payload)
        r.raise_for_status()
        return r.json()

def nowpay_verify_signature(raw_body: bytes, signature: str) -> bool:
    """
    Verify x-nowpayments-sig using HMAC-SHA512 with NOWPAY_IPN_SECRET over
    the canonical JSON string (sorted keys). We accept raw_body as-is.
    NOWPayments sends the body JSON; we recompute hash on sorted JSON.

    If NOWPAY_IPN_SECRET is missing, fail closed (return False).
    """
    try:
        if not NOWPAY_IPN_SECRET:
            return False
        # Load JSON, re-dump with sorted keys (canonical)
        body = json.loads(raw_body.decode("utf-8"))
        canonical = json.dumps(body, sort_keys=True, separators=(",", ":"))
        calc = hmac.new(
            NOWPAY_IPN_SECRET.encode("utf-8"),
            msg=canonical.encode("utf-8"),
            digestmod=hashlib.sha512,
        ).hexdigest()
        return hmac.compare_digest(calc, (signature or "").lower())
    except Exception as e:
        log.error(f"IPN verify error: {e}")
        return False

def _invite_for_tier(tier: str) -> str:
    t = tier.upper()
    return {"BASIC": TELEGRAM_LINK_BASIC, "PRO": TELEGRAM_LINK_PRO, "VIP": TELEGRAM_LINK_VIP}.get(t, "")

async def _notify_user_paid(tier: str, telegram_id: int):
    link = _invite_for_tier(tier)
    if not telegram_id:
        return
    try:
        text = (
            f"✅ Payment confirmed for *{tier}*.\n"
            f"Join your private group: {link if link else '(ask admin for invite link)'}"
        )
        await telegram_app.bot.send_message(chat_id=telegram_id, text=text, disable_web_page_preview=True, parse_mode=None)
    except Exception as e:
        log.error(f"Failed to DM user {telegram_id}: {e}")

# ==========================================================================================
# Webhook lifecycle
# ==========================================================================================
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

# ==========================================================================================
# FastAPI App (lifespan)
# ==========================================================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    global telegram_app, scheduler
    if not TELEGRAM_BOT_TOKEN:
        log.error("❌ TELEGRAM_BOT_TOKEN missing.")
        yield
        return

    await load_state()

    telegram_app = build_telegram_app()
    await set_webhook()

    scheduler = AsyncIOScheduler(timezone=SCHED_TZ)
    # Engine every minute at :05
    scheduler.add_job(run_engine, CronTrigger(second=5))
    # Settlement every 30s
    scheduler.add_job(settle_due_trades, CronTrigger(second="*/30"))
    # Reports: daily 23:59:30 UTC and weekly Sun 23:59:40 UTC
    scheduler.add_job(send_daily_weekly_reports, CronTrigger(hour=23, minute=59, second=30))
    scheduler.add_job(send_daily_weekly_reports, CronTrigger(day_of_week="sun", hour=23, minute=59, second=40))
    scheduler.start()
    log.info("⏱️ Schedulers started (engine, settlement, reports)")

    yield

    try:
        if scheduler:
            scheduler.shutdown(wait=False)
        await unset_webhook()
    except Exception as e:
        log.error(f"Lifespan shutdown error: {e}")

app = FastAPI(lifespan=lifespan)

# ==========================================================================================
# Routes
# ==========================================================================================
@app.get("/", response_class=PlainTextResponse)
async def root_get():
    return "Pocket Option Signals — OANDA→PO (Webhook, OTC-aware, Tiers, W/L/D, NOWPayments)"

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
    now = datetime.now(timezone.utc)
    dkey = now.strftime("%Y-%m-%d")
    wkey = _week_key(now)
    async with _stats_lock:
        daily = _stats.get("daily", {}).get(dkey, {"win": 0, "loss": 0, "draw": 0})
        weekly = _stats.get("weekly", {}).get(wkey, {"win": 0, "loss": 0, "draw": 0})
    async with _quota_lock:
        quotas_today = {cid: _quota.get(cid, {}).get(dkey, 0) for cid, _, _ in _tiers()}
    async with _unsupported_lock:
        unsupported = sorted(list(_unsupported))
    return JSONResponse({
        "ok": True,
        "otc_now": is_otc_now(),
        "oanda_env": OANDA_ENV,
        "instruments": OANDA_INSTRUMENTS,
        "granularity": OANDA_GRANULARITY,
        "expiry_min": PO_EXPIRY_MIN,
        "entry_delay_sec": PO_ENTRY_DELAY_SEC,
        "daily": daily,
        "weekly": weekly,
        "quota_today": quotas_today,
        "unsupported": unsupported,
    })

# ------------------------------------------------------------------------------------------
# Payments: simple helpers to sanity-check env and create demo invoices via HTTP GET (optional)
# ------------------------------------------------------------------------------------------
@app.get("/pay/create")
async def pay_create(
    tier: str = Query(..., pattern="^(?i)(BASIC|PRO|VIP)$"),
    telegram_id: int = Query(..., description="Telegram user id to DM invite after payment"),
):
    if not NOWPAY_API_KEY:
        raise HTTPException(503, "NOWPayments unavailable")
    t = tier.upper()
    amt = _plan_price(t)
    if not amt:
        raise HTTPException(400, "Unknown tier")
    order_id = f"tier:{t}:uid:{telegram_id}:{int(datetime.now(timezone.utc).timestamp())}"
    desc = f"Pocket Option Signals — {t} plan for Telegram user {telegram_id}"
    try:
        invoice = await nowpay_create_invoice(
            price_amount=amt,
            price_currency=NOWPAY_PRICE_CCY,
            order_id=order_id,
            order_description=desc,
            pay_currency=NOWPAY_PAY_CCY or None,
            success_url=f"{BASE_URL}/pay/success?tier={t}",
            cancel_url=f"{BASE_URL}/pay/cancel?tier={t}",
            ipn_callback_url=f"{BASE_URL}/pay/ipn",
        )
    except Exception as e:
        log.error(f"Invoice error: {e}")
        raise HTTPException(500, "Could not create invoice")
    async with _orders_lock:
        _orders[order_id] = {"tier": t, "telegram_id": telegram_id, "created_at": datetime.now(timezone.utc).isoformat()}
        await save_orders()
    return invoice

@app.get("/pay/success", response_class=PlainTextResponse)
async def pay_success(tier: str = "UNKNOWN"):
    return f"Payment success page placeholder. Tier={tier}"

@app.get("/pay/cancel", response_class=PlainTextResponse)
async def pay_cancel(tier: str = "UNKNOWN"):
    return f"Payment canceled. Tier={tier}"

# ------------------------------------------------------------------------------------------
# IPN endpoint — NOWPayments callback (verify HMAC, mark paid, DM user invite link)
# ------------------------------------------------------------------------------------------
@app.post("/pay/ipn")
async def pay_ipn(request: Request, x_nowpayments_sig: str = Header(None)):
    raw = await request.body()
    if not nowpay_verify_signature(raw, x_nowpayments_sig or ""):
        raise HTTPException(403, "Bad signature")

    body = json.loads(raw.decode("utf-8"))
    # NOWPayments sends fields like: payment_status, payment_id, order_id, price_amount, price_currency, actually_paid, pay_currency, etc.
    order_id = body.get("order_id") or body.get("orderId") or ""
    status   = (body.get("payment_status") or body.get("paymentStatus") or "").lower()

    # Accept statuses when funds are confirmed/finished
    paid_ok = status in ("finished", "confirmed", "partially_paid", "sending")  # broadened; you may restrict to finished/confirmed

    if not order_id:
        return {"ok": False, "reason": "missing order_id", "status": status}

    async with _orders_lock:
        ctx = _orders.get(order_id, {})
    tier = (ctx.get("tier") or "VIP").upper()
    telegram_id = int(ctx.get("telegram_id") or 0)

    if paid_ok and telegram_id:
        await _notify_user_paid(tier, telegram_id)

    # Optionally mark as fulfilled
    async with _orders_lock:
        if order_id in _orders:
            _orders[order_id]["last_status"] = status
            _orders[order_id]["updated_at"] = datetime.now(timezone.utc).isoformat()
            await save_orders()

    return {"ok": True, "order_id": order_id, "status": status, "notified": bool(paid_ok and telegram_id)}
