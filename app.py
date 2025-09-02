# app.py
# Pocket Option Signals — Accuracy-focused, Paid Tiers, Auto Subscriptions
# - Advanced MTF strategy (M15 trend → M5 context → M1 trigger) with tighter defaults
# - FX live window: local-time session parsing (America/Port_of_Spain) 08:00–16:00
# - OTC/weekend guard (no signals)
# - Per-instrument cooldown (default 10 min)
# - Quotas: FREE(3), BASIC(6), PRO(15), VIP(∞)
# - Daily/Weekly tallies broadcast to ALL groups (does NOT consume quotas)
# - NOWPayments invoices, IPN verify, subscription activate & auto-remove on expiry
# - Webhook-based Telegram bot (single Render service)
# ------------------------------------------------------------------------------------------

import os
import json
import hmac
import hashlib
import asyncio
import logging
from datetime import datetime, timedelta, timezone, date, time as dtime
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

try:
    from zoneinfo import ZoneInfo  # stdlib (Py3.9+)
except Exception:  # pragma: no cover
    ZoneInfo = None

# =========================
# ENV
# =========================
TELEGRAM_BOT_TOKEN   = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
BASE_URL             = os.getenv("BASE_URL", "").rstrip("/")
WEBHOOK_PATH         = os.getenv("WEBHOOK_PATH", "/telegram/webhook").strip()
WEBHOOK_SECRET       = os.getenv("WEBHOOK_SECRET", "change-me").strip()

# Groups & quotas
TELEGRAM_GROUP_ID    = os.getenv("TELEGRAM_GROUP_ID", "").strip()  # optional fallback VIP
TELEGRAM_CHAT_FREE   = os.getenv("TELEGRAM_CHAT_FREE", "").strip()
TELEGRAM_CHAT_BASIC  = os.getenv("TELEGRAM_CHAT_BASIC", "").strip()
TELEGRAM_CHAT_PRO    = os.getenv("TELEGRAM_CHAT_PRO", "").strip()
TELEGRAM_CHAT_VIP    = os.getenv("TELEGRAM_CHAT_VIP", "").strip()

LIMIT_FREE           = int(os.getenv("LIMIT_FREE", "3"))
LIMIT_BASIC          = int(os.getenv("LIMIT_BASIC", "6"))
LIMIT_PRO            = int(os.getenv("LIMIT_PRO", "15"))

# Optional static invite links (recommended)
TELEGRAM_LINK_BASIC  = os.getenv("TELEGRAM_LINK_BASIC", "").strip()
TELEGRAM_LINK_PRO    = os.getenv("TELEGRAM_LINK_PRO", "").strip()
TELEGRAM_LINK_VIP    = os.getenv("TELEGRAM_LINK_VIP", "").strip()

# OANDA
OANDA_API_KEY        = os.getenv("OANDA_API_KEY", "").strip()
OANDA_ACCOUNT_ID     = os.getenv("OANDA_ACCOUNT_ID", "").strip()
OANDA_ENV            = os.getenv("OANDA_ENV", "practice").strip().lower()
OANDA_INSTRUMENTS    = [s.strip() for s in os.getenv(
    "OANDA_INSTRUMENTS",
    "EUR_USD,AUD_CAD,AUD_CHF,AUD_JPY,AUD_USD,CAD_CHF,CAD_JPY,CHF_JPY,EUR_AUD,EUR_CAD,EUR_CHF,EUR_GBP,EUR_JPY,GBP_AUD,GBP_CAD,GBP_CHF,GBP_JPY,GBP_USD,USD_CAD,USD_CHF,USD_JPY"
).split(",") if s.strip()]
OANDA_GRANULARITY    = os.getenv("OANDA_GRANULARITY", "M1").strip().upper()

PO_EXPIRY_MIN        = int(os.getenv("PO_EXPIRY_MIN", "5"))
PO_ENTRY_DELAY_SEC   = int(os.getenv("PO_ENTRY_DELAY_SEC", "2"))
ALERT_COOLDOWN_MIN   = int(os.getenv("ALERT_COOLDOWN_MIN", "10"))  # stricter

# Baseline indicators (kept for fallback if needed)
RSI_MIN_BUY          = float(os.getenv("RSI_MIN_BUY", "50"))
RSI_MAX_SELL         = float(os.getenv("RSI_MAX_SELL", "50"))
EMA_FAST             = int(os.getenv("EMA_FAST", "9"))
EMA_SLOW             = int(os.getenv("EMA_SLOW", "21"))
ATR_FILTER_MULT      = float(os.getenv("ATR_FILTER_MULT", "0.0"))

# OTC/weekend gate
PO_BLOCK_DURING_OTC  = int(os.getenv("PO_BLOCK_DURING_OTC", "1"))  # keep enabled
FOREX_LIVE_24X5      = int(os.getenv("FOREX_LIVE_24X5", "1"))      # Mon–Fri live, Sat/Sun OTC
FOREX_DAILY_BLACKOUT_UTC = os.getenv("FOREX_DAILY_BLACKOUT_UTC", "20:58-21:05").strip()

# Local-time session window (your operating hours)
SESSION_TZ           = os.getenv("SESSION_TZ", "America/Port_of_Spain").strip()
SIGNAL_SESSION_LOCAL = os.getenv("SIGNAL_SESSION_LOCAL", "08:00-16:00").strip()  # your requirement
# You can also set SIGNAL_SESSION_UTC="HH:MM-HH:MM" instead; local takes precedence if set.

# NOWPayments (subscriptions)
NOWPAY_API_KEY       = os.getenv("NOWPAY_API_KEY", "").strip()
NOWPAY_IPN_SECRET    = os.getenv("NOWPAY_IPN_SECRET", "").strip()
NOWPAY_PRICE_CCY     = os.getenv("NOWPAY_PRICE_CCY", "usd").strip().lower()
NOWPAY_PAY_CCY       = os.getenv("NOWPAY_PAY_CCY", "usdttrc20").strip().lower()
PRICE_BASIC          = float(os.getenv("PRICE_BASIC", "29.0"))
PRICE_PRO            = float(os.getenv("PRICE_PRO", "49.0"))
PRICE_VIP            = float(os.getenv("PRICE_VIP", "79.0"))
NOWPAY_SUB_MONTHS    = int(os.getenv("NOWPAY_SUB_MONTHS", "1"))
SUB_GRACE_HOURS      = int(os.getenv("SUB_GRACE_HOURS", "12"))     # grace before kick
SUB_KICK_ENABLED     = int(os.getenv("SUB_KICK_ENABLED", "1"))

# Advanced strategy (accuracy-first)
ADVANCED_STRATEGY      = int(os.getenv("ADVANCED_STRATEGY", "1"))
SIGNAL_SESSION_UTC     = os.getenv("SIGNAL_SESSION_UTC", "").strip()  # optional if you prefer UTC
NEWS_UTC_BLOCKS        = os.getenv("NEWS_UTC_BLOCKS", "").strip()

ATR_PERCENTILE_MIN     = float(os.getenv("ATR_PERCENTILE_MIN", "0.45"))
ATR_PERCENTILE_MAX     = float(os.getenv("ATR_PERCENTILE_MAX", "0.90"))
BODY_ATR_MIN           = float(os.getenv("BODY_ATR_MIN", "0.40"))

M15_EMA_TREND_LEN      = int(os.getenv("M15_EMA_TREND_LEN", "200"))
M15_EMA_SLOPE_MIN_PIPS = float(os.getenv("M15_EMA_SLOPE_MIN_PIPS", "0.25"))  # tighter
M5_EMA_CONTEXT_LEN     = int(os.getenv("M5_EMA_CONTEXT_LEN", "50"))
M5_RSI_BULL            = float(os.getenv("M5_RSI_BULL", "56.0"))
M5_RSI_BEAR            = float(os.getenv("M5_RSI_BEAR", "44.0"))
CONFIDENCE_MIN         = int(os.getenv("CONFIDENCE_MIN", "5"))

# Schedules / files
SCHED_TZ             = timezone.utc
DATA_DIR             = os.getenv("DATA_DIR", "./data").strip()
STATS_FILE           = os.path.join(DATA_DIR, "stats.json")
OPEN_TRADES_FILE     = os.path.join(DATA_DIR, "open_trades.json")
QUOTA_FILE           = os.path.join(DATA_DIR, "quota.json")
UNSUPPORTED_FILE     = os.path.join(DATA_DIR, "unsupported.json")
ORDERS_FILE          = os.path.join(DATA_DIR, "orders.json")
LAST_SIGNAL_FILE     = os.path.join(DATA_DIR, "last_signal.json")
SUBS_FILE            = os.path.join(DATA_DIR, "subs.json")         # telegram_id -> {tier, expires_at}

# =========================
# Logging / Globals
# =========================
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("pocket-option-signals")

telegram_app: Application | None = None
scheduler: AsyncIOScheduler | None = None

_last_candle_time: Dict[str, datetime] = {}
_last_signal_time: Dict[str, str] = {}

_stats_lock = asyncio.Lock()
_trades_lock = asyncio.Lock()
_quota_lock = asyncio.Lock()
_unsupported_lock = asyncio.Lock()
_orders_lock = asyncio.Lock()
_subs_lock = asyncio.Lock()
_io_lock = asyncio.Lock()

_stats: Dict[str, Any] = {"daily": {}, "weekly": {}}
_open_trades: List[Dict[str, Any]] = []
_quota: Dict[str, Dict[str, int]] = {}
_unsupported: Set[str] = set()
_orders: Dict[str, Dict[str, Any]] = {}
_subs: Dict[str, Dict[str, Any]] = {}  # key = str(telegram_id)

# =========================
# FS helpers
# =========================
def _ensure_dirs():
    os.makedirs(DATA_DIR, exist_ok=True)

async def _read_json(path: str, default: Any):
    try:
        if os.path.exists(path):
            async with _io_lock:
                with open(path, "r") as f:
                    return json.load(f)
    except Exception as e:
        log.error(f"Read error {path}: {e}")
    return default

async def _write_json(path: str, data: Any):
    try:
        _ensure_dirs()
        async with _io_lock:
            with open(path, "w") as f:
                json.dump(data, f)
    except Exception as e:
        log.error(f"Write error {path}: {e}")

def _week_key(dt: datetime) -> str:
    iso = dt.isocalendar()
    return f"{iso.year}-W{iso.week:02d}"

async def load_state():
    global _stats, _open_trades, _quota, _unsupported, _orders, _last_signal_time, _subs
    _stats = await _read_json(STATS_FILE, {"daily": {}, "weekly": {}})
    _open_trades = await _read_json(OPEN_TRADES_FILE, [])
    _quota = await _read_json(QUOTA_FILE, {})
    _unsupported = set(await _read_json(UNSUPPORTED_FILE, []))
    _orders = await _read_json(ORDERS_FILE, {})
    _last_signal_time = await _read_json(LAST_SIGNAL_FILE, {})
    _subs = await _read_json(SUBS_FILE, {})

async def save_stats():        await _write_json(STATS_FILE, _stats)
async def save_open_trades():  await _write_json(OPEN_TRADES_FILE, _open_trades)
async def save_quota():        await _write_json(QUOTA_FILE, _quota)
async def save_unsupported():  await _write_json(UNSUPPORTED_FILE, sorted(list(_unsupported)))
async def save_orders():       await _write_json(ORDERS_FILE, _orders)
async def save_last_signal():  await _write_json(LAST_SIGNAL_FILE, _last_signal_time)
async def save_subs():         await _write_json(SUBS_FILE, _subs)

# =========================
# Telegram commands
# =========================
HELP_TEXT = (
    "Pocket Option Signals 🤖 (Accuracy Mode)\n"
    "• Tiers: FREE(3/d), BASIC(6/d), PRO(15/d), VIP(∞)\n"
    "• FX live only, Session: 08:00–16:00 local\n"
    "• W/L/D daily & weekly (broadcasted to all tiers)\n\n"
    "Commands:\n"
    "/start  — intro\n"
    "/ping   — health\n"
    "/plans  — prices\n"
    "/upgrade <BASIC|PRO|VIP> — crypto invoice\n"
)

async def start_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(HELP_TEXT)

async def ping_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("pong")

def _plan_price(tier: str) -> Optional[float]:
    return {"BASIC": PRICE_BASIC, "PRO": PRICE_PRO, "VIP": PRICE_VIP}.get(tier.upper())

def _plan_link(tier: str) -> str:
    return {"BASIC": TELEGRAM_LINK_BASIC, "PRO": TELEGRAM_LINK_PRO, "VIP": TELEGRAM_LINK_VIP}.get(tier.upper(), "")

async def plans_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = (
        f"Plans (monthly, pay in {NOWPAY_PAY_CCY.upper()}):\n"
        f"• BASIC — {PRICE_BASIC:.2f} {NOWPAY_PRICE_CCY.upper()} — 6/day\n"
        f"• PRO   — {PRICE_PRO:.2f} {NOWPAY_PRICE_CCY.upper()} — 15/day\n"
        f"• VIP   — {PRICE_VIP:.2f} {NOWPAY_PRICE_CCY.upper()} — Unlimited\n\n"
        f"Use /upgrade BASIC|PRO|VIP to get your invoice."
    )
    await update.message.reply_text(msg)

async def upgrade_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not NOWPAY_API_KEY:
        await update.message.reply_text("Payments unavailable. Try again later.")
        return
    args = context.args or []
    if len(args) < 1:
        await update.message.reply_text("Usage: /upgrade BASIC|PRO|VIP")
        return
    tier = args[0].upper()
    price = _plan_price(tier)
    if price is None:
        await update.message.reply_text("Unknown tier. Use BASIC, PRO, or VIP.")
        return

    user = update.effective_user
    user_id = user.id if user else 0
    order_id = f"tier:{tier}:uid:{user_id}:{int(datetime.now(timezone.utc).timestamp())}"
    description = f"Pocket Option Signals — {tier} plan for Telegram user {user_id}"

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
        await update.message.reply_text("Could not create invoice. Please try later.")
        return

    invoice_url = invoice.get("invoice_url") or invoice.get("url") or ""
    async with _orders_lock:
        _orders[order_id] = {"tier": tier, "telegram_id": user_id, "created_at": datetime.now(timezone.utc).isoformat()}
        await save_orders()
    await update.message.reply_text(
        f"✅ Invoice for {tier}: {price:.2f} {NOWPAY_PRICE_CCY.upper()}\nPay here: {invoice_url}\n"
        f"You'll receive your private invite link after confirmation."
    )

def build_telegram_app() -> Application:
    app = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", start_cmd))
    app.add_handler(CommandHandler("ping",  ping_cmd))
    app.add_handler(CommandHandler("plans", plans_cmd))
    app.add_handler(CommandHandler("upgrade", upgrade_cmd))
    return app

# =========================
# OANDA
# =========================
def oanda_base_url() -> str:
    return "https://api-fxtrade.oanda.com" if OANDA_ENV == "live" else "https://api-fxpractice.oanda.com"

def _auth_headers() -> Dict[str, str]:
    return {"Authorization": f"Bearer {OANDA_API_KEY}", "Content-Type": "application/json"}

async def fetch_oanda_candles(instrument: str, granularity: str, count: int = 300) -> pd.DataFrame:
    url = f"{oanda_base_url()}/v3/instruments/{instrument}/candles"
    params = {"count": str(count), "granularity": granularity, "price": "M"}
    async with httpx.AsyncClient(timeout=25) as client:
        r = await client.get(url, params=params, headers=_auth_headers())
        if r.status_code == 400:
            async with _unsupported_lock:
                if instrument not in _unsupported:
                    _unsupported.add(instrument)
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

# =========================
# Time windows: local session & news blocks
# =========================
def _parse_hhmm(s: str) -> dtime:
    hh, mm = s.split(":")
    return dtime(int(hh), int(mm))

def _build_utc_window_from_local(local_range: str) -> Tuple[dtime, dtime]:
    """Convert today's local window to UTC times based on SESSION_TZ."""
    if not ZoneInfo:
        raise RuntimeError("zoneinfo not available")
    tz = ZoneInfo(SESSION_TZ)
    start_s, end_s = [x.strip() for x in local_range.split("-")]
    today = date.today()
    # local datetimes
    ls = datetime.combine(today, _parse_hhmm(start_s), tzinfo=tz)
    le = datetime.combine(today, _parse_hhmm(end_s), tzinfo=tz)
    # to UTC times (same date; we only compare time-of-day)
    ls_utc = ls.astimezone(timezone.utc).time()
    le_utc = le.astimezone(timezone.utc).time()
    return ls_utc, le_utc

def _parse_blocks_csv(csv_ranges: str) -> List[Tuple[dtime, dtime]]:
    blocks = []
    if not csv_ranges:
        return blocks
    for part in csv_ranges.split(","):
        part = part.strip()
        if not part:
            continue
        a, b = part.split("-")
        blocks.append((_parse_hhmm(a), _parse_hhmm(b)))
    return blocks

def _time_in_blocks(t: dtime, blocks: List[Tuple[dtime, dtime]]) -> bool:
    for a, b in blocks:
        if a <= t <= b:
            return True
    return False

# build session blocks (prefer local)
try:
    _SESSION_UTC_RANGE: Optional[Tuple[dtime, dtime]] = _build_utc_window_from_local(SIGNAL_SESSION_LOCAL) if SIGNAL_SESSION_LOCAL else None
except Exception as e:
    log.warning(f"Local session parse failed, falling back to UTC: {e}")
    _SESSION_UTC_RANGE = None

_SESSION_UTC_BLOCKS = _parse_blocks_csv(SIGNAL_SESSION_UTC) if (SIGNAL_SESSION_UTC and not _SESSION_UTC_RANGE) else []
_NEWS_BLOCKS        = _parse_blocks_csv(NEWS_UTC_BLOCKS)

def in_signal_session(now_utc: Optional[datetime] = None) -> bool:
    now = (now_utc or datetime.now(timezone.utc))
    t = now.time()
    if _SESSION_UTC_RANGE:
        a, b = _SESSION_UTC_RANGE
        return a <= t <= b
    if _SESSION_UTC_BLOCKS:
        return _time_in_blocks(t, _SESSION_UTC_BLOCKS)
    return True  # no session filter

# =========================
# OTC / 24x5 FX gate
# =========================
def _parse_hhmm_utc(s: str) -> Tuple[int, int]:
    hh, mm = s.split(":")
    return int(hh), int(mm)

def _within_utc_range(now: datetime, start: str, end: str) -> bool:
    sh, sm = _parse_hhmm_utc(start)
    eh, em = _parse_hhmm_utc(end)
    a = now.replace(hour=sh, minute=sm, second=0, microsecond=0)
    b = now.replace(hour=eh, minute=em, second=0, microsecond=0)
    return a <= now <= b

def is_otc_now(now_utc: Optional[datetime] = None) -> bool:
    """True when we consider market 'OTC' (no signals)."""
    if PO_BLOCK_DURING_OTC != 1:
        return False
    now = (now_utc or datetime.now(timezone.utc))
    wd = now.weekday()  # Mon=0..Sun=6
    # 24x5 live by default: Sat/Sun OTC
    if FOREX_LIVE_24X5 == 1:
        if wd in (5, 6):
            return True
        blk = FOREX_DAILY_BLACKOUT_UTC
        if blk:
            try:
                a, b = blk.split("-")
                if _within_utc_range(now, a, b):
                    return True
            except Exception:
                pass
        return False
    # Fallback: always live weekdays in your session only
    return wd in (5, 6)

# =========================
# News block flag
# =========================
def in_news_block(now_utc: Optional[datetime] = None) -> bool:
    if not _NEWS_BLOCKS:
        return False
    now = (now_utc or datetime.now(timezone.utc))
    return _time_in_blocks(now.time(), _NEWS_BLOCKS)

# =========================
# ATR percentile & scoring
# =========================
def atr_percentile(series: pd.Series, lookback: int = 300) -> float:
    look = series.dropna().iloc[-lookback:] if len(series) >= lookback else series.dropna()
    if look.empty:
        return 0.0
    cur = look.iloc[-1]
    rank = (look <= cur).sum() / len(look)
    return float(rank)

def score_confidence(flags: List[bool]) -> int:
    return int(sum(1 for f in flags if f))

# =========================
# Strategy: Baseline (kept for fallback)
# =========================
def compute_signal_baseline(df: pd.DataFrame, idx: int) -> Dict[str, Any]:
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

    cur, prev = df.iloc[idx], df.iloc[idx - 1]
    if ATR_FILTER_MULT > 0 and (cur["atr14"] or 0) < ATR_FILTER_MULT * (cur["c"] * 1e-5):
        return {"signal": "HOLD", "reason": "Low volatility (ATR filter)"}

    signal, reason, direction = "HOLD", "No cross", None
    if (prev["ema_fast"] <= prev["ema_slow"]) and (cur["ema_fast"] > cur["ema_slow"]) and (cur["rsi14"] >= RSI_MIN_BUY):
        signal, reason, direction = "CALL", f"EMA{EMA_FAST}↑EMA{EMA_SLOW} & RSI≥{RSI_MIN_BUY}", "UP"
    elif (prev["ema_fast"] >= prev["ema_slow"]) and (cur["ema_fast"] < cur["ema_slow"]) and (cur["rsi14"] <= RSI_MAX_SELL):
        signal, reason, direction = "PUT",  f"EMA{EMA_FAST}↓EMA{EMA_SLOW} & RSI≤{RSI_MAX_SELL}", "DOWN"

    return {
        "signal": signal, "reason": reason, "direction": direction,
        "price": float(cur["c"]), "time": cur["time"].to_pydatetime().replace(tzinfo=timezone.utc),
        "ema_fast": float(cur["ema_fast"]), "ema_slow": float(cur["ema_slow"]), "rsi14": float(cur["rsi14"]),
        "atr14": float(cur["atr14"]), "confidence": 0
    }

# =========================
# Strategy: ADVANCED MTF (accuracy-first)
# =========================
async def compute_signal_mtf(inst: str, df1: pd.DataFrame, idx1: int) -> Dict[str, Any]:
    # higher TFs
    df5  = await fetch_oanda_candles(inst, "M5",  count=300)
    df15 = await fetch_oanda_candles(inst, "M15", count=300)
    if idx1 < 1 or df5.empty or df15.empty:
        return {"signal": "HOLD", "reason": "Not enough data"}

    # M15 trend
    ema200_15 = ta.ema(df15["c"].astype(float), length=M15_EMA_TREND_LEN)
    df15 = df15.copy(); df15["ema200"] = ema200_15
    if len(df15) < 6 or pd.isna(df15["ema200"].iloc[-1]):
        return {"signal": "HOLD", "reason": "M15 EMA200 insufficient"}
    slope_pips = (df15["ema200"].iloc[-1] - df15["ema200"].iloc[-6]) * 10000.0
    trend_up   = slope_pips >= M15_EMA_SLOPE_MIN_PIPS
    trend_down = slope_pips <= -M15_EMA_SLOPE_MIN_PIPS
    if not (trend_up or trend_down):
        return {"signal": "HOLD", "reason": "M15 slope too flat"}

    # M5 context
    ema50_5 = ta.ema(df5["c"].astype(float), length=M5_EMA_CONTEXT_LEN)
    rsi5    = ta.rsi(df5["c"].astype(float), length=14)
    df5 = df5.copy(); df5["ema50"] = ema50_5; df5["rsi"] = rsi5
    if pd.isna(df5["ema50"].iloc[-1]) or pd.isna(df5["rsi"].iloc[-1]):
        return {"signal": "HOLD", "reason": "M5 context insufficient"}
    price5 = float(df5["c"].iloc[-1])
    ema5   = float(df5["ema50"].iloc[-1])
    rsi5v  = float(df5["rsi"].iloc[-1])
    m5_bull = (price5 > ema5) and (rsi5v >= M5_RSI_BULL)
    m5_bear = (price5 < ema5) and (rsi5v <= M5_RSI_BEAR)

    # M1 trigger
    close1 = df1["c"].astype(float)
    ema_fast = ta.ema(close1, length=EMA_FAST)
    ema_slow = ta.ema(close1, length=EMA_SLOW)
    rsi1     = ta.rsi(close1, length=14)
    atr1     = ta.atr(df1["h"], df1["l"], close1, length=14)

    df1 = df1.copy()
    df1["ema_fast"] = ema_fast; df1["ema_slow"] = ema_slow
    df1["rsi1"]     = rsi1;     df1["atr1"]     = atr1

    cur1, prev1 = df1.iloc[idx1], df1.iloc[idx1 - 1]
    if any(pd.isna([cur1["ema_fast"], cur1["ema_slow"], cur1["rsi1"], cur1["atr1"]])):
        return {"signal": "HOLD", "reason": "M1 indicators insufficient"}

    # quality gates
    body = abs(float(cur1["c"]) - float(cur1["o"]))
    if float(cur1["atr1"]) <= 0 or body < BODY_ATR_MIN * float(cur1["atr1"]):
        return {"signal": "HOLD", "reason": "Weak candle body vs ATR"}
    atrp = atr_percentile(df1["atr1"], lookback=300)
    if not (ATR_PERCENTILE_MIN <= atrp <= ATR_PERCENTILE_MAX):
        return {"signal": "HOLD", "reason": "ATR percentile outside gate"}

    bull_trigger = (prev1["ema_fast"] <= prev1["ema_slow"]) and (cur1["ema_fast"] > cur1["ema_slow"]) and (cur1["rsi1"] >= 50.0)
    bear_trigger = (prev1["ema_fast"] >= prev1["ema_slow"]) and (cur1["ema_fast"] < cur1["ema_slow"]) and (cur1["rsi1"] <= 50.0)

    long_ok  = (slope_pips >= M15_EMA_SLOPE_MIN_PIPS) and m5_bull and bull_trigger
    short_ok = (slope_pips <= -M15_EMA_SLOPE_MIN_PIPS) and m5_bear and bear_trigger

    flags = [
        (slope_pips >= M15_EMA_SLOPE_MIN_PIPS) or (slope_pips <= -M15_EMA_SLOPE_MIN_PIPS),
        m5_bull if long_ok else m5_bear,
        bull_trigger if long_ok else bear_trigger,
        (ATR_PERCENTILE_MIN <= atrp <= ATR_PERCENTILE_MAX),
        body >= BODY_ATR_MIN * float(cur1["atr1"]),
        True  # placeholder to keep scale 0..6
    ]
    score = score_confidence(flags)

    if long_ok and score >= CONFIDENCE_MIN:
        return {
            "signal": "CALL", "reason": f"MTF long (score {score}, slope {slope_pips:.2f} pips)",
            "direction": "UP", "price": float(cur1["c"]),
            "time": cur1["time"].to_pydatetime().replace(tzinfo=timezone.utc),
            "ema_fast": float(cur1["ema_fast"]), "ema_slow": float(cur1["ema_slow"]),
            "rsi": float(cur1["rsi1"]), "atr": float(cur1["atr1"]),
            "confidence": score
        }
    if short_ok and score >= CONFIDENCE_MIN:
        return {
            "signal": "PUT", "reason": f"MTF short (score {score}, slope {slope_pips:.2f} pips)",
            "direction": "DOWN", "price": float(cur1["c"]),
            "time": cur1["time"].to_pydatetime().replace(tzinfo=timezone.utc),
            "ema_fast": float(cur1["ema_fast"]), "ema_slow": float(cur1["ema_slow"]),
            "rsi": float(cur1["rsi1"]), "atr": float(cur1["atr1"]),
            "confidence": score
        }
    return {"signal": "HOLD", "reason": f"No confluence (score {score})"}

# =========================
# Tiers / Quota
# =========================
def _tiers() -> List[Tuple[str, Optional[int], str]]:
    out: List[Tuple[str, Optional[int], str]] = []
    if TELEGRAM_CHAT_FREE:  out.append((TELEGRAM_CHAT_FREE,  LIMIT_FREE,  "FREE"))
    if TELEGRAM_CHAT_BASIC: out.append((TELEGRAM_CHAT_BASIC, LIMIT_BASIC, "BASIC"))
    if TELEGRAM_CHAT_PRO:   out.append((TELEGRAM_CHAT_PRO,   LIMIT_PRO,   "PRO"))
    if TELEGRAM_CHAT_VIP:   out.append((TELEGRAM_CHAT_VIP,   None,        "VIP"))
    if TELEGRAM_GROUP_ID:   out.append((TELEGRAM_GROUP_ID,   None,        "VIP"))
    return out

def _today_key() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")

async def quota_try_consume(chat_id: str, cap: Optional[int]) -> Tuple[bool, int, int]:
    if cap is None:
        return True, -1, -1
    day = _today_key()
    async with _quota_lock:
        used = int(_quota.get(chat_id, {}).get(day, 0))
        if used >= cap:
            return False, used, cap
        _quota.setdefault(chat_id, {})
        _quota[chat_id][day] = used + 1
        await save_quota()
        return True, used + 1, cap

async def send_to_tiers(text: str):
    """Send SIGNALS (consumes quota)."""
    targets = _tiers()
    if not targets:
        log.warning("No Telegram chats configured; skipping send.")
        return
    for chat_id, cap, label in targets:
        allowed, used_after, cap_val = await quota_try_consume(chat_id, cap)
        if not allowed:
            log.info(f"Quota reached for {label} ({chat_id}) {used_after}/{cap_val}. Skipping.")
            continue
        try:
            suffix = ""
            if cap_val != -1 and used_after != -1:
                suffix = f" [{label} {used_after}/{cap_val}]"
            await telegram_app.bot.send_message(chat_id=chat_id, text=text + suffix, disable_web_page_preview=True)
        except Exception as e:
            log.error(f"Telegram send error ({label}/{chat_id}): {e}")

async def send_to_all_chats_no_quota(text: str):
    """Broadcast reports/tallies WITHOUT consuming quotas."""
    for chat_id, _, label in _tiers():
        try:
            await telegram_app.bot.send_message(chat_id=chat_id, text=text, disable_web_page_preview=True)
        except Exception as e:
            log.error(f"Report send error ({label}/{chat_id}): {e}")

# =========================
# Settlement & stats
# =========================
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
        "instrument": inst, "direction": direction,
        "entry_price": entry_price, "entry_time": entry_time.isoformat(),
        "expiry_time": expiry_time.isoformat(),
        "settled": False, "result": None,
    }
    async with _trades_lock:
        _open_trades.append(trade)
        await save_open_trades()

def _bump_stats(dt: datetime, result: str):
    dkey = dt.strftime("%Y-%m-%d")
    wkey = _week_key(dt)
    _stats.setdefault("daily", {}); _stats.setdefault("weekly", {})
    _stats["daily"].setdefault(dkey, {"win": 0, "loss": 0, "draw": 0})
    _stats["weekly"].setdefault(wkey, {"win": 0, "loss": 0, "draw": 0})
    _stats["daily"][dkey][result.lower()] += 1
    _stats["weekly"][wkey][result.lower()] += 1

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
        f"Weekly ({wkey}) — W:{w['win']} L:{w['loss']} D:{w['draw']}\n"
        "Upgrade with /plans to get more signals and VIP access."
    )
    await send_to_all_chats_no_quota(msg)

# =========================
# Engine w/ cooldowns & gates
# =========================
def pocket_asset_name(instrument: str) -> str:
    return instrument.replace("_", "/")

def is_supported(inst: str) -> bool:
    return inst not in _unsupported

def _cooldown_ok(inst: str, now: datetime) -> bool:
    iso = _last_signal_time.get(inst)
    if not iso:
        return True
    try:
        last = datetime.fromisoformat(iso)
    except Exception:
        return True
    return (now - last) >= timedelta(minutes=ALERT_COOLDOWN_MIN)

async def _note_signal_time(inst: str, when: datetime):
    _last_signal_time[inst] = when.isoformat()
    await save_last_signal()

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
            return
        _last_candle_time[inst] = last_time

        now = datetime.now(timezone.utc)
        if is_otc_now(now):
            return
        if not in_signal_session(now):
            return
        if in_news_block(now):
            return
        if not _cooldown_ok(inst, now):
            return

        # Strategy
        if ADVANCED_STRATEGY == 1:
            res = await compute_signal_mtf(inst, df, idx)
        else:
            res = compute_signal_baseline(df, idx)

        sig = res.get("signal", "HOLD")
        if sig in ("CALL", "PUT"):
            entry_time = now + timedelta(seconds=PO_ENTRY_DELAY_SEC)
            asset = pocket_asset_name(inst)
            conf = res.get("confidence", 0)
            msg = (
                f"⚡️ PO Signal — {asset}\n"
                f"• Action: {sig} ({res.get('direction')})\n"
                f"• Price: {res.get('price')}\n"
                f"• TF: {OANDA_GRANULARITY} (MTF)\n"
                f"• Entry: {entry_time.strftime('%H:%M:%S UTC')} (+{PO_ENTRY_DELAY_SEC}s)\n"
                f"• Expiry: {PO_EXPIRY_MIN} min\n"
                f"• Conf: {conf}/{max(CONFIDENCE_MIN,6)}\n"
                f"• Reason: {res.get('reason')}\n"
                f"• Candle: {res.get('time').strftime('%Y-%m-%d %H:%M UTC')}\n"
                f"• Source: OANDA→PO"
            )
            await send_to_tiers(msg)
            await record_open_trade(inst, sig, float(res["price"]), entry_time, PO_EXPIRY_MIN)
            await _note_signal_time(inst, now)
        else:
            log.info(f"{inst}: HOLD ({res.get('reason')})")
    except httpx.HTTPStatusError as e:
        if e.response is not None and e.response.status_code == 400:
            async with _unsupported_lock:
                if inst not in _unsupported:
                    _unsupported.add(inst)
                    await save_unsupported()
            log.error(f"{inst} marked unsupported (400).")
        else:
            log.error(f"{inst} HTTP error: {e}")
    except Exception as e:
        log.error(f"{inst} error: {e}")

async def run_engine():
    if not OANDA_API_KEY or not OANDA_ACCOUNT_ID:
        log.warning("OANDA credentials missing; engine skipped.")
        return
    for i in OANDA_INSTRUMENTS:
        await process_instrument(i)

# =========================
# NOWPayments
# =========================
NOWPAY_BASE = "https://api.nowpayments.io/v1"

async def nowpay_create_invoice(
    *, price_amount: float, price_currency: str, order_id: str, order_description: str,
    pay_currency: Optional[str], success_url: str, cancel_url: str, ipn_callback_url: str,
) -> Dict[str, Any]:
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
    try:
        if not NOWPAY_IPN_SECRET:
            return False
        body = json.loads(raw_body.decode("utf-8"))
        canonical = json.dumps(body, sort_keys=True, separators=(",", ":"))
        calc = hmac.new(NOWPAY_IPN_SECRET.encode("utf-8"), msg=canonical.encode("utf-8"), digestmod=hashlib.sha512).hexdigest()
        return hmac.compare_digest(calc, (signature or "").lower())
    except Exception as e:
        log.error(f"IPN verify error: {e}")
        return False

def _invite_for_tier(tier: str) -> str:
    return {"BASIC": TELEGRAM_LINK_BASIC, "PRO": TELEGRAM_LINK_PRO, "VIP": TELEGRAM_LINK_VIP}.get(tier.upper(), "")

async def _notify_user_paid(tier: str, telegram_id: int):
    if not telegram_id:
        return
    link = _invite_for_tier(tier)
    try:
        text = (
            f"✅ Payment confirmed for *{tier}*.\n"
            f"Join your private group: {link if link else '(ask admin for invite)'}"
        )
        await telegram_app.bot.send_message(chat_id=telegram_id, text=text, disable_web_page_preview=True)
    except Exception as e:
        log.error(f"DM to user {telegram_id} failed: {e}")

def _tier_chat_id(tier: str) -> Optional[str]:
    t = tier.upper()
    if t == "BASIC": return TELEGRAM_CHAT_BASIC or None
    if t == "PRO":   return TELEGRAM_CHAT_PRO or None
    if t == "VIP":   return TELEGRAM_CHAT_VIP or TELEGRAM_GROUP_ID or None
    return None

async def _kick_from_paid_chats(telegram_id: int, tiers: List[str]):
    """Remove a user from associated paid chats after grace period. Bot must be admin."""
    for t in tiers:
        chat_id = _tier_chat_id(t)
        if not chat_id:
            continue
        try:
            await telegram_app.bot.ban_chat_member(chat_id=chat_id, user_id=telegram_id)
            await telegram_app.bot.unban_chat_member(chat_id=chat_id, user_id=telegram_id)  # kick without ban
        except Exception as e:
            log.error(f"Kick failed uid={telegram_id} from {t}/{chat_id}: {e}")

# =========================
# Webhook lifecycle
# =========================
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
    except Exception:
        pass
    try:
        await telegram_app.stop()
        await telegram_app.shutdown()
    except Exception:
        pass

# =========================
# FastAPI & schedulers
# =========================
app = FastAPI()

@asynccontextmanager
async def lifespan(app_: FastAPI):
    global telegram_app, scheduler
    if not TELEGRAM_BOT_TOKEN:
        log.error("❌ TELEGRAM_BOT_TOKEN missing.")
        yield
        return

    await load_state()
    telegram_app = build_telegram_app()
    await set_webhook()

    scheduler = AsyncIOScheduler(
        timezone=SCHED_TZ,
        job_defaults={"coalesce": True, "max_instances": 1, "misfire_grace_time": 20},
    )
    scheduler.add_job(run_engine, CronTrigger(second=5), id="engine")
    scheduler.add_job(settle_due_trades, CronTrigger(second="*/30"), id="settle")
    # Daily tallies (to ALL groups, no quota)
    scheduler.add_job(send_daily_weekly_reports, CronTrigger(hour=23, minute=59, second=30), id="daily")
    scheduler.add_job(send_daily_weekly_reports, CronTrigger(day_of_week="sun", hour=23, minute=59, second=40), id="weekly")
    # Subscription expiry checks (hourly)
    scheduler.add_job(check_subscriptions, CronTrigger(minute=0, second=10), id="subs_check")
    scheduler.start()
    log.info("⏱️ Scheduler started.")
    try:
        yield
    finally:
        try:
            if scheduler:
                scheduler.shutdown(wait=False)
        except Exception:
            pass
        await unset_webhook()

app.router.lifespan_context = lifespan

# =========================
# Routes
# =========================
@app.get("/", response_class=PlainTextResponse)
async def root_get():
    return "PO Signals — Accuracy Mode (FX live, tiers, quotas, paywall, auto subs)"

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
    async with _subs_lock:
        active_subs = sum(1 for v in _subs.values() if v.get("expires_at") and datetime.fromisoformat(v["expires_at"]) > now)
    return JSONResponse({
        "ok": True,
        "otc_now": is_otc_now(),
        "in_session": in_signal_session(),
        "in_news_block": in_news_block(),
        "oanda_env": OANDA_ENV,
        "instruments": OANDA_INSTRUMENTS,
        "granularity": OANDA_GRANULARITY,
        "expiry_min": PO_EXPIRY_MIN,
        "entry_delay_sec": PO_ENTRY_DELAY_SEC,
        "daily": daily,
        "weekly": weekly,
        "quota_today": quotas_today,
        "unsupported": unsupported,
        "cooldown_min": ALERT_COOLDOWN_MIN,
        "advanced_strategy": ADVANCED_STRATEGY,
        "confidence_min": CONFIDENCE_MIN,
        "atr_percentile_gate": [ATR_PERCENTILE_MIN, ATR_PERCENTILE_MAX],
        "body_atr_min": BODY_ATR_MIN,
        "session_local": SIGNAL_SESSION_LOCAL,
        "session_tz": SESSION_TZ,
        "active_subscriptions": active_subs,
    })

# --- Payments ---
@app.get("/pay/create")
async def pay_create(
    tier: str = Query(..., pattern="^(?i)(BASIC|PRO|VIP)$"),
    telegram_id: int = Query(...),
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
    return f"Payment success (Tier={tier})."

@app.get("/pay/cancel", response_class=PlainTextResponse)
async def pay_cancel(tier: str = "UNKNOWN"):
    return f"Payment canceled (Tier={tier})."

@app.post("/pay/ipn")
async def pay_ipn(request: Request, x_nowpayments_sig: str = Header(None)):
    raw = await request.body()
    if not nowpay_verify_signature(raw, x_nowpayments_sig or ""):
        raise HTTPException(403, "Bad signature")
    body = json.loads(raw.decode("utf-8"))
    order_id = body.get("order_id") or body.get("orderId") or ""
    status   = (body.get("payment_status") or body.get("paymentStatus") or "").lower()

    if not order_id:
        return {"ok": False, "reason": "missing order_id", "status": status}

    async with _orders_lock:
        ctx = _orders.get(order_id, {})

    tier = (ctx.get("tier") or "VIP").upper()
    telegram_id = int(ctx.get("telegram_id") or 0)

    # Activate on confirmed payments
    paid_ok  = status in ("finished", "confirmed", "partially_paid", "sending")
    if paid_ok and telegram_id:
        # set/extend subscription
        async with _subs_lock:
            current = _subs.get(str(telegram_id), {})
            start = datetime.now(timezone.utc)
            # if still active, extend from existing expiry
            base = datetime.fromisoformat(current["expires_at"]).astimezone(timezone.utc) if current.get("expires_at") else start
            if base < start:
                base = start
            new_expiry = base + timedelta(days=30 * NOWPAY_SUB_MONTHS)
            _subs[str(telegram_id)] = {"tier": tier, "expires_at": new_expiry.isoformat()}
            await save_subs()

        await _notify_user_paid(tier, telegram_id)

    # store last known status
    async with _orders_lock:
        if order_id in _orders:
            _orders[order_id]["last_status"] = status
            _orders[order_id]["updated_at"] = datetime.now(timezone.utc).isoformat()
            await save_orders()
    return {"ok": True, "order_id": order_id, "status": status, "activated": bool(paid_ok and telegram_id)}

# =========================
# Subscription checks
# =========================
async def check_subscriptions():
    """Kick expired paid users (after grace)."""
    if SUB_KICK_ENABLED != 1:
        return
    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(hours=SUB_GRACE_HOURS)
    expired: List[int] = []
    async with _subs_lock:
        for uid, rec in _subs.items():
            exp_s = rec.get("expires_at")
            tier = (rec.get("tier") or "").upper()
            if not exp_s or not tier:
                continue
            try:
                exp = datetime.fromisoformat(exp_s).astimezone(timezone.utc)
            except Exception:
                continue
            if exp < cutoff:
                expired.append(int(uid))
    for uid in expired:
        try:
            await _kick_from_paid_chats(uid, ["BASIC", "PRO", "VIP"])
        except Exception as e:
            log.error(f"Kick error uid={uid}: {e}")

# =========================
# End
# =========================
