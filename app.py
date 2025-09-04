# app.py
# Pocket Option Signals — BLW M1→5 (FX live only) + Exact-Expiry Settlement via Deriv
# Features:
# - Exact pairs (your winners list)
# - 1m candles, 5m expiry (configurable), per-instrument cooldown
# - FX live window (Mon–Fri 08:00–16:00 America/Port_of_Spain)
# - Per-tier Telegram quotas (FREE/BASIC/PRO/VIP)
# - Daily & weekly tallies to all groups
# - NOWPayments invoices + IPN to deliver plan links
# - Exact settlement on Deriv last tick at expiry (fallback to OANDA M1 close)
# ------------------------------------------------------------------------------------

import os
import json
import hmac
import hashlib
import asyncio
import logging
from typing import Dict, Any, List, Optional, Tuple, Set
from datetime import datetime, timedelta, timezone, time as dtime

import pandas as pd
import pandas_ta as ta
import httpx
import websockets

from fastapi import FastAPI, Request, Header
from fastapi.responses import PlainTextResponse, JSONResponse

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes

# =========================
# ENV
# =========================
TELEGRAM_BOT_TOKEN   = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
BASE_URL             = os.getenv("BASE_URL", "").rstrip("/")
WEBHOOK_PATH         = os.getenv("WEBHOOK_PATH", "/telegram/webhook").strip()
WEBHOOK_SECRET       = os.getenv("WEBHOOK_SECRET", "change-me").strip()
ENABLE_BOT_POLLING   = os.getenv("ENABLE_BOT_POLLING", "0").strip()

# Chats (as strings; they can be negative for channels)
TELEGRAM_CHAT_FREE   = os.getenv("TELEGRAM_CHAT_FREE", "").strip()
TELEGRAM_CHAT_BASIC  = os.getenv("TELEGRAM_CHAT_BASIC", "").strip()
TELEGRAM_CHAT_PRO    = os.getenv("TELEGRAM_CHAT_PRO", "").strip()
TELEGRAM_CHAT_VIP    = os.getenv("TELEGRAM_CHAT_VIP", "").strip()

# Invite links (optional; you can leave blank if you add users manually)
TELEGRAM_LINK_FREE   = os.getenv("TELEGRAM_LINK_FREE", "").strip()
TELEGRAM_LINK_BASIC  = os.getenv("TELEGRAM_LINK_BASIC", "").strip()
TELEGRAM_LINK_PRO    = os.getenv("TELEGRAM_LINK_PRO", "").strip()
TELEGRAM_LINK_VIP    = os.getenv("TELEGRAM_LINK_VIP", "").strip()

# Quotas
LIMIT_FREE           = int(os.getenv("LIMIT_FREE", "3"))
LIMIT_BASIC          = int(os.getenv("LIMIT_BASIC", "6"))
LIMIT_PRO            = int(os.getenv("LIMIT_PRO", "15"))
# VIP is unlimited

# Strategy (BLW M1→5 tuned)
PO_EXPIRY_MIN        = int(os.getenv("PO_EXPIRY_MIN", "5"))
ALERT_COOLDOWN_MIN   = int(os.getenv("ALERT_COOLDOWN_MIN", "10"))
ADX_MIN              = float(os.getenv("ADX_MIN", "18"))
BODY_ATR_MIN         = float(os.getenv("BODY_ATR_MIN", "0.30"))
ATR_PCTL_MIN         = float(os.getenv("ATR_PCTL_MIN", "0.30"))
ATR_PCTL_MAX         = float(os.getenv("ATR_PCTL_MAX", "0.92"))
RSI_UP               = float(os.getenv("RSI_UP", "52"))
RSI_DN               = float(os.getenv("RSI_DN", "48"))

# OANDA (for candles and fallback settlement)
OANDA_API_KEY        = os.getenv("OANDA_API_KEY", "").strip()
OANDA_ACCOUNT_ID     = os.getenv("OANDA_ACCOUNT_ID", "").strip()
OANDA_ENV            = os.getenv("OANDA_ENV", "practice").strip().lower()
OANDA_GRANULARITY    = os.getenv("OANDA_GRANULARITY", "M1").strip().upper()

# EXACT WINNER PAIRS (defaulted; override via env OANDA_INSTRUMENTS if you need)
OANDA_INSTRUMENTS    = [s.strip() for s in os.getenv(
    "OANDA_INSTRUMENTS",
    "AUD_USD,EUR_GBP,EUR_JPY,AUD_CHF,CAD_JPY,GBP_AUD,EUR_CAD,EUR_CHF,AUD_CAD,CAD_CHF,CHF_JPY,USD_CHF,GBP_CHF"
).split(",") if s.strip()]

# Session / OTC guard (FX live only)
SESSION_TZ_NAME      = os.getenv("SESSION_TZ", "America/Port_of_Spain")
LIVE_START_LOCAL     = os.getenv("LIVE_START_LOCAL", "08:00")  # 08:00 local
LIVE_END_LOCAL       = os.getenv("LIVE_END_LOCAL", "16:00")    # 16:00 local

# NOWPayments
NOWPAY_API_KEY       = os.getenv("NOWPAY_API_KEY", "").strip()
NOWPAY_IPN_SECRET    = os.getenv("NOWPAY_IPN_SECRET", "").strip()
NOWPAY_PRICE_CCY     = os.getenv("NOWPAY_PRICE_CCY", "usd").strip().lower()
NOWPAY_PAY_CCY       = os.getenv("NOWPAY_PAY_CCY", "usdttrc20").strip().lower()
PRICE_BASIC          = float(os.getenv("PRICE_BASIC", "29.0"))
PRICE_PRO            = float(os.getenv("PRICE_PRO", "49.0"))
PRICE_VIP            = float(os.getenv("PRICE_VIP", "79.0"))

# Deriv ticks endpoint for exact settlement
DERIV_ENDPOINT       = os.getenv("DERIV_ENDPOINT", "wss://ws.deriv.com/websockets/v3?app_id=1089").strip()

# Storage
DATA_DIR             = os.getenv("DATA_DIR", "./data").strip()
STATS_FILE           = os.path.join(DATA_DIR, "stats.json")
OPEN_TRADES_FILE     = os.path.join(DATA_DIR, "open_trades.json")
QUOTA_FILE           = os.path.join(DATA_DIR, "quota.json")
LAST_SIGNAL_FILE     = os.path.join(DATA_DIR, "last_signal.json")
UNSUPPORTED_FILE     = os.path.join(DATA_DIR, "unsupported.json")
ORDERS_FILE          = os.path.join(DATA_DIR, "orders.json")

# Scheduler / globals
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("pocket-option-signals")

app_fastapi = FastAPI()
telegram_app: Optional[Application] = None
scheduler: Optional[AsyncIOScheduler] = None

_stats_lock = asyncio.Lock()
_trades_lock = asyncio.Lock()
_quota_lock = asyncio.Lock()
_io_lock = asyncio.Lock()
_orders_lock = asyncio.Lock()
_unsupported_lock = asyncio.Lock()

_stats: Dict[str, Any] = {"daily": {}, "weekly": {}}
_open_trades: List[Dict[str, Any]] = []
_quota: Dict[str, Dict[str, int]] = {}
_last_signal_time: Dict[str, str] = {}
_unsupported: Set[str] = set()
_orders: Dict[str, Any] = {}

# =========================
# Utils & FS
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

async def load_state():
    global _stats, _open_trades, _quota, _last_signal_time, _unsupported, _orders
    _stats = await _read_json(STATS_FILE, {"daily": {}, "weekly": {}})
    _open_trades = await _read_json(OPEN_TRADES_FILE, [])
    _quota = await _read_json(QUOTA_FILE, {})
    _last_signal_time = await _read_json(LAST_SIGNAL_FILE, {})
    _unsupported = set(await _read_json(UNSUPPORTED_FILE, []))
    _orders = await _read_json(ORDERS_FILE, {})

async def save_stats():        await _write_json(STATS_FILE, _stats)
async def save_open_trades():  await _write_json(OPEN_TRADES_FILE, _open_trades)
async def save_quota():        await _write_json(QUOTA_FILE, _quota)
async def save_last_signal():  await _write_json(LAST_SIGNAL_FILE, _last_signal_time)
async def save_unsupported():  await _write_json(UNSUPPORTED_FILE, sorted(list(_unsupported)))
async def save_orders():       await _write_json(ORDERS_FILE, _orders)

# =========================
# Session time
# =========================
try:
    from zoneinfo import ZoneInfo
except Exception:  # pragma: no cover
    from backports.zoneinfo import ZoneInfo

def _parse_hhmm(s: str) -> dtime:
    hh, mm = s.split(":")
    return dtime(int(hh), int(mm))

LOCAL_TZ = ZoneInfo(SESSION_TZ_NAME)
LIVE_START = _parse_hhmm(LIVE_START_LOCAL)
LIVE_END   = _parse_hhmm(LIVE_END_LOCAL)

def fx_live_now(now_utc: Optional[datetime] = None) -> bool:
    now = (now_utc or datetime.now(timezone.utc)).astimezone(LOCAL_TZ)
    if now.weekday() > 4:  # Sat, Sun
        return False
    t = now.time()
    return LIVE_START <= t <= LIVE_END

# =========================
# OANDA helpers
# =========================
def oanda_base_url() -> str:
    return "https://api-fxtrade.oanda.com" if OANDA_ENV == "live" else "https://api-fxpractice.oanda.com"

def auth_headers() -> Dict[str, str]:
    return {"Authorization": f"Bearer {OANDA_API_KEY}", "Content-Type": "application/json"}

async def fetch_oanda(instrument: str, granularity: str, count: int = 500) -> pd.DataFrame:
    url = f"{oanda_base_url()}/v3/instruments/{instrument}/candles"
    params = {"count": str(count), "granularity": granularity, "price": "M"}
    async with httpx.AsyncClient(timeout=25) as client:
        r = await client.get(url, params=params, headers=auth_headers())
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
                "time": pd.to_datetime(c["time"]),
                "o": float(mid.get("o", "nan")),
                "h": float(mid.get("h", "nan")),
                "l": float(mid.get("l", "nan")),
                "c": float(mid.get("c", "nan")),
                "volume": int(c.get("volume", 0)),
                "complete": bool(c.get("complete", False)),
            })
        except Exception:
            continue
    df = pd.DataFrame(rows).dropna().sort_values("time").reset_index(drop=True)
    return df

def last_completed_idx(df: pd.DataFrame) -> Optional[int]:
    if df.empty:
        return None
    if bool(df.iloc[-1].get("complete", False)):
        return int(df.index[-1])
    if len(df) >= 2 and bool(df.iloc[-2].get("complete", False)):
        return int(df.index[-2])
    return None

def atr_percentile(series: pd.Series, lookback: int = 300) -> float:
    s = series.dropna()
    if len(s) == 0:
        return 0.0
    if len(s) > lookback:
        s = s.iloc[-lookback:]
    cur = s.iloc[-1]
    rank = (s <= cur).sum() / len(s)
    return float(rank)

# =========================
# Deriv exact settlement (ticks)
# =========================
DERIV_SYMBOLS = {
    "EUR_USD": "frxEURUSD", "GBP_USD": "frxGBPUSD", "USD_JPY": "frxUSDJPY",
    "USD_CHF": "frxUSDCHF", "USD_CAD": "frxUSDCAD", "EUR_JPY": "frxEURJPY",
    "GBP_JPY": "frxGBPJPY", "EUR_GBP": "frxEURGBP", "AUD_USD": "frxAUDUSD",
    "AUD_JPY": "frxAUDJPY", "AUD_CAD": "frxAUDCAD", "AUD_CHF": "frxAUDCHF",
    "CAD_JPY": "frxCADJPY", "CAD_CHF": "frxCADCHF", "CHF_JPY": "frxCHFJPY",
    "EUR_CHF": "frxEURCHF", "EUR_AUD": "frxEURAUD", "EUR_CAD": "frxEURCAD",
    "GBP_AUD": "frxGBPAUD", "GBP_CAD": "frxGBPCAD", "GBP_CHF": "frxGBPCHF",
}

async def _deriv_ticks_history(symbol: str, start_epoch: int, end_epoch: int) -> list:
    if not DERIV_ENDPOINT:
        raise RuntimeError("DERIV_ENDPOINT not set")
    req = {
        "ticks_history": symbol,
        "start": start_epoch,
        "end": end_epoch,
        "style": "ticks",
        "adjust_start_time": 1,
    }
    async with websockets.connect(DERIV_ENDPOINT, ping_interval=20, ping_timeout=20) as ws:
        await ws.send(json.dumps(req))
        resp = json.loads(await ws.recv())
    if "error" in resp:
        raise RuntimeError(f"Deriv ticks_history error: {resp['error']}")
    # Normalize shapes
    if "history" in resp and "prices" in resp["history"] and "times" in resp["history"]:
        return [{"epoch": int(t), "quote": float(p)} for t, p in zip(resp["history"]["times"], resp["history"]["prices"])]
    if "ticks" in resp and isinstance(resp["ticks"], list):
        return resp["ticks"]
    if "prices" in resp and "times" in resp:
        return [{"epoch": int(t), "quote": float(p)} for t, p in zip(resp["times"], resp["prices"])]
    return resp.get("candles") or resp.get("history") or []

async def _deriv_ticks_series(symbol: str, start_epoch: int, end_epoch: int) -> List[Tuple[int, float]]:
    raw = await _deriv_ticks_history(symbol, start_epoch, end_epoch)
    out: List[Tuple[int, float]] = []
    for r in raw:
        try:
            ep = int(r.get("epoch") or r.get("time") or r.get("t"))
            px = float(r.get("quote") or r.get("price") or r.get("p") or r.get("close"))
            if ep and px:
                out.append((ep, px))
        except Exception:
            continue
    out.sort(key=lambda x: x[0])
    return out

async def price_at_expiry(symbol_fx: str, expiry_dt_utc: datetime, lookback_sec: int = 90, lookahead_sec: int = 15) -> Optional[float]:
    deriv_symbol = DERIV_SYMBOLS.get(symbol_fx)
    if not deriv_symbol:
        return None
    exp_epoch = int(expiry_dt_utc.timestamp())
    start_ep  = exp_epoch - max(1, lookback_sec)
    end_ep    = exp_epoch + max(1, lookahead_sec)
    try:
        ticks = await _deriv_ticks_series(deriv_symbol, start_ep, end_ep)
    except Exception as e:
        log.error(f"Deriv ticks fetch failed for {symbol_fx}: {e}")
        return None
    if not ticks:
        return None
    le = [t for t in ticks if t[0] <= exp_epoch]
    if le:
        return float(le[-1][1])
    ge = [t for t in ticks if t[0] > exp_epoch]
    if ge:
        return float(ge[0][1])
    return None

# =========================
# BLW M1→5 Strategy (1m trigger, 5m expiry)
# =========================
def compute_blw_m1x5(inst: str, m1: pd.DataFrame, idx: Optional[int]) -> Dict[str, Any]:
    if idx is None or idx < 1 or m1.empty:
        return {"signal": "HOLD", "reason": "insufficient data"}

    df = m1.copy()
    close = df["c"].astype(float)

    df["ema9"]  = ta.ema(close, length=9)
    df["ema21"] = ta.ema(close, length=21)
    df["rsi"]   = ta.rsi(close, length=14)
    df["atr"]   = ta.atr(df["h"], df["l"], close, length=14)
    adx_df = ta.adx(df["h"], df["l"], close, length=14)
    df["adx"]   = adx_df["ADX_14"] if adx_df is not None and "ADX_14" in adx_df.columns else pd.Series([None]*len(df))

    if idx >= len(df):
        idx = len(df) - 1
    cur, prev = df.iloc[idx], df.iloc[idx - 1]

    # must be completed candle
    if not bool(cur.get("complete", False)):
        return {"signal": "HOLD", "reason": "last candle not complete"}

    # body filter
    body = abs(float(cur["c"]) - float(cur["o"]))
    if float(cur["atr"]) <= 0 or body < BODY_ATR_MIN * float(cur["atr"]):
        return {"signal": "HOLD", "reason": "small body vs ATR"}

    # volatility percentile
    p_atr = atr_percentile(df["atr"])
    if not (ATR_PCTL_MIN <= p_atr <= ATR_PCTL_MAX):
        return {"signal": "HOLD", "reason": f"ATR pct {p_atr:.2f} out of window"}

    # trend strength
    try:
        adxv = float(cur["adx"])
    except Exception:
        adxv = 0.0
    if adxv < ADX_MIN:
        return {"signal": "HOLD", "reason": f"ADX {adxv:.1f} < {ADX_MIN}"}

    # EMA cross + RSI side
    signal, direction, reason = "HOLD", None, "no cross"
    if (prev["ema9"] <= prev["ema21"]) and (cur["ema9"] > cur["ema21"]) and float(cur["rsi"]) >= RSI_UP:
        signal, direction, reason = "CALL", "UP", f"EMA9↑EMA21 + RSI≥{RSI_UP}"
    elif (prev["ema9"] >= prev["ema21"]) and (cur["ema9"] < cur["ema21"]) and float(cur["rsi"]) <= RSI_DN:
        signal, direction, reason = "PUT", "DOWN", f"EMA9↓EMA21 + RSI≤{RSI_DN}"

    return {
        "signal": signal,
        "direction": direction,
        "reason": reason,
        "time": pd.to_datetime(cur["time"]).to_pydatetime().replace(tzinfo=timezone.utc),
        "price": float(cur["c"]),
    }

# =========================
# Telegram commands
# =========================
HELP_TEXT = (
    "Pocket Option BLW (M1→5) 🤖\n"
    "• FX live only (Mon–Fri 08:00–16:00 America/Port_of_Spain)\n"
    "• Tiers: FREE(3/d), BASIC(6/d), PRO(15/d), VIP(∞)\n"
    "• 1m candles, 5m expiry, 10m per-pair cooldown\n"
    "• Auto daily & weekly tallies\n\n"
    "Commands:\n"
    "/start — intro\n"
    "/ping  — health\n"
    "/limits — today’s remaining quota\n"
    "/plans — pricing\n"
    "/upgrade <BASIC|PRO|VIP> — crypto invoice\n"
)

def build_telegram_app() -> Application:
    app = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", start_cmd))
    app.add_handler(CommandHandler("ping", ping_cmd))
    app.add_handler(CommandHandler("limits", limits_cmd))
    app.add_handler(CommandHandler("plans", plans_cmd))
    app.add_handler(CommandHandler("upgrade", upgrade_cmd))
    return app

async def start_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(HELP_TEXT)

async def ping_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("pong")

def resolve_tier_and_limit(chat_id: str) -> Tuple[str, Optional[int]]:
    if TELEGRAM_CHAT_VIP and chat_id == TELEGRAM_CHAT_VIP:
        return ("VIP", None)
    if TELEGRAM_CHAT_PRO and chat_id == TELEGRAM_CHAT_PRO:
        return ("PRO", LIMIT_PRO)
    if TELEGRAM_CHAT_BASIC and chat_id == TELEGRAM_CHAT_BASIC:
        return ("BASIC", LIMIT_BASIC)
    if TELEGRAM_CHAT_FREE and chat_id == TELEGRAM_CHAT_FREE:
        return ("FREE", LIMIT_FREE)
    return ("FREE", LIMIT_FREE)

async def limits_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id if update.effective_chat else None
    if chat_id is None:
        return
    tier, lim = resolve_tier_and_limit(str(chat_id))
    used = await quota_get_today(str(chat_id))
    rem = "∞" if lim is None else max(lim - used, 0)
    await update.message.reply_text(f"Tier: {tier}\nUsed today: {used}\nRemaining: {rem}")

async def tg_send(chat_id: str, text: str):
    if not telegram_app:
        return
    try:
        await telegram_app.bot.send_message(chat_id=chat_id, text=text, disable_web_page_preview=True)
    except Exception as e:
        log.error(f"Telegram send failed to {chat_id}: {e}")

# Pricing + Upgrade (NOWPayments)
async def plans_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = (
        f"Plans (monthly, pay in {NOWPAY_PAY_CCY.upper()}):\n"
        f"• BASIC — {PRICE_BASIC:.2f} {NOWPAY_PRICE_CCY.upper()} — 6/day\n"
        f"• PRO   — {PRICE_PRO:.2f} {NOWPAY_PRICE_CCY.upper()} — 15/day\n"
        f"• VIP   — {PRICE_VIP:.2f} {NOWPAY_PRICE_CCY.upper()} — Unlimited\n\n"
        f"Use /upgrade BASIC|PRO|VIP to get your invoice."
    )
    await update.message.reply_text(msg)

async def nowpay_create_invoice(price_amount: float, price_currency: str, order_id: str,
                                order_description: str, pay_currency: Optional[str] = None,
                                success_url: Optional[str] = None, cancel_url: Optional[str] = None,
                                ipn_callback_url: Optional[str] = None) -> Dict[str, Any]:
    url = "https://api.nowpayments.io/v1/invoice"
    payload = {
        "price_amount": price_amount,
        "price_currency": price_currency,
        "order_id": order_id,
        "order_description": order_description,
    }
    if pay_currency:
        payload["pay_currency"] = pay_currency
    if success_url:
        payload["success_url"] = success_url
    if cancel_url:
        payload["cancel_url"] = cancel_url
    if ipn_callback_url:
        payload["ipn_callback_url"] = ipn_callback_url

    headers = {"x-api-key": NOWPAY_API_KEY, "Content-Type": "application/json"}
    async with httpx.AsyncClient(timeout=25) as client:
        r = await client.post(url, json=payload, headers=headers)
        r.raise_for_status()
        return r.json()

def _plan_price(tier: str) -> Optional[float]:
    return {"BASIC": PRICE_BASIC, "PRO": PRICE_PRO, "VIP": PRICE_VIP}.get(tier.upper())

def _plan_link(tier: str) -> str:
    return {"BASIC": TELEGRAM_LINK_BASIC, "PRO": TELEGRAM_LINK_PRO, "VIP": TELEGRAM_LINK_VIP}.get(tier.upper(), "")

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
            ipn_callback_url=f"{BASE_URL}/pay/ipn",
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
        f"✅ Invoice for {tier}: {price:.2f} {NOWPAY_PRICE_CCY.upper()}\n"
        f"Pay here: {invoice_url}\n"
        f"You'll receive your private invite link after confirmation."
    )

# =========================
# Quotas & stats
# =========================
def _day_key(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%d")

def _week_key(dt: datetime) -> str:
    iso = dt.isocalendar()
    return f"{iso.year}-W{iso.week:02d}"

async def quota_inc_today(chat_id: str):
    today = _day_key(datetime.now(timezone.utc))
    async with _quota_lock:
        _quota.setdefault(today, {})
        _quota[today][chat_id] = _quota[today].get(chat_id, 0) + 1
        await save_quota()

async def quota_get_today(chat_id: str) -> int:
    today = _day_key(datetime.now(timezone.utc))
    async with _quota_lock:
        return _quota.get(today, {}).get(chat_id, 0) or 0

async def stats_on_open(inst: str):
    now = datetime.now(timezone.utc)
    dk, wk = _day_key(now), _week_key(now)
    async with _stats_lock:
        _stats["daily"].setdefault(dk, {"signals": 0, "W": 0, "L": 0, "D": 0})
        _stats["weekly"].setdefault(wk, {"signals": 0, "W": 0, "L": 0, "D": 0})
        _stats["daily"][dk]["signals"] += 1
        _stats["weekly"][wk]["signals"] += 1
        await save_stats()

async def stats_on_settle(result: str):
    now = datetime.now(timezone.utc)
    dk, wk = _day_key(now), _week_key(now)
    key = result  # W|L|D
    if key not in ("W", "L", "D"):
        return
    async with _stats_lock:
        _stats["daily"].setdefault(dk, {"signals": 0, "W": 0, "L": 0, "D": 0})
        _stats["weekly"].setdefault(wk, {"signals": 0, "W": 0, "L": 0, "D": 0})
        _stats["daily"][dk][key] += 1
        _stats["weekly"][wk][key] += 1
        await save_stats()

async def send_tally(scope: str):
    now = datetime.now(timezone.utc)
    if scope == "daily":
        key = _day_key(now)
        bucket = _stats.get("daily", {}).get(key, {"signals": 0, "W": 0, "L": 0, "D": 0})
        title = f"📊 Daily Tally ({key})"
    else:
        key = _week_key(now)
        bucket = _stats.get("weekly", {}).get(key, {"signals": 0, "W": 0, "L": 0, "D": 0})
        title = f"📈 Weekly Tally ({key})"

    s, w, l, d = bucket.get("signals", 0), bucket.get("W", 0), bucket.get("L", 0), bucket.get("D", 0)
    wr = (w / max(w + l, 1)) * 100.0
    text = (
        f"{title}\n"
        f"Signals: {s}\n"
        f"Results: ✅ {w} | ❌ {l} | ➖ {d}\n"
        f"Win rate: {wr:.2f}%"
    )
    for cid in [TELEGRAM_CHAT_FREE, TELEGRAM_CHAT_BASIC, TELEGRAM_CHAT_PRO, TELEGRAM_CHAT_VIP]:
        if cid:
            await tg_send(cid, text)

# =========================
# Engine: signal, cooldown, settlement
# =========================
def _cooldown_ok(inst: str) -> bool:
    last_iso = _last_signal_time.get(inst)
    if not last_iso:
        return True
    last = datetime.fromisoformat(last_iso)
    return (datetime.now(timezone.utc) - last) >= timedelta(minutes=ALERT_COOLDOWN_MIN)

def _mark_cooldown(inst: str):
    _last_signal_time[inst] = datetime.now(timezone.utc).isoformat()

async def try_send_signal(inst: str):
    if not fx_live_now():
        return
    if not _cooldown_ok(inst):
        return
    try:
        m1 = await fetch_oanda(inst, "M1", 400)
        idx = last_completed_idx(m1)
        sig = compute_blw_m1x5(inst, m1, idx)
        if sig.get("signal") not in ("CALL", "PUT"):
            return

        # targets in order VIP -> PRO -> BASIC -> FREE with per-tier quota
        targets: List[Tuple[str, Optional[int]]] = []
        ordered = [
            (TELEGRAM_CHAT_VIP, None),
            (TELEGRAM_CHAT_PRO, LIMIT_PRO),
            (TELEGRAM_CHAT_BASIC, LIMIT_BASIC),
            (TELEGRAM_CHAT_FREE, LIMIT_FREE),
        ]
        for cid, lim in ordered:
            if not cid:
                continue
            used = await quota_get_today(cid)
            if (lim is None) or (used < lim):
                targets.append((cid, lim))

        if not targets:
            return

        direction_arrow = "🟢CALL" if sig["direction"] == "UP" else "🔴PUT"
        msg = (
            f"⚡ {inst} | {direction_arrow}\n"
            f"Entry: {sig['price']:.5f}\n"
            f"Expiry: {PO_EXPIRY_MIN} min\n"
            f"Why: {sig['reason']}"
        )

        deliver_to: List[str] = []
        for cid, _ in targets:
            await tg_send(cid, msg)
            await quota_inc_today(cid)
            deliver_to.append(cid)

        # store open trade for exact settlement
        settle_at = (datetime.now(timezone.utc) + timedelta(minutes=PO_EXPIRY_MIN)).isoformat()
        trade = {
            "instrument": inst,
            "direction": sig["direction"],
            "entry": sig["price"],
            "opened_at": datetime.now(timezone.utc).isoformat(),
            "settle_at": settle_at,
            "targets": deliver_to,
        }
        async with _trades_lock:
            _open_trades.append(trade)
            await save_open_trades()

        await stats_on_open(inst)
        _mark_cooldown(inst)

    except Exception as e:
        log.error(f"{inst} signal error: {e}")

async def run_engine():
    for inst in OANDA_INSTRUMENTS:
        await try_send_signal(inst)

async def settle_due_trades():
    """
    Settle positions exactly at expiry timestamp using Deriv ticks.
    Fallback to OANDA M1 close if ticks unavailable.
    """
    now = datetime.now(timezone.utc)
    to_remove = []

    async with _trades_lock:
        for i, tr in enumerate(list(_open_trades)):
            settle_at_iso = tr.get("settle_at")
            if not settle_at_iso:
                to_remove.append(i)
                continue
            settle_dt = datetime.fromisoformat(settle_at_iso)
            if now < settle_dt:
                continue

            inst = tr["instrument"]
            entry = float(tr["entry"])
            direction = tr["direction"]
            targets = tr.get("targets", [])

            # exact price via Deriv ticks
            px = await price_at_expiry(inst, settle_dt)

            # fallback to candle at/just before expiry
            if px is None:
                try:
                    df = await fetch_oanda(inst, "M1", 15)
                    if not df.empty:
                        df2 = df[df["time"] <= pd.to_datetime(settle_at_iso)].tail(1)
                        if not df2.empty:
                            px = float(df2.iloc[-1]["c"])
                except Exception as e:
                    log.error(f"Settlement fallback failed {inst}: {e}")

            if px is None:
                # try next cycle if we didn't get data yet
                continue

            # decide W/L/D
            if abs(px - entry) < 1e-12:
                res = "D"
            elif direction == "UP":
                res = "W" if px > entry else "L"
            else:
                res = "W" if px < entry else "L"

            await stats_on_settle(res)
            msg = (
                f"🔔 {inst} result: {res}\n"
                f"Entry: {entry:.5f}\n"
                f"Settle: {px:.5f} @ {settle_dt.strftime('%H:%M:%S')} UTC"
            )
            for cid in targets:
                await tg_send(cid, msg)

            to_remove.append(i)

        # remove settled
        for idx in reversed(to_remove):
            _open_trades.pop(idx)
        if to_remove:
            await save_open_trades()

# =========================
# FastAPI endpoints
# =========================
@app_fastapi.get("/", response_class=PlainTextResponse)
async def root():
    return "Pocket Option BLW M1→5 — FX live only (Deriv exact-settlement)"

@app_fastapi.get("/healthz", response_class=PlainTextResponse)
async def healthz():
    ok = all([
        bool(TELEGRAM_BOT_TOKEN),
        bool(OANDA_API_KEY),
        len(OANDA_INSTRUMENTS) > 0,
    ])
    return "ok" if ok else "not-ok"

@app_fastapi.get("/status")
async def status():
    now = datetime.now(timezone.utc).isoformat()
    return {
        "now": now,
        "fx_live_now": fx_live_now(),
        "pairs": OANDA_INSTRUMENTS,
        "cooldown_min": ALERT_COOLDOWN_MIN,
        "expiry_min": PO_EXPIRY_MIN,
    }

# Telegram webhook (optional if not polling)
@app_fastapi.post(WEBHOOK_PATH)
async def telegram_webhook(request: Request, x_telegram_bot_api_secret_token: str = Header(None)):
    if not telegram_app:
        return PlainTextResponse("no-telegram", status_code=503)
    if WEBHOOK_SECRET and x_telegram_bot_api_secret_token != WEBHOOK_SECRET:
        return PlainTextResponse("unauthorized", status_code=401)
    try:
        data = await request.json()
    except Exception:
        return PlainTextResponse("bad-json", status_code=400)
    update = Update.de_json(data, telegram_app.bot)
    await telegram_app.process_update(update)
    return PlainTextResponse("ok")

# NOWPayments IPN
def _hmac_ok(body_bytes: bytes, received_hmac: str) -> bool:
    digest = hmac.new(NOWPAY_IPN_SECRET.encode("utf-8"), body_bytes, hashlib.sha512).hexdigest()
    return hmac.compare_digest(digest, received_hmac.lower())

@app_fastapi.post("/pay/ipn")
async def pay_ipn(request: Request):
    if not NOWPAY_IPN_SECRET:
        return PlainTextResponse("ipn-not-configured", status_code=503)
    raw = await request.body()
    h = request.headers.get("x-nowpayments-sig", "")
    if not _hmac_ok(raw, h or ""):
        return PlainTextResponse("bad-hmac", status_code=401)
    try:
        payload = json.loads(raw.decode("utf-8"))
    except Exception:
        return PlainTextResponse("bad-json", status_code=400)

    order_id = payload.get("order_id", "")
    payment_status = payload.get("payment_status", "").lower()
    if not order_id:
        return PlainTextResponse("no-order", status_code=400)

    async with _orders_lock:
        order = _orders.get(order_id)

    if not order:
        return PlainTextResponse("unknown-order", status_code=404)

    tier = order.get("tier")
    tg_user_id = order.get("telegram_id")

    # On confirmed/finished, send invite link (or DM)
    if payment_status in ("finished", "confirmed", "completed", "paid"):
        link = _plan_link(tier)
        try:
            if telegram_app and tg_user_id:
                await telegram_app.bot.send_message(
                    chat_id=tg_user_id,
                    text=f"✅ Payment confirmed for {tier}!\nJoin link: {link or 'Ask admin to add you.'}"
                )
        except Exception as e:
            log.error(f"Invite send failed: {e}")
        return PlainTextResponse("ok")
    return PlainTextResponse("ignored")

# =========================
# Startup / Shutdown
# =========================
@app_fastapi.on_event("startup")
async def on_startup():
    global telegram_app, scheduler
    await load_state()

    # Telegram
    telegram_app = build_telegram_app()
    await telegram_app.initialize()
    await telegram_app.start()
    if ENABLE_BOT_POLLING == "1":
        await telegram_app.bot.delete_webhook(drop_pending_updates=True)
        log.info("🤖 Telegram bot: polling started")
    else:
        await telegram_app.bot.set_webhook(url=f"{BASE_URL}{WEBHOOK_PATH}", secret_token=WEBHOOK_SECRET, drop_pending_updates=True)
        log.info(f"🤖 Telegram bot: webhook set {BASE_URL}{WEBHOOK_PATH}")

    # Scheduler
    scheduler = AsyncIOScheduler(timezone=timezone.utc)
    # Engine: every minute at second 5
    scheduler.add_job(run_engine, CronTrigger(second="5"))
    # Settlement: every 30s
    scheduler.add_job(settle_due_trades, CronTrigger(second="*/30"))
    # Daily tally: 20:05 UTC ~ 16:05 Port_of_Spain (no DST)
    scheduler.add_job(lambda: asyncio.create_task(send_tally("daily")), CronTrigger(hour="20", minute="5"))
    # Weekly tally: Sun 20:10 UTC
    scheduler.add_job(lambda: asyncio.create_task(send_tally("weekly")), CronTrigger(day_of_week="sun", hour="20", minute="10"))
    scheduler.start()
    log.info("⏰ Scheduler started")

@app_fastapi.on_event("shutdown")
async def on_shutdown():
    if scheduler:
        scheduler.shutdown(wait=False)
    if telegram_app:
        await telegram_app.stop()

# Uvicorn entry
app = app_fastapi
