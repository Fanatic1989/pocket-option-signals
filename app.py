# app.py
# -----------------------------------------------------------------------------
# Pocket Option Signals — BLW M1→5 (Deriv feed, FX live only)
# - Deriv candles (matches binary charts much closer than OANDA)
# - EXACT features kept:
#   • Tiers: FREE/BASIC/PRO/VIP with per-chat quotas (VIP ∞)
#   • 1m candles, 5m expiry, 10m per-instrument cooldown
#   • FX live session guard (Mon–Fri, local 08:00–16:00)
#   • W/L/D settlement and auto daily/weekly tallies to all groups
#   • NOWPayments paywall: /plans, /upgrade, /pay/ipn (HMAC verify)
#   • Webhook or polling for Telegram
# - Environment variables listed near the top.
# -----------------------------------------------------------------------------

from __future__ import annotations

import os
import json
import hmac
import hashlib
import asyncio
import logging
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone, time as dtime
from typing import Dict, Any, List, Optional, Tuple, Set

import pandas as pd
import pandas_ta as ta
import httpx

from fastapi import FastAPI, Request, Header
from fastapi.responses import PlainTextResponse, JSONResponse

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes

# ---- Deriv fetcher -----------------------------------------------------------
from deriv_fetcher import deriv_m1, last_completed_idx

# =========================
# ENV
# =========================
# Telegram
TELEGRAM_BOT_TOKEN   = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
BASE_URL             = os.getenv("BASE_URL", "").rstrip("/")
WEBHOOK_PATH         = os.getenv("WEBHOOK_PATH", "/telegram/webhook").strip()
WEBHOOK_SECRET       = os.getenv("WEBHOOK_SECRET", "change-me").strip()
ENABLE_BOT_POLLING   = os.getenv("ENABLE_BOT_POLLING", "1").strip()  # default polling

# Chats (channels/supergroups)
TELEGRAM_CHAT_FREE   = os.getenv("TELEGRAM_CHAT_FREE", "").strip()
TELEGRAM_CHAT_BASIC  = os.getenv("TELEGRAM_CHAT_BASIC", "").strip()
TELEGRAM_CHAT_PRO    = os.getenv("TELEGRAM_CHAT_PRO", "").strip()
TELEGRAM_CHAT_VIP    = os.getenv("TELEGRAM_CHAT_VIP", "").strip()

# Per-chat daily quotas
LIMIT_FREE           = int(os.getenv("LIMIT_FREE", "3"))
LIMIT_BASIC          = int(os.getenv("LIMIT_BASIC", "6"))
LIMIT_PRO            = int(os.getenv("LIMIT_PRO", "15"))
# VIP unlimited

# Deriv feed
DERIV_ENDPOINT       = os.getenv("DERIV_ENDPOINT", "").strip()
# You can set DERIV_SYMBOLS directly, e.g. "frxEURUSD,frxGBPUSD,..."
DERIV_SYMBOLS_ENV    = os.getenv("DERIV_SYMBOLS", "").strip()
# Back-compat: convert OANDA_INSTRUMENTS like "EUR_USD" -> "frxEURUSD" if DERIV_SYMBOLS unset
OANDA_INSTRUMENTS    = os.getenv("OANDA_INSTRUMENTS", "").strip()

# Strategy (BLW M1→5 tuned)
PO_EXPIRY_MIN        = int(os.getenv("PO_EXPIRY_MIN", "5"))
ALERT_COOLDOWN_MIN   = int(os.getenv("ALERT_COOLDOWN_MIN", "10"))
ADX_MIN              = float(os.getenv("ADX_MIN", "18"))
BODY_ATR_MIN         = float(os.getenv("BODY_ATR_MIN", "0.30"))
ATR_PCTL_MIN         = float(os.getenv("ATR_PCTL_MIN", "0.30"))
ATR_PCTL_MAX         = float(os.getenv("ATR_PCTL_MAX", "0.92"))
RSI_UP               = float(os.getenv("RSI_UP", "52"))
RSI_DN               = float(os.getenv("RSI_DN", "48"))

# Session / OTC guard (FX live only)
SESSION_TZ_NAME      = os.getenv("SESSION_TZ", "America/Port_of_Spain")
LIVE_START_LOCAL     = os.getenv("LIVE_START_LOCAL", "08:00")
LIVE_END_LOCAL       = os.getenv("LIVE_END_LOCAL", "16:00")

# Payments (NOWPayments)
NOWPAY_API_KEY       = os.getenv("NOWPAY_API_KEY", "").strip()
NOWPAY_IPN_SECRET    = os.getenv("NOWPAY_IPN_SECRET", "").strip()
NOWPAY_PRICE_CCY     = os.getenv("NOWPAY_PRICE_CCY", "usd").strip().lower()
NOWPAY_PAY_CCY       = os.getenv("NOWPAY_PAY_CCY", "usdttrc20").strip().lower()

PRICE_BASIC          = float(os.getenv("PRICE_BASIC", "29.0"))
PRICE_PRO            = float(os.getenv("PRICE_PRO", "49.0"))
PRICE_VIP            = float(os.getenv("PRICE_VIP", "79.0"))

# Invite links to DM after payment
TELEGRAM_LINK_BASIC  = os.getenv("TELEGRAM_LINK_BASIC", "").strip()
TELEGRAM_LINK_PRO    = os.getenv("TELEGRAM_LINK_PRO", "").strip()
TELEGRAM_LINK_VIP    = os.getenv("TELEGRAM_LINK_VIP", "").strip()

# Storage
DATA_DIR             = os.getenv("DATA_DIR", "./data").strip()
STATS_FILE           = os.path.join(DATA_DIR, "stats.json")
OPEN_TRADES_FILE     = os.path.join(DATA_DIR, "open_trades.json")
QUOTA_FILE           = os.path.join(DATA_DIR, "quota.json")
LAST_SIGNAL_FILE     = os.path.join(DATA_DIR, "last_signal.json")
UNSUPPORTED_FILE     = os.path.join(DATA_DIR, "unsupported.json")
ORDERS_FILE          = os.path.join(DATA_DIR, "orders.json")
SUBS_FILE            = os.path.join(DATA_DIR, "subs.json")  # subscriptions (telegram_id -> tier/expiry)

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
_subs_lock = asyncio.Lock()

_stats: Dict[str, Any] = {"daily": {}, "weekly": {}}
_open_trades: List[Dict[str, Any]] = []
_quota: Dict[str, Dict[str, int]] = {}
_last_signal_time: Dict[str, str] = {}
_orders: Dict[str, Dict[str, Any]] = {}
_subs: Dict[str, Dict[str, Any]] = {}

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
    global _stats, _open_trades, _quota, _last_signal_time, _orders, _subs
    _stats = await _read_json(STATS_FILE, {"daily": {}, "weekly": {}})
    _open_trades = await _read_json(OPEN_TRADES_FILE, [])
    _quota = await _read_json(QUOTA_FILE, {})
    _last_signal_time = await _read_json(LAST_SIGNAL_FILE, {})
    _orders = await _read_json(ORDERS_FILE, {})
    _subs = await _read_json(SUBS_FILE, {})

async def save_stats():        await _write_json(STATS_FILE, _stats)
async def save_open_trades():  await _write_json(OPEN_TRADES_FILE, _open_trades)
async def save_quota():        await _write_json(QUOTA_FILE, _quota)
async def save_last_signal():  await _write_json(LAST_SIGNAL_FILE, _last_signal_time)
async def save_orders():       await _write_json(ORDERS_FILE, _orders)
async def save_subs():         await _write_json(SUBS_FILE, _subs)

# =========================
# Symbols (Deriv)
# =========================
def _sanitize_list_csv(s: str) -> List[str]:
    if not s:
        return []
    out = []
    for part in s.split(","):
        part = part.strip().strip('"').strip("'")
        if part:
            out.append(part)
    return out

def _oanda_to_deriv(sym: str) -> Optional[str]:
    # "EUR_USD" -> "frxEURUSD"
    if "_" not in sym:
        return None
    a, b = sym.split("_", 1)
    if len(a) != 3 or len(b) != 3:
        return None
    return f"frx{a}{b}"

def load_deriv_symbols() -> List[str]:
    if DERIV_SYMBOLS_ENV:
        return _sanitize_list_csv(DERIV_SYMBOLS_ENV)
    oanda_list = _sanitize_list_csv(OANDA_INSTRUMENTS)
    derived: List[str] = []
    for s in oanda_list:
        m = _oanda_to_deriv(s)
        if m:
            derived.append(m)
    if not derived:
        # default: your winners list mapped to Deriv
        defaults = "AUD_USD,EUR_GBP,EUR_JPY,AUD_CHF,CAD_JPY,GBP_AUD,EUR_CAD,EUR_CHF,AUD_CAD,CAD_CHF,CHF_JPY,USD_CHF,GBP_CHF"
        for s in _sanitize_list_csv(defaults):
            m = _oanda_to_deriv(s)
            if m:
                derived.append(m)
    # de-dupe
    seen = set()
    clean = []
    for s in derived:
        if s not in seen:
            seen.add(s)
            clean.append(s)
    return clean

DERIV_SYMBOLS = load_deriv_symbols()

# =========================
# Time / Session guard
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
    """FX live window ONLY: Mon–Fri, 08:00–16:00 local time."""
    now = (now_utc or datetime.now(timezone.utc)).astimezone(LOCAL_TZ)
    if now.weekday() > 4:
        return False
    t = now.time()
    return LIVE_START <= t <= LIVE_END

# =========================
# Indicators / Strategy
# =========================
def atr_percentile(series: pd.Series, lookback: int = 300) -> float:
    s = series.dropna()
    if len(s) == 0:
        return 0.0
    if len(s) > lookback:
        s = s.iloc[-lookback:]
    cur = s.iloc[-1]
    rank = (s <= cur).sum() / len(s)
    return float(rank)

def compute_blw_m1x5(inst: str, m1: pd.DataFrame, idx: int) -> Dict[str, Any]:
    """
    BLW-like M1 trigger with 5-min expiry:
      - EMA 9/21 cross + RSI(14) side + body >= BODY_ATR_MIN * ATR(14)
      - ADX(14) >= ADX_MIN
      - ATR percentile between [ATR_PCTL_MIN, ATR_PCTL_MAX]
      - last candle must be complete
    Returns {signal: CALL|PUT|HOLD, direction, reason, time, price}
    """
    if idx is None or idx < 1 or m1.empty:
        return {"signal": "HOLD", "reason": "insufficient data"}

    df = m1.copy()
    close = df["c"].astype(float)

    df["ema9"]  = ta.ema(close, length=9)
    df["ema21"] = ta.ema(close, length=21)
    df["rsi"]   = ta.rsi(close, length=14)
    df["atr"]   = ta.atr(df["h"], df["l"], close, length=14)
    adx_df      = ta.adx(df["h"], df["l"], close, length=14)
    df["adx"]   = adx_df["ADX_14"] if adx_df is not None and "ADX_14" in adx_df else pd.NA

    if idx >= len(df):
        idx = len(df) - 1
    cur, prev = df.iloc[idx], df.iloc[idx - 1]

    # must be completed candle
    if not bool(cur.get("complete", False)):
        return {"signal": "HOLD", "reason": "last candle not complete"}

    # body vs ATR
    body = abs(float(cur["c"]) - float(cur["o"]))
    if float(cur["atr"]) <= 0 or body < BODY_ATR_MIN * float(cur["atr"]):
        return {"signal": "HOLD", "reason": "small body vs ATR"}

    # volatility percentile
    p_atr = atr_percentile(df["atr"])
    if not (ATR_PCTL_MIN <= p_atr <= ATR_PCTL_MAX):
        return {"signal": "HOLD", "reason": f"ATR pct {p_atr:.2f} out of window"}

    # trend strength
    if pd.isna(cur["adx"]) or float(cur["adx"]) < ADX_MIN:
        return {"signal": "HOLD", "reason": f"ADX {float(cur['adx']) if not pd.isna(cur['adx']) else 'nan'} < {ADX_MIN}"}

    # Cross + RSI side
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
# Telegram bot
# =========================
HELP_TEXT = (
    "Pocket Option BLW (M1→5, Deriv) 🤖\n"
    "• FX live only (Mon–Fri 08:00–16:00 America/Port_of_Spain)\n"
    "• Tiers: FREE(3/d), BASIC(6/d), PRO(15/d), VIP(∞)\n"
    "• 1m candles, 5m expiry, 10m per-pair cooldown\n"
    "• Daily & weekly tallies auto-posted\n\n"
    "Commands:\n"
    "/start — intro\n"
    "/ping  — health\n"
    "/plans — pricing\n"
    "/upgrade <BASIC|PRO|VIP> — crypto invoice\n"
    "/limits — today’s remaining quota (per chat)\n"
)

def build_telegram_app() -> Application:
    app = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", start_cmd))
    app.add_handler(CommandHandler("ping", ping_cmd))
    app.add_handler(CommandHandler("plans", plans_cmd))
    app.add_handler(CommandHandler("upgrade", upgrade_cmd))
    app.add_handler(CommandHandler("limits", limits_cmd))
    return app

async def start_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(HELP_TEXT)

async def ping_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("pong")

def _plan_price(tier: str) -> Optional[float]:
    return {"BASIC": PRICE_BASIC, "PRO": PRICE_PRO, "VIP": PRICE_VIP}.get(tier.upper())

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
        _orders[order_id] = {
            "tier": tier, "telegram_id": user_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "invoice": invoice
        }
        await save_orders()

    await update.message.reply_text(
        f"✅ Invoice for {tier}: {price:.2f} {NOWPAY_PRICE_CCY.upper()}\n"
        f"Pay here: {invoice_url}\n"
        f"You'll receive your private invite link after confirmation."
    )

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

# =========================
# Quotas & stats (daily / weekly)
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
        return _quota.get(today, {}).get(chat_id, 0)

async def stats_on_open():
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
    key = result  # "W" | "L" | "D"
    if key not in ("W", "L", "D"):
        return
    async with _stats_lock:
        _stats["daily"].setdefault(dk, {"signals": 0, "W": 0, "L": 0, "D": 0})
        _stats["weekly"].setdefault(wk, {"signals": 0, "W": 0, "L": 0, "D": 0})
        _stats["daily"][dk][key] += 1
        _stats["weekly"][wk][key] += 1
        await save_stats()

async def send_tally(scope: str):
    """scope: 'daily' or 'weekly' — send to all configured groups."""
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
    text = (f"{title}\n"
            f"Signals: {s}\n"
            f"Results: ✅ {w} | ❌ {l} | ➖ {d}\n"
            f"Win rate: {wr:.2f}%")
    for cid in [TELEGRAM_CHAT_FREE, TELEGRAM_CHAT_BASIC, TELEGRAM_CHAT_PRO, TELEGRAM_CHAT_VIP]:
        if cid:
            await tg_send(cid, text)

# =========================
# Tier resolution
# =========================
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

# =========================
# NOWPayments helpers
# =========================
NOWPAY_BASE = "https://api.nowpayments.io/v1"

async def nowpay_create_invoice(
    price_amount: float,
    price_currency: str,
    order_id: str,
    order_description: str,
    pay_currency: Optional[str] = None,
    success_url: Optional[str] = None,
    cancel_url: Optional[str] = None,
    ipn_callback_url: Optional[str] = None
) -> Dict[str, Any]:
    headers = {"x-api-key": NOWPAY_API_KEY, "Content-Type": "application/json"}
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

    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.post(f"{NOWPAY_BASE}/invoice", headers=headers, json=payload)
        r.raise_for_status()
        return r.json()

def verify_nowpay_hmac(raw_body: bytes, x_sig: str) -> bool:
    if not NOWPAY_IPN_SECRET or not x_sig:
        return False
    hm = hmac.new(NOWPAY_IPN_SECRET.encode("utf-8"), raw_body, hashlib.sha512).hexdigest()
    return hmac.compare_digest(hm, x_sig)

# =========================
# Engine: cooldown, signal, settlement
# =========================
def _cooldown_ok(inst: str) -> bool:
    last_iso = _last_signal_time.get(inst)
    if not last_iso:
        return True
    last = datetime.fromisoformat(last_iso)
    return (datetime.now(timezone.utc) - last) >= timedelta(minutes=ALERT_COOLDOWN_MIN)

def _mark_cooldown(inst: str):
    _last_signal_time[inst] = datetime.now(timezone.utc).isoformat()

async def settle_due_trades():
    now = datetime.now(timezone.utc)
    remove_idx: List[int] = []
    async with _trades_lock:
        for i, tr in enumerate(list(_open_trades)):
            if now >= datetime.fromisoformat(tr["settle_at"]):
                inst = tr["instrument"]
                try:
                    # get last closed candle at or before settle_at
                    df = await deriv_m1(inst, 20)
                    if df.empty:
                        continue
                    df2 = df[df["time"] <= pd.to_datetime(tr["settle_at"])].tail(1)
                    if df2.empty:
                        df2 = df.tail(1)
                    close = float(df2.iloc[-1]["c"])
                    entry = float(tr["entry"])
                    direction = tr["direction"]
                    if abs(close - entry) < 1e-12:
                        res = "D"
                    elif direction == "UP":
                        res = "W" if close > entry else "L"
                    else:
                        res = "W" if close < entry else "L"
                    await stats_on_settle(res)
                    msg = f"🔔 {inst} result: {res} (entry {entry:.5f} → close {close:.5f})"
                    for cid in tr.get("targets", []):
                        await tg_send(cid, msg)
                    remove_idx.append(i)
                except Exception as e:
                    log.error(f"Settlement error {inst}: {e}")
        for j in reversed(remove_idx):
            _open_trades.pop(j)
        if remove_idx:
            await save_open_trades()

async def try_send_signal(inst: str):
    # live-session guard
    if not fx_live_now():
        return
    # cooldown guard
    if not _cooldown_ok(inst):
        return

    try:
        m1 = await deriv_m1(inst, 400)
        idx = last_completed_idx(m1)
        sig = compute_blw_m1x5(inst, m1, idx)
        if sig.get("signal") not in ("CALL", "PUT"):
            return

        # targets by tier respecting quotas
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

        # open trade for settlement
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

        await stats_on_open()
        _mark_cooldown(inst)

    except Exception as e:
        log.error(f'{inst} signal error: {e}')

async def run_engine():
    # iterate sequentially to keep resource usage low
    for inst in DERIV_SYMBOLS:
        await try_send_signal(inst)

# =========================
# Tally jobs
# =========================
async def daily_tally_job():
    await send_tally("daily")

async def weekly_tally_job():
    await send_tally("weekly")

# =========================
# FastAPI endpoints
# =========================
@app_fastapi.get("/", response_class=PlainTextResponse)
async def root():
    return "Pocket Option BLW M1→5 — FX live only (Deriv)"

@app_fastapi.get("/healthz", response_class=PlainTextResponse)
async def healthz():
    ok = all([
        bool(TELEGRAM_BOT_TOKEN),
        bool(DERIV_ENDPOINT),
        len(DERIV_SYMBOLS) > 0,
    ])
    return "ok" if ok else "not-ok"

@app_fastapi.get("/status")
async def status():
    now = datetime.now(timezone.utc).isoformat()
    return {
        "now": now,
        "fx_live_now": fx_live_now(),
        "pairs": DERIV_SYMBOLS,
        "cooldown_min": ALERT_COOLDOWN_MIN,
        "expiry_min": PO_EXPIRY_MIN,
    }

# Telegram webhook (optional; polling is default if ENABLE_BOT_POLLING=1)
@app_fastapi.post(WEBHOOK_PATH)
async def telegram_webhook(request: Request):
    if not telegram_app:
        return PlainTextResponse("no-telegram", status_code=503)
    try:
        data = await request.json()
    except Exception:
        return PlainTextResponse("bad-json", status_code=400)
    update = Update.de_json(data, telegram_app.bot)
    await telegram_app.process_update(update)
    return PlainTextResponse("ok")

# Payments endpoints
@app_fastapi.post("/pay/ipn")
async def pay_ipn(request: Request, x_nowpayments_sig: str = Header(default="")):
    raw = await request.body()
    if not verify_nowpay_hmac(raw, x_nowpayments_sig):
        return PlainTextResponse("invalid-signature", status_code=400)
    try:
        payload = await request.json()
    except Exception:
        return PlainTextResponse("bad-json", status_code=400)

    # We rely on order_id pattern created in /upgrade: tier:...:uid:...
    order_id = str(payload.get("order_id", ""))
    payment_status = (payload.get("payment_status") or "").lower()
    log.info(f"IPN for order_id={order_id} status={payment_status}")

    if not order_id or "tier:" not in order_id or ":uid:" not in order_id:
        return PlainTextResponse("ignored", status_code=200)

    # Confirmed / finished → grant access
    if payment_status not in ("finished", "confirmed"):
        return PlainTextResponse("pending", status_code=200)

    parts = order_id.split(":")
    tier = None
    user_id = None
    for i, p in enumerate(parts):
        if p == "tier" and i + 1 < len(parts):
            tier = parts[i + 1]
        if p == "uid" and i + 1 < len(parts):
            try:
                user_id = int(parts[i + 1])
            except Exception:
                pass

    if not tier or not user_id:
        return PlainTextResponse("bad-order", status_code=200)

    # Save subscription (30 days from now)
    expires = (datetime.now(timezone.utc) + timedelta(days=30)).isoformat()
    async with _subs_lock:
        _subs[str(user_id)] = {"tier": tier.upper(), "expires_at": expires}
        await save_subs()

    # Send invite link
    link = {"BASIC": TELEGRAM_LINK_BASIC, "PRO": TELEGRAM_LINK_PRO, "VIP": TELEGRAM_LINK_VIP}.get(tier.upper(), "")
    try:
        if telegram_app and link:
            await telegram_app.bot.send_message(chat_id=user_id,
                text=f"✅ Payment received for {tier.upper()}.\nJoin here: {link}")
    except Exception as e:
        log.error(f"DM invite failed user {user_id}: {e}")

    return PlainTextResponse("ok", status_code=200)

@app_fastapi.get("/pay/success")
async def pay_success():
    return PlainTextResponse("Payment success. Check your Telegram DM for the invite link.")

@app_fastapi.get("/pay/cancel")
async def pay_cancel():
    return PlainTextResponse("Payment canceled.")

# =========================
# Lifespan: start scheduler & telegram
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
        await telegram_app.bot.set_webhook(url=f"{BASE_URL}{WEBHOOK_PATH}",
                                           secret_token=WEBHOOK_SECRET,
                                           drop_pending_updates=True)
        log.info(f"🤖 Telegram bot: webhook set {BASE_URL}{WEBHOOK_PATH}")

    # Scheduler
    scheduler = AsyncIOScheduler(timezone=timezone.utc)
    # Engine every minute at second 5
    scheduler.add_job(run_engine, CronTrigger(second="5"))
    # Settlement every 30s
    scheduler.add_job(settle_due_trades, CronTrigger(second="*/30"))
    # Daily tally: 20:05 UTC ≈ 16:05 America/Port_of_Spain (UTC-4, no DST)
    scheduler.add_job(daily_tally_job, CronTrigger(hour="20", minute="5"))
    # Weekly tally: Sunday 20:10 UTC
    scheduler.add_job(weekly_tally_job, CronTrigger(day_of_week="sun", hour="20", minute="10"))
    scheduler.start()
    log.info("⏰ Scheduler started")

@app_fastapi.on_event("shutdown")
async def on_shutdown():
    if scheduler:
        scheduler.shutdown(wait=False)
    if telegram_app:
        await telegram_app.stop()

# =========================
# Uvicorn entry
# =========================
app = app_fastapi
