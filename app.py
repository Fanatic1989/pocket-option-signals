# app.py
# Dual Deriv Signals — BLW Trend + Mean-Reversion Chop (FX live only)
# - Deriv websocket data (resilient endpoints)
# - 1m candles, 5m expiry, wall-clock settlement, 10m per-pair cooldown
# - Tiers & quotas (FREE/BASIC/PRO/VIP)
# - Auto daily (16:00 local) & weekly tallies to all groups
# - Simple NOWPayments webhook to auto add/remove subscribers
# ------------------------------------------------------------------------------

import os
import json
import math
import asyncio
import logging
from typing import Any, Dict, List, Optional, Tuple, Set
from datetime import datetime, timedelta, timezone, time as dtime

import pandas as pd
import pandas_ta as ta
import websockets
import httpx

from fastapi import FastAPI, Request, Header
from fastapi.responses import PlainTextResponse, JSONResponse

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes

# =========================
# Logging
# =========================
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("dual-deriv-bot")

# =========================
# ENV
# =========================
TELEGRAM_BOT_TOKEN   = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
BASE_URL             = os.getenv("BASE_URL", "").rstrip("/")
WEBHOOK_PATH         = os.getenv("WEBHOOK_PATH", "/telegram/webhook").strip()
WEBHOOK_SECRET       = os.getenv("WEBHOOK_SECRET", "change-me").strip()
ENABLE_BOT_POLLING   = os.getenv("ENABLE_BOT_POLLING", "0").strip()

# Chats (set your real IDs; can be channel IDs or group IDs)
TELEGRAM_CHAT_FREE   = os.getenv("TELEGRAM_CHAT_FREE", "").strip()
TELEGRAM_CHAT_BASIC  = os.getenv("TELEGRAM_CHAT_BASIC", "").strip()
TELEGRAM_CHAT_PRO    = os.getenv("TELEGRAM_CHAT_PRO", "").strip()
TELEGRAM_CHAT_VIP    = os.getenv("TELEGRAM_CHAT_VIP", "").strip()

# Quotas
LIMIT_FREE           = int(os.getenv("LIMIT_FREE", "3"))
LIMIT_BASIC          = int(os.getenv("LIMIT_BASIC", "6"))
LIMIT_PRO            = int(os.getenv("LIMIT_PRO", "15"))
# VIP unlimited

# Deriv endpoints & instruments
DERIV_APP_ID         = os.getenv("DERIV_APP_ID", "99185").strip()
DERIV_ENDPOINT       = os.getenv("DERIV_ENDPOINT", "").strip()
DERIV_ENDPOINTS      = [
    DERIV_ENDPOINT or f"wss://ws.derivws.com/websockets/v3?app_id={DERIV_APP_ID}",
    f"wss://ws.binaryws.com/websockets/v3?app_id={DERIV_APP_ID}",
]

# EXACT PAIRS (Deriv symbols with 'frx' prefix). Map to pretty labels for messages.
# Use the same winners set you locked previously (converted to Deriv codes).
DERIV_SYMBOLS = [s.strip() for s in os.getenv(
    "DERIV_INSTRUMENTS",
    "frxAUDUSD,frxEURGBP,frxEURJPY,frxAUDCHF,frxCADJPY,frxGBPAUD,frxEURCAD,frxEURCHF,"
    "frxAUDCAD,frxCADCHF,frxCHFJPY,frxUSDCHF,frxGBPCHF"
).split(",") if s.strip()]

SYMBOL_PRETTY = {
    "frxAUDUSD": "AUD/USD",
    "frxEURGBP": "EUR/GBP",
    "frxEURJPY": "EUR/JPY",
    "frxAUDCHF": "AUD/CHF",
    "frxCADJPY": "CAD/JPY",
    "frxGBPAUD": "GBP/AUD",
    "frxEURCAD": "EUR/CAD",
    "frxEURCHF": "EUR/CHF",
    "frxAUDCAD": "AUD/CAD",
    "frxCADCHF": "CAD/CHF",
    "frxCHFJPY": "CHF/JPY",
    "frxUSDCHF": "USD/CHF",
    "frxGBPCHF": "GBP/CHF",
}

# Session / FX live window
SESSION_TZ_NAME      = os.getenv("SESSION_TZ", "America/Port_of_Spain")
LIVE_START_LOCAL     = os.getenv("LIVE_START_LOCAL", "08:00")
LIVE_END_LOCAL       = os.getenv("LIVE_END_LOCAL", "16:00")

# Strategy knobs
PO_EXPIRY_MIN        = int(os.getenv("PO_EXPIRY_MIN", "5"))
ALERT_COOLDOWN_MIN   = int(os.getenv("ALERT_COOLDOWN_MIN", "10"))

# BLW trend filters
ADX_MIN_TREND        = float(os.getenv("ADX_MIN_TREND", "20"))
M15_EMA_TREND_LEN    = int(os.getenv("M15_EMA_TREND_LEN", "200"))
M15_EMA_SLOPE_MIN_PIPS = float(os.getenv("M15_EMA_SLOPE_MIN_PIPS", "0.2"))  # per 15m bar
RSI_UP               = float(os.getenv("RSI_UP", "52"))
RSI_DN               = float(os.getenv("RSI_DN", "48"))
BODY_ATR_MIN         = float(os.getenv("BODY_ATR_MIN", "0.30"))
ATR_PCTL_MIN         = float(os.getenv("ATR_PCTL_MIN", "0.30"))
ATR_PCTL_MAX         = float(os.getenv("ATR_PCTL_MAX", "0.92"))

# Chop (mean reversion) filters
ADX_MAX_CHOP         = float(os.getenv("ADX_MAX_CHOP", "18"))
BB_LENGTH            = int(os.getenv("BB_LENGTH", "20"))
BB_STD               = float(os.getenv("BB_STD", "2.0"))

# Paywall / NOWPayments (optional)
PAYWALL_ENABLED      = os.getenv("PAYWALL_ENABLED", "0").strip() == "1"
NOWPAYMENTS_IPN_SECRET = os.getenv("NOWPAYMENTS_IPN_SECRET", "").strip()

# Storage paths
DATA_DIR             = os.getenv("DATA_DIR", "./data").strip()
STATS_FILE           = os.path.join(DATA_DIR, "stats.json")
OPEN_TRADES_FILE     = os.path.join(DATA_DIR, "open_trades.json")
QUOTA_FILE           = os.path.join(DATA_DIR, "quota.json")
LAST_SIGNAL_FILE     = os.path.join(DATA_DIR, "last_signal.json")
SUBS_FILE            = os.path.join(DATA_DIR, "subs.json")  # paywall state

# =========================
# Globals
# =========================
app_fastapi: FastAPI = FastAPI()
telegram_app: Optional[Application] = None
scheduler: Optional[AsyncIOScheduler] = None

_stats_lock = asyncio.Lock()
_trades_lock = asyncio.Lock()
_quota_lock = asyncio.Lock()
_io_lock = asyncio.Lock()

_stats: Dict[str, Any] = {"daily": {}, "weekly": {}}
_open_trades: List[Dict[str, Any]] = []
_quota: Dict[str, Dict[str, int]] = {}
_last_signal_time: Dict[str, str] = {}
_subs: Dict[str, Any] = {"active": {}}  # { user_id/email : {"tier":"BASIC"/"PRO"/"VIP","until":"iso"} }

# =========================
# Time / TZ helpers
# =========================
try:
    from zoneinfo import ZoneInfo
except Exception:
    from backports.zoneinfo import ZoneInfo

def _parse_hhmm(s: str) -> dtime:
    hh, mm = s.split(":")
    return dtime(int(hh), int(mm))

LOCAL_TZ = ZoneInfo(SESSION_TZ_NAME)
LIVE_START = _parse_hhmm(LIVE_START_LOCAL)
LIVE_END   = _parse_hhmm(LIVE_END_LOCAL)

def fx_live_now(now_utc: Optional[datetime] = None) -> bool:
    now = (now_utc or datetime.now(timezone.utc)).astimezone(LOCAL_TZ)
    if now.weekday() > 4:
        return False
    t = now.time()
    return LIVE_START <= t <= LIVE_END

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

async def load_state():
    global _stats, _open_trades, _quota, _last_signal_time, _subs
    _stats = await _read_json(STATS_FILE, {"daily": {}, "weekly": {}})
    _open_trades = await _read_json(OPEN_TRADES_FILE, [])
    _quota = await _read_json(QUOTA_FILE, {})
    _last_signal_time = await _read_json(LAST_SIGNAL_FILE, {})
    _subs = await _read_json(SUBS_FILE, {"active": {}})

# =========================
# Telegram
# =========================
HELP_TEXT = (
    "Dual Deriv Signals 🤖\n"
    "• FX live only (Mon–Fri 08:00–16:00 America/Port_of_Spain)\n"
    "• Tiers: FREE(3/d) BASIC(6/d) PRO(15/d) VIP(∞)\n"
    "• 1m candles • 5m expiry • 10m cooldown per pair\n"
    "• Auto daily & weekly tallies\n\n"
    "Commands:\n"
    "/start — intro\n"
    "/ping  — health\n"
    "/limits — today’s remaining quota\n"
)

def build_telegram_app() -> Application:
    app = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", start_cmd))
    app.add_handler(CommandHandler("ping", ping_cmd))
    app.add_handler(CommandHandler("limits", limits_cmd))
    return app

async def start_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(HELP_TEXT)

async def ping_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("pong")

async def limits_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = str(update.effective_chat.id) if update.effective_chat else None
    if chat_id is None:
        return
    tier, lim = resolve_tier_and_limit(chat_id)
    used = await quota_get_today(chat_id)
    rem = "∞" if lim is None else max(lim - used, 0)
    await update.message.reply_text(f"Tier: {tier}\nUsed today: {used}\nRemaining: {rem}")

async def tg_send(chat_id: str, text: str):
    if not telegram_app:
        return
    try:
        await telegram_app.bot.send_message(chat_id=chat_id, text=text, disable_web_page_preview=True)
    except Exception as e:
        log.error(f"Telegram send failed to {chat_id}: {e}")

def resolve_tier_and_limit(chat_id: str) -> Tuple[str, Optional[int]]:
    # Static groups -> fixed tiers
    if TELEGRAM_CHAT_VIP and chat_id == TELEGRAM_CHAT_VIP:
        return ("VIP", None)
    if TELEGRAM_CHAT_PRO and chat_id == TELEGRAM_CHAT_PRO:
        return ("PRO", LIMIT_PRO)
    if TELEGRAM_CHAT_BASIC and chat_id == TELEGRAM_CHAT_BASIC:
        return ("BASIC", LIMIT_BASIC)
    if TELEGRAM_CHAT_FREE and chat_id == TELEGRAM_CHAT_FREE:
        return ("FREE", LIMIT_FREE)
    # default FREE
    return ("FREE", LIMIT_FREE)

# =========================
# Deriv data
# =========================
async def deriv_candles(symbol: str, granularity_sec: int, count: int) -> pd.DataFrame:
    req = {
        "ticks_history": symbol,
        "granularity": granularity_sec,
        "count": count,
        "end": "latest",
        "style": "candles",
    }
    last_err: Optional[Exception] = None
    for ep in DERIV_ENDPOINTS:
        try:
            async with websockets.connect(ep, ping_interval=20, ping_timeout=20) as ws:
                await ws.send(json.dumps(req))
                raw = await ws.recv()
            resp = json.loads(raw)
            candles = resp.get("candles") or []
            rows = []
            for c in candles:
                ts = pd.to_datetime(int(c["epoch"]), unit="s", utc=True)
                rows.append({
                    "time": ts,
                    "o": float(c["open"]),
                    "h": float(c["high"]),
                    "l": float(c["low"]),
                    "c": float(c["close"]),
                    "volume": int(c.get("tick_count", 0)),
                    "complete": True
                })
            df = pd.DataFrame(rows).sort_values("time").reset_index(drop=True)
            if not df.empty:
                return df
        except Exception as e:
            last_err = e
            log.error(f"Deriv WS try failed ({ep}) for {symbol}: {e}")
            continue
    raise RuntimeError(f"All Deriv WS endpoints failed for {symbol}: {last_err}")

async def deriv_m1(symbol: str, count: int = 500) -> pd.DataFrame:
    return await deriv_candles(symbol, 60, count)

async def deriv_m15(symbol: str, count: int = 200) -> pd.DataFrame:
    return await deriv_candles(symbol, 900, count)

# =========================
# Indicators / helpers
# =========================
def last_completed_idx(df: pd.DataFrame) -> Optional[int]:
    if df.empty:
        return None
    return int(df.index[-1])

def atr_percentile(series: pd.Series, lookback: int = 300) -> float:
    s = series.dropna()
    if len(s) == 0:
        return 0.0
    if len(s) > lookback:
        s = s.iloc[-lookback:]
    cur = s.iloc[-1]
    rank = float((s <= cur).sum()) / float(len(s))
    return rank

def ema_slope_pips(series: pd.Series, pips_factor: float) -> float:
    if series is None or series.isna().all() or len(series) < 2:
        return 0.0
    return float(series.iloc[-1] - series.iloc[-2]) * pips_factor

def pips_factor_for(symbol: str) -> float:
    # rough pip size (J* pairs ~ 0.01; most FX ~ 0.0001). We’ll inspect last close to guess decimals.
    if symbol.endswith("JPY"):
        return 100.0  # 1 pip = 0.01
    return 10000.0   # 1 pip = 0.0001

# =========================
# Dual Strategy
# =========================
def decide_signal_trend(inst: str, m1: pd.DataFrame, m15: pd.DataFrame, idx: int) -> Dict[str, Any]:
    """BLW-like trend following on M1, gated by M15 trend filters."""
    if idx is None or idx < 1 or m1.empty or m15.empty:
        return {"signal": "HOLD", "reason": "insufficient data"}

    close = m1["c"].astype(float)
    m1_ema9  = ta.ema(close, length=9)
    m1_ema21 = ta.ema(close, length=21)
    m1_rsi   = ta.rsi(close, length=14)
    m1_atr   = ta.atr(m1["h"], m1["l"], close, length=14)
    m1_adx   = ta.adx(m1["h"], m1["l"], close, length=14)["ADX_14"]

    cur = idx
    prev = idx - 1
    if prev < 0:
        return {"signal": "HOLD", "reason": "insufficient history"}

    # body & volatility filters
    body = abs(float(m1.iloc[cur]["c"]) - float(m1.iloc[cur]["o"]))
    if float(m1_atr.iloc[cur]) <= 0 or body < BODY_ATR_MIN * float(m1_atr.iloc[cur]):
        return {"signal": "HOLD", "reason": "small body vs ATR"}
    p_atr = atr_percentile(m1_atr)
    if not (ATR_PCTL_MIN <= p_atr <= ATR_PCTL_MAX):
        return {"signal": "HOLD", "reason": f"ATR pct {p_atr:.2f} out of window"}

    # M15 trend filters
    pf = pips_factor_for(SYMBOL_PRETTY.get(inst, inst))
    m15_close = m15["c"].astype(float)
    m15_adx   = ta.adx(m15["h"], m15["l"], m15_close, length=14)["ADX_14"]
    m15_ema   = ta.ema(m15_close, length=M15_EMA_TREND_LEN)
    if len(m15_ema.dropna()) < 2 or pd.isna(m15_adx.iloc[-1]):
        return {"signal": "HOLD", "reason": "M15 filters not ready"}
    slope = ema_slope_pips(m15_ema, pf)
    if float(m15_adx.iloc[-1]) < ADX_MIN_TREND or abs(slope) < M15_EMA_SLOPE_MIN_PIPS:
        return {"signal": "HOLD", "reason": "weak M15 trend"}

    # BLW cross + RSI
    sig, direction, why = "HOLD", None, "no cross"
    if (m1_ema9.iloc[prev] <= m1_ema21.iloc[prev]) and (m1_ema9.iloc[cur] > m1_ema21.iloc[cur]) and float(m1_rsi.iloc[cur]) >= RSI_UP:
        sig, direction, why = "CALL", "UP", f"EMA9↑EMA21 + RSI≥{RSI_UP}"
    elif (m1_ema9.iloc[prev] >= m1_ema21.iloc[prev]) and (m1_ema9.iloc[cur] < m1_ema21.iloc[cur]) and float(m1_rsi.iloc[cur]) <= RSI_DN:
        sig, direction, why = "PUT", "DOWN", f"EMA9↓EMA21 + RSI≤{RSI_DN}"

    return {
        "signal": sig,
        "direction": direction,
        "reason": why,
        "time": pd.to_datetime(m1.iloc[cur]["time"]).to_pydatetime().replace(tzinfo=timezone.utc),
        "price": float(m1.iloc[cur]["c"]),
    }

def decide_signal_chop(inst: str, m1: pd.DataFrame, m15: pd.DataFrame, idx: int) -> Dict[str, Any]:
    """Mean-reversion for choppy markets using BB(20,2) & weak ADX on M15."""
    if idx is None or idx < 1 or m1.empty or m15.empty:
        return {"signal": "HOLD", "reason": "insufficient data"}

    m15_close = m15["c"].astype(float)
    m15_adx   = ta.adx(m15["h"], m15["l"], m15_close, length=14)["ADX_14"]
    if pd.isna(m15_adx.iloc[-1]) or float(m15_adx.iloc[-1]) > ADX_MAX_CHOP:
        return {"signal": "HOLD", "reason": "not choppy"}

    close = m1["c"].astype(float)
    basis = ta.sma(close, length=BB_LENGTH)
    dev   = ta.stdev(close, length=BB_LENGTH)
    upper = basis + BB_STD * dev
    lower = basis - BB_STD * dev

    cur = idx
    if pd.isna(upper.iloc[cur]) or pd.isna(lower.iloc[cur]) or pd.isna(basis.iloc[cur]):
        return {"signal": "HOLD", "reason": "BB not ready"}

    price = float(close.iloc[cur])
    # Fade extremes toward mean
    if price <= float(lower.iloc[cur]):
        sig, direction, why = "CALL", "UP", "Lower BB touch → revert to mean"
    elif price >= float(upper.iloc[cur]):
        sig, direction, why = "PUT", "DOWN", "Upper BB touch → revert to mean"
    else:
        return {"signal": "HOLD", "reason": "no BB extreme"}

    return {
        "signal": sig,
        "direction": direction,
        "reason": why,
        "time": pd.to_datetime(m1.iloc[cur]["time"]).to_pydatetime().replace(tzinfo=timezone.utc),
        "price": float(price),
    }

def decide_signal_dual(inst: str, m1: pd.DataFrame, m15: pd.DataFrame, idx: int) -> Dict[str, Any]:
    """Try trend first; if HOLD due to weak trend, try chop."""
    trend = decide_signal_trend(inst, m1, m15, idx)
    if trend.get("signal") in ("CALL", "PUT"):
        trend["mode"] = "TREND"
        return trend
    chop = decide_signal_chop(inst, m1, m15, idx)
    if chop.get("signal") in ("CALL", "PUT"):
        chop["mode"] = "CHOP"
        return chop
    return {"signal": "HOLD", "reason": f"TREND: {trend.get('reason')} | CHOP: {chop.get('reason')}"}

# =========================
# Quotas & Stats
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
        await _write_json(QUOTA_FILE, _quota)

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
        await _write_json(STATS_FILE, _stats)

async def stats_on_settle(result: str):
    if result not in ("W", "L", "D"):
        return
    now = datetime.now(timezone.utc)
    dk, wk = _day_key(now), _week_key(now)
    async with _stats_lock:
        _stats["daily"].setdefault(dk, {"signals": 0, "W": 0, "L": 0, "D": 0})
        _stats["weekly"].setdefault(wk, {"signals": 0, "W": 0, "L": 0, "D": 0})
        _stats["daily"][dk][result] += 1
        _stats["weekly"][wk][result] += 1
        await _write_json(STATS_FILE, _stats)

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
    text = f"{title}\nSignals: {s}\nResults: ✅ {w} | ❌ {l} | ➖ {d}\nWin rate: {wr:.2f}%"
    for cid in [TELEGRAM_CHAT_FREE, TELEGRAM_CHAT_BASIC, TELEGRAM_CHAT_PRO, TELEGRAM_CHAT_VIP]:
        if cid:
            await tg_send(cid, text)

# =========================
# Engine: cooldown & settlement
# =========================
def _cooldown_ok(inst: str) -> bool:
    iso = _last_signal_time.get(inst)
    if not iso:
        return True
    last = datetime.fromisoformat(iso)
    return (datetime.now(timezone.utc) - last) >= timedelta(minutes=ALERT_COOLDOWN_MIN)

def _mark_cooldown(inst: str):
    _last_signal_time[inst] = datetime.now(timezone.utc).isoformat()

async def settle_due_trades():
    now = datetime.now(timezone.utc)
    to_remove: List[int] = []
    async with _trades_lock:
        for i, tr in enumerate(list(_open_trades)):
            if now >= datetime.fromisoformat(tr["settle_at"]):
                inst = tr["instrument"]
                try:
                    # use last closed price at/just before settle_at (real clock)
                    df = await deriv_m1(inst, 20)
                    if df.empty:
                        continue
                    df2 = df[df["time"] <= pd.to_datetime(tr["settle_at"])].tail(1)
                    if df2.empty:
                        df2 = df.tail(1)
                    close = float(df2.iloc[-1]["c"])
                    entry = float(tr["entry"])
                    direction = tr["direction"]
                    if abs(close - entry) < 1e-10:
                        res = "D"
                    elif direction == "UP":
                        res = "W" if close > entry else "L"
                    else:
                        res = "W" if close < entry else "L"
                    await stats_on_settle(res)
                    pretty = SYMBOL_PRETTY.get(inst, inst)
                    msg = f"⏱ {pretty} result: {res} (entry {entry:.5f} → close {close:.5f})"
                    for cid in tr.get("targets", []):
                        await tg_send(cid, msg)
                    to_remove.append(i)
                except Exception as e:
                    log.error(f"Settlement error {inst}: {e}")
        for idx in reversed(to_remove):
            _open_trades.pop(idx)
        if to_remove:
            await _write_json(OPEN_TRADES_FILE, _open_trades)

# =========================
# Signal flow
# =========================
def build_signal_text(inst: str, direction: str) -> str:
    pretty = SYMBOL_PRETTY.get(inst, inst)
    if direction == "UP":
        side = "🟢 BUY"
    else:
        side = "🔴 PUT"
    return f"{pretty} {side} • 1m candle • {PO_EXPIRY_MIN}m expiry"

async def try_send_signal(inst: str):
    if not fx_live_now():
        return
    if not _cooldown_ok(inst):
        return
    try:
        m1 = await deriv_m1(inst, 400)
        m15 = await deriv_m15(inst, 200)
        idx = last_completed_idx(m1)
        sig = decide_signal_dual(inst, m1, m15, idx)
        if sig.get("signal") not in ("CALL", "PUT"):
            return

        # choose target chats obeying quotas
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

        msg = build_signal_text(inst, sig["direction"])
        deliver: List[str] = []
        for cid, _ in targets:
            await tg_send(cid, msg)
            await quota_inc_today(cid)
            deliver.append(cid)

        # track for settlement
        settle_at = (datetime.now(timezone.utc) + timedelta(minutes=PO_EXPIRY_MIN)).isoformat()
        trade = {
            "instrument": inst,
            "direction": sig["direction"],
            "entry": sig["price"],
            "opened_at": datetime.now(timezone.utc).isoformat(),
            "settle_at": settle_at,
            "targets": deliver,
            "mode": sig.get("mode", "?"),
            "why": sig.get("reason", ""),
        }
        async with _trades_lock:
            _open_trades.append(trade)
            await _write_json(OPEN_TRADES_FILE, _open_trades)

        await stats_on_open()
        _mark_cooldown(inst)
        await _write_json(LAST_SIGNAL_FILE, _last_signal_time)

    except Exception as e:
        log.error(f"{inst} signal error: {e}")

async def run_engine():
    for inst in DERIV_SYMBOLS:
        await try_send_signal(inst)

# =========================
# Tally jobs (16:00 local, weekly Sundays)
# =========================
async def daily_tally_local():
    await send_tally("daily")

async def weekly_tally_utc():
    await send_tally("weekly")

# =========================
# Paywall (NOWPayments IPN) — minimal
# =========================
def _subs_set(user_key: str, tier: str, days: int):
    until = datetime.now(timezone.utc) + timedelta(days=days)
    _subs["active"][user_key] = {"tier": tier, "until": until.isoformat()}

def _subs_cleanup():
    now = datetime.now(timezone.utc)
    expired = [k for k, v in _subs["active"].items() if datetime.fromisoformat(v["until"]) < now]
    for k in expired:
        _subs["active"].pop(k, None)

@app_fastapi.post("/nowpayments/webhook")
async def nowpayments_webhook(request: Request, x_nowpayments_sig: Optional[str] = Header(default=None)):
    if not PAYWALL_ENABLED:
        return PlainTextResponse("disabled", status_code=200)
    # If you configured NOWPAYMENTS_IPN_SECRET, verify signature here (omitted for brevity)
    try:
        payload = await request.json()
    except Exception:
        return PlainTextResponse("bad json", status_code=400)

    # Expect your own metadata (e.g., user email or TG username) and plan → days
    status = (payload.get("payment_status") or "").lower()
    meta = payload.get("order_description") or ""  # e.g., "user:foo@example.com plan:VIP30"
    # naive parse:
    user_key = None
    if "user:" in meta:
        user_key = meta.split("user:")[-1].split()[0]
    plan_days = 30
    tier = "VIP"
    if "plan:" in meta:
        p = meta.split("plan:")[-1].split()[0].lower()
        if "basic" in p: tier, plan_days = "BASIC", 30
        elif "pro" in p: tier, plan_days = "PRO", 30
        elif "vip" in p: tier, plan_days = "VIP", 30
    if status in ("finished", "confirmed") and user_key:
        _subs_set(user_key, tier, plan_days)
        await _write_json(SUBS_FILE, _subs)
        return PlainTextResponse("ok", status_code=200)
    return PlainTextResponse("ignored", status_code=200)

# =========================
# FastAPI endpoints
# =========================
@app_fastapi.get("/", response_class=PlainTextResponse)
async def root():
    return "Dual Deriv Signals — FX live only"

@app_fastapi.get("/healthz", response_class=PlainTextResponse)
async def healthz():
    ok = all([
        bool(TELEGRAM_BOT_TOKEN),
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

# Telegram webhook (optional; polling default)
@app_fastapi.post(WEBHOOK_PATH)
async def telegram_webhook(request: Request):
    if not telegram_app:
        return PlainTextResponse("no-telegram", status_code=503)
    # Optional simple secret header check
    if request.headers.get("X-Telegram-Bot-Api-Secret-Token", "") != WEBHOOK_SECRET:
        return PlainTextResponse("forbidden", status_code=403)
    try:
        data = await request.json()
    except Exception:
        return PlainTextResponse("bad-json", status_code=400)
    update = Update.de_json(data, telegram_app.bot)
    await telegram_app.process_update(update)
    return PlainTextResponse("ok")

# =========================
# Lifespan
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
        if BASE_URL:
            await telegram_app.bot.set_webhook(
                url=f"{BASE_URL}{WEBHOOK_PATH}",
                secret_token=WEBHOOK_SECRET,
                drop_pending_updates=True
            )
            log.info(f"🤖 Telegram bot: webhook set {BASE_URL}{WEBHOOK_PATH}")
        else:
            log.warning("BASE_URL not set; webhook not configured. Bot still started.")

    # Scheduler (UTC)
    scheduler = AsyncIOScheduler(timezone=timezone.utc)
    # Engine: every minute, second 5
    scheduler.add_job(run_engine, CronTrigger(second="5"))
    # Settlement & housekeeping
    scheduler.add_job(settle_due_trades, CronTrigger(second="*/30"))
    # Daily tally at 16:00 local -> compute equivalent UTC cron (fixed offset; Port_of_Spain is UTC-4 all year)
    # 16:00 local == 20:00 UTC
    scheduler.add_job(daily_tally_local, CronTrigger(hour="20", minute="0"))
    # Weekly tally: Sundays 20:05 UTC
    scheduler.add_job(weekly_tally_utc, CronTrigger(day_of_week="sun", hour="20", minute="5"))
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
