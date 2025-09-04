# app.py
# Pocket Option Signals — Deriv-only BLW M1→5 Strategy
# Features:
# - Deriv candles (WebSocket)
# - 1m trigger, 5m timed expiry (actual timestamp)
# - Live session only (Mon–Fri 08:00–16:00 America/Port_of_Spain)
# - Per-pair 10m cooldown
# - Tiers/quotas: FREE(3), BASIC(6), PRO(15), VIP(∞)
# - Daily/weekly tally pushes to all groups
# -----------------------------------------------------

import os
import json
import hmac
import hashlib
import asyncio
import logging
from datetime import datetime, timedelta, timezone, time as dtime
from typing import Any, Dict, List, Optional, Tuple, Set

import pandas as pd
import pandas_ta as ta
import websockets
import httpx

from fastapi import FastAPI, Request
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
ENABLE_BOT_POLLING   = os.getenv("ENABLE_BOT_POLLING", "1").strip()  # default polling

# Chats
TELEGRAM_CHAT_FREE   = os.getenv("TELEGRAM_CHAT_FREE", "").strip()
TELEGRAM_CHAT_BASIC  = os.getenv("TELEGRAM_CHAT_BASIC", "").strip()
TELEGRAM_CHAT_PRO    = os.getenv("TELEGRAM_CHAT_PRO", "").strip()
TELEGRAM_CHAT_VIP    = os.getenv("TELEGRAM_CHAT_VIP", "").strip()

# Quotas
LIMIT_FREE           = int(os.getenv("LIMIT_FREE", "3"))
LIMIT_BASIC          = int(os.getenv("LIMIT_BASIC", "6"))
LIMIT_PRO            = int(os.getenv("LIMIT_PRO", "15"))
# VIP is unlimited

# Deriv feed
DERIV_ENDPOINT       = os.getenv("DERIV_ENDPOINT", "wss://ws.binaryws.com/websockets/v3?app_id=1089").strip()
# Use Deriv symbols (not OANDA): e.g., frxEURUSD, frxGBPUSD, frxUSDJPY, etc.
DERIV_INSTRUMENTS    = [s.strip() for s in os.getenv(
    "DERIV_INSTRUMENTS",
    "frxEURUSD,frxEURGBP,frxEURJPY,frxAUDCHF,frxCADJPY,frxGBPAUD,frxEURCAD,frxEURCHF,frxAUDCAD,frxCADCHF,frxCHFJPY,frxUSDCHF,frxGBPCHF"
).split(",") if s.strip()]

# Strategy (BLW tuned)
PO_EXPIRY_MIN        = int(os.getenv("PO_EXPIRY_MIN", "5"))   # timed expiry in minutes
ALERT_COOLDOWN_MIN   = int(os.getenv("ALERT_COOLDOWN_MIN", "10"))
ADX_MIN              = float(os.getenv("ADX_MIN", "18"))
BODY_ATR_MIN         = float(os.getenv("BODY_ATR_MIN", "0.30"))
ATR_PCTL_MIN         = float(os.getenv("ATR_PCTL_MIN", "0.30"))
ATR_PCTL_MAX         = float(os.getenv("ATR_PCTL_MAX", "0.92"))
RSI_UP               = float(os.getenv("RSI_UP", "52"))
RSI_DN               = float(os.getenv("RSI_DN", "48"))

# Session window (FX live)
SESSION_TZ_NAME      = os.getenv("SESSION_TZ", "America/Port_of_Spain")
LIVE_START_LOCAL     = os.getenv("LIVE_START_LOCAL", "08:00")
LIVE_END_LOCAL       = os.getenv("LIVE_END_LOCAL", "16:00")

# Storage
DATA_DIR             = os.getenv("DATA_DIR", "./data").strip()
STATS_FILE           = os.path.join(DATA_DIR, "stats.json")
OPEN_TRADES_FILE     = os.path.join(DATA_DIR, "open_trades.json")
QUOTA_FILE           = os.path.join(DATA_DIR, "quota.json")
LAST_SIGNAL_FILE     = os.path.join(DATA_DIR, "last_signal.json")

# =========================
# Globals / Locks
# =========================
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("pocket-option-signals")

app_fastapi = FastAPI()
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
    global _stats, _open_trades, _quota, _last_signal_time
    _stats = await _read_json(STATS_FILE, {"daily": {}, "weekly": {}})
    _open_trades = await _read_json(OPEN_TRADES_FILE, [])
    _quota = await _read_json(QUOTA_FILE, {})
    _last_signal_time = await _read_json(LAST_SIGNAL_FILE, {})

async def save_stats():        await _write_json(STATS_FILE, _stats)
async def save_open_trades():  await _write_json(OPEN_TRADES_FILE, _open_trades)
async def save_quota():        await _write_json(QUOTA_FILE, _quota)
async def save_last_signal():  await _write_json(LAST_SIGNAL_FILE, _last_signal_time)

# =========================
# Time / Session guard
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
    if now.weekday() > 4:  # Sat(5), Sun(6)
        return False
    t = now.time()
    return LIVE_START <= t <= LIVE_END

# =========================
# Deriv fetcher (WebSocket)
# =========================
async def deriv_candles(symbol: str, granularity_sec: int = 60, count: int = 500) -> pd.DataFrame:
    req = {
        "ticks_history": symbol,
        "granularity": granularity_sec,
        "count": count,
        "end": "latest",
        "style": "candles"
    }
    async with websockets.connect(DERIV_ENDPOINT, ping_interval=20, ping_timeout=20) as ws:
        await ws.send(json.dumps(req))
        resp_raw = await ws.recv()
    try:
        resp = json.loads(resp_raw)
    except Exception:
        log.error(f"Deriv bad JSON for {symbol}")
        return pd.DataFrame()

    candles = resp.get("candles") or []
    rows = []
    for c in candles:
        try:
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
        except Exception:
            continue
    df = pd.DataFrame(rows).sort_values("time").reset_index(drop=True)
    return df

async def deriv_m1(symbol: str, count: int = 500) -> pd.DataFrame:  return await deriv_candles(symbol, 60, count)
async def deriv_m5(symbol: str, count: int = 500) -> pd.DataFrame:  return await deriv_candles(symbol, 300, count)
async def deriv_m15(symbol: str, count: int = 500) -> pd.DataFrame: return await deriv_candles(symbol, 900, count)

def last_completed_idx(df: pd.DataFrame) -> Optional[int]:
    if df.empty: return None
    return int(df.index[-1])

def atr_percentile(series: pd.Series, lookback: int = 300) -> float:
    s = series.dropna()
    if len(s) == 0: return 0.0
    if len(s) > lookback: s = s.iloc[-lookback:]
    cur = s.iloc[-1]
    rank = (s <= cur).sum() / len(s)
    return float(rank)

# =========================
# BLW M1→5 Strategy
# =========================
def compute_blw_m1x5(inst: str, m1: pd.DataFrame, idx: int) -> Dict[str, Any]:
    if idx is None or idx < 1 or m1.empty:
        return {"signal": "HOLD", "reason": "insufficient data"}
    df = m1.copy()
    close = df["c"].astype(float)
    df["ema9"]  = ta.ema(close, length=9)
    df["ema21"] = ta.ema(close, length=21)
    df["rsi"]   = ta.rsi(close, length=14)
    df["atr"]   = ta.atr(df["h"], df["l"], close, length=14)
    adx_df      = ta.adx(df["h"], df["l"], close, length=14)
    df["adx"]   = adx_df["ADX_14"] if adx_df is not None and "ADX_14" in adx_df.columns else pd.NA

    cur, prev = df.iloc[idx], df.iloc[idx - 1]

    body = abs(float(cur["c"]) - float(cur["o"]))
    if float(cur["atr"]) <= 0 or body < BODY_ATR_MIN * float(cur["atr"]):
        return {"signal": "HOLD", "reason": "small body vs ATR"}

    p_atr = atr_percentile(df["atr"])
    if not (ATR_PCTL_MIN <= p_atr <= ATR_PCTL_MAX):
        return {"signal": "HOLD", "reason": f"ATR pct {p_atr:.2f} out of window"}

    if pd.isna(cur["adx"]) or float(cur["adx"]) < ADX_MIN:
        return {"signal": "HOLD", "reason": f"ADX {float(cur['adx']) if not pd.isna(cur['adx']) else 'nan'} < {ADX_MIN}"}

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
# Telegram
# =========================
HELP_TEXT = (
    "Pocket Option (Deriv) — BLW M1→5 🤖\n"
    "• FX live only (Mon–Fri 08:00–16:00 America/Port_of_Spain)\n"
    "• Tiers: FREE(3/d), BASIC(6/d), PRO(15/d), VIP(∞)\n"
    "• 1m candles, 5m timed expiry, 10m per-pair cooldown\n"
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
    chat_id = update.effective_chat.id if update.effective_chat else None
    if chat_id is None: return
    tier, lim = resolve_tier_and_limit(str(chat_id))
    used = await quota_get_today(str(chat_id))
    rem = "∞" if lim is None else max(lim - used, 0)
    await update.message.reply_text(f"Tier: {tier}\nUsed today: {used}\nRemaining: {rem}")

async def tg_send(chat_id: str, text: str):
    if not telegram_app: return
    try:
        await telegram_app.bot.send_message(chat_id=chat_id, text=text, disable_web_page_preview=True)
    except Exception as e:
        log.error(f"Telegram send failed to {chat_id}: {e}")

def resolve_tier_and_limit(chat_id: str) -> Tuple[str, Optional[int]]:
    if TELEGRAM_CHAT_VIP and chat_id == TELEGRAM_CHAT_VIP:     return ("VIP", None)
    if TELEGRAM_CHAT_PRO and chat_id == TELEGRAM_CHAT_PRO:     return ("PRO", LIMIT_PRO)
    if TELEGRAM_CHAT_BASIC and chat_id == TELEGRAM_CHAT_BASIC: return ("BASIC", LIMIT_BASIC)
    if TELEGRAM_CHAT_FREE and chat_id == TELEGRAM_CHAT_FREE:   return ("FREE", LIMIT_FREE)
    return ("FREE", LIMIT_FREE)

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
    if key not in ("W", "L", "D"): return
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
# Engine (signals & settlement)
# =========================
def _cooldown_ok(inst: str) -> bool:
    last_iso = _last_signal_time.get(inst)
    if not last_iso: return True
    last = datetime.fromisoformat(last_iso)
    return (datetime.now(timezone.utc) - last) >= timedelta(minutes=ALERT_COOLDOWN_MIN)

def _mark_cooldown(inst: str):
    _last_signal_time[inst] = datetime.now(timezone.utc).isoformat()

def _symbol_to_pretty(symbol: str) -> str:
    # frxEURUSD -> EUR/USD, etc.
    if symbol.startswith("frx") and len(symbol) >= 9:
        a = symbol[3:6]
        b = symbol[6:9]
        return f"{a}/{b}"
    return symbol

async def settle_due_trades():
    """Use actual expiry timestamp; pull the last candle <= settle_at."""
    now = datetime.now(timezone.utc)
    to_remove: List[int] = []
    async with _trades_lock:
        for i, tr in enumerate(list(_open_trades)):
            settle_at = datetime.fromisoformat(tr["settle_at"])
            if now < settle_at:
                continue
            inst = tr["instrument"]
            try:
                df = await deriv_m1(inst, 30)
                if df.empty:
                    continue
                df2 = df[df["time"] <= pd.to_datetime(settle_at, utc=True)].tail(1)
                if df2.empty:
                    continue
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
                pretty = _symbol_to_pretty(inst)
                msg = f"🔔 {pretty} result: {res} (entry {entry:.5f} → close {close:.5f})"
                for cid in tr.get("targets", []):
                    await tg_send(cid, msg)
                to_remove.append(i)
            except Exception as e:
                log.error(f"Settlement error {inst}: {e}")
        for idx in reversed(to_remove):
            _open_trades.pop(idx)
        if to_remove:
            await save_open_trades()

async def try_send_signal(inst: str):
    if not fx_live_now():
        return
    if not _cooldown_ok(inst):
        return
    try:
        m1 = await deriv_m1(inst, 400)
        idx = last_completed_idx(m1)
        sig = compute_blw_m1x5(inst, m1, idx)
        if sig.get("signal") not in ("CALL", "PUT"):
            return

        # choose target chats with quotas
        targets: List[Tuple[str, Optional[int]]] = []
        ordered = [
            (TELEGRAM_CHAT_VIP, None),
            (TELEGRAM_CHAT_PRO, LIMIT_PRO),
            (TELEGRAM_CHAT_BASIC, LIMIT_BASIC),
            (TELEGRAM_CHAT_FREE, LIMIT_FREE),
        ]
        for cid, lim in ordered:
            if not cid: continue
            used = await quota_get_today(cid)
            if (lim is None) or (used < lim):
                targets.append((cid, lim))

        if not targets:
            return

        pretty = _symbol_to_pretty(inst)
        arrow  = "🟢 BUY" if sig["direction"] == "UP" else "🔴 PUT"
        text = f"{pretty} {arrow} • 1m candle • {PO_EXPIRY_MIN}m expiry"

        deliver_to: List[str] = []
        for cid, _ in targets:
            await tg_send(cid, text)
            await quota_inc_today(cid)
            deliver_to.append(cid)

        settle_at = (datetime.now(timezone.utc) + timedelta(minutes=PO_EXPIRY_MIN)).astimezone(timezone.utc).isoformat()
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
        log.error(f"{inst} signal error: {e}")

# =========================
# Scheduler jobs
# =========================
async def run_engine():
    for inst in DERIV_INSTRUMENTS:
        await try_send_signal(inst)

async def daily_tally_job():
    await send_tally("daily")

async def weekly_tally_job():
    await send_tally("weekly")

# =========================
# FastAPI endpoints
# =========================
@app_fastapi.get("/", response_class=PlainTextResponse)
async def root():
    return "Pocket Option (Deriv) — BLW M1→5 — FX live only"

@app_fastapi.get("/healthz", response_class=PlainTextResponse)
async def healthz():
    ok = bool(TELEGRAM_BOT_TOKEN) and bool(DERIV_ENDPOINT) and len(DERIV_INSTRUMENTS) > 0
    return "ok" if ok else "not-ok"

@app_fastapi.get("/status")
async def status():
    return {
        "now": datetime.now(timezone.utc).isoformat(),
        "fx_live_now": fx_live_now(),
        "pairs": DERIV_INSTRUMENTS,
        "cooldown_min": ALERT_COOLDOWN_MIN,
        "expiry_min": PO_EXPIRY_MIN,
    }

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

# =========================
# Lifespan
# =========================
@app_fastapi.on_event("startup")
async def on_startup():
    global telegram_app, scheduler
    await load_state()

    telegram_app = build_telegram_app()
    await telegram_app.initialize()
    await telegram_app.start()
    if ENABLE_BOT_POLLING == "1":
        await telegram_app.bot.delete_webhook(drop_pending_updates=True)
        log.info("🤖 Telegram bot: polling started")
    else:
        await telegram_app.bot.set_webhook(url=f"{BASE_URL}{WEBHOOK_PATH}", secret_token=WEBHOOK_SECRET, drop_pending_updates=True)
        log.info(f"🤖 Telegram bot: webhook set {BASE_URL}{WEBHOOK_PATH}")

    scheduler = AsyncIOScheduler(timezone=timezone.utc)
    scheduler.add_job(run_engine, CronTrigger(second="5"))            # every minute at :05
    scheduler.add_job(settle_due_trades, CronTrigger(second="*/30"))  # settle twice a minute
    # Daily tally at local 16:00 => Port of Spain is UTC-4 (no DST): 20:00 UTC
    scheduler.add_job(daily_tally_job, CronTrigger(hour="20", minute="0"))
    # Weekly tally Sunday 20:10 UTC
    scheduler.add_job(weekly_tally_job, CronTrigger(day_of_week="sun", hour="20", minute="10"))
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
