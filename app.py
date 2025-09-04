#!/usr/bin/env python3
# Pocket Option Signals — Deriv Dual (BLW TREND + CHOP mean-reversion)
# - Deriv candles & ticks (binary-aligned)
# - Dual "AI" regime: Trending vs Choppy auto-switch
# - M1 trigger, true 5m expiry (tick-settlement at expiry timestamp)
# - 10m pair cooldown, FX live only (Mon–Fri 08:00–16:00 local)
# - Quotas per tier (FREE/BASIC/PRO/VIP)
# - Daily tally (16:00 local) + Weekly tally (Sun 16:05 local)
# - NOWPayments IPN: auto-issue invite links; expiry watcher can kick on lapse

import os, json, asyncio, logging, math, hmac, hashlib, re
from datetime import datetime, timedelta, timezone, time as dtime
from typing import Dict, Any, List, Optional, Tuple

import pandas as pd
import pandas_ta as ta
import websockets

from fastapi import FastAPI, Request, Header
from fastapi.responses import PlainTextResponse
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

from telegram import Update, ChatInviteLink
from telegram.ext import Application, CommandHandler, ContextTypes

# =========================
# ENV
# =========================
TELEGRAM_BOT_TOKEN   = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
BASE_URL             = os.getenv("BASE_URL", "").rstrip("/")
WEBHOOK_PATH         = os.getenv("WEBHOOK_PATH", "/telegram/webhook").strip()
WEBHOOK_SECRET       = os.getenv("WEBHOOK_SECRET", "change-me").strip()
ENABLE_BOT_POLLING   = os.getenv("ENABLE_BOT_POLLING", "1").strip()  # "1" polling (default), "0" webhook

# Chats (strings, e.g. "-1001...")
TELEGRAM_CHAT_FREE   = os.getenv("TELEGRAM_CHAT_FREE", "").strip()
TELEGRAM_CHAT_BASIC  = os.getenv("TELEGRAM_CHAT_BASIC", "").strip()
TELEGRAM_CHAT_PRO    = os.getenv("TELEGRAM_CHAT_PRO", "").strip()
TELEGRAM_CHAT_VIP    = os.getenv("TELEGRAM_CHAT_VIP", "").strip()

# Quotas per group/day
LIMIT_FREE           = int(os.getenv("LIMIT_FREE", "3"))
LIMIT_BASIC          = int(os.getenv("LIMIT_BASIC", "6"))
LIMIT_PRO            = int(os.getenv("LIMIT_PRO", "15"))  # VIP unlimited

# Deriv endpoint & pairs
DERIV_ENDPOINT       = os.getenv("DERIV_ENDPOINT", "wss://ws.derivws.com/websockets/v3?app_id=1089").strip()
DERIV_PAIRS          = [s.strip() for s in os.getenv(
    "DERIV_PAIRS",
    # winners mapping to Deriv codes by default:
    "frxAUDUSD,frxEURGBP,frxEURJPY,frxAUDCHF,frxCADJPY,frxGBPAUD,frxEURCAD,frxEURCHF,frxAUDCAD,frxCADCHF,frxCHFJPY,frxUSDCHF,frxGBPCHF"
).split(",") if s.strip()]

# Strategy knobs (shared)
PO_EXPIRY_MIN        = int(os.getenv("PO_EXPIRY_MIN", "5"))
ALERT_COOLDOWN_MIN   = int(os.getenv("ALERT_COOLDOWN_MIN", "10"))
BODY_ATR_MIN         = float(os.getenv("BODY_ATR_MIN", "0.30"))
ATR_PCTL_MIN         = float(os.getenv("ATR_PCTL_MIN", "0.30"))
ATR_PCTL_MAX         = float(os.getenv("ATR_PCTL_MAX", "0.92"))

# TREND (BLW-like)
ADX_MIN_TREND        = float(os.getenv("ADX_MIN_TREND", "20"))
RSI_UP               = float(os.getenv("RSI_UP", "52"))
RSI_DN               = float(os.getenv("RSI_DN", "48"))
EMA_SLOPE_LEN        = int(os.getenv("EMA_SLOPE_LEN", "200"))
EMA_SLOPE_MIN_PIPS   = float(os.getenv("EMA_SLOPE_MIN_PIPS", "0.2"))

# CHOP (mean-reversion)
ADX_MAX_CHOP         = float(os.getenv("ADX_MAX_CHOP", "18"))
BB_LENGTH            = int(os.getenv("BB_LENGTH", "20"))
BB_STD               = float(os.getenv("BB_STD", "2.0"))
RSI_MID              = float(os.getenv("RSI_MID", "50"))

# Tallies
ENABLE_WEEKLY_TALLY  = os.getenv("ENABLE_WEEKLY_TALLY", "1").strip()

# Session (local time)
SESSION_TZ_NAME      = os.getenv("SESSION_TZ", "America/Port_of_Spain").strip()
LIVE_START_LOCAL     = os.getenv("LIVE_START_LOCAL", "08:00").strip()
LIVE_END_LOCAL       = os.getenv("LIVE_END_LOCAL", "16:00").strip()

# NOWPayments
NP_IPN_SECRET        = os.getenv("NOWPAYMENTS_IPN_SECRET", "").strip()
# OrderID convention you send when creating invoice: "tg:<USER_ID>|plan:<VIP|PRO|BASIC>|days:<N>"
# Example: "tg:123456789|plan:PRO|days:30"

# Storage
DATA_DIR             = os.getenv("DATA_DIR", "./data").strip()
STATS_FILE           = os.path.join(DATA_DIR, "stats.json")
QUOTA_FILE           = os.path.join(DATA_DIR, "quota.json")
TRADES_FILE          = os.path.join(DATA_DIR, "open_trades.json")
COOLDOWN_FILE        = os.path.join(DATA_DIR, "cooldowns.json")
MEMBERS_FILE         = os.path.join(DATA_DIR, "members.json")  # {user_id:{tier, expires_at, chat_id, invite_link}}

# =========================
# Logging & globals
# =========================
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("dual-deriv-bot")

app_fastapi: FastAPI = FastAPI()
telegram_app: Optional[Application] = None
scheduler: Optional[AsyncIOScheduler] = None

_stats_lock = asyncio.Lock()
_quota_lock = asyncio.Lock()
_trades_lock = asyncio.Lock()
_io_lock = asyncio.Lock()
_members_lock = asyncio.Lock()

_stats: Dict[str, Any] = {"daily": {}, "weekly": {}}
_quota: Dict[str, Dict[str, int]] = {}
_open_trades: List[Dict[str, Any]] = []
_cooldowns: Dict[str, str] = {}
_members: Dict[str, Dict[str, Any]] = {}  # user_id => {tier, expires_at, chat_id, invite_link}
_last_daily_sent_for: Optional[str] = None
_last_weekly_sent_for: Optional[str] = None

# =========================
# Time helpers / session
# =========================
try:
    from zoneinfo import ZoneInfo
except Exception:
    from backports.zoneinfo import ZoneInfo  # type: ignore

LOCAL_TZ = ZoneInfo(SESSION_TZ_NAME)

def _parse_hhmm(s: str) -> dtime:
    hh, mm = s.split(":")
    return dtime(int(hh), int(mm))

LIVE_START = _parse_hhmm(LIVE_START_LOCAL)
LIVE_END   = _parse_hhmm(LIVE_END_LOCAL)

def now_local() -> datetime:
    return datetime.now(timezone.utc).astimezone(LOCAL_TZ)

def fx_live_now() -> bool:
    n = now_local()
    if n.weekday() > 4:
        return False
    return LIVE_START <= n.time() <= LIVE_END

def is_tally_moment_daily(n: Optional[datetime] = None) -> bool:
    n = n or now_local()
    return n.time().hour == LIVE_END.hour and n.time().minute == LIVE_END.minute

def is_tally_moment_weekly(n: Optional[datetime] = None) -> bool:
    n = n or now_local()
    # Sunday 16:05 local
    return (n.weekday() == 6) and (n.time().hour == LIVE_END.hour) and (n.time().minute == (LIVE_END.minute + 5) % 60)

# =========================
# FS utils
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
    global _stats, _quota, _open_trades, _cooldowns, _members
    _stats = await _read_json(STATS_FILE, {"daily": {}, "weekly": {}})
    _quota = await _read_json(QUOTA_FILE, {})
    _open_trades = await _read_json(TRADES_FILE, [])
    _cooldowns = await _read_json(COOLDOWN_FILE, {})
    _members = await _read_json(MEMBERS_FILE, {})

async def save_stats():    await _write_json(STATS_FILE, _stats)
async def save_quota():    await _write_json(QUOTA_FILE, _quota)
async def save_trades():   await _write_json(TRADES_FILE, _open_trades)
async def save_cooldowns():await _write_json(COOLDOWN_FILE, _cooldowns)
async def save_members():  await _write_json(MEMBERS_FILE, _members)

# =========================
# Deriv fetchers
# =========================
async def deriv_candles(symbol: str, granularity_sec: int, count: int = 400) -> pd.DataFrame:
    req = {
        "ticks_history": symbol,
        "granularity": granularity_sec,
        "count": count,
        "end": "latest",
        "style": "candles",
    }
    async with websockets.connect(DERIV_ENDPOINT, ping_interval=20, ping_timeout=20) as ws:
        await ws.send(json.dumps(req))
        resp = json.loads(await ws.recv())
    rows = []
    for c in resp.get("candles", []):
        ts = pd.to_datetime(c["epoch"], unit="s", utc=True)
        rows.append({
            "time": ts,
            "o": float(c["open"]),
            "h": float(c["high"]),
            "l": float(c["low"]),
            "c": float(c["close"]),
            "complete": True
        })
    return pd.DataFrame(rows).sort_values("time").reset_index(drop=True)

async def deriv_m1(symbol: str, count: int = 400) -> pd.DataFrame:
    return await deriv_candles(symbol, 60, count)

async def deriv_tick_at_or_before(symbol: str, when_utc: datetime) -> Optional[float]:
    req = {"ticks_history": symbol, "end": int(when_utc.timestamp()), "count": 1, "style": "ticks"}
    async with websockets.connect(DERIV_ENDPOINT, ping_interval=20, ping_timeout=20) as ws:
        await ws.send(json.dumps(req))
        resp = json.loads(await ws.recv())
    # history.prices or ticks
    prices = None
    if "history" in resp and "prices" in resp["history"]:
        prices = resp["history"]["prices"]
    elif "ticks" in resp and isinstance(resp["ticks"], list):
        prices = resp["ticks"]
    if prices:
        return float(prices[-1])
    return None

# =========================
# Indicators / utilities
# =========================
def last_completed_idx(df: pd.DataFrame) -> Optional[int]:
    if df is None or df.empty: return None
    return int(df.index[-1])

def atr_percentile(series: pd.Series, lookback: int = 300) -> float:
    s = series.dropna()
    if s.empty: return 0.0
    if len(s) > lookback: s = s.iloc[-lookback:]
    cur = s.iloc[-1]
    return float((s <= cur).sum() / len(s))

def pip_size(symbol: str) -> float:
    # crude: if JPY in code → 0.01 else 0.0001
    s = symbol.upper()
    return 0.01 if "JPY" in s else 0.0001

def fmt_pair(symbol: str) -> str:
    # frxEURUSD -> EUR/USD
    s = symbol.replace("frx", "")
    return f"{s[:3]}/{s[3:]}" if len(s) == 6 else s

# =========================
# Dual "AI" regime + strategies
# =========================
def regime_classifier(df: pd.DataFrame, symbol: str) -> str:
    """
    Lightweight regime selection:
      - TREND if ADX>=ADX_MIN_TREND and EMA(EMA_SLOPE_LEN) slope >= EMA_SLOPE_MIN_PIPS/min
      - CHOP  if ADX<=ADX_MAX_CHOP
      - HOLD otherwise
    """
    if df is None or df.empty or len(df) < (EMA_SLOPE_LEN + 2):
        return "HOLD"
    close = df["c"].astype(float)
    adx = ta.adx(df["h"], df["l"], close, length=14)["ADX_14"]
    ema = ta.ema(close, length=EMA_SLOPE_LEN)

    ps = pip_size(symbol)
    # slope in pips per candle (1m)
    slope = (ema.iloc[-1] - ema.iloc[-EMA_SLOPE_LEN]) / max(EMA_SLOPE_LEN - 1, 1)
    slope_pips = abs(float(slope) / ps)

    adx_v = float(adx.iloc[-1]) if not pd.isna(adx.iloc[-1]) else 0.0
    if (adx_v >= ADX_MIN_TREND) and (slope_pips >= EMA_SLOPE_MIN_PIPS):
        return "TREND"
    if adx_v <= ADX_MAX_CHOP:
        return "CHOP"
    return "HOLD"

def trend_signal(df: pd.DataFrame, idx: int) -> Dict[str, Any]:
    """
    BLW-like:
      - EMA9/EMA21 cross + RSI side
      - ADX >= ADX_MIN_TREND
      - Body >= BODY_ATR_MIN * ATR
      - ATR percentile in [ATR_PCTL_MIN, ATR_PCTL_MAX]
    """
    if idx is None or idx < 1 or df.empty: return {"signal":"HOLD","reason":"insufficient"}
    df = df.copy()
    close = df["c"].astype(float)
    df["ema9"]  = ta.ema(close, length=9)
    df["ema21"] = ta.ema(close, length=21)
    df["rsi"]   = ta.rsi(close, length=14)
    df["atr"]   = ta.atr(df["h"], df["l"], close, length=14)
    df["adx"]   = ta.adx(df["h"], df["l"], close, length=14)["ADX_14"]

    cur, prev = df.iloc[idx], df.iloc[idx-1]
    # body & atr pctl
    body = abs(float(cur["c"]) - float(cur["o"]))
    if float(cur["atr"]) <= 0 or body < BODY_ATR_MIN * float(cur["atr"]):
        return {"signal":"HOLD","reason":"small body"}
    p = atr_percentile(df["atr"])
    if not (ATR_PCTL_MIN <= p <= ATR_PCTL_MAX):
        return {"signal":"HOLD","reason":"atr pctl out"}
    if pd.isna(cur["adx"]) or float(cur["adx"]) < ADX_MIN_TREND:
        return {"signal":"HOLD","reason":"weak adx"}

    if (prev["ema9"] <= prev["ema21"]) and (cur["ema9"] > cur["ema21"]) and float(cur["rsi"]) >= RSI_UP:
        return {"signal":"CALL","direction":"UP","price":float(cur["c"]),"time":pd.to_datetime(cur["time"]).to_pydatetime().replace(tzinfo=timezone.utc),"reason":"EMA9>21 & RSI up"}
    if (prev["ema9"] >= prev["ema21"]) and (cur["ema9"] < cur["ema21"]) and float(cur["rsi"]) <= RSI_DN:
        return {"signal":"PUT","direction":"DOWN","price":float(cur["c"]),"time":pd.to_datetime(cur["time"]).to_pydatetime().replace(tzinfo=timezone.utc),"reason":"EMA9<21 & RSI down"}
    return {"signal":"HOLD","reason":"no cross"}

def chop_signal(df: pd.DataFrame, idx: int) -> Dict[str, Any]:
    """
    Mean-reversion:
      - Bollinger Bands (BB_LEN, BB_STD)
      - Signal when close pierces outer band and RSI is beyond mid; look for revert
      - Keep body >= BODY_ATR_MIN*ATR and ATR pctl window for basic quality
    """
    if idx is None or idx < 2 or df.empty: return {"signal":"HOLD","reason":"insufficient"}
    df = df.copy()
    close = df["c"].astype(float)
    bb = ta.bbands(close, length=BB_LENGTH, std=BB_STD)  # columns: BBL_20_2.0, BBM, BBU
    df["bb_l"], df["bb_m"], df["bb_u"] = bb.iloc[:,0], bb.iloc[:,1], bb.iloc[:,2]
    df["rsi"] = ta.rsi(close, length=14)
    df["atr"] = ta.atr(df["h"], df["l"], close, length=14)

    cur, prev = df.iloc[idx], df.iloc[idx-1]
    if pd.isna(cur["bb_l"]) or pd.isna(cur["bb_u"]): return {"signal":"HOLD","reason":"bb na"}
    # body & atr window
    body = abs(float(cur["c"]) - float(cur["o"]))
    if float(cur["atr"]) <= 0 or body < BODY_ATR_MIN * float(cur["atr"]):
        return {"signal":"HOLD","reason":"small body"}
    p = atr_percentile(df["atr"])
    if not (ATR_PCTL_MIN <= p <= ATR_PCTL_MAX):
        return {"signal":"HOLD","reason":"atr pctl out"}

    # If previous closed below lower band and RSI<RSI_MID → expect reversion up (CALL)
    if (float(prev["c"]) < float(prev["bb_l"])) and float(prev["rsi"]) < RSI_MID:
        return {"signal":"CALL","direction":"UP","price":float(cur["c"]),"time":pd.to_datetime(cur["time"]).to_pydatetime().replace(tzinfo=timezone.utc),"reason":"revert from lower BB"}
    # If previous closed above upper band and RSI>RSI_MID → expect reversion down (PUT)
    if (float(prev["c"]) > float(prev["bb_u"])) and float(prev["rsi"]) > RSI_MID:
        return {"signal":"PUT","direction":"DOWN","price":float(cur["c"]),"time":pd.to_datetime(cur["time"]).to_pydatetime().replace(tzinfo=timezone.utc),"reason":"revert from upper BB"}

    return {"signal":"HOLD","reason":"no band pierce"}

# =========================
# Quotas & stats
# =========================
def _day_key(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%d")

def _week_key(dt: datetime) -> str:
    iso = dt.isocalendar()
    return f"{iso.year}-W{iso.week:02d}"

def resolve_tier_and_limit(chat_id: str) -> Tuple[str, Optional[int]]:
    if TELEGRAM_CHAT_VIP and chat_id == TELEGRAM_CHAT_VIP:    return ("VIP", None)
    if TELEGRAM_CHAT_PRO and chat_id == TELEGRAM_CHAT_PRO:    return ("PRO", LIMIT_PRO)
    if TELEGRAM_CHAT_BASIC and chat_id == TELEGRAM_CHAT_BASIC:return ("BASIC", LIMIT_BASIC)
    if TELEGRAM_CHAT_FREE and chat_id == TELEGRAM_CHAT_FREE:  return ("FREE", LIMIT_FREE)
    return ("FREE", LIMIT_FREE)

async def quota_get_today(chat_id: str) -> int:
    today = _day_key(datetime.now(timezone.utc))
    async with _quota_lock:
        return int(_quota.get(today, {}).get(chat_id, 0))

async def quota_inc_today(chat_id: str):
    today = _day_key(datetime.now(timezone.utc))
    async with _quota_lock:
        _quota.setdefault(today, {})
        _quota[today][chat_id] = int(_quota[today].get(chat_id, 0)) + 1
        await save_quota()

async def stats_on_open():
    now = datetime.now(timezone.utc)
    dk, wk = _day_key(now), _week_key(now)
    async with _stats_lock:
        _stats["daily"].setdefault(dk, {"signals":0,"W":0,"L":0,"D":0})
        _stats["daily"][dk]["signals"] += 1
        _stats["weekly"].setdefault(wk, {"signals":0,"W":0,"L":0,"D":0})
        _stats["weekly"][wk]["signals"] += 1
        await save_stats()

async def stats_on_settle(result: str):
    if result not in ("W","L","D"): return
    now = datetime.now(timezone.utc)
    dk, wk = _day_key(now), _week_key(now)
    async with _stats_lock:
        _stats["daily"].setdefault(dk, {"signals":0,"W":0,"L":0,"D":0})
        _stats["daily"][dk][result] += 1
        _stats["weekly"].setdefault(wk, {"signals":0,"W":0,"L":0,"D":0})
        _stats["weekly"][wk][result] += 1
        await save_stats()

# =========================
# Telegram
# =========================
HELP_TEXT = (
    "Deriv Dual Signals 🤖 (BLW Trend + MR Chop)\n"
    "• FX live only (Mon–Fri 08:00–16:00 local)\n"
    "• 1m candle, 5m expiry, 10m cooldown\n"
    "• Tiers: FREE(3/d), BASIC(6/d), PRO(15/d), VIP(∞)\n"
    "• Daily tally 16:00 + Weekly tally Sun 16:05\n\n"
    "Commands:\n"
    "/start — intro\n"
    "/ping  — health\n"
    "/limits — today’s remaining quota\n"
)

def build_telegram_app() -> Application:
    app = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", start_cmd))
    app.add_handler(CommandHandler("ping",  ping_cmd))
    app.add_handler(CommandHandler("limits", limits_cmd))
    return app

async def start_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(HELP_TEXT)

async def ping_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("pong")

async def limits_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = str(update.effective_chat.id) if update.effective_chat else ""
    used = await quota_get_today(chat_id)
    tier, lim = resolve_tier_and_limit(chat_id)
    rem = "∞" if lim is None else max(lim - used, 0)
    await update.message.reply_text(f"Tier: {tier}\nUsed today: {used}\nRemaining: {rem}")

async def tg_send(chat_id: str, text: str):
    if not telegram_app: return
    try:
        await telegram_app.bot.send_message(chat_id=chat_id, text=text, disable_web_page_preview=True)
    except Exception as e:
        log.error(f"Telegram send failed to {chat_id}: {e}")

# =========================
# Engine
# =========================
def cooldown_ok(inst: str) -> bool:
    iso = _cooldowns.get(inst)
    if not iso: return True
    last = datetime.fromisoformat(iso)
    return (datetime.now(timezone.utc) - last) >= timedelta(minutes=ALERT_COOLDOWN_MIN)

def mark_cooldown(inst: str):
    _cooldowns[inst] = datetime.now(timezone.utc).isoformat()

async def try_send_signal(inst: str):
    if not fx_live_now(): return
    if not cooldown_ok(inst): return

    try:
        df = await deriv_m1(inst, 400)
        idx = last_completed_idx(df)
        if idx is None: return

        regime = regime_classifier(df, inst)
        if regime == "TREND":
            sig = trend_signal(df, idx)
        elif regime == "CHOP":
            sig = chop_signal(df, idx)
        else:
            return

        if sig.get("signal") not in ("CALL","PUT"):
            return

        # concise message with explicit candle/expiry
        side = "🟢 BUY" if sig["direction"] == "UP" else "🔴 PUT"
        text = f"{fmt_pair(inst)} {side} | 1m candle ⏱ {PO_EXPIRY_MIN}m expiry"

        # broadcast per tier respecting quotas
        for cid, lim in [
            (TELEGRAM_CHAT_VIP, None),
            (TELEGRAM_CHAT_PRO, LIMIT_PRO),
            (TELEGRAM_CHAT_BASIC, LIMIT_BASIC),
            (TELEGRAM_CHAT_FREE, LIMIT_FREE),
        ]:
            if not cid: continue
            used = await quota_get_today(cid)
            if (lim is None) or (used < lim):
                await tg_send(cid, text)
                await quota_inc_today(cid)

        # register open trade for tick settlement at true expiry time
        settle_at = datetime.now(timezone.utc) + timedelta(minutes=PO_EXPIRY_MIN)
        trade = {
            "instrument": inst,
            "direction": sig["direction"],
            "entry": sig["price"],
            "opened_at": datetime.now(timezone.utc).isoformat(),
            "settle_at": settle_at.isoformat(),
        }
        async with _trades_lock:
            _open_trades.append(trade)
            await save_trades()

        await stats_on_open()
        mark_cooldown(inst)
        await save_cooldowns()

    except Exception as e:
        log.error(f"{inst} signal error: {e}")

async def settle_due_trades():
    """Use Deriv tick at EXACT expiry to decide W/L/D."""
    now = datetime.now(timezone.utc)
    changed = False
    async with _trades_lock:
        rem: List[Dict[str, Any]] = []
        for tr in _open_trades:
            t_settle = datetime.fromisoformat(tr["settle_at"])
            if now >= t_settle:
                try:
                    px = await deriv_tick_at_or_before(tr["instrument"], t_settle)
                    if px is None:
                        # fallback to latest m1 close at/before settle
                        m1 = await deriv_m1(tr["instrument"], 10)
                        m1c = m1[m1["time"] <= pd.to_datetime(t_settle)].tail(1)
                        px = float(m1c.iloc[-1]["c"]) if not m1c.empty else float(tr["entry"])

                    entry = float(tr["entry"])
                    if math.isclose(px, entry, abs_tol=1e-10):
                        res = "D"
                    elif tr["direction"] == "UP":
                        res = "W" if px > entry else "L"
                    else:
                        res = "W" if px < entry else "L"
                    await stats_on_settle(res)
                    changed = True
                except Exception as e:
                    log.error(f"Settlement error {tr['instrument']}: {e}")
                    rem.append(tr)
            else:
                rem.append(tr)
        _open_trades[:] = rem
        if changed:
            await save_trades()

# =========================
# Tallies (daily & weekly)
# =========================
async def send_tally(scope: str):
    now = datetime.now(timezone.utc)
    if scope == "daily":
        key = _day_key(now)
        bucket = _stats.get("daily", {}).get(key, {"signals":0,"W":0,"L":0,"D":0})
        title = f"📊 Daily Tally ({key})"
    else:
        key = _week_key(now)
        bucket = _stats.get("weekly", {}).get(key, {"signals":0,"W":0,"L":0,"D":0})
        title = f"📈 Weekly Tally ({key})"

    s,w,l,d = bucket["signals"], bucket["W"], bucket["L"], bucket["D"]
    wr = (w / max(w+l,1)) * 100.0
    txt = f"{title}\nSignals: {s}\nResults: ✅ {w} | ❌ {l} | ➖ {d}\nWin rate: {wr:.2f}%"
    for cid in [TELEGRAM_CHAT_FREE, TELEGRAM_CHAT_BASIC, TELEGRAM_CHAT_PRO, TELEGRAM_CHAT_VIP]:
        if cid: await tg_send(cid, txt)

async def maybe_send_daily_tally():
    global _last_daily_sent_for
    n = now_local()
    if not is_tally_moment_daily(n): return
    key = n.strftime("%Y-%m-%d")
    if _last_daily_sent_for == key: return
    await send_tally("daily")
    _last_daily_sent_for = key

async def maybe_send_weekly_tally():
    if ENABLE_WEEKLY_TALLY != "1": return
    global _last_weekly_sent_for
    n = now_local()
    if not is_tally_moment_weekly(n): return
    key = _week_key(datetime.now(timezone.utc))
    if _last_weekly_sent_for == key: return
    await send_tally("weekly")
    _last_weekly_sent_for = key

# =========================
# Membership (NOWPayments)
# =========================
def _verify_np_signature(raw: bytes, sig_header: str) -> bool:
    if not NP_IPN_SECRET or not sig_header: return False
    calc = hmac.new(NP_IPN_SECRET.encode(), raw, hashlib.sha512).hexdigest()
    return hmac.compare_digest(calc, sig_header)

def _parse_order_id(order_id: str) -> Tuple[Optional[str], Optional[str], Optional[int]]:
    """
    Expected form: "tg:<USER_ID>|plan:<VIP|PRO|BASIC>|days:<N>"
    Returns (user_id, plan, days)
    """
    try:
        parts = dict(p.split(":",1) for p in order_id.split("|"))
        uid = parts.get("tg")
        plan = parts.get("plan")
        days = int(parts.get("days","0"))
        return uid, plan, days
    except Exception:
        return None, None, None

def _plan_to_chat(plan: str) -> Optional[str]:
    plan = (plan or "").upper()
    if plan == "VIP":   return TELEGRAM_CHAT_VIP or None
    if plan == "PRO":   return TELEGRAM_CHAT_PRO or None
    if plan == "BASIC": return TELEGRAM_CHAT_BASIC or None
    return None

async def grant_membership(user_id: str, plan: str, days: int) -> Dict[str, Any]:
    """
    Create a (revocable) invite link for the target group, store expiry,
    DM user with the link. Requires bot admin in the group(s).
    """
    target_chat = _plan_to_chat(plan)
    if not target_chat or not telegram_app:
        return {"ok": False, "error": "chat-or-bot-missing"}

    expires_at = now_local() + timedelta(days=max(days,1))
    invite: Optional[ChatInviteLink] = None
    try:
        # create invite link with expiry
        invite = await telegram_app.bot.create_chat_invite_link(
            chat_id=target_chat,
            expire_date=int(expires_at.timestamp())
        )
    except Exception as e:
        log.error(f"Create invite failed: {e}")

    async with _members_lock:
        _members[user_id] = {
            "tier": plan.upper(),
            "expires_at": expires_at.astimezone(timezone.utc).isoformat(),
            "chat_id": target_chat,
            "invite_link": invite.invite_link if invite else None,
        }
        await save_members()

    # DM the user (works if user has started the bot at least once)
    try:
        if invite:
            await telegram_app.bot.send_message(
                chat_id=int(user_id),
                text=f"✅ Your {plan.upper()} access is active until {expires_at.strftime('%Y-%m-%d %H:%M %Z')}.\nJoin: {invite.invite_link}"
            )
        else:
            await telegram_app.bot.send_message(
                chat_id=int(user_id),
                text=f"✅ Your {plan.upper()} access is active until {expires_at.strftime('%Y-%m-%d %H:%M %Z')}.\n(Ask admin to share the group link.)"
            )
    except Exception as e:
        log.error(f"DM user failed: {e}")

    return {"ok": True}

async def membership_sweeper():
    """Daily: if expired, try kicking from group (requires admin)."""
    if not telegram_app: return
    now_utc = datetime.now(timezone.utc)
    to_delete = []
    async with _members_lock:
        for uid, m in _members.items():
            exp = datetime.fromisoformat(m.get("expires_at"))
            if now_utc > exp:
                chat_id = m.get("chat_id")
                try:
                    # attempt kick (ban+unban to remove)
                    if chat_id:
                        await telegram_app.bot.ban_chat_member(chat_id=chat_id, user_id=int(uid))
                        await telegram_app.bot.unban_chat_member(chat_id=chat_id, user_id=int(uid))
                except Exception as e:
                    log.warning(f"Kick fail uid={uid}: {e}")
                to_delete.append(uid)
        for uid in to_delete:
            _members.pop(uid, None)
        if to_delete:
            await save_members()

# =========================
# FastAPI
# =========================
app_fastapi = FastAPI()

@app_fastapi.get("/", response_class=PlainTextResponse)
async def root():
    return "Deriv Dual (BLW Trend + MR Chop) — FX live only"

@app_fastapi.get("/healthz", response_class=PlainTextResponse)
async def healthz():
    ok = bool(TELEGRAM_BOT_TOKEN) and bool(DERIV_ENDPOINT) and len(DERIV_PAIRS) > 0
    return "ok" if ok else "not-ok"

@app_fastapi.get("/status")
async def status():
    return {
        "now_utc": datetime.now(timezone.utc).isoformat(),
        "local_time": now_local().isoformat(),
        "fx_live_now": fx_live_now(),
        "pairs": DERIV_PAIRS,
        "cooldown_min": ALERT_COOLDOWN_MIN,
        "expiry_min": PO_EXPIRY_MIN,
    }

# Telegram webhook (optional; use if ENABLE_BOT_POLLING="0")
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

# NOWPayments IPN
@app_fastapi.post("/nowpayments/ipn")
async def nowpayments_ipn(request: Request, x_nowpayments_signature: str = Header(default="")):
    raw = await request.body()
    if not _verify_np_signature(raw, x_nowpayments_signature):
        return PlainTextResponse("bad-signature", status_code=401)
    try:
        payload = await request.json()
    except Exception:
        return PlainTextResponse("bad-json", status_code=400)

    status = (payload.get("payment_status") or "").lower()
    order_id = str(payload.get("order_id", ""))
    uid, plan, days = _parse_order_id(order_id)

    # Accept when finished/confirmed
    if status not in ("finished", "confirmed"):
        return {"ok": True, "note": f"ignored status={status}"}

    if not uid or not plan or not days:
        return {"ok": False, "error": "bad-order-id"}

    res = await grant_membership(uid, plan, days)
    return res

# =========================
# Startup / Shutdown
# =========================
async def run_engine():
    if not fx_live_now(): return
    for inst in DERIV_PAIRS:
        await try_send_signal(inst)

async def housekeeping():
    await settle_due_trades()
    await maybe_send_daily_tally()
    await maybe_send_weekly_tally()

@app_fastapi.on_event("startup")
async def on_startup():
    global telegram_app, scheduler
    await load_state()

    telegram_app = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
    telegram_app.add_handler(CommandHandler("start", start_cmd))
    telegram_app.add_handler(CommandHandler("ping",  ping_cmd))
    telegram_app.add_handler(CommandHandler("limits", limits_cmd))
    await telegram_app.initialize()
    await telegram_app.start()

    if ENABLE_BOT_POLLING == "0":
        await telegram_app.bot.set_webhook(url=f"{BASE_URL}{WEBHOOK_PATH}", secret_token=WEBHOOK_SECRET, drop_pending_updates=True)
    else:
        await telegram_app.bot.delete_webhook(drop_pending_updates=True)

    scheduler = AsyncIOScheduler(timezone=LOCAL_TZ)
    scheduler.add_job(run_engine,   CronTrigger(second="5"))         # every minute at :05s
    scheduler.add_job(housekeeping, CronTrigger(second="*/30"))      # settle + tallies check
    scheduler.add_job(membership_sweeper, CronTrigger(hour="0", minute="5"))  # daily expiry watcher
    # backup daily & weekly tallies
    scheduler.add_job(maybe_send_daily_tally,  CronTrigger(hour=LIVE_END.hour, minute=LIVE_END.minute))
    scheduler.add_job(maybe_send_weekly_tally, CronTrigger(day_of_week="sun", hour=LIVE_END.hour, minute=(LIVE_END.minute + 5) % 60))
    scheduler.start()
    log.info("⏰ Scheduler started")

@app_fastapi.on_event("shutdown")
async def on_shutdown():
    if scheduler:
        scheduler.shutdown(wait=False)
    if telegram_app:
        await telegram_app.stop()

# Uvicorn entrypoint
app = app_fastapi
