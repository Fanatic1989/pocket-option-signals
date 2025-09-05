#!/usr/bin/env python3
# backtest.py — Terminal backtester for Pocket Option Signals strategies
# - Matches live logic: TREND → CHOP → BASE
# - Market-hours gate: Trinidad & Tobago (America/Port_of_Spain) Mon–Fri 08:00–17:00
# - Uses Deriv WebSocket history (1m candles) + optional authorize
# - 5-minute expiry by default (configurable)
# - Outputs summary + optional CSV with all simulated trades

import os, sys, ssl, json, asyncio, csv, io, math, time, html, argparse, logging
from collections import deque, defaultdict
from datetime import datetime, timezone, timedelta, time as dtime
from zoneinfo import ZoneInfo

try:
    from dotenv import load_dotenv
except Exception:
    def load_dotenv(*args, **kwargs): pass

try:
    import websockets
except Exception as e:
    print("ERROR: websockets package is required. Install with: pip install websockets python-dotenv", file=sys.stderr)
    raise

# --------------------
# ENV / Defaults
# --------------------
load_dotenv()

APP_NAME = "Pocket Option Signals (Backtest)"

DERIV_APP_ID    = os.getenv("DERIV_APP_ID", "99185")
DERIV_API_TOKEN = os.getenv("DERIV_API_TOKEN", "")  # optional

# Strategy params (match live defaults)
CANDLE_GRANULARITY = int(os.getenv("CANDLE_GRANULARITY", "60"))  # seconds
SMA_FAST = int(os.getenv("SMA_FAST", "9"))
SMA_SLOW = int(os.getenv("SMA_SLOW", "21"))
PULLBACK_MIN = float(os.getenv("PULLBACK_MIN", "0.00010"))
ATR_WINDOW = int(os.getenv("ATR_WINDOW", "14"))
BB_WINDOW = int(os.getenv("BB_WINDOW", "20"))
BB_STD_MULT = float(os.getenv("BB_STD_MULT", "1.0"))
SLOPE_ATR_MULT = float(os.getenv("SLOPE_ATR_MULT", "0.20"))

ENABLE_TRENDING = os.getenv("ENABLE_TRENDING", "1") == "1"
ENABLE_CHOPPY   = os.getenv("ENABLE_CHOPPY", "1") == "1"

# Message/expiry
EXPIRY_MIN = int(os.getenv("EXPIRY_MIN", "5"))

# Trinidad trading gate (local)
TRADING_TZ = os.getenv("TRADING_TZ", "America/Port_of_Spain")
LOCAL_TRADING_DAYS = [d.strip() for d in os.getenv("LOCAL_TRADING_DAYS", "Mon-Fri").split(",") if d.strip()]
LOCAL_START_LOCAL = os.getenv("LOCAL_TRADING_START", "08:00")
LOCAL_END_LOCAL   = os.getenv("LOCAL_TRADING_END",   "17:00")

UTC = timezone.utc
LOG = logging.getLogger("backtest")
logging.basicConfig(
    level=getattr(logging, os.getenv("LOG_LEVEL", "INFO").upper(), logging.INFO),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    stream=sys.stdout,
)

# --------------------
# Helpers
# --------------------
def to_dt(epoch):
    try:
        return datetime.fromtimestamp(float(epoch), tz=UTC)
    except Exception:
        return None

def weekday_name_idx(idx: int):
    return ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"][idx]

def expand_day_tokens(tokens):
    ref = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
    days = set()
    for token in tokens:
        if "-" in token:
            a, b = token.split("-", 1)
            ia, ib = ref.index(a), ref.index(b)
            if ia <= ib:
                rng = ref[ia:ib+1]
            else:
                rng = ref[ia:]+ref[:ib+1]
            days.update(rng)
        else:
            days.add(token)
    return days

def parse_hhmm_local(s: str, tz: ZoneInfo):
    hh, mm = s.split(":")
    return dtime(int(hh), int(mm), tzinfo=tz)

def market_is_open_trinidad(ts_utc: datetime) -> bool:
    tz = ZoneInfo(TRADING_TZ)
    local_dt = ts_utc.astimezone(tz)
    wd_name = weekday_name_idx(local_dt.weekday())
    allowed_days = expand_day_tokens(LOCAL_TRADING_DAYS)
    if wd_name not in allowed_days:
        return False
    start_t = parse_hhmm_local(LOCAL_START_LOCAL, tz)
    end_t   = parse_hhmm_local(LOCAL_END_LOCAL, tz)
    t = local_dt.timetz()
    return start_t <= t <= end_t

def pretty_symbol(sym: str) -> str:
    if sym.startswith("frx") and len(sym) == 8:
        p = sym[3:]
        return f"{p[:3]}/{p[3:]}"
    if len(sym) >= 6 and sym[-3:].isalpha() and sym[:3].isalpha():
        return f"{sym[:3]}/{sym[3:]}"
    return sym

def safe_float(x, default=None):
    try:
        return float(x)
    except Exception:
        return default

# --------------------
# Rolling indicators
# --------------------
class RollingState:
    def __init__(self):
        self.fast = deque(maxlen=SMA_FAST); self.sum_fast = 0.0
        self.slow = deque(maxlen=SMA_SLOW); self.sum_slow = 0.0
        self.prev_fast = None; self.prev_slow = None
        self.closes = deque(maxlen=BB_WINDOW); self.sum_close = 0.0; self.sumsq_close = 0.0
        self.trs = deque(maxlen=ATR_WINDOW); self.sum_tr = 0.0
        self.prev_close = None

    def update_sma(self, close_price: float):
        if len(self.fast) == self.fast.maxlen:
            self.sum_fast -= self.fast[0]
        self.fast.append(close_price); self.sum_fast += close_price
        fast = self.sum_fast / len(self.fast)

        if len(self.slow) == self.slow.maxlen:
            self.sum_slow -= self.slow[0]
        self.slow.append(close_price); self.sum_slow += close_price
        slow = self.sum_slow / len(self.slow)

        prev_fast, prev_slow = self.prev_fast, self.prev_slow
        self.prev_fast, self.prev_slow = fast, slow
        return prev_fast, prev_slow, fast, slow

    def update_bb(self, close_price: float):
        if len(self.closes) == self.closes.maxlen:
            old = self.closes[0]; self.sum_close -= old; self.sumsq_close -= old*old
        self.closes.append(close_price); self.sum_close += close_price; self.sumsq_close += close_price*close_price
        n = len(self.closes)
        if n == 0: return None, None
        mean = self.sum_close / n
        var = max(self.sumsq_close / n - mean * mean, 0.0)
        std = math.sqrt(var)
        return mean, std

    def update_atr(self, high, low, close):
        pc = self.prev_close
        tr = abs(high - low) if pc is None else max(abs(high-low), abs(high-pc), abs(low-pc))
        if len(self.trs) == self.trs.maxlen:
            self.sum_tr -= self.trs[0]
        self.trs.append(tr); self.sum_tr += tr; self.prev_close = close
        n = len(self.trs)
        return (self.sum_tr / n) if n > 0 else None

# --------------------
# Signal logic (matches live cascade: TREND → CHOP → BASE)
# --------------------
def decide_signal(state: RollingState, close_price, high, low):
    prev_fast, prev_slow, fast, slow = state.update_sma(close_price)
    mean, std = state.update_bb(close_price)
    atr = state.update_atr(high, low, close_price)

    # TREND
    if ENABLE_TRENDING and len(state.slow) >= SMA_SLOW and atr is not None:
        slope = abs(slow - (prev_slow if prev_slow is not None else slow))
        slope_thresh = SLOPE_ATR_MULT * atr
        trending = slope >= slope_thresh and abs(close_price - slow) >= (0.25 * atr if atr and atr > 0 else PULLBACK_MIN)
        if trending:
            crossed_up = prev_fast is not None and prev_slow is not None and prev_fast <= prev_slow and fast > slow
            crossed_dn = prev_fast is not None and prev_slow is not None and prev_fast >= prev_slow and fast < slow
            if crossed_up:
                return "CALL", "TREND"
            if crossed_dn:
                return "PUT", "TREND"

    # CHOP
    if ENABLE_CHOPPY and mean is not None and std is not None and state.prev_slow is not None and atr is not None:
        slope = abs(state.prev_slow - state.slow[-1] if len(state.slow) else 0.0)
        slope_thresh = SLOPE_ATR_MULT * atr
        if slope < slope_thresh:
            upper = mean + BB_STD_MULT * std
            lower = mean - BB_STD_MULT * std
            if close_price >= upper:
                return "PUT", "CHOP"
            if close_price <= lower:
                return "CALL", "CHOP"

    # BASE
    if len(state.slow) >= SMA_SLOW:
        crossed_up = state.prev_fast is not None and state.prev_slow is not None and state.prev_fast <= state.prev_slow and state.fast[-1] > state.slow[-1]
        crossed_dn = state.prev_fast is not None and state.prev_slow is not None and state.prev_fast >= state.prev_slow and state.fast[-1] < state.slow[-1]
        if crossed_up and abs(close_price - state.slow[-1]) >= PULLBACK_MIN:
            return "CALL", "BASE"
        if crossed_dn and abs(close_price - state.slow[-1]) >= PULLBACK_MIN:
            return "PUT", "BASE"

    return None, None

# --------------------
# Deriv WS history fetch
# --------------------
async def fetch_candles(symbol: str, start_epoch: int, end_epoch: int, app_id: str, token: str = ""):
    """
    Returns list of candles dicts: [{"open","high","low","close","epoch"}, ...] 1m granularity.
    """
    uri = f"wss://ws.derivws.com/websockets/v3?app_id={app_id}"
    ssl_ctx = ssl.create_default_context()
    candles_all = []

    async with websockets.connect(uri, ssl=ssl_ctx, ping_interval=20, ping_timeout=20, max_queue=1024) as ws:
        # Optional authorize
        if token:
            await ws.send(json.dumps({"authorize": token}))
            auth = json.loads(await ws.recv())
            if "error" in auth:
                LOG.error(f"Authorize error: {auth['error']}")
            else:
                LOG.info("Authorized with Deriv token")

        # We’ll request the whole range in chunks of days to be safe
        CHUNK_DAYS = 7
        s = start_epoch
        while s < end_epoch:
            e = min(s + CHUNK_DAYS * 86400, end_epoch)

            req = {
                "ticks_history": symbol,
                "style": "candles",
                "granularity": CANDLE_GRANULARITY,
                "start": s,
                "end": e
            }
            await ws.send(json.dumps(req))
            resp = json.loads(await ws.recv())
            if "error" in resp:
                LOG.error(f"history error: {resp['error']}")
                break

            # Two possible shapes: candles[] or history with prices/times
            chunk = []
            if "candles" in resp:
                chunk = resp["candles"]
            elif "history" in resp and "prices" in resp["history"]:
                # reconstruct candles with close-only (fallback)
                prices = resp["history"]["prices"]
                times = resp["history"]["times"]
                for p, t in zip(prices, times):
                    # no high/low/open provided in this shape
                    chunk.append({"open": p, "high": p, "low": p, "close": p, "epoch": t})

            if chunk:
                candles_all.extend(chunk)
                LOG.info(f"Fetched {len(chunk)} candles {symbol} {datetime.fromtimestamp(s, tz=UTC)} → {datetime.fromtimestamp(e, tz=UTC)}")
            else:
                LOG.warning(f"No candles returned for {symbol} in chunk {s}→{e}")

            # advance
            s = e
            await asyncio.sleep(0.05)

    # Deduplicate/sort by epoch
    seen = set(); out = []
    for c in candles_all:
        ep = int(c["epoch"])
        if ep in seen: continue
        seen.add(ep); out.append(c)
    out.sort(key=lambda x: int(x["epoch"]))
    return out

# --------------------
# Backtest runner
# --------------------
def eval_trade(direction: str, entry_close: float, expiry_close: float, tie_is_win: bool = False) -> int:
    """
    Returns +1 win, 0 tie (if tie_is_win False), -1 loss.
    """
    if direction == "CALL":
        if expiry_close > entry_close: return +1
        if expiry_close == entry_close: return +1 if tie_is_win else 0
        return -1
    else:
        if expiry_close < entry_close: return +1
        if expiry_close == entry_close: return +1 if tie_is_win else 0
        return -1

async def run_backtest(symbol: str, start: str, end: str, csv_path: str = "", tie_is_win=False):
    start_epoch = int(datetime.fromisoformat(start).replace(tzinfo=UTC).timestamp())
    # include end-of-day by adding 23:59:59 if only date passed
    try:
        end_dt = datetime.fromisoformat(end)
        if end_dt.tzinfo is None:
            end_dt = end_dt.replace(hour=23, minute=59, second=59, tzinfo=UTC)
        end_epoch = int(end_dt.timestamp())
    except Exception:
        end_epoch = int(datetime.fromisoformat(end).replace(tzinfo=UTC).timestamp())

    LOG.info(f"Loading candles {symbol} {start} → {end} (UTC)")
    candles = await fetch_candles(symbol, start_epoch, end_epoch, DERIV_APP_ID, DERIV_API_TOKEN)
    LOG.info(f"Total candles loaded: {len(candles)}")

    state = RollingState()
    trades = []
    wins = losses = ties = 0
    per_strategy = defaultdict(lambda: {"wins":0,"losses":0,"ties":0,"count":0})

    # Iterate over candles in order; one decision per *closed* candle
    closes = [c for c in candles if c.get("close") is not None]
    for i in range(len(closes)):
        c = closes[i]
        when = to_dt(c["epoch"])
        if not when:
            continue

        # Enforce Trinidad window
        if not market_is_open_trinidad(when):
            # still update indicators so state progresses across closed periods
            close = safe_float(c["close"])
            high = safe_float(c.get("high"), close)
            low  = safe_float(c.get("low"),  close)
            state.update_sma(close)
            state.update_bb(close)
            state.update_atr(high, low, close)
            continue

        close = safe_float(c["close"])
        high = safe_float(c.get("high"), close)
        low  = safe_float(c.get("low"),  close)

        direction, strat = decide_signal(state, close, high, low)

        if direction:
            # expiry N candles ahead
            expiry_idx = min(i + EXPIRY_MIN, len(closes) - 1)
            expiry_close = safe_float(closes[expiry_idx]["close"], close)
            result = eval_trade(direction, close, expiry_close, tie_is_win=tie_is_win)
            if result > 0: wins += 1; per_strategy[strat]["wins"] += 1
            elif result < 0: losses += 1; per_strategy[strat]["losses"] += 1
            else: ties += 1; per_strategy[strat]["ties"] += 1

            per_strategy[strat]["count"] += 1
            trades.append({
                "time_utc": when.strftime("%Y-%m-%d %H:%M:%S"),
                "symbol": symbol,
                "symbol_pretty": pretty_symbol(symbol),
                "direction": direction,
                "strategy": strat,
                "entry": f"{close:.6f}",
                "expiry_min": EXPIRY_MIN,
                "expiry_close": f"{expiry_close:.6f}",
                "result": "WIN" if result > 0 else ("LOSS" if result < 0 else "TIE")
            })

    total = wins + losses + ties
    winrate = (wins / (wins + losses)) * 100 if (wins + losses) > 0 else 0.0

    # Print summary
    print("\n================= BACKTEST SUMMARY =================")
    print(f"Symbol: {symbol} ({pretty_symbol(symbol)})")
    print(f"Period: {start} → {end} (UTC)")
    print(f"Local Gate: {LOCAL_TRADING_DAYS} {LOCAL_START_LOCAL}–{LOCAL_END_LOCAL} ({TRADING_TZ})")
    print(f"Granularity: {CANDLE_GRANULARITY}s  |  Expiry: {EXPIRY_MIN}m  |  Tie= {'Win' if tie_is_win else 'No-Count'}")
    print("----------------------------------------------------")
    print(f"Signals: {total}  |  Wins: {wins}  Losses: {losses}  Ties: {ties}  |  Winrate: {winrate:.2f}%")
    print("By Strategy:")
    for strat in ("TREND","CHOP","BASE"):
        s = per_strategy[strat]
        w, l, t, n = s["wins"], s["losses"], s["ties"], s["count"]
        wr = (w/(w+l)*100) if (w+l)>0 else 0.0
        print(f"  {strat:<5}  count={n:<4}  wins={w:<4}  losses={l:<4}  ties={t:<4}  wr={wr:5.2f}%")
    print("====================================================\n")

    # CSV
    if csv_path:
        fieldnames = ["time_utc","symbol","symbol_pretty","strategy","direction","entry","expiry_min","expiry_close","result"]
        with open(csv_path, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(trades)
        print(f"Saved trades CSV → {csv_path}")

# --------------------
# CLI
# --------------------
def main():
    p = argparse.ArgumentParser(description="Backtest Pocket Option Signals strategies against Deriv candles.")
    p.add_argument("--symbol", default=os.getenv("BT_SYMBOL", "frxGBPUSD"), help="e.g. frxGBPUSD, frxEURUSD")
    p.add_argument("--start",  required=True, help="Start date (UTC) YYYY-MM-DD or YYYY-MM-DDTHH:MM")
    p.add_argument("--end",    required=True, help="End date (UTC) YYYY-MM-DD or YYYY-MM-DDTHH:MM")
    p.add_argument("--csv",    default="", help="Optional path to write trades CSV")
    p.add_argument("--expiry", type=int, default=int(os.getenv("EXPIRY_MIN", EXPIRY_MIN)), help="Expiry minutes (default from env)")
    p.add_argument("--gran",   type=int, default=int(os.getenv("CANDLE_GRANULARITY", CANDLE_GRANULARITY)), help="Granularity seconds (default 60)")
    p.add_argument("--tie-win", action="store_true", help="Count exact tie as WIN (default is no-count)")
    args = p.parse_args()

    # allow CLI to override global expiry/granularity
    global EXPIRY_MIN, CANDLE_GRANULARITY
    EXPIRY_MIN = args.expiry
    CANDLE_GRANULARITY = args.gran

    print(f"{APP_NAME} — Deriv app_id={DERIV_APP_ID}  token={'set' if bool(DERIV_API_TOKEN) else 'none'}")
    asyncio.run(run_backtest(args.symbol, args.start, args.end, csv_path=args.csv, tie_is_win=args.tie_win))

if __name__ == "__main__":
    main()
