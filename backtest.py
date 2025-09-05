#!/usr/bin/env python3
# backtest.py — Terminal backtester for Pocket Option Signals
# - Uses Deriv WS candles (1m) with optional authorize
# - TREND → CHOP → BASE strategies (SMA/BB/ATR), same defaults as live
# - Trinidad & Tobago trading window: Mon–Fri 08:00–17:00 (America/Port_of_Spain)
# - Simulates 5-minute expiry (configurable)
# - Prints summary and (optional) writes trades CSV

import os, sys, ssl, json, asyncio, csv, math, time, argparse, logging
from collections import deque, defaultdict
from datetime import datetime, timezone, time as dtime
from zoneinfo import ZoneInfo

# Optional .env
try:
    from dotenv import load_dotenv
except Exception:
    def load_dotenv(*args, **kwargs): pass

try:
    import websockets
except Exception:
    print("ERROR: websockets is required. Install: pip install websockets python-dotenv", file=sys.stderr)
    raise

# -------------------- ENV --------------------
load_dotenv()

APP_NAME = "Pocket Option Signals (Backtest)"

DERIV_APP_ID    = os.getenv("DERIV_APP_ID", "99185")
DERIV_API_TOKEN = os.getenv("DERIV_API_TOKEN", "")  # optional

# Strategy defaults (match live app)
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

# Expiry minutes (default 5)
EXPIRY_MIN = int(os.getenv("EXPIRY_MIN", "5"))

# Trinidad trading gate (local)
TRADING_TZ = os.getenv("TRADING_TZ", "America/Port_of_Spain")  # UTC-4, no DST
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

# -------------------- Helpers --------------------
def to_dt(epoch):
    try:
        return datetime.fromtimestamp(float(epoch), tz=UTC)
    except Exception:
        return None

def expand_day_tokens(tokens):
    ref = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
    days = set()
    for token in tokens:
        if "-" in token:
            a, b = token.split("-", 1)
            ia, ib = ref.index(a), ref.index(b)
            if ia <= ib: rng = ref[ia:ib+1]
            else: rng = ref[ia:]+ref[:ib+1]
            days.update(rng)
        else:
            days.add(token)
    return days

def weekday_name_idx(i): return ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"][i]

def parse_hhmm_local(s: str, tz: ZoneInfo):
    hh, mm = s.split(":")
    return dtime(int(hh), int(mm), tzinfo=tz)

def market_is_open_trinidad(ts_utc: datetime) -> bool:
    tz = ZoneInfo(TRADING_TZ)
    local_dt = ts_utc.astimezone(tz)
    wd = weekday_name_idx(local_dt.weekday())
    if wd not in expand_day_tokens(LOCAL_TRADING_DAYS): return False
    start_t = parse_hhmm_local(LOCAL_START_LOCAL, tz)
    end_t   = parse_hhmm_local(LOCAL_END_LOCAL, tz)
    t = local_dt.timetz()
    return start_t <= t <= end_t

def pretty_symbol(sym: str) -> str:
    if sym.startswith("frx") and len(sym) == 8:
        p = sym[3:]; return f"{p[:3]}/{p[3:]}"
    if len(sym) >= 6 and sym[-3:].isalpha() and sym[:3].isalpha():
        return f"{sym[:3]}/{sym[3:]}"
    return sym

def safe_float(x, d=None):
    try: return float(x)
    except Exception: return d

# -------------------- Rolling indicators --------------------
class RollingState:
    def __init__(self):
        self.fast = deque(maxlen=SMA_FAST); self.sum_fast = 0.0
        self.slow = deque(maxlen=SMA_SLOW); self.sum_slow = 0.0
        self.prev_fast = None; self.prev_slow = None
        self.closes = deque(maxlen=BB_WINDOW); self.sum_close = 0.0; self.sumsq_close = 0.0
        self.trs = deque(maxlen=ATR_WINDOW); self.sum_tr = 0.0
        self.prev_close = None
    def update_sma(self, close):
        if len(self.fast) == self.fast.maxlen: self.sum_fast -= self.fast[0]
        self.fast.append(close); self.sum_fast += close
        fast = self.sum_fast / len(self.fast)
        if len(self.slow) == self.slow.maxlen: self.sum_slow -= self.slow[0]
        self.slow.append(close); self.sum_slow += close
        slow = self.sum_slow / len(self.slow)
        prev_fast, prev_slow = self.prev_fast, self.prev_slow
        self.prev_fast, self.prev_slow = fast, slow
        return prev_fast, prev_slow, fast, slow
    def update_bb(self, close):
        if len(self.closes) == self.closes.maxlen:
            old = self.closes[0]; self.sum_close -= old; self.sumsq_close -= old*old
        self.closes.append(close); self.sum_close += close; self.sumsq_close += close*close
        n = len(self.closes)
        if n == 0: return None, None
        mean = self.sum_close / n
        var = max(self.sumsq_close / n - mean*mean, 0.0)
        return mean, math.sqrt(var)
    def update_atr(self, high, low, close):
        pc = self.prev_close
        tr = abs(high - low) if pc is None else max(abs(high-low), abs(high-pc), abs(low-pc))
        if len(self.trs) == self.trs.maxlen: self.sum_tr -= self.trs[0]
        self.trs.append(tr); self.sum_tr += tr; self.prev_close = close
        n = len(self.trs); return (self.sum_tr / n) if n>0 else None

# -------------------- Strategy (TREND → CHOP → BASE) --------------------
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
            if crossed_up: return "CALL", "TREND"
            if crossed_dn: return "PUT",  "TREND"

    # CHOP
    if ENABLE_CHOPPY and mean is not None and std is not None and state.prev_slow is not None and atr is not None:
        slope = abs(state.prev_slow - state.slow[-1] if len(state.slow) else 0.0)
        if slope < (SLOPE_ATR_MULT * atr):
            upper = mean + BB_STD_MULT * std
            lower = mean - BB_STD_MULT * std
            if close_price >= upper: return "PUT",  "CHOP"
            if close_price <= lower: return "CALL", "CHOP"

    # BASE
    if len(state.slow) >= SMA_SLOW:
        crossed_up = state.prev_fast is not None and state.prev_slow is not None and state.prev_fast <= state.prev_slow and state.fast[-1] > state.slow[-1]
        crossed_dn = state.prev_fast is not None and state.prev_slow is not None and state.prev_fast >= state.prev_slow and state.fast[-1] < state.slow[-1]
        if crossed_up and abs(close_price - state.slow[-1]) >= PULLBACK_MIN: return "CALL", "BASE"
        if crossed_dn and abs(close_price - state.slow[-1]) >= PULLBACK_MIN: return "PUT",  "BASE"

    return None, None

# -------------------- Deriv WS history fetch --------------------
async def fetch_candles(symbol: str, start_epoch: int, end_epoch: int, app_id: str, token: str = ""):
    """
    Returns list of candles dicts: [{"open","high","low","close","epoch"}, ...] at 1m.
    """
    uri = f"wss://ws.derivws.com/websockets/v3?app_id={app_id}"
    ssl_ctx = ssl.create_default_context()
    out = []

    async with websockets.connect(uri, ssl=ssl_ctx, ping_interval=20, ping_timeout=20, max_queue=1024) as ws:
        if token:
            await ws.send(json.dumps({"authorize": token}))
            auth = json.loads(await ws.recv())
            if "error" in auth:
                LOG.error(f"Authorize error: {auth['error']}")
            else:
                LOG.info("Authorized.")

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

            if "candles" in resp:
                out.extend(resp["candles"])
                LOG.info(f"Fetched {len(resp['candles'])} candles {datetime.fromtimestamp(s, tz=UTC)} → {datetime.fromtimestamp(e, tz=UTC)}")
            elif "history" in resp and "prices" in resp["history"]:
                prices = resp["history"]["prices"]; times = resp["history"]["times"]
                chunk = [{"open": p, "high": p, "low": p, "close": p, "epoch": t} for p, t in zip(prices, times)]
                out.extend(chunk)
                LOG.info(f"Fetched {len(chunk)} closes {datetime.fromtimestamp(s, tz=UTC)} → {datetime.fromtimestamp(e, tz=UTC)}")
            else:
                LOG.warning(f"No data {s}→{e}")

            s = e
            await asyncio.sleep(0.03)

    # Dedup/sort
    seen = set(); cleaned = []
    for c in out:
        ep = int(c["epoch"])
        if ep in seen: continue
        seen.add(ep); cleaned.append(c)
    cleaned.sort(key=lambda x: int(x["epoch"]))
    return cleaned

# -------------------- Backtest core --------------------
def eval_trade(direction: str, entry_close: float, expiry_close: float, tie_is_win: bool = False) -> int:
    if direction == "CALL":
        if expiry_close > entry_close: return +1
        if expiry_close == entry_close: return +1 if tie_is_win else 0
        return -1
    else:
        if expiry_close < entry_close: return +1
        if expiry_close == entry_close: return +1 if tie_is_win else 0
        return -1

async def run_backtest(symbol: str, start_iso: str, end_iso: str, csv_path: str = "", tie_is_win=False):
    # Interpret dates as UTC timestamps
    start_dt = datetime.fromisoformat(start_iso)
    if start_dt.tzinfo is None: start_dt = start_dt.replace(tzinfo=UTC)
    end_dt = datetime.fromisoformat(end_iso)
    if end_dt.tzinfo is None: end_dt = end_dt.replace(hour=23, minute=59, second=59, tzinfo=UTC)
    start_epoch, end_epoch = int(start_dt.timestamp()), int(end_dt.timestamp())

    LOG.info(f"Loading {symbol} {start_iso} → {end_iso} (UTC)")
    candles = await fetch_candles(symbol, start_epoch, end_epoch, DERIV_APP_ID, DERIV_API_TOKEN)
    LOG.info(f"Total candles: {len(candles)}")

    state = RollingState()
    closes = [c for c in candles if c.get("close") is not None]
    trades = []
    wins = losses = ties = 0
    per_strategy = defaultdict(lambda: {"wins":0,"losses":0,"ties":0,"count":0})

    for i, c in enumerate(closes):
        when = to_dt(c["epoch"])
        if not when:
            continue

        close = safe_float(c["close"])
        high = safe_float(c.get("high"), close)
        low  = safe_float(c.get("low"),  close)

        # Always update indicators so state stays consistent across closed periods
        # but only take signals during local open window
        if market_is_open_trinidad(when):
            direction, strat = decide_signal(state, close, high, low)
            if direction:
                expiry_idx = min(i + EXPIRY_MIN, len(closes) - 1)
                expiry_close = safe_float(closes[expiry_idx]["close"], close)
                r = eval_trade(direction, close, expiry_close, tie_is_win=tie_is_win)
                if r > 0: wins += 1; per_strategy[strat]["wins"] += 1
                elif r < 0: losses += 1; per_strategy[strat]["losses"] += 1
                else: ties += 1; per_strategy[strat]["ties"] += 1
                per_strategy[strat]["count"] += 1
                trades.append({
                    "time_utc": when.strftime("%Y-%m-%d %H:%M:%S"),
                    "symbol": symbol,
                    "symbol_pretty": pretty_symbol(symbol),
                    "strategy": strat,
                    "direction": direction,
                    "entry": f"{close:.6f}",
                    "expiry_min": EXPIRY_MIN,
                    "expiry_close": f"{expiry_close:.6f}",
                    "result": "WIN" if r > 0 else ("LOSS" if r < 0 else "TIE")
                })
        else:
            # still evolve indicators
            state.update_sma(close); state.update_bb(close); state.update_atr(high, low, close)

    total = wins + losses + ties
    winrate = (wins / (wins + losses)) * 100 if (wins + losses) > 0 else 0.0

    print("\n================= BACKTEST SUMMARY =================")
    print(f"Symbol: {symbol} ({pretty_symbol(symbol)})")
    print(f"Period: {start_iso} → {end_iso} (UTC)")
    print(f"Local Gate: {LOCAL_TRADING_DAYS} {LOCAL_START_LOCAL}–{LOCAL_END_LOCAL} ({TRADING_TZ})")
    print(f"Granularity: {CANDLE_GRANULARITY}s  |  Expiry: {EXPIRY_MIN}m  |  Tie= {'Win' if tie_is_win else 'No-Count'}")
    print("----------------------------------------------------")
    print(f"Signals: {total}  |  Wins: {wins}  Losses: {losses}  Ties: {ties}  |  Winrate: {winrate:.2f}%")
    print("By Strategy:")
    for strat in ("TREND","CHOP","BASE"):
        s = per_strategy[strat]
        w,l,t,n = s["wins"], s["losses"], s["ties"], s["count"]
        wr = (w/(w+l)*100) if (w+l)>0 else 0.0
        print(f"  {strat:<5}  count={n:<4}  wins={w:<4}  losses={l:<4}  ties={t:<4}  wr={wr:5.2f}%")
    print("====================================================\n")

    if csv_path:
        fields = ["time_utc","symbol","symbol_pretty","strategy","direction","entry","expiry_min","expiry_close","result"]
        with open(csv_path, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=fields); w.writeheader(); w.writerows(trades)
        print(f"Saved trades CSV → {csv_path}")

# -------------------- CLI --------------------
def main():
    p = argparse.ArgumentParser(description="Backtest Pocket Option Signals strategies against Deriv candles.")
    p.add_argument("--symbol", default=os.getenv("BT_SYMBOL", "frxGBPUSD"), help="e.g. frxGBPUSD, frxEURUSD")
    p.add_argument("--start",  required=True, help="Start date/time (UTC) YYYY-MM-DD or YYYY-MM-DDTHH:MM")
    p.add_argument("--end",    required=True, help="End date/time (UTC) YYYY-MM-DD or YYYY-MM-DDTHH:MM")
    p.add_argument("--csv",    default="", help="Optional path to write trades CSV")
    p.add_argument("--expiry", type=int, default=int(os.getenv("EXPIRY_MIN", EXPIRY_MIN)), help="Expiry minutes (default 5)")
    p.add_argument("--gran",   type=int, default=int(os.getenv("CANDLE_GRANULARITY", CANDLE_GRANULARITY)), help="Granularity seconds (default 60)")
    p.add_argument("--tie-win", action="store_true", help="Count exact tie as WIN (default is no-count)")
    args = p.parse_args()

    # override globals from CLI if provided
    global EXPIRY_MIN, CANDLE_GRANULARITY
    EXPIRY_MIN = args.expiry
    CANDLE_GRANULARITY = args.gran

    print(f"{APP_NAME} — Deriv app_id={DERIV_APP_ID}  token={'set' if bool(DERIV_API_TOKEN) else 'none'}")
    asyncio.run(run_backtest(args.symbol, args.start, args.end, csv_path=args.csv, tie_is_win=args.tie_win))

if __name__ == "__main__":
    main()
