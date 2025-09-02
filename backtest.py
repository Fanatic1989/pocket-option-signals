#!/usr/bin/env python3
"""
Pocket Option Signals Backtest
Runs your MTF Confluence strategy over the past month of OANDA data.
"""

import os
import asyncio
import pandas as pd
import pandas_ta as ta
from datetime import datetime, timedelta, timezone
import httpx
from zoneinfo import ZoneInfo

# --- ENV CONFIG ---
OANDA_API_KEY = os.getenv("OANDA_API_KEY", "")
OANDA_ACCOUNT_ID = os.getenv("OANDA_ACCOUNT_ID", "")
OANDA_ENV = os.getenv("OANDA_ENV", "practice").lower()
OANDA_INSTRUMENTS = [s.strip() for s in os.getenv("OANDA_INSTRUMENTS", "").split(",") if s.strip()]
SESSION_TZ = ZoneInfo(os.getenv("SESSION_TZ", "America/Port_of_Spain"))
SESSION_START = datetime.strptime("08:00", "%H:%M").time()
SESSION_END = datetime.strptime("16:00", "%H:%M").time()

EMA_FAST = int(os.getenv("EMA_FAST", "9"))
EMA_SLOW = int(os.getenv("EMA_SLOW", "21"))
RSI_MIN_BUY = float(os.getenv("RSI_MIN_BUY", "50"))
RSI_MAX_SELL = float(os.getenv("RSI_MAX_SELL", "50"))

# --- OANDA HELPERS ---
def oanda_base_url():
    return "https://api-fxtrade.oanda.com" if OANDA_ENV == "live" else "https://api-fxpractice.oanda.com"

def auth_headers():
    return {"Authorization": f"Bearer {OANDA_API_KEY}"}

async def fetch_candles(pair, granularity="M1", days=30):
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=days)
    url = f"{oanda_base_url()}/v3/instruments/{pair}/candles"
    params = {
        "granularity": granularity,
        "price": "M",
        "from": start.isoformat(),
        "to": end.isoformat(),
    }
    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.get(url, params=params, headers=auth_headers())
        r.raise_for_status()
        data = r.json()
    rows = []
    for c in data.get("candles", []):
        if not c.get("complete"):  # skip incomplete candles
            continue
        mid = c.get("mid", {})
        rows.append({
            "time": pd.to_datetime(c["time"]),
            "o": float(mid.get("o", "nan")),
            "h": float(mid.get("h", "nan")),
            "l": float(mid.get("l", "nan")),
            "c": float(mid.get("c", "nan")),
        })
    return pd.DataFrame(rows)

# --- STRATEGY (SAME AS APP) ---
def compute_signal(df, idx):
    """Baseline EMA9/EMA21 cross + RSI filter"""
    close = df["c"]
    ema_fast = ta.ema(close, length=EMA_FAST)
    ema_slow = ta.ema(close, length=EMA_SLOW)
    rsi14 = ta.rsi(close, length=14)

    cur, prev = df.iloc[idx], df.iloc[idx-1]
    ef, es, rsi = ema_fast.iloc[idx], ema_slow.iloc[idx], rsi14.iloc[idx]
    pf, ps, rsi_p = ema_fast.iloc[idx-1], ema_slow.iloc[idx-1], rsi14.iloc[idx-1]

    if pd.isna([ef, es, rsi]).any():
        return None

    if pf <= ps and ef > es and rsi >= RSI_MIN_BUY:
        return "CALL"
    if pf >= ps and ef < es and rsi <= RSI_MAX_SELL:
        return "PUT"
    return None

def in_session(ts):
    local = ts.astimezone(SESSION_TZ)
    return SESSION_START <= local.time() <= SESSION_END and local.weekday() < 5

# --- BACKTEST ---
async def backtest_pair(pair):
    df = await fetch_candles(pair, "M1", days=30)
    if df.empty: return pair, {"signals": 0, "W": 0, "L": 0, "D": 0}

    signals, wins, losses, draws = 0, 0, 0, 0

    # Simple win/loss simulation: if price moved in predicted direction in next 5 candles
    for i in range(21, len(df)-5):
        ts = df.iloc[i]["time"]
        if not in_session(ts):
            continue
        sig = compute_signal(df, i)
        if not sig:
            continue
        signals += 1
        entry = df.iloc[i]["c"]
        future = df.iloc[i+1:i+6]["c"]
        if sig == "CALL":
            if future.max() > entry: wins += 1
            elif future.min() < entry: losses += 1
            else: draws += 1
        elif sig == "PUT":
            if future.min() < entry: wins += 1
            elif future.max() > entry: losses += 1
            else: draws += 1

    return pair, {"signals": signals, "W": wins, "L": losses, "D": draws}

async def main():
    results = {}
    tasks = [backtest_pair(p) for p in OANDA_INSTRUMENTS]
    for coro in asyncio.as_completed(tasks):
        pair, res = await coro
        results[pair] = res

    total_s = sum(r["signals"] for r in results.values())
    total_w = sum(r["W"] for r in results.values())
    total_l = sum(r["L"] for r in results.values())
    total_d = sum(r["D"] for r in results.values())

    print("=== BACKTEST RESULTS (30 DAYS) ===")
    for p, r in results.items():
        print(f"{p}: {r}")
    print(f"\nTOTAL: {total_s} signals | W:{total_w} L:{total_l} D:{total_d}")
    winrate = total_w / max(total_s, 1) * 100
    print(f"Win rate: {winrate:.2f}%")

if __name__ == "__main__":
    asyncio.run(main())
