import os
import time
import csv
import json
import requests
from pathlib import Path
from datetime import datetime, timezone, timedelta

import pandas as pd
import pandas_ta as ta
from oanda_utils import oanda_get_candle
import yfinance as yf
import yaml

def _signal_header(now, interval, expiry_min):
    return [
        f"📡 Pocket Option Signals — {now:%Y-%m-%d %H:%M UTC}",
        f"Candle: {interval} | Expiry: {expiry_min}m",
        ""
    ]

import io, requests
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass


# ========= Config / Env =========
INTERVAL   = os.getenv("INTERVAL", "5m")
LOOKBACK   = os.getenv("LOOKBACK", "5d")
EXPIRY_MIN = int(os.getenv("EXPIRY_MIN", "10"))
RSI_BUY    = int(os.getenv("RSI_BUY", "30"))
RSI_SELL   = int(os.getenv("RSI_SELL", "70"))
MIN_SCORE  = int(os.getenv("MIN_SCORE", "2"))
MUST_TRADE = int(os.getenv("MUST_TRADE", "0"))
SUPPRESS_EMPTY = int(os.getenv("SUPPRESS_EMPTY", "1"))

DATA_DIR    = Path("data"); DATA_DIR.mkdir(parents=True, exist_ok=True)
SIGNALS_CSV = DATA_DIR / "signals.csv"
PERF_CSV    = DATA_DIR / "perf.csv"
SYMBOLS_YML = Path("symbols.yaml")

# ========= Signal Script =========
def load_symbols():
    with open(SYMBOLS_YML, "r") as f:
        cfg = yaml.safe_load(f)
    return [(s["yf"], s["name"]) for s in cfg["symbols"] if s.get("enabled", True)]

def send_telegram(msg: str):
    bot = os.getenv("TELEGRAM_BOT_TOKEN")
    chat = os.getenv("TELEGRAM_CHAT_PRO")
    if not bot or not chat:
        print("⚠️ Telegram not configured.")
        return
    r = requests.post(
        f"https://api.telegram.org/bot{bot}/sendMessage",
        data={"chat_id": chat, "text": msg, "parse_mode":"Markdown"},
        timeout=30
    )
    print("Telegram response:", r.status_code, r.text)

def send_to_tiers(msg: str):
    for tier in ["TELEGRAM_CHAT_FREE","TELEGRAM_CHAT_BASIC","TELEGRAM_CHAT_PRO","TELEGRAM_CHAT_VIP"]:
        chat = os.getenv(tier)
        bot  = os.getenv("TELEGRAM_BOT_TOKEN")
        if not chat or not bot: 
            continue
        try:
            requests.post(
                f"https://api.telegram.org/bot{bot}/sendMessage",
                data={"chat_id": chat, "text": msg, "parse_mode":"Markdown"},
                timeout=30
            )
            print(f"✅ Sent to {tier}")
        except Exception as e:
            print(f"⚠️ Failed {tier}:", e)


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty: return pd.DataFrame()
    df = df.copy()
    df["ema50"]  = ta.ema(df["close"], length=50)
    df["ema200"] = ta.ema(df["close"], length=200)
    macd = ta.macd(df["close"], fast=12, slow=26, signal=9)
    df["macd"]  = macd["MACD_12_26_9"]
    df["macds"] = macd["MACDs_12_26_9"]
    df["rsi14"] = ta.rsi(df["close"], length=14)
    return df.dropna()

def classify(prev, cur):
    score=0; reasons=[]
    up = cur["ema50"] > cur["ema200"]
    reasons.append("EMA50>EMA200" if up else "EMA50<EMA200")
    macd_up   = (prev["macd"] <= prev["macds"]) and (cur["macd"] > cur["macds"])
    macd_down = (prev["macd"] >= prev["macds"]) and (cur["macd"] < cur["macds"])
    if macd_up:   reasons.append("MACD↑"); score+=1
    if macd_down: reasons.append("MACD↓"); score+=1
    if cur["rsi14"] <= int(os.getenv("RSI_BUY","30")):  reasons.append("RSI≤BUY");  score+=1
    if cur["rsi14"] >= int(os.getenv("RSI_SELL","70")): reasons.append("RSI≥SELL"); score+=1
    sig=None
    if up and macd_up: sig="BUY"
    if (not up) and macd_down: sig="SELL"
    if sig is None:
        if cur["rsi14"] <= int(os.getenv("RSI_BUY","30")): sig="BUY"
        elif cur["rsi14"] >= int(os.getenv("RSI_SELL","70")): sig="SELL"
    return sig, score, ", ".join(reasons)

def fetch_df(yf_symbol: str) -> pd.DataFrame:
    # No Yahoo. Try Stooq (FX/etc) then Binance (crypto).
    try:
        df = fetch_stooq(yf_symbol, INTERVAL)
        if df is not None and not df.empty:
            return df
    except Exception:
        pass
    try:
        df = fetch_binance(yf_symbol, INTERVAL)
        if df is not None and not df.empty:
            return df
    except Exception:
        pass
    return pd.DataFrame()

def main():
    t0 = time.time()
    now = datetime.now(timezone.utc)
    lines = [
        f"📡 Pocket Option Signals — {now.strftime('%Y-%m-%d %H:%M UTC')}",
        f"Candle: {INTERVAL} | Expiry: {EXPIRY_MIN}m",
        
    any_sig = False

    # load symbols
    with open(SYMBOLS_YML,"r") as f:
        cfg = yaml.safe_load(f)
    syms = [(s["yf"], s["name"]) for s in cfg["symbols"] if s.get("enabled", True)]

    for yf_symbol, name in syms:
        try:
            df = add_indicators(fetch_df(yf_symbol))
            if df is None or df.empty or len(df) < 3:
                lines.append(f"⚪ {name} — no/short data")
                continue
            prev, cur = df.iloc[-2], df.iloc[-1]
            px = float(cur["close"])
            sig, score, why = classify(prev, cur)
            if sig and score >= int(os.getenv("MIN_SCORE","2")):
                any_sig = True
                arrow = "🟢 BUY" if sig=="BUY" else "🔴 SELL"
                lines.append(f"{arrow} *{name}* @ `{px:.5f}`\n• {why} (score {score})")
            else:
                lines.append(f"⚪ {name} — no setup (score {score if sig else 0})")
        except Exception as e:
            lines.append(f"⚠️ {name} error: {e}")

    if int(os.getenv("MUST_TRADE","0")) and not any_sig:
        lines.append("\n🧪 *TEST SIGNAL* — forced pick (MUST_TRADE=1)")
        any_sig = True

    lines.append(f"\n⏱ Build time: `{time.time()-t0:.1f}s`")
    if int(os.getenv("SUPPRESS_EMPTY","1")) and not any_sig:
        return []
    return lines



if __name__ == "__main__":
    try:
        lines = []
        try:
            result = main()
            if result:
                lines = result
        except Exception as inner:
            print("⚠️ Error running main():", inner)

        if not lines:
            lines = ["⚠️ No signals generated (debug)."]

        full_msg = "\n\n".join(lines)

        try:
            send_to_tiers(full_msg)
        except NameError:
            print("send_to_tiers() missing, falling back to send_telegram()")
            send_telegram(full_msg)

        print("✅ Signals run completed.")
    except Exception as e:
        import traceback
        print("❌ Fatal error:", e)
        traceback.print_exc()
