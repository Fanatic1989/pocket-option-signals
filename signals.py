import os, time, csv
from pathlib import Path
from datetime import datetime, timezone, timedelta

import numpy as np
import pandas as pd
import yfinance as yf
import yaml
import requests

# ===== Config (env-overridable) =====
INTERVAL   = os.getenv("INTERVAL", "5m")            # candle size
LOOKBACK   = os.getenv("LOOKBACK", "15d")           # safer lookback for 5m
EXPIRY_MIN = int(os.getenv("EXPIRY_MIN", "10"))     # option expiry (minutes)
RSI_BUY    = int(os.getenv("RSI_BUY", "30"))
RSI_SELL   = int(os.getenv("RSI_SELL", "70"))
MIN_SCORE  = int(os.getenv("MIN_SCORE", "2"))       # filter weak confluence
MIN_ROWS   = 220                                     # EMA200 + buffer
RETRIES    = 3
RETRY_SLEEP= 2.0

# ===== Session filter (to avoid OTC) =====
# Format "HHMM-HHMM" in UTC, weekdays "12345" = Mon–Fri
SESSION_UTC = os.getenv("SESSION_UTC", "0700-1700")
WEEKDAYS    = os.getenv("WEEKDAYS", "12345")        # Mon=1 ... Sun=7
# Groups that must follow session gate (CRYPTO runs 24/7 by default)
GATED_GROUPS = set(os.getenv("GATED_GROUPS", "FX,COMMODITY,INDEX").split(","))

DATA_DIR    = Path("data"); DATA_DIR.mkdir(parents=True, exist_ok=True)
SIGNALS_CSV = DATA_DIR / "signals.csv"
SYMBOLS_YML = Path("symbols.yaml")

BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
CHAT_ID   = os.getenv("TELEGRAM_CHAT_ID")  # e.g. @YourChannelUsername

# --------- Indicators (no pandas_ta) ---------
def ema(series: pd.Series, length: int) -> pd.Series:
    return series.ewm(span=length, adjust=False).mean()

def rsi(series: pd.Series, length: int = 14) -> pd.Series:
    delta = series.diff()
    gain  = delta.clip(lower=0)
    loss  = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/length, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/length, adjust=False).mean().replace(0, np.nan)
    rs = avg_gain / avg_loss
    return (100 - (100 / (1 + rs))).fillna(50)

def macd(series: pd.Series, fast=12, slow=26, signal=9):
    macd_line = ema(series, fast) - ema(series, slow)
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist

# --------- Session gate helpers ---------
def _parse_session(s: str):
    try:
        a, b = s.split("-")
        return int(a), int(b)  # HHMM ints
    except Exception:
        return 0, 2359

def in_session_utc(now: datetime, group: str) -> bool:
    # Crypto always allowed unless you explicitly add it to GATED_GROUPS
    if group.upper() not in GATED_GROUPS:
        return True
    wd = str((now.isoweekday()))  # 1..7
    if wd not in set(WEEKDAYS):
        return False
    start, end = _parse_session(SESSION_UTC)
    hhmm = now.hour*100 + now.minute
    return start <= hhmm <= end

# ----------------------------------------------
def load_symbols():
    with open(SYMBOLS_YML, "r") as f:
        cfg = yaml.safe_load(f)
    syms = [(s["yf"], s["name"], str(s.get("group","FX")).upper()) for s in cfg["symbols"] if s.get("enabled", True)]
    if not syms:
        raise SystemExit("No enabled symbols in symbols.yaml")
    return syms

def fetch_df(yf_symbol: str) -> pd.DataFrame | None:
    for attempt in range(1, RETRIES + 1):
        try:
            df = yf.download(
                tickers=yf_symbol,
                interval=INTERVAL,
                period=LOOKBACK,
                progress=False,
                auto_adjust=False,
                threads=False,
            )
            if df is None or df.empty:
                raise ValueError("empty dataframe")
            df = df.dropna().rename(columns=str.lower)
            if len(df) < MIN_ROWS:
                raise ValueError(f"too few rows ({len(df)})")
            return df
        except Exception:
            if attempt == RETRIES:
                return None
            time.sleep(RETRY_SLEEP)
    return None

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    close = df["close"]
    df["ema50"]  = ema(close, 50)
    df["ema200"] = ema(close, 200)
    macd_line, signal_line, _ = macd(close, 12, 26, 9)
    df["macd"]  = macd_line
    df["macds"] = signal_line
    df["rsi14"] = rsi(close, 14)
    return df.dropna()

def confluence(prev, cur):
    score=0; reasons=[]
    uptrend = cur["ema50"] > cur["ema200"]
    reasons.append("EMA50>EMA200" if uptrend else "EMA50<EMA200")
    macd_up   = (prev["macd"] <= prev["macds"]) and (cur["macd"] > cur["macds"])
    macd_down = (prev["macd"] >= prev["macds"]) and (cur["macd"] < cur["macds"])
    if macd_up:   reasons.append("MACD↑"); score+=1
    if macd_down: reasons.append("MACD↓"); score+=1
    if cur["rsi14"] <= RSI_BUY:  reasons.append(f"RSI≤{RSI_BUY}");  score+=1
    if cur["rsi14"] >= RSI_SELL: reasons.append(f"RSI≥{RSI_SELL}"); score+=1
    return score, ", ".join(reasons), uptrend, macd_up, macd_down

def classify(prev, cur):
    score, why, up, m_up, m_down = confluence(prev, cur)
    signal=None
    if up and m_up: signal="BUY"
    if (not up) and m_down: signal="SELL"
    if signal is None:
        if cur["rsi14"] <= RSI_BUY: signal="BUY"
        elif cur["rsi14"] >= RSI_SELL: signal="SELL"
    return signal, score, why

def send_telegram(text: str):
    if not BOT_TOKEN or not CHAT_ID: return
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    r = requests.post(url, data={"chat_id": CHAT_ID, "text": text, "parse_mode":"Markdown"}, timeout=30)
    r.raise_for_status()

def append_signal(row: dict):
    header = ["ts_utc","symbol_yf","symbol_pretty","signal","price","expiry_min","evaluate_at_utc","status","score","why"]
    exists = SIGNALS_CSV.exists()
    with open(SIGNALS_CSV, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        if not exists: w.writeheader()
        w.writerow(row)

def main():
    t0 = time.time()
    now = datetime.now(timezone.utc)

    enabled = load_symbols()
    lines = [f"📡 *Pocket Option Signals* — {now:%Y-%m-%d %H:%M UTC}\nInterval: {INTERVAL} | Expiry: {EXPIRY_MIN}m\nSession: {SESSION_UTC} UTC, Weekdays {WEEKDAYS}\n"]

    for yf_sym, pretty, group in enabled:
        if not in_session_utc(now, group):
            lines.append(f"⚪ *{pretty}* — skipped (outside session for {group})")
            continue

        df = fetch_df(yf_sym)
        if df is None or len(df) < MIN_ROWS:
            lines.append(f"⚪ *{pretty}* — skipped (no/short data)")
            continue

        try:
            df = add_indicators(df)
            if len(df) < 2:
                lines.append(f"⚪ *{pretty}* — skipped (insufficient after indicators)")
                continue
            prev, cur = df.iloc[-2], df.iloc[-1]
            px = float(cur["close"])
            sig, score, why = classify(prev, cur)

            if sig and score >= MIN_SCORE:
                arrow = "🟢 BUY" if sig=="BUY" else "🔴 SELL"
                lines.append(f"{arrow} *{pretty}* @ `{px:.5f}`\n• {why} (score {score})")
                evaluate_at = now + timedelta(minutes=EXPIRY_MIN)
                append_signal({
                    "ts_utc": now.strftime("%Y-%m-%d %H:%M:%S"),
                    "symbol_yf": yf_sym,
                    "symbol_pretty": pretty,
                    "signal": sig,
                    "price": f"{px:.8f}",
                    "expiry_min": EXPIRY_MIN,
                    "evaluate_at_utc": evaluate_at.strftime("%Y-%m-%d %H:%M:%S"),
                    "status": "open",
                    "score": score,
                    "why": why
                })
            else:
                lines.append(f"⚪ *{pretty}* — No setup (score {score if sig else 0})")
        except Exception as e:
            lines.append(f"⚠️ *{pretty}* error: {e}")

    build_time = time.time() - t0
    lines.append(f"\n⏱ Build time: `{build_time:.1f}s`")
    send_telegram("\n\n".join(lines))

if __name__ == "__main__":
    main()
