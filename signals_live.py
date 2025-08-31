"""
signals_live.py - OANDA M1 scalper runner (MACD+RSI+EMA trend)
- Uses instruments listed in symbols.yaml with an 'oanda' key (e.g. EUR_USD)
- Sends only confirmed trades (🟢/🔴) to all tiers
Env:
  INTERVAL=1m (fixed for OANDA M1)
  EXPIRY_MIN=5
  MIN_SCORE=1..4  (1=more signals, 3+=stricter)
  SUPPRESS_EMPTY=1 to skip posts when nothing confirms
"""
import os, requests, yaml
from datetime import datetime, timezone
import pandas as pd

from scalper import add_indicators, classify
from telegram_send import send_to_tiers

OANDA_API_KEY = os.getenv("OANDA_API_KEY","")
OANDA_ENV     = os.getenv("OANDA_ENV","practice").lower()
OANDA_HOST    = "https://api-fxpractice.oanda.com" if OANDA_ENV!="live" else "https://api-fxtrade.oanda.com"

INTERVAL   = os.getenv("INTERVAL","1m")  # display-only; fetch is M1
EXPIRY_MIN = int(os.getenv("EXPIRY_MIN","5"))
MIN_SCORE  = int(os.getenv("MIN_SCORE","1"))
SUPPRESS_EMPTY = os.getenv("SUPPRESS_EMPTY","1") == "1"

def oanda_fetch_m1(instrument: str, count: int = 220) -> pd.DataFrame:
    """Return DataFrame with columns: time, open, high, low, close (floats)."""
    if not OANDA_API_KEY:
        raise RuntimeError("OANDA_API_KEY not set")
    url = f"{OANDA_HOST}/v3/instruments/{instrument}/candles"
    params = {"granularity":"M1","count":count,"price":"M"}
    headers = {"Authorization": f"Bearer {OANDA_API_KEY}"}
    r = requests.get(url, params=params, headers=headers, timeout=20)
    r.raise_for_status()
    data = r.json().get("candles", [])
    if not data:
        return pd.DataFrame()
    rows = []
    for c in data:
        mid = c.get("mid", {})
        rows.append({
            "time":  c["time"],
            "open":  float(mid["o"]),
            "high":  float(mid["h"]),
            "low":   float(mid["l"]),
            "close": float(mid["c"]),
        })
    df = pd.DataFrame(rows)
    # Ensure proper order (oldest->newest) and index
    df = df.sort_values("time").reset_index(drop=True)
    return df

def load_oanda_list():
    """Read symbols.yaml and return [(instrument, pretty_name)]."""
    try:
        cfg = yaml.safe_load(open("symbols.yaml"))
    except Exception:
        return []
    out = []
    for s in cfg.get("symbols", []):
        if not s.get("enabled", True):
            continue
        inst = s.get("oanda") or s.get("instrument")
        name = s.get("name") or s.get("pretty") or inst
        if inst:
            out.append((inst, name))
    return out

def header_lines(now):
    return [
        f"📡 Pocket Option Signals — {now:%Y-%m-%d %H:%M UTC}",
        f"Candle: {INTERVAL} | Expiry: {EXPIRY_MIN}m",
        ""
    ]

def main() -> list[str]:
    now = datetime.now(timezone.utc)
    lines = header_lines(now)

    universe = load_oanda_list()
    if not universe:
        # Fallback small list if symbols.yaml lacks 'oanda' keys
        universe = [("EUR_USD","EUR/USD"),("GBP_USD","GBP/USD"),("USD_JPY","USD/JPY"),("XAU_USD","Gold")]
    confirmed = []

    for inst, pretty in universe:
        try:
            df = oanda_fetch_m1(inst, count=220)
            if df is None or df.empty or len(df) < 60:  # need enough bars for EMA200, etc.
                continue
            df = add_indicators(df)
            if df is None or df.empty or len(df) < 2:
                continue
            prev, cur = df.iloc[-2], df.iloc[-1]

            side, score, why = classify(prev, cur, min_score=MIN_SCORE)
            if side and score >= MIN_SCORE:
                emoji = "🟢" if side == "BUY" else "🔴"
                entry = f"{cur['close']:.5f}" if abs(cur['close']) < 100 else f"{cur['close']:.3f}"
                confirmed.append(f"{emoji} {pretty} — {side} @ {entry}")
                confirmed.append(f"• {why} (score {score})")
                confirmed.append("")  # spacer
        except Exception as e:
            # keep quiet in production; uncomment to log:
            # lines.append(f"⚠️ {pretty} — {e}")
            pass

    if confirmed:
        lines.extend(confirmed)
    else:
        # nothing confirmed; respect SUPPRESS_EMPTY
        if SUPPRESS_EMPTY:
            return []  # special: caller will skip send
        else:
            lines.append("⚠️ No confirmed setups this run.")

    return lines

if __name__ == "__main__":
    out = main()
    if not out:
        print("✋ suppressed empty (no confirmed signals).")
    else:
        msg = "\n".join(out)
        print(msg)
