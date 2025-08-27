import os, time, csv, csv, json
from pathlib import Path
from datetime import datetime, timezone, timedelta

import numpy as np
import pandas as pd
import yfinance as yf
import yaml
import requests

import io, requests

def stooq_symbol(yf_sym: str) -> str | None:
    """
    Map Yahoo FX and metals to Stooq codes.
    Examples:
      EURUSD=X -> eurusd
      GBPUSD=X -> gbpusd
      USDJPY=X -> usdjpy
      XAUUSD=X -> xauusd
    Returns lowercase stooq symbol or None if unmappable.
    """
    if yf_sym.endswith("=X"):
        base = yf_sym[:-2].lower()
        return base  # e.g., 'eurusd'
    if yf_sym.upper() in ("XAUUSD=X","XAGUSD=X"):
        return yf_sym[:6].lower()
    return None

def stooq_fetch_one(yf_sym: str, interval: str, period: str) -> pd.DataFrame:
    """
    Fetch intraday CSV from Stooq. Supported i=1,5,10,15,30,60.
    period ignored; Stooq returns recent history.
    """
    sym = stooq_symbol(yf_sym)
    if not sym: 
        raise RuntimeError("no stooq mapping")
    # map interval like '5m' -> '5', '1m' -> '1'
    mins = ''.join(ch for ch in interval if ch.isdigit())
    if not mins: mins = '5'
    url = f"https://stooq.com/q/d/l/?s={sym}&i={mins}"
    r = requests.get(url, timeout=15, headers={"User-Agent":"Mozilla/5.0"})
    r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text))
    if df.empty: 
        raise RuntimeError("stooq empty")
    # Normalize to yfinance-like columns
    rename = {"Date":"Datetime","Open":"open","High":"high","Low":"low","Close":"close","Volume":"volume"}
    df = df.rename(columns=rename)
    # Stooq timestamps are UTC already
    df["Datetime"] = pd.to_datetime(df["Datetime"], utc=True, errors="coerce")
    df = df.dropna(subset=["Datetime"])
    df = df.set_index("Datetime").sort_index()
    return df[["open","high","low","close","volume"]].dropna()

def stooq_fetch_bulk(yf_symbols: list[str], interval: str) -> dict[str,pd.DataFrame]:
    out={}
    for s in yf_symbols:
        try:
            out[s]=stooq_fetch_one(s, interval, "ignored")
        except Exception:
            pass
    # Try Stooq for any missing symbols
    for sym in yf_symbols:
        if sym not in out or out[sym].empty:
            try:
                out[sym] = stooq_fetch_one(sym, interval, period)
            except Exception:
                pass
    for sym in yf_symbols:
        if sym not in out or out[sym].empty:
            try:
                out[sym] = stooq_fetch_one(sym, interval, period)
            except Exception:
                pass
    return out
def bulk_fetch(*args, **kwargs):
    yf_symbols = args[0] if args else kwargs.get('yf_symbols')
    interval   = args[1] if len(args)>1 else kwargs.get('interval', INTERVAL)
    return oanda_fetch_bulk(yf_symbols, interval)



# === OANDA-ONLY LOCAL FETCH ENABLED ===
print('🔄 Using OANDA-only local fetchers (no Yahoo/Stooq)')


# ---------- Tier routing ----------
def _queue_write(path, when_iso, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["available_at_utc","text"])
        if not exists: w.writeheader()
        w.writerow({"available_at_utc": when_iso, "text": text})

def _post_now(chat_id, text):
    if not BOT_TOKEN or not chat_id: return
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    import requests
    r = requests.post(url, data={"chat_id": chat_id, "text": text, "parse_mode":"Markdown"}, timeout=30)
    r.raise_for_status()

def send_to_tiers(text, now_utc=None):
    from datetime import datetime, timezone, timedelta
    now_utc = now_utc or datetime.now(timezone.utc)
    # VIP & PRO: instant if configured
    if 'CHAT_VIP' in globals() and CHAT_VIP:   _post_now(CHAT_VIP, text)
    if 'CHAT_PRO' in globals() and CHAT_PRO:   _post_now(CHAT_PRO, text)

    # BASIC: queue or instant
    if 'CHAT_BASIC' in globals() and CHAT_BASIC:
        if BASIC_DELAY_MIN > 0:
            when = now_utc + timedelta(minutes=BASIC_DELAY_MIN)
            _queue_write(Path("data/basic_queue.csv"), when.strftime("%Y-%m-%d %H:%M:%S"), text)
        else:
            _post_now(CHAT_BASIC, text)

    # FREE: queue or instant
    if 'CHAT_FREE' in globals() and CHAT_FREE:
        if FREE_DELAY_MIN > 0:
            when = now_utc + timedelta(minutes=FREE_DELAY_MIN)
            _queue_write(Path("data/free_queue.csv"), when.strftime("%Y-%m-%d %H:%M:%S"), text)
        else:
            _post_now(CHAT_FREE, text)

# Final dispatch
try:
    send_to_tiers("\n\n".join(lines))
except Exception:
    send_telegram("\n\n".join(lines))
