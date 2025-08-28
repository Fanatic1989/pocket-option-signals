import os, time, csv, json, csv, json
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


# ---------- Tier limits & counters ----------
_LIMITS_FILE = Path("tier_limits.json")
_DATA_DIR = Path("data"); _DATA_DIR.mkdir(parents=True, exist_ok=True)

def _load_limits():
    try:
        return json.load(open(_LIMITS_FILE))
    except Exception:
        # sensible defaults if file missing
        return {"FREE": 5, "BASIC": 15, "PRO": 40, "VIP": 999999}

def _count_file_for_today():
    d = datetime.datetime.utcnow().strftime("%Y%m%d")
    return _DATA_DIR / f"tier_counts_{d}.json"

def _load_counts():
    f = _count_file_for_today()
    if f.exists():
        try:
            return json.load(open(f))
        except Exception:
            return {}
    return {}

def _save_counts(c):
    json.dump(c, open(_count_file_for_today(), "w"))

def _tier_can_send(tier: str) -> bool:
    lim = _load_limits()
    cap = int(lim.get(tier.upper(), 999999))
    counts = _load_counts()
    return int(counts.get(tier.upper(), 0)) < cap

def _bump_tier_count(tier: str):
    counts = _load_counts()
    key = tier.upper()
    counts[key] = int(counts.get(key, 0)) + 1
    _save_counts(counts)


def send_to_tiers(text, now_utc=None):
    from datetime import datetime, timezone, timedelta
    now_utc = now_utc or datetime.now(timezone.utc)

    def try_send(tier_name, chat_id):
        if not chat_id:
            return
        # VIP always allowed (large cap), others respect caps
        if _tier_can_send(tier_name):
            _post_now(chat_id, text)
            _bump_tier_count(tier_name)

    # VIP & PRO: instant
    try_send("VIP",   CHAT_VIP   if 'CHAT_VIP'   in globals() else None)
    try_send("PRO",   CHAT_PRO   if 'CHAT_PRO'   in globals() else None)

    # BASIC & FREE: now instant too (caps still apply)
    try_send("BASIC", CHAT_BASIC if 'CHAT_BASIC' in globals() else None)
    try_send("FREE",  CHAT_FREE  if 'CHAT_FREE'  in globals() else None)


# Final dispatch
try:
    send_to_tiers("\n\n".join(lines))
except Exception:
    send_telegram("\n\n".join(lines))
