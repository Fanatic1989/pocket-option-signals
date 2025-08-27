import os, time, csv, json
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
