# strategy.py — indicator prep + cores (BASE/TREND/CHOP/WIDE) for signals
from __future__ import annotations
import re
from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd

# --------- indicators ----------
def _sma(series: pd.Series, period: int) -> pd.Series:
    period = max(int(period or 1), 1)
    return series.rolling(period, min_periods=period).mean()

def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    period = max(int(period or 1), 1)
    delta = close.diff()
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    roll_up = pd.Series(gain, index=close.index).rolling(period, min_periods=period).mean()
    roll_down = pd.Series(loss, index=close.index).rolling(period, min_periods=period).mean()
    rs = roll_up / (roll_down.replace(0, np.nan))
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return pd.Series(rsi, index=close.index).bfill()

def _stoch(high: pd.Series, low: pd.Series, close: pd.Series, k: int = 14, d: int = 3) -> Tuple[pd.Series, pd.Series]:
    k = max(int(k or 1), 1); d = max(int(d or 1), 1)
    ll = low.rolling(k, min_periods=k).min()
    hh = high.rolling(k, min_periods=k).max()
    rng = (hh - ll).replace(0, np.nan)
    raw_k = (close - ll) / rng * 100.0
    k_line = raw_k.rolling(d, min_periods=1).mean()
    d_line = k_line.rolling(d, min_periods=1).mean()
    return k_line.fillna(50.0), d_line.fillna(50.0)

def _slope(series: pd.Series, lookback: int = 5) -> pd.Series:
    lookback = max(int(lookback or 1), 1)
    idx = np.arange(len(series))
    out = np.full(len(series), np.nan)
    for i in range(lookback - 1, len(series)):
        x = idx[i - lookback + 1 : i + 1]
        y = series.iloc[i - lookback + 1 : i + 1].values
        x_mean = x.mean(); y_mean = y.mean()
        num = ((x - x_mean) * (y - y_mean)).sum()
        den = ((x - x_mean) ** 2).sum()
        out[i] = num / den if den != 0 else 0.0
    return pd.Series(out, index=series.index)

# --------- expiry mapping ----------
def _expiry_to_bars(tf: str, expiry: str) -> int:
    tf = (tf or "M5").upper()
    expiry = (expiry or "5m").upper()
    tf_sec_map = {"M1":60,"M2":120,"M3":180,"M5":300,"M10":600,"M15":900,"M30":1800,"H1":3600,"H4":14400,"D1":86400}
    tf_sec = tf_sec_map.get(tf, 300)
    e = expiry.upper().replace("MIN","M").replace("MINS","M").replace("HR","H").replace("HRS","H")
    m = re.fullmatch(r"(\d+)\s*([MH]?)", e)
    if not m:
        if e.endswith("M"): secs = int(e[:-1]) * 60
        elif e.startswith("H"): secs = int(e[1:]) * 3600
        else: secs = 300
    else:
        n = int(m.group(1)); unit = m.group(2) or "M"
        secs = n * (60 if unit == "M" else 3600)
    bars = max(int(round(secs / tf_sec)), 1)
    return bars

# --------- prep ----------
def _ensure_cols(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d.columns = [c.strip().lower() for c in d.columns]
    if "timestamp" not in d.columns:
        d["timestamp"] = range(len(d))
    for c in ("open","high","low","close"):
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d.dropna(subset=["close"]).reset_index(drop=True)
    return d

def _prep_indicators(d: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
    sma_period = int(params.get("sma_period", 20))
    rsi_period = int(params.get("rsi_period", 14))
    st_k = int(params.get("stoch_k", 14))
    st_d = int(params.get("stoch_d", 3))
    d["sma"] = _sma(d["close"], sma_period)
    d["rsi"] = _rsi(d["close"], rsi_period)
    d["sk"], d["sd"] = _stoch(d["high"], d["low"], d["close"], st_k, st_d)
    d["sma_slope"] = _slope(d["sma"], lookback=5)
    return d

# --------- helpers ----------
def _cross_up(a_prev, a_now, b_prev, b_now) -> bool:
    try: return a_prev < b_prev and a_now > b_now
    except Exception: return False

def _cross_dn(a_prev, a_now, b_prev, b_now) -> bool:
    try: return a_prev > b_prev and a_now < b_now
    except Exception: return False

# --------- cores ----------
def _base_conditions(d: pd.DataFrame):
    buy = pd.Series(False, index=d.index)
    sell = pd.Series(False, index=d.index)
    for i in range(1, len(d)):
        buy.iloc[i] = (
            d["close"].iloc[i] > d["sma"].iloc[i] and
            d["rsi"].iloc[i] > 55 and
            _cross_up(d["sk"].iloc[i-1], d["sk"].iloc[i], d["sd"].iloc[i-1], d["sd"].iloc[i])
        )
        sell.iloc[i] = (
            d["close"].iloc[i] < d["sma"].iloc[i] and
            d["rsi"].iloc[i] < 45 and
            _cross_dn(d["sk"].iloc[i-1], d["sk"].iloc[i], d["sd"].iloc[i-1], d["sd"].iloc[i])
        )
    return buy, sell

def _trend_conditions(d: pd.DataFrame):
    buy = pd.Series(False, index=d.index)
    sell = pd.Series(False, index=d.index)
    for i in range(2, len(d)):
        buy.iloc[i] = (
            (d["sma_slope"].iloc[i] > 0) and
            (d["close"].iloc[i-1] <= d["sma"].iloc[i-1]) and
            (d["close"].iloc[i]   >  d["sma"].iloc[i]) and
            (d["rsi"].iloc[i-1] <= 50) and (d["rsi"].iloc[i] > 50)
        )
        sell.iloc[i] = (
            (d["sma_slope"].iloc[i] < 0) and
            (d["close"].iloc[i-1] >= d["sma"].iloc[i-1]) and
            (d["close"].iloc[i]   <  d["sma"].iloc[i]) and
            (d["rsi"].iloc[i-1] >= 50) and (d["rsi"].iloc[i] < 50)
        )
    return buy, sell

def _chop_conditions(d: pd.DataFrame):
    buy = pd.Series(False, index=d.index)
    sell = pd.Series(False, index=d.index)
    for i in range(1, len(d)):
        buy.iloc[i] = (d["rsi"].iloc[i] < 30) and (d["sk"].iloc[i] > d["sd"].iloc[i])
        sell.iloc[i] = (d["rsi"].iloc[i] > 70) and (d["sk"].iloc[i] < d["sd"].iloc[i])
    return buy, sell

def _wide_conditions(d: pd.DataFrame):
    """
    More permissive than BASE:
      BUY  = close > SMA and RSI >= 50 and (K >= D or K rising)
      SELL = close < SMA and RSI <= 50 and (K <= D or K falling)
    """
    buy = pd.Series(False, index=d.index)
    sell = pd.Series(False, index=d.index)
    for i in range(1, len(d)):
        k_now, k_prev = d["sk"].iloc[i], d["sk"].iloc[i-1]
        d_now, d_prev = d["sd"].iloc[i], d["sd"].iloc[i-1]
        k_ge_d = (k_now >= d_now)
        k_up   = (k_now > k_prev)
        k_le_d = (k_now <= d_now)
        k_dn   = (k_now < k_prev)
        buy.iloc[i] = (d["close"].iloc[i] > d["sma"].iloc[i]) and (d["rsi"].iloc[i] >= 50.0) and (k_ge_d or k_up)
        sell.iloc[i] = (d["close"].iloc[i] < d["sma"].iloc[i]) and (d["rsi"].iloc[i] <= 50.0) and (k_le_d or k_dn)
    return buy, sell

# (exported names are imported by worker_inline)
