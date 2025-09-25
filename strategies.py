# strategies.py
from __future__ import annotations
import re
import json
from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd

# -------------------- Indicators & helpers --------------------

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

# -------------------- Expiry mapping --------------------

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

# -------------------- Data housekeeping --------------------

def _ensure_cols(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    # normalize column names to lower case
    d.columns = [c.strip().lower() for c in d.columns]
    # ensure timestamp exists
    if "timestamp" not in d.columns:
        d["timestamp"] = range(len(d))
    # coerce numeric OHLC
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
    # add indicators
    d["sma"] = _sma(d["close"], sma_period)
    d["rsi"] = _rsi(d["close"], rsi_period)
    d["sk"], d["sd"] = _stoch(d.get("high", d["close"]), d.get("low", d["close"]), d["close"], st_k, st_d)
    d["sma_slope"] = _slope(d["sma"], lookback=5)
    return d

# -------------------- small helpers --------------------

def _cross_up(a_prev, a_now, b_prev, b_now) -> bool:
    try:
        return (a_prev < b_prev) and (a_now > b_now)
    except Exception:
        return False

def _cross_dn(a_prev, a_now, b_prev, b_now) -> bool:
    try:
        return (a_prev > b_prev) and (a_now < b_now)
    except Exception:
        return False

# -------------------- Strategy cores --------------------

def _base_conditions(d: pd.DataFrame):
    buy = pd.Series(False, index=d.index)
    sell = pd.Series(False, index=d.index)
    for i in range(1, len(d)):
        try:
            buy.iloc[i] = (
                (d["close"].iloc[i] > d["sma"].iloc[i]) and
                (d["rsi"].iloc[i] > 55) and
                _cross_up(d["sk"].iloc[i-1], d["sk"].iloc[i], d["sd"].iloc[i-1], d["sd"].iloc[i])
            )
            sell.iloc[i] = (
                (d["close"].iloc[i] < d["sma"].iloc[i]) and
                (d["rsi"].iloc[i] < 45) and
                _cross_dn(d["sk"].iloc[i-1], d["sk"].iloc[i], d["sd"].iloc[i-1], d["sd"].iloc[i])
            )
        except Exception:
            buy.iloc[i] = False
            sell.iloc[i] = False
    return buy, sell

def _trend_conditions(d: pd.DataFrame):
    buy = pd.Series(False, index=d.index)
    sell = pd.Series(False, index=d.index)
    for i in range(2, len(d)):
        try:
            buy.iloc[i] = (
                (d["sma_slope"].iloc[i] > 0) and
                (d["close"].iloc[i-1] <= d["sma"].iloc[i-1]) and
                (d["close"].iloc[i] > d["sma"].iloc[i]) and
                (d["rsi"].iloc[i-1] <= 50) and (d["rsi"].iloc[i] > 50)
            )
            sell.iloc[i] = (
                (d["sma_slope"].iloc[i] < 0) and
                (d["close"].iloc[i-1] >= d["sma"].iloc[i-1]) and
                (d["close"].iloc[i] < d["sma"].iloc[i]) and
                (d["rsi"].iloc[i-1] >= 50) and (d["rsi"].iloc[i] < 50)
            )
        except Exception:
            buy.iloc[i] = False
            sell.iloc[i] = False
    return buy, sell

def _chop_conditions(d: pd.DataFrame):
    buy = pd.Series(False, index=d.index)
    sell = pd.Series(False, index=d.index)
    for i in range(1, len(d)):
        try:
            buy.iloc[i] = (d["rsi"].iloc[i] < 30) and (d["sk"].iloc[i] > d["sd"].iloc[i])
            sell.iloc[i] = (d["rsi"].iloc[i] > 70) and (d["sk"].iloc[i] < d["sd"].iloc[i])
        except Exception:
            buy.iloc[i] = False
            sell.iloc[i] = False
    return buy, sell

def _wide_conditions(d: pd.DataFrame):
    """
    More permissive core: useful for getting more signals while debugging.
    BUY when price > sma, rsi >= 50 and K >= D or K rising
    SELL when price < sma, rsi <= 50 and K <= D or K falling
    """
    buy = pd.Series(False, index=d.index)
    sell = pd.Series(False, index=d.index)
    for i in range(1, len(d)):
        try:
            k_now, k_prev = d["sk"].iloc[i], d["sk"].iloc[i-1]
            d_now, d_prev = d["sd"].iloc[i], d["sd"].iloc[i-1]
            k_ge_d = (k_now >= d_now)
            k_up = (k_now > k_prev)
            k_le_d = (k_now <= d_now)
            k_dn = (k_now < k_prev)

            buy.iloc[i] = (d["close"].iloc[i] > d["sma"].iloc[i]) and (d["rsi"].iloc[i] >= 50.0) and (k_ge_d or k_up)
            sell.iloc[i] = (d["close"].iloc[i] < d["sma"].iloc[i]) and (d["rsi"].iloc[i] <= 50.0) and (k_le_d or k_dn)
        except Exception:
            buy.iloc[i] = False
            sell.iloc[i] = False
    return buy, sell

# -------------------- Custom eval (hardened) --------------------

def _to_rule(x: Any) -> Dict[str, Any]:
    if isinstance(x, dict):
        return x
    if isinstance(x, str):
        try:
            j = json.loads(x)
            return j if isinstance(j, dict) else {}
        except Exception:
            return {}
    return {}

def _eval_custom(d: pd.DataFrame, custom: Dict[str, Any]):
    """
    Supports a small set of JSON rules:
      - price_above_sma (bool)
      - price_below_sma (bool)
      - rsi_gt / rsi_lt (numbers)
      - stoch_cross_up / stoch_cross_dn (bool)
    custom can be dict or JSON string containing buy_rule/sell_rule or buy/sell
    """
    if not isinstance(custom, dict):
        custom = _to_rule(custom)

    br_raw = custom.get("buy_rule") or custom.get("buy") or {}
    sr_raw = custom.get("sell_rule") or custom.get("sell") or {}
    br = _to_rule(br_raw)
    sr = _to_rule(sr_raw)

    # If no custom provided, fall back to base
    if not br and not sr:
        return _base_conditions(d)

    def build(rule: Dict[str, Any]) -> pd.Series:
        cond = pd.Series(False, index=d.index)
        if not rule:
            return cond
        for i in range(1, len(d)):
            ok = True
            try:
                if rule.get("price_above_sma"): ok = ok and (d["close"].iloc[i] > d["sma"].iloc[i])
                if rule.get("price_below_sma"): ok = ok and (d["close"].iloc[i] < d["sma"].iloc[i])
                if "rsi_gt" in rule: ok = ok and (d["rsi"].iloc[i] > float(rule["rsi_gt"]))
                if "rsi_lt" in rule: ok = ok and (d["rsi"].iloc[i] < float(rule["rsi_lt"]))
                if rule.get("stoch_cross_up"): ok = ok and _cross_up(d["sk"].iloc[i-1], d["sk"].iloc[i], d["sd"].iloc[i-1], d["sd"].iloc[i])
                if rule.get("stoch_cross_dn"): ok = ok and _cross_dn(d["sk"].iloc[i-1], d["sk"].iloc[i], d["sd"].iloc[i-1], d["sd"].iloc[i])
            except Exception:
                ok = False
            cond.iloc[i] = ok
        return cond

    buy = build(br)
    sell = build(sr)
    return buy, sell

# -------------------- Exports --------------------
# worker_inline.py expects to import:
# _ensure_cols, _prep_indicators, _base_conditions, _trend_conditions,
# _chop_conditions, _eval_custom, _expiry_to_bars, _wide_conditions

__all__ = [
    "_ensure_cols", "_prep_indicators",
    "_base_conditions", "_trend_conditions", "_chop_conditions", "_wide_conditions",
    "_eval_custom", "_expiry_to_bars"
]
