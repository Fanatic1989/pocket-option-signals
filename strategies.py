from __future__ import annotations
import math
import re
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

import pandas as pd
import numpy as np


# -------------------- Indicator helpers (local, robust) --------------------

def _sma(series: pd.Series, period: int) -> pd.Series:
    period = max(int(period or 1), 1)
    return series.rolling(period, min_periods=period).mean()

def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    period = max(int(period or 1), 1)
    delta = close.diff()
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    roll_up = pd.Series(gain).rolling(period, min_periods=period).mean()
    roll_down = pd.Series(loss).rolling(period, min_periods=period).mean()
    rs = roll_up / (roll_down.replace(0, np.nan))
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return pd.Series(rsi, index=close.index).fillna(method="bfill")

def _stoch(high: pd.Series, low: pd.Series, close: pd.Series, k: int = 14, d: int = 3) -> (pd.Series, pd.Series):
    k = max(int(k or 1), 1); d = max(int(d or 1), 1)
    ll = low.rolling(k, min_periods=k).min()
    hh = high.rolling(k, min_periods=k).max()
    raw_k = (close - ll) / (hh - ll).replace(0, np.nan) * 100.0
    k_line = raw_k.rolling(d, min_periods=1).mean()
    d_line = k_line.rolling(d, min_periods=1).mean()
    return k_line.fillna(50.0), d_line.fillna(50.0)

def _slope(series: pd.Series, lookback: int = 5) -> pd.Series:
    lookback = max(int(lookback or 1), 1)
    # simple slope over lookback using linear regression on index
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
    # timeframe to seconds
    tf_sec_map = {
        "M1":60,"M2":120,"M3":180,"M5":300,"M10":600,"M15":900,"M30":1800,"H1":3600,"H4":14400,"D1":86400
    }
    tf_sec = tf_sec_map.get(tf, 300)

    # expiry string to seconds
    m = re.fullmatch(r"(\d+)\s*([MH]?)", expiry.replace("MIN","M").replace("HR","H").replace("HRS","H"))
    if not m:
        # try canonical forms like 1M / 3M / 5M / 30M / H1 / H4
        if expiry.endswith("M"):
            secs = int(expiry[:-1]) * 60
        elif expiry.startswith("H"):
            secs = int(expiry[1:]) * 3600
        else:
            secs = 300
    else:
        n = int(m.group(1))
        unit = m.group(2) or "M"
        secs = n * (60 if unit == "M" else 3600)
    bars = max(int(round(secs / tf_sec)), 1)
    return bars


# -------------------- Backtest Container --------------------

@dataclass
class BTResult:
    trades: int = 0
    wins: int = 0
    losses: int = 0
    draws: int = 0
    rows: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def winrate(self) -> float:
        return (self.wins / self.trades) if self.trades else 0.0


# -------------------- Core signal builders --------------------

def _ensure_cols(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d.columns = [c.strip().lower() for c in d.columns]
    if "timestamp" not in d.columns:
        # synth ts if missing
        d["timestamp"] = range(len(d))
    # coerce OHLC
    for c in ("open","high","low","close"):
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d.dropna(subset=["close"]).reset_index(drop=True)
    return d

def _prep_indicators(d: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
    sma_period = int(params.get("sma_period", 50))
    rsi_period = int(params.get("rsi_period", 14))
    st_k = int(params.get("stoch_k", 14))
    st_d = int(params.get("stoch_d", 3))
    d["sma"] = _sma(d["close"], sma_period)
    d["rsi"] = _rsi(d["close"], rsi_period)
    d["sk"], d["sd"] = _stoch(d["high"], d["low"], d["close"], st_k, st_d)
    d["sma_slope"] = _slope(d["sma"], lookback=5)
    return d

def _cross_up(a_prev, a_now, b_prev, b_now) -> bool:
    try:
        return a_prev < b_prev and a_now > b_now
    except Exception:
        return False

def _cross_dn(a_prev, a_now, b_prev, b_now) -> bool:
    try:
        return a_prev > b_prev and a_now < b_now
    except Exception:
        return False


# -------------------- Strategy conditions --------------------

def _base_conditions(d: pd.DataFrame) -> pd.Series:
    # Buy: price above SMA, RSI > 55, Stoch K cross above D
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

def _trend_conditions(d: pd.DataFrame) -> pd.Series:
    # Trend: SMA slope > 0, pullback that respects SMA and RSI bounces from 50 → buy.
    buy = pd.Series(False, index=d.index)
    sell = pd.Series(False, index=d.index)
    for i in range(2, len(d)):
        # Uptrend buy
        buy.iloc[i] = (
            (d["sma_slope"].iloc[i] > 0) and
            (d["close"].iloc[i-1] <= d["sma"].iloc[i-1]) and
            (d["close"].iloc[i]   >  d["sma"].iloc[i]) and
            (d["rsi"].iloc[i-1] <= 50) and (d["rsi"].iloc[i] > 50)
        )
        # Downtrend sell
        sell.iloc[i] = (
            (d["sma_slope"].iloc[i] < 0) and
            (d["close"].iloc[i-1] >= d["sma"].iloc[i-1]) and
            (d["close"].iloc[i]   <  d["sma"].iloc[i]) and
            (d["rsi"].iloc[i-1] >= 50) and (d["rsi"].iloc[i] < 50)
        )
    return buy, sell

def _chop_conditions(d: pd.DataFrame) -> pd.Series:
    # Mean reversion: RSI < 30 → buy, RSI > 70 → sell, with mild stoch confirmation
    buy = pd.Series(False, index=d.index)
    sell = pd.Series(False, index=d.index)
    for i in range(1, len(d)):
        buy.iloc[i] = (d["rsi"].iloc[i] < 30) and (d["sk"].iloc[i] > d["sd"].iloc[i])
        sell.iloc[i] = (d["rsi"].iloc[i] > 70) and (d["sk"].iloc[i] < d["sd"].iloc[i])
    return buy, sell


# -------------------- Custom rule evaluation --------------------

def _eval_custom(d: pd.DataFrame, custom: Dict[str, Any]) -> (pd.Series, pd.Series):
    """
    Very simple engine: expects dict rules from your natural-language parser,
    with keys under 'buy'/'sell' arrays OR empty → falls back to BASE.
    Rule keys supported (boolean AND):
      price_above_sma / price_below_sma
      rsi_gt / rsi_lt (number)
      stoch_cross_up / stoch_cross_dn (bool)
    """
    if not custom:
        return _base_conditions(d)

    br = (custom.get("buy_rule") or custom.get("buy") or {})
    sr = (custom.get("sell_rule") or custom.get("sell") or {})
    if not br and not sr:
        return _base_conditions(d)

    def build(rule: Dict[str, Any]) -> pd.Series:
        cond = pd.Series(True, index=d.index)
        if not rule:
            return pd.Series(False, index=d.index)
        for i in range(1, len(d)):
            ok = True
            if rule.get("price_above_sma"): ok = ok and (d["close"].iloc[i] > d["sma"].iloc[i])
            if rule.get("price_below_sma"): ok = ok and (d["close"].iloc[i] < d["sma"].iloc[i])
            if "rsi_gt" in rule:           ok = ok and (d["rsi"].iloc[i] > float(rule["rsi_gt"]))
            if "rsi_lt" in rule:           ok = ok and (d["rsi"].iloc[i] < float(rule["rsi_lt"]))
            if rule.get("stoch_cross_up"): ok = ok and _cross_up(d["sk"].iloc[i-1], d["sk"].iloc[i], d["sd"].iloc[i-1], d["sd"].iloc[i])
            if rule.get("stoch_cross_dn"): ok = ok and _cross_dn(d["sk"].iloc[i-1], d["sk"].iloc[i], d["sd"].iloc[i-1], d["sd"].iloc[i])
            cond.iloc[i] = ok
        return cond

    buy = build(br)
    sell = build(sr)
    return buy, sell


# -------------------- Backtest core (binary) --------------------

def run_backtest_core_binary(
    df: pd.DataFrame,
    core: str,
    cfg: Dict[str, Any],
    tf: str,
    expiry: str,
) -> BTResult:
    """
    Simple binary options backtest:
      - Enter on the bar AFTER a condition becomes true.
      - Exit exactly N bars later (N from expiry/timeframe).
      - Outcome: compare exit close vs entry close (ties count as LOSS).
    """
    d = _ensure_cols(df)
    # Minimal guard
    if len(d) < 30:
        return BTResult()

    # Indicator params (can be overridden from cfg['indicators'], but we use defaults regardless of toggles)
    inds_cfg = (cfg or {}).get("indicators", {})
    ip = {
        "sma_period": int(inds_cfg.get("sma", {}).get("period", 50)),
        "rsi_period": int(inds_cfg.get("rsi", {}).get("period", 14)),
        "stoch_k":    int(inds_cfg.get("stoch", {}).get("k", 14)),
        "stoch_d":    int(inds_cfg.get("stoch", {}).get("d", 3)),
    }
    d = _prep_indicators(d, ip)

    bars = _expiry_to_bars(tf, expiry)
    result = BTResult()

    core = (core or "BASE").upper()
    if core == "TREND":
        buy_sig, sell_sig = _trend_conditions(d)
    elif core == "CHOP":
        buy_sig, sell_sig = _chop_conditions(d)
    elif core == "CUSTOM":
        custom = (cfg or {}).get("custom", {}) or {}
        buy_sig, sell_sig = _eval_custom(d, custom)
    else:
        buy_sig, sell_sig = _base_conditions(d)

    for i in range(1, len(d) - bars):
        # Entry on next candle when signal is true at i
        if buy_sig.iloc[i]:
            entry_idx = i + 1
            exit_idx  = entry_idx + bars
            if exit_idx >= len(d): break
            entry = float(d["close"].iloc[entry_idx])
            exitp = float(d["close"].iloc[exit_idx])
            outcome = "WIN" if exitp > entry else "LOSS" if exitp < entry else "LOSS"
            result.trades += 1
            if outcome == "WIN": result.wins += 1
            elif outcome == "LOSS": result.losses += 1
            else: result.draws += 1
            result.rows.append({
                "time_in":  d["timestamp"].iloc[entry_idx],
                "dir":      "BUY",
                "entry":    entry,
                "time_out": d["timestamp"].iloc[exit_idx],
                "exit":     exitp,
                "outcome":  outcome
            })
        if sell_sig.iloc[i]:
            entry_idx = i + 1
            exit_idx  = entry_idx + bars
            if exit_idx >= len(d): break
            entry = float(d["close"].iloc[entry_idx])
            exitp = float(d["close"].iloc[exit_idx])
            outcome = "WIN" if exitp < entry else "LOSS" if exitp > entry else "LOSS"
            result.trades += 1
            if outcome == "WIN": result.wins += 1
            elif outcome == "LOSS": result.losses += 1
            else: result.draws += 1
            result.rows.append({
                "time_in":  d["timestamp"].iloc[entry_idx],
                "dir":      "SELL",
                "entry":    entry,
                "time_out": d["timestamp"].iloc[exit_idx],
                "exit":     exitp,
                "outcome":  outcome
            })

    return result
