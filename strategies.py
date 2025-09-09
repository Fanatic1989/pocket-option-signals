from dataclasses import dataclass, field
from typing import List, Dict, Any
import pandas as pd
import pandas_ta as ta
import numpy as np

@dataclass
class BTResult:
    trades: int = 0
    wins: int = 0
    losses: int = 0
    draws: int = 0
    winrate: float = 0.0
    rows: List[dict] = field(default_factory=list)

def _expiry_to_bars(expiry_label: str, tf: str) -> int:
    # expiry like "1m","3m","5m","30m","H1","H4"
    tf_map = {"M1":1,"M2":2,"M3":3,"M5":5,"M10":10,"M15":15,"M30":30,"H1":60,"H4":240,"D1":1440}
    e_map = {"1m":1,"3m":3,"5m":5,"30m":30,"h1":60,"h4":240}
    tf_min = tf_map.get(tf.upper(), 5)
    e = e_map.get(expiry_label.lower(), 5)
    return max(1, int(round(e / tf_min)))

def _ensure_indicators(df: pd.DataFrame, cfg: Dict[str,Any]) -> pd.DataFrame:
    out = df.copy()

    # --- Moving Averages (period driven) ---
    period_sma = int(cfg["indicators"].get("sma",{}).get("period",50))
    out["sma50"] = ta.sma(out["close"], length=period_sma)

    if cfg["indicators"].get("ema",{}).get("enabled"):
        out["ema"] = ta.ema(out["close"], length=int(cfg["indicators"]["ema"].get("period",21)))
    if cfg["indicators"].get("wma",{}).get("enabled"):
        out["wma"] = ta.wma(out["close"], length=int(cfg["indicators"]["wma"].get("period",20)))
    if cfg["indicators"].get("smma",{}).get("enabled"):
        # SMMA (RMA) approximation using ta.rma
        out["smma"] = ta.rma(out["close"], length=int(cfg["indicators"]["smma"].get("period",20)))
    if cfg["indicators"].get("tma",{}).get("enabled"):
        # TMA approximation via 2x SMA
        l = int(cfg["indicators"]["tma"].get("period",20))
        out["tma"] = ta.sma(ta.sma(out["close"], length=l), length=l)

    # --- RSI ---
    out["rsi"] = ta.rsi(out["close"], length=int(cfg["indicators"].get("rsi",{}).get("period",14)))

    # --- Bollinger Bands ---
    bb_len = int(cfg["indicators"].get("bb",{}).get("period",20))
    bb_std = float(cfg["indicators"].get("bb",{}).get("std",2.0))
    bb = ta.bbands(out["close"], length=bb_len, std=bb_std)
    if bb is not None and isinstance(bb, pd.DataFrame):
        out["bb_low"] = bb.iloc[:,0]
        out["bb_mid"] = bb.iloc[:,1]
        out["bb_up"] = bb.iloc[:,2]
    else:
        out["bb_low"]=np.nan; out["bb_mid"]=np.nan; out["bb_up"]=np.nan

    # --- Stochastic ---
    st = ta.stoch(high=out["high"], low=out["low"], close=out["close"],
                  k=int(cfg["indicators"].get("stoch",{}).get("k",14)),
                  d=int(cfg["indicators"].get("stoch",{}).get("d",3)),
                  smooth_k=int(cfg["indicators"].get("stoch",{}).get("smooth_k",3)))
    if st is not None and isinstance(st, pd.DataFrame):
        out["stoch_k"] = st.iloc[:,0]; out["stoch_d"]=st.iloc[:,1]
    else:
        out["stoch_k"]=np.nan; out["stoch_d"]=np.nan

    # (Optional) more indicators could be added here as enabled
    return out

def _signal_base(row_prev, row):
    # Example BASE logic: trend-following around SMA(midline) + RSI
    if row["close"] > row["sma50"] and row["rsi"] > 50: return "BUY"
    if row["close"] < row["sma50"] and row["rsi"] < 50: return "SELL"
    return ""

def _signal_trend(row_prev, row):
    # Stricter momentum flavor
    if row["close"] > row["sma50"] and row["rsi"] > 52 and row_prev["rsi"] < row["rsi"]:
        return "BUY"
    if row["close"] < row["sma50"] and row["rsi"] < 48 and row_prev["rsi"] > row["rsi"]:
        return "SELL"
    return ""

def _signal_chop(row_prev, row):
    # Mean-reversion off the bands back to midline
    if not np.isnan(row["bb_low"]) and row["close"] < row["bb_low"] and row["close"] > row["bb_mid"]:
        return "BUY"
    if not np.isnan(row["bb_up"]) and row["close"] > row["bb_up"] and row["close"] < row["bb_mid"]:
        return "SELL"
    return ""

def _signal_custom(cfg_custom, row_prev, row):
    # Interprets dict produced by rules.parse_natural_rule
    def match(rule: dict, up: bool):
        if not rule: return False
        ok = True
        if "sma_respect" in rule:
            sma = row.get("sma50")
            if not np.isnan(sma):
                if up:   ok &= row["close"] >= sma
                else:    ok &= row["close"] <= sma
        if "rsi_bounce" in rule:
            if rule["rsi_bounce"] == "up":
                ok &= row_prev["rsi"] <= 50 and row["rsi"] > 50
            elif rule["rsi_bounce"] == "down":
                ok &= row_prev["rsi"] >= 50 and row["rsi"] < 50
        if "stoch_cross" in rule:
            if rule["stoch_cross"] == "up":
                ok &= row_prev["stoch_k"] <= row_prev["stoch_d"] and row["stoch_k"] > row["stoch_d"]
            elif rule["stoch_cross"] == "down":
                ok &= row_prev["stoch_k"] >= row_prev["stoch_d"] and row["stoch_k"] < row["stoch_d"]
        return ok

    buy_rule = cfg_custom.get("buy_rule_dict") or {}
    sell_rule = cfg_custom.get("sell_rule_dict") or {}
    if match(buy_rule, True): return "BUY"
    if match(sell_rule, False): return "SELL"
    return ""

def _prepare_custom_dicts(cfg: dict):
    c = cfg.get("custom", {})
    def ensure_dict(x): return x if isinstance(x, dict) else {}
    c["buy_rule_dict"] = ensure_dict(c.get("buy_rule"))
    c["sell_rule_dict"] = ensure_dict(c.get("sell_rule"))
    return c

def run_backtest_core_binary(df: pd.DataFrame, strategy: str, cfg: Dict[str,Any], tf: str, expiry_label: str) -> BTResult:
    if df is None or df.empty:
        return BTResult()

    data = df.copy()
    for col in ("open","high","low","close"):
        if col not in data.columns:
            raise ValueError("CSV must include open, high, low, close, timestamp")

    data = data.sort_values("timestamp").reset_index(drop=True)
    data = _ensure_indicators(data, cfg)
    bars_ahead = _expiry_to_bars(expiry_label, tf)

    rows = []
    wins=losses=draws=trades=0

    strat = (strategy or "BASE").upper()
    custom_cfg = _prepare_custom_dicts(cfg)

    for i in range(50, len(data)-bars_ahead):
        prev = data.iloc[i-1]
        cur  = data.iloc[i]
        sig = ""
        if strat == "BASE":
            sig = _signal_base(prev, cur)
        elif strat == "TREND":
            sig = _signal_trend(prev, cur)
        elif strat == "CHOP":
            sig = _signal_chop(prev, cur)
        elif strat == "CUSTOM":
            sig = _signal_custom(custom_cfg, prev, cur)

        if not sig:
            continue

        entry = float(cur["close"])
        out = data.iloc[i+bars_ahead]
        exitp = float(out["close"])

        outcome = "DRAW"
        if sig == "BUY":
            if exitp > entry: outcome="WIN"
            elif exitp < entry: outcome="LOSS"
        else:
            if exitp < entry: outcome="WIN"
            elif exitp > entry: outcome="LOSS"

        trades += 1
        if outcome=="WIN": wins+=1
        elif outcome=="LOSS": losses+=1
        else: draws+=1

        rows.append({
            "idx": int(i),
            "time_in": str(cur.get("timestamp")),
            "dir": sig,
            "entry": entry,
            "time_out": str(out.get("timestamp")),
            "exit": exitp,
            "outcome": outcome
        })

    winrate = (wins/ trades) if trades else 0.0
    if len(rows) > 20:
        rows = rows[-20:]  # small preview

    return BTResult(trades=trades, wins=wins, losses=losses, draws=draws, winrate=winrate, rows=rows)
