# strategies.py
import pandas as pd
from indicators import apply_indicators
from rules import eval_rule

TF_MINUTES = {"M1":1, "M2":2, "M3":3, "M5":5, "M10":10, "M15":15, "M30":30, "H1":60, "H4":240, "D1":1440}
EXPIRY_MINUTES = {"1m":1, "3m":3, "5m":5, "30m":30, "H1":60, "H4":240}

def expiry_bars(tf: str, expiry_label: str) -> int:
    tf_m = TF_MINUTES.get(tf.upper(), 1)
    ex_m = EXPIRY_MINUTES.get(expiry_label.upper(), 5)
    return max(1, int(round(ex_m / tf_m)))

def first_enabled_ma_column(df: pd.DataFrame, cfg):
    order = [("EMA","ema"), ("SMA","sma"), ("WMA","wma"), ("SMMA","smma"), ("TMA","tma"), ("MA","ma")]
    for key, col in order:
        if cfg["indicators"].get(key,{}).get("enabled", False) and col in df.columns:
            return col
    return None

def strat_BASE(df: pd.DataFrame, cfg):
    out = []
    macol = first_enabled_ma_column(df, cfg)
    if not macol: return out
    price, ma = df["close"], df[macol]
    cross_up = (price > ma) & (price.shift(1) <= ma.shift(1))
    cross_dn = (price < ma) & (price.shift(1) >= ma.shift(1))
    for i in range(len(df)):
        if bool(getattr(cross_up, "iloc", cross_up)[i]): out.append((i,"BUY"))
        elif bool(getattr(cross_dn, "iloc", cross_dn)[i]): out.append((i,"SELL"))
    return out

def strat_TREND(df: pd.DataFrame, cfg):
    out = []
    macol = first_enabled_ma_column(df, cfg)
    if not macol: return out
    ind = cfg["indicators"]
    adx_thr = int(ind.get("ADX",{}).get("threshold",20)) if ind.get("ADX",{}).get("enabled",False) else 0
    for i in range(2, len(df)):
        ma_now, ma_prev = df[macol].iloc[i], df[macol].iloc[i-1]
        price = df["close"].iloc[i]
        rising = pd.notna(ma_now) and pd.notna(ma_prev) and (ma_now > ma_prev)
        falling= pd.notna(ma_now) and pd.notna(ma_prev) and (ma_now < ma_prev)
        cond_adx = True
        if ind.get("ADX",{}).get("enabled",False) and "adx" in df:
            val = df["adx"].iloc[i]; cond_adx = pd.notna(val) and (val >= adx_thr)
        if rising and price > ma_now and cond_adx:  out.append((i,"BUY"))
        if falling and price < ma_now and cond_adx: out.append((i,"SELL"))
    return out

def strat_CHOP(df: pd.DataFrame, cfg):
    out = []
    for i in range(2, len(df)):
        ref = df["bb_mid"].iloc[i] if "bb_mid" in df and pd.notna(df["bb_mid"].iloc[i]) else None
        if ref is None:
            macol = first_enabled_ma_column(df, cfg)
            if macol and pd.notna(df[macol].iloc[i]): ref = df[macol].iloc[i]
        if ref is None: continue
        price, prev = df["close"].iloc[i], df["close"].iloc[i-1]
        if price > ref and prev <= ref: out.append((i,"SELL"))
        elif price < ref and prev >= ref: out.append((i,"BUY"))
    return out

def strat_CUSTOM(df: pd.DataFrame, cfg):
    out = []
    c = cfg.get("custom", {})
    if not c.get("enabled"): return out
    buy_expr = c.get("buy_rule","").strip()
    sell_expr = c.get("sell_rule","").strip()
    if not buy_expr and not sell_expr: return out
    for i in range(1, len(df)):
        did = None
        if buy_expr and eval_rule(buy_expr, df, i, c): did = "BUY"
        if sell_expr and eval_rule(sell_expr, df, i, c): did = "SELL" if did is None else did
        if did: out.append((i, did))
    return out

STRATEGY_FUNCS = {"BASE":strat_BASE, "TREND":strat_TREND, "CHOP":strat_CHOP, "CUSTOM":strat_CUSTOM}

def run_backtest_core_binary(df: pd.DataFrame, strategy: str, cfg, tf: str, expiry_label: str):
    df = df.copy(); df.columns = [c.lower() for c in df.columns]
    if "timestamp" in df.columns:
        try: df["timestamp"] = pd.to_datetime(df["timestamp"])
        except: pass
    df = apply_indicators(df, cfg)
    sigs = STRATEGY_FUNCS[strategy](df, cfg)

    bars = expiry_bars(tf, expiry_label)
    wins = losses = draws = 0
    rows = []

    for (i, direction) in sigs:
        if i + bars >= len(df): break
        entry = float(df["close"].iloc[i])
        exit_price = float(df["close"].iloc[i + bars])
        if abs(exit_price - entry) < 1e-12:
            outcome = "DRAW"; draws += 1
        elif direction == "BUY":
            outcome = "WIN" if exit_price > entry else "LOSS"
            if outcome == "WIN": wins += 1
            else: losses += 1
        else:
            outcome = "WIN" if exit_price < entry else "LOSS"
            if outcome == "WIN": wins += 1
            else: losses += 1

        rows.append({
            "idx": i,
            "time_in": str(df["timestamp"].iloc[i]) if "timestamp" in df.columns else "",
            "dir": direction,
            "entry": entry,
            "time_out": str(df["timestamp"].iloc[i + bars]) if "timestamp" in df.columns else f"+{bars} bars",
            "exit": exit_price,
            "outcome": outcome
        })

    trades = wins + losses + draws
    winrate = (wins / trades) if trades > 0 else 0.0
    return {"trades": trades, "wins": wins, "losses": losses, "draws": draws, "winrate": winrate, "rows": rows[-20:]}
