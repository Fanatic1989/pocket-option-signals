import pandas as pd
import pandas_ta as ta

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    macd = ta.macd(df["close"], fast=12, slow=26, signal=9)
    df["macd"]  = macd["MACD_12_26_9"]
    df["macds"] = macd["MACDs_12_26_9"]
    df["rsi"]   = ta.rsi(df["close"], length=14)
    df["ema50"] = ta.ema(df["close"], length=50)
    df["ema200"]= ta.ema(df["close"], length=200)
    adx = ta.adx(high=df["high"], low=df["low"], close=df["close"], length=14)
    df["adx"]   = adx["ADX_14"]
    return df.dropna()

def classify(prev, cur,
             rsi_buy=48, rsi_sell=52,
             adx_min=20,
             min_score=3):
    """
    Score:
      +1 Trend (ema50 vs ema200)
      +1 MACD cross in trend direction
      +1 RSI side (<=48 for BUY, >=52 for SELL)
      +1 ADX >= adx_min
    """
    score, why, side = 0, [], None

    uptrend = cur["ema50"] > cur["ema200"]
    why.append("EMA50>EMA200" if uptrend else "EMA50<EMA200")
    score += 1

    macd_up   = (prev["macd"] <= prev["macds"]) and (cur["macd"] > cur["macds"])
    macd_down = (prev["macd"] >= prev["macds"]) and (cur["macd"] < cur["macds"])
    if uptrend and macd_up:
        score += 1; why.append("MACD↑")
    if (not uptrend) and macd_down:
        score += 1; why.append("MACD↓")

    if uptrend and macd_up and cur["rsi"] <= rsi_buy:
        side = "BUY"; why.append(f"RSI≤{rsi_buy}")
    if (not uptrend) and macd_down and cur["rsi"] >= rsi_sell:
        side = "SELL"; why.append(f"RSI≥{rsi_sell}")

    if cur["adx"] >= adx_min:
        score += 1; why.append(f"ADX≥{adx_min}")

    if side and score >= min_score:
        return side, score, ", ".join(why)
    return None, score, ", ".join(why)
