import pandas as pd
import pandas_ta as ta

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    # requires columns: open, high, low, close
    macd = ta.macd(df["close"], fast=12, slow=26, signal=9)
    df["macd"]  = macd["MACD_12_26_9"]
    df["macds"] = macd["MACDs_12_26_9"]
    df["rsi"]   = ta.rsi(df["close"], length=14)
    df["ema50"] = ta.ema(df["close"], length=50)
    df["ema200"]= ta.ema(df["close"], length=200)
    return df.dropna()

def classify(prev, cur, rsi_buy=50, rsi_sell=50, min_score=1):
    score, reasons, side = 0, [], None
    uptrend = cur["ema50"] > cur["ema200"]
    if uptrend: reasons.append("EMA50>EMA200")
    else:       reasons.append("EMA50<EMA200")

    macd_up   = (prev["macd"] <= prev["macds"]) and (cur["macd"] > cur["macds"])
    macd_down = (prev["macd"] >= prev["macds"]) and (cur["macd"] < cur["macds"])

    if macd_up:   reasons.append("MACD↑"); score += 1
    if macd_down: reasons.append("MACD↓"); score += 1

    if cur["rsi"] >= rsi_sell and macd_down:
        side = "SELL"
        reasons.append(f"RSI≥{rsi_sell}")
    elif cur["rsi"] <= rsi_buy and macd_up:
        side = "BUY"
        reasons.append(f"RSI≤{rsi_buy}")

    return side, score, ", ".join(reasons)
