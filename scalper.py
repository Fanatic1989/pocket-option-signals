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
    df["ema20_5m"] = pd.NA  # caller can fill for multi-timeframe confirm
    df["ema50_5m"] = pd.NA
    df["adx"]   = ta.adx(high=df["high"], low=df["low"], close=df["close"], length=14)["ADX_14"]
    df["atr"]   = ta.atr(high=df["high"], low=df["low"], close=df["close"], length=14)
    # ATR percentile over last 200 bars (needs enough history)
    df["atr_pctl"] = df["atr"].rolling(200, min_periods=50).apply(
        lambda x: 100.0 * (x.rank(pct=True).iloc[-1] if len(x.dropna()) else 0), raw=False
    )
    return df.dropna()

def classify(prev, cur,
             rsi_buy=48, rsi_sell=52,
             adx_min=22, atr_pctl_min=40,
             mtf_ema20=None, mtf_ema50=None,
             min_score=3):
    """
    Returns: (side, score, why)
    Scoring:
      +1 Trend (EMA50 vs EMA200)
      +1 MACD cross in trend direction
      +1 RSI cross (48/52 default)
      +1 ADX >= adx_min
      +1 ATR percentile >= atr_pctl_min
      +1 MTF 5m confirm (ema20 > ema50 for BUY, < for SELL) if provided
      +1 Price location (above/below ema50) & candle color agrees
    """
    score, why, side = 0, [], None

    uptrend = cur["ema50"] > cur["ema200"]
    why.append("EMA50>EMA200" if uptrend else "EMA50<EMA200")
    if uptrend: score += 1

    # MACD crosses
    macd_up   = (prev["macd"] <= prev["macds"]) and (cur["macd"] > cur["macds"])
    macd_down = (prev["macd"] >= prev["macds"]) and (cur["macd"] < cur["macds"])
    if macd_up:   why.append("MACD↑"); score += 1
    if macd_down: why.append("MACD↓"); score += 1

    # RSI timing (mid-zone cross)
    rsi_up   = (prev["rsi"] <= rsi_buy)  and (cur["rsi"] > rsi_buy)
    rsi_down = (prev["rsi"] >= rsi_sell) and (cur["rsi"] < rsi_sell)
    if rsi_up:   why.append(f"RSI↗>{rsi_buy}"); score += 1
    if rsi_down: why.append(f"RSI↘<{rsi_sell}"); score += 1

    # Strength & volatility filters
    if cur["adx"] >= adx_min:
        why.append(f"ADX≥{adx_min}"); score += 1
    if cur["atr_pctl"] >= atr_pctl_min:
        why.append(f"ATR%≥{atr_pctl_min}"); score += 1

    # Price location + candle color
    candle_up = cur["close"] > cur["open"]
    candle_dn = cur["close"] < cur["open"]
    if uptrend and cur["close"] > cur["ema50"] and candle_up:
        why.append("Price>EMA50 & green"); score += 1
    if (not uptrend) and cur["close"] < cur["ema50"] and candle_dn:
        why.append("Price<EMA50 & red"); score += 1

    # Side decision
    if uptrend and (macd_up or rsi_up):
        side = "BUY"
    elif (not uptrend) and (macd_down or rsi_down):
        side = "SELL"

    # Optional MTF confirm
    if pd.notna(mtf_ema20) and pd.notna(mtf_ema50):
        if side == "BUY"  and (mtf_ema20 > mtf_ema50):
            why.append("MTF(5m)↑"); score += 1
        if side == "SELL" and (mtf_ema20 < mtf_ema50):
            why.append("MTF(5m)↓"); score += 1

    return (side, score, ", ".join(why))
