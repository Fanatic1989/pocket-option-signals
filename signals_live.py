import os, requests
from datetime import datetime, timezone
import pandas as pd
import pandas_ta as ta
import yaml

# --- ENV ---
INTERVAL = os.getenv("INTERVAL","1m")
EXPIRY   = int(os.getenv("EXPIRY_MIN","5"))
MIN_SCORE = int(os.getenv("MIN_SCORE","3"))

OANDA_API_KEY = os.getenv("OANDA_API_KEY","")
OANDA_ENV     = os.getenv("OANDA_ENV","practice").lower()
HOST = "https://api-fxtrade.oanda.com" if OANDA_ENV=="live" else "https://api-fxpractice.oanda.com"
HDR  = {"Authorization": f"Bearer {OANDA_API_KEY}"} if OANDA_API_KEY else {}

def oanda_fetch_m1(instr:str, count:int=220)->pd.DataFrame:
    r = requests.get(f"{HOST}/v3/instruments/{instr}/candles",
                     params={"granularity":"M1","count":count,"price":"M"}, headers=HDR, timeout=20)
    r.raise_for_status()
    js = r.json()
    rows=[]
    for c in js.get("candles",[]):
        if not c.get("complete"): continue
        t = c["time"]
        o = float(c["mid"]["o"]); h=float(c["mid"]["h"]); l=float(c["mid"]["l"]); cl=float(c["mid"]["c"])
        rows.append((t,o,h,l,cl))
    df = pd.DataFrame(rows, columns=["time","open","high","low","close"])
    return df

def add_indicators(df:pd.DataFrame)->pd.DataFrame:
    macd = ta.macd(df["close"], fast=12, slow=26, signal=9)
    df["macd"]  = macd["MACD_12_26_9"]
    df["macds"] = macd["MACDs_12_26_9"]
    df["rsi"]   = ta.rsi(df["close"], length=14)
    df["ema50"] = ta.ema(df["close"], length=50)
    df["ema200"]= ta.ema(df["close"], length=200)
    return df.dropna()

def classify(prev, cur, rsi_buy=48, rsi_sell=52):
    score, why, side = 0, [], None
    if cur["ema50"] > cur["ema200"]: why.append("EMA50>EMA200"); trend="UP"
    else:                            why.append("EMA50<EMA200"); trend="DN"
    macd_up   = (prev["macd"] <= prev["macds"]) and (cur["macd"] > cur["macds"])
    macd_down = (prev["macd"] >= prev["macds"]) and (cur["macd"] < cur["macds"])
    if macd_up:   why.append("MACD↑"); score += 1
    if macd_down: why.append("MACD↓"); score += 1
    if cur["rsi"] <= rsi_buy and macd_up and trend=="UP":
        side = "BUY";  why.append(f"RSI≤{rsi_buy}"); score += 1
    if cur["rsi"] >= rsi_sell and macd_down and trend=="DN":
        side = "SELL"; why.append(f"RSI≥{rsi_sell}"); score += 1
    # price location
    if cur["close"] > cur["ema50"] and side=="BUY":  why.append("Price>EMA50 & green"); score += 1
    if cur["close"] < cur["ema50"] and side=="SELL": why.append("Price<EMA50 & red");  score += 1
    return side, score, ", ".join(why)

def within_hours(symbol_group:str)->bool:
    now = datetime.now(timezone.utc)
    wd = now.isoweekday()  # 1..7 (Mon..Sun)
    def ok(block, weekdays):
        hh = os.getenv(block, "")
        wd_ok = str(wd) in os.getenv(weekdays, "12345")
        if not hh: return True
        try:
            start,end = hh.split("-")
            start=(int(start[:2]), int(start[2:])); end=(int(end[:2]), int(end[2:]))
            cur = (now.hour, now.minute)
            return wd_ok and (start <= cur <= end)
        except: return wd_ok
    g = (symbol_group or "FX").upper()
    if g=="FX":        return ok("FX_HOURS_UTC","WEEKDAYS_FX")
    if g=="METAL":     return ok("COMMODITY_HOURS_UTC","WEEKDAYS_COMMODITY")
    if g in ("INDEX","STOCK"): 
        key = "INDEX_HOURS_UTC" if g=="INDEX" else "STOCK_HOURS_UTC"
        days= "WEEKDAYS_INDEX" if g=="INDEX" else "WEEKDAYS_STOCK"
        return ok(key, days)
    return True

def main()->list[str]:
    now = datetime.now(timezone.utc)
    lines = [
        f"📡 Pocket Option Signals — {now:%Y-%m-%d %H:%M UTC}",
        f"Candle: {INTERVAL} | Expiry: {EXPIRY}m",
        ""
    ]
    # load universe
    cfg = yaml.safe_load(open("symbols.yaml"))
    for s in cfg.get("symbols", []):
        if not s.get("enabled", True): continue
        inst = s["oanda"]; pretty = s.get("name", inst.replace("_","/")); group = s.get("group","FX")
        if not within_hours(group): continue
        try:
            df = oanda_fetch_m1(inst, 220)
            df = add_indicators(df)
            if len(df) < 2: 
                continue
            prev, cur = df.iloc[-2], df.iloc[-1]
            side, score, why = classify(prev, cur)
            if side and score >= MIN_SCORE:
                emoji = "🟢" if side=="BUY" else "🔴"
                price = f"{cur['close']:.5f}" if "/" in pretty else f"{cur['close']:.3f}"
                lines.append(f"{emoji} {pretty} — {side} @ {price}")
                lines.append(f"• {why} (score {score})")
                lines.append("")
        except Exception as e:
            # log locally, but do not spam chats
            print(f"⚠️ {pretty} — error: {e}")
            continue
    return lines
