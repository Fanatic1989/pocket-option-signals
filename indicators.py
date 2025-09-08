# indicators.py
import pandas as pd
import pandas_ta as ta

INDICATOR_SPECS = {
    "EMA":   {"name":"EMA",  "params":{"period":20}, "fields":["ema"]},
    "SMA":   {"name":"SMA",  "params":{"period":20}, "fields":["sma"]},
    "WMA":   {"name":"WMA",  "params":{"period":20}, "fields":["wma"]},
    "SMMA":  {"name":"SMMA (RMA)", "params":{"period":20}, "fields":["smma"]},
    "TMA":   {"name":"TMA (Triangular)", "params":{"period":20}, "fields":["tma"]},
    "MA":    {"name":"Moving Average", "params":{"period":20}, "fields":["ma"]},

    "RSI":   {"name":"RSI", "params":{"length":14, "overbought":70, "oversold":30}, "fields":["rsi"]},
    "STOCH": {"name":"Stochastic Oscillator", "params":{"k":14, "d":3, "smooth_k":3}, "fields":["stoch_k","stoch_d"]},
    "MACD":  {"name":"MACD", "params":{"fast":12, "slow":26, "signal":9}, "fields":["macd","macd_signal","macd_hist"]},
    "OSMA":  {"name":"OsMA", "params":{"fast":12, "slow":26, "signal":9}, "fields":["osma"]},
    "AO":    {"name":"Awesome Oscillator", "params":{"fast":5, "slow":34}, "fields":["ao"]},
    "AC":    {"name":"Accelerator Oscillator", "params":{"fast":5, "slow":34, "sma":5}, "fields":["ac"]},
    "MOM":   {"name":"Momentum", "params":{"length":10}, "fields":["mom"]},
    "ROC":   {"name":"Rate of Change", "params":{"length":10}, "fields":["roc"]},
    "WILLR": {"name":"Williams %R", "params":{"length":14}, "fields":["willr"]},
    "CCI":   {"name":"CCI", "params":{"length":20}, "fields":["cci"]},
    "STC":   {"name":"Schaff Trend Cycle", "params":{"tclen":10,"fast":23,"slow":50,"factor":0.5}, "fields":["stc"]},

    "ADX":   {"name":"ADX", "params":{"length":14, "threshold":20}, "fields":["adx"]},
    "ATR":   {"name":"Average True Range", "params":{"length":14}, "fields":["atr"]},
    "AROON": {"name":"Aroon", "params":{"length":25}, "fields":["aroon_up","aroon_down","aroon_osc"]},
    "VORTEX":{"name":"Vortex", "params":{"length":14}, "fields":["vortex_pos","vortex_neg"]},
    "SUPERTREND":{"name":"SuperTrend", "params":{"length":10, "multiplier":3.0}, "fields":["supertrend_dir","supertrend_upper","supertrend_lower"]},

    "BB":    {"name":"Bollinger Bands", "params":{"length":20, "std":2.0}, "fields":["bb_low","bb_mid","bb_high"]},
    "BBW":   {"name":"Bollinger Bands Width", "params":{}, "fields":["bb_width"]},
    "KC":    {"name":"Keltner Channel", "params":{"length":20, "mult":2.0}, "fields":["kc_low","kc_mid","kc_high"]},
    "DON":   {"name":"Donchian Channels", "params":{"lower":20, "upper":20}, "fields":["don_low","don_mid","don_high"]},
    "ENV":   {"name":"Envelopes", "params":{"length":20, "offset_pct":1.5}, "fields":["env_low","env_mid","env_high"]},

    "PSAR":  {"name":"Parabolic SAR", "params":{"step":0.02, "max":0.2}, "fields":["psar"]},
    "ICHI":  {"name":"Ichimoku", "params":{"tenkan":9, "kijun":26, "senkou_b":52}, "fields":["ichi_tenkan","ichi_kijun","ichi_senkou_a","ichi_senkou_b"]},
    "FRACTAL":{"name":"Fractal", "params":{"left":2, "right":2}, "fields":["fractal_high","fractal_low"]},
    "FCB":   {"name":"Fractal Chaos Bands (approx.)", "params":{"length":20}, "fields":["fcb_low","fcb_high"]},
    "ALLIGATOR":{"name":"Alligator", "params":{"jaw":13,"teeth":8,"lips":5,"jaw_offset":8,"teeth_offset":5,"lips_offset":3},
                 "fields":["jaw","teeth","lips"]},
    "BULL_POWER":{"name":"Bulls Power", "params":{"length":13}, "fields":["bull_power"]},
    "BEAR_POWER":{"name":"Bears Power", "params":{"length":13}, "fields":["bear_power"]},
    "ZIGZAG":{"name":"ZigZag", "params":{"dev_pct":5.0}, "fields":["zigzag"]},
}

def apply_indicators(df, cfg):
    ind = cfg["indicators"]
    def ena(k): return ind.get(k,{}).get("enabled", False)
    def p(k, key, default=None): return ind.get(k,{}).get(key, default)

    df = df.copy()
    df.columns = [c.lower() for c in df.columns]

    if ena("EMA"):   df["ema"]   = ta.ema(df["close"], length=int(p("EMA","period",20)))
    if ena("SMA"):   df["sma"]   = ta.sma(df["close"], length=int(p("SMA","period",20)))
    if ena("WMA"):   df["wma"]   = ta.wma(df["close"], length=int(p("WMA","period",20)))
    if ena("SMMA"):  df["smma"]  = ta.rma(df["close"], length=int(p("SMMA","period",20)))
    if ena("TMA"):   df["tma"]   = ta.sma(ta.sma(df["close"], length=int(p("TMA","period",20))), length=int(p("TMA","period",20)))
    if ena("MA"):    df["ma"]    = ta.sma(df["close"], length=int(p("MA","period",20)))

    if ena("RSI"): df["rsi"] = ta.rsi(df["close"], length=int(p("RSI","length",14)))
    if ena("STOCH"):
        s = ta.stoch(df["high"], df["low"], df["close"],
                     k=int(p("STOCH","k",14)), d=int(p("STOCH","d",3)), smooth_k=int(p("STOCH","smooth_k",3)))
        if s is not None and s.shape[1] >= 2:
            df["stoch_k"], df["stoch_d"] = s.iloc[:,0], s.iloc[:,1]

    m = ta.macd(df["close"], fast=int(ind.get("MACD",{}).get("fast",12)),
                slow=int(ind.get("MACD",{}).get("slow",26)),
                signal=int(ind.get("MACD",{}).get("signal",9)))
    if m is not None and m.shape[1] >= 3:
        if ind.get("MACD",{}).get("enabled",False):
            df["macd"], df["macd_signal"], df["macd_hist"] = m.iloc[:,0], m.iloc[:,1], m.iloc[:,2]
        if ind.get("OSMA",{}).get("enabled",False):
            df["osma"] = m.iloc[:,0] - m.iloc[:,1]

    if ind.get("AO",{}).get("enabled",False) or ind.get("AC",{}).get("enabled",False):
        ao = ta.ao(df["high"], df["low"], fast=int(ind.get("AO",{}).get("fast",5)), slow=int(ind.get("AO",{}).get("slow",34)))
        if ao is not None:
            if ind.get("AO",{}).get("enabled",False): df["ao"] = ao
            if ind.get("AC",{}).get("enabled",False): df["ac"] = ao - ta.sma(ao, length=int(ind.get("AC",{}).get("sma",5)))

    if ind.get("MOM",{}).get("enabled",False): df["mom"] = ta.mom(df["close"], length=int(ind["MOM"]["length"]))
    if ind.get("ROC",{}).get("enabled",False): df["roc"] = ta.roc(df["close"], length=int(ind["ROC"]["length"]))
    if ind.get("WILLR",{}).get("enabled",False): df["willr"] = ta.willr(df["high"], df["low"], df["close"], length=int(ind["WILLR"]["length"]))
    if ind.get("CCI",{}).get("enabled",False): df["cci"] = ta.cci(df["high"], df["low"], df["close"], length=int(ind["CCI"]["length"]))
    if ind.get("STC",{}).get("enabled",False): df["stc"] = ta.stc(df["close"], tclen=int(ind["STC"]["tclen"]), fast=int(ind["STC"]["fast"]), slow=int(ind["STC"]["slow"]), factor=float(ind["STC"]["factor"]))

    if ind.get("ADX",{}).get("enabled",False):
        adx = ta.adx(df["high"], df["low"], df["close"], length=int(ind["ADX"]["length"]))
        if adx is not None: df["adx"] = adx.filter(like="ADX").iloc[:,0]
    if ind.get("ATR",{}).get("enabled",False): df["atr"] = ta.atr(df["high"], df["low"], df["close"], length=int(ind["ATR"]["length"]))
    if ind.get("AROON",{}).get("enabled",False):
        a = ta.aroon(df["high"], df["low"], length=int(ind["AROON"]["length"]))
        if a is not None:
            df["aroon_up"] = a.filter(like="UP").iloc[:,0]
            df["aroon_down"] = a.filter(like="DOWN").iloc[:,0]
            osc = a.filter(like="OSC")
            if not osc.empty: df["aroon_osc"] = osc.iloc[:,0]
    if ind.get("VORTEX",{}).get("enabled",False):
        vx = ta.vortex(df["high"], df["low"], df["close"], length=int(ind["VORTEX"]["length"]))
        if vx is not None and vx.shape[1] >= 2:
            df["vortex_pos"], df["vortex_neg"] = vx.iloc[:,0], vx.iloc[:,1]
    if ind.get("SUPERTREND",{}).get("enabled",False):
        st = ta.supertrend(df["high"], df["low"], df["close"],
                           length=int(ind["SUPERTREND"]["length"]), multiplier=float(ind["SUPERTREND"]["multiplier"]))
        if isinstance(st, pd.DataFrame) and st.shape[1] >= 3:
            dirc = st.filter(like="dir"); ub = st.filter(like="ub") or st.filter(like="upper"); lb = st.filter(like="lb") or st.filter(like="lower")
            if not dirc.empty: df["supertrend_dir"] = dirc.iloc[:,0]
            if not ub.empty:  df["supertrend_upper"] = ub.iloc[:,0]
            if not lb.empty:  df["supertrend_lower"] = lb.iloc[:,0]

    bb = ta.bbands(df["close"], length=int(ind.get("BB",{}).get("length",20)), std=float(ind.get("BB",{}).get("std",2.0)))
    if bb is not None and bb.shape[1] >= 3:
        if ind.get("BB",{}).get("enabled",False):
            df["bb_low"], df["bb_mid"], df["bb_high"] = bb.iloc[:,0], bb.iloc[:,1], bb.iloc[:,2]
        if ind.get("BBW",{}).get("enabled",False):
            upper, mid, lower = bb.iloc[:,2], bb.iloc[:,1], bb.iloc[:,0]
            width = (upper - lower); df["bb_width"] = ((width / mid.replace(0, pd.NA)) * 100.0).fillna(width)

    if ind.get("KC",{}).get("enabled",False):
        kc = ta.kc(df["high"], df["low"], df["close"], length=int(ind["KC"]["length"]), mult=float(ind["KC"]["mult"]))
        if kc is not None and kc.shape[1] >= 3:
            df["kc_low"], df["kc_mid"], df["kc_high"] = kc.iloc[:,0], kc.iloc[:,1], kc.iloc[:,2]
    if ind.get("DON",{}).get("enabled",False):
        d = ta.donchian(df["high"], df["low"], lower_length=int(ind["DON"]["lower"]), upper_length=int(ind["DON"]["upper"]))
        if d is not None and d.shape[1] >= 3:
            df["don_low"], df["don_mid"], df["don_high"] = d.iloc[:,0], d.iloc[:,1], d.iloc[:,2]
    if ind.get("ENV",{}).get("enabled",False):
        mid = ta.sma(df["close"], length=int(ind["ENV"]["length"]))
        off = float(ind["ENV"]["offset_pct"])/100.0
        df["env_mid"], df["env_high"], df["env_low"] = mid, mid*(1+off), mid*(1-off)

    if ind.get("PSAR",{}).get("enabled",False):
        ps = ta.psar(df["high"], df["low"], df["close"], step=float(ind["PSAR"]["step"]), max=float(ind["PSAR"]["max"]))
        if ps is not None:
            cols = [c for c in ps.columns if "PSAR" in c]; ser = None
            for c in cols: ser = ps[c] if ser is None else ser.combine_first(ps[c])
            if ser is not None: df["psar"] = ser
    return df
