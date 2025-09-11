# indicators.py — full indicator catalog + calculator (pandas_ta)
import pandas as pd
import pandas_ta as ta

# For the dashboard form (labels + default params)
INDICATOR_SPECS = {
    # ---------- Moving Averages (overlay) ----------
    "EMA":   {"name": "EMA",   "kind":"overlay", "params": {"period": 20}, "fields": ["ema"]},
    "SMA":   {"name": "SMA",   "kind":"overlay", "params": {"period": 20}, "fields": ["sma"]},
    "WMA":   {"name": "WMA",   "kind":"overlay", "params": {"period": 20}, "fields": ["wma"]},
    "SMMA":  {"name": "SMMA (RMA)", "kind":"overlay", "params": {"period": 20}, "fields": ["smma"]},
    "TMA":   {"name": "TMA (Triangular)", "kind":"overlay", "params": {"period": 20}, "fields": ["tma"]},
    "MA":    {"name": "Moving Average", "kind":"overlay", "params": {"period": 20}, "fields": ["ma"]},

    # ---------- Oscillators / Panels ----------
    "RSI":   {"name":"RSI", "kind":"panel", "params":{"length":14, "overbought":70, "oversold":30}, "fields":["rsi"]},
    "STOCH": {"name":"Stochastic Oscillator", "kind":"panel", "params":{"k":14, "d":3, "smooth_k":3}, "fields":["stoch_k","stoch_d"]},
    "MACD":  {"name":"MACD", "kind":"panel", "params":{"fast":12, "slow":26, "signal":9}, "fields":["macd","macd_signal","macd_hist"]},
    "OSMA":  {"name":"OsMA", "kind":"panel", "params":{"fast":12, "slow":26, "signal":9}, "fields":["osma"]},
    "AO":    {"name":"Awesome Oscillator", "kind":"panel", "params":{"fast":5, "slow":34}, "fields":["ao"]},
    "AC":    {"name":"Accelerator Oscillator", "kind":"panel", "params":{"fast":5, "slow":34, "sma":5}, "fields":["ac"]},
    "MOM":   {"name":"Momentum", "kind":"panel", "params":{"length":10}, "fields":["mom"]},
    "ROC":   {"name":"Rate of Change", "kind":"panel", "params":{"length":10}, "fields":["roc"]},
    "WILLR": {"name":"Williams %R", "kind":"panel", "params":{"length":14}, "fields":["willr"]},
    "CCI":   {"name":"CCI", "kind":"panel", "params":{"length":20}, "fields":["cci"]},
    "STC":   {"name":"Schaff Trend Cycle", "kind":"panel", "params":{"tclen":10,"fast":23,"slow":50,"factor":0.5}, "fields":["stc"]},

    "ADX":   {"name":"ADX", "kind":"panel", "params":{"length":14, "threshold":20}, "fields":["adx"]},
    "ATR":   {"name":"Average True Range", "kind":"panel", "params":{"length":14}, "fields":["atr"]},
    "AROON": {"name":"Aroon", "kind":"panel", "params":{"length":25}, "fields":["aroon_up","aroon_down","aroon_osc"]},
    "VORTEX":{"name":"Vortex", "kind":"panel", "params":{"length":14}, "fields":["vortex_pos","vortex_neg"]},

    # ---------- Channels / Bands (overlay or panel) ----------
    "BB":    {"name":"Bollinger Bands", "kind":"overlay", "params":{"length":20, "std":2.0}, "fields":["bb_low","bb_mid","bb_high"]},
    "BBW":   {"name":"Bollinger Bands Width", "kind":"panel", "params":{}, "fields":["bb_width"]},
    "KC":    {"name":"Keltner Channel", "kind":"overlay", "params":{"length":20, "mult":2.0}, "fields":["kc_low","kc_mid","kc_high"]},
    "DON":   {"name":"Donchian Channels", "kind":"overlay", "params":{"lower":20, "upper":20}, "fields":["don_low","don_mid","don_high"]},
    "ENV":   {"name":"Envelopes", "kind":"overlay", "params":{"length":20, "offset_pct":1.5}, "fields":["env_low","env_mid","env_high"]},

    # ---------- Overlays / Signals ----------
    "PSAR":  {"name":"Parabolic SAR", "kind":"overlay", "params":{"step":0.02, "max":0.2}, "fields":["psar"]},
    "ICHI":  {"name":"Ichimoku", "kind":"overlay", "params":{"tenkan":9, "kijun":26, "senkou_b":52}, "fields":["ichi_tenkan","ichi_kijun","ichi_senkou_a","ichi_senkou_b"]},
    "FRACTAL":{"name":"Fractal", "kind":"overlay", "params":{"left":2, "right":2}, "fields":["fractal_high","fractal_low"]},
    "FCB":   {"name":"Fractal Chaos Bands (approx.)", "kind":"overlay", "params":{"length":20}, "fields":["fcb_low","fcb_high"]},
    "ALLIGATOR":{"name":"Alligator", "kind":"overlay",
                 "params":{"jaw":13,"teeth":8,"lips":5,"jaw_offset":8,"teeth_offset":5,"lips_offset":3},
                 "fields":["jaw","teeth","lips"]},
    "BULL_POWER":{"name":"Bulls Power", "kind":"panel", "params":{"length":13}, "fields":["bull_power"]},
    "BEAR_POWER":{"name":"Bears Power", "kind":"panel", "params":{"length":13}, "fields":["bear_power"]},
    "ZIGZAG":{"name":"ZigZag", "kind":"overlay", "params":{"dev_pct":5.0}, "fields":["zigzag"]},
}

def apply_indicators(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Compute all enabled indicators into columns the plotter/strategies can use."""
    ind = cfg.get("indicators", {})
    def ena(k): return ind.get(k,{}).get("enabled", False)
    def p(k, key, default=None): return ind.get(k,{}).get(key, default)

    df = df.copy()
    df.columns = [c.lower() for c in df.columns]

    # --- MAs
    if ena("EMA"):   df["ema"]   = ta.ema(df["close"], length=int(p("EMA","period",20)))
    if ena("SMA"):   df["sma"]   = ta.sma(df["close"], length=int(p("SMA","period",20)))
    if ena("WMA"):   df["wma"]   = ta.wma(df["close"], length=int(p("WMA","period",20)))
    if ena("SMMA"):  df["smma"]  = ta.rma(df["close"], length=int(p("SMMA","period",20)))
    if ena("TMA"):
        per = int(p("TMA","period",20))
        df["tma"] = ta.sma(ta.sma(df["close"], length=per), length=per)
    if ena("MA"):    df["ma"]    = ta.sma(df["close"], length=int(p("MA","period",20)))

    # --- Oscillators
    if ena("RSI"): df["rsi"] = ta.rsi(df["close"], length=int(p("RSI","length",14)))

    if ena("STOCH"):
        s = ta.stoch(df["high"], df["low"], df["close"],
                     k=int(p("STOCH","k",14)), d=int(p("STOCH","d",3)), smooth_k=int(p("STOCH","smooth_k",3)))
        if isinstance(s, pd.DataFrame) and s.shape[1] >= 2:
            df["stoch_k"], df["stoch_d"] = s.iloc[:,0], s.iloc[:,1]

    macd = ta.macd(df["close"], fast=int(p("MACD","fast",12)),
                   slow=int(p("MACD","slow",26)), signal=int(p("MACD","signal",9)))
    if isinstance(macd, pd.DataFrame) and macd.shape[1] >= 3:
        if ena("MACD"):
            df["macd"], df["macd_signal"], df["macd_hist"] = macd.iloc[:,0], macd.iloc[:,1], macd.iloc[:,2]
        if ena("OSMA"):
            df["osma"] = macd.iloc[:,0] - macd.iloc[:,1]

    if ena("AO") or ena("AC"):
        ao = ta.ao(df["high"], df["low"], fast=int(p("AO","fast",5)), slow=int(p("AO","slow",34)))
        if ao is not None:
            if ena("AO"): df["ao"] = ao
            if ena("AC"): df["ac"] = ao - ta.sma(ao, length=int(p("AC","sma",5)))

    if ena("MOM"):   df["mom"] = ta.mom(df["close"], length=int(p("MOM","length",10)))
    if ena("ROC"):   df["roc"] = ta.roc(df["close"], length=int(p("ROC","length",10)))
    if ena("WILLR"): df["willr"] = ta.willr(df["high"], df["low"], df["close"], length=int(p("WILLR","length",14)))
    if ena("CCI"):   df["cci"] = ta.cci(df["high"], df["low"], df["close"], length=int(p("CCI","length",20)))
    if ena("STC"):   df["stc"] = ta.stc(df["close"], tclen=int(p("STC","tclen",10)),
                                        fast=int(p("STC","fast",23)), slow=int(p("STC","slow",50)),
                                        factor=float(p("STC","factor",0.5)))

    if ena("ADX"):
        adx = ta.adx(df["high"], df["low"], df["close"], length=int(p("ADX","length",14)))
        if isinstance(adx, pd.DataFrame): df["adx"] = adx.filter(like="ADX").iloc[:,0]
    if ena("ATR"): df["atr"] = ta.atr(df["high"], df["low"], df["close"], length=int(p("ATR","length",14)))
    if ena("AROON"):
        a = ta.aroon(df["high"], df["low"], length=int(p("AROON","length",25)))
        if isinstance(a, pd.DataFrame):
            df["aroon_up"]   = a.filter(like="UP").iloc[:,0]
            df["aroon_down"] = a.filter(like="DOWN").iloc[:,0]
            osc = a.filter(like="OSC")
            if not osc.empty: df["aroon_osc"] = osc.iloc[:,0]
    if ena("VORTEX"):
        vx = ta.vortex(df["high"], df["low"], df["close"], length=int(p("VORTEX","length",14)))
        if isinstance(vx, pd.DataFrame) and vx.shape[1] >= 2:
            df["vortex_pos"], df["vortex_neg"] = vx.iloc[:,0], vx.iloc[:,1]

    # --- Bands / Channels
    bb = ta.bbands(df["close"], length=int(p("BB","length",20)), std=float(p("BB","std",2.0)))
    if isinstance(bb, pd.DataFrame) and bb.shape[1] >= 3:
        if ena("BB"):
            df["bb_low"], df["bb_mid"], df["bb_high"] = bb.iloc[:,0], bb.iloc[:,1], bb.iloc[:,2]
        if ena("BBW"):
            width = (bb.iloc[:,2] - bb.iloc[:,0])
            mid   = bb.iloc[:,1].replace(0, pd.NA)
            df["bb_width"] = ((width / mid) * 100.0).fillna(width)

    if ena("KC"):
        kc = ta.kc(df["high"], df["low"], df["close"], length=int(p("KC","length",20)), mult=float(p("KC","mult",2.0)))
        if isinstance(kc, pd.DataFrame) and kc.shape[1] >= 3:
            df["kc_low"], df["kc_mid"], df["kc_high"] = kc.iloc[:,0], kc.iloc[:,1], kc.iloc[:,2]
    if ena("DON"):
        d = ta.donchian(df["high"], df["low"], lower_length=int(p("DON","lower",20)), upper_length=int(p("DON","upper",20)))
        if isinstance(d, pd.DataFrame) and d.shape[1] >= 3:
            df["don_low"], df["don_mid"], df["don_high"] = d.iloc[:,0], d.iloc[:,1], d.iloc[:,2]
    if ena("ENV"):
        mid = ta.sma(df["close"], length=int(p("ENV","length",20)))
        off = float(p("ENV","offset_pct",1.5))/100.0
        df["env_mid"], df["env_high"], df["env_low"] = mid, mid*(1+off), mid*(1-off)

    # --- Overlays / signals
    if ena("PSAR"):
        ps = ta.psar(df["high"], df["low"], df["close"], step=float(p("PSAR","step",0.02)), max=float(p("PSAR","max",0.2)))
        if isinstance(ps, pd.DataFrame):
            cols = [c for c in ps.columns if "PSAR" in c.upper()]
            ser = None
            for c in cols: ser = ps[c] if ser is None else ser.combine_first(ps[c])
            if ser is not None: df["psar"] = ser

    if ena("ICHI"):
        tenkan = int(p("ICHI","tenkan",9)); kijun = int(p("ICHI","kijun",26)); senkou_b = int(p("ICHI","senkou_b",52))
        conv = ta.ichimoku_conversion(df["high"], df["low"], length=tenkan)
        base = ta.ichimoku_base(df["high"], df["low"], length=kijun)
        spanB = ta.ichimoku_b(df["high"], df["low"], length=senkou_b)
        if conv is not None: df["ichi_tenkan"] = conv
        if base is not None: df["ichi_kijun"]   = base
        if spanB is not None:
            # span A typically = (tenkan + kijun)/2 shifted forward; we store raw values
            if "ichi_tenkan" in df and "ichi_kijun" in df:
                df["ichi_senkou_a"] = (df["ichi_tenkan"] + df["ichi_kijun"]) / 2.0
            df["ichi_senkou_b"] = spanB

    if ena("FRACTAL"):
        # simple fractal highs/lows (left/right bars)
        left = int(p("FRACTAL","left",2)); right = int(p("FRACTAL","right",2))
        df["fractal_high"] = df["high"].rolling(left+right+1, center=True).max()
        df["fractal_low"]  = df["low"].rolling(left+right+1, center=True).min()

    if ena("FCB"):
        l = int(p("FCB","length",20))
        rolling_high = df["high"].rolling(l).max()
        rolling_low  = df["low"].rolling(l).min()
        df["fcb_high"], df["fcb_low"] = rolling_high, rolling_low

    if ena("ALLIGATOR"):
        jaw_p, teeth_p, lips_p = int(p("ALLIGATOR","jaw",13)), int(p("ALLIGATOR","teeth",8)), int(p("ALLIGATOR","lips",5))
        jaw_off, teeth_off, lips_off = int(p("ALLIGATOR","jaw_offset",8)), int(p("ALLIGATOR","teeth_offset",5)), int(p("ALLIGATOR","lips_offset",3))
        df["jaw"]   = ta.sma(df["close"], length=jaw_p).shift(jaw_off)
        df["teeth"] = ta.sma(df["close"], length=teeth_p).shift(teeth_off)
        df["lips"]  = ta.sma(df["close"], length=lips_p).shift(lips_off)

    if ena("BULL_POWER"):
        l = int(p("BULL_POWER","length",13))
        ema = ta.ema(df["close"], length=l)
        df["bull_power"] = df["high"] - ema
    if ena("BEAR_POWER"):
        l = int(p("BEAR_POWER","length",13))
        ema = ta.ema(df["close"], length=l)
        df["bear_power"] = df["low"] - ema

    if ena("ZIGZAG"):
        dev = float(p("ZIGZAG","dev_pct",5.0)) / 100.0
        # very lightweight approx: mark swing when change exceeds dev from last pivot
        pivots = [pd.NA] * len(df)
        last_p = df["close"].iloc[0] if len(df) else pd.NA
        last_dir = 0
        for i, price in enumerate(df["close"]):
            if pd.isna(last_p): last_p = price
            if price >= last_p * (1 + dev) and last_dir <= 0:
                pivots[i] = price; last_p = price; last_dir = +1
            elif price <= last_p * (1 - dev) and last_dir >= 0:
                pivots[i] = price; last_p = price; last_dir = -1
        df["zigzag"] = pd.Series(pivots, index=df.index)

    return df
