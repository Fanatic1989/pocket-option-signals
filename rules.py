# rules.py
import re, math, pandas as pd, pandas_ta as ta

def cross_over(a_prev, a_now, b_prev, b_now): return (a_now > b_now) and (a_prev <= b_prev)
def cross_under(a_prev, a_now, b_prev, b_now): return (a_now < b_now) and (a_prev >= b_prev)
def near(a, b, tol_pct=0.1):
    if b == 0: return abs(a-b) < 1e-12
    try: return abs((a-b)/b)*100.0 <= float(tol_pct)
    except: return False

def bounce(series_vals, level: float, lookback: int = 3, direction: str = "up", tol: float = 0.1):
    if len(series_vals) < lookback+1: return False
    window = series_vals.iloc[-(lookback+1):-1]
    touched = any(near(float(v), float(level), tol_pct=tol) for v in window if pd.notna(v))
    if not touched: return False
    last = series_vals.iloc[-1]; prev = series_vals.iloc[-2]
    if direction == "up":   return pd.notna(last) and pd.notna(prev) and (last > prev) and last >= level
    if direction == "down": return pd.notna(last) and pd.notna(prev) and (last < prev) and last <= level
    return False

def _ensure_series(df: pd.DataFrame, name: str, arg=None) -> pd.Series:
    name = name.upper()
    key = f"{name}{'' if arg is None else arg}"
    if key in df.columns: return df[key]
    if name in ("SMA","EMA","WMA","SMMA","TMA"):
        n = int(arg or 20)
        s = None
        if name == "SMA": s = ta.sma(df["close"], length=n)
        elif name == "EMA": s = ta.ema(df["close"], length=n)
        elif name == "WMA": s = ta.wma(df["close"], length=n)
        elif name == "SMMA": s = ta.rma(df["close"], length=n)
        else: s = ta.sma(ta.sma(df["close"], length=n), length=n)
        df[key] = s; return s
    if name == "RSI":
        n = int(arg or 14); s = ta.rsi(df["close"], length=n); df[key] = s; return s
    if name == "CLOSE_SER":
        s = df["close"]; df[key]=s; return s
    raise ValueError(name)

def build_rule_env(df: pd.DataFrame, i: int, cfg_custom):
    def SMA(n): return _ensure_series(df, "SMA", n)
    def EMA(n): return _ensure_series(df, "EMA", n)
    def WMA(n): return _ensure_series(df, "WMA", n)
    def SMMA(n): return _ensure_series(df, "SMMA", n)
    def TMA(n): return _ensure_series(df, "TMA", n)
    def RSI(n=None): return _ensure_series(df, "RSI", n)
    def _CLOSE_SER(): return _ensure_series(df, "CLOSE_SER", None)

    def VAL(s): 
        import pandas as pd
        return float(s.iloc[i]) if pd.notna(s.iloc[i]) else float("nan")
    def PREV(s):
        import pandas as pd
        return float(s.iloc[i-1]) if i>0 and pd.notna(s.iloc[i-1]) else float("nan")

    env = {
        "SMA": SMA, "EMA": EMA, "WMA": WMA, "SMMA": SMMA, "TMA": TMA, "RSI": RSI, "_CLOSE_SER": _CLOSE_SER,
        "VAL": VAL, "PREV": PREV,
        "near": near, "bounce": lambda series, level, lookback=None, direction='up', tol=None:
            bounce(series, float(level), int(lookback or cfg_custom.get('lookback',3)), direction=direction, tol=float(tol or cfg_custom.get('tol_pct',0.1))),
        "cross_over": cross_over, "cross_under": cross_under, "math": math
    }
    return env

def eval_rule(expr: str, df: pd.DataFrame, i: int, cfg_custom) -> bool:
    if not expr or not expr.strip(): return False
    try:
        return bool(eval(expr, {"__builtins__": {}}, build_rule_env(df, i, cfg_custom)))
    except Exception:
        return False

def _norm(s): 
    import re
    return re.sub(r"\s+", " ", (s or "").strip().lower())

def parse_natural_rule(text: str) -> str:
    s = _norm(text)
    if not s: return ""
    parts = [p.strip() for p in re.split(r"\b(and|or|then)\b", s) if p.strip() and p not in ('and','or','then')]
    dsl = []
    for c in parts:
        if "respect" in c and ("sma" in c or "ema" in c or "wma" in c or "smma" in c or "tma" in c):
            import re
            m = re.search(r"(\d+)", c); period = int(m.group(1)) if m else 50
            which = "SMA"
            for w in ("ema","sma","wma","smma","tma"):
                if w in c: which = w.upper()
            dsl.append(f"near(VAL(_CLOSE_SER()), VAL({which}({period})), tol_pct=0.2)")
        elif "rsi" in c and "bounce" in c:
            direction = "up" if "up" in c else ("down" if "down" in c else "up")
            dsl.append(f"bounce(RSI(), 50, lookback=3, direction='{direction}')")
        elif "stoch" in c and "cross" in c:
            # map to RSI/SMA crossover stand-in for simplicity in simple mode
            if "down" in c: dsl.append("cross_under(PREV(RSI()), VAL(RSI()), PREV(SMA(50)), VAL(SMA(50)))")
            else: dsl.append("cross_over(PREV(RSI()), VAL(RSI()), PREV(SMA(50)), VAL(SMA(50)))")
    if not dsl: return ""
    expr = dsl[0]
    for i in range(1, len(dsl)): expr = f"({expr}) and ({dsl[i]})"
    return expr
