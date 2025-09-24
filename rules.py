# rules.py
import re
import math
import pandas as pd
import pandas_ta as ta

# ---------- helpers used at runtime evaluation ----------

def cross_over(a_prev, a_now, b_prev, b_now):
    return (a_now > b_now) and (a_prev <= b_prev)

def cross_under(a_prev, a_now, b_prev, b_now):
    return (a_now < b_now) and (a_prev >= b_prev)

def near(a, b, tol_pct=0.1):
    if b == 0:
        return abs(a - b) < 1e-12
    try:
        return abs((a - b) / b) * 100.0 <= float(tol_pct)
    except Exception:
        return False

def bounce(series_vals, level: float, lookback: int = 3, direction: str = "up", tol: float = 0.1):
    """Detect a 'touch near level' within lookback, followed by a move away in direction."""
    if len(series_vals) < lookback + 1:
        return False
    window = series_vals.iloc[-(lookback+1):-1]
    touched = any(near(float(v), float(level), tol_pct=tol) for v in window if pd.notna(v))
    if not touched:
        return False
    last = series_vals.iloc[-1]; prev = series_vals.iloc[-2]
    if direction == "up":
        return pd.notna(last) and pd.notna(prev) and (last > prev) and last >= level
    if direction == "down":
        return pd.notna(last) and pd.notna(prev) and (last < prev) and last <= level
    return False

# ---------- series getters used by eval sandbox ----------

def _ensure_series(df: pd.DataFrame, name: str, arg=None) -> pd.Series:
    name = name.upper()
    key = f"{name}{'' if arg is None else arg}"
    if key in df.columns:
        return df[key]

    if name in ("SMA", "EMA", "WMA", "SMMA", "TMA"):
        n = int(arg or 20)
        if name == "SMA":
            s = ta.sma(df["close"], length=n)
        elif name == "EMA":
            s = ta.ema(df["close"], length=n)
        elif name == "WMA":
            s = ta.wma(df["close"], length=n)
        elif name == "SMMA":
            s = ta.rma(df["close"], length=n)
        else:  # TMA
            s = ta.sma(ta.sma(df["close"], length=n), length=n)
        df[key] = s
        return s

    if name == "RSI":
        n = int(arg or 14)
        s = ta.rsi(df["close"], length=n)
        df[key] = s
        return s

    if name == "STOCHK":
        # using pandas_ta stoch (K,D) when available
        st = ta.stoch(df["high"], df["low"], df["close"], k=14, d=3, smooth_k=3)
        s = st.iloc[:, 0] if st is not None and st.shape[1] >= 2 else pd.Series([float("nan")] * len(df))
        df[key] = s
        return s

    if name == "STOCHD":
        st = ta.stoch(df["high"], df["low"], df["close"], k=14, d=3, smooth_k=3)
        s = st.iloc[:, 1] if st is not None and st.shape[1] >= 2 else pd.Series([float("nan")] * len(df))
        df[key] = s
        return s

    if name == "CLOSE_SER":
        s = df["close"]
        df[key] = s
        return s

    raise ValueError(name)

def build_rule_env(df: pd.DataFrame, i: int, cfg_custom):
    def SMA(n):   return _ensure_series(df, "SMA", n)
    def EMA(n):   return _ensure_series(df, "EMA", n)
    def WMA(n):   return _ensure_series(df, "WMA", n)
    def SMMA(n):  return _ensure_series(df, "SMMA", n)
    def TMA(n):   return _ensure_series(df, "TMA", n)
    def RSI(n=14): return _ensure_series(df, "RSI", n)
    def STOCHK(): return _ensure_series(df, "STOCHK")
    def STOCHD(): return _ensure_series(df, "STOCHD")
    def _CLOSE_SER(): return _ensure_series(df, "CLOSE_SER")

    def VAL(s):
        return float(s.iloc[i]) if pd.notna(s.iloc[i]) else float("nan")
    def PREV(s):
        return float(s.iloc[i-1]) if i > 0 and pd.notna(s.iloc[i-1]) else float("nan")

    env = {
        "SMA": SMA, "EMA": EMA, "WMA": WMA, "SMMA": SMMA, "TMA": TMA,
        "RSI": RSI, "STOCHK": STOCHK, "STOCHD": STOCHD, "_CLOSE_SER": _CLOSE_SER,
        "VAL": VAL, "PREV": PREV,
        "near": near,
        "bounce": lambda series, level, lookback=None, direction='up', tol=None:
            bounce(series, float(level), int(lookback or cfg_custom.get('lookback', 3)),
                   direction=direction, tol=float(tol or cfg_custom.get('tol_pct', 0.1))),
        "cross_over": cross_over, "cross_under": cross_under, "math": math
    }
    return env

def eval_rule(expr: str, df: pd.DataFrame, i: int, cfg_custom) -> bool:
    if not expr or not expr.strip():
        return False
    try:
        return bool(eval(expr, {"__builtins__": {}}, build_rule_env(df, i, cfg_custom)))
    except Exception:
        return False

# ---------- natural language parsing ----------

def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip().lower())

def _extract_ma_period(s: str, default_period=50):
    m = re.search(r"\b(\d{1,3})\s*(sma|ema|wma|smma|tma)\b", s)
    if not m:
        m = re.search(r"\b(\d{1,3})\s*(?:period|ma)\b", s)
    period = int(m.group(1)) if m else default_period
    which = "sma"
    if m and len(m.groups()) >= 2 and m.group(2):
        which = m.group(2).lower()
    return which.upper(), period

def _trend_dir(s: str):
    # returns "up", "down" or None
    if "uptrend" in s or "trend up" in s or "goes back up" in s or "go back up" in s or "go up" in s:
        return "up"
    if "downtrend" in s or "trend down" in s or "goes back down" in s or "go back down" in s or "go down" in s:
        return "down"
    return None

def _make_ma_symbol(which: str, period: int):
    return f"{which}({period})"

def _price_respect_ma(which: str, period: int, tol_pct=0.2):
    # near(close, MA(period))
    return f"near(VAL(_CLOSE_SER()), VAL({_make_ma_symbol(which, period)}), tol_pct={tol_pct})"

def _ma_trend(which: str, period: int, want="up"):
    # slope condition
    sign = ">" if want == "up" else "<"
    return f"VAL({_make_ma_symbol(which, period)}) {sign} PREV({_make_ma_symbol(which, period)})"

def _rsi_bounce_dir(direction="up"):
    # bounce(RSI(), 50, direction=...)
    d = "up" if direction == "up" else "down"
    return f"bounce(RSI(), 50, lookback=3, direction='{d}')"

def _stoch_cross(direction="up"):
    # use %K/%D cross as proxy
    if direction == "up":
        return "cross_over(PREV(STOCHK()), VAL(STOCHK()), PREV(STOCHD()), VAL(STOCHD()))"
    else:
        return "cross_under(PREV(STOCHK()), VAL(STOCHK()), PREV(STOCHD()), VAL(STOCHD()))"

def _contains_sell_clause(s: str) -> bool:
    return bool(re.search(r"\bthen\s+sell\b", s))

def _contains_buy_clause(s: str) -> bool:
    return bool(re.search(r"\bthen\s+buy\b", s))

def parse_natural_rule(text: str) -> str:
    """
    Legacy: parse a single-clause text into a BUY/SELL DSL expression.
    If your UI uses separate boxes this is fine. For combined text,
    use parse_natural_pair() instead.
    """
    s = _norm(text)
    if not s:
        return ""

    # Determine direction bias from phrasing
    direction = _trend_dir(s) or ("up" if "buy" in s else ("down" if "sell" in s else "up"))

    # MA part
    which, period = _extract_ma_period(s, default_period=50)
    ma_respect = _price_respect_ma(which, period, tol_pct=0.2)

    # Trend part (optional)
    trend_cond = ""
    if "uptrend" in s or "downtrend" in s:
        trend_cond = _ma_trend(which, period, want="up" if "up" in s else "down")

    # Retest = near MA again; we already use near() so treat as alias (no extra cond)
    # RSI bounce
    rsi_part = _rsi_bounce_dir(direction)

    # Stoch cross
    stoch_part = _stoch_cross(direction)

    parts = [ma_respect]
    if trend_cond:
        parts.append(trend_cond)
    parts.append(rsi_part)
    parts.append(stoch_part)
    expr = " and ".join(f"({p})" for p in parts)
    return expr

def parse_natural_pair(text: str):
    """
    NEW: Accept a long sentence that includes BOTH buy and sell logic.
    Returns tuple (buy_expr, sell_expr). Either can be "" if not found.
    """
    s = _norm(text)
    if not s:
        return "", ""

    # Split around "then buy" / "then sell"
    # We’ll collect the clause BEFORE each "then <side>"
    # Example: "... then buy and ... then sell"
    clauses = re.split(r"\bthen\s+(buy|sell)\b", s)
    # clauses becomes [pre, side1, post1, side2, post2, ...]
    buy_txt, sell_txt = "", ""
    if len(clauses) >= 3:
        pre = clauses[0]
        side1, post1 = clauses[1], clauses[2]
        text1 = (pre + " " + post1).strip()
        if side1 == "buy":
            buy_txt = text1
        else:
            sell_txt = text1
        # If more sides present
        if len(clauses) >= 5:
            side2, post2 = clauses[3], clauses[4]
            if side2 == "buy":
                buy_txt = (post2 or "").strip()
            else:
                sell_txt = (post2 or "").strip()
    else:
        # If no explicit "then buy/sell", try to infer:
        if "buy" in s:
            buy_txt = s
        if "sell" in s:
            sell_txt = s

    # Build rules per side with bias directions
    def side_to_expr(side_txt: str, side: str) -> str:
        if not side_txt:
            return ""
        # Force direction bias for RSI/Stoch/trend based on side
        which, period = _extract_ma_period(side_txt, default_period=50)
        trend_bias = "up" if side == "buy" else "down"
        ma_respect = _price_respect_ma(which, period, tol_pct=0.2)

        # If user explicitly said up/downtrend, respect it; else use bias
        tr = _trend_dir(side_txt) or trend_bias
        trend_cond = _ma_trend(which, period, want=tr)

        rsi_part = _rsi_bounce_dir("up" if tr == "up" else "down")
        stoch_part = _stoch_cross("up" if tr == "up" else "down")

        parts = [ma_respect, trend_cond, rsi_part, stoch_part]
        return " and ".join(f"({p})" for p in parts)

    buy_expr = side_to_expr(buy_txt, "buy")
    sell_expr = side_to_expr(sell_txt, "sell")
    return buy_expr, sell_expr
# --- Add this to rules.py ---

def get_symbol_strategies():
    """
    Return a list of strategies the worker should evaluate each cycle.
    Symbols shown use Deriv naming (e.g., frxEURUSD). Adjust if your
    data_fetch.py expects plain 'EURUSD' instead.
    """
    return [
        # ===== EURUSD (Deriv: frxEURUSD) =====
        {
            "name":   "EURUSD M1 BASE",
            "symbol": "frxEURUSD",
            "tf":     "M1",
            "expiry": "1m",
            "core":   "BASE",
            "cfg": {
                "indicators": {
                    "sma":   {"period": 20},
                    "rsi":   {"period": 14},
                    "stoch": {"k": 14, "d": 3},
                }
            }
        },
        {
            "name":   "EURUSD M1 TREND",
            "symbol": "frxEURUSD",
            "tf":     "M1",
            "expiry": "1m",
            "core":   "TREND",
            "cfg": {
                "indicators": {
                    "sma": {"period": 20},
                    "rsi": {"period": 14},
                }
            }
        },

        # ===== GBPUSD =====
        {
            "name":   "GBPUSD M1 BASE",
            "symbol": "frxGBPUSD",
            "tf":     "M1",
            "expiry": "1m",
            "core":   "BASE",
            "cfg": {
                "indicators": {
                    "sma":   {"period": 20},
                    "rsi":   {"period": 14},
                    "stoch": {"k": 14, "d": 3},
                }
            }
        },

        # ===== A more explicit CUSTOM example (tune thresholds as you like) =====
        {
            "name":   "EURUSD M1 CUSTOM",
            "symbol": "frxEURUSD",
            "tf":     "M1",
            "expiry": "1m",
            "core":   "CUSTOM",
            "cfg": {
                "indicators": {
                    "sma":   {"period": 20},
                    "rsi":   {"period": 14},
                    "stoch": {"k": 14, "d": 3},
                },
                # _eval_custom supports: price_above_sma/price_below_sma (bool),
                # rsi_gt/rsi_lt (number), stoch_cross_up/stoch_cross_dn (bool)
                "custom": {
                    "buy_rule":  {"price_above_sma": True, "rsi_gt": 52, "stoch_cross_up": True},
                    "sell_rule": {"price_below_sma": True, "rsi_lt": 48, "stoch_cross_dn": True}
                }
            }
        },
    ]
