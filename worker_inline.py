# worker_inline.py
from __future__ import annotations
import os
import json
import time
import traceback
import re
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Tuple

import pandas as pd

# ------------------ Try imports (be tolerant) ------------------
# Strategy helpers: prefer `strategy.py` then `strategies.py`
try:
    from strategy import (
        _ensure_cols, _prep_indicators,
        _base_conditions, _trend_conditions, _chop_conditions, _eval_custom,
        _expiry_to_bars, _wide_conditions
    )
except Exception:
    try:
        from strategies import (
            _ensure_cols, _prep_indicators,
            _base_conditions, _trend_conditions, _chop_conditions, _eval_custom,
            _expiry_to_bars, _wide_conditions
        )
    except Exception:
        # If imports fail, create safe stubs so worker reports helpful error
        def _ensure_cols(df: pd.DataFrame) -> pd.DataFrame: raise ImportError("missing strategy helpers")
        def _prep_indicators(d, p): raise ImportError("missing strategy helpers")
        def _base_conditions(d): raise ImportError("missing strategy helpers")
        def _trend_conditions(d): raise ImportError("missing strategy helpers")
        def _chop_conditions(d): raise ImportError("missing strategy helpers")
        def _eval_custom(d, c): raise ImportError("missing strategy helpers")
        def _expiry_to_bars(tf, expiry): return 1
        def _wide_conditions(d): raise ImportError("missing strategy helpers")

# Data fetch - your project should expose a function to fetch candles
try:
    from data_fetch import fetch_latest_candles
except Exception:
    # stub that raises so the worker fails loudly if missing
    def fetch_latest_candles(symbol: str, tf: str, limit: int = 300) -> Optional[pd.DataFrame]:
        raise ImportError("data_fetch.fetch_latest_candles is missing")

# Strategies provider (rules.py)
try:
    from rules import get_symbol_strategies
except Exception:
    def get_symbol_strategies() -> List[Dict[str, Any]]:
        return []

# Engine (send to Telegram)
try:
    from live_engine import ENGINE
except Exception:
    class _StubEngine:
        def send_to_tier(self, tier, text): return {"ok": False, "error": "ENGINE not available"}
    ENGINE = _StubEngine()

# ------------------ Config loader ------------------
def _load_local_config() -> Dict[str, Any]:
    """
    Load optional config.json from project root. Expected keys:
      - live_tf (e.g. "M1")
      - live_expiry (e.g. "5m")
    """
    cfg = {}
    try:
        here = os.path.dirname(__file__) or "."
        path = os.path.join(here, "config.json")
        if os.path.exists(path):
            with open(path, "r", encoding="utf8") as fh:
                cfg = json.load(fh) or {}
    except Exception:
        cfg = {}
    return cfg

_LOCAL_CFG = _load_local_config()

# ------------------ Normalizers ------------------
def normalize_tf(raw: Any) -> str:
    if not raw: return os.getenv("LIVE_TF", "M1").upper()
    s = str(raw).strip().upper()
    return s

def normalize_expiry(raw: Any) -> str:
    """
    Accepts numbers, '5', '5m', '5min', '300s', '1h', etc.
    Returns canonical string like '5m', '300s', '1h'.
    """
    if raw is None:
        return os.getenv("LIVE_EXPIRY", _LOCAL_CFG.get("live_expiry", "5m"))
    if isinstance(raw, (int, float)):
        return f"{int(raw)}m"
    s = str(raw).strip().lower()
    if s.isdigit():
        return f"{int(s)}m"
    # minutes forms
    m = re.match(r"^(\d+)\s*(m|min|mins|minute|minutes)$", s)
    if m:
        return f"{int(m.group(1))}m"
    # seconds forms
    m2 = re.match(r"^(\d+)\s*(s|sec|secs|second|seconds)$", s)
    if m2:
        secs = int(m2.group(1))
        if secs % 60 == 0:
            return f"{secs // 60}m"
        return f"{secs}s"
    # hours
    m3 = re.match(r"^(\d+)\s*(h|hr|hour|hours)$", s)
    if m3:
        return f"{int(m3.group(1))}h"
    # already something like '5m','1h' -> return as-is
    return s

# ------------------ Throttling / dedupe ------------------
_last_sent_key: Dict[str, str] = {}
_last_send_timestamps: List[float] = []
MIN_GAP_SEC = float(os.getenv("WORKER_MIN_GAP_SEC", "2"))

def _throttle_ok() -> bool:
    now = time.time()
    # purge older than 60s to keep list small
    while _last_send_timestamps and now - _last_send_timestamps[0] > 60:
        _last_send_timestamps.pop(0)
    if _last_send_timestamps and now - _last_send_timestamps[-1] < MIN_GAP_SEC:
        return False
    return True

def _mark_sent() -> None:
    _last_send_timestamps.append(time.time())

# ------------------ Signal detection helper ------------------
def _detect_latest_signal_from_df(df: pd.DataFrame, core: str, cfg: Dict[str, Any], tf: str, expiry: str) -> Tuple[Optional[str], Optional[int], Dict[str, Any]]:
    """
    Returns (sig['BUY'|'SELL'|None], ref_bar_timestamp, ctx)
    ctx contains rsi/sma/close/sk/sd/bars_to_expiry to assist debugging.
    """
    try:
        d = _ensure_cols(df)
    except Exception as e:
        return None, None, {"reason": f"ensure_cols_error: {e}"}

    if len(d) < 10:
        return None, None, {"reason": "insufficient_bars", "len": len(d)}

    # prepare indicator params from cfg
    ind = (cfg.get("indicators") or {}) if isinstance(cfg, dict) else {}
    ip = {
        "sma_period": int((ind.get("sma") or {}).get("period", ind.get("sma_period", 20))),
        "rsi_period": int((ind.get("rsi") or {}).get("period", ind.get("rsi_period", 14))),
        "stoch_k":    int((ind.get("stoch") or {}).get("k", ind.get("stoch_k", 14))),
        "stoch_d":    int((ind.get("stoch") or {}).get("d", ind.get("stoch_d", 3))),
    }

    try:
        d = _prep_indicators(d, ip)
    except Exception as e:
        return None, None, {"reason": f"prep_indicators_error: {e}"}

    cu = (core or "BASE").upper()
    try:
        if cu == "TREND":
            buy, sell = _trend_conditions(d)
        elif cu == "CHOP":
            buy, sell = _chop_conditions(d)
        elif cu == "WIDE":
            buy, sell = _wide_conditions(d)
        elif cu == "CUSTOM":
            custom = cfg.get("custom", {}) if isinstance(cfg, dict) else {}
            buy, sell = _eval_custom(d, custom)
        else:
            buy, sell = _base_conditions(d)
    except Exception as e:
        return None, None, {"reason": f"core_eval_error: {e}"}

    # latest closed bar is index -2 (last is in-progress)
    i = len(d) - 2
    if i < 1:
        return None, None, {"reason": "not_enough_closed_bars"}

    ctx = {
        "rsi": float(d["rsi"].iloc[i]) if "rsi" in d else None,
        "sma": float(d["sma"].iloc[i]) if "sma" in d else None,
        "close": float(d["close"].iloc[i]) if "close" in d else None,
        "sk": float(d["sk"].iloc[i]) if "sk" in d else None,
        "sd": float(d["sd"].iloc[i]) if "sd" in d else None,
        "bars_to_expiry": _expiry_to_bars(tf, expiry) if callable(_expiry_to_bars) else None,
    }

    sig = None
    try:
        if bool(buy.iloc[i]):
            sig = "BUY"
        elif bool(sell.iloc[i]):
            sig = "SELL"
    except Exception:
        sig = None

    return sig, int(d["timestamp"].iloc[i]) if "timestamp" in d.columns else None, ctx

# ------------------ Engine sender wrapper ------------------
def _send_all_via_engine(text: str) -> Dict[str, Any]:
    """Sends the same text to all tiers via live_engine.ENGINE"""
    results = {}
    for t in ["free", "basic", "pro", "vip"]:
        try:
            results[t] = ENGINE.send_to_tier(t, text)
        except Exception as e:
            results[t] = {"ok": False, "error": str(e)}
    return {"ok": True, "results": results}

# ------------------ Public entrypoint ------------------
def one_cycle(api_base: str = "", api_key: str | None = None) -> Dict[str, Any]:
    """
    Run one pass over get_symbol_strategies() items.
    - api_base and api_key are optional (kept for compatibility)
    - returns summary dict: { sent:int, details:[...], errors:[...] }
    """
    out: Dict[str, Any] = {"sent": 0, "details": [], "errors": []}

    try:
        strategies = list(get_symbol_strategies() or [])
        if not strategies:
            out["errors"].append("no_strategies_defined")
            return out

        # global fallbacks
        cfg_tf = normalize_tf(_LOCAL_CFG.get("live_tf") or os.getenv("LIVE_TF", "M1"))
        cfg_expiry = normalize_expiry(_LOCAL_CFG.get("live_expiry") or os.getenv("LIVE_EXPIRY", "5m"))

        for s in strategies:
            try:
                symbol = s.get("symbol") or s.get("sym") or s.get("name") or "EURUSD"
                tf_raw = s.get("tf") or s.get("timeframe") or cfg_tf
                tf = normalize_tf(tf_raw)
                expiry_raw = s.get("expiry") or s.get("live_expiry") or cfg_expiry
                expiry = normalize_expiry(expiry_raw)
                core = (s.get("core") or "BASE").upper()
                name = s.get("name") or f"{symbol} {tf} {core}"
                cfg = s.get("cfg") or {}

                # fetch candles
                try:
                    df = fetch_latest_candles(symbol, tf, limit=int(s.get("count", 300) or 300))
                except Exception as e:
                    out["details"].append({"symbol": symbol, "tf": tf, "skipped": "fetch_error", "error": str(e)})
                    continue

                if df is None or df.empty:
                    out["details"].append({"symbol": symbol, "tf": tf, "skipped": "empty_df"})
                    continue

                sig, ref_ts, ctx = _detect_latest_signal_from_df(df, core, cfg, tf, expiry)
                if not sig:
                    # include expiry used for debugging
                    ctx["expiry_used"] = expiry
                    out["details"].append({"symbol": symbol, "tf": tf, "skipped": "no_signal", "ctx": ctx})
                    continue

                # dedupe key per symbol (avoid repeat same signal from same ref bar)
                key = f"{symbol}:{tf}:{core}:{ref_ts}:{sig}:{expiry}"
                if _last_sent_key.get(symbol) == key:
                    out["details"].append({"symbol": symbol, "tf": tf, "skipped": "dedup", "ref_ts": ref_ts})
                    continue

                if not _throttle_ok():
                    out["details"].append({"symbol": symbol, "tf": tf, "skipped": "throttle"})
                    continue

                nowz = datetime.now(timezone.utc).isoformat(timespec="seconds")
                direction_text = "CALL/BUY" if sig == "BUY" else "PUT/SELL"
                # Ensure expiry displayed is the normalized expiry (expiry variable above)
                msg = (
                    f"⚡ <b>{symbol}</b> • {tf} • {name}\n"
                    f"Signal: <b>{direction_text}</b>\n"
                    f"Expiry: {expiry}\n"
                    f"Ref bar ts: {ref_ts}\n"
                    f"Now: {nowz}Z\n"
                    f"RSI: {ctx.get('rsi')} | SMA: {ctx.get('sma')} | bars→exp: {ctx.get('bars_to_expiry')}\n"
                )

                # send via ENGINE
                res = _send_all_via_engine(msg)
                out["details"].append({"symbol": symbol, "tf": tf, "sig": sig, "expiry_used": expiry, "api": res})

                # update dedupe and throttle
                _last_sent_key[symbol] = key
                _mark_sent()
                out["sent"] += 1

            except Exception as e:
                out["errors"].append(f"symbol_loop_error:{symbol}:{type(e).__name__}:{e}")
                out["errors"].append(traceback.format_exc(limit=2))
                # continue with next symbol
                continue

        return out

    except Exception as e:
        out["errors"].append(f"worker_error:{type(e).__name__}:{e}")
        out["errors"].append(traceback.format_exc())
        return out

# Allow importing one_cycle directly
__all__ = ["one_cycle"]
