# worker_inline.py
from __future__ import annotations
import time, traceback
from datetime import datetime, timezone
from typing import Dict, Any, List
import pandas as pd

# --- import strategy helpers: try 'strategy', then fallback to 'strategies'
try:
    from strategy import (
        _ensure_cols, _prep_indicators,
        _base_conditions, _trend_conditions, _chop_conditions, _eval_custom,
        _expiry_to_bars,
    )
except ImportError:
    from strategies import (
        _ensure_cols, _prep_indicators,
        _base_conditions, _trend_conditions, _chop_conditions, _eval_custom,
        _expiry_to_bars,
    )

# Candle source
from data_fetch import fetch_latest_candles

# Strategy list provider (from your rules.py)
try:
    from rules import get_symbol_strategies
except Exception:
    def get_symbol_strategies():
        return []

# Use the in-process ENGINE to avoid HTTP self-calls (deadlocks on 1 worker)
from live_engine import ENGINE

# ------------------ internal throttles / dedupe ------------------
_last_sent_key: Dict[str, str] = {}   # per symbol
_last_send_ts: List[float] = []
MIN_GAP_SEC  = 3

def _throttle_ok() -> bool:
    now = time.time()
    while _last_send_ts and now - _last_send_ts[0] > 60:
        _last_send_ts.pop(0)
    if _last_send_ts and now - _last_send_ts[-1] < MIN_GAP_SEC:
        return False
    return True

def _mark_sent(): _last_send_ts.append(time.time())

# ------------------ engine send ------------------
def _send_all_engine(text: str) -> Dict[str, Any]:
    results = {}
    for t in ["free", "basic", "pro", "vip"]:
        results[t] = ENGINE.send_to_tier(t, text)
    return {"ok": True, "results": results}

# ------------------ signal detection ------------------
def _detect_latest_signal(df: pd.DataFrame, core: str, cfg: Dict[str, Any],
                          tf: str, expiry: str):
    d = _ensure_cols(df)
    if len(d) < 30:
        return None, None, {"reason": "insufficient_bars"}

    ind = (cfg.get("indicators") or {}) if isinstance(cfg.get("indicators"), dict) else {}
    ip = {
        "sma_period": int((ind.get("sma") or {}).get("period", 20)),
        "rsi_period": int((ind.get("rsi") or {}).get("period", 14)),
        "stoch_k":    int((ind.get("stoch") or {}).get("k", 14)),
        "stoch_d":    int((ind.get("stoch") or {}).get("d", 3)),
    }
    d = _prep_indicators(d, ip)

    cu = (core or "BASE").upper()
    if cu == "TREND":
        buy, sell = _trend_conditions(d)
    elif cu == "CHOP":
        buy, sell = _chop_conditions(d)
    elif cu == "CUSTOM":
        custom = (cfg.get("custom") or {}) if isinstance(cfg.get("custom"), dict) else {}
        buy, sell = _eval_custom(d, custom)
    else:
        buy, sell = _base_conditions(d)

    i = len(d) - 2  # latest CLOSED bar
    if i < 1:
        return None, None, {"reason": "short_after_prep"}

    sig = "BUY" if bool(buy.iloc[i]) else ("SELL" if bool(sell.iloc[i]) else None)
    ctx = {
        "rsi": float(d["rsi"].iloc[i]) if "rsi" in d else None,
        "sma": float(d["sma"].iloc[i]) if "sma" in d else None,
        "bars_to_expiry": _expiry_to_bars(tf, expiry),
    }
    return sig, int(d["timestamp"].iloc[i]), ctx

# ------------------ public entry ------------------
def one_cycle(api_base: str, api_key: str | None) -> Dict[str, Any]:
    """
    Runs one pass over all strategies:
      * fetch candles (data_fetch.py / Deriv)
      * evaluate latest closed bar with strategy helpers
      * broadcast to all tiers via ENGINE (caps enforced)
    """
    out: Dict[str, Any] = {"sent": 0, "details": [], "errors": []}

    try:
        strategies = list(get_symbol_strategies())
        if not strategies:
            out["errors"].append("no_strategies_defined")
            return out

        for s in strategies:
            symbol = s.get("symbol", "EURUSD")
            tf     = (s.get("tf") or "M1").upper()
            expiry = s.get("expiry", "1m")
            core   = (s.get("core") or "BASE").upper()
            name   = s.get("name") or f"{symbol} {tf} {core}"
            cfg    = s.get("cfg", {}) or {}

            df = fetch_latest_candles(symbol, tf, limit=300)
            if df is None or df.empty or "close" not in df.columns:
                out["details"].append({"symbol": symbol, "tf": tf, "skipped": "empty_df"})
                continue

            sig, ref_ts, ctx = _detect_latest_signal(df, core, cfg, tf, expiry)
            if not sig:
                out["details"].append({"symbol": symbol, "tf": tf, "skipped": "no_signal"})
                continue

            key = f"{symbol}:{tf}:{core}:{ref_ts}:{sig}"
            if _last_sent_key.get(symbol) == key:
                out["details"].append({"symbol": symbol, "tf": tf, "skipped": "dedup"})
                continue

            if not _throttle_ok():
                out["details"].append({"symbol": symbol, "tf": tf, "skipped": "throttle"})
                continue

            nowz = datetime.now(timezone.utc).isoformat(timespec="seconds")
            msg = (
                f"⚡ <b>{symbol}</b> • {tf} • {name}\n"
                f"Signal: <b>{'CALL/BUY' if sig=='BUY' else 'PUT/SELL'}</b>\n"
                f"Expiry: {expiry}\n"
                f"Ref bar ts: {ref_ts}\n"
                f"Now: {nowz}Z\n"
                f"RSI: {ctx.get('rsi')} | SMA: {ctx.get('sma')} | bars→exp: {ctx.get('bars_to_expiry')}\n"
            )

            res = _send_all_engine(msg)
            ok = res.get("ok") is True or isinstance(res.get("results"), dict)
            out["details"].append({"symbol": symbol, "tf": tf, "sig": sig, "api": res})

            if ok:
                _last_sent_key[symbol] = key
                _mark_sent()
                out["sent"] += 1

        return out

    except Exception as e:
        out["errors"].append(f"{type(e).__name__}: {e}")
        out["errors"].append(traceback.format_exc(limit=2))
        return out
