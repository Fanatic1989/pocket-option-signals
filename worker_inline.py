# worker_inline.py
from __future__ import annotations
import time, json, traceback
from datetime import datetime, timezone
from typing import Dict, Any, List
import requests
import pandas as pd

from data_fetch import fetch_latest_candles
from strategy import (
    _ensure_cols, _prep_indicators,
    _base_conditions, _trend_conditions, _chop_conditions, _eval_custom,
    _expiry_to_bars,
)
try:
    from rules import get_symbol_strategies
except Exception:
    def get_symbol_strategies():
        return []

# web API wiring (same as your app)
CORE_SEND_URL   = None
CORE_STATUS_URL = None
HEADERS = {}

_last_sent_key: Dict[str, str] = {}   # symbol -> last key
MIN_GAP_SEC  = 3
_last_send_ts: List[float] = []

def _init(api_base: str, api_key: str|None):
    global CORE_SEND_URL, CORE_STATUS_URL, HEADERS
    CORE_SEND_URL   = f"{api_base.rstrip('/')}/api/core/send"
    CORE_STATUS_URL = f"{api_base.rstrip('/')}/api/core/status"
    HEADERS = {"Content-Type": "application/json"}
    if api_key: HEADERS["X-API-Key"] = api_key

def _throttle_ok() -> bool:
    now = time.time()
    while _last_send_ts and now - _last_send_ts[0] > 60:
        _last_send_ts.pop(0)
    if _last_send_ts and now - _last_send_ts[-1] < MIN_GAP_SEC:
        return False
    return True

def _mark_sent(): _last_send_ts.append(time.time())

def _send_all(text: str) -> Dict[str, Any]:
    r = requests.post(CORE_SEND_URL, json={"tier":"all","text":text}, headers=HEADERS, timeout=15)
    try: return r.json()
    except Exception: return {"ok": False, "error": f"HTTP {r.status_code}", "raw": r.text[:300]}

def _detect_latest_signal(df: pd.DataFrame, core: str, cfg: Dict[str, Any], tf: str, expiry: str):
    d = _ensure_cols(df)
    if len(d) < 30:
        return None, None, {"reason":"insufficient_bars"}
    ind = (cfg.get("indicators") or {}) if isinstance(cfg.get("indicators"), dict) else {}
    ip = {
        "sma_period": int((ind.get("sma") or {}).get("period", 50)),
        "rsi_period": int((ind.get("rsi") or {}).get("period", 14)),
        "stoch_k":    int((ind.get("stoch") or {}).get("k", 14)),
        "stoch_d":    int((ind.get("stoch") or {}).get("d", 3)),
    }
    d = _prep_indicators(d, ip)
    cu = (core or "BASE").upper()
    if cu == "TREND":   buy, sell = _trend_conditions(d)
    elif cu == "CHOP":  buy, sell = _chop_conditions(d)
    elif cu == "CUSTOM":
        custom = (cfg.get("custom") or {}) if isinstance(cfg.get("custom"), dict) else {}
        buy, sell = _eval_custom(d, custom)
    else:
        buy, sell = _base_conditions(d)
    i = len(d)-2
    if i < 1: return None, None, {"reason":"short_after_prep"}
    sig = "BUY" if bool(buy.iloc[i]) else ("SELL" if bool(sell.iloc[i]) else None)
    ctx = {"rsi": float(d.get("rsi", pd.Series([None]*len(d))).iloc[i]) if "rsi" in d else None,
           "sma": float(d.get("sma", pd.Series([None]*len(d))).iloc[i]) if "sma" in d else None,
           "bars_to_expiry": _expiry_to_bars(tf, expiry)}
    return sig, int(d["timestamp"].iloc[i]), ctx

def one_cycle(api_base: str, api_key: str|None) -> Dict[str, Any]:
    """Run one pass over all strategies; returns a JSON summary."""
    _init(api_base, api_key)
    out = {"sent": 0, "errors": [], "details": []}
    try:
        for s in get_symbol_strategies():
            symbol = s.get("symbol","EURUSD")
            tf     = (s.get("tf") or "M1").upper()
            expiry = s.get("expiry","1m")
            core   = (s.get("core") or "BASE").upper()
            name   = s.get("name") or f"{symbol} {tf} {core}"
            cfg    = s.get("cfg", {}) or {}

            df = fetch_latest_candles(symbol, tf, limit=300)
            if df is None or df.empty or "close" not in df.columns:
                out["details"].append({"symbol":symbol, "tf":tf, "skipped":"empty_df"})
                continue

            sig, ref_ts, ctx = _detect_latest_signal(df, core, cfg, tf, expiry)
            if not sig:
                out["details"].append({"symbol":symbol, "tf":tf, "skipped":"no_signal"})
                continue

            key = f"{symbol}:{tf}:{core}:{ref_ts}:{sig}"
            if _last_sent_key.get(symbol) == key:
                out["details"].append({"symbol":symbol, "tf":tf, "skipped":"dedup"})
                continue

            if not _throttle_ok():
                out["details"].append({"symbol":symbol, "tf":tf, "skipped":"throttle"})
                continue

            nowz = datetime.now(timezone.utc).isoformat(timespec="seconds")
            msg = (f"⚡ <b>{symbol}</b> • {tf} • {name}\n"
                   f"Signal: <b>{'CALL/BUY' if sig=='BUY' else 'PUT/SELL'}</b>\n"
                   f"Expiry: {expiry}\n"
                   f"Ref bar ts: {ref_ts}\n"
                   f"Now: {nowz}Z\n"
                   f"RSI: {ctx.get('rsi')} | SMA: {ctx.get('sma')} | bars→exp: {ctx.get('bars_to_expiry')}\n")
            res = _send_all(msg)
            ok = res.get("ok") is True or isinstance(res.get("results"), dict)
            _last_sent_key[symbol] = key if ok else _last_sent_key.get(symbol, "")
            if ok:
                _mark_sent()
                out["sent"] += 1
            out["details"].append({"symbol":symbol, "tf":tf, "sig":sig, "api":res})
        return out
    except Exception as e:
        tb = traceback.format_exc(limit=2)
        out["errors"].append(f"{type(e).__name__}: {e}")
        out["errors"].append(tb)
        return out
