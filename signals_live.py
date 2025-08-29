import os, json, time, requests, hashlib

from hashlib import sha1
from pathlib import Path

def _strip_no_setups(msg: str) -> str:
    lines = [ln for ln in msg.splitlines() if not ln.lstrip().startswith("⚪")]
    return "\n".join(lines).strip()

def _has_signal(msg: str) -> bool:
    return ("🟢" in msg) or ("🔴" in msg)

def _is_duplicate(msg: str, ttl_sec: int = 1800) -> bool:
    data = Path("data"); data.mkdir(exist_ok=True, parents=True)
    f = data / "last_sent.sha1"
    now = int(__import__("time").time())
    h = sha1(msg.encode("utf-8")).hexdigest()
    if f.exists():
        try:
            old = f.read_text().strip().split(",")
            old_hash, old_ts = (old[0], int(old[1]))
            if old_hash == h and (now - old_ts) < ttl_sec:
                print("🔁 Duplicate within TTL — suppressed.")
                return True
        except Exception:
            pass
    f.write_text(f"{h},{now}")
    return False
from pathlib import Path
from datetime import datetime, timezone, timedelta

import pandas as pd
import pandas_ta as ta
import yaml

# --------- ENV / Defaults ----------
INTERVAL    = os.getenv("INTERVAL", "1m")     # candle size shown in header (fetch is M1 from OANDA)
EXPIRY_MIN  = int(os.getenv("EXPIRY_MIN", "5"))
MIN_SCORE   = int(os.getenv("MIN_SCORE", "1"))
MUST_TRADE  = int(os.getenv("MUST_TRADE", "0"))
SUPPRESS_EMPTY = int(os.getenv("SUPPRESS_EMPTY", "1"))
MAX_PER_RUN = int(os.getenv("MAX_PER_RUN", "8"))

# Hours/Weekdays per asset class (UTC). Change in Actions env if needed.
FX_HOURS_UTC        = os.getenv("FX_HOURS_UTC",        "0700-1700")
COMMODITY_HOURS_UTC = os.getenv("COMMODITY_HOURS_UTC", "0700-1700")
INDEX_HOURS_UTC     = os.getenv("INDEX_HOURS_UTC",     "1330-2000")
STOCK_HOURS_UTC     = os.getenv("STOCK_HOURS_UTC",     "1330-2000")

WEEKDAYS_FX         = os.getenv("WEEKDAYS_FX",         "12345")
WEEKDAYS_COMMODITY  = os.getenv("WEEKDAYS_COMMODITY",  "12345")
WEEKDAYS_INDEX      = os.getenv("WEEKDAYS_INDEX",      "12345")
WEEKDAYS_STOCK      = os.getenv("WEEKDAYS_STOCK",      "12345")

# OANDA creds (set in .env locally / Secrets in Actions)
OANDA_API_KEY = os.getenv("OANDA_API_KEY","")
OANDA_ENV     = os.getenv("OANDA_ENV","practice").lower()
OANDA_HOST    = "https://api-fxpractice.oanda.com" if OANDA_ENV != "live" else "https://api-fxtrade.oanda.com"

DATA_DIR     = Path("data"); DATA_DIR.mkdir(exist_ok=True)
SIGNALS_CSV  = DATA_DIR/"signals.csv"
STATE_FILE   = DATA_DIR/"last_sent.json"  # de-dup per symbol/timestamp
SYMBOLS_YML  = Path("symbols.yaml")

# ------------- utils -------------
def is_session_open(group: str, now: datetime) -> bool:
    """Respect NO-OTC windows by asset class."""
    if group.upper() == "FX":
        hours, wd = FX_HOURS_UTC, WEEKDAYS_FX
    elif group.upper() == "COMMODITY":
        hours, wd = COMMODITY_HOURS_UTC, WEEKDAYS_COMMODITY
    elif group.upper() == "INDEX":
        hours, wd = INDEX_HOURS_UTC, WEEKDAYS_INDEX
    else:
        hours, wd = STOCK_HOURS_UTC, WEEKDAYS_STOCK

    # weekdays like "12345" (Mon=1 .. Sun=7)
    if str(((now.isoweekday()))) not in set(list(wd)):
        return False

    try:
        start, end = hours.split("-")
        hhmm = int(now.strftime("%H%M"))
        return int(start) <= hhmm <= int(end)
    except Exception:
        # if malformed env, be conservative: off
        return False

def load_symbols():
    cfg = yaml.safe_load(open(SYMBOLS_YML, "r"))
    rows = []
    for s in cfg.get("symbols", []):
        if not s.get("enabled", True): continue
        yf = s.get("yf")
        pretty = s.get("name", yf)
        group = str(s.get("group", "FX")).upper()
        if not yf: continue
        # map Yahoo FX 'EURUSD=X' -> OANDA 'EUR_USD'; BTC-USD -> BTC_USD
        core = yf.replace("=X","").replace("-","").upper()  # EURUSD, BTCUSD, XAUUSD, etc
        if len(core) >= 6:
            oanda = core[:3]+"_"+core[3:]
        else:
            # fallback: leave as-is (you can extend mapping later)
            oanda = core
        rows.append((yf, pretty, group, oanda))
    return rows

def oanda_fetch_m1(instrument: str, count: int = 220) -> pd.DataFrame:
    if not OANDA_API_KEY:
        raise RuntimeError("OANDA_API_KEY not set")
    url = f"{OANDA_HOST}/v3/instruments/{instrument}/candles"
    r = requests.get(url, params={"granularity":"M1", "count":count, "price":"M"},
                     headers={"Authorization": f"Bearer {OANDA_API_KEY}"}, timeout=25)
    r.raise_for_status()
    js = r.json()
    recs = []
    for c in js.get("candles", []):
        if not c.get("complete"):  # skip forming candle
            continue
        mid = c["mid"]
        recs.append({
            "ts": datetime.fromisoformat(c["time"].replace("Z","+00:00")),
            "open": float(mid["o"]),
            "high": float(mid["h"]),
            "low":  float(mid["l"]),
            "close":float(mid["c"]),
        })
    if not recs:
        return pd.DataFrame()
    df = pd.DataFrame(recs).set_index("ts").sort_index()
    return df

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    macd = ta.macd(df["close"], fast=12, slow=26, signal=9)
    df["macd"]  = macd["MACD_12_26_9"]
    df["macds"] = macd["MACDs_12_26_9"]
    df["rsi"]   = ta.rsi(df["close"], length=14)
    df["ema50"] = ta.ema(df["close"], length=50)
    df["ema200"]= ta.ema(df["close"], length=200)
    return df.dropna()

def classify(prev, cur, min_score=1):
    score, reasons, side = 0, [], None
    uptrend = cur["ema50"] > cur["ema200"]
    reasons.append("EMA50>EMA200" if uptrend else "EMA50<EMA200")

    macd_up   = (prev["macd"] <= prev["macds"]) and (cur["macd"] > cur["macds"])
    macd_down = (prev["macd"] >= prev["macds"]) and (cur["macd"] < cur["macds"])
    if macd_up:   reasons.append("MACD↑"); score += 1
    if macd_down: reasons.append("MACD↓"); score += 1

    # RSI near 50 as scalper trigger with MACD confirmation
    if cur["rsi"] >= 50 and macd_down:
        side = "SELL"; reasons.append("RSI≥50")
    elif cur["rsi"] <= 50 and macd_up:
        side = "BUY";  reasons.append("RSI≤50")

    if side and score < min_score:
        side = None
    return side, score, ", ".join(reasons)

def load_state():
    if STATE_FILE.exists():
        try: return json.load(open(STATE_FILE))
        except: pass
    return {}

def save_state(st):
    json.dump(st, open(STATE_FILE,"w"))

def already_sent(st, key: str) -> bool:
    # key = hash(symbol + last_bar_ts + side)
    when = st.get(key)
    if not when: return False
    # keep de-dup window 15 minutes
    try:
        ts = datetime.fromisoformat(when)
        return (datetime.now(timezone.utc) - ts) < timedelta(minutes=15)
    except Exception:
        return True

def mark_sent(st, key: str):
    st[key] = datetime.now(timezone.utc).isoformat()

# ------------- main -------------
def main():
    now = datetime.now(timezone.utc)
    header = [
        f"📡 Pocket Option Signals — {now:%Y-%m-%d %H:%M UTC}",
        f"Candle: {INTERVAL} | Expiry: {EXPIRY_MIN}m",
        ""
    ]
    out_lines = []
    sent_count = 0

    st = load_state()
    universe = load_symbols()

    for yf_sym, pretty, group, oanda_inst in universe:
        if sent_count >= MAX_PER_RUN:
            break

        if not is_session_open(group, now):
            # NO OTC — skip outside session silently
            continue

        try:
            df = oanda_fetch_m1(oanda_inst, count=220)
            if df.empty or len(df) < 60:
                continue
            df = add_indicators(df)
            prev, cur = df.iloc[-2], df.iloc[-1]
            side, score, why = classify(prev, cur, MIN_SCORE)
            if not side and not MUST_TRADE:
                continue

            px = float(cur["close"])
            side = side or ("BUY" if px >= px else "BUY")  # never hit; just a guard if MUST_TRADE set

            # de-dup key
            last_bar_ts = df.index[-1].isoformat()
            key_raw = f"{oanda_inst}|{last_bar_ts}|{side}"
            key = hashlib.sha1(key_raw.encode()).hexdigest()[:16]
            if already_sent(st, key):
                continue

            emoji = "🟢" if side == "BUY" else "🔴"
            out_lines.append(f"{emoji} {pretty} — {side} @ {px:.5f}")
            out_lines.append(f"• {why} (score {score if side else 0})")
            out_lines.append("")  # spacer

            # log to CSV (append)
            try:
                exists = SIGNALS_CSV.exists()
                with open(SIGNALS_CSV, "a") as f:
                    if not exists:
                        f.write("ts_utc,symbol_yf,symbol_pretty,signal,price,expiry_min,evaluate_at_utc,status,score,why\n")
                    eval_at = now + timedelta(minutes=EXPIRY_MIN)
                    f.write(",".join([
                        now.strftime("%Y-%m-%d %H:%M:%S"),
                        yf_sym,
                        pretty.replace(",","/"),
                        side,
                        f"{px:.8f}",
                        str(EXPIRY_MIN),
                        eval_at.strftime("%Y-%m-%d %H:%M:%S"),
                        "open",
                        str(score if side else 0),
                        why.replace(",",";")
                    ]) + "\n")
            except Exception:
                pass

            mark_sent(st, key)
            sent_count += 1

        except Exception as e:
            # keep quiet on errors (don’t spam)
            continue

    save_state(st)

    # Only return confirmed trades. If empty and SUPPRESS_EMPTY=1, caller will skip sending.
    return header + out_lines

# --------- send ----------
def send():
    from telegram_send import send_to_tiers
    lines = main()
    # filter empties: only send if at least one 🟢/🔴 present or SUPPRESS_EMPTY=0
    txt = "\n".join(lines).strip()
    has_signal = ("🟢" in txt) or ("🔴" in txt)
    if not has_signal and SUPPRESS_EMPTY:
        print("No confirmed setups; suppressed.")
        return
    send_to_tiers(txt)

if __name__ == "__main__":
    # MUST_TRADE respected by strategy; enforce again here
    MT = os.getenv("MUST_TRADE", "0") == "1"

    out = main() or []
    # Build message
    msg = "
".join(out)
    # Remove "no setup" rows
    msg2 = _strip_no_setups(msg)

    # If nothing confirmed and not forcing trades -> do not send
    if not _has_signal(msg2) and not MT:
        print("ℹ️ No confirmed setups — not sending.")
        raise SystemExit(0)

    # De-duplicate last send (30 min TTL)
    if _is_duplicate(msg2, ttl_sec=1800):
        raise SystemExit(0)

    # Ship
    send_to_tiers(msg2)
    print("✅ Signals run completed.")
