import os, csv
from pathlib import Path
from datetime import datetime, timezone
import pandas as pd

# We depend on oanda_utils.oanda_price_at(instrument: str, when_utc: datetime) -> float
from oanda_utils import oanda_price_at

DATA = Path("data")
SIG  = DATA/"signals.csv"
EPS  = 1e-8  # draw tolerance

def yf_to_oanda(yf_sym: str) -> str:
    """
    Map Yahoo symbols like 'EURUSD=X' -> 'EUR_USD', 'XAUUSD=X' -> 'XAU_USD'.
    Fallback: pass-through upper-case.
    """
    if not yf_sym:
        return ""
    s = yf_sym.replace("-", "_")
    if s.endswith("=X"):
        s = s[:-2]
    u = s.upper()
    if u == "XAUUSD": return "XAU_USD"
    if u == "XAGUSD": return "XAG_USD"
    # 6 letters -> ABCDEF -> ABC_DEF
    if len(u) == 6 and u.isalpha():
        return f"{u[:3]}_{u[3:]}"
    return u

def load_signals_df() -> pd.DataFrame:
    if not SIG.exists():
        return pd.DataFrame()
    df = pd.read_csv(SIG)
    if df.empty:
        return df
    # normalize timestamps
    for col in ("ts_utc","evaluate_at_utc"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], utc=True, errors="coerce")
    # fill missing columns
    for col in ("status","result_price","outcome"):
        if col not in df.columns:
            df[col] = ""
    return df

def eval_outcome(side: str, entry_px: float, settle_px: float):
    # DRAW if virtually equal
    if abs(settle_px - entry_px) <= EPS:
        return "DRAW"
    if side == "BUY":
        return "WIN" if settle_px > entry_px else "LOSS"
    else:
        return "WIN" if settle_px < entry_px else "LOSS"

def main():
    # ensure numeric dtype for result_price
    try:
        df["result_price"] = pd.to_numeric(df.get("result_price", []), errors="coerce")
    except Exception: pass

    now = datetime.now(timezone.utc)
    df = load_signals_df()
    if df.empty:
        print("No signals file.")
        return

    due_mask = (df.get("status","").astype(str) != "closed")
    if "evaluate_at_utc" in df.columns:
        due_mask &= (df["evaluate_at_utc"].notna()) & (df["evaluate_at_utc"] <= now)
    due = df[due_mask].copy()
    if due.empty:
        print("No signals due for evaluation.")
        return

    changed = False
    for idx, r in due.iterrows():
        try:
            yf_sym   = (r.get("symbol_yf") or r.get("pair") or r.get("symbol") or "").strip()
            inst     = yf_to_oanda(yf_sym)
            side     = (r.get("signal") or r.get("side") or "").strip().upper()
            entry_px = float(r.get("price"))
            eval_at  = r["evaluate_at_utc"]
            if not inst or not side or pd.isna(eval_at):
                continue

            settle = float(oanda_price_at(inst, eval_at))
            outcome = eval_outcome(side, entry_px, settle)

            df.at[idx, "result_price"] = round(settle, 8)
            df.at[idx, "outcome"]      = outcome
            df.at[idx, "status"]       = "closed"
            changed = True
            print(f"✔ {yf_sym} @ {eval_at} → {settle:.8f} → {outcome}")
        except Exception as e:
            print(f"⚠️ pricing error for {r.get('symbol_yf', '?')} at {r.get('evaluate_at_utc')}: {e}")
            # leave row open; will retry on next run

    if changed:
        # write CSV preserving column order
        cols = list(df.columns)
        df.to_csv(SIG, index=False, columns=cols)
        print("Saved updates to", SIG)
    else:
        print("No changes.")

if __name__ == "__main__":
    main()
