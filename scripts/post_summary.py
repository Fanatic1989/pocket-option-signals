import os, csv, requests
from pathlib import Path
import pandas as pd

DATA = Path("data"); DATA.mkdir(exist_ok=True)
SIG  = DATA / "signals.csv"

TG_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN","")
TG_IDS = {
    "FREE":  os.environ.get("TELEGRAM_CHAT_FREE",""),
    "BASIC": os.environ.get("TELEGRAM_CHAT_BASIC",""),
    "PRO":   os.environ.get("TELEGRAM_CHAT_PRO",""),
    "VIP":   os.environ.get("TELEGRAM_CHAT_VIP",""),
}

def send(text: str, chat_id: str):
    if not TG_TOKEN or not chat_id: 
        return
    url = f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage"
    r = requests.post(url, data={"chat_id": chat_id, "text": text, "parse_mode":"Markdown"})
    try:
        r.raise_for_status()
        print("✅ sent to", chat_id)
    except Exception as e:
        print("⚠️ telegram error:", e, "resp:", getattr(r, "text", ""))

def build_summary(df: pd.DataFrame) -> str:
    # TODAY, closed only
    df["ts_utc"] = pd.to_datetime(df["ts_utc"], utc=True, errors="coerce")
    df = df.dropna(subset=["ts_utc"])
    today = pd.Timestamp.utcnow().date()
    day = df[(df["status"]=="closed") & (df["ts_utc"].dt.date==today)].copy()
    if day.empty:
        return ""

    wins = int((day["outcome"]=="WIN").sum())
    loss = int((day["outcome"]=="LOSS").sum())
    draw = int((day["outcome"]=="DRAW").sum()) if "outcome" in day.columns else 0
    total = wins + loss + draw
    acc = round(100 * wins / max(1, (wins+loss)), 1)

    # Optional: tiny table of last few
    tail = day[["symbol_pretty","signal","outcome"]].tail(5)
    last_lines = "\n".join(
        f"• {('🟢' if r.signal=='BUY' else '🔴')} {r.symbol_pretty}: {r.signal} → {r.outcome}"
        for r in tail.itertuples()
    )

    text = (
      "🏁 *VIP Official Tally — Today (UTC)*\n"
      f"Total trades: *{total}*\n"
      f"W: *{wins}* | L: *{loss}* | D: *{draw}*\n"
      f"Accuracy (excl. draws): *{acc}%*\n\n"
      f"_Recent:_\n{last_lines if last_lines else '—'}\n\n"
      "Source: VIP signals (authoritative)."
    )
    return text

def main():
    if not SIG.exists():
        print("No signals.csv yet; nothing to summarize."); return 0
    df = pd.read_csv(SIG)
    if df.empty or "status" not in df.columns:
        print("No closed trades yet."); return 0

    text = build_summary(df)
    if not text:
        print("No closed trades today; skipping broadcast.")
        return 0

    # broadcast to all tiers
    for cid in [TG_IDS["FREE"], TG_IDS["BASIC"], TG_IDS["PRO"], TG_IDS["VIP"]]:
        if cid: send(text, cid)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
