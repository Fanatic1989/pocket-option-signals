import os, pandas as pd
from pathlib import Path
import requests

DATA = Path("data"); DATA.mkdir(exist_ok=True)
SIG  = DATA/"signals.csv"

TG_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN","")
CHANS = [os.getenv("TELEGRAM_CHAT_FREE",""), os.getenv("TELEGRAM_CHAT_BASIC",""),
         os.getenv("TELEGRAM_CHAT_PRO",""),  os.getenv("TELEGRAM_CHAT_VIP","")]

def send(text: str):
    if not TG_TOKEN: return
    url = f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage"
    for cid in CHANS:
        if not cid: continue
        r = requests.post(url, data={"chat_id": cid, "text": text, "parse_mode":"Markdown"})
        try: r.raise_for_status()
        except Exception as e: print("telegram error:", e, getattr(r,"text",""))

def per_pair(df):
    # exclude draws for accuracy
    x = df[df["outcome"]!="DRAW"].copy()
    if x.empty: return ""
    g = x.groupby("symbol_pretty")["outcome"]
    rows=[]
    for k,v in g:
        n=len(v); acc=round(100*v.eq("WIN").mean(),1)
        rows.append((k,n,acc))
    rows.sort(key=lambda t:(-t[2], -t[1], t[0]))
    top = rows[:8]
    lines = [f"• {k}: {acc}% ({n} trades)" for k,n,acc in top]
    return "\n".join(lines)

def build_daily(df):
    today = pd.Timestamp.utcnow().date()
    day = df[(df["status"]=="closed") & (df["ts_utc"].dt.date==today)]
    if day.empty: return ""
    wins = int((day["outcome"]=="WIN").sum())
    loss = int((day["outcome"]=="LOSS").sum())
    draw = int((day["outcome"]=="DRAW").sum())
    tot  = wins+loss+draw
    acc  = round(100 * wins / max(1, wins+loss), 1)
    pairs = per_pair(day)
    return (
      f"🏁 *VIP Official Tally — Today (UTC)*\n"
      f"Total trades: *{tot}*\nW: *{wins}* | L: *{loss}* | D: *{draw}*\n"
      f"Accuracy (excl. draws): *{acc}%*\n\n"
      f"*Per-pair (top 8)*:\n{pairs if pairs else '—'}\n\n"
      "Source: VIP signals (authoritative)."
    )

def build_weekly(df):
    now = pd.Timestamp.utcnow()
    wk = df[(df["status"]=="closed") & (df["ts_utc"] >= now - pd.Timedelta(days=7))]
    if wk.empty: return ""
    wins = int((wk["outcome"]=="WIN").sum())
    loss = int((wk["outcome"]=="LOSS").sum())
    draw = int((wk["outcome"]=="DRAW").sum())
    tot  = wins+loss+draw
    acc  = round(100 * wins / max(1, wins+loss), 1)
    pairs = per_pair(wk)
    return (
      f"📈 *VIP Weekly Tally — Last 7 days (UTC)*\n"
      f"Total trades: *{tot}*\nW: *{wins}* | L: *{loss}* | D: *{draw}*\n"
      f"Accuracy (excl. draws): *{acc}%*\n\n"
      f"*Per-pair (top 8)*:\n{pairs if pairs else '—'}\n\n"
      "Source: VIP signals (authoritative)."
    )

def main(period="daily"):
    if not SIG.exists(): return 0
    df = pd.read_csv(SIG)
    if df.empty or "status" not in df.columns: return 0
    df["ts_utc"] = pd.to_datetime(df["ts_utc"], utc=True, errors="coerce")
    df = df.dropna(subset=["ts_utc"])
    text = build_daily(df) if period=="daily" else build_weekly(df)
    if text: send(text)
    return 0

if __name__ == "__main__":
    import sys
    period = sys.argv[1] if len(sys.argv)>1 else "daily"
    raise SystemExit(main(period))
