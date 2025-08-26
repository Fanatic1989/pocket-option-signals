from pathlib import Path
from datetime import datetime, timezone
import pandas as pd, os, requests

DATA = Path("data"); DATA.mkdir(exist_ok=True)
SIG  = DATA/"signals.csv"
if not SIG.exists():
    raise SystemExit(0)

df = pd.read_csv(SIG)
if df.empty or "status" not in df.columns:
    raise SystemExit(0)

# normalize timestamps to UTC date
df["ts_utc"] = pd.to_datetime(df["ts_utc"], utc=True, errors="coerce")
df = df.dropna(subset=["ts_utc"])
today = datetime.now(timezone.utc).date()

day_df = df[(df["status"]=="closed") & (df["ts_utc"].dt.date==today)]
if day_df.empty:
    raise SystemExit(0)

wins = int((day_df["outcome"]=="WIN").sum())
loss = int((day_df["outcome"]=="LOSS").sum())
draw = int((day_df["outcome"]=="DRAW").sum())
total = wins + loss + draw
acc = round(100 * wins / max(1, (wins + loss)), 1)

text = (
    f"📊 *Session Summary* — {today} UTC\n"
    f"Total: {total}\n"
    f"Wins: {wins} | Losses: {loss} | Draws: {draw}\n"
    f"Accuracy (excl. draws): {acc}%"
)

token = os.environ["TELEGRAM_BOT_TOKEN"]
chat  = os.environ["TELEGRAM_CHAT_ID"]
url = f"https://api.telegram.org/bot{token}/sendMessage"
resp = requests.post(url, data={"chat_id": chat, "text": text, "parse_mode": "Markdown"}, timeout=30)
resp.raise_for_status()
