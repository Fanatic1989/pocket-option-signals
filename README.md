# Pocket Option Signals (Serverless)

**What it does**
- Every 5 minutes: generates signals for enabled markets in `symbols.yaml`
- Logs each signal to `data/signals.csv`
- After `EXPIRY_MIN` (default 10m), evaluates WIN/LOSS and updates `data/signals.csv`
- Appends rolling accuracy to `data/perf.csv`
- Posts signals + build time to Telegram; posts a daily accuracy summary

**Setup**
1. Create Telegram bot (@BotFather) → get token
2. Create Telegram channel → add bot as Admin
3. In GitHub repo → Settings → Secrets and variables → Actions:
   - `TELEGRAM_BOT_TOKEN` = your bot token
   - `TELEGRAM_CHAT_ID`   = @YourChannelUsername
4. (Optional) Edit `symbols.yaml` to enable/disable markets

**Tuning (via env in workflow)**
- `INTERVAL` (default `5m`)
- `EXPIRY_MIN` (default `10`)
- `MIN_SCORE` (default `2`) — higher = fewer, stronger signals
- `RSI_BUY` / `RSI_SELL` (defaults 30/70)

**Disclaimers**
- Educational signals only. Binary options are high-risk. Not financial advice.
