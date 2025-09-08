# Pocket Option Signals (Flask)

Modular dashboard for binary options signals with indicator toggles, natural-language rules, Deriv pull, and backtester.

## Features
- Indicator toggles (EMA, SMA, WMA, SMMA, TMA, RSI, MACD, ADX, BB, etc.)
- Strategies: **BASE**, **TREND**, **CHOP**, **CUSTOM**
- Simple-English → DSL parser for custom criteria
- Binary backtester with TF: M1..D1 and expiry: 1m, 3m, 5m, 30m, H1, H4
- Deriv candle fetch (websocket) + server CSV
- Uptime endpoints for UptimeRobot: `/_up` and `/health` (HEAD/GET)
- Procfile for Render deployment

## Setup (Local)
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python app.py
# open: http://localhost:8000/
```
Login password (default): `admin123` (change with `ADMIN_PASSWORD` env).

## Run with Gunicorn (Prod)
```bash
gunicorn app:app --workers=2 --threads=2 --timeout=120
```

## Deploy on Render (GitHub)
1) Connect your GitHub account to Render and select the repo **Fanatic1989/pocket-option-signals**.
2) Render will auto-detect the **Procfile** and use Gunicorn.
3) Set environment variables (optional):
   - `FLASK_SECRET` (random string)
   - `ADMIN_PASSWORD` (your admin pass)
   - `TIMEZONE` (default `America/Port_of_Spain`)

## Backtesting data
- Upload a CSV with columns: `timestamp,open,high,low,close` **or**
- Use **Pull from Deriv** (saves `/tmp/deriv_last_month.csv`) then tick **Use server data**.

## Custom Rules
Example simple BUY rule:
> price respect 50 sma and rsi bounce off mid line upwards and stochastic lines cross upwards

If parsing fails, switch to Expert and use DSL.

---
Made for Chris. Enjoy 🚀
