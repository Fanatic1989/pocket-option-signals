import os, requests, yaml
from pathlib import Path

ACC  = os.getenv("OANDA_ACCOUNT_ID","").strip()
KEY  = os.getenv("OANDA_API_KEY","").strip()
ENV  = os.getenv("OANDA_ENV","practice").lower().strip()
HOST = "https://api-fxtrade.oanda.com" if ENV=="live" else "https://api-fxpractice.oanda.com"

if not ACC or not KEY:
    raise SystemExit("Set OANDA_ACCOUNT_ID and OANDA_API_KEY in .env and reload (set -a; . ./.env; set +a).")

r = requests.get(f"{HOST}/v3/accounts/{ACC}/instruments",
                 headers={"Authorization": f"Bearer {KEY}"}, timeout=30)
r.raise_for_status()
inst = r.json().get("instruments", [])

def yf_fx(name):
    # EUR_USD -> EURUSD=X ; fallback “=X” format for compatibility/evaluation
    b,q = name.split("_",1)
    return f"{b}{q}=X"

fx, metals, indices = [], [], []
for it in inst:
    name = it["name"]           # e.g., EUR_USD, XAU_USD, US30_USD
    typ  = it.get("type","")
    if typ=="CURRENCY" and "_" in name:
        b,q = name.split("_",1)
        fx.append({"oanda": name, "yf": yf_fx(name), "name": f"{b}/{q}", "group": "FX", "enabled": True})
    elif name in ("XAU_USD","XAG_USD"):
        nm = "GOLD (XAU/USD)" if name=="XAU_USD" else "SILVER (XAG/USD)"
        metals.append({"oanda": name, "yf": "XAUUSD=X" if name=="XAU_USD" else "XAGUSD=X",
                       "name": nm, "group": "COMMODITY", "enabled": True})
    elif name in ("US30_USD","NAS100_USD","SPX500_USD","DE30_EUR","UK100_GBP","JP225_USD"):
        pretty = {"US30_USD":"US30","NAS100_USD":"NAS100","SPX500_USD":"S&P 500",
                  "DE30_EUR":"DAX","UK100_GBP":"FTSE 100","JP225_USD":"Nikkei 225"}[name]
        indices.append({"oanda": name, "yf": pretty, "name": pretty, "group": "INDEX", "enabled": True})

symbols = fx + metals + indices
Path("symbols.yaml").write_text(yaml.safe_dump({"symbols": symbols}, sort_keys=False), encoding="utf-8")
print(f"symbols.yaml written with {len(symbols)} instruments (FX+metals+major indices).")
