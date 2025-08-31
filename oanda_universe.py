import os, sys, requests, yaml
from pathlib import Path

API = os.getenv("OANDA_API_KEY")
ENV = os.getenv("OANDA_ENV","practice").lower()
ACCT= os.getenv("OANDA_ACCOUNT_ID")
if not (API and ACCT):
    print("OANDA creds missing (.env).", file=sys.stderr); sys.exit(1)

HOST = "https://api-fxpractice.oanda.com" if ENV!="live" else "https://api-fxtrade.oanda.com"
HEAD = {"Authorization": f"Bearer {API}"}

def fetch_instruments():
    r = requests.get(f"{HOST}/v3/accounts/{ACCT}/instruments", headers=HEAD, timeout=30)
    r.raise_for_status()
    return r.json().get("instruments", [])

def group_for(inst):
    name = inst["name"].upper()
    typ  = inst.get("type","").upper()
    if typ=="CURRENCY":
        if "XAU" in name or "XAG" in name: return "COMMODITY"
        return "FX"
    if typ=="CFD":
        if any(k in name for k in ("WTI","BRENT","NGAS","NATGAS","OIL","USOIL","UKOIL")): return "COMMODITY"
        return "INDEX"
    return "OTHER"

rows=[]
for inst in fetch_instruments():
    rows.append({
        "src": "OANDA",
        "oanda": inst["name"],                    # e.g. EUR_USD, US30_USD, XAU_USD
        "name": inst.get("displayName", inst["name"].replace("_","/")),
        "group": group_for(inst),
        "enabled": True
    })

# Add a concise crypto set from Binance (expand later)
for sym, disp in [("BTCUSDT","BTC/USDT"),("ETHUSDT","ETH/USDT"),
                  ("SOLUSDT","SOL/USDT"),("XRPUSDT","XRP/USDT"),
                  ("LTCUSDT","LTC/USDT"),("DOGEUSDT","DOGE/USDT")]:
    rows.append({"src":"BINANCE","binance":sym,"name":disp,"group":"CRYPTO","enabled":True})

out={"symbols": rows}
Path("symbols.yaml").write_text(yaml.safe_dump(out, sort_keys=False), encoding="utf-8")
print(f"✅ Wrote {len(rows)} symbols to symbols.yaml")
