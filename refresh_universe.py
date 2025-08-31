import os, json, requests, sys, yaml, time
from pathlib import Path

OANDA_API_KEY  = os.getenv("OANDA_API_KEY")
OANDA_ENV      = os.getenv("OANDA_ENV","practice").lower()
OANDA_ACCOUNT  = os.getenv("OANDA_ACCOUNT_ID")
if not (OANDA_API_KEY and OANDA_ACCOUNT):
    print("ERROR: OANDA_API_KEY/OANDA_ACCOUNT_ID missing (.env).", file=sys.stderr)
    sys.exit(1)

HOST = "https://api-fxpractice.oanda.com" if OANDA_ENV!="live" else "https://api-fxtrade.oanda.com"
HEAD = {"Authorization": f"Bearer {OANDA_API_KEY}"}

def oanda_instruments():
    url = f"{HOST}/v3/accounts/{OANDA_ACCOUNT}/instruments"
    r = requests.get(url, headers=HEAD, timeout=30)
    r.raise_for_status()
    data = r.json()
    return data.get("instruments", [])

def map_group(inst_type, name):
    # OANDA types: CURRENCY, CFD, METAL (metals show as CURRENCY with 'XAU', 'XAG' pairs),
    # indices/commodities usually CFD.
    n=name.upper()
    if inst_type=="CURRENCY":
        if "XAU" in n or "XAG" in n: return "COMMODITY"
        return "FX"
    if inst_type=="CFD":
        # crude heuristics
        if any(k in n for k in ["WTI","BRENT","NGAS","NATGAS","OIL","UKOIL","USOIL"]): return "COMMODITY"
        return "INDEX"
    return "OTHER"

def build_from_oanda():
    out=[]
    for inst in oanda_instruments():
        name  = inst["name"]         # e.g. "EUR_USD", "US30_USD", "XAU_USD"
        typ   = inst.get("type","")
        disp  = inst.get("displayName", name.replace("_","/"))
        group = map_group(typ, name)
        out.append({
            "src": "OANDA",
            "oanda": name,
            "name": disp.replace("_","/"),
            "group": group,
            "enabled": True
        })
    return out

def build_crypto_binance():
    # Pocket Option lists major crypto. Add a focused set (expand anytime).
    tickers = [
        ("BTCUSDT","BTC/USDT"),
        ("ETHUSDT","ETH/USDT"),
        ("SOLUSDT","SOL/USDT"),
        ("XRPUSDT","XRP/USDT"),
        ("LTCUSDT","LTC/USDT"),
        ("DOGEUSDT","DOGE/USDT"),
        ("ADAUSDT","ADA/USDT"),
        ("BNBUSDT","BNB/USDT"),
    ]
    out=[]
    for sym, disp in tickers:
        out.append({
            "src":"BINANCE",
            "binance": sym,
            "name": disp,
            "group":"CRYPTO",
            "enabled": True
        })
    return out

def write_symbols_yaml(rows):
    # unify into your existing format (with src routing)
    data={"symbols": rows}
    Path("symbols.yaml").write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    print(f"✅ Wrote {len(rows)} symbols → symbols.yaml")

def main():
    oanda_rows = build_from_oanda()
    crypto_rows= build_crypto_binance()
    # de-dup by (src,id)
    seen=set(); merged=[]
    for r in oanda_rows + crypto_rows:
        key=(r["src"], r.get("oanda") or r.get("binance") or r.get("yf") or r["name"])
        if key in seen: continue
        seen.add(key); merged.append(r)
    write_symbols_yaml(merged)

if __name__=="__main__":
    main()
