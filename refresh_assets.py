import re, requests, yaml
from pathlib import Path

URLS = [
    "https://pocketoption.com/en/assets/",
    "https://m.pocketoption.com/en/assets-current/",
]

# Simple keyword → Yahoo Finance ticker mapping for non-FX
MAP_EXACT = {
    "GOLD": "XAUUSD=X",
    "SILVER": "XAGUSD=X",
    "BRENT": "BZ=F",          # ICE Brent futures
    "WTI": "CL=F",            # WTI futures
    "NATGAS": "NG=F",
    "NATURALGAS": "NG=F",

    "S&P500": "^GSPC",
    "SP500": "^GSPC",
    "NASDAQ": "^NDX",
    "DOWJONES": "^DJI",
    "DAX": "^GDAXI",
    "FTSE": "^FTSE",
    "CAC40": "^FCHI",
    "EUROSTOXX50": "^STOXX50E",
    "NIKKEI": "^N225",
    "HANGSENG": "^HSI",
    "ASX200": "^AXJO",
}

# Crypto (fallbacks) that Pocket Option commonly lists
CRYPTO_SET = {"BTC", "ETH", "LTC", "XRP", "BCH", "ADA", "DOGE", "SOL", "DOT"}

def fetch_html():
    best = ""
    for url in URLS:
        try:
            r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=20)
            if r.ok and len(r.text) > len(best):
                best = r.text
        except Exception:
            pass
    if not best:
        raise RuntimeError("Could not fetch Pocket Option assets page")
    return best

def parse_assets(html: str):
    """
    Grab things that look like assets:
      • FX like EUR/USD, GBP/JPY, USD/CHF
      • Keywords like GOLD, SILVER, BRENT, WTI, NATURAL GAS
      • Indices like NASDAQ, S&P 500, DOW JONES, DAX, FTSE, CAC 40, NIKKEI, etc.
      • Crypto tickers (BTC, ETH...) – we’ll convert to BTC-USD, ETH-USD, etc.
    """
    found = set()

    # FX pairs e.g., EUR/USD
    for m in re.findall(r'\b([A-Z]{3}/[A-Z]{3})\b', html):
        found.add(m.upper())

    # Keywords (commodities / indices / crypto words)
    for m in re.findall(r'\b([A-Z][A-Z0-9&\-\s]{2,})\b', html):
        token = m.upper().replace(" ", "")
        # Filter to likely asset words
        if any(k in token for k in ["GOLD","SILVER","BRENT","WTI","NAT","GAS","S&P","SP500","NASDAQ","DOW","DAX","FTSE","CAC","EUROSTOXX","NIKKEI","HANGSENG","ASX","BTC","ETH","LTC","XRP","BCH","ADA","DOGE","SOL","DOT"]):
            found.add(token)

    return sorted(found)

def map_to_yf(symbols):
    out = []
    seen = set()

    for s in symbols:
        name = s

        # FX like EUR/USD -> EURUSD=X
        if "/" in s and len(s) == 7:
            yf = s.replace("/", "") + "=X"
            group = "FX"
        # Exact keyword maps (commodities & indices)
        elif s in MAP_EXACT:
            yf = MAP_EXACT[s]
            group = "INDEX" if yf.startswith("^") else "COMMODITY"
        # Crypto names -> CRYPTO-USD
        elif s in CRYPTO_SET:
            yf = f"{s}-USD"
            group = "CRYPTO"
        # NATURALGAS / NATGAS variants
        elif s in {"NATURALGAS", "NATGAS"}:
            yf = "NG=F"; group = "COMMODITY"
        # S&P variants seen without ampersand
        elif s in {"SP500", "SANDP500"}:
            yf = "^GSPC"; group = "INDEX"
        # CAC40/ASX200 without space
        elif s in {"CAC40"}:
            yf="^FCHI"; group="INDEX"
        elif s in {"ASX200"}:
            yf="^AXJO"; group="INDEX"
        # If it looks like a crypto asset ending with "USD" already (rare)
        elif re.fullmatch(r'[A-Z]{2,5}-USD', s):
            yf = s; group = "CRYPTO"
        else:
            # Skip unknowns—keeps file clean
            continue

        key = (yf, group)
        if key in seen:
            continue
        seen.add(key)
        out.append({"yf": yf, "name": name.replace("SANDP","S&P"), "group": group, "enabled": True})

    return out

def write_yaml(rows):
    Path("symbols.yaml").write_text(yaml.safe_dump({"symbols": rows}, sort_keys=False), encoding="utf-8")
    print(f"✅ Wrote {len(rows)} symbols to symbols.yaml")

if __name__ == "__main__":
    html = fetch_html()
    raw  = parse_assets(html)
    rows = map_to_yf(raw)
    write_yaml(rows)
