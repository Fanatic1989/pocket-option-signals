import re, yaml, requests
from pathlib import Path

URLS = [
  "https://m.pocketoption.com/en/assets-current/",
  "https://pocketoption.com/en/assets/"
]

def fetch_assets():
    html = ""
    for u in URLS:
        try:
            r = requests.get(u, timeout=15, headers={"User-Agent":"Mozilla/5.0"})
            if r.ok and len(r.text) > len(html):
                html = r.text
        except Exception:
            pass
    return html

def parse_assets(html):
    # Very forgiving extraction: look for common symbols like EUR/USD, BTC/USD, GOLD, etc.
    # You can harden with BeautifulSoup later if needed.
    raw = set()
    for m in re.findall(r'([A-Z]{3}\/[A-Z]{3}|[A-Z]{3,5}[- ]USD|GOLD|BRENT|WTI|SILVER|NATURAL GAS|NASDAQ|S&P 500|DOW JONES)', html, re.I):
        raw.add(m.upper().replace(" ", ""))

    # Map to Yahoo/our groups quickly (you can expand this mapping later)
    def map_symbol(s):
        s = s.replace(" ", "")
        if "/" in s:  # FX like EUR/USD
            return {"yf": s.replace("/","") + "=X", "name": s, "group": "FX", "enabled": True}
        if s.endswith("-USD") or s.endswith("USD"):  # CRYPTO guess
            base = s.replace("-USD","").replace("USD","")
            return {"yf": f"{base}-USD", "name": f"{base}/USD", "group": "CRYPTO", "enabled": True}
        if s in {"GOLD"}:
            return {"yf": "XAUUSD=X", "name": "GOLD (XAU/USD)", "group": "COMMODITY", "enabled": True}
        return None

    items = []
    for s in sorted(raw):
        m = map_symbol(s)
        if m: items.append(m)
    return items

def write_symbols(items):
    path = Path("symbols.yaml")
    data = {"symbols": items}
    with open(path, "w") as f:
        yaml.safe_dump(data, f, sort_keys=False)
    print(f"Written {len(items)} symbols to {path}")

if __name__ == "__main__":
    html = fetch_assets()
    if not html:
        raise SystemExit("Could not fetch asset page")
    items = parse_assets(html)
    write_symbols(items)
