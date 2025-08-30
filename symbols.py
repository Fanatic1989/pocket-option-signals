import yaml, re
# Convert common names/YF-ish codes to OANDA instruments
def to_oanda(sym:str)->str:
    s = sym.strip().upper()
    if s.endswith("=X"): s = s[:-2]           # EURUSD=X -> EURUSD
    s = s.replace("-","").replace("/","")     # EUR/USD -> EURUSD
    # Indexes/commodities not covered here; keep FX/simple crypto
    if re.fullmatch(r"[A-Z]{6,8}", s):
        # map BTCUSD -> BTC_USD (for OANDA crypto/fx style)
        if len(s)>=6:
            return f"{s[:3]}_{s[3:]}"
    return s
with open("symbols.yaml","r") as f:
    cfg = yaml.safe_load(f)
SYMBOLS = []
for s in cfg.get("symbols", []):
    if not s.get("enabled", True): continue
    yf = s.get("yf") or ""
    name = s.get("name") or yf or "UNKNOWN"
    oanda = s.get("oanda") or to_oanda(yf or name)
    SYMBOLS.append({"oanda": oanda, "name": name})
