import os, requests, pandas as pd, yfinance as yf

def fetch_oanda_5m_close_series(pair: str) -> pd.DataFrame:
    API = os.getenv("OANDA_API_KEY")
    ENV = os.getenv("OANDA_ENV","practice").lower()
    host = "https://api-fxpractice.oanda.com" if ENV!="live" else "https://api-fxtrade.oanda.com"
    head = {"Authorization": f"Bearer {API}"}
    r = requests.get(f"{host}/v3/instruments/{pair}/candles",
                     headers=head, params={"count":300,"granularity":"M5","price":"M","includeFirst":"false"}, timeout=30)
    r.raise_for_status()
    js = r.json().get("candles",[])
    rows=[(c["time"], float(c["mid"]["c"])) for c in js if c.get("complete")]
    if not rows: raise RuntimeError(f"OANDA empty for {pair}")
    df = pd.DataFrame(rows, columns=["time","close"])
    return df

def fetch_binance_5m(symbol: str) -> pd.DataFrame:
    r = requests.get("https://api.binance.com/api/v3/klines",
                     params={"symbol":symbol,"interval":"5m","limit":300}, timeout=30)
    r.raise_for_status()
    k = r.json()
    rows=[(i[6], float(i[4])) for i in k]  # closeTime, close
    return pd.DataFrame(rows, columns=["time","close"])

def fetch_yf(interval: str, ticker: str, period: str="5d") -> pd.DataFrame:
    df = yf.download(tickers=ticker, interval=interval, period=period, progress=False)
    return df.dropna().rename(columns=str.lower)

def fetch_df_from_symbol(sym: dict, interval: str="5m", lookback: str="5d") -> pd.DataFrame:
    src = (sym.get("src") or "YF").upper()
    if src=="OANDA" and sym.get("oanda"):
        return fetch_oanda_5m_close_series(sym["oanda"])
    if src=="BINANCE" and sym.get("binance"):
        return fetch_binance_5m(sym["binance"])
    # fallback (YF)
    yf_t = sym.get("yf")
    if not yf_t:
        # try to guess FX form for YF if needed
        name = sym.get("name","").replace("/","")
        yf_t = name + ("=X" if len(name)==6 else "")
    return fetch_yf(interval, yf_t, lookback)
