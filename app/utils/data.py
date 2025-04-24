# -------- utils/data.py ----------
import datetime as dt, pandas as pd, yfinance as yf, streamlit as st
import financedatabase as fd

@st.cache_data
def master_tickers():
    df = pd.concat([
        fd.ETFs().select().reset_index()[["symbol","name"]],
        fd.Equities().select().reset_index()[["symbol","name"]],
    ])
    df = df[df.symbol.notna()]
    df.symbol = df.symbol.str.replace("^","",regex=False)
    df["display"] = df.symbol + " - " + df.name
    return df

def clearbit_logo(tkr):
    try:
        web = yf.Ticker(tkr).info.get("website","")
        if web:
            dom = web.replace("https://www.","").replace("https://","").split("/")[0]
            return f"https://logo.clearbit.com/{dom}"
    except Exception:
        pass
    return None

@st.cache_data(show_spinner=False)
def prices(tickers:list, start:dt.date, end:dt.date)->pd.DataFrame:
    if not tickers:
        return pd.DataFrame()
    df = yf.download(tickers, start=start, end=end, progress=False, threads=False)["Close"]
    return df.reset_index().melt(id_vars="Date", var_name="Ticker", value_name="Price")
