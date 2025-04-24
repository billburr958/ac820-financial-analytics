# app/app.py

import streamlit as st  # must be first Streamlit command
st.set_page_config(
    page_title="📈 Portfolio Analysis & Optimizer",
    layout="wide",
    initial_sidebar_state="expanded"
)

import datetime as dt
import pandas as pd

# ── Load custom CSS for financial dashboard styling ──────
with open("assets/style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# ── Load Google Font ────────────────────────────────────
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap" rel="stylesheet">
<style>
  body, .stText, .stMetric div[data-testid="stMetricValue"] {
    font-family: 'Inter', sans-serif;
  }
</style>
""", unsafe_allow_html=True)

# ── Title & Description ─────────────────────────────────
st.title("📈 Portfolio Analysis & Optimizer Tool")

st.markdown("""
**What you can do with this app:**

- **Portfolio Overview:** Track adjusted closing prices and daily performance across your selected tickers.
- **Financials:** View fundamental metrics and ratio analyses for each company.
- **Optimizer:** Get suggestions on portfolio allocation to maximize return for a given risk level.
- **Performance & Metrics:** Deep dive into KPIs, cumulative returns, and benchmark comparisons.
- **Fraud Detection:** Assess potential financial misreporting via machine learning and rule‑based models.
- **10‑K Explorer:** Download and query recent SEC 10‑K filings interactively.
""", unsafe_allow_html=True)

# ── Imports & logger ─────────────────────────────────────
from utils.logger import get_logger
from utils import data as d
from config import THEME_CSS  # if you have additional CSS

from tabs import (
    portfolio,
    performance,
    optimizer,
    financials,
    fraud_detection,
    tenk_explorer
)

log = get_logger(__name__)
log.info("====== App start ======")

# ── Sidebar ──────────────────────────────────────────────
master_df   = d.master_tickers()
display_lst = master_df.display.tolist()

with st.sidebar:
    st.header("🔎 Choose Stocks")
    chosen_disp = st.multiselect("Tickers (search)", display_lst)
    chosen      = master_df.loc[
        master_df.display.isin(chosen_disp), "symbol"
    ].tolist()
    log.info("Tickers selected: %s", chosen)

    st.markdown("---")
    st.header("📅 Date Range")
    start_date = st.date_input("Start", dt.date.today() - dt.timedelta(days=365))
    end_date   = st.date_input("End",   dt.date.today())
    if start_date >= end_date:
        st.error("Start date must be before end date.")
    log.info("Date range %s – %s", start_date, end_date)

# ── Load price data ───────────────────────────────────────
log.info("Downloading price data …")
price_df = d.prices(chosen, start_date, end_date)
log.info("Rows downloaded: %d", len(price_df))

# ── Compute helper DataFrame ─────────────────────────────
if price_df.empty:
    group_df = pd.DataFrame()
else:
    group_df = price_df.copy()
    group_df["Price_First"]   = group_df.groupby("Ticker")["Price"].transform("first")
    group_df["Price_End"]     = group_df.groupby("Ticker")["Price"].transform("last")
    group_df["Price_%_Daily"] = group_df.groupby("Ticker")["Price"].pct_change()
    group_df["Price_%"]       = (
        group_df["Price_End"] - group_df["Price_First"]
    ) / group_df["Price_First"]

# ── Rename for plotting tabs ──────────────────────────────
price_plot = price_df.rename(columns={"Price": "Adjusted Closing Price"})
group_plot = group_df.rename(columns={"Price": "Adjusted Closing Price"})

# ── Define tabs in desired order ──────────────────────────
tab_port, tab_fin, tab_opt, tab_perf, tab_fraud, tab_10k = st.tabs([
    "🏠 Portfolio Overview",
    "🧮 Financials",
    "⚙️ Optimizer",
    "📊 Performance & Metrics",
    "🚨 Fraud Detection",
    "📑 10-K Explorer"
])

with tab_port:
    log.info("[Tab] Portfolio Overview")
    portfolio.render(price_plot, group_plot, chosen)

with tab_fin:
    log.info("[Tab] Financials")
    financials.render(chosen)

with tab_opt:
    log.info("[Tab] Optimizer")
    optimizer.render(price_df, chosen)

with tab_perf:
    log.info("[Tab] Performance & Metrics")
    performance.render(price_plot, group_plot, chosen, start_date, end_date)

with tab_fraud:
    log.info("[Tab] Fraud Detection")
    fraud_detection.render(chosen)

with tab_10k:
    log.info("[Tab] 10-K Explorer")
    tenk_explorer.render()

log.info("====== Render cycle complete ======")
