# -------------------------------------------------------------
# tabs/portfolio.py   –  “Portfolio Overview & Individual Stock Insights”
# -------------------------------------------------------------
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px


# ------------------------------------------------------------------
# Helper to fetch Clearbit logo
# ------------------------------------------------------------------
def get_company_logo(ticker: str) -> str | None:
    try:
        stock = yf.Ticker(ticker)
        website = stock.info.get("website", None)
        if website:
            domain = (
                website.replace("https://www.", "")
                       .replace("https://", "")
                       .split("/")[0]
            )
            return f"https://logo.clearbit.com/{domain}"
    except Exception:
        pass
    return None


# ------------------------------------------------------------------
# render()  — call this from app.py
# ------------------------------------------------------------------
def render(
    data: pd.DataFrame,
    data_grouped: pd.DataFrame,
    tickers: list[str],
):
    """
    Parameters
    ----------
    data : pd.DataFrame
        Melted price DataFrame with columns
        ['Date', 'Ticker', 'Adjusted Closing Price']
    data_grouped : pd.DataFrame
        Same data with extra helper columns already
        computed (Price_First, Price_End, etc.)
    tickers : list[str]
        List of user-selected ticker symbols
    """

    # Inject CSS for smaller metric font
    st.markdown(
        """
        <style>
            .small-metric-label { font-size: 12px; color: #666; margin-bottom:2px; }
            .small-metric-value { font-size: 18px; font-weight: bold; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### Portfolio Overview & Individual Stock Insights")

    st.markdown(
        """
**Overview:**  
View historical price trends and key metrics for your selected stocks.
    
**Charts & Metrics:**  
- **Portfolio Composition** — line chart of each ticker’s adjusted close.  
- **Individual Stock Panels** — show compact metrics (50-day avg, 1-year low/high) and optional indicators.
"""
    )

    # ------------------------------------------------------------------
    # Optional extra metrics
    # ------------------------------------------------------------------
    additional_metrics = st.multiselect(
        "Select Additional Metrics to Display",
        options=[
            "20-Day SMA",
            "200-Day SMA",
            "Annualized Volatility",
            "Cumulative Return",
        ],
        default=[],
    )

    # ------------------------------------------------------------------
    # Guard-rail if no data
    # ------------------------------------------------------------------
    if data.empty or len(tickers) == 0:
        st.info("Please select at least one ticker and a valid date range.")
        return

    # ------------------------------------------------------------------
    # Portfolio-level line chart
    # ------------------------------------------------------------------
    fig_portfolio = px.line(
        data,
        x="Date",
        y="Adjusted Closing Price",
        color="Ticker",
        title="Portfolio Composition",
        template="plotly_white",
    )
    fig_portfolio.update_layout(
        xaxis_title="Date",
        yaxis_title="Price ($)",
        hovermode="x unified"
    )
    fig_portfolio.update_yaxes(tickprefix="$")
    st.plotly_chart(fig_portfolio, use_container_width=True)

    # ------------------------------------------------------------------
    # Individual stock panels
    # ------------------------------------------------------------------
    st.markdown("#### Individual Stock Metrics")

    cols = st.columns(3)
    for i, ticker in enumerate(tickers):
        col = cols[i % 3]

        # logo or header
        logo_url = get_company_logo(ticker)
        if logo_url:
            col.image(logo_url, width=65)
        else:
            col.subheader(ticker)

        # subset for this ticker
        tdata = data_grouped.query("Ticker == @ticker").sort_values("Date")
        if tdata.empty:
            col.warning("No data available.")
            continue

        prices = tdata["Adjusted Closing Price"]

        # compute core metrics
        ma50    = prices.tail(50).mean()    if len(prices) >= 50 else np.nan
        yr_low  = prices.tail(365).min()    if len(prices) >= 1  else np.nan
        yr_high = prices.tail(365).max()    if len(prices) >= 1  else np.nan

        # display metrics with smaller font via HTML
        m1, m2, m3 = col.columns(3)
        m1.markdown(
            f"<div class='small-metric-label'>50d Avg ($)</div>"
            f"<div class='small-metric-value'>{ma50:.2f}</div>",
            unsafe_allow_html=True,
        )
        m2.markdown(
            f"<div class='small-metric-label'>1y Low ($)</div>"
            f"<div class='small-metric-value'>{yr_low:.2f}</div>",
            unsafe_allow_html=True,
        )
        m3.markdown(
            f"<div class='small-metric-label'>1y High ($)</div>"
            f"<div class='small-metric-value'>{yr_high:.2f}</div>",
            unsafe_allow_html=True,
        )

        # optional extra metrics
        if "20-Day SMA" in additional_metrics:
            sma20 = prices.tail(20).mean() if len(prices) >= 20 else np.nan
            col.write(f"**20d SMA:** ${sma20:.2f}")

        if "200-Day SMA" in additional_metrics:
            sma200 = prices.tail(200).mean() if len(prices) >= 200 else np.nan
            col.write(f"**200d SMA:** ${sma200:.2f}")

        if "Annualized Volatility" in additional_metrics:
            daily_ret = prices.pct_change().dropna()
            ann_vol = daily_ret.std() * np.sqrt(252) if not daily_ret.empty else np.nan
            col.write(f"**Ann Vol:** {ann_vol:.2%}")

        if "Cumulative Return" in additional_metrics:
            cum_ret = (prices.iloc[-1] / prices.iloc[0]) - 1
            col.write(f"**Cum Ret:** {cum_ret:.2%}")

        # individual price trend
        fig_stock = px.line(
            tdata,
            x="Date",
            y="Adjusted Closing Price",
            title=f"{ticker} Price Trend",
            markers=True,
            template="plotly_white",
        )
        fig_stock.update_layout(
            xaxis_title=None,
            yaxis_title=None,
            hovermode="x unified"
        )
        col.plotly_chart(fig_stock, use_container_width=True)
