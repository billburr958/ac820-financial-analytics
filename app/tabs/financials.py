# -------------------------------------------------------------
# tabs/financials.py  –  “Financials” tab
# -------------------------------------------------------------
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf


def _format_billions(x):
    """Helper to prettify large numbers."""
    return f"${x/1e9:.2f} B" if pd.notnull(x) else "N/A"


# -------------------------------------------------------------
# render() – main Financials tab
# -------------------------------------------------------------
def render(tickers: list[str]) -> None:
    """
    Parameters
    ----------
    tickers : list[str]
        User-selected ticker symbols from the sidebar.
    """
    st.markdown("### Company Financials")
    st.markdown(
        """
**Overview**  
View key financial metrics for each selected stock.

*Choose either **Annual** or **Quarterly** data and then pick a report date for each ticker.*  
Metrics displayed:

- Trailing / Forward P E  
- Dividend Yield  
- Market Cap  
- Revenue & Net Income (for the chosen period)
"""
    )

    if not tickers:
        st.info("Please select at least one ticker.")
        return

    fin_type = st.radio("Select Financial Data Type", ("Annual", "Quarterly"))

    fin_rows = []
    for tkr in tickers:
        try:
            yfin = yf.Ticker(tkr)
            fin_df = yfin.financials if fin_type == "Annual" else yfin.quarterly_financials
            if fin_df.empty:
                st.warning(f"No {fin_type} financial data available for {tkr}.")
                continue

            dates_available = list(fin_df.columns)
            pick_date = st.selectbox(
                f"{tkr} – choose report date",
                options=dates_available,
                key=f"{tkr}_{fin_type}",
            )

            fin_rows.append(
                {
                    "Ticker": tkr,
                    "Report Date": pick_date,
                    "Trailing P/E": yfin.info.get("trailingPE", np.nan),
                    "Forward P/E": yfin.info.get("forwardPE", np.nan),
                    "Dividend Yield": yfin.info.get("dividendYield", np.nan),
                    "Market Cap": yfin.info.get("marketCap", np.nan),
                    "Revenue": fin_df.loc["Total Revenue", pick_date]
                    if "Total Revenue" in fin_df.index
                    else np.nan,
                    "Net Income": fin_df.loc["Net Income", pick_date]
                    if "Net Income" in fin_df.index
                    else np.nan,
                }
            )

        except Exception as e:
            st.warning(f"Could not retrieve financials for {tkr}: {e}")

    if not fin_rows:
        st.info("No financial data available for the selected tickers.")
        return

    # Make pretty DataFrame for display
    fin_out = pd.DataFrame(fin_rows)
    fin_out["Market Cap"] = fin_out["Market Cap"].apply(_format_billions)
    fin_out["Revenue"] = fin_out["Revenue"].apply(_format_billions)
    fin_out["Net Income"] = fin_out["Net Income"].apply(_format_billions)

    st.dataframe(fin_out, use_container_width=True)
