# -------------------------------------------------------------
# tabs/performance.py  –  “Performance & Metrics” tab
# -------------------------------------------------------------
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import yfinance as yf

# Re-use the logo helper from the Portfolio tab
from tabs.portfolio import get_company_logo


def render(
    data: pd.DataFrame,
    data_grouped: pd.DataFrame,
    tickers: list[str],
    date_start,
    date_end,
):
    # Inject CSS to shrink metric font sizes
    st.markdown(
        """
        <style>
          /* Metric label text */
          .stMetric label {
            font-size: 0.8rem !important;
          }
          /* Metric value text */
          .stMetric div[data-testid="stMetricValue"] {
            font-size: 1.2rem !important;
          }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### Performance Analysis & Portfolio Metrics")
    st.markdown(
        """
**Instructions:**  
1. **Investment Amounts:** Enter the dollar amount you plan to invest for each stock.  
2. **Portfolio Simulation:** The tool simulates your portfolio's historical growth.  
3. **Performance Metrics:** Various risk–return measures are computed, including benchmark comparisons.
"""
    )

    # ------------------------------------------------------------------
    # Guard if no data
    # ------------------------------------------------------------------
    if data.empty or len(tickers) == 0:
        st.info("Please select tickers and ensure valid data.")
        return

    # ------------------------------------------------------------------
    # Step 1 – Input investment amounts
    # ------------------------------------------------------------------
    st.markdown("#### Step 1 • Enter Investment Amounts")

    col_alloc, col_total = st.columns((0.25, 0.75))
    total_inv = 0
    allocations = {}

    for tkr in tickers:
        row = col_alloc.columns((0.1, 0.3))
        logo = get_company_logo(tkr)
        if logo:
            row[0].image(logo, width=65)
        else:
            row[0].subheader(tkr)

        amt = row[1].number_input(
            f"Amount for {tkr}",
            key=f"amt_{tkr}",
            step=50,
            label_visibility="collapsed",
        )
        total_inv += amt
        allocations[tkr] = amt

    col_total.subheader(f"Total Investment: ${total_inv:,.0f}")

    # ------------------------------------------------------------------
    # Step 2 – Optional benchmark
    # ------------------------------------------------------------------
    st.markdown("#### Step 2 • (Optional) Select a Benchmark")
    benchmark_tkr = st.text_input("Benchmark Ticker (default: SPY)", value="SPY")

    benchmark_df = pd.DataFrame()
    if benchmark_tkr:
        try:
            raw = yf.download(
                benchmark_tkr, start=date_start, end=date_end, progress=False
            ).reset_index()
            if "Adj Close" in raw.columns:
                benchmark_df = raw[["Date", "Adj Close"]].rename(
                    columns={"Adj Close": "Benchmark Price"}
                )
            elif "Close" in raw.columns:
                benchmark_df = raw[["Date", "Close"]].rename(
                    columns={"Close": "Benchmark Price"}
                )
            else:
                st.warning(
                    "Benchmark data lacks a recognized price column (Adj Close or Close)."
                )
        except Exception as e:
            st.warning(f"Benchmark download failed: {e}")

    # ------------------------------------------------------------------
    # Step 3 – Growth simulation
    # ------------------------------------------------------------------
    st.markdown("#### Step 3 • Simulated Portfolio Growth")

    # Display the formula with proper LaTeX rendering
    st.write("**Growth Formula**")
    st.latex(r"Investment_{t} = Allocation_{0} \times \left(1 + \frac{P_{t} - P_{0}}{P_{0}}\right)")

    df_sim = data.copy()
    df_sim["RelFactor"] = df_sim.groupby("Ticker")["Adjusted Closing Price"].transform(
        lambda x: (x - x.iloc[0]) / x.iloc[0]
    )
    df_sim["Investment"] = df_sim["Ticker"].map(allocations) * (1 + df_sim["RelFactor"])

    fig_sim = px.area(
        df_sim,
        x="Date",
        y="Investment",
        color="Ticker",
        title="Simulated Portfolio Value Over Time",
        template="plotly_white",
        labels={"Investment": "Investment Value ($)"},
    )
    fig_sim.update_layout(hovermode="x unified")
    st.plotly_chart(fig_sim, use_container_width=True)

    # ------------------------------------------------------------------
    # Step 4 – Performance metrics
    # ------------------------------------------------------------------
    st.markdown("#### Step 4 • Portfolio Performance Metrics")

    st.markdown(
        """
**Metric Definitions**  
- *Annual Return* = geometric yearly return  
- *Annual Volatility* = daily std × √252  
- *Sharpe* = Ann Return / Ann Volatility  
- *Sortino* = Ann Return / Downside Volatility  
- *Calmar* = Ann Return / |Max Drawdown|  
- *VaR 95 & CVaR 95* = 5th-percentile loss and its mean  
- *Beta & R²* versus benchmark  
"""
    )

    # build wide price pivot
    price_pivot = (
        data.pivot(index="Date", columns="Ticker", values="Adjusted Closing Price")
        .dropna(how="all")
        .copy()
    )
    if price_pivot.empty:
        st.info("Not enough data to compute metrics.")
        return

    # Daily returns & weights
    daily_ret = price_pivot.pct_change().dropna()
    total_alloc = sum(allocations.values())

    if total_alloc == 0:
        st.warning("Enter at least one positive investment amount.")
        return

    weights = {t: allocations[t] / total_alloc for t in tickers}
    w_series = pd.Series(weights)
    w_vec = w_series.reindex(daily_ret.columns).fillna(0).values

    port_ret = daily_ret.mul(w_vec, axis=1).sum(axis=1)
    cum_ret = (1 + port_ret).cumprod()

    # --- core metrics
    n_days = len(port_ret)
    ann_return = cum_ret.iloc[-1] ** (252 / n_days) - 1
    ann_vol = port_ret.std() * np.sqrt(252)
    sharpe = ann_return / ann_vol if ann_vol else np.nan

    downside = port_ret[port_ret < 0]
    down_vol = downside.std() * np.sqrt(252) if not downside.empty else np.nan
    sortino = ann_return / down_vol if down_vol else np.nan

    run_max = cum_ret.cummax()
    drawdown = (cum_ret - run_max) / run_max
    max_dd = drawdown.min()
    calmar = ann_return / abs(max_dd) if max_dd else np.nan

    var95 = np.percentile(port_ret, 5)
    cvar95 = port_ret[port_ret <= var95].mean()

    # --- beta / R²
    beta = r2 = np.nan
    if not benchmark_df.empty and "Benchmark Price" in benchmark_df.columns:
        bench = benchmark_df.copy()
        bench["BenchRet"] = bench["Benchmark Price"].pct_change()
        bench = bench.set_index("Date")["BenchRet"].dropna()

        common = port_ret.index.intersection(bench.index)
        if not common.empty:
            port_aligned = port_ret.loc[common]
            bench_aligned = bench.loc[common]
            cov = np.cov(port_aligned, bench_aligned)[0, 1]
            var_b = np.var(bench_aligned)
            beta = cov / var_b if var_b else np.nan
            r2 = np.corrcoef(port_aligned, bench_aligned)[0, 1] ** 2

    # --- Show metrics
    m = st.columns(7)
    m[0].metric("Annual Return", f"{ann_return:.2%}")
    m[1].metric("Volatility", f"{ann_vol:.2%}")
    m[2].metric("Sharpe", f"{sharpe:.2f}")
    m[3].metric("Sortino", f"{sortino:.2f}")
    m[4].metric("Calmar", f"{calmar:.2f}")
    m[5].metric("Max Drawdown", f"{max_dd:.2%}")
    m[6].metric("VaR 95%", f"{var95:.2%}")
    st.markdown(f"**CVaR 95%:** {cvar95:.2%}")

    if not np.isnan(beta):
        st.markdown(f"**Beta vs {benchmark_tkr}:** {beta:.2f} (R² = {r2:.2f})")
