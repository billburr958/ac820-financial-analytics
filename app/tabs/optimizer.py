# -------------------------------------------------------------
# tabs/optimizer.py  –  “Optimizer” tab (enhanced)
# -------------------------------------------------------------
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from utils.optimize import bayes_weights  # Bayesian helper


def markowitz_random_portfolios(
    daily_returns: pd.DataFrame,
    num_portfolios: int = 5000,
    rf: float = 0.0,
    min_weight: float = 0.0,
    max_weight: float = 1.0,
) -> pd.DataFrame:
    results = []
    mean_returns = daily_returns.mean() * 252
    cov_matrix = daily_returns.cov() * 252
    tickers = daily_returns.columns.tolist()
    n = len(tickers)
    attempts = 0
    while len(results) < num_portfolios and attempts < num_portfolios * 10:
        w = np.random.dirichlet(np.ones(n))
        if np.all(w >= min_weight) and np.all(w <= max_weight):
            p_ret = np.dot(w, mean_returns)
            p_vol = np.sqrt(w @ cov_matrix @ w)
            sharpe = (p_ret - rf) / p_vol if p_vol != 0 else np.nan
            results.append({
                "Return":    p_ret,
                "Volatility": p_vol,
                "Sharpe":    sharpe,
                "Weights":   dict(zip(tickers, w))
            })
        attempts += 1
    return pd.DataFrame(results)


def render(price_df: pd.DataFrame, tickers: list[str]):
    st.markdown("### Optimizer 🔎")

    if price_df.empty or not tickers:
        st.info("Please select tickers and a valid date range.")
        return

    # Prepare returns
    price_pivot = price_df.pivot(index="Date", columns="Ticker", values="Price")
    daily_ret   = price_pivot.pct_change().dropna()
    if daily_ret.empty:
        st.info("Not enough data for optimization.")
        return

    # Choose method
    opt_method = st.radio("Optimization Method", ["Markowitz", "Bayesian Optimization"])

    # ──────── MARKOWITZ ───────────────────────────────────────────
    if opt_method == "Markowitz":
        c1, c2, c3, c4, c5 = st.columns((0.2, 0.2, 0.2, 0.2, 0.2))
        num_portfolios = c1.number_input("Portfolios", min_value=100, step=100, value=2000)
        rf_rate        = c2.number_input("Risk-Free Rate (%)", 0.0, 10.0, step=0.1, value=0.0) / 100
        min_w          = c3.number_input("Min Weight (%)", 0.0, 100.0, step=1.0, value=0.0) / 100
        max_w          = c4.number_input("Max Weight (%)", 0.0, 100.0, step=1.0, value=100.0) / 100
        run_btn        = c5.button("Run")

        if run_btn:
            st.markdown("Running Markowitz Monte-Carlo…")
            res_df = markowitz_random_portfolios(
                daily_ret,
                num_portfolios=num_portfolios,
                rf=rf_rate,
                min_weight=min_w,
                max_weight=max_w
            )
            if res_df.empty:
                st.warning("No portfolios met weight constraints.")
                return

            # Efficient frontier
            fig = px.scatter(
                res_df,
                x="Volatility",
                y="Return",
                color="Sharpe",
                color_continuous_scale="Turbo",
                title="Efficient Frontier",
                template="plotly_white",
                hover_data=["Volatility", "Return", "Sharpe"]
            )
            best_sharpe = res_df.loc[res_df["Sharpe"].idxmax()]
            min_vol     = res_df.loc[res_df["Volatility"].idxmin()]

            fig.add_scatter(
                x=[best_sharpe["Volatility"]],
                y=[best_sharpe["Return"]],
                mode="markers",
                marker=dict(color="red", size=12, symbol="star"),
                name="Max Sharpe"
            )
            fig.add_scatter(
                x=[min_vol["Volatility"]],
                y=[min_vol["Return"]],
                mode="markers",
                marker=dict(color="green", size=12, symbol="star"),
                name="Min Volatility"
            )
            fig.update_layout(
                xaxis_title="Volatility (Ann. Std Dev)",
                yaxis_title="Annualized Return"
            )
            st.plotly_chart(fig, use_container_width=True)

            # Display results with pie charts
            st.markdown("#### Optimal Portfolios")
            col1, col2 = st.columns(2)

            # Max Sharpe
            with col1:
                st.subheader("Max Sharpe Portfolio")
                st.metric("Return", f"{best_sharpe['Return']:.2%}")
                st.metric("Volatility", f"{best_sharpe['Volatility']:.2%}")
                st.metric("Sharpe Ratio", f"{best_sharpe['Sharpe']:.2f}")

                # Weights table
                w_df = pd.DataFrame.from_dict(best_sharpe["Weights"], orient="index", columns=["Weight"])
                w_df["Weight %"] = (w_df["Weight"] * 100).round(2).astype(str) + "%"
                st.table(w_df[["Weight %"]])

                # Pie chart
                pie1 = px.pie(
                    w_df,
                    values="Weight",
                    names=w_df.index,
                    hole=0.4,
                    title="Allocation",
                    template="plotly_white"
                )
                pie1.update_traces(textinfo="percent+label")
                st.plotly_chart(pie1, use_container_width=True)

            # Min Volatility
            with col2:
                st.subheader("Min Volatility Portfolio")
                st.metric("Return", f"{min_vol['Return']:.2%}")
                st.metric("Volatility", f"{min_vol['Volatility']:.2%}")
                st.metric("Sharpe Ratio", f"{min_vol['Sharpe']:.2f}")

                w2_df = pd.DataFrame.from_dict(min_vol["Weights"], orient="index", columns=["Weight"])
                w2_df["Weight %"] = (w2_df["Weight"] * 100).round(2).astype(str) + "%"
                st.table(w2_df[["Weight %"]])

                pie2 = px.pie(
                    w2_df,
                    values="Weight",
                    names=w2_df.index,
                    hole=0.4,
                    title="Allocation",
                    template="plotly_white"
                )
                pie2.update_traces(textinfo="percent+label")
                st.plotly_chart(pie2, use_container_width=True)

            st.success("Markowitz optimization complete!")

    # ──────── BAYESIAN OPTIMIZATION ───────────────────────────────
    else:
        st.markdown("### Bayesian Optimization")
        st.markdown("Uses Bayesian search to find weights that maximize Sharpe Ratio.")

        # Single-asset fallback
        if daily_ret.shape[1] == 1:
            bayes_w = np.array([1.0])
        else:
            bayes_w = bayes_weights(daily_ret, n_calls=50)

        bayes_dict = dict(zip(daily_ret.columns, bayes_w))

        # Display weights beautifully
        st.markdown("#### Optimized Weights")
        bw_df = pd.DataFrame.from_dict(bayes_dict, orient="index", columns=["Weight"])
        bw_df["Weight %"] = (bw_df["Weight"] * 100).round(2).astype(str) + "%"
        st.table(bw_df[["Weight %"]])

        # Pie chart
        pie_b = px.pie(
            bw_df,
            values="Weight",
            names=bw_df.index,
            hole=0.4,
            title="Bayesian Allocation",
            template="plotly_white"
        )
        pie_b.update_traces(textinfo="percent+label")
        st.plotly_chart(pie_b, use_container_width=True)

        st.info("Use these weights in the Performance tab to simulate results.")
