# -------------------------------------------------------------
# tabs/fraud_detection.py  – Fraud Detection tab (final/fixed)
# -------------------------------------------------------------
import streamlit as st
import pandas as pd
import shap
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import streamlit.components.v1 as components
from utils import fraud as F

# ───────────────────────────────────────────────────────────────
# Plain-language dictionary for every feature
# ───────────────────────────────────────────────────────────────
FEATURE_DOC = {
    "DSRI":  "Days-Sales-in-Receivables Index — compares receivables growth to sales growth; high values can signal revenue inflation.",
    "GMI":   "Gross-Margin Index — deterioration (> 1) may motivate manipulation.",
    "AQI":   "Asset-Quality Index — larger share of intangible / illiquid assets; higher values are riskier.",
    "SGI":   "Sales-Growth Index — rapid growth can pressure managers.",
    "DEPI":  "Depreciation Index — slower depreciation inflates earnings.",
    "SGAI":  "SG&A Index — rising SG&A relative to sales may indicate cost pressure.",
    "LVGI":  "Leverage Index — rising leverage raises default risk and incentives.",
    "TATA":  "Total Accruals × Assets — large positive accruals often accompany earnings manipulation.",
    "leverage":        "Total liabilities ÷ assets.",
    "profitability":   "Net income ÷ assets.",
    "liquidity":       "Current assets ÷ current liabilities.",
    "EBIT_to_assets":  "Operating profit ÷ assets.",
    "soft_asset_ratio":"Share of assets that are ‘soft’ (not cash/PPE).",
}

# ───────────────────────────────────────────────────────────────
# Helpers
# ───────────────────────────────────────────────────────────────
def _impact(shap_arr, feat_row):
    """Return dataframe sorted by |SHAP| desc."""
    df = pd.DataFrame(
        {"feature": feat_row.index,
         "value":   feat_row.values,
         "shap":    shap_arr}
    )
    df["abs"] = df["shap"].abs()
    return df.sort_values("abs", ascending=False)

# ───────────────────────────────────────────────────────────────
# Main render function
# ───────────────────────────────────────────────────────────────
def render(tickers):
    st.markdown("## Fraud Detection 🔍")
    st.markdown("""
This tab lets you assess a company’s financial statements for potential fraud using three complementary methods:

- **Custom Trained Model (ML)** – a CatBoost-based classifier we trained on financial ratios; outputs a fraud probability plus SHAP explainability.  
  ▶️ [View Development Notebook](https://drive.google.com/file/d/1gWQ33mYepsZ9MHvhqKuyAPehqOECSNoP/view?usp=drive_link)  
- **Beneish M-Score** – a classic rule-based approach using seven financial ratios (flag if M-Score > –2.22).  
- **Piotroski F-Score** – a nine-rule quality screen (flag if F-Score ≤ 3).

**Training Data Source**  
The Custom Trained Model was built using the [JarFraud FraudDetection dataset](https://github.com/JarFraud/FraudDetection/tree/master), which aggregates key financial indices—like Days-Sales-in-Receivables, Gross-Margin, Asset-Quality, and others—from public filings.

Use this tab to compare methods, drill into feature impacts with SHAP, and explore detailed ratio values.
""")

    if not tickers:
        st.info("Select at least one ticker in the sidebar.")
        return

    tkr  = st.selectbox("Ticker", tickers)
    meth = st.selectbox("Method", ["Custom Trained Model", "Beneish M-Score", "Piotroski F-Score"])
    if not st.button("Run Prediction"):
        st.stop()

    # Map display name back to internal method key
    method_key = "CatBoost" if meth == "Custom Trained Model" else meth
    res = F.predict(tkr, method_key)
    if res is None:
        st.error("Could not retrieve two full fiscal years for this ticker.")
        return

    feats = res["feats"]

    # ── Branch: Custom Trained Model ───────────────────────────────
    if meth == "Custom Trained Model":
        prob      = res["prob"]
        shap_vals = res["shap"]
        shap_row  = shap_vals[0]

        st.metric("Fraud Probability", f"{prob:.1%}",
                  help="> 50% indicates elevated fraud risk")

        # SHAP bar
        df_imp = _impact(shap_row.values, feats.iloc[0])
        top10  = df_imp.head(10)
        colors = ["crimson" if s > 0 else "seagreen" for s in top10["shap"]]

        fig_bar = go.Figure(go.Bar(
            x=top10["shap"], y=top10["feature"], orientation="h",
            marker_color=colors,
            customdata=top10["value"],
            hovertemplate="<b>%{y}</b><br>Impact: %{x:.3f}<br>Value: %{customdata:.2f}<extra></extra>",
        ))
        fig_bar.update_layout(
            title="Top SHAP Feature Impacts",
            xaxis_title="SHAP value (log-odds)",
            template="plotly_white", height=450)
        st.plotly_chart(fig_bar, use_container_width=True)

        # Dependence scatter
        with st.expander("Dependence Plot • explore any ratio"):
            feat_ch = st.selectbox("Pick ratio", feats.columns, key="dep")
            fig, ax = plt.subplots(figsize=(6,4))
            shap.plots.scatter(shap_vals[:, feat_ch], ax=ax, show=False)
            ax.set_xlabel(feat_ch); ax.set_ylabel("SHAP value")
            st.pyplot(fig)

        # Waterfall
        with st.expander("Waterfall (detailed)"):
            plt.close("all")
            shap.plots.waterfall(shap_row, max_display=14, show=False)
            fig = plt.gcf()
            fig.set_size_inches(8, 5)
            fig.set_dpi(80)
            fig.tight_layout()
            st.pyplot(fig)

    # ── Branch: Beneish M-Score ────────────────────────────────────
    elif meth == "Beneish M-Score":
        m_score = res["extra"]["M-Score"]
        flag    = res["extra"]["flag"]
        st.metric("M-Score", f"{m_score:.2f}", help="> -2.22 ⇒ likely manipulation")
        st.write("**Verdict:** ", "🚨 Flagged" if flag else "✅ No red flag")

    # ── Branch: Piotroski F-Score ───────────────────────────────────
    else:
        f_score = res["extra"]["F-Score"]
        flag    = res["extra"]["flag"]
        st.metric("F-Score", f"{f_score}/9", help="≤ 3 considered weak / risky")
        st.write("**Verdict:** ", "🚨 Weak (≤ 3)" if flag else "✅ Acceptable (> 3)")

    # ── Shared sections ─────────────────────────────────────────
    with st.expander("Numeric Feature Values"):
        st.dataframe(feats.T.rename(columns={0: "value"}))

    with st.expander("📖 Feature Glossary"):
        for k, v in FEATURE_DOC.items():
            st.markdown(f"* **{k}** — {v}")
