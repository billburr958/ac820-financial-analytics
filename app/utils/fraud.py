# -------------------------------------------------------------
# utils/fraud.py  –  CatBoost + rule-based scores
# -------------------------------------------------------------
import numpy as np, pandas as pd, requests, joblib, shap
from functools import lru_cache
from config import FMP_API_KEY
import streamlit as st


FMP_API_KEY = st.secrets["FMP_API_KEY"]

# ------------ load CatBoost model once ------------
CB_MODEL = joblib.load("best_model.pkl")

# ------------ API helper (two fiscal years) --------
@lru_cache(maxsize=128)
def _two_years(tkr):
    base = "https://financialmodelingprep.com/api/v3"
    eps  = ["income-statement", "balance-sheet-statement", "cash-flow-statement"]
    inc, bal, cf = (requests.get(f"{base}/{ep}/{tkr}?limit=2&apikey={FMP_API_KEY}").json() for ep in eps)
    if not inc or not bal or not cf:
        return None
    df = pd.concat([pd.DataFrame(inc), pd.DataFrame(bal), pd.DataFrame(cf)], axis=1)
    df = df.loc[:, ~df.columns.duplicated()].sort_values("date").fillna(0)
    return df.tail(2) if df.shape[0] >= 2 else None

# ------------ ratio calculations (same as before) --
def _div(n, d, fallback=0.0):
    try:
        if d == 0 or np.isnan(d):
            return fallback
        return n / d
    except:
        return fallback


def _ratios(df2):
    c, p = df2.iloc[-1], df2.iloc[-2]
    g = lambda row, k: float(row.get(k, 0))

    rectC, rectP = g(c, "netReceivables"), g(p, "netReceivables")
    saleC, saleP = g(c, "revenue"), g(p, "revenue")
    cogsC, cogsP = g(c, "costOfRevenue"), g(p, "costOfRevenue")
    actC, actP = g(c, "totalCurrentAssets"), g(p, "totalCurrentAssets")
    ppeC, ppeP = g(c, "propertyPlantEquipmentNet"), g(p, "propertyPlantEquipmentNet")
    atC, atP = g(c, "totalAssets"), g(p, "totalAssets")
    dpC, dpP = g(c, "depreciationAndAmortization"), g(p, "depreciationAndAmortization")
    niC, niP = g(c, "netIncome"), g(p, "netIncome")
    ltC, ltP = g(c, "totalLiabilities"), g(p, "totalLiabilities")
    lctC = g(c, "totalCurrentLiabilities")
    ebitC = g(c, "ebit") if "ebit" in c else niC

    return pd.DataFrame([{
        "DSRI": _div(rectC / saleC, rectP / saleP),
        "GMI": _div((saleP - cogsP) / saleP, (saleC - cogsC) / saleC),
        "AQI": _div(1 - (actC + ppeC) / atC, 1 - (actP + ppeP) / atP),
        "SGI": _div(saleC, saleP),
        "DEPI": _div(dpP / (dpP + ppeP), dpC / (dpC + ppeC)),
        "SGAI": _div((saleC - cogsC - niC) / saleC, (saleP - cogsP - niP) / saleP),
        "LVGI": _div(ltC / atC, ltP / atP),
        "TATA": _div(actC - lctC - dpC, atC),
        # extras
        "leverage":       _div(ltC, atC),
        "profitability":  _div(niC, atC),
        "liquidity":      _div(actC, lctC),
        "EBIT_to_assets": _div(ebitC, atC),
        "soft_asset_ratio": _div(actC - ppeC, atC),
        # Piotroski helpers
        "ROA": _div(niC, atC),
        "Prev_ROA": _div(niP, atP),
        "CFO": _div(g(c, "operatingCashFlow"), atC),
        "GrossMargin": _div(saleC - cogsC, saleC),
        "Prev_GM": _div(saleP - cogsP, saleP),
        "AssetTurn": _div(saleC, atC),
        "Prev_AT": _div(saleP, atP),
        "CurrRatio": _div(actC, lctC),
        "Prev_CR": _div(actP, g(p, "totalCurrentLiabilities")),
        "Shares": g(c, "weightedAverageShsOut"),
        "Prev_Shares": g(p, "weightedAverageShsOut"),
        "Prev_leverage": _div(ltP, atP),
    }])


# ------------ rule scorers -------------------------------------
def beneish_score(r):
    m = (
        -4.84
        + 0.92  * r.DSRI
        + 0.528 * r.GMI
        + 0.404 * r.AQI
        + 0.892 * r.SGI
        + 0.115 * r.DEPI
        - 0.172 * r.SGAI
        + 4.679 * r.TATA
    )
    flag = m > -2.22
    return float(m), bool(flag)


# ------------ rule scorers -------------------------------------
def piotroski_score(r):
    """
    Returns (F-Score, flag)
    Flag is True when F-Score ≤ 3  (considered weak / risky)
    """
    # Helper metrics
    roa_cur  = r.ROA
    roa_prev = r.Prev_ROA
    cfo      = r.CFO
    accrual  = cfo > roa_cur
    gm_cur   = r.GrossMargin
    gm_prev  = r.Prev_GM
    at_cur   = r.AssetTurn
    at_prev  = r.Prev_AT
    lev_cur  = r.leverage
    lev_prev = r.Prev_leverage
    cr_cur   = r.CurrRatio
    cr_prev  = r.Prev_CR
    share_change = r.Shares <= r.Prev_Shares

    score = 0
    score += roa_cur > 0                 # 1. ROA positive
    score += cfo > 0                     # 2. CFO positive
    score += (roa_cur - roa_prev) > 0    # 3. ΔROA positive
    score += accrual                     # 4. CFO > ROA (quality)
    score += lev_cur < lev_prev          # 5. ΔLeverage down
    score += cr_cur > cr_prev            # 6. ΔCurrent Ratio up
    score += (gm_cur - gm_prev) > 0      # 7. ΔGross Margin up
    score += (at_cur - at_prev) > 0      # 8. ΔAsset Turnover up
    score += share_change                # 9. Shares not issued

    flag = score <= 3
    return int(score), bool(flag)


# ------------ public dispatcher ---------------------------------
def predict(tkr: str, method: str = "CatBoost"):
    """
    Returns a dict:
        prob, shap_vals, feats, extra   (prob None for rule scores)
    """
    df2 = _two_years(tkr)
    if df2 is None:
        return None

    feats = _ratios(df2)
    r = feats.iloc[0]

    if method == "CatBoost":
        prob = float(CB_MODEL.predict_proba(feats)[0, 1])
        shap_vals = shap.TreeExplainer(CB_MODEL)(feats, check_additivity=False)
        return {"prob": prob, "shap": shap_vals, "feats": feats, "extra": None}

    elif method == "Beneish M-Score":
        m, flag = beneish_score(r)
        return {"prob": None, "shap": None, "feats": feats,
                "extra": {"M-Score": m, "flag": flag}}

    elif method == "Piotroski F-Score":
        fscore, flag = piotroski_score(r)
        return {"prob": None, "shap": None, "feats": feats,
                "extra": {"F-Score": fscore, "flag": flag}}

    else:
        raise ValueError("Unknown method")
# -------------------------------------------------------------
