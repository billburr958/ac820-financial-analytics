# -------- utils/optimize.py -------
import numpy as np, pandas as pd
from skopt import gp_minimize

def bayes_weights(daily_returns:pd.DataFrame, n_calls:int=50)->np.ndarray:
    def obj(w):
        w=np.array(w); w=w/w.sum() if w.sum()!=0 else w
        dr=daily_returns.dot(w); 
        if dr.isnull().any(): return 1e6
        ann_ret=(1+dr).prod()**(252/len(dr))-1
        ann_vol=dr.std()*np.sqrt(252)
        return -ann_ret/ann_vol if ann_vol else 1e6
    if daily_returns.shape[1]==1:
        return np.array([1.0])
    bounds=[(0.,1.)]*daily_returns.shape[1]
    res=gp_minimize(obj,bounds,n_calls=n_calls,random_state=42)
    w=np.array(res.x); return w/w.sum()
