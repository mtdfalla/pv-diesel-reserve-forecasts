"""Forecast evaluation metrics and the Diebold-Mariano test.
"""
import numpy as np
from scipy import stats

def mae(y,p):  return float(np.mean(np.abs(y-p)))
def rmse(y,p): return float(np.sqrt(np.mean((y-p)**2)))
def nrmse(y,p):return float(rmse(y,p)/np.mean(y)) if np.mean(y)!=0 else np.nan
def r2(y,p):
    ss=np.sum((y-np.mean(y))**2); return float(1-np.sum((y-p)**2)/ss) if ss>0 else np.nan
def skill(y,p,ref):  # forecast skill vs reference RMSE
    r=rmse(y,ref); return float(1-rmse(y,p)/r) if r>0 else np.nan

def all_metrics(y,p,ref=None):
    m=dict(n=int(len(y)),MAE=mae(y,p),RMSE=rmse(y,p),nRMSE=nrmse(y,p),R2=r2(y,p))
    if ref is not None: m["skill_vs_ref"]=skill(y,p,ref)
    return m

def diebold_mariano(y,p1,p2,h=1,power=2):
    """DM test: H0 equal accuracy. d = loss(p1)-loss(p2).
    Negative DM => p1 better. Returns (DM_stat, p_value).
    """
    e1=np.abs(y-p1)**power; e2=np.abs(y-p2)**power
    d=e1-e2; n=len(d); dbar=d.mean()
    # HLN small-sample autocov correction up to lag h-1
    gamma0=np.mean((d-dbar)**2)
    var=gamma0
    for k in range(1,h):
        ck=np.mean((d[k:]-dbar)*(d[:-k]-dbar)); var+=2*ck
    if var<=0: return np.nan,np.nan
    DM=dbar/np.sqrt(var/n)
    DM*=np.sqrt((n+1-2*h+h*(h-1)/n)/n)  # HLN correction
    pval=2*stats.t.cdf(-abs(DM),df=n-1)
    return float(DM),float(pval)
