"""Split-conformal prediction methods in GHI space: marginal ICP, clear-sky-normalized ICP,
Mondrian (per weather regime) and CQR (Romano et al., 2019).
"""
import numpy as np

def conformal_q(scores, cov):
    """Finite-sample conformal quantile of nonconformity scores for coverage `cov`."""
    n = len(scores); k = int(np.ceil((n+1)*cov))
    k = min(max(k,1), n)
    return np.sort(scores)[k-1]

# ---- ICP marginal ----
def icp_fit(y_cal, pred_cal): return np.abs(y_cal - pred_cal)
def icp_interval(pred_te, scores_cal, cov):
    q = conformal_q(scores_cal, cov)
    lo = np.clip(pred_te - q, 0, None); hi = pred_te + q
    return lo, hi

# ---- ICP normalized by scale sigma (e.g., clear-sky) ----
def icpn_fit(y_cal, pred_cal, sigma_cal): return np.abs(y_cal - pred_cal)/sigma_cal
def icpn_interval(pred_te, sigma_te, scores_cal, cov):
    q = conformal_q(scores_cal, cov)
    lo = np.clip(pred_te - q*sigma_te, 0, None); hi = pred_te + q*sigma_te
    return lo, hi

# ---- Mondrian (regime-conditional) ----
def mondrian_fit(y_cal, pred_cal, grp_cal):
    s = np.abs(y_cal - pred_cal); out={}
    for g in np.unique(grp_cal): out[g]=s[grp_cal==g]
    return out
def mondrian_interval(pred_te, grp_te, scores_by_grp, cov, fallback):
    q_te=np.empty(len(pred_te))
    for g in np.unique(grp_te):
        s = scores_by_grp.get(g, fallback)
        q_te[grp_te==g]=conformal_q(s, cov)
    lo=np.clip(pred_te-q_te,0,None); hi=pred_te+q_te
    return lo, hi

# ---- CQR ----
def cqr_fit(y_cal, qlo_cal, qhi_cal): return np.maximum(qlo_cal - y_cal, y_cal - qhi_cal)
def cqr_interval(qlo_te, qhi_te, scores_cal, cov):
    q = conformal_q(scores_cal, cov)
    lo=np.clip(qlo_te - q,0,None); hi=qhi_te + q
    return lo, hi

# ---- interval metrics ----
def picp(y, lo, hi): return float(np.mean((y>=lo)&(y<=hi)))
def pinaw(y, lo, hi):
    rng = y.max()-y.min(); return float(np.mean(hi-lo)/rng) if rng>0 else np.nan
def winkler(y, lo, hi, cov):
    alpha=1-cov; w=hi-lo
    pen=np.where(y<lo, (2/alpha)*(lo-y), np.where(y>hi,(2/alpha)*(y-hi),0.0))
    return float(np.mean(w+pen))
def pinball(y, q, tau):  # quantile/pinball loss
    d=y-q; return float(np.mean(np.maximum(tau*d,(tau-1)*d)))
def crps_from_quantiles(y, qdict):
    """Approximate CRPS as 2*mean over quantile levels of pinball loss."""
    taus=sorted(qdict); return float(2*np.mean([pinball(y,qdict[t],t) for t in taus]))
