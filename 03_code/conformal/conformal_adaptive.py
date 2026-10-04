"""Adaptive and regime-conditional conformal methods: ACI (Gibbs and Candes, 2021) with a daily
reset, regime-conditional ACI, Mondrian-CQR (Mondrian calibration with CQR), and coverage over
time buckets.
"""
import numpy as np
import conformal as CP  # base module (same package dir)


# ---------------------------------------------------------------------------
# Conformal quantile of a FIXED score pool at an arbitrary coverage level.
# Returns +inf when the requested coverage is not attainable from n scores
# (i.e. the interval must cover everything) -> ACI then widens to full range.
# ---------------------------------------------------------------------------
def _q_at_cov(sorted_scores, cov):
    n = len(sorted_scores)
    if cov >= 1.0:
        return np.inf
    if cov <= 0.0:
        return 0.0
    k = int(np.ceil((n + 1) * cov))
    if k > n:
        return np.inf
    return sorted_scores[max(k, 1) - 1]


# ---------------------------------------------------------------------------
# Adaptive Conformal Inference (marginal), online, with per-day reset.
#   alpha: nominal miscoverage (e.g. 0.10 for 90%)
#   gamma: ACI learning rate (step size of the alpha_t update)
#   day_id: integer/label per test point identifying its day (reset key)
# Update: alpha_{t+1} = alpha_t + gamma * (alpha - err_t),  err_t = 1 if missed.
# At each new day, alpha_t is reset to the nominal alpha.
# ---------------------------------------------------------------------------
def aci_run(pred_te, y_te, scores_cal, alpha, gamma, day_id, reset_daily=True):
    s = np.sort(np.asarray(scores_cal, float))
    n = len(pred_te)
    lo = np.empty(n); hi = np.empty(n); a_trace = np.empty(n)
    a_t = alpha
    prev_day = None
    for t in range(n):
        if reset_daily and day_id[t] != prev_day:
            a_t = alpha
            prev_day = day_id[t]
        a_eff = min(max(a_t, 0.0), 1.0)
        q = _q_at_cov(s, 1.0 - a_eff)
        if not np.isfinite(q):
            q = s[-1] if len(s) else 0.0  # widen to max observed score
        lo[t] = max(pred_te[t] - q, 0.0)
        hi[t] = pred_te[t] + q
        a_trace[t] = a_eff
        covered = (y_te[t] >= lo[t]) and (y_te[t] <= hi[t])
        err = 0.0 if covered else 1.0
        a_t = a_t + gamma * (alpha - err)
    return lo, hi, a_trace


# ---------------------------------------------------------------------------
# Regime-conditional ACI: independent alpha_t and score pool per regime.
# (Online adaptivity *within* each weather regime.)
# ---------------------------------------------------------------------------
def aci_regime_run(pred_te, y_te, grp_te, scores_by_grp, alpha, gamma, day_id,
                   fallback_scores, reset_daily=True):
    sorted_by_grp = {g: np.sort(np.asarray(v, float)) for g, v in scores_by_grp.items()}
    fb = np.sort(np.asarray(fallback_scores, float))
    n = len(pred_te)
    lo = np.empty(n); hi = np.empty(n); a_trace = np.empty(n)
    a_t = {}; prev_day = None
    groups = list(sorted_by_grp.keys())
    for g in groups:
        a_t[g] = alpha
    for t in range(n):
        if reset_daily and day_id[t] != prev_day:
            for g in groups:
                a_t[g] = alpha
            prev_day = day_id[t]
        g = grp_te[t]
        s = sorted_by_grp.get(g, fb)
        if g not in a_t:
            a_t[g] = alpha
        a_eff = min(max(a_t[g], 0.0), 1.0)
        q = _q_at_cov(s, 1.0 - a_eff)
        if not np.isfinite(q):
            q = s[-1] if len(s) else 0.0
        lo[t] = max(pred_te[t] - q, 0.0)
        hi[t] = pred_te[t] + q
        a_trace[t] = a_eff
        covered = (y_te[t] >= lo[t]) and (y_te[t] <= hi[t])
        err = 0.0 if covered else 1.0
        a_t[g] = a_t[g] + gamma * (alpha - err)
    return lo, hi, a_trace


# ---------------------------------------------------------------------------
# Mondrian-CQR: regime-conditional conformalized quantile regression.
# Fit a separate CQR correction E_i = max(qlo - y, y - qhi) per regime, then
# apply the per-regime conformal quantile on the test set.
# ---------------------------------------------------------------------------
def mondrian_cqr_fit(y_cal, qlo_cal, qhi_cal, grp_cal):
    s = np.maximum(qlo_cal - y_cal, y_cal - qhi_cal)
    return {g: s[grp_cal == g] for g in np.unique(grp_cal)}

def mondrian_cqr_interval(qlo_te, qhi_te, grp_te, scores_by_grp, cov, fallback):
    q_te = np.empty(len(qlo_te))
    for g in np.unique(grp_te):
        sc = scores_by_grp.get(g, fallback)
        q_te[grp_te == g] = CP.conformal_q(sc, cov)
    lo = np.clip(qlo_te - q_te, 0, None)
    hi = qhi_te + q_te
    return lo, hi


# ---------------------------------------------------------------------------
# Reliability over time: empirical coverage (and mean width) in time buckets.
#   bucket: array of bucket labels (e.g. year-month string) per test point.
# Returns list of dicts.
# ---------------------------------------------------------------------------
def reliability_over_time(y, lo, hi, bucket):
    out = []
    bucket = np.asarray(bucket)
    for b in sorted(set(bucket.tolist())):
        m = bucket == b
        if m.sum() < 30:
            continue
        out.append(dict(bucket=str(b), n=int(m.sum()),
                        PICP=float(np.mean((y[m] >= lo[m]) & (y[m] <= hi[m]))),
                        width=float(np.mean(hi[m] - lo[m]))))
    return out
