"""ACI with h-step feedback delay (v1 code). The interval issued at index t uses only miscoverage
events whose outcomes are realised by t, that is events j with j + h <= t (cf. Szabadvary,
2024).
"""
import numpy as np

from conformal_adaptive import _q_at_cov


def aci_run_delayed(pred_te, y_te, scores_cal, alpha, gamma, day_id, h_steps,
                    reset_daily=True):
    """Marginal ACI with h-step feedback delay.

    Identical signature to `conformal_adaptive.aci_run` plus `h_steps`.
    Returns (lo, hi, alpha_trace).
    """
    s = np.sort(np.asarray(scores_cal, float))
    n = len(pred_te)
    h = int(h_steps)
    lo = np.empty(n); hi = np.empty(n); a_trace = np.empty(n)
    err = np.full(n, np.nan)  # miscoverage indicator per issue time
    a_t = alpha
    prev_day = None
    day_start = 0
    for t in range(n):
        if reset_daily and day_id[t] != prev_day:
            a_t = alpha
            prev_day = day_id[t]
            day_start = t  # clears the pending buffer for the new day
        a_eff = min(max(a_t, 0.0), 1.0)
        q = _q_at_cov(s, 1.0 - a_eff)
        if not np.isfinite(q):
            q = s[-1] if len(s) else 0.0
        lo[t] = max(pred_te[t] - q, 0.0)
        hi[t] = pred_te[t] + q
        a_trace[t] = a_eff
        err[t] = 0.0 if (y_te[t] >= lo[t]) and (y_te[t] <= hi[t]) else 1.0
        # --- feedback that has become observable by t+1 ---
        j = t + 1 - h
        if j >= day_start:
            a_t = a_t + gamma * (alpha - err[j])
    return lo, hi, a_trace


def aci_regime_run_delayed(pred_te, y_te, grp_te, scores_by_grp, alpha, gamma,
                           day_id, fallback_scores, h_steps, reset_daily=True):
    """Regime-conditional ACI with h-step feedback delay.

    The delayed event of index j updates the alpha of the regime *of index j*, which
    is the regime whose interval produced that miscoverage -- not the regime of the
    current index.
    """
    sorted_by_grp = {g: np.sort(np.asarray(v, float)) for g, v in scores_by_grp.items()}
    fb = np.sort(np.asarray(fallback_scores, float))
    n = len(pred_te)
    h = int(h_steps)
    lo = np.empty(n); hi = np.empty(n); a_trace = np.empty(n)
    err = np.full(n, np.nan)
    a_t = {}
    prev_day = None
    day_start = 0
    groups = list(sorted_by_grp.keys())
    for g in groups:
        a_t[g] = alpha
    for t in range(n):
        if reset_daily and day_id[t] != prev_day:
            for g in list(a_t):
                a_t[g] = alpha
            prev_day = day_id[t]
            day_start = t
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
        err[t] = 0.0 if (y_te[t] >= lo[t]) and (y_te[t] <= hi[t]) else 1.0
        j = t + 1 - h
        if j >= day_start:
            gj = grp_te[j]
            if gj not in a_t:
                a_t[gj] = alpha
            a_t[gj] = a_t[gj] + gamma * (alpha - err[j])
    return lo, hi, a_trace
