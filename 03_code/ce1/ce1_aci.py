"""Adaptive conformal inference with delayed feedback in event time.

The outcome of the interval issued at time tau_j is observed at tau_j + h x 5 min. The interval
issued at tau_t uses exactly the miscoverage events with tau_j + h x 5 min <= tau_t, within the
same day after the daily reset. On gap-free days this equals the row-based update (self-test).
The latent level is unprojected, as in standard ACI (Gibbs and Candes, 2021). The projected
variant and the anticipative update (outcome known at the next issue time) are kept as
comparators.

Usage (from the repository root unless the path starts at ce1/, then from 03_code):
    python ce1/ce1_aci.py --selftest
"""
from __future__ import annotations

import sys
from collections import deque
from pathlib import Path

import numpy as np

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE / "conformal"))
sys.path.insert(0, str(CODE / "r1"))

STEP = np.timedelta64(5, "m")


def _q_at_cov(sorted_scores, cov):
    """Identical to conformal/conformal_adaptive.py::_q_at_cov (v1)."""
    n = len(sorted_scores)
    if cov >= 1.0:
        return np.inf
    if cov <= 0.0:
        return 0.0
    k = int(np.ceil((n + 1) * cov))
    if k > n:
        return np.inf
    return sorted_scores[max(k, 1) - 1]


def _interval(pred, s, a_lat):
    a_eff = min(max(a_lat, 0.0), 1.0)
    q = _q_at_cov(s, 1.0 - a_eff)
    if not np.isfinite(q):
        q = s[-1] if len(s) else 0.0
    return max(pred - q, 0.0), pred + q, a_eff


def aci_event(pred, y, ts, scores_cal, alpha, gamma, h_steps, grp=None,
              scores_by_grp=None, reset_daily=True, project=False, return_info=False):
    """ACI (marginal if grp is None, regime-conditional otherwise) with event-time
    feedback. `ts` are the issue timestamps (datetime64), strictly increasing.

    Returns lo, hi, alpha_eff trace (and an info dict if return_info).
    """
    ts = np.asarray(ts, dtype="datetime64[ns]")
    n = len(pred)
    assert len(y) == n and len(ts) == n
    if n > 1:
        assert (np.diff(ts) > np.timedelta64(0, "ns")).all(), "issue times must increase"
    delay = int(h_steps) * STEP
    days = ts.astype("datetime64[D]")
    regime = grp is not None
    if regime:
        pools = {g: np.sort(np.asarray(v, float)) for g, v in scores_by_grp.items()}
        fb = np.sort(np.asarray(scores_cal, float))
    else:
        s_all = np.sort(np.asarray(scores_cal, float))
    lo = np.empty(n); hi = np.empty(n); tr = np.empty(n)
    err = np.empty(n)
    a = {}  # latent level per group ("_" for marginal)
    queue = deque()  # rows issued today whose outcome has not arrived
    prev_day = None
    applied = 0
    row_delay_ne_h = 0  # issue times where the arrived set differs from row rule

    def upd(key, e):
        v = a.get(key, alpha) + gamma * (alpha - e)
        a[key] = min(max(v, 0.0), 1.0) if project else v

    for t in range(n):
        if reset_daily and days[t] != prev_day:
            a.clear()
            queue.clear()
            prev_day = days[t]
            day_start = t
        # apply every event observed by the issue time of row t
        while queue and ts[queue[0]] + delay <= ts[t]:
            j = queue.popleft()
            upd(grp[j] if regime else "_", err[j])
            applied += 1
        # v1 row rule would have applied exactly the rows j <= t - h of today
        n_row_rule = max(0, t - int(h_steps) - day_start + 1)
        n_event = (t - day_start) - len(queue)
        if n_row_rule != n_event:
            row_delay_ne_h += 1
        key = grp[t] if regime else "_"
        s = pools.get(key, fb) if regime else s_all
        lo[t], hi[t], tr[t] = _interval(pred[t], s, a.get(key, alpha))
        err[t] = 0.0 if (y[t] >= lo[t]) and (y[t] <= hi[t]) else 1.0
        queue.append(t)
    if return_info:
        return lo, hi, tr, dict(events_applied=applied, rows_where_event_rule_differs=row_delay_ne_h,
                                n=n)
    return lo, hi, tr


# ----------------------------------------------------------------------------------------
# Self-tests
# ----------------------------------------------------------------------------------------
def _reference(pred, y, ts, scores_cal, alpha, gamma, h_steps, grp=None, scores_by_grp=None):
    """Brute force, written independently: at each issue time the latent level of the
    relevant group is alpha + gamma * sum(alpha - err_j) over same-day rows j whose outcome
    time tau_j + h*5min is <= tau_t.
    """
    ts = np.asarray(ts, dtype="datetime64[ns]")
    delay = int(h_steps) * STEP
    days = ts.astype("datetime64[D]")
    n = len(pred)
    lo = np.empty(n); hi = np.empty(n); err = np.empty(n)
    for t in range(n):
        key = grp[t] if grp is not None else None
        acc = alpha
        for j in range(t):
            if days[j] != days[t] or ts[j] + delay > ts[t]:
                continue
            if grp is not None and grp[j] != key:
                continue
            acc += gamma * (alpha - err[j])
        s = (np.sort(scores_by_grp.get(key, scores_cal)) if grp is not None
             else np.sort(scores_cal))
        lo[t], hi[t], _ = _interval(pred[t], s, acc)
        err[t] = 0.0 if lo[t] <= y[t] <= hi[t] else 1.0
    return lo, hi


def _stream(n_days, gaps, seed):
    rng = np.random.default_rng(seed)
    ts = []
    for d in range(n_days):
        day0 = np.datetime64("2030-01-01T07:00") + np.timedelta64(d, "D")
        k = np.arange(144)
        if gaps:
            keep = rng.random(144) > 0.15
            keep[: int(rng.integers(0, 3))] = False
            k = k[keep]
        ts.extend(day0 + k * STEP)
    ts = np.array(ts, dtype="datetime64[ns]")
    n = len(ts)
    pred = 400 + 200 * rng.standard_normal(n)
    y = pred + 80 * rng.standard_normal(n) * (1 + (rng.random(n) < 0.2) * 3)
    grp = rng.choice(np.array(["clear", "transitional", "cloudy"]), size=n)
    scores = np.abs(80 * rng.standard_normal(3000))
    sbg = {g: np.abs((60 + 30 * i) * rng.standard_normal(1000))
           for i, g in enumerate(["clear", "transitional", "cloudy"])}
    return pred, y, ts, grp, scores, sbg


def selftest(verbose=True):
    import r1_conformal_delayed as R1D  # frozen v1 code, imported unchanged
    import conformal_adaptive as R0A
    res = {}

    def ok(name, cond, msg=""):
        res[name] = bool(cond)
        if verbose:
            print(f"  [{'PASS' if cond else 'FAIL'}] {name} {msg}")

    alpha, gamma = 0.10, 0.05
    # 1. gap-free days: identical to the v1 row rule, every horizon, both variants
    pred, y, ts, grp, sc, sbg = _stream(6, gaps=False, seed=1)
    day_id = ts.astype("datetime64[D]")
    for h in (1, 3, 6, 12):
        a1 = aci_event(pred, y, ts, sc, alpha, gamma, h)
        b1 = R1D.aci_run_delayed(pred, y, sc, alpha, gamma, day_id, h)
        ok(f"gapfree_equals_R1_marginal_h{h}", all(np.array_equal(u, v) for u, v in zip(a1, b1)))
        a2 = aci_event(pred, y, ts, sc, alpha, gamma, h, grp=grp, scores_by_grp=sbg)
        b2 = R1D.aci_regime_run_delayed(pred, y, grp, sbg, alpha, gamma, day_id, sc, h)
        ok(f"gapfree_equals_R1_regime_h{h}", all(np.array_equal(u, v) for u, v in zip(a2, b2)))
    a0 = aci_event(pred, y, ts, sc, alpha, gamma, 1)
    b0 = R0A.aci_run(pred, y, sc, alpha, gamma, day_id)
    ok("h1_equals_R0_undelayed", all(np.array_equal(u, v) for u, v in zip(a0, b0)))

    # 2. with gaps: equals the independent brute-force reference; row rule differs
    pred, y, ts, grp, sc, sbg = _stream(4, gaps=True, seed=2)
    for h in (1, 3, 6, 12):
        lo, hi, _, info = aci_event(pred, y, ts, sc, alpha, gamma, h, return_info=True)
        rl, rh = _reference(pred, y, ts, sc, alpha, gamma, h)
        ok(f"gaps_equals_reference_marginal_h{h}", np.array_equal(lo, rl) and np.array_equal(hi, rh),
           f"(issue times where event rule differs from row rule: {info['rows_where_event_rule_differs']})")
        lo, hi, _ = aci_event(pred, y, ts, sc, alpha, gamma, h, grp=grp, scores_by_grp=sbg)
        rl, rh = _reference(pred, y, ts, sc, alpha, gamma, h, grp=grp, scores_by_grp=sbg)
        ok(f"gaps_equals_reference_regime_h{h}", np.array_equal(lo, rl) and np.array_equal(hi, rh))
    _, _, _, info = aci_event(pred, y, ts, sc, alpha, gamma, 6, return_info=True)
    ok("gaps_event_rule_is_faster_than_row_rule", info["rows_where_event_rule_differs"] > 0)
    # causality: perturbing outcomes that arrive after tau_t never changes interval t
    h = 6
    lo, hi, _ = aci_event(pred, y, ts, sc, alpha, gamma, h)
    bad = 0
    rng = np.random.default_rng(5)
    for t in rng.choice(len(ts), 100, replace=False):
        y2 = y.copy()
        late = ts + h * STEP > ts[t]
        y2[late] = y2[late] + 1e4 * rng.standard_normal(late.sum())
        l2, h2, _ = aci_event(pred, y2, ts, sc, alpha, gamma, h)
        bad += int(l2[t] != lo[t] or h2[t] != hi[t])
    ok("no_feedback_before_arrival", bad == 0, f"(violations {bad}/100)")

    # 3. projected = latent while unsaturated; they differ once the latent level leaves [0, 1]
    pred, y, ts, grp, sc, sbg = _stream(3, gaps=True, seed=3)
    small = 0.002
    l1 = aci_event(pred, y, ts, sc, alpha, small, 3)
    l2 = aci_event(pred, y, ts, sc, alpha, small, 3, project=True)
    ok("projected_equals_latent_unsaturated",
       all(np.array_equal(u, v) for u, v in zip(l1, l2)) and 0 < l1[2].min() and l1[2].max() < 1)
    big = 0.5
    l1 = aci_event(pred, y, ts, sc, alpha, big, 3)
    l2 = aci_event(pred, y, ts, sc, alpha, big, 3, project=True)
    ok("projected_differs_when_saturated", not np.array_equal(l1[2], l2[2]))
    return res


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        r = selftest()
        sys.exit(0 if all(r.values()) else 1)
