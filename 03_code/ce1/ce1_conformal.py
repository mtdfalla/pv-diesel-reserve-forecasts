"""Stage S5: the interval layer for every site, fold and horizon, from the stage S2 caches.

Static split conformal (marginal, Mondrian, CQR, Mondrian-CQR, normalized variants), the
probabilistic baselines and ACI with event-time delayed feedback (ce1_aci.py). Nothing is
refitted. Rules: D-089.

Usage (from the repository root unless the path starts at ce1/, then from 03_code):
    python ce1/ce1_conformal.py --stage S5 [--smoke] [--only yulara:2024:3]
    python ce1/ce1_conformal.py --selftest
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
CODE = HERE.parent
for p in (HERE, CODE / "utils", CODE / "conformal", CODE / "r1"):
    sys.path.insert(0, str(p))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import conformal as CP  # noqa: E402  (v1, imported unchanged)
import conformal_adaptive as CA  # noqa: E402  (v1, imported unchanged)
import r1_conformal_delayed as R1D  # noqa: E402  (v1, imported unchanged)
import ce1_aci as A  # noqa: E402

COVS = [0.80, 0.90, 0.95]
GAMMA = 0.05
GAMMA_SWEEP = [0.01, 0.02, 0.05, 0.10]
TAUS_Q = [0.025, 0.05, 0.10, 0.90, 0.95, 0.975]
CS_FLOOR = 50.0
CAP = 1.5
MIN_GRP = 30  # tod: marginal pool for an hour with <= 30 scores
MIN_SCOPE = 30
CALIB_MONTHS = [0.5, 1, 2, 3, 6, 9, 12]
REGIMES = ["clear", "transitional", "cloudy"]
STATIC = ["icp", "icp_norm", "mondrian", "cqr", "mondrian_cqr", "raw_q", "tod", "nicp_var"]
FOLDS = {"yulara": [2019, 2020, 2021, 2022, 2023, 2024], "asp": [2024]}
HORIZONS = [1, 3, 6, 12]
SETTINGS = dict(covs=COVS, gamma=GAMMA, gamma_sweep=GAMMA_SWEEP, taus=TAUS_Q, cs_floor=CS_FLOOR,
                cap=CAP, min_grp=MIN_GRP, calib_months=CALIB_MONTHS, static=STATIC, version=1)


def log(m):
    print(f"[S5] {m}", flush=True)


def md5(p: Path) -> str:
    h = hashlib.md5()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def qcol(tau):
    return f"q{tau:.6f}"


def band(cov):
    a = 1 - cov
    return round(a / 2, 3), round(1 - a / 2, 3)


def hours(ts):
    ts = np.asarray(ts, dtype="datetime64[ns]")
    return ((ts - ts.astype("datetime64[D]")) // np.timedelta64(1, "h")).astype(int)


def tod_fit(y_cal, p_cal, h_cal):
    s = np.abs(y_cal - p_cal)
    return {int(k): s[h_cal == k] for k in np.unique(h_cal)}, s


def tod_interval(p_te, h_te, pools, s_all, cov):
    q = np.empty(len(p_te))
    for k in np.unique(h_te):
        pk = pools.get(int(k))
        pool = pk if (pk is not None and len(pk) > MIN_GRP) else s_all
        q[h_te == k] = CP.conformal_q(pool, cov)
    return np.clip(p_te - q, 0, None), p_te + q


def nicp_var_scale(gcs, v, beta):
    return np.asarray(gcs, float) * (np.asarray(v, float) + beta)


def raw_interval(qlo, qhi):
    return np.clip(qlo, 0, None), np.asarray(qhi, float).copy()


# ------------------------------------------------------------------ metrics
def pinball_obs(y, q, tau):
    d = y - q
    return np.maximum(tau * d, (tau - 1) * d)


def crps_obs(y, qd):
    taus = sorted(qd)
    return 2.0 * np.mean(np.column_stack([pinball_obs(y, qd[t], t) for t in taus]), axis=1)


def crossing_stats(qd, taus, y_mean):
    M = np.column_stack([qd[t] for t in taus])
    dif = np.diff(M, axis=1)
    viol = dif < 0
    any_v = viol.any(axis=1)
    worst = np.where(any_v, -dif.min(axis=1), 0.0)
    n_pts = int(any_v.sum())
    return dict(n=int(M.shape[0]), n_crossing=n_pts, rate=round(n_pts / M.shape[0], 6),
                mean_violation_Wm2=round(float(worst[worst > 0].mean()) if n_pts else 0.0, 4),
                max_violation_Wm2=round(float(worst.max()), 4),
                mean_violation_pct_of_mean_GHI=round(float(worst[worst > 0].mean() / y_mean * 100)
                                                     if n_pts else 0.0, 4))


def rearrange(qd):
    taus = sorted(qd)
    M = np.sort(np.column_stack([qd[t] for t in taus]), axis=1)
    return {t: M[:, i] for i, t in enumerate(taus)}


# ------------------------------------------------------------------ one job
def split_cache(d: pd.DataFrame):
    c, t = d[d["split"] == "calib"], d[d["split"] == "test"]
    assert c.index.is_monotonic_increasing and t.index.is_monotonic_increasing
    return c, t


def static_intervals(c, t, cov):
    """Every static method at one nominal level. Returns {method: (lo, hi)}."""
    yc, pc, pt = c["y"].values, c["p"].values, t["p"].values
    gc, gt = c["g"].values.astype(str), t["g"].values.astype(str)
    lt, ht = band(cov)
    qc_lo, qc_hi, qt_lo, qt_hi = (c[qcol(lt)].values, c[qcol(ht)].values,
                                  t[qcol(lt)].values, t[qcol(ht)].values)
    icp_s = CP.icp_fit(yc, pc)
    iv = {}
    iv["icp"] = CP.icp_interval(pt, icp_s, cov)
    iv["icp_norm"] = CP.icpn_interval(pt, np.maximum(t["gcs"].values, CS_FLOOR),
                                      CP.icpn_fit(yc, pc, np.maximum(c["gcs"].values, CS_FLOOR)), cov)
    iv["mondrian"] = CP.mondrian_interval(pt, gt, CP.mondrian_fit(yc, pc, gc), cov, icp_s)
    cqr_s = CP.cqr_fit(yc, qc_lo, qc_hi)
    iv["cqr"] = CP.cqr_interval(qt_lo, qt_hi, cqr_s, cov)
    iv["mondrian_cqr"] = CA.mondrian_cqr_interval(qt_lo, qt_hi, gt,
                                                  CA.mondrian_cqr_fit(yc, qc_lo, qc_hi, gc), cov, cqr_s)
    iv["raw_q"] = raw_interval(qt_lo, qt_hi)
    pools, s_all = tod_fit(yc, pc, hours(c.index.values))
    iv["tod"] = tod_interval(pt, hours(t.index.values), pools, s_all, cov)
    beta = float(np.median(c["v"].values))
    iv["nicp_var"] = CP.icpn_interval(pt, nicp_var_scale(t["gcs"], t["v"], beta),
                                      CP.icpn_fit(yc, pc, nicp_var_scale(c["gcs"], c["v"], beta)), cov)
    return iv, dict(beta=beta)


def in_sample_coverage(c, cov):
    """Coverage of the static methods on their own calibration rows (>= cov by the
    finite-sample construction; a stage check).
    """
    iv, _ = static_intervals(c, c, cov)
    y = c["y"].values
    out = {m: CP.picp(y, *iv[m]) for m in ("icp", "icp_norm", "cqr", "tod", "nicp_var")}
    g = c["g"].values.astype(str)
    for m in ("mondrian", "mondrian_cqr"):
        out[m] = min(CP.picp(y[g == r], iv[m][0][g == r], iv[m][1][g == r]) for r in np.unique(g))
    return out


def run_job(d: pd.DataFrame, site: str, fold: int, h: int, obs=True, calib_size=False):
    c, t = split_cache(d)
    yt, pt = t["y"].values, t["p"].values
    ts = t.index.values
    day = t.index.normalize().values
    month = t.index.to_period("M").astype(str).values
    g_it, g_tt = t["g"].values.astype(str), t["g_t"].values.astype(str)
    yc, pc = c["y"].values, c["p"].values
    gc = c["g"].values.astype(str)
    icp_s = np.abs(yc - pc)
    mon_s = CP.mondrian_fit(yc, pc, gc)
    scopes = [("all", np.ones(len(yt), bool))]
    scopes += [(r, g_it == r) for r in REGIMES] + [(f"tt_{r}", g_tt == r) for r in REGIMES]
    base = dict(site=site, fold=fold, horizon_min=h * 5)
    recs, crps_recs, rel, capr = [], [], [], []
    O = {}  # per-observation arrays
    info = dict(n_calib=int(len(c)), n_test=int(len(t)), n_days_test=int(len(np.unique(day))))

    def add(method, variant, cov, lo, hi, scoped=True):
        cv = (yt >= lo) & (yt <= hi)
        for sname, m in (scopes if scoped else scopes[:1]):
            if m.sum() < MIN_SCOPE:
                continue
            p = float(cv[m].mean())
            recs.append(dict(base, method=method, variant=variant, nominal=cov, scope=sname,
                             n=int(m.sum()), PICP=round(p, 4), ACE=round(p - cov, 4),
                             PINAW=round(CP.pinaw(yt[m], lo[m], hi[m]), 4),
                             width_Wm2=round(float(np.mean(hi[m] - lo[m])), 2),
                             Winkler=round(CP.winkler(yt[m], lo[m], hi[m], cov), 2)))
        if obs:
            O[f"cov__{method}|{variant}|{cov}"] = cv
            O[f"wid__{method}|{variant}|{cov}"] = (hi - lo).astype(np.float32)

    # ---------------------------------------------------------------- static
    qstore = {m: {} for m in STATIC}
    qrear = {m: {} for m in ("cqr", "mondrian_cqr")}
    qt_full = {tau: t[qcol(tau)].values for tau in TAUS_Q}; qt_full[0.5] = pt
    qc_full = {tau: c[qcol(tau)].values for tau in TAUS_Q}; qc_full[0.5] = pc
    qt_r, qc_r = rearrange(qt_full), rearrange(qc_full)
    for cov in COVS:
        iv, extra = static_intervals(c, t, cov)
        info["nicp_var_beta"] = extra["beta"]
        lt, ht = band(cov)
        for m, (lo, hi) in iv.items():
            add(m, "static", cov, lo, hi)
            qstore[m][lt], qstore[m][ht] = lo, hi
            if abs(cov - 0.90) < 1e-9:
                for r in CA.reliability_over_time(yt, lo, hi, month):
                    rel.append(dict(base, method=m, variant="static", **r))
            hi_c = np.minimum(hi, CAP * np.maximum(t["gcs"].values, 0.0))
            capr.append(dict(base, method=m, nominal=cov, n=int(len(yt)),
                             frac_capped=round(float((hi_c < hi - 1e-9).mean()), 6),
                             PICP=round(CP.picp(yt, lo, hi), 4), PICP_capped=round(CP.picp(yt, lo, hi_c), 4),
                             PINAW=round(CP.pinaw(yt, lo, hi), 4), PINAW_capped=round(CP.pinaw(yt, lo, hi_c), 4),
                             Winkler=round(CP.winkler(yt, lo, hi, cov), 2),
                             Winkler_capped=round(CP.winkler(yt, lo, hi_c, cov), 2)))
        # CQR family on rearranged quantiles (reported variant)
        s_r = CP.cqr_fit(yc, qc_r[lt], qc_r[ht])
        lo, hi = CP.cqr_interval(qt_r[lt], qt_r[ht], s_r, cov)
        add("cqr", "rearranged", cov, lo, hi, scoped=False)
        qrear["cqr"][lt], qrear["cqr"][ht] = lo, hi
        lo, hi = CA.mondrian_cqr_interval(qt_r[lt], qt_r[ht], g_it,
                                          CA.mondrian_cqr_fit(yc, qc_r[lt], qc_r[ht], gc), cov, s_r)
        add("mondrian_cqr", "rearranged", cov, lo, hi, scoped=False)
        qrear["mondrian_cqr"][lt], qrear["mondrian_cqr"][ht] = lo, hi

    # ---------------------------------------------------------------- adaptive
    aci_q = {k: {} for k in ("aci|delayed", "aci_regime|delayed", "aci|anticipative",
                             "aci_regime|anticipative")}
    ident = []
    same_h1 = []
    ev_info = None
    for cov in COVS:
        a = 1 - cov
        lt, ht = band(cov)
        runs = {}
        runs[("aci", "anticipative")] = CA.aci_run(pt, yt, icp_s, a, GAMMA, day)
        runs[("aci_regime", "anticipative")] = CA.aci_regime_run(pt, yt, g_it, mon_s, a, GAMMA, day, icp_s)
        e1 = A.aci_event(pt, yt, ts, icp_s, a, GAMMA, 1)
        e2 = A.aci_event(pt, yt, ts, icp_s, a, GAMMA, 1, grp=g_it, scores_by_grp=mon_s)
        ident.append(all(np.array_equal(u, v) for u, v in zip(e1, runs[("aci", "anticipative")]))
                     and all(np.array_equal(u, v) for u, v in zip(e2, runs[("aci_regime", "anticipative")])))
        # delayed, event time (primary) and projected (sensitivity)
        lo, hi, tr, inf = A.aci_event(pt, yt, ts, icp_s, a, GAMMA, h, return_info=True)
        runs[("aci", "delayed")] = (lo, hi, tr)
        ev_info = inf
        runs[("aci_regime", "delayed")] = A.aci_event(pt, yt, ts, icp_s, a, GAMMA, h, grp=g_it,
                                                      scores_by_grp=mon_s)
        runs[("aci", "delayed_projected")] = A.aci_event(pt, yt, ts, icp_s, a, GAMMA, h, project=True)
        runs[("aci_regime", "delayed_projected")] = A.aci_event(pt, yt, ts, icp_s, a, GAMMA, h, grp=g_it,
                                                                scores_by_grp=mon_s, project=True)
        if h == 1:
            same_h1.append(all(np.array_equal(u, v) for k in ("aci", "aci_regime")
                               for u, v in zip(runs[(k, "delayed")], runs[(k, "anticipative")])))
        if abs(cov - 0.90) < 1e-9:
            runs[("aci", "delayed_rows_r1")] = R1D.aci_run_delayed(pt, yt, icp_s, a, GAMMA, day, h)
            runs[("aci_regime", "delayed_rows_r1")] = R1D.aci_regime_run_delayed(pt, yt, g_it, mon_s, a,
                                                                                 GAMMA, day, icp_s, h)
        for (m, var), (lo, hi, tr) in runs.items():
            add(m, var, cov, lo, hi)
            if f"{m}|{var}" in aci_q:
                aci_q[f"{m}|{var}"][lt], aci_q[f"{m}|{var}"][ht] = lo, hi
            if abs(cov - 0.90) < 1e-9:
                if var in ("anticipative", "delayed"):
                    for r in CA.reliability_over_time(yt, lo, hi, month):
                        rel.append(dict(base, method=m, variant=var, **r))
                if obs:
                    O[f"alpha__{m}|{var}"] = np.asarray(tr, np.float32)
        # learning-rate sweep (all scope)
        for gm in GAMMA_SWEEP:
            for var in ("anticipative", "delayed"):
                if gm == GAMMA:
                    lo, hi, _ = runs[("aci", var)]
                elif var == "anticipative":
                    lo, hi, _ = CA.aci_run(pt, yt, icp_s, a, gm, day)
                else:
                    lo, hi, _ = A.aci_event(pt, yt, ts, icp_s, a, gm, h)
                p = CP.picp(yt, lo, hi)
                recs.append(dict(base, method=f"aci_g{gm}", variant=var, nominal=cov, scope="all",
                                 n=int(len(yt)), PICP=round(p, 4), ACE=round(p - cov, 4),
                                 PINAW=round(CP.pinaw(yt, lo, hi), 4),
                                 width_Wm2=round(float(np.mean(hi - lo)), 2),
                                 Winkler=round(CP.winkler(yt, lo, hi, cov), 2)))
    info["anticipative_equals_event_rule_h1"] = bool(all(ident))
    info["delayed_equals_anticipative_at_h1"] = bool(all(same_h1)) if h == 1 else None
    info["event_rule"] = ev_info

    # ---------------------------------------------------------------- CRPS proxy
    def crps_block(store, variant):
        for m, qd in store.items():
            qd = dict(qd); qd[0.5] = pt
            co = crps_obs(yt, qd)
            if obs:
                O[f"crps__{m}|{variant}"] = co.astype(np.float32)
            for sname, msk in scopes:
                if msk.sum() >= MIN_SCOPE:
                    crps_recs.append(dict(base, method=m, variant=variant, scope=sname, n=int(msk.sum()),
                                          CRPS=round(float(co[msk].mean()), 3)))
    crps_block(qstore, "static")
    crps_block(qrear, "rearranged")
    crps_block({k.split("|")[0]: v for k, v in aci_q.items() if k.endswith("|delayed")}, "delayed")
    crps_block({k.split("|")[0]: v for k, v in aci_q.items() if k.endswith("|anticipative")}, "anticipative")

    # ---------------------------------------------------------------- crossing
    crossing = {"test_quantiles_only": crossing_stats({k: v for k, v in qt_full.items() if k != 0.5},
                                                      TAUS_Q, float(np.mean(yt))),
                "test_with_point_as_median": crossing_stats(qt_full, sorted(qt_full), float(np.mean(yt))),
                "calib_quantiles_only": crossing_stats({k: v for k, v in qc_full.items() if k != 0.5},
                                                       TAUS_Q, float(np.mean(yc)))}
    for cov in COVS:
        lt, ht = band(cov)
        bad = qt_full[lt] > qt_full[ht]
        crossing[f"band_edges_{int(round(cov * 100))}"] = dict(n=int(len(yt)), n_crossing=int(bad.sum()),
                                                               rate=round(float(bad.mean()), 6))
    # ---------------------------------------------------------------- calibration size
    cs_rows = []
    if calib_size:
        last = c.index.max()
        for months in CALIB_MONTHS:
            msk = c.index >= last - pd.DateOffset(days=int(months * 30))
            if msk.sum() < 100:
                continue
            ivs, _ = static_intervals(c[msk], t, 0.90)
            for m in ("icp", "mondrian", "cqr", "mondrian_cqr"):
                lo, hi = ivs[m]
                tr_m = g_it == "transitional"
                cs_rows.append(dict(base, calib_months=months, calib_n=int(msk.sum()), method=m,
                                    PICP=round(CP.picp(yt, lo, hi), 4), PINAW=round(CP.pinaw(yt, lo, hi), 4),
                                    PICP_transitional=round(CP.picp(yt[tr_m], lo[tr_m], hi[tr_m]), 4)))
    info["in_sample_coverage_90"] = {k: round(v, 5) for k, v in in_sample_coverage(c, 0.90).items()}
    if obs:
        O["y"] = yt.astype(np.float32); O["p"] = pt.astype(np.float32)
        ud, dcode = np.unique(day, return_inverse=True)
        O["day_code"] = dcode.astype(np.int32)
        O["reg_it"] = np.array([REGIMES.index(x) if x in REGIMES else -1 for x in g_it], np.int8)
        O["reg_tt"] = np.array([REGIMES.index(x) if x in REGIMES else -1 for x in g_tt], np.int8)
        O["ts_ns"] = t.index.values.astype("datetime64[ns]").astype(np.int64)
    return dict(intervals=recs, crps=crps_recs, reliability=rel, bound_cap=capr, crossing=crossing,
                calib_size=cs_rows, info=info), O


# ------------------------------------------------------------------ self-tests
def _synthetic(n_days=60, seed=0, gaps=True, shift=0.0):
    rng = np.random.default_rng(seed)
    ts = []
    for dd in range(n_days):
        k = np.arange(20, 130)
        if gaps:
            k = k[rng.random(len(k)) > 0.1]
        ts.extend(np.datetime64("2030-01-01T00:00") + np.timedelta64(dd, "D") + k * A.STEP)
    idx = pd.DatetimeIndex(np.array(ts, dtype="datetime64[ns]"), name="ts")
    n = len(idx)
    hr = idx.hour.values + idx.minute.values / 60
    gcs = np.clip(1000 * np.sin(np.pi * (hr - 6) / 12), 0, None) + 5
    g = rng.choice(np.array(REGIMES), size=n, p=[0.55, 0.35, 0.10])
    sd = np.where(g == "clear", 20, np.where(g == "transitional", 120, 60)) * (1 + shift)
    p = gcs * rng.uniform(0.4, 1.0, n)
    y = np.clip(p + sd * rng.standard_normal(n), 0, None)
    v = np.where(g == "clear", 0.02, np.where(g == "transitional", 0.2, 0.1)) + 0.01 * rng.random(n)
    d = pd.DataFrame(dict(y=y, p=p, gcs=gcs, g=g, g_t=np.roll(g, -1), v=v), index=idx)
    for tau in TAUS_Q:
        from scipy.stats import norm
        d[qcol(tau)] = np.clip(p + sd * norm.ppf(tau) * 0.8, 0, None)
    half = idx[len(idx) // 2].normalize()
    d["split"] = np.where(idx < half, "calib", "test")
    return d


def selftest(verbose=True):
    res = {}

    def ok(name, cond, msg=""):
        res[name] = bool(cond)
        if verbose:
            print(f"  [{'PASS' if cond else 'FAIL'}] {name} {msg}", flush=True)

    d = _synthetic()
    c, t = split_cache(d)
    # 1. in-sample coverage >= nominal (finite-sample construction)
    for cov in COVS:
        ins = in_sample_coverage(c, cov)
        ok(f"in_sample_coverage_ge_nominal_{cov}", all(v >= cov - 1e-12 for v in ins.values()),
           f"(min {min(ins.values()):.4f})")
    # 2. exchangeable data: test coverage near nominal for every static conformal method
    iv, _ = static_intervals(c, t, 0.90)
    cov_t = {m: CP.picp(t["y"].values, *iv[m]) for m in iv}
    ok("exchangeable_test_coverage_near_nominal",
       all(abs(cov_t[m] - 0.90) < 0.02 for m in iv if m != "raw_q"),
       "(" + ", ".join(f"{m} {v:.3f}" for m, v in cov_t.items()) + ")")
    # 3. tod with a single clock hour equals marginal ICP; nicp_var with constant scale equals ICP
    d1 = d.copy()
    d1.index = pd.DatetimeIndex(d1.index.normalize() + pd.Timedelta(hours=10)
                                + pd.to_timedelta(np.arange(len(d1)) % 700, unit="s"), name="ts")
    d1 = d1[~d1.index.duplicated()].sort_index()
    c1, t1 = split_cache(d1)
    iv1, _ = static_intervals(c1, t1, 0.9)
    ok("tod_single_hour_equals_icp", np.array_equal(iv1["tod"][0], iv1["icp"][0])
       and np.array_equal(iv1["tod"][1], iv1["icp"][1]))
    d2 = d.assign(gcs=400.0, v=0.1)
    c2, t2 = split_cache(d2)
    iv2, _ = static_intervals(c2, t2, 0.9)
    ok("nicp_var_constant_scale_equals_icp", np.allclose(iv2["nicp_var"][0], iv2["icp"][0])
       and np.allclose(iv2["nicp_var"][1], iv2["icp"][1]))
    # 4. raw_q is CQR without correction
    lt, ht = band(0.9)
    ok("raw_q_is_clipped_quantiles", np.array_equal(iv["raw_q"][0], np.clip(t[qcol(lt)].values, 0, None))
       and np.array_equal(iv["raw_q"][1], t[qcol(ht)].values))
    # 5. tod falls back to the marginal pool for a sparse hour
    pools, s_all = tod_fit(c["y"].values, c["p"].values, hours(c.index.values))
    hsp = np.array([99] * 3)
    lo, hi = tod_interval(np.array([500.0] * 3), hsp, {99: np.array([1.0, 2, 3])}, s_all, 0.9)
    ok("tod_sparse_hour_falls_back", np.allclose(hi - 500, CP.conformal_q(s_all, 0.9)))
    # 6. crps proxy equals the v1 aggregate function
    qd = {0.05: iv["icp"][0], 0.95: iv["icp"][1], 0.5: t["p"].values}
    ok("crps_obs_mean_equals_R1", np.isclose(crps_obs(t["y"].values, qd).mean(),
                                             CP.crps_from_quantiles(t["y"].values, qd)))
    # 7. rearrangement: monotone, identity when nothing crosses
    rq = rearrange({0.1: np.array([3.0, 1]), 0.5: np.array([2.0, 2]), 0.9: np.array([1.0, 3])})
    ok("rearrange_monotone_and_identity", np.array_equal(rq[0.1], [1, 1]) and np.array_equal(rq[0.9], [3, 3])
       and np.array_equal(rq[0.5], [2, 2]))
    # 8. the whole job: identities and shapes
    out, O = run_job(d, "syn", 2030, 3, obs=True, calib_size=True)
    ok("job_anticipative_equals_event_rule_h1", out["info"]["anticipative_equals_event_rule_h1"])
    out1, _ = run_job(d, "syn", 2030, 1, obs=False)
    ok("job_delayed_equals_anticipative_at_h1", out1["info"]["delayed_equals_anticipative_at_h1"])
    n_t = len(t)
    ok("obs_arrays_have_test_length", all(len(v) == n_t for v in O.values()), f"({len(O)} arrays)")
    ok("no_nan_in_records", all(np.isfinite(r["PICP"]) and np.isfinite(r["Winkler"]) for r in out["intervals"])
       and all(np.isfinite(r["CRPS"]) for r in out["crps"]))
    iv80, _ = static_intervals(c, t, 0.80)
    iv95, _ = static_intervals(c, t, 0.95)
    ok("widths_nested_80_90_95", all((np.mean(iv80[m][1] - iv80[m][0]) <= np.mean(iv[m][1] - iv[m][0])
                                      <= np.mean(iv95[m][1] - iv95[m][0])) for m in iv))
    sc = {r["scope"] for r in out["intervals"]}
    ok("issue_and_target_time_scopes", {"all", "clear", "transitional", "tt_clear", "tt_transitional"} <= sc)
    ok("calib_size_rows", len(out["calib_size"]) > 0)
    # 9. Mondrian beats ICP on the transitional regime when regimes differ in noise
    tr = t["g"].values == "transitional"
    ok("mondrian_fixes_transitional_undercoverage",
       CP.picp(t["y"].values[tr], iv["icp"][0][tr], iv["icp"][1][tr]) < 0.85
       and abs(CP.picp(t["y"].values[tr], iv["mondrian"][0][tr], iv["mondrian"][1][tr]) - 0.9) < 0.03)
    # 10. drift: the delayed ACI moves coverage toward nominal relative to static ICP
    dd = _synthetic(seed=4)
    te = dd["split"] == "test"
    dd.loc[te, "y"] = np.clip(dd.loc[te, "p"] + 1.6 * (dd.loc[te, "y"] - dd.loc[te, "p"]), 0, None)
    o3, _ = run_job(dd, "syn", 2030, 3, obs=False)
    get = lambda m, v: [r["PICP"] for r in o3["intervals"] if r["method"] == m and r["variant"] == v
                        and r["scope"] == "all" and r["nominal"] == 0.9][0]
    ok("aci_delayed_corrects_drift", abs(get("aci", "delayed") - 0.9) < abs(get("icp", "static") - 0.9),
       f"(icp {get('icp', 'static'):.3f}, aci delayed {get('aci', 'delayed'):.3f})")
    # 11. module under it
    r_aci = A.selftest(verbose=False)
    ok("ce1_aci_selftests", all(r_aci.values()), f"({sum(r_aci.values())}/{len(r_aci)})")
    return res


# ------------------------------------------------------------------ stage
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="S5")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--only", default=None)
    ap.add_argument("--s2", default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if a.selftest:
        r = selftest()
        sys.exit(0 if all(r.values()) else 1)
    import config as C
    import ce1_folds as F
    t0 = time.time()
    s2 = Path(a.s2) if a.s2 else C.RESULTS / "ce1" / "S2"
    out = Path(a.out) if a.out else C.RESULTS / "ce1" / (a.stage + ("_smoke" if a.smoke else ""))
    runs = C.RESULTS / "ce1" / "_runs" if not a.out else out / "_runs"
    out.mkdir(parents=True, exist_ok=True); runs.mkdir(parents=True, exist_ok=True)
    checks = {}

    def check(name, cond, msg=""):
        checks[name] = dict(ok=bool(cond), msg=msg)
        log(f"{'PASS' if cond else 'FAIL'}  {name}  {msg}")

    st = selftest(verbose=False)
    check("selftests", all(st.values()), f"({sum(st.values())}/{len(st)}"
          + ("" if all(st.values()) else "; failed " + ", ".join(k for k, v in st.items() if not v)) + ")")
    jobs = [(s, y, h) for s in FOLDS for y in FOLDS[s] for h in HORIZONS]
    if a.smoke:
        jobs = [("yulara", 2024, 3)]
    if a.only:
        jobs = [(p.split(":")[0], int(p.split(":")[1]), int(p.split(":")[2])) for p in a.only.split(",")]
    tabs = {k: [] for k in ("intervals", "crps", "reliability", "calib_size", "bound_cap")}
    jobrows = []
    inputs = {}
    for site, yr, h in jobs:
        stem = f"{site}_f{yr}_h{h}"
        src = s2 / f"{stem}.parquet"
        inputs[src.name] = md5(src)
        pj, pn = out / f"{stem}.json", out / f"{stem}_obs.npz"
        if pj.exists() and pn.exists():
            rec = json.load(open(pj))
            if rec.get("settings") == SETTINGS and rec.get("input_md5") == inputs[src.name]:
                log(f"{stem}: done earlier, skipped")
                for k in tabs:
                    tabs[k] += rec.get(k, [])
                jobrows.append(rec["info"] | dict(site=site, fold=yr, h=h))
                continue
        t1 = time.time()
        d = F.load_cache(src)
        if a.smoke:
            keep = d.index.normalize().isin(pd.Index(d.index.normalize().unique()[::4]))
            d = d[keep]
        res, O = run_job(d, site, yr, h, obs=True, calib_size=(yr == 2024))
        res["info"]["seconds"] = round(time.time() - t1, 1)
        json.dump(dict(res, settings=SETTINGS, input_md5=inputs[src.name]), open(pj, "w"), default=str)
        np.savez_compressed(pn, **O)
        for k in tabs:
            tabs[k] += res[k]
        jobrows.append(res["info"] | dict(site=site, fold=yr, h=h))
        log(f"{stem}: test {res['info']['n_test']} rows, {res['info']['seconds']:.0f}s")
    for k, rows in tabs.items():
        pd.DataFrame(rows).to_csv(out / f"{k}.csv", index=False)
    J = pd.DataFrame([{k: v for k, v in r.items() if not isinstance(v, dict)} for r in jobrows])
    J.to_csv(out / "jobs.csv", index=False)
    I = pd.DataFrame(tabs["intervals"])
    done = {(r["site"], r["fold"], r["h"]) for r in jobrows}
    check("every_job_done", all(j in done for j in jobs), f"({len(done)}/{len(jobs)})")
    check("anticipative_equals_event_rule_h1_every_job",
          all(r["anticipative_equals_event_rule_h1"] for r in jobrows))
    h1 = [r for r in jobrows if r["h"] == 1]
    check("delayed_equals_anticipative_at_5min", all(r["delayed_equals_anticipative_at_h1"] for r in h1),
          f"({len(h1)} jobs)")
    ins = [min(r["in_sample_coverage_90"].values()) for r in jobrows]
    check("in_sample_calibration_coverage_ge_nominal", min(ins) >= 0.90 - 1e-9, f"(min {min(ins):.5f})")
    check("records_finite", bool(len(I)) and np.isfinite(I[["PICP", "PINAW", "Winkler"]].values).all())
    st90 = I[(I.variant == "static") & (I.scope == "all")]
    w = st90.pivot_table(index=["site", "fold", "horizon_min", "method"], columns="nominal", values="width_Wm2")
    check("static_widths_nested_80_90_95", bool(((w[0.8] <= w[0.9]) & (w[0.9] <= w[0.95])).all()))
    cvr = I[(I.variant == "static") & (I.scope == "all") & (I.nominal == 0.9)
            & (I.method.isin(["icp", "mondrian", "cqr", "mondrian_cqr"]))]
    check("static_marginal_coverage_plausible", bool(len(cvr)) and cvr.PICP.between(0.80, 0.97).all(),
          f"(range {cvr.PICP.min():.3f}-{cvr.PICP.max():.3f}; a diagnostic bound, not a result)")
    ok = all(c["ok"] for c in checks.values())
    detail = dict(stage=a.stage, smoke=a.smoke, ok=ok, seconds=round(time.time() - t0, 1), checks=checks,
                  jobs=len(jobs), settings=SETTINGS,
                  outputs={p.name: md5(p) for p in sorted(out.glob("*")) if p.is_file()},
                  inputs=inputs,
                  code={p: md5(HERE / p) for p in ("ce1_conformal.py", "ce1_aci.py", "ce1_folds.py",
                                                   "../conformal/conformal.py", "../conformal/conformal_adaptive.py",
                                                   "../r1/r1_conformal_delayed.py")},
                  versions=dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__),
                  host=platform.node())
    json.dump(detail, open(runs / f"{a.stage}{'_smoke' if a.smoke else ''}.detail.json", "w"), indent=1)
    log(f"{'ALL CHECKS PASSED' if ok else 'SOME CHECKS FAILED'} in {time.time() - t0:.0f}s")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
