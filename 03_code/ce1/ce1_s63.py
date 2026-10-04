"""Stage S63: forecasting diagnostics registered in D-122 (order statistics, lower-tail tests,
complete-day coverage and related items), Yulara, six folds, from the cached forecasts.

Usage (from the repository root unless the path starts at ce1/, then from 03_code):
    python3 03_code/ce1/ce1_s63.py --root .
    python3 03_code/ce1/ce1_s63.py --selftest
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "utils"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402
import ce1_dispatch as D  # noqa: E402

FOLDS = [2019, 2020, 2021, 2022, 2023, 2024]
H = 3
SHAPES = ("main", "shapeB")
CP = ["icp", "mondrian", "cqr", "mondrian_cqr"]
THP = CP + ["raw_q", "tod_q", "nicp_var"]
REGS = ["clear", "transitional", "cloudy"]  # S5 codes 0, 1, 2 (ce1_stats.REGIMES)
TGT = 2e-5
EPS = 1e-15
MONDRIAN_MIN = 30  # ce1_dispatch: cell used if > 30 rows
TAIL_THETAS = [0.90, 0.95, 0.975, 0.99, 0.995, 0.998, 0.999]
KUPIEC_CRIT = float(stats.chi2.ppf(0.95, 1))
PROXY_TAUS = [0.025, 0.05, 0.10, 0.5, 0.90, 0.95, 0.975]
PERSIST_STEPS = 12
MONO_TOL = 1e-9
MONO_LIMIT = 0.001
B, SEED, BLOCK = 10000, 42, 7
COV = 0.9
S5_STATIC = ["icp", "icp_norm", "mondrian", "cqr", "mondrian_cqr", "raw_q", "tod", "nicp_var"]


def md5(p: Path) -> str:
    h = hashlib.md5()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def log(m):
    print(time.strftime("%H:%M:%S"), f"[S63] {m}", flush=True)


# ================================================================ pure functions
def os_rank(n: int, theta: float) -> int:
    """Rank of the conformal order statistic, ceil((n + 1) theta), guarded against float noise."""
    return int(math.ceil(round((n + 1) * float(theta), 9)))


def order_stat(s: np.ndarray, theta: float):
    """ceil((n+1) theta)-th smallest score, or None when beyond support."""
    s = np.asarray(s, float)
    n = len(s)
    k = os_rank(n, theta)
    if k > n or n == 0:
        return None
    return float(np.partition(s, k - 1)[k - 1])


def os_bound(policy, theta, fit, ev):
    """One-sided lower GHI bound with the conformal order statistic in place of np.quantile.
    Returns (bound, beyond) where beyond marks steps whose cell is beyond support (bound 0 there:
    no PV counted, the handbook level). Scores and Mondrian fallback as ce1_dispatch.counted_ghi.
    """
    nev = len(ev["p"])
    if policy in ("icp", "mondrian"):
        s_all = fit["p"] - fit["y"]  # score of the lower bound, -(y - p)
        base = ev["p"]
    elif policy in ("cqr", "mondrian_cqr"):
        tk = D.tau_key(theta)
        s_all = fit["q"][tk] - fit["y"]
        base = ev["q"][tk]
    else:
        raise ValueError(policy)
    out = np.empty(nev)
    beyond = np.zeros(nev, bool)
    if policy in ("icp", "cqr"):
        k = order_stat(s_all, theta)
        if k is None:
            return np.zeros(nev), np.ones(nev, bool)
        return np.clip(base - k, 0, None), beyond
    for g in np.unique(ev["g"]):
        sg = s_all[fit["g"] == g]
        use = sg if len(sg) > MONDRIAN_MIN else s_all
        k = order_stat(use, theta)
        m = ev["g"] == g
        if k is None:
            out[m] = 0.0
            beyond[m] = True
        else:
            out[m] = np.clip(base[m] - k, 0, None)
    return out, beyond


def kupiec(x: int, n: int, p: float):
    """Kupiec POF likelihood ratio and p-value for x exceedances in n at expected rate p."""
    if n == 0:
        return np.nan, np.nan
    ph = x / n

    def ll(q):
        a = (n - x) * math.log1p(-q) if x < n else 0.0
        b = x * math.log(q) if x > 0 else 0.0
        return a + b
    lr = -2.0 * (ll(p) - (ll(ph) if 0 < ph < 1 else 0.0))
    lr = max(lr, 0.0)
    return lr, float(stats.chi2.sf(lr, 1))


def pinball(y, b, tau):
    u = np.asarray(y, float) - np.asarray(b, float)
    return float(np.mean(np.maximum(tau * u, (tau - 1) * u)))


def crps_obs(y, qd):
    """S5 CRPS proxy (seven-level WIS form), as ce1_wp58.crps_obs."""
    taus = sorted(qd)
    return 2.0 * np.mean(np.column_stack([np.maximum(t * (y - qd[t]), (t - 1) * (y - qd[t]))
                                          for t in taus]), axis=1)


def block_index(nd: int, b: int, L: int, seed=SEED) -> np.ndarray:
    """(b, nd) day indices. L = 1: ce1_stats.DayBoot (iid day draws). L > 1: circular blocks
    of L consecutive days, the ce1_wp55.block_counts construction.
    """
    rng = np.random.default_rng(seed)
    if L == 1:
        return rng.integers(0, nd, size=(b, nd))
    nb = int(np.ceil(nd / L))
    st = rng.integers(0, nd, size=(b, nb))
    return ((st[:, :, None] + np.arange(L)[None, None, :]) % nd).reshape(b, -1)[:, :nd]


def boot_mean_ci(v, mask, code, nd, idx, alpha=0.05):
    """Ratio-of-sums interval over resampled days (ce1_stats.DayBoot.mean_ci with a given index)."""
    dc = np.asarray(code)[mask]
    S = np.bincount(dc, weights=np.asarray(v, float)[mask], minlength=nd)
    C = np.bincount(dc, minlength=nd).astype(float)
    num = S[idx].sum(axis=1)
    den = C[idx].sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        bb = num / den
    bb = bb[np.isfinite(bb)]
    pt = float(S.sum() / C.sum()) if C.sum() > 0 else np.nan
    return pt, float(np.percentile(bb, 100 * alpha / 2)), float(np.percentile(bb, 100 * (1 - alpha / 2)))


def status(lo, hi, nominal=COV):
    return "below" if hi < nominal else ("above" if lo > nominal else "includes")


def complete_days_fc(idx, is_day: pd.Series, h: int, min_share=0.95) -> set:
    """Days whose scored issue times cover at least min_share of the issue times at which a
    forecast can exist by construction: t and its 13-step input window t-12..t in daytime (night
    is never an input, ce1_causal.make_xy) and the target t+h in daytime. (ce1_dispatch.
    complete_days counts every daytime t with a daytime target, including the first hour of the
    day that no forecast can cover, so no day reaches 0.95 there; D-123.)
    """
    s = is_day.astype(bool)
    poss = (s.astype(int).rolling(13, min_periods=13).min() == 1) & s.shift(-h, fill_value=False)
    pdays = poss.groupby(s.index.normalize()).sum()
    ts = pd.DatetimeIndex(idx)
    sc = pd.Series(1, index=ts).groupby(ts.normalize()).sum()
    share = (sc / pdays.reindex(sc.index)).fillna(0)
    return set(share.index[share >= min_share].values.astype("datetime64[D]"))


def rearrange(C: np.ndarray) -> np.ndarray:
    """Counted PV of one policy (levels in ascending theta on axis 0) made non-increasing in theta
    at every step by sorting (NaN, no forecast row, stays NaN in every level).
    """
    nan = np.isnan(C)
    S = -np.sort(-np.where(nan, -np.inf, C), axis=0)
    S[nan] = np.nan
    return S


def mono_share(C: np.ndarray, tol=MONO_TOL):
    """Share of steps (columns with a forecast) where counted PV rises between adjacent levels."""
    ok = ~np.isnan(C).any(axis=0)
    d = np.diff(C[:, ok], axis=0)
    viol = (d > tol).any(axis=0)
    return float(viol.mean()) if ok.any() else np.nan, int(viol.sum()), int(ok.sum())


def mono_size(C: np.ndarray, tol=MONO_TOL) -> dict:
    """Descriptive size of the rises (not part of the registered rule): share of adjacent level
    pairs that rise, mean rising pairs per step, median / 99th percentile / largest rise, and
    the largest gap between the counted value and its rearranged (sorted) value.
    """
    ok = ~np.isnan(C).any(axis=0)
    X = C[:, ok]
    d = np.diff(X, axis=0)
    up = d > tol
    r = d[up]
    gap = np.abs(rearrange(X) - X)
    return dict(pair_share=float(up.mean()) if up.size else np.nan,
                pairs_per_step=float(up.sum(axis=0).mean()) if up.size else np.nan,
                rise_median=float(np.median(r)) if r.size else 0.0,
                rise_p99=float(np.percentile(r, 99)) if r.size else 0.0,
                rise_max=float(r.max()) if r.size else 0.0,
                sorted_gap_p99=float(np.percentile(gap, 99)) if gap.size else 0.0,
                sorted_gap_max=float(gap.max()) if gap.size else 0.0)


def persistence_members(kt_in: pd.Series, issue_idx, steps=PERSIST_STEPS) -> np.ndarray:
    """(n, steps) issue-time clear-sky index at t, t-5 min, ..., t-(steps-1)*5 min."""
    pos = kt_in.index.get_indexer(pd.DatetimeIndex(issue_idx))
    assert (pos >= 0).all()
    v = kt_in.values
    M = np.full((len(pos), steps), np.nan)
    for j in range(steps):
        ok = pos - j >= 0
        M[ok, j] = v[pos[ok] - j]
    return M


def ens_quantiles(M: np.ndarray, gcs: np.ndarray):
    """Quantiles at PROXY_TAUS of the member rows (NaN dropped), times the clear-sky GHI."""
    Q = np.nanquantile(M, PROXY_TAUS, axis=1).T  # (n, 7); median slot = ensemble median
    Q = Q * np.asarray(gcs, float)[:, None]
    return {t: Q[:, i] for i, t in enumerate(PROXY_TAUS)}


# ================================================================ self-tests
def selftest(verbose=True):
    res = {}

    def ok(name, cond, msg=""):
        res[name] = bool(cond)
        if verbose:
            print(f"  [{'PASS' if cond else 'FAIL'}] {name} {msg}")

    ok("os_rank_basic", os_rank(99, 0.9) == 90 and os_rank(9, 0.9) == 9 and os_rank(9, 0.95) == 10)
    s = np.arange(1.0, 10.0)  # n = 9
    ok("order_stat_kth", order_stat(s, 0.9) == 9.0 and order_stat(s, 0.5) == 5.0)
    ok("order_stat_beyond_support", order_stat(s, 0.95) is None)
    # the order statistic is never below the interpolated quantile's lower bracket and equals
    # the 'higher' quantile method at rank ceil((n+1) theta) when that rank is inside the sample
    rng = np.random.default_rng(1)
    x = rng.normal(size=1001)
    k = os_rank(1001, 0.99)
    ok("order_stat_equals_sorted", order_stat(x, 0.99) == np.sort(x)[k - 1])
    # os_bound: marginal ICP equals p - s_(k); Mondrian falls back to the marginal set at <= 30
    n = 400
    fit = dict(y=rng.uniform(100, 900, n), p=rng.uniform(100, 900, n), g=np.array(["a"] * 380 + ["b"] * 20))
    ev = dict(y=rng.uniform(100, 900, 50), p=rng.uniform(100, 900, 50), g=np.array(["a"] * 25 + ["b"] * 25))
    b, bey = os_bound("icp", 0.9, fit, ev)
    kk = np.sort(fit["p"] - fit["y"])[os_rank(n, 0.9) - 1]
    ok("os_bound_icp", np.allclose(b, np.clip(ev["p"] - kk, 0, None)) and not bey.any())
    bm, beym = os_bound("mondrian", 0.9, fit, ev)
    ka = np.sort((fit["p"] - fit["y"])[:380])[os_rank(380, 0.9) - 1]
    ok("os_bound_mondrian_cells_and_fallback", np.allclose(bm[:25], np.clip(ev["p"][:25] - ka, 0, None))
       and np.allclose(bm[25:], np.clip(ev["p"][25:] - kk, 0, None)))
    fit2 = dict(fit, g=np.array(["a"] * 360 + ["b"] * 40))
    bm2, bey2 = os_bound("mondrian", 0.99, fit2, ev)  # cell b: n = 40, rank ceil(41 * 0.99) = 41 > 40
    ok("os_bound_mondrian_beyond_support_is_handbook", bey2[25:].all() and (bm2[25:] == 0).all() and not bey2[:25].any())
    # interpolated and order-statistic bounds are close on a large sample
    fitL = dict(y=rng.uniform(100, 900, 20000), p=rng.uniform(100, 900, 20000), g=np.array(["a"] * 20000))
    bi = D.counted_ghi("icp", 0.95, fitL, ev)
    bo, _ = os_bound("icp", 0.95, fitL, ev)
    ok("os_close_to_interpolated", np.max(np.abs(bi - bo)) < 2.0, f"(max {np.max(np.abs(bi - bo)):.3f} W/m2)")
    # Kupiec: LR 0 at x = n p; known value; p-value monotone
    lr0, p0 = kupiec(10, 1000, 0.01)
    lr1, p1 = kupiec(20, 1000, 0.01)
    ref = -2 * (980 * math.log(0.99) + 20 * math.log(0.01) - 980 * math.log(0.98) - 20 * math.log(0.02))
    ok("kupiec_values", abs(lr0) < 1e-9 and abs(lr1 - ref) < 1e-9 and p1 < 0.05 < p0, f"(LR {lr1:.3f})")
    lrz, pz = kupiec(0, 1000, 0.01)
    ok("kupiec_zero_exceedances", abs(lrz - (-2 * 1000 * math.log(0.99))) < 1e-9)
    ok("pinball_simple", abs(pinball([1.0], [0.0], 0.1) - 0.1) < 1e-12 and abs(pinball([0.0], [1.0], 0.1) - 0.9) < 1e-12)
    # block index: L = 1 is DayBoot's draw; L = 7 rows are consecutive runs mod nd
    rng2 = np.random.default_rng(SEED)
    ok("block1_is_dayboot", np.array_equal(block_index(30, 20, 1), rng2.integers(0, 30, size=(20, 30))))
    bi7 = block_index(30, 20, 7)
    run_ok = all(((bi7[r, 7 * j + 1:7 * j + 7] - bi7[r, 7 * j:7 * j + 6]) % 30 == 1).all() for r in range(20) for j in range(4))
    ok("block7_consecutive_runs", bi7.shape == (20, 30) and run_ok)
    # rearrangement: non-increasing, NaN kept, a monotone column unchanged
    C = np.array([[5.0, 1.0, np.nan], [6.0, 2.0, np.nan], [4.0, 0.5, np.nan]])
    Rr = rearrange(C)
    ok("rearrange_sorted_nan_kept", np.array_equal(Rr[:, 0], [6.0, 5.0, 4.0]) and np.isnan(Rr[:, 2]).all()
       and np.array_equal(Rr[:, 1], [2.0, 1.0, 0.5]))
    sh, nv, nt = mono_share(C)
    ok("mono_share_counts", nv == 2 and nt == 2 and sh == 1.0)
    ok("mono_share_monotone_zero", mono_share(Rr)[1] == 0)
    ms = mono_size(C)
    ok("mono_size_values", abs(ms["rise_max"] - 1.0) < 1e-12 and abs(ms["pair_share"] - 0.5) < 1e-12
       and abs(ms["sorted_gap_max"] - 1.0) < 1e-12, str(ms))
    # persistence members and quantiles: a constant index gives a point forecast = kt x gcs
    ix = pd.date_range("2024-01-01 06:00", periods=40, freq="5min")
    kt = pd.Series(0.8, index=ix)
    kt.iloc[5] = np.nan
    M = persistence_members(kt, ix[[20, 3, 10]])
    ok("persistence_members_window", M.shape == (3, 12) and (~np.isnan(M[0])).all() and np.isnan(M[1, 4:]).all()
       and (~np.isnan(M[1, :4])).all() and np.isnan(M[2, 5]) and np.isnan(M[2, 11]) and int(np.isnan(M[2]).sum()) == 2)
    M = M[:2]
    Q = ens_quantiles(M, np.array([1000.0, 500.0]))
    ok("persistence_constant_is_point", all(np.allclose(Q[t], [800.0, 400.0]) for t in PROXY_TAUS))
    y = np.array([700.0, 450.0])
    # complete days: a day with every possible issue time scored qualifies, one with a gap does not
    gi = pd.date_range("2024-01-01", "2024-01-03 23:55", freq="5min")
    dayf = pd.Series((gi.hour >= 6) & (gi.hour < 18), index=gi)
    poss = gi[(dayf.astype(int).rolling(13, min_periods=13).min() == 1).values & dayf.shift(-3, fill_value=False).values]
    sc_idx = poss[~((poss >= "2024-01-02 09:00") & (poss < "2024-01-02 11:00"))]
    cdset = complete_days_fc(sc_idx, dayf, 3)
    ok("complete_days_fc", cdset == {np.datetime64("2024-01-01"), np.datetime64("2024-01-03")}, str(sorted(cdset)))
    ok("crps_point_equals_abs_error", np.allclose(crps_obs(y, {t: np.array([800.0, 400.0]) for t in PROXY_TAUS}), [100.0, 50.0]))
    return res


# ================================================================ helpers on the record
def as_dict(c: pd.DataFrame) -> dict:
    return dict(idx=c.index.values, y=c["y"].values.astype(float), p=c["p"].values.astype(float),
                g=c["g"].values, v=c["v"].values.astype(float), gcs=c["gcs"].values.astype(float),
                q={round(float(c_[1:]), 6): c[c_].values.astype(float) for c_ in c.columns if c_.startswith("q")})


def bound_any(pol, theta, fit, ev):
    """Interpolated bound of the dispatch (limit level 1: no PV)."""
    if theta >= 1.0:
        return np.zeros(len(ev["y"]))
    return D.counted_ghi(pol, theta, fit, ev)


# ================================================================ analysis
def analyse(root: Path, skip_rearranged: bool = False) -> bool:
    import ce1_folds as FC
    import ce1_causal as K
    import ce1_load as LM
    import ce1_s6 as S6
    R = root / "04_results" / "ce1"
    out = R / "S63"
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    started = time.strftime("%Y-%m-%dT%H:%M:%S")
    checks, inputs = {}, {}

    def check(name, cond, msg=""):
        checks[name] = dict(ok=bool(cond), msg=str(msg))
        log(f"{'PASS' if cond else 'FAIL'}  {name}  {msg}")

    def inp(rel):
        inputs[rel] = md5(R / rel)
        return R / rel

    for k, v in selftest(verbose=False).items():
        check(f"selftest_{k}", v)
    D.extend_theta_low()
    grid_theta = [float(t) for t in D.param_grid("icp")]
    geo = pd.read_parquet(inp("S1/yulara_geo_5min.parquet"))
    scal = json.load(open(inp("S1/scalars.json")))["yulara"]
    inp("S1M/available_pv_map.npz"); inp("S1L/load_profile.npz"); inp("S1L/load_scales.json")
    geo_, m, unit96, scales, _ = S6.load_inputs(R / "S1", R / "S1M", R / "S1L")
    is_day = geo["is_day"].astype(bool)
    wp57 = pd.read_csv(inp("WP57/cases.csv"))
    ev61 = pd.read_csv(inp("WP61/events.csv"))

    caches, sel = {}, {}
    for yr in FOLDS:
        caches[(yr, 3)] = FC.load_cache_ext(inp(f"S2/yulara_f{yr}_h3.parquet"), inp(f"S2X/yulara_f{yr}_h3.parquet"))
        caches[(yr, 1)] = FC.load_cache(inp(f"S2/yulara_f{yr}_h1.parquet"))
        sel[yr] = json.load(open(inp(f"S6E/f{yr}_h3_select.json")))["selections"]

    rows_os, rows_os_cells, rows_sup, rows_pexc, rows_tail, rows_pers = [], [], [], [], [], []
    rows_mono, rows_mono_ar, rows_cdd, rows_ev = [], [], [], []
    for yr in FOLDS:
        t1 = time.time()
        c = caches[(yr, 3)]
        ca, te = c[c.split == "calib"], c[c.split == "test"]
        fit, ev = as_dict(ca), as_dict(te)
        H1, H2 = D.split_halves(fit)
        tgt = pd.DatetimeIndex(ev["idx"]) + pd.Timedelta(minutes=5 * H)
        grid = S6.year_grid(f"{yr}-01-01", f"{yr}-12-31 23:55")
        pav = LM.available_pv_on_grid(grid, geo["ghi"].reindex(grid).values, m)
        pv_of = S6.PVAt(tgt, m)
        day = LM.day_ids(grid)
        tot_e = pd.read_parquet(inp(f"S6E/f{yr}_h3_totals.parquet"))
        dz = np.load(inp(f"S6E/f{yr}_h3_daily.npz"))

        # ---------------- item 1: order statistics at the A-R level, rerun of the A-R test dispatch
        for sname in SHAPES:
            sc = S6.SCENARIOS[sname]
            L = S6.load_for(grid, unit96, scales, sc)
            kw = dict(ratings=LM.fleet_for_peak(L.max()), min_load=sc["min_load"], fuel=sc["fuel"], battery=sc["battery"])
            Crows, meta = [np.zeros(len(ev["y"]))], [("handbook", np.nan, "anchor", 0)]
            for pol in CP:
                th = float(sel[yr][sname]["policies"][pol]["A_R"]["param"])
                bi = bound_any(pol, th, fit, ev)
                if th >= 1.0:
                    bo, bey = np.zeros(len(ev["y"])), np.zeros(len(ev["y"]), bool)
                else:
                    bo, bey = os_bound(pol, th, fit, ev)
                Crows += [pv_of(bi), np.where(bey, 0.0, pv_of(bo))]
                meta += [(pol, th, "interpolated", 0), (pol, th, "order_stat", int(bey.sum()))]
                d_ghi = bo - bi
                rows_os_cells.append(dict(fold=yr, scenario=sname, policy=pol, theta=th, n_calib=len(fit["y"]),
                                          beyond_support_steps=int(bey.sum()), d_bound_mean_Wm2=float(np.mean(d_ghi)),
                                          d_bound_min_Wm2=float(np.min(d_ghi)), d_bound_max_Wm2=float(np.max(d_ghi)),
                                          share_steps_changed=float(np.mean(np.abs(d_ghi) > 1e-9))))
            Cg = S6.place(grid, tgt, np.vstack(Crows))
            r = LM.simulate(L, pav, Cg, day, H, **kw)["total"]
            ref = tot_e[(tot_e.scenario == sname) & (tot_e.set == "test")]
            hb_ref = ref[ref.policy == "handbook"].iloc[0]
            worst = abs(r["fuel_L"][0] / hb_ref.fuel_L - 1)
            for j, (pol, th, kind, nb) in enumerate(meta):
                if kind == "interpolated":
                    rr = ref[(ref.policy == pol) & np.isclose(ref.param.astype(float), th)].iloc[0]
                    worst = max(worst, abs(r["fuel_L"][j] / rr.fuel_L - 1), abs(r["ue_kWh"][j] - rr.ue_kWh) / max(rr.ue_kWh, 1.0))
                if kind != "anchor":
                    rows_os.append(dict(fold=yr, scenario=sname, policy=pol, theta=th, bound=kind,
                                        beyond_support_steps=nb, fuel_L=float(r["fuel_L"][j]),
                                        ue_share=float(r["ue_share"][j]), ue_kWh=float(r["ue_kWh"][j]),
                                        starts=float(r["starts"][j]), fuel_handbook=float(r["fuel_L"][0]),
                                        meets=bool(r["ue_share"][j] <= TGT + EPS),
                                        saving_pct=100 * (1 - float(r["fuel_L"][j]) / float(r["fuel_L"][0]))))
            check(f"interpolated_reproduces_S6E_AR_f{yr}_{sname}", worst < 1e-9, f"(worst relative {worst:.1e})")

        # ---------------- item 2: support per regime (15 min), calibration year and its H1 half
        for part, F in (("calib", fit), ("H1", H1)):
            days = pd.DatetimeIndex(F["idx"]).normalize()
            for reg in REGS + ["all"]:
                msk = np.ones(len(F["y"]), bool) if reg == "all" else (F["g"] == reg)
                n = int(msk.sum())
                rows_sup.append(dict(fold=yr, part=part, regime=reg, issue_times=n, days=int(days[msk].nunique()),
                                     theta_max_one_expected=(1 - 1 / n) if n else np.nan,
                                     theta_max_order_stat=(n / (n + 1)) if n else np.nan))

        # ---------------- items 3 and 4: pair exceedance at the A-R levels, lower-tail reliability
        g_it, g_tt = ev["g"], te["g_t"].values
        y = ev["y"]
        for pol in THP:
            for sname in SHAPES:
                th = float(sel[yr][sname]["policies"][pol]["A_R"]["param"])
                e = y < bound_any(pol, th, fit, ev)
                for a in REGS:
                    for b in REGS:
                        mm = (g_it == a) & (g_tt == b)
                        rows_pexc.append(dict(fold=yr, h=H, policy=pol, level=f"A_R_{sname}", theta=th, issue=a, target=b,
                                              n=int(mm.sum()), exceed=float(e[mm].mean()) if mm.any() else np.nan,
                                              expected=1 - th))
            for th in TAIL_THETAS:
                b_ = D.counted_ghi(pol, th, fit, ev)
                e = y < b_
                for reg in REGS + ["all"]:
                    mm = np.ones(len(y), bool) if reg == "all" else (g_it == reg)
                    n, x = int(mm.sum()), int(e[mm].sum())
                    lr, pv = kupiec(x, n, 1 - th)
                    rows_tail.append(dict(fold=yr, h=H, policy=pol, theta=th, regime=reg, n=n, exceed_n=x,
                                          exceed=x / n if n else np.nan, expected=1 - th, expected_n=n * (1 - th),
                                          pinball=pinball(y[mm], b_[mm], 1 - th), kupiec_lr=lr, kupiec_p=pv,
                                          reject_5pct=bool(lr > KUPIEC_CRIT)))

        # ---------------- item 5: short-window persistence ensemble
        sc_ = float(scal[str(yr)]["scalar"])
        base = K.build_base(geo, sc_, "locf6")
        reg_b = base["regime"].reindex(pd.DatetimeIndex(ev["idx"])).values
        check(f"build_base_regime_equals_cache_g_f{yr}", np.array_equal(reg_b.astype(str), g_it.astype(str)),
              f"({int((reg_b.astype(str) != g_it.astype(str)).sum())} differ)")
        gcs_t = geo["ghi_cs_raw"].reindex(tgt).values * sc_
        check(f"gcs_equals_raw_x_scalar_f{yr}", np.allclose(ev["gcs"], gcs_t, rtol=1e-9))
        M = persistence_members(base["kt_in"], ev["idx"])
        nmem = (~np.isnan(M)).sum(axis=1)
        check(f"persistence_members_present_f{yr}", (nmem >= 1).all() and (~np.isnan(M[:, 0])).all(),
              f"(min {int(nmem.min())}, median {float(np.median(nmem))}, rows with 12: {float(np.mean(nmem == 12)):.4f})")
        Qp = ens_quantiles(M, ev["gcs"])
        cr_ref = crps_obs(y, Qp)
        z5 = np.load(inp(f"S5/yulara_f{yr}_h3_obs.npz"))
        ts5 = pd.DatetimeIndex(pd.to_datetime(z5["ts_ns"]))
        check(f"s5_rows_equal_cache_f{yr}", len(ts5) == len(te) and (ts5 == te.index).all()
              and np.allclose(te["y"].values.astype(np.float32), z5["y"]))
        codes = {r_: i for i, r_ in enumerate(REGS)}
        check(f"s5_regime_codes_f{yr}", all((z5["reg_it"][g_it == r_] == codes[r_]).all() for r_ in REGS)
              and all((z5["reg_tt"][g_tt == r_] == codes[r_]).all() for r_ in REGS))
        crk = [k for k in z5.files if k.startswith("crps__")]
        for reg in ["all"] + REGS:
            mm = np.ones(len(y), bool) if reg == "all" else (g_it == reg)
            cref = float(cr_ref[mm].mean())
            for k in crk:
                v = float(z5[k][mm].astype(float).mean())
                rows_pers.append(dict(fold=yr, h=H, scope=reg, n=int(mm.sum()), method=k[len("crps__"):],
                                      CRPS=v, CRPS_persistence=cref, skill=1 - v / cref))

        # ---------------- item 6: monotonicity in theta (test year and H2), counted PV
        tgtH2 = pd.DatetimeIndex(H2["idx"]) + pd.Timedelta(minutes=5 * H)
        pv_H2 = S6.PVAt(tgtH2, m)
        levels = [t for t in grid_theta if t < 1.0]
        for pol in THP:
            gT = np.vstack([D.counted_ghi(pol, t, fit, ev) for t in levels])
            CT = np.vstack([pv_of(g_) for g_ in gT])
            CH = np.vstack([pv_H2(D.counted_ghi(pol, t, H1, H2)) for t in levels])
            shT, nvT, nT = mono_share(CT)
            shH, nvH, nH = mono_share(CH)
            shG, nvG, _ = mono_share(gT)
            rows_mono.append(dict(fold=yr, h=H, policy=pol, n_levels=len(levels), test_steps=nT, test_viol=nvT, test_share=shT,
                                  H2_steps=nH, H2_viol=nvH, H2_share=shH, test_share_ghi=shG,
                                  **{f"pv_kW_{k}": v for k, v in mono_size(CT).items()},
                                  **{f"ghi_Wm2_{k}": v for k, v in mono_size(gT).items()}))

        # ---------------- item 7 (dispatch part): complete days, A-R UE on complete days
        cdays = complete_days_fc(ev["idx"], is_day, H, 0.95)
        gd = pd.DatetimeIndex(grid).normalize().unique().values.astype("datetime64[D]")
        keep = np.array([d_ in cdays for d_ in gd])
        pol_r = np.asarray(dz["rows_policy"]).astype(str)
        par_r = np.asarray(dz["rows_param"]).astype(float)
        rix = {(p_, round(float(x_), 6)): i for i, (p_, x_) in enumerate(zip(pol_r, par_r))}
        hb_i = int(np.where(pol_r == "handbook")[0][0])
        for sname in SHAPES:
            ld = dz[f"{sname}__load_kWh"][hb_i].astype(float)
            fh = dz[f"{sname}__fuel_L"][hb_i].astype(float)
            for pol in THP + ["frac_pv"]:
                th = float(sel[yr][sname]["policies"][pol]["A_R"]["param"])
                i = rix[(pol, round(th, 6))]
                u = dz[f"{sname}__ue_kWh"][i].astype(float)
                f = dz[f"{sname}__fuel_L"][i].astype(float)
                ld_i = dz[f"{sname}__load_kWh"][i].astype(float)
                rows_cdd.append(dict(fold=yr, scenario=sname, policy=pol, theta=th, days=len(gd), complete_days=int(keep.sum()),
                                     ue_share_all=float(u.sum() / ld_i.sum()), ue_share_complete=float(u[keep].sum() / ld_i[keep].sum()),
                                     meets_all=bool(u.sum() / ld_i.sum() <= TGT + EPS),
                                     meets_complete=bool(u[keep].sum() / ld_i[keep].sum() <= TGT + EPS),
                                     saving_all_pct=100 * (1 - f.sum() / fh.sum()),
                                     saving_complete_pct=100 * (1 - f[keep].sum() / fh[keep].sum())))

        # ---------------- item 9: UE events by issue-time regime of the forecast at the first step
        gmap = pd.Series(g_it, index=pd.DatetimeIndex(ev["idx"]))
        e_ = ev61[ev61.fold == yr]
        iss = pd.DatetimeIndex(pd.to_datetime(e_.start)) - pd.Timedelta(minutes=5 * H)
        regs = gmap.reindex(iss).fillna("fallback").values
        for (proto, sname, pol), grp in e_.assign(reg=regs).groupby(["protocol", "scenario", "policy"]):
            for reg in REGS + ["fallback"]:
                rows_ev.append(dict(fold=yr, protocol=proto, scenario=sname, policy=pol, regime=reg,
                                    events=int((grp.reg == reg).sum()), n_events=len(grp)))
        log(f"fold {yr}: {time.time() - t1:.0f} s")

    pd.DataFrame(rows_os).to_csv(out / "order_stats.csv", index=False)
    pd.DataFrame(rows_os_cells).to_csv(out / "order_stats_cells.csv", index=False)
    pd.DataFrame(rows_pexc).to_csv(out / "pair_exceedance.csv", index=False)
    tail = pd.DataFrame(rows_tail)
    tail.to_csv(out / "tail.csv", index=False)
    pers = pd.DataFrame(rows_pers)
    pers.to_csv(out / "persistence.csv", index=False)
    mono = pd.DataFrame(rows_mono)
    mono.to_csv(out / "monotonicity.csv", index=False)
    pd.DataFrame(rows_cdd).to_csv(out / "complete_days_dispatch.csv", index=False)
    evr = pd.DataFrame(rows_ev)
    evr.to_csv(out / "events_regime.csv", index=False)

    # ---------------- item 2 (flags): every selected level against the support limits
    sup = pd.DataFrame(rows_sup)
    flags = []
    s51 = {yr: np.load(inp(f"S51/f{yr}_h3_paths.npz")) for yr in FOLDS}
    for yr in FOLDS:
        pz = s51[yr]
        specs = list(pz["specs"].astype(str))
        for pol in THP:
            lv = []
            for sname in SHAPES:
                lv.append((f"A_R_{sname}", float(sel[yr][sname]["policies"][pol]["A_R"]["param"])))
                w = wp57[(wp57.fold == yr) & (wp57.scenario == sname) & (wp57.policy == pol)]
                for r_ in w.itertuples():
                    lv.append((f"{r_.selection}_{sname}", float(r_.param)))
                kk = f"{pol}|0.005|0.5"
                if kk in specs:
                    lv.append((f"A_D_max_{sname}", float(np.nanmax(pz[f"{sname}__level"][specs.index(kk)]))))
            regs_ = REGS if pol in ("mondrian", "mondrian_cqr") else ["all"]
            for name, th in lv:
                if th >= 1.0:
                    continue  # the limit level counts no PV
                for part in ("calib", "H1"):
                    for reg in regs_:
                        s_ = sup[(sup.fold == yr) & (sup.part == part) & (sup.regime == reg)].iloc[0]
                        flags.append(dict(fold=yr, policy=pol, level=name, theta=th, part=part, regime=reg,
                                          n=int(s_.issue_times), theta_max_one_expected=float(s_.theta_max_one_expected),
                                          above_one_expected=bool(th > s_.theta_max_one_expected + 1e-12),
                                          above_order_stat=bool(th > s_.theta_max_order_stat + 1e-12)))
    sup.to_csv(out / "support.csv", index=False)
    fl = pd.DataFrame(flags)
    fl.to_csv(out / "support_flags.csv", index=False)

    # ---------------- item 3 (coverage) and 7 (coverage) and 8 (7-day blocks) from S5 records
    rows_pc, rows_cdc, rows_b7 = [], [], []
    s7 = pd.read_csv(inp("S7/coverage_ci.csv"))
    s46 = pd.read_csv(inp("S46/coverage.csv"))
    s46b = s46[s46.definition == "base"]
    worst7, worst46, n7, n46 = 0.0, 0.0, 0, 0
    jobs = [("yulara", yr, h) for yr in FOLDS for h in (1, 3)] + [("yulara", 2024, 6), ("yulara", 2024, 12)] \
        + [("asp", 2024, h) for h in (1, 3, 6, 12)]
    for site, yr, h in jobs:
        z = np.load(inp(f"S5/{site}_f{yr}_h{h}_obs.npz"))
        code = z["day_code"].astype(int)
        nd = int(code.max()) + 1
        ts = pd.DatetimeIndex(pd.to_datetime(z["ts_ns"]))
        dcode = np.unique(ts.normalize().values, return_inverse=True)[1]
        check(f"day_code_is_sorted_days_{site}_f{yr}_h{h}", np.array_equal(dcode, code))
        rit, rtt = z["reg_it"], z["reg_tt"]
        n = len(code)
        scopes = [("all", np.ones(n, bool))] + [(r_, rit == i) for i, r_ in enumerate(REGS)] \
            + [(f"tt_{r_}", rtt == i) for i, r_ in enumerate(REGS)]
        idx1, idx7 = block_index(nd, B, 1), block_index(nd, B, BLOCK)
        covk = [k for k in z.files if k.startswith("cov__") and k.endswith("|static|0.9")]
        # complete days (Yulara only; the geo record is Yulara's)
        cd_mask = None
        if site == "yulara":
            cd = complete_days_fc(ts.values, is_day, h, 0.95)
            cd_mask = np.isin(ts.normalize().values.astype("datetime64[D]"), np.array(sorted(cd), dtype="datetime64[D]"))
        for k in covk:
            meth = k.split("__")[1].split("|")[0]
            hit = z[k].astype(float)
            # item 3: pairs (Yulara, 5 and 15 min)
            if site == "yulara" and h in (1, 3):
                for a, ra in enumerate(REGS):
                    for b_, rb in enumerate(REGS):
                        mm = (rit == a) & (rtt == b_)
                        rows_pc.append(dict(fold=yr, horizon_min=5 * h, method=meth, issue=ra, target=rb, n=int(mm.sum()),
                                            picp=float(hit[mm].mean()) if mm.any() else np.nan))
            for sc, mm in scopes:
                if mm.sum() < 30:
                    continue
                if cd_mask is not None:
                    m2 = mm & cd_mask
                    rows_cdc.append(dict(site=site, fold=yr, horizon_min=5 * h, method=meth, scope=sc, n=int(mm.sum()),
                                         n_complete=int(m2.sum()), days=nd, complete_days=int(np.unique(code[cd_mask]).size),
                                         picp=float(hit[mm].mean()), picp_complete=float(hit[m2].mean()) if m2.any() else np.nan))
                in7 = (site == "yulara" and yr == 2024) or site == "asp"
                in46 = site == "yulara" and h in (1, 3)
                if not (in7 or in46):
                    continue
                p1, lo1, hi1 = boot_mean_ci(hit, mm, code, nd, idx1)
                p7, lo7, hi7 = boot_mean_ci(hit, mm, code, nd, idx7)
                rows_b7.append(dict(site=site, fold=yr, horizon_min=5 * h, method=meth, scope=sc, n=int(mm.sum()), days=nd,
                                    picp=p1, ci1_lo=lo1, ci1_hi=hi1, ci7_lo=lo7, ci7_hi=hi7,
                                    status1=status(lo1, hi1), status7=status(lo7, hi7), s7_cited=in7, s46_cited=in46))
                if in7:
                    a7 = s7[(s7.site == site) & (s7.fold == yr) & (s7.horizon_min == 5 * h) & (s7.method == meth)
                            & (s7.variant == "static") & (s7.nominal == COV) & (s7.scope == sc)]
                    if len(a7) == 1:
                        n7 += 1
                        worst7 = max(worst7, abs(round(lo1, 4) - a7.PICP_ci_lo.iloc[0]), abs(round(hi1, 4) - a7.PICP_ci_hi.iloc[0]),
                                     abs(round(p1, 4) - a7.PICP.iloc[0]))
                if in46:
                    a46 = s46b[(s46b.fold == yr) & (s46b.horizon_min == 5 * h) & (s46b.method == meth) & (s46b.scope == sc)]
                    if len(a46) == 1 and np.isfinite(a46.ci_lo.iloc[0]):
                        n46 += 1
                        worst46 = max(worst46, abs(lo1 - a46.ci_lo.iloc[0]), abs(hi1 - a46.ci_hi.iloc[0]), abs(p1 - a46.picp.iloc[0]))
        log(f"S5 {site} f{yr} h{h} done ({time.time() - t0:.0f} s)")
    check("block1_reproduces_S7_intervals", n7 > 100 and worst7 < 1.01e-4, f"({n7} cells, worst {worst7:.1e} at 4 places)")
    check("block1_reproduces_S46_base_intervals", n46 > 100 and worst46 < 1e-9, f"({n46} cells, worst {worst46:.1e})")
    pd.DataFrame(rows_pc).to_csv(out / "pair_coverage.csv", index=False)
    cdc = pd.DataFrame(rows_cdc)
    cdc.to_csv(out / "complete_days_coverage.csv", index=False)
    b7 = pd.DataFrame(rows_b7)
    b7.to_csv(out / "blocks7.csv", index=False)

    # ---------------- item 10: row sets of the same calendar year in two fold caches
    rows_rs = []
    for h in (1, 3):
        for yr in FOLDS[:-1]:
            a = caches[(yr, h)]
            b = caches[(yr + 1, h)]
            ta, cb = a[a.split == "test"], b[b.split == "calib"]
            ia, ib = ta.index, cb.index
            sh = ia.intersection(ib)
            rows_rs.append(dict(year=yr, horizon_min=5 * h, test_rows_fold_Y=len(ia), calib_rows_fold_Y1=len(ib),
                                shared=len(sh), only_test=len(ia.difference(ib)), only_calib=len(ib.difference(ia)),
                                y_equal_share=float(np.mean(np.isclose(ta.loc[sh, "y"].values, cb.loc[sh, "y"].values, rtol=0, atol=1e-6))),
                                g_equal_share=float(np.mean(ta.loc[sh, "g"].values == cb.loc[sh, "g"].values)),
                                g_t_equal_share=float(np.mean(ta.loc[sh, "g_t"].values == cb.loc[sh, "g_t"].values)),
                                transitional_share_test=float(np.mean(ta.g.values == "transitional")),
                                transitional_share_calib=float(np.mean(cb.g.values == "transitional"))))
    rs = pd.DataFrame(rows_rs)
    rs.to_csv(out / "rowsets.csv", index=False)

    # ---------------- item 6 (continued): rearranged variant for policies above 0.1%
    over = mono.groupby("policy")[["test_share", "H2_share"]].max()
    viol_pols = [p_ for p_ in THP if max(over.loc[p_, "test_share"], over.loc[p_, "H2_share"]) > MONO_LIMIT]
    log(f"monotonicity: policies above {MONO_LIMIT:.1%} of steps: {viol_pols}")
    if viol_pols and not skip_rearranged:
        for yr in FOLDS:
            t1 = time.time()
            c = caches[(yr, 3)]
            rows = [("handbook", np.nan)] + [(p_, float(t)) for p_ in viol_pols for t in grid_theta]
            prep = S6.prepare(c, H, yr, geo, m, rows)
            tot_e = pd.read_parquet(R / "S6E" / f"f{yr}_h3_totals.parquet")
            for part in ("H2", "test"):
                C = prep[part]["C"]
                for p_ in viol_pols:
                    ix = [i for i, r_ in enumerate(rows) if r_[0] == p_]
                    lv = np.array([rows[i][1] for i in ix])
                    assert np.all(np.diff(lv) > 0)
                    C[ix] = rearrange(C[ix])
                prep[part]["C"] = C
            for sname in SHAPES:
                sc = S6.SCENARIOS[sname]
                res = {}
                for part in ("H2", "test"):
                    P = prep[part]
                    L = S6.load_for(P["grid"], unit96, scales, sc)
                    res[part] = LM.simulate(L, P["pav"], P["C"], P["day"], H, ratings=LM.fleet_for_peak(L.max()),
                                            min_load=sc["min_load"], fuel=sc["fuel"], battery=sc["battery"])["total"]
                hbT = float(res["test"]["fuel_L"][0])
                ref_hb = float(tot_e[(tot_e.scenario == sname) & (tot_e.set == "test") & (tot_e.policy == "handbook")].fuel_L.iloc[0])
                check(f"rearranged_handbook_equals_S6E_f{yr}_{sname}", abs(hbT / ref_hb - 1) < 1e-9)
                for p_ in viol_pols:
                    s_ = S6.select_policy(res["H2"], res["test"], rows, p_)
                    orig = sel[yr][sname]["policies"][p_]["A_R"]
                    rows_mono_ar.append(dict(fold=yr, scenario=sname, policy=p_, theta_rearranged=s_["A_R"]["param"],
                                             theta_original=float(orig["param"]),
                                             ue_share_rearranged=s_["A_R"]["test"]["ue_share"], ue_share_original=orig["test"]["ue_share"],
                                             meets_rearranged=bool(s_["A_R"]["test"]["ue_share"] <= TGT + EPS),
                                             meets_original=bool(orig["test"]["ue_share"] <= TGT + EPS),
                                             saving_rearranged_pct=100 * (1 - s_["A_R"]["test"]["fuel_L"] / hbT),
                                             saving_original_pct=100 * (1 - orig["test"]["fuel_L"] / hbT)))
            log(f"rearranged fold {yr}: {time.time() - t1:.0f} s")
    mar = pd.DataFrame(rows_mono_ar)
    mar.to_csv(out / "monotone_rearranged.csv", index=False)

    # ================================================================ reading rules (D-122)
    rd = {}
    osd = pd.DataFrame(rows_os)
    I, O = osd[osd.bound == "interpolated"], osd[osd.bound == "order_stat"]
    rd["1_order_stats"] = dict(met_interpolated=int(I.meets.sum()), met_order_stat=int(O.meets.sum()), n=len(I),
                               median_saving_interpolated=float(I.saving_pct.median()),
                               median_saving_order_stat=float(O.saving_pct.median()),
                               beyond_support_cells=int((O.beyond_support_steps > 0).sum()),
                               main_text=bool(abs(int(O.meets.sum()) - int(I.meets.sum())) > 2
                                              or abs(O.saving_pct.median() - I.saving_pct.median()) > 0.05),
                               rule="main text beside the interpolated result if met changes by more than 2 of 48 or the "
                                    "median conformal saving by more than 0.05 points, else one Supplement sentence")
    rd["2_support"] = dict(flags_calib_one_expected=int(fl[(fl.part == "calib")].above_one_expected.sum()),
                           flags_calib_order_stat=int(fl[(fl.part == "calib")].above_order_stat.sum()),
                           flags_H1_one_expected=int(fl[(fl.part == "H1")].above_one_expected.sum()),
                           flags_H1_order_stat=int(fl[(fl.part == "H1")].above_order_stat.sum()),
                           min_calib_cloudy_issue_times=int(sup[(sup.part == "calib") & (sup.regime == "cloudy")].issue_times.min()),
                           min_calib_cloudy_days=int(sup[(sup.part == "calib") & (sup.regime == "cloudy")].days.min()))
    # item 4 reading
    t4 = {}
    for pol in THP:
        ok_all = True
        detail = {}
        for th in [t for t in TAIL_THETAS if t <= 0.99]:
            g = tail[(tail.policy == pol) & (tail.theta == th) & (tail.regime.isin(REGS))]
            per_fold = g.groupby("fold").apply(lambda d: int((~d.reject_5pct).sum()))
            folds_ok = int((per_fold >= 2).sum())
            detail[str(th)] = folds_ok
            ok_all &= folds_ok >= 4
        t4[pol] = dict(reliable_lower_tail=bool(ok_all), folds_with_2_of_3_regimes_not_rejected=detail)
    rd["4_tail"] = dict(per_policy=t4, rule="reliable only if at every theta up to 0.99 the Kupiec test does not reject in "
                                            "at least 2 of 3 regimes in at least 4 of 6 folds")
    # item 5 reading
    mc = pers[(pers.method == "mondrian_cqr|static") & (pers.scope == "transitional")]
    rd["5_persistence"] = dict(mondrian_cqr_transitional_skill_by_fold={int(r_.fold): float(r_.skill) for r_ in mc.itertuples()},
                               folds_negative=int((mc.skill < 0).sum()),
                               no_crps_superiority_claim=bool((mc.skill < 0).sum() >= 4),
                               median_skill_all_static={mm_: float(pers[(pers.method == f"{mm_}|static") & (pers.scope == "all")].skill.median())
                                                        for mm_ in S5_STATIC})
    rd["6_monotonicity"] = dict(max_share_by_policy={p_: float(max(over.loc[p_, "test_share"], over.loc[p_, "H2_share"])) for p_ in THP},
                                size_median_over_folds={p_: {k: float(mono[mono.policy == p_][k].median())
                                                             for k in ("pv_kW_pair_share", "pv_kW_rise_p99", "pv_kW_rise_max",
                                                                       "pv_kW_sorted_gap_max", "ghi_Wm2_pair_share", "ghi_Wm2_rise_p99",
                                                                       "ghi_Wm2_rise_max", "test_share_ghi")} for p_ in THP},
                                policies_above_limit=viol_pols,
                                rearranged=(dict(met_rearranged=int(mar.meets_rearranged.sum()), met_original=int(mar.meets_original.sum()),
                                                 n=len(mar), median_saving_rearranged=float(mar.saving_rearranged_pct.median()),
                                                 median_saving_original=float(mar.saving_original_pct.median()))
                                            if len(mar) else None))
    # item 7 reading: headline coverage cells (main text) and their complete-day values
    head = [("yulara", 2024, 5, "icp", "transitional"), ("yulara", 2024, 5, "mondrian_cqr", "transitional"),
            ("yulara", 2024, 5, "icp", "clear"), ("yulara", 2024, 5, "icp", "cloudy")]
    head += [("yulara", y_, 5, m_, "transitional") for y_ in FOLDS for m_ in ("icp", "mondrian", "mondrian_cqr")]
    head += [("yulara", 2024, hm, "icp", "all") for hm in (5, 15, 30, 60)]
    head += [("yulara", 2024, hm, mm_, "transitional") for hm in (5, 15, 30, 60) for mm_ in ("icp", "tod", "mondrian_cqr")]
    # the lists above name four cells twice (fold 2024, 5 min, ICP and
    # Mondrian-CQR transitional, in the per-fold and the per-horizon lists). Count each cell once.
    head = list(dict.fromkeys(head))
    hv = []
    for s_, y_, hm, mm_, sc in head:
        r_ = cdc[(cdc.site == s_) & (cdc.fold == y_) & (cdc.horizon_min == hm) & (cdc.method == mm_) & (cdc.scope == sc)]
        if len(r_) == 1:
            hv.append(dict(fold=y_, horizon_min=hm, method=mm_, scope=sc, picp=float(r_.picp.iloc[0]),
                           picp_complete=float(r_.picp_complete.iloc[0]), change=float(r_.picp_complete.iloc[0] - r_.picp.iloc[0])))
    hv = pd.DataFrame(hv)
    cdd = pd.DataFrame(rows_cdd)
    rd["7_complete_days"] = dict(headline_cells=len(hv), max_abs_change=float(hv.change.abs().max()),
                                 cells_over_0_01=hv[hv.change.abs() > 0.01].to_dict("records"),
                                 AR_met_all_conformal=int(cdd[cdd.policy.isin(CP)].meets_all.sum()),
                                 AR_met_complete_conformal=int(cdd[cdd.policy.isin(CP)].meets_complete.sum()),
                                 AR_met_all_quantile=int(cdd[cdd.policy.isin(["raw_q", "cqr", "mondrian_cqr"])].meets_all.sum()),
                                 AR_met_complete_quantile=int(cdd[cdd.policy.isin(["raw_q", "cqr", "mondrian_cqr"])].meets_complete.sum()),
                                 complete_day_share_median=float((cdd.complete_days / cdd.days).median()))
    ch = b7[b7.status1 != b7.status7]
    rd["8_blocks7"] = dict(cells=len(b7), status_changes=len(ch),
                           changes_cited=ch[(ch.method.isin(["icp", "mondrian", "cqr", "mondrian_cqr", "tod"]))
                                            & (ch.scope.isin(["transitional", "all", "clear", "cloudy"]))]
                           [["site", "fold", "horizon_min", "method", "scope", "picp", "ci1_lo", "ci1_hi", "ci7_lo", "ci7_hi",
                             "status1", "status7"]].to_dict("records"))
    def st_(site, fold, hm, meth, sc):
        r_ = b7[(b7.site == site) & (b7.fold == fold) & (b7.horizon_min == hm) & (b7.method == meth) & (b7.scope == sc)]
        return (r_.status1.iloc[0], r_.status7.iloc[0]) if len(r_) == 1 else (None, None)
    n121 = [st_("yulara", y_, hm, "mondrian_cqr", "transitional") for y_ in FOLDS for hm in (5, 15)]
    rd["8_blocks7"]["cited"] = dict(
        mondrian_cqr_transitional_6_folds_5_15min=dict(below_1day=sum(a == "below" for a, _ in n121),
                                                             below_7day=sum(b == "below" for _, b in n121),
                                                             above_1day=sum(a == "above" for a, _ in n121),
                                                             above_7day=sum(b == "above" for _, b in n121), cells=len(n121)),
        yulara_2024={f"{mm_}|{sc}|{hm}": st_("yulara", 2024, hm, mm_, sc) for mm_ in ("icp", "tod", "mondrian", "cqr", "mondrian_cqr")
                     for sc in ("transitional", "clear", "cloudy", "all") for hm in (5, 15, 30, 60)},
        asp_2024={f"{mm_}|{sc}|{hm}": st_("asp", 2024, hm, mm_, sc) for mm_ in ("icp", "mondrian", "cqr", "mondrian_cqr")
                  for sc in ("transitional", "all") for hm in (5, 15, 30, 60)})
    ev_all = evr.groupby(["protocol", "regime"]).events.sum().unstack(fill_value=0)
    stp = pd.concat([pd.Series(caches[(yr, 3)][caches[(yr, 3)].split == "test"].g.values) for yr in FOLDS]).value_counts(normalize=True)
    rd["9_events"] = dict(event_share_by_protocol={p_: {k: float(v / ev_all.loc[p_].sum()) for k, v in ev_all.loc[p_].items()}
                                                   for p_ in ev_all.index},
                          scored_step_share={k: float(v) for k, v in stp.items()})
    rd["10_rowsets"] = dict(max_symmetric_difference=int((rs.only_test + rs.only_calib).max()),
                            min_g_equal_share=float(rs.g_equal_share.min()), min_y_equal_share=float(rs.y_equal_share.min()))
    json.dump(rd, open(out / "reading.json", "w"), indent=1, default=float)

    # ---------------- sanity checks on the outputs
    check("order_stats_48_cells", len(I) == 48 and len(O) == 48)
    check("tail_rows_complete", len(tail) == 6 * len(THP) * len(TAIL_THETAS) * 4 and tail.n.gt(0).all())
    check("pair_exceedance_nonempty", len(rows_pexc) == 6 * len(THP) * 2 * 9)
    check("events_counted_once", int(evr.drop_duplicates(["fold", "protocol", "scenario", "policy"]).n_events.sum()) == len(ev61),
          f"({len(ev61)} events)")
    check("persistence_rows", pers.CRPS_persistence.gt(0).all() and len(pers) > 0)
    check("no_nan_in_key_outputs", np.isfinite(osd.fuel_L).all() and np.isfinite(tail.exceed).all() and np.isfinite(b7.ci7_lo).all())

    ok = all(c_["ok"] for c_ in checks.values())
    outs = {p.name: md5(p) for p in sorted(out.glob("*")) if p.is_file() and p.name != "checks.json"}
    code = {p: md5(HERE / p) for p in ("ce1_s63.py", "ce1_dispatch.py", "ce1_load.py", "ce1_s6.py", "ce1_folds.py",
                                       "ce1_causal.py", "ce1_pvmap.py")}
    env = dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__,
               pvlib=__import__("pvlib").__version__, scipy=__import__("scipy").__version__)
    json.dump(dict(checks=checks, inputs=inputs, outputs=outs, code=code, env=env), open(out / "checks.json", "w"), indent=1)
    (R / "_runs").mkdir(exist_ok=True)
    detail = dict(stage="S63", ok=ok, seconds=round(time.time() - t0, 1), checks=checks, inputs=inputs, outputs=outs,
                  code=code, env=env)
    json.dump(detail, open(R / "_runs" / "S63.detail.json", "w"), indent=1)
    json.dump(dict(stage="S63", ok=ok, error="", started=started, finished=time.strftime("%Y-%m-%dT%H:%M:%S"),
                   minutes=round((time.time() - t0) / 60, 1), host=platform.node() + "", smoke=False,
                   n_checks=len(checks), n_failed=sum(not c_["ok"] for c_ in checks.values())),
              open(R / "_runs" / "S63.json", "w"), indent=1)
    log(f"{'ALL CHECKS PASS' if ok else 'CHECK FAILED'} ({len(checks)} checks, "
        f"{sum(not c_['ok'] for c_ in checks.values())} failed, {time.time() - t0:.0f} s)")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--skip-rearranged", action="store_true")
    a = ap.parse_args()
    st = selftest()
    if not all(st.values()):
        sys.exit(1)
    if a.selftest:
        sys.exit(0)
    sys.exit(0 if analyse(Path(a.root), a.skip_rearranged) else 1)


if __name__ == "__main__":
    main()
