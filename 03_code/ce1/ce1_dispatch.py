"""Decision layer: reserve policies as one-sided lower bounds on GHI at level theta, mapped to
counted PV, on the theta set 0.750 to 0.990 (step 0.005) plus 0.991 to 0.999 (step 0.001),
united with the critical fractile of every cost ratio. The physical model and parameters are
those of r1/r1_dispatch.py, evaluated in calendar time.

Usage (from the repository root unless the path starts at ce1/, then from 03_code):
    python ce1/ce1_dispatch.py --selftest
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

CODE = Path(__file__).resolve().parents[1]

# ---------------------------------------------------------------- parameters (as v1)
DT = 5 / 60.0
C_O_DEF = 0.30
RATIO_DEF = 10.0
RATIOS = [3.0, 5.0, 10.0, 19.0]
BATT = {"small": (250.0, 250.0), "default": (600.0, 400.0), "large": (1500.0, 600.0)}
ETA = 0.949
SOC_MIN, SOC_MAX = 0.10, 1.0
LAMBDAS = [0.0, 0.25, 0.5, 0.75, 1.0]
CVAR_ALPHA = 0.95
CP_POLICIES = ["icp", "mondrian", "cqr", "mondrian_cqr"]
MODES = ["reset", "carry", "settle", "strict"]
BASELINE_THETA = ["raw_q", "tod_q", "nicp_var"]
BASELINE_POLICIES = BASELINE_THETA + ["frac_pv"]
ALL_POLICIES = CP_POLICIES + BASELINE_POLICIES
FRAC_GRID = np.round(np.arange(0.0, 1.0 + 1e-9, 0.02), 4)  # frac_pv parameter
SEED = 42

THETA_FINE = np.round(np.arange(0.75, 0.99 + 1e-9, 0.005), 4)
THETA_TAIL = np.round(np.arange(0.991, 0.999 + 1e-9, 0.001), 4)
THETA_FINE = np.array(sorted(set(THETA_FINE.tolist()) | set(THETA_TAIL.tolist())))
THETA_LOW = np.round(np.arange(0.50, 0.745 + 1e-9, 0.005), 4)


def extend_theta_low() -> np.ndarray:
    """D-092: add THETA_LOW to the one theta set of both decision models (idempotent).
    Every function reading THETA_FINE (theta_set, tau_key, param_grid, taus_needed) then
    works on the extended set.
    """
    global THETA_FINE
    THETA_FINE = np.array(sorted(set(THETA_FINE.tolist()) | set(THETA_LOW.tolist())))
    return THETA_FINE


def taus_low() -> list[float]:
    """The base-quantile levels of THETA_LOW (0.255..0.500), fitted by stage S2X."""
    return sorted(round(1.0 - float(t), 6) for t in THETA_LOW)


def fractile(ratio: float) -> float:
    return round(ratio / (1.0 + ratio), 4)


def theta_set(ratios=(RATIO_DEF,)) -> np.ndarray:
    return np.array(sorted(set(THETA_FINE.tolist()) | {fractile(r) for r in ratios}))


def tau_key(theta: float) -> float:
    """Base-quantile level for the one-sided CQR bound at level theta. Fractile levels are
    fitted at their exact value 1/(1+ratio); keys are rounded to 6 decimals.
    """
    return round(1.0 - float(theta), 6) if round(theta, 4) in THETA_FINE else \
        round(1.0 / (1.0 + _ratio_of(theta)), 6)


def _ratio_of(theta):
    for r in RATIOS:
        if abs(fractile(r) - theta) < 1e-9:
            return r
    raise KeyError(f"theta {theta} is neither on the fine grid nor a known fractile")


def param_grid(policy, ratio=RATIO_DEF) -> np.ndarray:
    """Parameter set a policy is tuned over: FRAC_GRID for frac_pv, the theta set otherwise."""
    return FRAC_GRID if policy == "frac_pv" else theta_set((ratio,))


def nicp_scale(d, beta):
    """Scale of the variability-normalized ICP: G_cs * (v + beta)."""
    return np.asarray(d["gcs"], float) * (np.asarray(d["v"], float) + beta)


def _hour(d):
    ts = np.asarray(d["idx"], dtype="datetime64[ns]")
    return ((ts - ts.astype("datetime64[D]")) // np.timedelta64(1, "h")).astype(int)


def _grouped_offset(e_fit, g_fit, g_ev, p_ev, level):
    """Mondrian offset: (level) quantile of fit errors per group, marginal if n <= 30."""
    out = np.empty(len(p_ev))
    for g in np.unique(g_ev):
        eg = e_fit[g_fit == g]
        off = np.quantile(eg if len(eg) > 30 else e_fit, level)
        out[g_ev == g] = np.clip(p_ev[g_ev == g] + off, 0, None)
    return out


def taus_needed() -> list[float]:
    """Every lower quantile level the dispatch needs (for the quantile caches)."""
    return sorted({tau_key(t) for t in theta_set(RATIOS)})


# ---------------------------------------------------------------- policies
def counted_ghi(policy, theta, fit, ev):
    """GHI the operator counts on: one-sided lower bound at level theta (v1 definitions).
    fit/ev: dicts with y, p, g and q (dict tau_key -> array).
    """
    if policy == "deterministic":
        return ev["p"].copy()
    if policy == "oracle":
        return ev["y"].copy()
    if policy == "icp":
        e = fit["y"] - fit["p"]
        return np.clip(ev["p"] + np.quantile(e, 1 - theta), 0, None)
    if policy == "mondrian":
        e = fit["y"] - fit["p"]
        out = np.empty(len(ev["p"]))
        for g in np.unique(ev["g"]):
            eg = e[fit["g"] == g]
            off = np.quantile(eg if len(eg) > 30 else e, 1 - theta)
            out[ev["g"] == g] = np.clip(ev["p"][ev["g"] == g] + off, 0, None)
        return out
    if policy in ("cqr", "mondrian_cqr"):
        tk = tau_key(theta)
        if tk not in fit["q"] or tk not in ev["q"]:
            raise KeyError(f"no base quantile at tau={tk} for theta={theta}")
        s_all = fit["q"][tk] - fit["y"]
        if policy == "cqr":
            return np.clip(ev["q"][tk] - np.quantile(s_all, theta), 0, None)
        out = np.empty(len(ev["p"]))
        for g in np.unique(ev["g"]):
            sg = s_all[fit["g"] == g]
            adj = np.quantile(sg, theta) if len(sg) > 30 else np.quantile(s_all, theta)
            out[ev["g"] == g] = np.clip(ev["q"][tk][ev["g"] == g] - adj, 0, None)
        return out
    if policy == "raw_q":
        tk = tau_key(theta)
        if tk not in ev["q"]:
            raise KeyError(f"no base quantile at tau={tk} for theta={theta}")
        return np.clip(ev["q"][tk], 0, None)
    if policy == "tod_q":
        return _grouped_offset(fit["y"] - fit["p"], _hour(fit), _hour(ev), ev["p"], 1 - theta)
    if policy == "nicp_var":
        beta = float(np.median(fit["v"]))
        sf, se = nicp_scale(fit, beta), nicp_scale(ev, beta)
        r = (fit["y"] - fit["p"]) / sf
        return np.clip(ev["p"] + se * np.quantile(r, 1 - theta), 0, None)
    if policy == "frac_pv":
        return ev["p"].copy()  # the fraction is applied in PV space (counted_pv)
    raise ValueError(policy)


def counted_pv(policy, par, fit, ev, pv_of):
    """PV the operator counts on. Equal to pv_of(counted_ghi) except for frac_pv, where the
    reserve is a fraction of forecast PV: (1 - f) * pv_of(p).
    """
    if policy == "frac_pv":
        return (1.0 - float(par)) * pv_of(ev["p"])
    return pv_of(counted_ghi(policy, par, fit, ev))


# ---------------------------------------------------------------- time structure
def day_structure(idx):
    """Row order is time order. Returns day index per row and first/last row per day."""
    ts = np.asarray(idx, dtype="datetime64[ns]")
    assert (np.diff(ts) > np.timedelta64(0, "ns")).all(), "rows must be in strict time order"
    day = ts.astype("datetime64[D]")
    starts = np.r_[True, day[1:] != day[:-1]]
    ends = np.r_[day[1:] != day[:-1], True]
    return day, starts, ends


def missing_step_summary(idx, is_day_grid=None):
    """Retained scored steps per day on the 5-min clock. If a daytime grid is given
    (boolean Series over the full clock), the missing daytime steps are counted too.
    """
    ts = np.asarray(idx, dtype="datetime64[ns]")
    day = ts.astype("datetime64[D]")
    ud, cnt = np.unique(day, return_counts=True)
    out = dict(n_days=int(len(ud)), n_scored_steps=int(len(ts)))
    if is_day_grid is not None:
        g = is_day_grid[is_day_grid]
        gd = np.asarray(g.index, dtype="datetime64[ns]").astype("datetime64[D]")
        gdays, gcnt = np.unique(gd, return_counts=True)
        tot = dict(zip(gdays, gcnt))
        denom = sum(tot.get(d, 0) for d in ud)
        out.update(n_daytime_steps=int(denom), share_scored=round(len(ts) / denom, 4))
    return out


def complete_days(idx, is_day_grid, h_steps, min_share=0.95):
    """Days whose scored steps cover at least min_share of the steps that could be scored
    (daytime issue times whose target is also daytime).
    """
    import pandas as pd
    s = is_day_grid.astype(bool)
    possible = s & s.shift(-h_steps, fill_value=False)
    pd_days = pd.Series(possible.values, index=s.index).groupby(s.index.normalize()).sum()
    ts = pd.DatetimeIndex(idx)
    sc = pd.Series(1, index=ts).groupby(ts.normalize()).sum()
    share = (sc / pd_days.reindex(sc.index)).fillna(0)
    return set(share.index[share >= min_share].values.astype("datetime64[D]"))


# ---------------------------------------------------------------- simulation
def simulate_batch(counted_K, actual, idx, E_max, P_max, c_u, c_o, pv_of, mode="reset",
                   counted_is_pv=False):
    """Dispatch of K policies at once along the time-ordered rows.

    counted_K: (K, n) counted GHI (or counted PV if counted_is_pv); actual: (n) GHI.
    Returns dict with
      daily  (K, n_days) daily cost,  soc_end (K, n_days) end-of-day SoC fraction,
      soc_lo, soc_hi (K) extreme SoC fractions,
      e_chg (K, n_days) energy taken from PV surplus into the battery (kWh, PV side),
      e_dis (K, n_days) energy delivered by the battery to cover deficits (kWh, load side),
      e_def, e_sur (K, n_days) gross deficit and surplus energy before the battery (kWh).
    """
    assert mode in MODES
    counted_K = np.atleast_2d(counted_K)
    K, n = counted_K.shape
    day, starts, ends = day_structure(idx)
    dnum = np.cumsum(starts) - 1
    nd = int(dnum[-1] + 1) if n else 0
    C = counted_K.astype(float) if counted_is_pv else pv_of(counted_K)
    A = pv_of(actual)[None, :]
    e0 = SOC_MIN * E_max
    soc = np.full(K, e0)
    daily = np.zeros((K, nd))
    soc_end = np.zeros((K, nd))
    lo = np.full(K, e0); hi = np.full(K, e0)
    e_chg = np.zeros((K, nd)); e_dis = np.zeros((K, nd))
    e_def = np.zeros((K, nd)); e_sur = np.zeros((K, nd))
    for i in range(n):
        if starts[i] and mode != "carry":
            soc = np.full(K, e0)
        diff = A[:, i] - C[:, i]
        deficit = np.maximum(-diff, 0.0)
        surplus = np.maximum(diff, 0.0)
        dis = np.clip(np.minimum(np.minimum(deficit, P_max), (soc - e0) * ETA / DT), 0, None)
        chg = np.clip(np.minimum(np.minimum(surplus, P_max), (SOC_MAX * E_max - soc) / (ETA * DT)),
                      0, None)
        soc = soc - dis * DT / ETA + chg * ETA * DT
        wasted = surplus if mode == "strict" else surplus - chg
        daily[:, dnum[i]] += c_u * (deficit - dis) * DT + c_o * wasted * DT
        e_chg[:, dnum[i]] += chg * DT; e_dis[:, dnum[i]] += dis * DT
        e_def[:, dnum[i]] += deficit * DT; e_sur[:, dnum[i]] += surplus * DT
        lo = np.minimum(lo, soc); hi = np.maximum(hi, soc)
        if ends[i]:
            if mode == "settle":
                daily[:, dnum[i]] += c_o * (soc - e0) / ETA
            soc_end[:, dnum[i]] = soc / E_max
    return dict(daily=daily, soc_end=soc_end, soc_lo=lo / E_max, soc_hi=hi / E_max,
                e_chg=e_chg, e_dis=e_dis, e_def=e_def, e_sur=e_sur)


def simulate_calendar(counted, actual, idx, E_max, P_max, c_u, c_o, pv_of, mode="reset"):
    """Reference for the self-test: every day laid out on its full 288-step clock, missing
    steps hold the state of charge and cost nothing. One policy, plain loops.
    """
    ts = np.asarray(idx, dtype="datetime64[ns]")
    day = ts.astype("datetime64[D]")
    slot = ((ts - day.astype("datetime64[ns]")) // np.timedelta64(5, "m")).astype(int)
    udays = np.unique(day)
    e0 = SOC_MIN * E_max
    soc = e0
    daily = np.zeros(len(udays)); soc_end = np.zeros(len(udays))
    pc, pa = pv_of(counted), pv_of(actual)
    for d_i, d in enumerate(udays):
        if mode != "carry":
            soc = e0
        rows = np.where(day == d)[0]
        clock = {int(slot[r]): r for r in rows}
        for k in range(288):
            if k not in clock:
                continue  # missing step: hold SoC, no cost
            r = clock[k]
            diff = pa[r] - pc[r]
            deficit, surplus = max(-diff, 0.0), max(diff, 0.0)
            dis = max(min(deficit, P_max, (soc - e0) * ETA / DT), 0.0)
            chg = max(min(surplus, P_max, (SOC_MAX * E_max - soc) / (ETA * DT)), 0.0)
            soc = soc - dis * DT / ETA + chg * ETA * DT
            wasted = surplus if mode == "strict" else surplus - chg
            daily[d_i] += c_u * (deficit - dis) * DT + c_o * wasted * DT
        if mode == "settle":
            daily[d_i] += c_o * (soc - e0) / ETA
        soc_end[d_i] = soc / E_max
    return daily, soc_end


def cvar(x, alpha=CVAR_ALPHA):
    k = max(1, int(np.ceil((1 - alpha) * len(x))))
    return float(np.mean(np.sort(x)[::-1][:k]))


def objective(daily_vec, lam):
    return (1 - lam) * float(daily_vec.mean()) + lam * cvar(daily_vec)


# ---------------------------------------------------------------- sweeps and protocols
def _unpack(r, k):
    d = dict(daily=r["daily"][k], soc_end=r["soc_end"][k], soc_lo=float(r["soc_lo"][k]),
             soc_hi=float(r["soc_hi"][k]))
    d.update({e: r[e][k] for e in ("e_chg", "e_dis", "e_def", "e_sur")})
    return d


def _bound(pv_of, ev):
    """A time-aware PV map (an object with .bind(ev), e.g. ce1_s6.PVBinder) is bound
    to the target times of ev's rows; a plain function passes through unchanged.
    """
    return pv_of.bind(ev) if hasattr(pv_of, "bind") else pv_of


def sweep(fit, ev, thetas, E, P, c_u, c_o, pv_of, mode, policies=CP_POLICIES):
    """Daily costs for every (policy, parameter) on ev (scores from fit), one batch per
    policy. Theta policies use `thetas`; frac_pv uses FRAC_GRID.
    """
    pv_of = _bound(pv_of, ev)
    out = {}
    for pol in policies:
        grid = FRAC_GRID if pol == "frac_pv" else thetas
        CK = np.vstack([counted_pv(pol, th, fit, ev, pv_of) for th in grid])
        r = simulate_batch(CK, ev["y"], ev["idx"], E, P, c_u, c_o, pv_of, mode,
                           counted_is_pv=True)
        for k, th in enumerate(grid):
            out[(pol, round(float(th), 4))] = _unpack(r, k)
    return out


ANCHORS = ["deterministic", "oracle", "pv_rule_100"]


def anchors(ev, E, P, c_u, c_o, pv_of, mode):
    """deterministic (commit the point forecast), oracle (commit the actual) and
    pv_rule_100 (the operator's 100%-of-PV reserve: nothing is counted on).
    """
    pv_of = _bound(pv_of, ev)
    CK = np.vstack([pv_of(ev["p"]), pv_of(ev["y"]), np.zeros(len(ev["y"]))])
    r = simulate_batch(CK, ev["y"], ev["idx"], E, P, c_u, c_o, pv_of, mode, counted_is_pv=True)
    return {name: _unpack(r, k) for k, name in enumerate(ANCHORS)}


def argmin_theta(sw, pol, thetas, lam=0.0):
    best, arg = np.inf, None
    for th in thetas:
        v = objective(sw[(pol, round(float(th), 4))]["daily"], lam)
        if v < best - 1e-12:
            best, arg = v, round(float(th), 4)
    return arg, best


def local_minima(sw, pol, thetas, lam=0.0):
    """Levels where the objective is below both grid neighbours (for the cost-surface
    report: a sharp surface shows several minima within a few thousandths).
    """
    v = np.array([objective(sw[(pol, round(float(t), 4))]["daily"], lam) for t in thetas])
    m = [i for i in range(len(v)) if (i == 0 or v[i] < v[i - 1]) and (i == len(v) - 1 or v[i] < v[i + 1])]
    return [round(float(thetas[i]), 4) for i in m]


def split_halves(calib):
    import pandas as pd
    m = pd.DatetimeIndex(calib["idx"]).month <= 6

    def sub(mask):
        d = {k: np.asarray(v)[mask] for k, v in calib.items() if k != "q"}
        d["q"] = {k: v[mask] for k, v in calib["q"].items()}
        return d
    return sub(np.asarray(m)), sub(~np.asarray(m))


def run_protocols(calib, test, pv_of, mode="reset", ratio=RATIO_DEF, batt="default",
                  c_o=C_O_DEF, lambdas=LAMBDAS, policies=ALL_POLICIES):
    """Protocol A (select on calibration H2 with scores from H1), A swapped, B (fractile),
    and the test-tuned reference, all on the same parameter set per policy, in one energy
    mode. Rows carry absolute costs (mean and CVaR, per day) and the energy flows.
    """
    c_u = c_o * ratio
    E, Pw = BATT[batt]
    thetas = theta_set((ratio,))
    H1, H2 = split_halves(calib)
    selA = sweep(H1, H2, thetas, E, Pw, c_u, c_o, pv_of, mode, policies)
    selS = sweep(H2, H1, thetas, E, Pw, c_u, c_o, pv_of, mode, policies)
    ev = sweep(calib, test, thetas, E, Pw, c_u, c_o, pv_of, mode, policies)
    anc = anchors(test, E, Pw, c_u, c_o, pv_of, mode)
    det = anc["deterministic"]["daily"]
    dm, dc = float(det.mean()), cvar(det)
    rows, choices = [], []

    def rec(pol, th, protocol, lam, sel_obj=None, ref=None):
        r = ev[(pol, round(float(th), 4))]
        m, cv = float(r["daily"].mean()), cvar(r["daily"])
        row = dict(mode=mode, ratio=ratio, batt=batt, protocol=protocol, policy=pol,
                   param_kind="f" if pol == "frac_pv" else "theta",
                   theta=round(float(th), 4), lam=lam, mean_daily=m, cvar95_daily=cv,
                   **{f"{e}_mean": float(r[e].mean()) for e in ("e_chg", "e_dis", "e_def", "e_sur")},
                   vc_mean=1 - m / dm if dm > 0 else np.nan,
                   vc_cvar=1 - cv / dc if dc > 0 else np.nan,
                   soc_end_mean=float(r["soc_end"].mean()), soc_end_max=float(r["soc_end"].max()),
                   soc_lo=r["soc_lo"], soc_hi=r["soc_hi"], selection_objective=sel_obj)
        if ref is not None:
            row["regret_signed"] = objective(r["daily"], lam) - ref
        rows.append(row)

    for name, a in anc.items():
        m_a, cv_a = float(a["daily"].mean()), cvar(a["daily"])
        om, oc = float(anc["oracle"]["daily"].mean()), cvar(anc["oracle"]["daily"])
        rows.append(dict(mode=mode, ratio=ratio, batt=batt, protocol="anchor", policy=name,
                         param_kind=None, theta=np.nan, lam=np.nan, mean_daily=m_a,
                         cvar95_daily=cv_a,
                         **{f"{e}_mean": float(a[e].mean()) for e in ("e_chg", "e_dis", "e_def", "e_sur")},
                         vc_mean=(dm - m_a) / (dm - om) if dm != om else np.nan,
                         vc_cvar=(dc - cv_a) / (dc - oc) if dc != oc else np.nan,
                         soc_end_mean=float(a["soc_end"].mean()), soc_end_max=float(a["soc_end"].max()),
                         soc_lo=a["soc_lo"], soc_hi=a["soc_hi"], selection_objective=None))
    for pol in policies:
        grid = FRAC_GRID if pol == "frac_pv" else thetas
        for lam in lambdas:
            thR, objR = argmin_theta(ev, pol, grid, lam)
            thA, objA = argmin_theta(selA, pol, grid, lam)
            thS, objS = argmin_theta(selS, pol, grid, lam)
            thB = fractile(ratio) if pol != "frac_pv" else None
            rec(pol, thR, "reference_test_tuned", lam, ref=objR)
            rec(pol, thA, "A", lam, sel_obj=objA, ref=objR)
            rec(pol, thS, "A_swapped", lam, sel_obj=objS, ref=objR)
            if lam == 0.0 and thB is not None:
                rec(pol, thB, "B_fractile", lam, ref=objR)
            choices.append(dict(mode=mode, ratio=ratio, batt=batt, policy=pol, lam=lam,
                                theta_A=thA, theta_A_swapped=thS, theta_B=thB, theta_ref=thR,
                                minima_selection=local_minima(selA, pol, grid, lam),
                                minima_test=local_minima(ev, pol, grid, lam)))
    surface = [dict(mode=mode, ratio=ratio, batt=batt, set=sname, policy=pol, theta=th,
                    mean_daily=float(v["daily"].mean()), cvar95_daily=cvar(v["daily"]),
                    soc_end_mean=float(v["soc_end"].mean()))
               for sname, sw in (("selection_H2", selA), ("test", ev)) for (pol, th), v in sw.items()]
    daily = {f"{pol}|{th}": v["daily"] for (pol, th), v in ev.items()}
    daily.update({f"{k}|na": v["daily"] for k, v in anc.items()})
    return dict(rows=rows, choices=choices, surface=surface, daily=daily)


# ---------------------------------------------------------------- self-tests
def _toy(seed=0, n_days=30, gaps=True, every=1):
    rng = np.random.default_rng(seed)
    ts = []
    for d in range(n_days):
        day0 = np.datetime64("2023-01-01T07:00") + np.timedelta64(d * every, "D")
        k = np.arange(140)
        if gaps:
            k = k[rng.random(140) > 0.1]
        ts.extend(day0 + k * np.timedelta64(5, "m"))
    idx = np.array(ts, dtype="datetime64[ns]")
    n = len(idx)
    hod = (idx - idx.astype("datetime64[D]")) / np.timedelta64(1, "h")
    cs = np.clip(1000 * np.sin(np.pi * (hod - 6) / 12), 0, None)
    y = np.clip(cs * np.clip(0.8 + 0.25 * rng.standard_normal(n), 0, 1.3), 0, None)
    p = np.clip(0.7 * y + 0.3 * cs * 0.8 + 60 * rng.standard_normal(n), 0, None)
    g = np.where(rng.random(n) < 0.5, "clear", "transitional")
    taus = taus_needed()
    q = {t: np.clip(p + np.quantile(y - p, t) + 20 * rng.standard_normal(n), 0, None) for t in taus}
    v = rng.gamma(2.0, 0.03, n)  # trailing variability of the clear-sky index
    return dict(idx=idx, y=y, p=p, g=g, q=q, v=v, gcs=np.maximum(cs, 20.0))


def _pv(g):
    return np.clip(np.asarray(g) * 1.0, 0, 1000.0)


def selftest(verbose=True):
    res = {}

    def ok(name, cond, msg=""):
        res[name] = bool(cond)
        if verbose:
            print(f"  [{'PASS' if cond else 'FAIL'}] {name} {msg}")

    th = theta_set(RATIOS)
    ok("theta_set_contains_fractiles", all(fractile(r) in th for r in RATIOS),
       f"({len(th)} levels, {th.min()}..{th.max()})")
    ok("every_theta_has_exact_tau", all(abs(tau_key(t) - (1 - t)) < 6e-5 for t in th))
    ok("grid_reaches_0.99", th.max() >= 0.99)

    ev = _toy(1)
    E, P = BATT["default"]
    c_o, c_u = C_O_DEF, C_O_DEF * RATIO_DEF
    # 1. calendar layout = batched time-ordered rows, every mode, gappy days
    for mode in MODES:
        cg = counted_ghi("icp", 0.9, ev, ev)
        b = simulate_batch(cg[None, :], ev["y"], ev["idx"], E * 0.2, P * 0.2, c_u, c_o, _pv, mode)
        cd, ce = simulate_calendar(cg, ev["y"], ev["idx"], E * 0.2, P * 0.2, c_u, c_o, _pv, mode)
        ok(f"calendar_equals_packed_{mode}", np.allclose(b["daily"][0], cd, atol=1e-9)
           and np.allclose(b["soc_end"][0], ce, atol=1e-12))
    # 2. reset mode reproduces the frozen v1 simulate (vectorised over days, packed)
    R1 = _load_r1_dispatch()
    cg = counted_ghi("mondrian", 0.925, ev, ev)
    _, M = R1.day_matrix(__import__("pandas").DatetimeIndex(ev["idx"]))
    r1_cost, _, _ = R1.simulate(cg, ev["y"], M, E * 0.3, P * 0.3, c_u, c_o)
    b = simulate_batch(cg[None, :], ev["y"], ev["idx"], E * 0.3, P * 0.3, c_u, c_o, R1.pv_of, "reset")
    ok("reset_equals_R1_simulate", np.allclose(b["daily"][0], r1_cost, atol=1e-9))
    # v1 policy definitions reproduced exactly on the v1 grid
    same = all(np.array_equal(counted_ghi(pol, t, ev, ev),
                              R1.counted_ghi(pol, t, ev, ev, sorted(ev["q"])))
               for pol in CP_POLICIES for t in (0.75, 0.8, 0.85, 0.9, 0.95))
    ok("policies_equal_R1_on_R1_grid", same)
    # 3. settle = reset + c_o * (E_end - E_start) / eta, per day
    br = simulate_batch(cg[None, :], ev["y"], ev["idx"], E * 0.3, P * 0.3, c_u, c_o, _pv, "reset")
    bs = simulate_batch(cg[None, :], ev["y"], ev["idx"], E * 0.3, P * 0.3, c_u, c_o, _pv, "settle")
    extra = c_o * (bs["soc_end"][0] * E * 0.3 - SOC_MIN * E * 0.3) / ETA
    ok("settle_identity", np.allclose(bs["daily"][0], br["daily"][0] + extra, atol=1e-9))
    # 4. carry = reset when no energy is ever stored (no surplus: counted >= actual)
    hi_c = ev["y"] + 50
    bc = simulate_batch(hi_c[None, :], ev["y"], ev["idx"], E, P, c_u, c_o, _pv, "carry")
    br2 = simulate_batch(hi_c[None, :], ev["y"], ev["idx"], E, P, c_u, c_o, _pv, "reset")
    ok("carry_equals_reset_without_storage", np.allclose(bc["daily"], br2["daily"]))
    # carry differs from reset when energy is stored overnight
    lo_c = ev["y"] * 0.5
    bc = simulate_batch(lo_c[None, :], ev["y"], ev["idx"], E, P, c_u, c_o, _pv, "carry")
    br3 = simulate_batch(lo_c[None, :], ev["y"], ev["idx"], E, P, c_u, c_o, _pv, "reset")
    ok("carry_differs_with_overnight_energy", not np.allclose(bc["daily"], br3["daily"]))
    # 5. protocols: the reference is a minimum over the same set, so nothing beats it
    cal, te = _toy(2, 40, every=9), _toy(3, 30)
    out = run_protocols(cal, te, _pv, mode="reset", lambdas=[0.0, 0.5])
    import pandas as pd
    R = pd.DataFrame(out["rows"])
    worst = float(R[R.protocol != "anchor"]["regret_signed"].min())
    ok("reference_is_a_minimum", worst >= -1e-9, f"(min signed regret {worst:.3g})")
    ok("B_never_beats_reference", (R[R.protocol == "B_fractile"]["regret_signed"] >= -1e-9).all())
    # 6. monotone bound: the ICP counted GHI falls as theta rises
    c1 = counted_ghi("icp", 0.80, ev, ev); c2 = counted_ghi("icp", 0.95, ev, ev)
    ok("icp_bound_monotone_in_theta", (c2 <= c1 + 1e-12).all())

    # ---- WP-45: strict accounting and energy flows
    E3, P3 = E * 0.3, P * 0.3
    cg = counted_ghi("mondrian", 0.9, ev, ev)
    br = simulate_batch(cg[None, :], ev["y"], ev["idx"], E3, P3, c_u, c_o, _pv, "reset")
    bt = simulate_batch(cg[None, :], ev["y"], ev["idx"], E3, P3, c_u, c_o, _pv, "strict")
    ok("strict_identity", np.allclose(bt["daily"], br["daily"] + c_o * br["e_chg"], atol=1e-9)
       and np.array_equal(bt["soc_end"], br["soc_end"]))
    ok("strict_ge_reset", (bt["daily"] >= br["daily"] - 1e-12).all()
       and (bt["daily"] > br["daily"] + 1e-9).any())
    bz = simulate_batch(cg[None, :], ev["y"], ev["idx"], E3, 0.0, c_u, c_o, _pv, "strict")
    bz0 = simulate_batch(cg[None, :], ev["y"], ev["idx"], E3, 0.0, c_u, c_o, _pv, "reset")
    ok("strict_equals_reset_without_battery", np.allclose(bz["daily"], bz0["daily"]))
    bal = br["soc_end"][0] * E3 - SOC_MIN * E3 - (ETA * br["e_chg"][0] - br["e_dis"][0] / ETA)
    ok("daily_energy_balance_closes", np.abs(bal).max() < 1e-6, f"(max {np.abs(bal).max():.2e} kWh)")
    gross = c_u * (br["e_def"] - br["e_dis"]) + c_o * (br["e_sur"] - br["e_chg"])
    ok("cost_equals_priced_energy_flows", np.allclose(gross, br["daily"], atol=1e-9))

    # ---- WP-45: baselines
    fit = _toy(4, 40, every=9)
    t = 0.9
    raw = counted_ghi("raw_q", t, fit, ev); cq = counted_ghi("cqr", t, fit, ev)
    both = (raw > 0) & (cq > 0)
    d = raw[both] - cq[both]
    ok("raw_q_is_cqr_without_correction", np.allclose(raw, np.clip(ev["q"][tau_key(t)], 0, None))
       and np.ptp(d) < 1e-9, f"(constant CQR shift {d.mean():.3f})")
    fit_h, ev_h = dict(fit, g=_hour(fit)), dict(ev, g=_hour(ev))
    ok("tod_q_equals_mondrian_on_hour_groups",
       np.array_equal(counted_ghi("tod_q", t, fit, ev), counted_ghi("mondrian", t, fit_h, ev_h)))
    fit_c, ev_c = dict(fit, v=np.full(len(fit["y"]), 0.07), gcs=np.full(len(fit["y"]), 500.0)), \
        dict(ev, v=np.full(len(ev["y"]), 0.07), gcs=np.full(len(ev["y"]), 500.0))
    ok("nicp_var_constant_scale_equals_icp",
       np.allclose(counted_ghi("nicp_var", t, fit_c, ev_c), counted_ghi("icp", t, fit, ev), atol=1e-9))
    beta_fit = float(np.median(fit["v"]))
    ev_v2 = dict(ev, v=ev["v"] * 2)
    lhs = counted_ghi("nicp_var", t, fit, ev_v2)
    rq = np.quantile((fit["y"] - fit["p"]) / nicp_scale(fit, beta_fit), 1 - t)
    ok("nicp_var_beta_from_fit_only",
       np.allclose(lhs, np.clip(ev["p"] + nicp_scale(ev_v2, beta_fit) * rq, 0, None)))
    anc = anchors(ev, E3, P3, c_u, c_o, _pv, "reset")
    f0 = simulate_batch(counted_pv("frac_pv", 0.0, fit, ev, _pv)[None, :], ev["y"], ev["idx"],
                        E3, P3, c_u, c_o, _pv, "reset", counted_is_pv=True)
    f1 = simulate_batch(counted_pv("frac_pv", 1.0, fit, ev, _pv)[None, :], ev["y"], ev["idx"],
                        E3, P3, c_u, c_o, _pv, "reset", counted_is_pv=True)
    ok("frac_pv_0_equals_deterministic", np.allclose(f0["daily"][0], anc["deterministic"]["daily"]))
    ok("frac_pv_1_equals_pv_rule_100", np.allclose(f1["daily"][0], anc["pv_rule_100"]["daily"]))
    ok("oracle_costs_nothing", np.allclose(anc["oracle"]["daily"], 0))
    cpv = np.vstack([counted_pv("frac_pv", f, fit, ev, _pv) for f in FRAC_GRID])
    ok("frac_pv_counted_monotone_in_f", (np.diff(cpv, axis=0) <= 1e-12).all())
    ok("counted_pv_equals_pv_of_counted_ghi",
       all(np.allclose(counted_pv(pp, t, fit, ev, _pv), _pv(counted_ghi(pp, t, fit, ev)))
           for pp in CP_POLICIES + BASELINE_THETA))
    # protocols with every policy and the strict mode
    out = run_protocols(cal, te, _pv, mode="strict", lambdas=[0.0, 0.5])
    R = pd.DataFrame(out["rows"])
    ok("all_policies_reported", set(R.policy) == set(ALL_POLICIES) | set(ANCHORS))
    worst = float(R[R.protocol != "anchor"]["regret_signed"].min())
    ok("reference_is_a_minimum_all_policies_strict", worst >= -1e-9, f"(min {worst:.3g})")
    Bp = set(R[R.protocol == "B_fractile"].policy)
    ok("B_defined_for_theta_policies_only", Bp == set(CP_POLICIES + BASELINE_THETA))
    fr = R[(R.policy == "frac_pv") & (R.protocol == "A")]
    ok("frac_pv_selects_on_frac_grid", fr.theta.isin(FRAC_GRID).all() and (fr.param_kind == "f").all())
    return res


def _load_r1_dispatch():
    """Import the frozen r1_dispatch.py with its GHI->PV map replaced by a toy map, without
    touching /tmp (the frozen module loads /tmp/ghi_pv_map.npz at import).
    """
    import types
    src = (CODE / "r1" / "r1_dispatch.py").read_text()
    src = src.replace('_m = np.load("/tmp/ghi_pv_map.npz")',
                      '_m = dict(centers=np.array([0.0, 1500.0]), vals=np.array([0.0, 1500.0]), cap=1000.0)')
    mod = types.ModuleType("r1_dispatch_frozen")
    sys.path.insert(0, str(CODE / "utils"))
    exec(compile(src, "r1_dispatch.py", "exec"), mod.__dict__)
    return mod


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        r = selftest()
        sys.exit(0 if all(r.values()) else 1)
