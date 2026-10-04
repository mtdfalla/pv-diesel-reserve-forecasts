"""Stage S51: adaptive reserve level (Protocol A-D).

The reserve level of each policy is updated once per day from the unserved energy (UE)
already observed in the test year:

    theta_{d+1} = max(theta_lo, theta_d + gamma * (UE_d - a_d) / a_bar)
    a_d   = kappa * 0.002% * load_d
    a_bar = kappa * 0.002% * mean daily scored load of the calibration year's second half

It starts from the policy's Protocol A-R level and applies, on day d, the smallest level of
the policy's set at or above the latent level. Primary gamma = 0.005, kappa = 0.5, with a
4 x 4 descriptive grid (rule D-094).

Usage (from the repository root unless the path starts at ce1/, then from 03_code):
    python3 03_code/ce1/ce1_adaptive.py --root . [--ext] [--s6 S6E --s6b S6E]
    python3 03_code/ce1/ce1_adaptive.py --selftest
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
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "utils"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import ce1_dispatch as D  # noqa: E402
import ce1_folds as FC  # noqa: E402
import ce1_load as LM  # noqa: E402
import ce1_s6 as S6  # noqa: E402
from ce1_decision import draw_counts, ci  # noqa: E402

# ---------------------------------------------------------------- D-094 constants
GAMMA, KAPPA = 0.005, 0.5
GAMMAS = (0.0025, 0.005, 0.01, 0.02)
KAPPAS = (0.25, 0.5, 0.75, 1.0)
VARIANTS = [(0.0, KAPPA)] + [(g, k) for g in GAMMAS for k in KAPPAS]  # first: constant (check)
TGT = LM.UE_TARGET
FOLDS = [2019, 2020, 2021, 2022, 2023, 2024]
H_PRIMARY = 3
H_SECONDARY = (1, 6, 12)
SCEN_PRIMARY = ("main", "shapeB")
SCEN_BATT = ("battery", "batteryB")
CP = ["icp", "mondrian", "cqr", "mondrian_cqr"]
POLICIES = list(D.ALL_POLICIES)
FOLD_MIN = 4
B_DEF, SEED = 10000, 42
LEVEL_TOL = 1e-9


def md5(p: Path) -> str:
    h = hashlib.md5()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def log(m):
    print(f"[S51] {m}", flush=True)


# ---------------------------------------------------------------- the controller
def level_index(levels, latent):
    """Index of the smallest level at or above the latent level (the last level if above all)."""
    i = int(np.searchsorted(levels, latent - LEVEL_TOL, side="left"))
    return min(i, len(levels) - 1)


def control(ue, load, theta0, gamma, kappa, abar_load, levels, lo):
    """Day-by-day levels from the daily UE and scored load of the path itself.

    ue, load (nd) kWh of the path's own days; abar_load the calibration H2 mean daily scored
    load. The level of day d depends on ue[:d] and load[:d] only. Returns (idx (nd) into
    levels, latent (nd+1) with latent[0] = theta0 and latent[d+1] after day d's update).
    """
    nd = len(ue)
    idx = np.empty(nd, np.int64)
    lat = np.empty(nd + 1)
    lat[0] = x = float(theta0)
    abar = kappa * TGT * abar_load
    for d in range(nd):
        idx[d] = level_index(levels, max(x, lo))
        if gamma:
            x = max(lo, x + gamma * (ue[d] - kappa * TGT * load[d]) / abar)
        lat[d + 1] = x
    return idx, lat


def control_lookup(rec_ue, load, theta0, gamma, kappa, abar_load, levels, lo):
    """First-pass path from the per-level daily record: rec_ue (n_levels, nd)."""
    nd = rec_ue.shape[1]
    idx = np.empty(nd, np.int64)
    x = float(theta0)
    abar = kappa * TGT * abar_load
    for d in range(nd):
        i = level_index(levels, max(x, lo))
        idx[d] = i
        if gamma:
            x = max(lo, x + gamma * (rec_ue[i, d] - kappa * TGT * load[d]) / abar)
    return idx


def path_row(Cpol, idx, day):
    """Counted PV of an adaptive path: each step takes the row of its day's level."""
    return Cpol[idx[day], np.arange(Cpol.shape[1])]


def causal_ok(Cpol, grid, h):
    """Every step whose counted PV differs between the levels is issued on its own day."""
    with np.errstate(invalid="ignore"):
        spread = np.nanmax(Cpol, axis=0) - np.nanmin(Cpol, axis=0)
    diff = np.nan_to_num(spread) > 0
    if not diff.any():
        return True, 0
    g = pd.DatetimeIndex(grid)[diff]
    issue = g - pd.Timedelta(minutes=5 * h)
    bad = int((issue.normalize() != g.normalize()).sum())
    return bad == 0, bad


def run_paths(L, pav, day, h, sim_kw, pol_C, specs, rec_ue, load_d, abar_load, max_pass=400):
    """Fixed-point simulation of every adaptive row of one cell.

    pol_C {policy: (levels, C (n_levels, n))}; specs list of (policy, gamma, kappa, theta0, lo).
    Rows of ce1_load.simulate are independent (the S6X subset identity), so a row whose
    recomputed levels equal the levels it was simulated with is at its fixed point and stays
    there: it is frozen, and later passes simulate the other rows only.
    Returns (result dict with daily and total arrays, idx paths (K, nd), latent (K, nd+1),
    passes, simulated rows summed over passes).
    """
    nd = int(day.max()) + 1
    K = len(specs)
    idx = np.empty((K, nd), np.int64)
    for k, (pol, g, kp, t0, lo) in enumerate(specs):
        levels, _ = pol_C[pol]
        idx[k] = control_lookup(rec_ue[pol], load_d, t0, g, kp, abar_load, levels, lo)
    lat = np.zeros((K, nd + 1))
    daily = {k: np.zeros((K, nd)) for k in LM.DAY_KEYS}
    total = {}
    active = np.arange(K)
    work = 0
    for p in range(1, max_pass + 1):
        C = np.vstack([path_row(pol_C[specs[k][0]][1], idx[k], day) for k in active])
        r = LM.simulate(L, pav, C, day, h, **sim_kw)
        work += len(active)
        ue = r["daily"]["ue_kWh"]
        keep = []
        for j, k in enumerate(active):
            pol, g, kp, t0, lo = specs[k]
            new, lat[k] = control(ue[j], load_d, t0, g, kp, abar_load, pol_C[pol][0], lo)
            if np.array_equal(new, idx[k]):
                for key in LM.DAY_KEYS:
                    daily[key][k] = r["daily"][key][j]
                for key, v in r["total"].items():
                    total.setdefault(key, np.full(K, np.nan))[k] = v[j]
            else:
                idx[k] = new
                keep.append(k)
        if not keep:
            return dict(daily=daily, total=total), idx, lat, p, work
        active = np.array(keep)
    raise RuntimeError(f"fixed point not reached in {max_pass} passes")


def bounds(ue_d, load_d, lat, gamma, kappa, abar_load, theta0):
    a = kappa * TGT * load_d
    abar = kappa * TGT * abar_load
    s = float(np.sum(ue_d - a))
    if not gamma:
        return True, np.nan
    ident = s <= abar * (lat[-1] - theta0) / gamma + 1e-6 * max(1.0, abs(s))
    rmax = max(0.0, float(np.max(ue_d - a)) / abar) if len(ue_d) else 0.0
    U = float(a.sum()) + abar * ((max(theta0, 1.0) - theta0) / gamma + rmax)
    return bool(ident), U


# ---------------------------------------------------------------- one fold and horizon
def h2_mean_daily_load(P, unit96, scales, sc):
    """Mean daily scored load (kWh) of the calibration year's H2 (a_bar's basis)."""
    L = S6.load_for(P["grid"], unit96, scales, sc)
    sc_mask = ~np.isnan(P["pav"])
    e = np.bincount(P["day"], weights=np.where(sc_mask, L, 0.0) * LM.DT)
    n = np.bincount(P["day"], weights=sc_mask.astype(float))
    return float(e[n > 0].mean())


def record_rows(z):
    pol = np.asarray(z["rows_policy"]).astype(str)
    par = np.asarray(z["rows_param"]).astype(float)
    return {(p, round(float(x), 6)): i for i, (p, x) in enumerate(zip(pol, par)) if p not in S6.ANCHORS}, \
        {p: i for i, p in enumerate(pol) if p in S6.ANCHORS}


def run_fold_h(yr, h, prep, rows, unit96, scales, rec, scenarios, check):
    """All adaptive cells of one fold and horizon. rec {scenario: (npz, select json)}."""
    P = prep["test"]
    grid, pav, day = P["grid"], P["pav"], P["day"]
    nd = int(day.max()) + 1
    cells, daily, paths = [], {}, {}
    pol_C = {}
    for pol in POLICIES:
        ix = [i for i, r in enumerate(rows) if r[0] == pol]
        lv = np.array([rows[i][1] for i in ix])
        o = np.argsort(lv)
        pol_C[pol] = (lv[o], P["C"][np.array(ix)[o]])
        fin = pol_C[pol][0] < 1.0 if pol != "frac_pv" else np.ones(len(o), bool)
        ok, bad = causal_ok(pol_C[pol][1][fin], grid, h)
        check(f"causal_f{yr}_h{h}_{pol}", ok, f"({bad} steps issued before their day)")
    for sname in scenarios:
        sc = S6.SCENARIOS[sname]
        z, sel = rec[sname]
        rmap, amap = record_rows(z)
        L = S6.load_for(grid, unit96, scales, sc)
        sim_kw = dict(ratings=LM.fleet_for_peak(L.max()), min_load=sc["min_load"], fuel=sc["fuel"],
                      battery=sc["battery"])
        abar_load = h2_mean_daily_load(prep["H2"], unit96, scales, sc)
        load_d = z[f"{sname}__load_kWh"][amap["handbook"]].astype(float)
        rec_ue = {}
        specs = []
        for pol in POLICIES:
            levels = pol_C[pol][0]
            rix = np.array([rmap[(pol, round(float(x), 6))] for x in levels])
            rec_ue[pol] = z[f"{sname}__ue_kWh"][rix].astype(float)
            t0 = float(sel["policies"][pol]["A_R"]["param"])
            assert np.any(np.abs(levels - t0) < LEVEL_TOL), f"A-R level {t0} of {pol} not in the set"
            lo = float(levels.min())
            specs += [(pol, g, kp, t0, lo) for g, kp in VARIANTS]
        t1 = time.time()
        r, idx, lat, passes, work = run_paths(L, pav, day, h, sim_kw, pol_C, specs, rec_ue, load_d, abar_load)
        #run_paths raises unless the fixed point is reached; the check records that it was reached
        # within the pass limit, and fails on a zero or exhausted pass count
        check(f"converged_f{yr}_h{h}_{sname}", 1 <= passes < 400, f"(fixed point after {passes} passes, {work} row-years, {time.time() - t1:.1f} s)")
        # scored load of each path equals the record's (load does not depend on the level)
        dl = r["daily"]["load_kWh"]
        check(f"load_identity_f{yr}_h{h}_{sname}", np.allclose(dl, load_d[None, :], rtol=1e-5, atol=1e-3))
        hb = amap["handbook"]
        hb_fuel = float(z[f"{sname}__fuel_L"][hb].astype(float).sum())
        tot = r["total"]
        for k, (pol, g, kp, t0, lo) in enumerate(specs):
            levels = pol_C[pol][0]
            lv = levels[idx[k]]
            ar = rmap[(pol, round(t0, 6))]
            if g == 0.0:
                same = (np.array_equal(r["daily"]["fuel_L"][k].astype(np.float32), z[f"{sname}__fuel_L"][ar])
                        and np.array_equal(r["daily"]["ue_kWh"][k].astype(np.float32), z[f"{sname}__ue_kWh"][ar]))
                check(f"constant_equals_record_f{yr}_h{h}_{sname}_{pol}", same)
            lim = lv >= 1.0 - LEVEL_TOL
            if lim.any():
                check(f"limit_zero_ue_f{yr}_h{h}_{sname}_{pol}_g{g}_k{kp}",
                      float(np.max(r["daily"]["ue_kWh"][k][lim])) <= 1e-9)
            ident, U = bounds(r["daily"]["ue_kWh"][k], load_d, lat[k], g, kp, abar_load, t0)
            if g:
                check(f"bound_f{yr}_h{h}_{sname}_{pol}_g{g}_k{kp}",
                      ident and float(tot["ue_kWh"][k]) <= U + 1e-6,
                      "" if ident else "(summed-update identity fails)")
            fuel_ar = float(z[f"{sname}__fuel_L"][ar].astype(float).sum())
            ue_ar = float(z[f"{sname}__ue_kWh"][ar].astype(float).sum())
            cells.append(dict(fold=yr, h=h, scenario=sname, policy=pol, gamma=g, kappa=kp, theta0=t0,
                              fuel_L=float(tot["fuel_L"][k]), ue_kWh=float(tot["ue_kWh"][k]),
                              load_kWh=float(tot["load_kWh"][k]), ue_share=float(tot["ue_share"][k]),
                              fuel_AR=fuel_ar, ue_AR=ue_ar, ue_share_AR=ue_ar / float(tot["load_kWh"][k]),
                              fuel_handbook=hb_fuel, level_mean=float(lv.mean()),
                              days_at_lo=int((np.abs(lv - lo) < LEVEL_TOL).sum()), days_at_limit=int(lim.sum()),
                              level_changes=int((np.diff(idx[k]) != 0).sum()), bound_kWh=U, passes=passes,
                              primary=bool(g == GAMMA and kp == KAPPA)))
        daily[sname] = dict(fuel_L=r["daily"]["fuel_L"].astype(np.float32),
                            ue_kWh=r["daily"]["ue_kWh"].astype(np.float32))
        paths[sname] = np.vstack([pol_C[s[0]][0][idx[k]] for k, s in enumerate(specs)]).astype(np.float32)
    return cells, daily, paths, [(s[0], s[1], s[2]) for s in specs]


# ---------------------------------------------------------------- reading (D-094 success test)
def success(cells: pd.DataFrame) -> dict:
    """The D-094 success test on the primary scope (15 min, main and shapeB, primary variant)."""
    c = cells[(cells.h == H_PRIMARY) & cells.primary & cells.scenario.isin(SCEN_PRIMARY)]
    met_ad = c.ue_share <= TGT + 1e-15
    met_ar = c.ue_share_AR <= TGT + 1e-15
    iscp = c.policy.isin(CP)
    out = dict(R_AD_cp=int(met_ad[iscp].sum()), R_AR_cp=int(met_ar[iscp].sum()), n_cp=int(iscp.sum()),
               R_AD_all=int(met_ad.sum()), R_AR_all=int(met_ar.sum()), n_all=int(len(c)))
    dr1 = {}
    for rule, fu, ue in (("A-D", "fuel_L", "ue_share"), ("A-R", "fuel_AR", "ue_share_AR")):
        dr1[rule] = {}
        for pol in POLICIES:
            folds = 0
            for yr in FOLDS:
                s = c[(c.policy == pol) & (c.fold == yr)]
                if len(s) == len(SCEN_PRIMARY) and ((s[fu] < s.fuel_handbook) & (s[ue] <= TGT + 1e-15)).all():
                    folds += 1
            dr1[rule][pol] = folds
    out["DR1_folds"] = dr1
    nA = sum(dr1["A-D"][p] >= FOLD_MIN for p in CP)
    nR = sum(dr1["A-R"][p] >= FOLD_MIN for p in CP)
    out.update(DR1D_cp_policies=nA, DR1_cp_policies=nR)
    if out["R_AD_cp"] > out["R_AR_cp"] and nA >= nR:
        out["claim"] = "a"
    elif out["R_AD_cp"] > out["R_AR_cp"]:
        out["claim"] = "b"
    else:
        out["claim"] = "c"
    return out


def boot_rows(cells, daily_all, specs_all, rec_all, B=B_DEF):
    """Paired day-block bootstrap intervals (D-091 draws) of A-D minus A-R and minus the
    handbook rule, test fuel, per primary-variant cell; conditional on the realized path.
    """
    out = []
    for (yr, h, sname), dd in daily_all.items():
        z = rec_all[(yr, h, sname)]
        rmap, amap = record_rows(z)
        nd = dd["fuel_L"].shape[1]
        Cn = draw_counts(nd, B)
        hb = z[f"{sname}__fuel_L"][amap["handbook"]].astype(float)
        for k, (pol, g, kp) in enumerate(specs_all[(yr, h)]):
            if not (g == GAMMA and kp == KAPPA):
                continue
            row = cells[(cells.fold == yr) & (cells.h == h) & (cells.scenario == sname) & (cells.policy == pol)
                        & cells.primary].iloc[0]
            ar = z[f"{sname}__fuel_L"][rmap[(pol, round(row.theta0, 6))]].astype(float)
            f = dd["fuel_L"][k].astype(float)
            d1, d2 = Cn @ (f - ar), Cn @ (f - hb)
            out.append(dict(fold=yr, h=h, scenario=sname, policy=pol, d_fuel_AR=float((f - ar).sum()),
                            d_fuel_AR_ci=ci(d1), d_fuel_handbook=float((f - hb).sum()), d_fuel_handbook_ci=ci(d2)))
    return pd.DataFrame(out)


# ---------------------------------------------------------------- self-tests
def selftest(verbose=True) -> dict:
    res = {}

    def ok(name, cond, msg=""):
        res[name] = bool(cond)
        if verbose:
            print(f"  [{'PASS' if cond else 'FAIL'}] {name} {msg}")

    rng = np.random.default_rng(11)
    levels = np.array([0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0])
    ok("level_rounds_up", level_index(levels, 0.61) == 2 and level_index(levels, 0.6) == 1
       and level_index(levels, 1.3) == 7 and level_index(levels, 0.2) == 0)
    # controller: gamma 0 constant; causal; floor; identity
    nd = 60
    ue = rng.exponential(1.0, nd) * (rng.random(nd) < 0.2)
    load = rng.uniform(900, 1100, nd) * 1e3
    idx0, _ = control(ue, load, 0.9, 0.0, 0.5, 1e6, levels, 0.5)
    ok("gamma0_constant", (levels[idx0] == 0.9).all())
    i1, lat1 = control(ue, load, 0.9, 0.005, 0.5, 1e6, levels, 0.5)
    ue2 = ue.copy(); ue2[30:] += 50.0
    i2, _ = control(ue2, load, 0.9, 0.005, 0.5, 1e6, levels, 0.5)
    ok("controller_causal", np.array_equal(i1[:31], i2[:31]) and not np.array_equal(i1, i2))
    ok("floor_held", lat1.min() >= 0.5 - 1e-12)
    ident, U = bounds(ue, load, lat1, 0.005, 0.5, 1e6, 0.9)
    ok("identity_holds", ident and ue.sum() <= U)
    # simulation on the toy grid: gamma 0 path equals the fixed row; fixed point is self-consistent
    grid, L, pav, cs = LM._toy(3, n_days=12)
    day = LM.day_ids(grid)
    C = LM._counted(pav, cs, np.random.default_rng(5), K=5)
    C = np.vstack([C, np.zeros(len(grid))])  # the limit level: counts no PV
    lv = np.array([0.6, 0.7, 0.8, 0.9, 0.95, 1.0])  # C rows ordered from least to most reserve
    C = C[[4, 3, 2, 1, 0, 5]]  # _counted: row 4 counts the most PV
    for batt in (False, True):
        kw = dict(battery=batt, batt_p=150.0, batt_e=80.0) if batt else dict(battery=False)
        fixed = LM.simulate(L, pav, C, day, 3, **kw)
        load_d = fixed["daily"]["load_kWh"][0]
        abar_load = float(load_d.mean())
        pol_C = {"p": (lv, C)}
        specs = [("p", 0.0, 0.5, 0.8, 0.6), ("p", 0.05, 0.5, 0.8, 0.6), ("p", 0.2, 1.0, 0.95, 0.6)]
        rec_ue = {"p": fixed["daily"]["ue_kWh"]}
        r, idx, lat, passes, _ = run_paths(L, pav, day, 3, kw, pol_C, specs, rec_ue, load_d, abar_load)
        ok(f"toy_gamma0_equals_fixed_{'batt' if batt else 'nobatt'}",
           np.array_equal(r["daily"]["fuel_L"][0], fixed["daily"]["fuel_L"][2])
           and np.array_equal(r["daily"]["ue_kWh"][0], fixed["daily"]["ue_kWh"][2]))
        cons = all(np.array_equal(control(r["daily"]["ue_kWh"][k], load_d, s[3], s[1], s[2], abar_load, lv, s[4])[0],
                                  idx[k]) for k, s in enumerate(specs))
        ok(f"toy_fixed_point_consistent_{'batt' if batt else 'nobatt'}", cons, f"({passes} passes)")
        # the path simulated alone equals its row in the batch (rows independent)
        Cp = path_row(C, idx[1], day)[None, :]
        solo = LM.simulate(L, pav, Cp, day, 3, **kw)
        ok(f"toy_path_row_independent_{'batt' if batt else 'nobatt'}",
           np.array_equal(solo["daily"]["fuel_L"][0], r["daily"]["fuel_L"][1]))
        tag = 'batt' if batt else 'nobatt'
        ok(f"toy_has_ue_and_level_changes_{tag}", float(fixed["daily"]["ue_kWh"][:5].sum()) > 1.0
           and int((np.diff(idx[1:], axis=1) != 0).sum()) > 0)
        wrong = {"p": np.zeros_like(fixed["daily"]["ue_kWh"])}  # a first guess with no UE at all
        r2, idx2, _, passes2, _ = run_paths(L, pav, day, 3, kw, pol_C, specs, wrong, load_d, abar_load)
        ok(f"toy_fixed_point_unique_{tag}", np.array_equal(idx2, idx)
           and np.array_equal(r2["daily"]["fuel_L"], r["daily"]["fuel_L"]), f"({passes2} passes from a wrong guess)")
        lim = lv[idx] >= 1.0
        ok(f"toy_limit_zero_ue_{'batt' if batt else 'nobatt'}",
           (not lim.any()) or float(np.max(r["daily"]["ue_kWh"][lim])) <= 1e-9)
    # causality check catches a pre-midnight difference
    g = pd.date_range("2023-06-01", periods=2 * 288, freq="5min")
    Cc = np.zeros((2, len(g))); Cc[1, 300] = 5.0
    ok("causal_ok_daylight", causal_ok(Cc, g, 3)[0])
    Cc[1, 289] = 5.0  # 00:05 target, issued 23:50 the day before
    ok("causal_flags_midnight", not causal_ok(Cc, g, 3)[0])
    # success test logic on a constructed table
    rows = []
    for yr in FOLDS:
        for sn in SCEN_PRIMARY:
            for pol in POLICIES:
                rows.append(dict(fold=yr, h=H_PRIMARY, scenario=sn, policy=pol, primary=True, fuel_L=90.0,
                                 fuel_AR=95.0, fuel_handbook=100.0, ue_share=TGT / 2,
                                 ue_share_AR=TGT * (2 if yr < 2021 else 0.5)))
    s = success(pd.DataFrame(rows))
    ok("success_logic", s["R_AD_cp"] == 48 and s["R_AR_cp"] == 32 and s["claim"] == "a"
       and s["DR1_folds"]["A-R"]["cqr"] == 4)
    return res


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--root", default=".")
    ap.add_argument("--ext", action="store_true", help="extended theta set (D-092): S2 + S2X caches")
    ap.add_argument("--s6", default="S6E")
    ap.add_argument("--s6b", default="S6E")
    ap.add_argument("--out", default="S51")
    ap.add_argument("--only", default=None, help="year:h[,year:h]")
    ap.add_argument("--primary-only", action="store_true")
    ap.add_argument("--B", type=int, default=B_DEF)
    a = ap.parse_args()
    if a.selftest:
        r = selftest()
        sys.exit(0 if all(r.values()) else 1)
    t0 = time.time()
    R = Path(a.root) / "04_results" / "ce1"
    out, runs = R / a.out, R / "_runs"
    out.mkdir(parents=True, exist_ok=True); runs.mkdir(parents=True, exist_ok=True)
    checks = {}

    def check(name, cond, msg=""):
        checks[name] = dict(ok=bool(cond), msg=msg)
        if not cond or not any(name.startswith(p) for p in ("causal_", "constant_", "limit_", "bound_")):
            log(f"{'PASS' if cond else 'FAIL'}  {name}  {msg}")

    st = selftest(verbose=False)
    check("selftests", all(st.values()), f"({sum(st.values())}/{len(st)})")
    geo, m, unit96, scales, _ = S6.load_inputs(R / "S1", R / "S1M", R / "S1L")
    if a.ext:
        D.extend_theta_low()
        load = lambda yr, h: FC.load_cache_ext(R / "S2" / f"yulara_f{yr}_h{h}.parquet",  # noqa: E731
                                                R / "S2X" / f"yulara_f{yr}_h{h}.parquet")
    else:
        load = lambda yr, h: FC.load_cache(R / "S2" / f"yulara_f{yr}_h{h}.parquet")  # noqa: E731
    rows = S6.row_table()
    jobs = [(y, H_PRIMARY) for y in FOLDS] + ([] if a.primary_only else [(y, h) for h in H_SECONDARY for y in FOLDS])
    if a.only:
        jobs = [(int(p.split(":")[0]), int(p.split(":")[1])) for p in a.only.split(",")]
    inputs = {}
    all_cells, daily_all, specs_all, rec_all = [], {}, {}, {}
    for yr, h in jobs:
        t1 = time.time()
        scen = list(SCEN_PRIMARY) if a.primary_only else list(SCEN_PRIMARY) + list(SCEN_BATT)
        rec = {}
        for sn in scen:
            d = R / (a.s6b if sn in SCEN_BATT else a.s6)
            fz, fs = d / f"f{yr}_h{h}_daily.npz", d / f"f{yr}_h{h}_select.json"
            inputs[str(fz.relative_to(R))] = md5(fz); inputs[str(fs.relative_to(R))] = md5(fs)
            rec[sn] = (np.load(fz), json.load(open(fs))["selections"][sn])
            rec_all[(yr, h, sn)] = rec[sn][0]
        cache = load(yr, h)
        prep = S6.prepare(cache, h, yr, geo, m, rows)
        cells, daily, paths, specs = run_fold_h(yr, h, prep, rows, unit96, scales, rec, scen, check)
        all_cells += cells
        specs_all[(yr, h)] = specs
        for sn in scen:
            daily_all[(yr, h, sn)] = daily[sn]
        np.savez_compressed(out / f"f{yr}_h{h}_paths.npz", specs=np.array([f"{p}|{g}|{k}" for p, g, k in specs]),
                            **{f"{sn}__level": paths[sn] for sn in scen},
                            **{f"{sn}__{k}": daily[sn][k] for sn in scen for k in daily[sn]})
        log(f"fold {yr} h={h}: {len(cells)} cells, {time.time() - t1:.0f} s")
    cells = pd.DataFrame(all_cells)
    cells.to_csv(out / "cells.csv", index=False)
    bt = boot_rows(cells, {k: v for k, v in daily_all.items() if k[2] in SCEN_PRIMARY or k[2] in SCEN_BATT},
                   specs_all, rec_all, B=a.B)
    bt.to_csv(out / "boot.csv", index=False)
    summ = success(cells) if any(h == H_PRIMARY for _, h in jobs) else {}
    json.dump(summ, open(out / "success.json", "w"), indent=1, default=int)
    ok = all(c["ok"] for c in checks.values())
    nfail = sum(not c["ok"] for c in checks.values())
    log(f"{len(checks)} checks, {nfail} failed; {time.time() - t0:.0f} s")
    outs = {p.name: md5(p) for p in sorted(out.glob("*")) if p.is_file()}
    now = time.strftime("%Y-%m-%dT%H:%M:%S")
    stage = a.out
    json.dump(dict(stage=stage, ok=ok, error="" if ok else f"{nfail} checks failed", finished=now,
                   seconds=round(time.time() - t0, 1), inputs=inputs, outputs=outs,
                   code={"ce1_adaptive.py": md5(Path(__file__))},
                   env=dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__)), open(runs / f"{stage}.json", "w"), indent=2)
    json.dump(dict(stage=stage, checks=checks, n_checks=len(checks), outputs=outs, inputs=inputs,
                   code={"ce1_adaptive.py": md5(Path(__file__))}), open(runs / f"{stage}.detail.json", "w"),
              indent=1, default=float)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
