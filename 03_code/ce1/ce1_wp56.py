"""Stage S56: dispatch-realism sensitivities (rule D-112): contingency reserve for the largest
online unit, doubled no-load fuel, and minimum loading of 30% and 50%, every policy and level,
15 min, six folds.

Usage (from the repository root unless the path starts at ce1/, then from 03_code):
    python ce1/ce1_wp56.py --stage S56 [--smoke]
    python ce1/ce1_wp56.py --selftest
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
import ce1_load as LM  # noqa: E402

FOLDS = [2019, 2020, 2021, 2022, 2023, 2024]
H = 3
NL = LM.FUEL["cat_c32"]
LM.FUEL.setdefault("cat_c32_nl2", (2 * NL[0], NL[1]))  # D-112, literal x2
LM.FUEL.setdefault("cat_c32_nl2eq", (2 * NL[0], NL[1] - NL[0]))  # D-112, full-load fuel kept
SCEN56 = {
    "cont":  dict(shape="A", scale="primary", min_load=LM.MIN_LOAD, battery=False, fuel="cat_c32", contingency=True),
    "contB": dict(shape="B", scale="primary", min_load=LM.MIN_LOAD, battery=False, fuel="cat_c32", contingency=True),
    "nl2":   dict(shape="A", scale="primary", min_load=LM.MIN_LOAD, battery=False, fuel="cat_c32_nl2"),
    "nl2eq": dict(shape="A", scale="primary", min_load=LM.MIN_LOAD, battery=False, fuel="cat_c32_nl2eq"),
    "min30": dict(shape="A", scale="primary", min_load=0.30, battery=False, fuel="cat_c32"),
    "min50": dict(shape="A", scale="primary", min_load=0.50, battery=False, fuel="cat_c32"),
}
BASE = {"cont": "main", "contB": "shapeB", "nl2": "main", "nl2eq": "main", "min30": "main", "min50": "main"}
CHUNK = 64
REC_ROWS = ("handbook", "oracle")  # rows recorded step by step for the N-1 check


def md5(p: Path) -> str:
    h = hashlib.md5()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def log(m):
    print(time.strftime("%H:%M:%S"), f"[S56] {m}", flush=True)


# ================================================================ contingency term
def combo_table(ratings):
    """Capacity and capacity minus largest unit of every count vector of ce1_load.Fleet, in
    the Fleet's own order (capacities strictly increasing is asserted: then the first set
    with capacity >= k is the set of capacity k).
    """
    fl = LM.Fleet(ratings)
    cls = np.asarray(fl.classes, float)
    largest = np.array([cls[np.flatnonzero(c)].max() for c in fl.counts])
    assert (np.diff(fl.cap) > 0).all(), "two unit sets share a capacity: k_min would be ambiguous"
    return fl.cap.astype(float), (fl.cap - largest).astype(float), float(fl.smallest)


def k_min(L, pc, phat, ratings, contingency=True):
    """Smallest feasible online capacity per row and step, and whether the PV term binds.
    L (n), pc (K, n) counted PV (NaN = no forecast row: 0, the handbook fallback),
    phat (n) available PV of the point forecast (NaN = no forecast row: 0).
    """
    cap, capn1, small = combo_table(ratings)
    k1 = np.maximum(L[None, :] - np.nan_to_num(pc), small)
    k2 = (L - np.nan_to_num(phat)) if contingency else np.full(len(L), -np.inf)
    ok2 = capn1[:, None] >= k2[None, :] - 1e-9  # (n_sets, n)
    out = np.full(k1.shape, cap[-1])
    first = np.full(k1.shape, cap[-1])
    feas = np.zeros(k1.shape, bool)
    for j in range(len(cap) - 1, -1, -1):  # descending: last write = smallest
        m1 = cap[j] >= k1 - 1e-9
        first = np.where(m1, cap[j], first)
        m = m1 & ok2[j][None, :]
        out = np.where(m, cap[j], out)
        feas |= m
    binds = out == first  # contingency adds nothing
    return out, binds, feas


def effective_counted(L, C, phat, ratings, contingency=True):
    """Counted PV that makes ce1_load.simulate request k_min (chunks of rows), plus per-row
    step masks for the binding share and infeasible steps.
    """
    K, n = C.shape
    Ce = np.empty((K, n))
    binds = np.zeros((K, n), bool)
    infeas = np.zeros((K, n), bool)
    for a in range(0, K, CHUNK):
        km, b, f = k_min(L, C[a:a + CHUNK], phat, ratings, contingency)
        Ce[a:a + CHUNK] = L[None, :] - km
        binds[a:a + CHUNK] = b
        infeas[a:a + CHUNK] = ~f
    return Ce, binds, infeas


# ---------------------------------------------------------------- v2.1: the N-1 constructions inside simulate (D-161)
def n1_registered() -> bool:
    """True when simulate's defaults are the registered v2.0.0 simulator: then the N-1 constructions use the
    registered transform (counted PV = load - k_min). Under v2.1 they run inside simulate (n1=..., p_hat).
    """
    return (LM.DEFAULT_COMMIT, LM.DEFAULT_GAP) == ("greedy", "hold")


def n1_infeasible(constr, L, C, phat, ratings):
    """Steps where no unit set meets the construction (the whole fleet dominates every set in capacity and in
    capacity minus the largest unit, so feasibility is the whole fleet's). As ce1_wp87._infeasible.
    """
    r = np.asarray(ratings, float)
    tot, n1 = r.sum(), r.sum() - r.max()
    pcz = np.nan_to_num(C)
    k1 = np.maximum(L[None, :] - pcz, r.min())
    need = (L[None, :] - pcz) if constr == "n1a" else np.broadcast_to(L - np.nan_to_num(phat), C.shape)
    return (k1 > tot + 1e-9) | (need > n1 + 1e-9)


def n1_setup(constr, L, C, phat, ratings):
    """(counted PV passed to simulate, infeasible-step mask, extra simulate keywords) of a construction:
    plain -> (C, none, {}); registered simulator -> the registered transform, {}; v2.1 -> C unchanged and
    n1=constr (p_hat = the point forecast's available PV for s56).
    """
    if constr in (None, "plain"):
        return C, np.zeros(C.shape, bool), {}
    assert constr in ("s56", "n1a"), constr
    if n1_registered():
        if constr == "s56":
            Ce, _, inf_ = effective_counted(L, C, phat, ratings, True)
            return Ce, inf_, {}
        import ce1_s69 as S69
        return S69.effective_n1a(L, C, ratings) + ({},)
    kw = dict(n1=constr)
    if constr == "s56":
        kw["p_hat"] = np.asarray(phat, float)
    return C, n1_infeasible(constr, L, C, phat, ratings), kw


def fallback_counts(C, pav):
    """Daytime scored steps without a forecast row, per row (what simulate counts)."""
    sc = np.isfinite(pav) & (pav > 0)
    return (np.isnan(C) & sc[None, :]).sum(axis=1)


# ================================================================ one fold
def run_fold(yr, R, geo, m, unit96, scales, rows, scen, smoke, check):
    import ce1_adaptive as A
    import ce1_folds as FC
    import ce1_s6 as S6
    s2 = R / "S2" / f"yulara_f{yr}_h{H}.parquet"
    s2x = R / "S2X" / f"yulara_f{yr}_h{H}.parquet"
    cache = FC.load_cache_ext(s2, s2x)
    prep = S6.prepare(cache, H, yr, geo, m, rows)
    det = [i for i, r in enumerate(rows) if r[0] == "deterministic"][0]
    hb = [i for i, r in enumerate(rows) if r[0] == "handbook"][0]
    orc = [i for i, r in enumerate(rows) if r[0] == "oracle"][0]
    s6e = pd.read_parquet(R / "S6E" / f"f{yr}_h{H}_totals.parquet")
    tables, sel, cells = [], {}, []
    for sname, sc in scen.items():
        res, ce_test, n1kw_test = {}, None, {}
        for part in ("H2", "test"):
            P = prep[part]
            L = S6.load_for(P["grid"], unit96, scales, sc)
            ratings = LM.fleet_for_peak(L.max())
            sim_kw = dict(ratings=ratings, min_load=sc["min_load"], fuel=sc["fuel"], battery=False)
            fb = fallback_counts(P["C"], P["pav"])
            if sc.get("contingency"):
                phat = P["C"][det]
                _, binds, infeas_t = effective_counted(L, P["C"], phat, ratings, True)
                Ce, infeas, n1kw = n1_setup("s56", L, P["C"], phat, ratings)
                check(f"n1_infeasible_equals_transform_f{yr}_{sname}_{part}", np.array_equal(infeas, infeas_t))
                dayt = np.isfinite(P["pav"]) & (P["pav"] > 0)
                share = binds[:, dayt].mean(axis=1)
                ninf = infeas[:, np.isfinite(P["pav"])].sum(axis=1)
                # identity: the same machinery without the contingency term is the plain run
                if part == "test":
                    pr = [hb, det, orc] + [i for i, r in enumerate(rows) if (r[0], r[1]) in set(S6.PROBES)]
                    C0, _, _ = effective_counted(L, P["C"][pr], phat, ratings, False)
                    a0 = LM.simulate(L, P["pav"], C0, P["day"], H, **sim_kw)["total"]
                    b0 = LM.simulate(L, P["pav"], P["C"][pr], P["day"], H, **sim_kw)["total"]
                    check(f"machinery_without_term_equals_plain_f{yr}_{sname}",
                          all(np.array_equal(a0[k], b0[k]) for k in ("fuel_L", "ue_kWh", "starts", "curt_kWh")))
            else:
                Ce, share, ninf, n1kw = P["C"], np.full(len(rows), np.nan), np.zeros(len(rows), int), {}
            if part == "test" and sc.get("contingency"):
                ce_test, n1kw_test = Ce, n1kw
            r = LM.simulate(L, P["pav"], Ce, P["day"], H, **sim_kw, **n1kw)
            tot = r["total"]
            tot["fallback_steps"] = fb.astype(float)
            res[part] = r
            t = pd.DataFrame({k: tot[k] for k in S6.TOT_KEYS})
            t.insert(0, "param", [x[1] for x in rows])
            t.insert(0, "policy", [x[0] for x in rows])
            t.insert(0, "set", part)
            t.insert(0, "scenario", sname)
            t["missing_steps"] = r["missing_steps"]
            t["pv_binds_share"] = share
            t["n1_infeasible_steps"] = ninf
            tables.append(t)
            check(f"handbook_zero_ue_f{yr}_{sname}_{part}", float(tot["ue_kWh"][hb]) == 0.0)
            check(f"oracle_fuel_le_handbook_f{yr}_{sname}_{part}",
                  float(tot["fuel_L"][orc]) <= float(tot["fuel_L"][hb]) + 1e-6)
            if sc.get("contingency") and part == "test":
                # N-1 holds on every recorded step where a feasible set exists
                rr = LM.simulate(L, P["pav"], Ce[[hb, orc]], P["day"], H, record=True, **sim_kw, **n1kw)
                fleet = np.asarray(ratings, float)
                on = rr["rec_on"]  # (2, n, U)
                lg = np.where(on, fleet[None, None, :], 0).max(axis=2)
                KU = rr["rec"]["KU"]
                phat0 = np.nan_to_num(P["C"][det])
                sc_m = np.isfinite(P["pav"])
                need = (L - phat0)[None, :]
                viol = (KU - lg < need - 1e-6) & sc_m[None, :] & ~infeas[[hb, orc]]
                check(f"n1_holds_on_recorded_steps_f{yr}_{sname}", int(viol.sum()) == 0,
                      f"({int(viol.sum())} violations, {int(infeas[[hb, orc]][:, sc_m].sum())} infeasible steps)")
        # the base configuration of this scenario reproduces S6E where nothing changed
        totH2, totT = res["H2"]["total"], res["test"]["total"]
        s = dict(scenario={k: v for k, v in sc.items()}, base=BASE[sname],
                 anchors={a: {k: float(totT[k][i]) for k in S6.TOT_KEYS} for i, a in enumerate(S6.ANCHORS)},
                 policies={p: S6.select_policy(totH2, totT, rows, p) for p in D.ALL_POLICIES})
        sel[sname] = s
        # ---- A-D, primary and constant variants, from this scenario's own record
        P = prep["test"]
        Ct = ce_test if ce_test is not None else P["C"]
        cells += ad_block(yr, sname, sc, prep, rows, Ct, res["test"], s, unit96, scales, check, hb, orc, n1kw_test)
        hb_fuel = float(res["test"]["total"]["fuel_L"][hb])
        totT = res["test"]["total"]
        log(f"fold {yr} {sname}: handbook {hb_fuel:,.0f} L, oracle ceiling "
            f"{100 * (1 - float(totT['fuel_L'][orc]) / hb_fuel):.2f}%")
    cells += ad_main_reference(yr, R, prep, rows, unit96, scales, check, hb, orc).to_dict("records")
    # probe: a scenario identical to main on the probe rows reproduces S6E (the prep is the record's)
    P = prep["test"]
    sc = dict(shape="A", scale="primary", min_load=LM.MIN_LOAD, battery=False, fuel="cat_c32")
    L = S6.load_for(P["grid"], unit96, scales, sc)
    pr = [i for i, r in enumerate(rows) if (r[0], r[1]) in set(S6.PROBES) or r[0] in S6.ANCHORS]
    t = LM.simulate(L, P["pav"], P["C"][pr], P["day"], H, ratings=LM.fleet_for_peak(L.max()),
                    min_load=sc["min_load"], fuel=sc["fuel"], battery=False)["total"]
    ref = s6e[(s6e.scenario == "main") & (s6e.set == "test")]
    worst = 0.0
    for j, i in enumerate(pr):
        pol, par = rows[i]
        rr = ref[(ref.policy == pol) & ((ref.param.isna() & (par != par)) | np.isclose(ref.param, par if par == par else -9))]
        for k in ("fuel_L", "ue_kWh", "starts"):
            worst = max(worst, abs(float(t[k][j]) - float(rr[k].iloc[0])) / max(abs(float(rr[k].iloc[0])), 1.0))
    check(f"probe_main_reproduces_S6E_f{yr}", worst < 1e-9, f"({len(pr)} rows, worst {worst:.2e})")
    return pd.concat(tables, ignore_index=True), sel, cells


def ad_block(yr, sname, sc, prep, rows, Ct, rtest, s, unit96, scales, check, hb, orc, n1kw=None):
    """A-D (primary) and the constant variant for one scenario, from its own test-year record
    (rtest: simulate output of every row) and its own A-R levels (s: select record).
    """
    import ce1_adaptive as A
    import ce1_s6 as S6
    P = prep["test"]
    L = S6.load_for(P["grid"], unit96, scales, sc)
    ratings = LM.fleet_for_peak(L.max())
    sim_kw = dict(ratings=ratings, min_load=sc["min_load"], fuel=sc["fuel"], battery=False, **(n1kw or {}))
    daily, totT = rtest["daily"], rtest["total"]
    load_d = daily["load_kWh"][hb]
    abar = A.h2_mean_daily_load(prep["H2"], unit96, scales, sc)
    pol_C, rec_ue, specs = {}, {}, []
    for pol in A.POLICIES:
        ix = [i for i, r in enumerate(rows) if r[0] == pol]
        lv = np.array([rows[i][1] for i in ix])
        o = np.argsort(lv)
        ixo = np.array(ix)[o]
        pol_C[pol] = (lv[o], Ct[ixo])
        rec_ue[pol] = daily["ue_kWh"][ixo]
        t0 = float(s["policies"][pol]["A_R"]["param"])
        lo = float(lv.min())
        specs += [(pol, 0.0, A.KAPPA, t0, lo), (pol, A.GAMMA, A.KAPPA, t0, lo)]
    rp, idx, lat, passes, work = A.run_paths(L, P["pav"], P["day"], H, sim_kw, pol_C, specs, rec_ue, load_d, abar)
    check(f"ad_converged_f{yr}_{sname}", 1 <= passes < 400 and bool(np.isfinite(rp["total"]["fuel_L"]).all()),
          f"({passes} passes, {work} row-years)")
    hb_fuel = float(totT["fuel_L"][hb])
    cells = []
    for k, (pol, g, kp, t0, lo) in enumerate(specs):
        ar = [i for i, r in enumerate(rows) if r[0] == pol and abs(r[1] - t0) < 1e-9][0]
        if g == 0.0:
            check(f"ad_constant_equals_AR_f{yr}_{sname}_{pol}",
                  np.array_equal(rp["daily"]["fuel_L"][k], daily["fuel_L"][ar])
                  and np.array_equal(rp["daily"]["ue_kWh"][k], daily["ue_kWh"][ar]))
        cells.append(dict(fold=yr, h=H, scenario=sname, policy=pol, gamma=g, kappa=kp, theta0=t0,
                          fuel_L=float(rp["total"]["fuel_L"][k]), ue_kWh=float(rp["total"]["ue_kWh"][k]),
                          load_kWh=float(rp["total"]["load_kWh"][k]), ue_share=float(rp["total"]["ue_share"][k]),
                          starts=float(rp["total"]["starts"][k]), fuel_AR=float(totT["fuel_L"][ar]),
                          ue_share_AR=float(totT["ue_share"][ar]), fuel_handbook=hb_fuel,
                          fuel_oracle=float(totT["fuel_L"][orc]), passes=passes,
                          primary=bool(g == A.GAMMA and kp == A.KAPPA)))
    return cells


def ad_main_reference(yr, R, prep, rows, unit96, scales, check, hb, orc):
    import ce1_s6 as S6
    sc = S6.SCENARIOS["main"]
    P = prep["test"]
    L = S6.load_for(P["grid"], unit96, scales, sc)
    r = LM.simulate(L, P["pav"], P["C"], P["day"], H, ratings=LM.fleet_for_peak(L.max()),
                    min_load=sc["min_load"], fuel=sc["fuel"], battery=False)
    sel = json.load(open(R / "S6E" / f"f{yr}_h{H}_select.json"))["selections"]["main"]
    c = pd.DataFrame(ad_block(yr, "main_ref", sc, prep, rows, P["C"], r, sel, unit96, scales, check, hb, orc))
    ref = pd.read_csv(R / "S51" / "cells.csv")
    ref = ref[(ref.fold == yr) & (ref.h == H) & (ref.scenario == "main") & ref.primary]
    mm = c[c.primary].merge(ref, on="policy", suffixes=("", "_s51"))
    worst = float(np.max(np.abs(mm.fuel_L / mm.fuel_L_s51 - 1))) if len(mm) else np.inf
    worst_ue = float(np.max(np.abs(mm.ue_kWh - mm.ue_kWh_s51))) if len(mm) else np.inf
    check(f"ad_wiring_reproduces_S51_main_f{yr}", len(mm) == int(c.primary.sum()) and worst < 1e-7 and worst_ue < 1e-3,
          f"({len(mm)} policies, worst relative fuel {worst:.1e}, worst UE {worst_ue:.1e} kWh)")
    return c



# ================================================================ self-test
def selftest(verbose=True):
    """Registered self-tests under the v2.0.0 simulator; the v2.1 tests set their own simulator (D-161)."""
    with LM.registered_v200():
        return _selftest(verbose)


def _selftest(verbose=True):
    res = {}

    def ok(name, cond, msg=""):
        res[name] = bool(cond)
        if verbose:
            print(f"  [{'PASS' if cond else 'FAIL'}] {name} {msg}")

    for ratings in (LM.UNITS_BASE, LM.UNITS_BASE + LM.UNITS_EXTRA):
        cap, capn1, small = combo_table(ratings)
        ok(f"caps_distinct_{len(ratings)}_units", (np.diff(cap) > 0).all(), f"({len(cap)} sets)")
    cap, capn1, small = combo_table(LM.UNITS_BASE)
    # brute force over every set for random loads
    rng = np.random.default_rng(3)
    n = 4000
    L = rng.uniform(1500, 6400, n)
    phat = np.where(rng.random(n) < 0.1, np.nan, rng.uniform(0, 1800, n))
    pc = np.vstack([np.where(rng.random(n) < 0.05, np.nan, np.nan_to_num(phat) * f) for f in (0.0, 0.5, 1.0)])
    km, binds, feas = k_min(L, pc, phat, LM.UNITS_BASE, True)
    bf = np.full(pc.shape, cap[-1]); bff = np.zeros(pc.shape, bool); first = np.full(pc.shape, np.inf)
    for kk in range(pc.shape[0]):
        for i in range(n):
            k1 = max(L[i] - (0 if np.isnan(pc[kk, i]) else pc[kk, i]), small)
            k2 = L[i] - (0 if np.isnan(phat[i]) else phat[i])
            c = [cap[j] for j in range(len(cap)) if cap[j] >= k1 - 1e-9 and capn1[j] >= k2 - 1e-9]
            if c:
                bf[kk, i] = min(c); bff[kk, i] = True
            first[kk, i] = min([cap[j] for j in range(len(cap)) if cap[j] >= k1 - 1e-9] or [cap[-1]])
    ok("k_min_equals_brute_force", np.array_equal(km, bf) and np.array_equal(feas, bff))
    ok("binds_means_no_extra_capacity", np.array_equal(binds, km == first))
    km0, b0, f0 = k_min(L, pc, phat, LM.UNITS_BASE, False)
    ok("without_term_k_min_is_first_set", np.array_equal(km0, first) and b0.all() and f0.all())
    ok("contingency_never_lowers_capacity", (km >= km0).all())
    # simulate requests exactly k_min through the effective counted PV
    grid, Lt, pav, cs = LM._toy(1)
    dd = LM.day_ids(grid)
    rng = np.random.default_rng(5)
    C = LM._counted(pav, cs, rng)
    ph = np.nan_to_num(pav) * 0.9
    Ce, bnd, inf_ = effective_counted(Lt, C, ph, LM.UNITS_BASE, True)
    r = LM.simulate(Lt, pav, Ce, dd, 3, record=True)
    km_t, _, feas_t = k_min(Lt, C, ph, LM.UNITS_BASE, True)
    sc = np.isfinite(pav)
    # the online capacity is at least k_min (minimum up time can only add units)
    ok("online_capacity_at_least_k_min", (r["rec"]["KU"][:, sc] >= km_t[:, sc] - 1e-6).all())
    fleet = np.asarray(LM.UNITS_BASE, float)
    lg = np.where(r["rec_on"], fleet[None, None, :], 0).max(axis=2)
    need = (Lt - ph)[None, :]
    viol = (r["rec"]["KU"] - lg < need - 1e-6) & sc[None, :] & feas_t
    ok("n1_holds_after_min_up_time", int(viol.sum()) == 0, f"({int(viol.sum())} violations)")
    # v2.1: the construction inside simulate (minimum feasible commitment, elapsed clocks): N-1 holds on every
    #    scored feasible step, the infeasible mask equals the transform's, online capacity >= k_min
    with LM.primary_v21():
        Ci, inf_i, kw = n1_setup("s56", Lt, C, ph, LM.UNITS_BASE)
        ri = LM.simulate(Lt, pav, Ci, dd, 3, record=True, **kw)
    ok("v21_n1_setup_is_internal", Ci is C and kw.get("n1") == "s56" and np.array_equal(inf_i, inf_))
    lgi = np.where(ri["rec_on"], fleet[None, None, :], 0).max(axis=2)
    vi = (ri["rec"]["KU"] - lgi < need - 1e-6) & sc[None, :] & feas_t
    ok("v21_n1_holds_internal", int(vi.sum()) == 0, f"({int(vi.sum())} violations)")
    ok("v21_online_capacity_at_least_k_min", (ri["rec"]["KU"][:, sc] >= km_t[:, sc] - 1e-6).all())
    with LM.registered_v200():
        Cr, inf_r, kwr = n1_setup("s56", Lt, C, ph, LM.UNITS_BASE)
    ok("registered_n1_setup_is_transform", kwr == {} and np.array_equal(Cr, Ce) and np.array_equal(inf_r, inf_))
    C0, _, _ = effective_counted(Lt, C, ph, LM.UNITS_BASE, False)
    a = LM.simulate(Lt, pav, C0, dd, 3)["total"]
    b = LM.simulate(Lt, pav, C, dd, 3)["total"]
    ok("machinery_without_term_equals_plain", all(np.array_equal(a[k], b[k]) for k in ("fuel_L", "ue_kWh", "starts")))
    a2 = LM.simulate(Lt, pav, C, dd, 3, fuel="cat_c32_nl2")["total"]
    a3 = LM.simulate(Lt, pav, C, dd, 3, fuel="cat_c32_nl2eq")["total"]
    extra = LM.FUEL["cat_c32"][0] * b["online_kW_h"]
    ok("nl2_adds_no_load_times_online_kWh", np.allclose(a2["fuel_L"] - b["fuel_L"], extra, rtol=1e-9))
    ok("nl2eq_identity", np.allclose(a3["fuel_L"] - b["fuel_L"], extra - LM.FUEL["cat_c32"][0] * b["thermal_kWh"], rtol=1e-9))
    return res


# ================================================================ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--root", default=None)
    ap.add_argument("--only", type=int, default=None)
    a = ap.parse_args()
    st = selftest()
    if a.selftest:
        sys.exit(0 if all(st.values()) else 1)
    if a.stage != "S56":
        ap.error("--stage S56 or --selftest")
    import ce1_s6 as S6
    root = Path(a.root) if a.root else HERE.parents[1]
    R = root / "04_results" / "ce1"
    out = R / ("S56_smoke" if a.smoke else "S56")
    runs = R / "_runs"
    out.mkdir(parents=True, exist_ok=True)
    runs.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    checks = {f"selftest_{k}": dict(ok=v, msg="") for k, v in st.items()}
    inputs, outputs = {}, {}

    def check(name, cond, msg=""):
        checks[name] = dict(ok=bool(cond), msg=msg)
        log(f"{'PASS' if cond else 'FAIL'}  {name}  {msg}")

    geo, m, unit96, scales, _ = S6.load_inputs(R / "S1", R / "S1M", R / "S1L")
    D.extend_theta_low()
    rows = S6.row_table()
    scen = SCEN56
    folds = FOLDS if a.only is None else [a.only]
    if a.smoke:
        keep = set(S6.ANCHORS) | {"icp", "mondrian_cqr"}
        rows = [r for r in rows if r[0] in keep]
        folds = [a.only or 2024]
    all_cells = []
    for yr in folds:
        f_tot, f_sel, f_cells = out / f"f{yr}_h{H}_totals.parquet", out / f"f{yr}_h{H}_select.json", out / f"f{yr}_h{H}_cells.csv"
        for nm in (f"S2/yulara_f{yr}_h{H}.parquet", f"S2X/yulara_f{yr}_h{H}.parquet", f"S6E/f{yr}_h{H}_totals.parquet"):
            inputs[nm] = md5(R / nm)
        if f_tot.exists() and f_sel.exists() and f_cells.exists() and not a.smoke:
            log(f"fold {yr}: exists, skipped")
            all_cells.append(pd.read_csv(f_cells))
        else:
            t1 = time.time()
            if a.smoke:
                A_pol = None
                import ce1_adaptive as A
                A_pol = A.POLICIES
                A.POLICIES = ["icp", "mondrian_cqr"]
                D_all = D.ALL_POLICIES
                D.ALL_POLICIES = ["icp", "mondrian_cqr"]
            tab, sel, cells = run_fold(yr, R, geo, m, unit96, scales, rows, scen, a.smoke, check)
            if a.smoke:
                A.POLICIES, D.ALL_POLICIES = A_pol, D_all
            tab.to_parquet(f_tot)
            json.dump(sel, open(f_sel, "w"), indent=1, default=float)
            pd.DataFrame(cells).to_csv(f_cells, index=False)
            all_cells.append(pd.DataFrame(cells))
            log(f"fold {yr}: {time.time() - t1:.0f} s")
        for f in (f_tot, f_sel, f_cells):
            outputs[f.name] = md5(f)
    ac = pd.concat(all_cells, ignore_index=True)
    ac.to_csv(out / "ad_cells.csv", index=False)
    outputs["ad_cells.csv"] = md5(out / "ad_cells.csv")
    ok = all(c["ok"] for c in checks.values())
    detail = dict(simulator=LM.sim_defaults(), stage="S56", ok=ok, smoke=a.smoke, seconds=round(time.time() - t0, 1), checks=checks,
                  inputs=inputs, outputs=outputs, code={"ce1_wp56.py": md5(Path(__file__))},
                  env=dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__))
    json.dump(detail, open(runs / ("S56_smoke.detail.json" if a.smoke else "S56.detail.json"), "w"), indent=1)
    log(f"{'ALL CHECKS PASS' if ok else 'CHECK FAILED'} ({len(checks)} checks, {time.time() - t0:.0f} s)")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
