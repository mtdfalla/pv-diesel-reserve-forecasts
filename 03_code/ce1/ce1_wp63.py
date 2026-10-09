"""Selection-rule table, sign tests, unserved-energy events and single-value traces, read from
existing stage outputs. Writes 04_results/ce1/WP63/.

Usage (from the repository root unless the path starts at ce1/, then from 03_code):
    python3 03_code/ce1/ce1_wp63.py --root .
    python3 03_code/ce1/ce1_wp63.py --selftest
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from math import comb
from pathlib import Path

import numpy as np
import pandas as pd

import ce1_literal as LT

FOLDS = [2019, 2020, 2021, 2022, 2023, 2024]
SCEN = ("main", "shapeB")
CP = ["icp", "mondrian", "cqr", "mondrian_cqr"]
QP = ["raw_q", "cqr", "mondrian_cqr"]
POLICIES = CP + ["raw_q", "tod_q", "nicp_var", "frac_pv"]
TGT = 2e-5
EPS = 1e-15
LIMIT = 1.0
LEVEL_TOL = 1e-6
B, SEED = 10000, 42
GAMMA, KAPPA = 0.005, 0.5
NOLOAD_L_PER_H_PER_KW = 0.0205  # ce1_load.FUEL["cat_c32"][0], checked against WP55 below
SU = {"low": (2.5, 1000.0), "central": (5.0, 1000.0), "high": (10.0, 2500.0)}  # ce1_wp55.SU_CASES
SELECTIONS = [
    ("A_R", "Year ahead (A-R), full allowance", "registered (DR1, D-091)"),
    ("b_xfit_0.002", "Static, cross-fitted on the calibration year, full allowance", "exploratory (D-112)"),
    ("c_prior_0.002", "Static, all earlier test years, full allowance", "exploratory (D-112)"),
    ("a_half_H2", "Year ahead, half allowance", "registered test PR-2 (D-110), after the A-D results"),
    ("b_xfit_0.001", "Static, cross-fitted on the calibration year, half allowance", "exploratory (D-112)"),
    ("c_prior_0.001", "Static, all earlier test years, half allowance", "exploratory (D-112)"),
    ("A_D", "Daily controller (A-D), half allowance", "registered (D-094)"),
    ("AW_clip", "Daily controller, latent level clipped at the limit", "diagnostic (WP-61)"),
]


def md5(p: Path) -> str:
    h = hashlib.md5()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def su_litres(case):
    mins, kw = SU[case]
    return mins / 60.0 * NOLOAD_L_PER_H_PER_KW * kw


def draw_counts(n, b, seed=SEED):
    """ce1_decision.draw_counts (iid day draws)."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(b, n))
    C = np.zeros((b, n))
    np.add.at(C, (np.repeat(np.arange(b), n), idx.ravel()), 1.0)
    return C


def block_counts(n, b, L, seed=SEED):
    """ce1_wp55.block_counts (circular blocks; L = 1 is draw_counts)."""
    if L == 1:
        return draw_counts(n, b, seed)
    rng = np.random.default_rng(seed)
    nb = int(np.ceil(n / L))
    st = rng.integers(0, n, size=(b, nb))
    idx = ((st[:, :, None] + np.arange(L)[None, None, :]) % n).reshape(b, -1)[:, :n]
    C = np.zeros((b, n))
    np.add.at(C, (np.repeat(np.arange(b), n), idx.ravel()), 1.0)
    return C


def ci(x, alpha=0.05):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return float(np.percentile(x, 100 * alpha / 2)), float(np.percentile(x, 100 * (1 - alpha / 2)))


def sign_p_upper(k, n):
    """One-sided exact sign test P(X >= k), X ~ Bin(n, 1/2)."""
    return sum(comb(n, i) for i in range(k, n + 1)) / 2 ** n


def sign_p_two(k, n):
    """Two-sided exact sign test."""
    return min(1.0, 2 * min(sign_p_upper(k, n), sign_p_upper(n - k, n)))


def breakeven(saving_L, d_starts, hb_L):
    """Litres per start at which the median net saving (% of handbook fuel) reaches zero,
    by bisection as ce1_wp55. inf when it stays positive up to 500 L, 0 when not positive at 0.
    """
    s, d, h = map(lambda a: np.asarray(a, float), (saving_L, d_starts, hb_L))
    f = lambda L: float(np.median((s - L * d) / h))
    if f(0.0) <= 0:
        return 0.0
    if f(500.0) > 0:
        return float("inf")
    lo, hi = 0.0, 500.0
    for _ in range(80):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if f(mid) > 0 else (lo, mid)
    return lo


# ================================================================ analysis
def analyse(root: Path) -> bool:
    R = root / "04_results" / "ce1"
    out = R / "WP63"
    out.mkdir(parents=True, exist_ok=True)
    checks, inputs = {}, {}
    LIT = LT.Literal(out, "ce1_wp63.py")

    def check(name, cond, msg=""):
        checks[name] = dict(ok=bool(cond), msg=str(msg))
        print(f"{'PASS' if cond else 'FAIL'}  {name}  {msg}", flush=True)

    def inp(rel):
        inputs[rel] = md5(R / rel)
        return R / rel

    cases = pd.read_csv(inp("WP57/cases.csv"))
    s51 = pd.read_csv(inp("S51/cells.csv"))
    ad = s51[(s51.h == 3) & s51.primary & s51.scenario.isin(SCEN)].copy()
    adt = pd.read_csv(inp("WP55/ad_totals.csv"))
    adt = adt[adt.primary & (adt.h == 3) & adt.scenario.isin(SCEN)]
    aw = pd.read_csv(inp("AW/cells.csv"))
    boot51 = pd.read_csv(inp("S51/boot.csv"))
    wp55 = json.load(open(inp("WP55/results.json")))["results"]
    check("noload_rate_matches_WP55", abs(wp55["PR1"]["L_per_start_central"] - su_litres("central")) < 1e-12,
          f"({su_litres('central'):.6f} L per start)")

    awc = aw[np.isclose(aw.clip_h.astype(float), 1.0)].copy()
    awcp = awc[awc.policy.isin(CP)]
    check("AW_clip_reproduces_body_row", len(awcp) == 48 and int(awcp.passes.sum() if "passes" in awcp else 0) >= 0
          and LIT("AW_clip_conformal_met", int((awcp.ue_share <= TGT + EPS).sum()), 0)
          and LIT("AW_clip_saving_median_pct_by_CP", [round(awcp[awcp.policy == p].saving_pct.median(), 2) for p in CP],
                  [0.41, 0.39, 0.51, 0.52]),
          [round(awcp[awcp.policy == p].saving_pct.median(), 2) for p in CP])

    recs, daily_rows = [], {}
    Cn = {}
    for yr in FOLDS:
        z = np.load(inp(f"S6E/f{yr}_h3_daily.npz"))
        tot = pd.read_parquet(inp(f"S6E/f{yr}_h3_totals.parquet"))
        pz = np.load(inp(f"S51/f{yr}_h3_paths.npz"))
        specs = list(pz["specs"].astype(str))
        pol_r = np.asarray(z["rows_policy"]).astype(str)
        par_r = np.asarray(z["rows_param"]).astype(float)
        rix = {(p, round(float(x), 6)): i for i, (p, x) in enumerate(zip(pol_r, par_r))}
        hb_i = int(np.where(pol_r == "handbook")[0][0])
        nd = z["main__fuel_L"].shape[1]
        if nd not in Cn:
            Cn[nd] = {L: block_counts(nd, B, L) for L in (1, 7)}
        for sname in SCEN:
            te = tot[(tot.scenario == sname) & (tot.set == "test")]
            hbt = te[te.policy == "handbook"].iloc[0]
            hb_d = z[f"{sname}__fuel_L"][hb_i].astype(float)
            load_d = z[f"{sname}__load_kWh"][hb_i].astype(float)
            scored_days = int((load_d > 0).sum())
            check(f"handbook_daily_sum_f{yr}_{sname}", abs(hb_d.sum() / hbt.fuel_L - 1) < 1e-6,
                  f"({hb_d.sum():.1f} vs {hbt.fuel_L:.1f})")
            for pol in POLICIES:
                cc = cases[(cases.fold == yr) & (cases.scenario == sname) & (cases.policy == pol)]
                for sel, _, _ in SELECTIONS:
                    if sel == "AW_clip":
                        a = awc[(awc.fold == yr) & (awc.scenario == sname) & (awc.policy == pol)]
                        if len(a) != 1:
                            continue
                        a = a.iloc[0]
                        recs.append(dict(fold=yr, scenario=sname, policy=pol, selection=sel, param=np.nan,
                                         fuel_L=a.fuel_L, ue_share=a.ue_share, meets=bool(a.ue_share <= TGT + EPS),
                                         fuel_handbook=float(hbt.fuel_L), starts=a.starts, handbook_starts=float(hbt.starts),
                                         days_departing=int(scored_days - a.days_at_limit), scored_days=scored_days,
                                         ue_days=int(a.ue_days), ci1_lo=np.nan, ci1_hi=np.nan, ci7_lo=np.nan, ci7_hi=np.nan))
                        continue
                    if sel == "A_D":
                        k = specs.index(f"{pol}|{GAMMA}|{KAPPA}")
                        f_d = pz[f"{sname}__fuel_L"][k].astype(float)
                        u_d = pz[f"{sname}__ue_kWh"][k].astype(float)
                        lv = pz[f"{sname}__level"][k].astype(float)
                        a = ad[(ad.fold == yr) & (ad.scenario == sname) & (ad.policy == pol)].iloc[0]
                        st = adt[(adt.fold == yr) & (adt.scenario == sname) & (adt.policy == pol)].iloc[0]
                        check(f"AD_daily_sum_f{yr}_{sname}_{pol}", abs(f_d.sum() / a.fuel_L - 1) < 1e-6)
                        dep = int(((lv < LIMIT - LEVEL_TOL) & (load_d > 0)).sum())
                        check(f"AD_days_departing_equal_S51_f{yr}_{sname}_{pol}",
                              int((lv >= LIMIT - LEVEL_TOL).sum()) == int(a.days_at_limit),
                              f"({int((lv >= LIMIT - LEVEL_TOL).sum())} vs {int(a.days_at_limit)})")
                        row = dict(param=np.nan, fuel_L=float(a.fuel_L), ue_share=float(a.ue_share),
                                   meets=bool(a.ue_share <= TGT + EPS), starts=float(st.starts), days_departing=dep)
                    else:
                        r = cc[cc.selection == sel]
                        assert len(r) == 1, (yr, sname, pol, sel)
                        r = r.iloc[0]
                        i = rix[(pol, round(float(r.param), 6))]
                        f_d = z[f"{sname}__fuel_L"][i].astype(float)
                        u_d = z[f"{sname}__ue_kWh"][i].astype(float)
                        check(f"static_daily_sum_f{yr}_{sname}_{pol}_{sel}", abs(f_d.sum() / r.fuel_L - 1) < 1e-6)
                        at_lim = float(r.param) >= LIMIT - LEVEL_TOL
                        row = dict(param=float(r.param), fuel_L=float(r.fuel_L), ue_share=float(r.ue_share),
                                   meets=bool(r.meets), starts=float(r.starts), days_departing=0 if at_lim else scored_days)
                    d = f_d - hb_d
                    c1 = ci(Cn[nd][1] @ d)
                    c7 = ci(Cn[nd][7] @ d)
                    recs.append(dict(fold=yr, scenario=sname, policy=pol, selection=sel, **row,
                                     fuel_handbook=float(hbt.fuel_L), handbook_starts=float(hbt.starts),
                                     scored_days=scored_days, ue_days=int((u_d > 1e-9).sum()),
                                     ci1_lo=c1[0], ci1_hi=c1[1], ci7_lo=c7[0], ci7_hi=c7[1]))
                    if sel == "A_D" and pol in CP + QP:
                        b = boot51[(boot51.fold == yr) & (boot51.h == 3) & (boot51.scenario == sname) & (boot51.policy == pol)]
                        if len(b) == 1:
                            lo = float(b.d_fuel_handbook_ci.iloc[0].strip("()").split(",")[0])
                            check(f"block1_reproduces_S51_handbook_ci_f{yr}_{sname}_{pol}", abs(c1[0] - lo) < 1e-3 * max(1, abs(lo)),
                                  f"({c1[0]:.3f} vs {lo:.3f})")
    df = pd.DataFrame(recs)
    df["saving_L"] = df.fuel_handbook - df.fuel_L
    df["saving_pct"] = 100 * df.saving_L / df.fuel_handbook
    df["saving_kL"] = df.saving_L / 1000.0
    df["d_starts"] = df.starts - df.handbook_starts
    for case in SU:
        df[f"net_pct_{case}"] = 100 * (df.saving_L - su_litres(case) * df.d_starts) / df.fuel_handbook
    df["breakeven_L_per_start_case"] = np.where(df.d_starts > 0, df.saving_L / df.d_starts.where(df.d_starts > 0),
                                                np.where(df.saving_L > 0, np.inf, 0.0))
    df["nonzero_saving"] = df.saving_pct > 0.005  # prints as at least 0.01 %
    df["met_nonzero"] = df.meets & df.nonzero_saving
    df["ci7_below_zero"] = df.ci7_hi < 0  # fuel below the handbook rule
    df["ci1_below_zero"] = df.ci1_hi < 0
    df.to_csv(out / "cases.csv", index=False)
    check("case_count", len(df) == 8 * 12 * len(SELECTIONS), len(df))
    sel57 = pd.read_csv(inp("WP57/selections.csv")).set_index("selection")
    for sel in sel57.index:
        g = df[(df.selection == sel) & df.policy.isin(CP)]
        check(f"reproduces_WP57_selections_{sel}", int(g.meets.sum()) == int(sel57.loc[sel, "meets"])
              and abs(g.saving_pct.median() - sel57.loc[sel, "saving_median_pct"]) < 1e-6,
              "(A-D handbook fuel: float32 daily sums in WP57 against float64 totals here, tolerance 1e-6 points)")

    # ---- per selection and policy (12 cases)
    lab = {s: (t, l) for s, t, l in SELECTIONS}
    rows = []
    for (sel, pol), g in df.groupby(["selection", "policy"], sort=False):
        py = g.groupby("fold").meets.sum()
        rows.append(dict(selection=sel, label=lab[sel][0], status=lab[sel][1], policy=pol, n=len(g),
                         met=int(g.meets.sum()), **{f"met_{y}": int(py.get(y, 0)) for y in FOLDS},
                         met_nonzero=int(g.met_nonzero.sum()),
                         saving_median_pct=g.saving_pct.median(), saving_min_pct=g.saving_pct.min(), saving_max_pct=g.saving_pct.max(),
                         saving_median_kL=g.saving_kL.median(),
                         year_median_min_pct=g.groupby("fold").saving_pct.median().min(),
                         year_median_max_pct=g.groupby("fold").saving_pct.median().max(),
                         days_departing_median=g.days_departing.median(), days_departing_max=g.days_departing.max(),
                         d_starts_median=g.d_starts.median(),
                         net_central_median_pct=g.net_pct_central.median(), net_high_median_pct=g.net_pct_high.median(),
                         breakeven_L_per_start=breakeven(g.saving_L, g.d_starts, g.fuel_handbook),
                         ue_days_median=g.ue_days.median(),
                         ci1_below_zero=int(g.ci1_below_zero.sum()) if g.ci1_hi.notna().all() else None,
                         ci7_below_zero=int(g.ci7_below_zero.sum()) if g.ci7_hi.notna().all() else None))
    t4 = pd.DataFrame(rows)
    t4.to_csv(out / "table4_rows.csv", index=False)

    rows = []
    for sel, _, _ in SELECTIONS:
        for sname_, pols in (("conformal", CP), ("quantile", QP), ("all", POLICIES)):
            g = df[(df.selection == sel) & df.policy.isin(pols)]
            py = g.groupby("fold").meets.sum()
            rows.append(dict(selection=sel, label=lab[sel][0], status=lab[sel][1], set=sname_, n=len(g), met=int(g.meets.sum()),
                             **{f"met_{y}": int(py.get(y, 0)) for y in FOLDS}, met_nonzero=int(g.met_nonzero.sum()),
                             saving_median_pct=g.saving_pct.median(), saving_median_kL=g.saving_kL.median(),
                             days_departing_median=g.days_departing.median(),
                             net_central_median_pct=g.net_pct_central.median(),
                             breakeven_L_per_start=breakeven(g.saving_L, g.d_starts, g.fuel_handbook),
                             ci1_below_zero=int(g.ci1_below_zero.sum()) if g.ci1_hi.notna().all() else None,
                             ci7_below_zero=int(g.ci7_below_zero.sum()) if g.ci7_hi.notna().all() else None))
    ts = pd.DataFrame(rows)
    ts.to_csv(out / "table4_sets.csv", index=False)
    T = ts.set_index(["selection", "set"])
    check("met_nonzero_xfit_half_conformal_is_26", LIT("met_nonzero_xfit_half_conformal", int(T.loc[("b_xfit_0.001", "conformal"), "met_nonzero"]), 26),
          int(T.loc[("b_xfit_0.001", "conformal"), "met_nonzero"]))
    check("met_nonzero_AD_conformal_is_46", LIT("met_nonzero_AD_conformal", int(T.loc[("A_D", "conformal"), "met_nonzero"]), 46),
          int(T.loc[("A_D", "conformal"), "met_nonzero"]))
    check("c_prior_full_meets_26_of_48", LIT("c_prior_full_conformal_met", int(T.loc[("c_prior_0.002", "conformal"), "met"]), 26))
    # the WP55 PR-1 numbers reproduce from this record (A-D quantile set, 36 cells)
    q = df[(df.selection == "A_D") & df.policy.isin(QP)]
    check("reproduces_WP55_net_central_median", abs(q.net_pct_central.median() - wp55["PR1"]["net_median_pct_central"]) < 1e-9,
          f"({q.net_pct_central.median():.4f})")
    check("reproduces_WP55_breakeven", abs(breakeven(q.saving_L, q.d_starts, q.fuel_handbook) - wp55["PR1"]["breakeven_L_per_start"]) < 1e-6,
          f"({breakeven(q.saving_L, q.d_starts, q.fuel_handbook):.3f} L)")

    # ---- per test year and sign tests
    py_rows, signs = [], {}
    for sel, _, _ in SELECTIONS:
        for sname_, pols in (("conformal", CP), ("quantile", QP)):
            g = df[(df.selection == sel) & df.policy.isin(pols)]
            fy = g.groupby("fold").agg(n=("meets", "size"), met=("meets", "sum"), saving_median_pct=("saving_pct", "median"),
                                       saving_min_pct=("saving_pct", "min"), saving_max_pct=("saving_pct", "max")).reset_index()
            fy.insert(0, "set", sname_)
            fy.insert(0, "selection", sel)
            py_rows.append(fy)
            k = int((fy.saving_median_pct > 0.005).sum())
            ka = int((fy.met == fy.n).sum())
            signs[f"{sel}|{sname_}"] = dict(years_positive_median_saving=k, years_all_met=ka, n_years=6,
                                            p_positive_one_sided=sign_p_upper(k, 6))
    pyd = pd.concat(py_rows, ignore_index=True)
    pyd.to_csv(out / "per_year.csv", index=False)
    v3 = {}
    for sel, _, _ in SELECTIONS:
        g = df[df.selection == sel]
        pv = g.pivot_table(index=["fold", "scenario"], columns="policy", values="saving_pct")
        mt = g.pivot_table(index=["fold", "scenario"], columns="policy", values="meets")
        best = pv[QP].max(axis=1)
        wins = int((pv["frac_pv"] > best).sum())
        yq = g[g.policy.isin(QP)].groupby("fold").saving_L.sum() / len(QP)
        yf = g[g.policy == "frac_pv"].groupby("fold").agg(s=("saving_L", "sum"), h=("fuel_handbook", "sum"))
        yd = 100 * (yf.s - yq) / yf.h
        ky = int((yd > 0).sum())
        v3[sel] = dict(frac_median_pct=float(pv["frac_pv"].median()), frac_met=int(mt["frac_pv"].sum()),
                       **{f"{p}_median_pct": float(pv[p].median()) for p in QP},
                       **{f"{p}_met": int(mt[p].sum()) for p in QP},
                       frac_beats_best_quantile=wins, n=len(pv),
                       best_minus_frac_median_points=float((best - pv["frac_pv"]).median()),
                       year_frac_minus_quantile_points={int(y): float(v) for y, v in yd.items()},
                       years_frac_ahead_pooled=ky, n_years=int(len(yd)), p_two_sided_years=sign_p_two(ky, int(len(yd))))
    signs["fixed_fraction_vs_best_quantile"] = v3
    want = {"A_R": (0.31, 6, 0.45, 0.46, 0.44, 1), "b_xfit_0.001": (0.25, 11, 0.28, 0.26, 0.28, 3),
            "c_prior_0.001": (0.28, 12, 0.32, 0.29, 0.33, 2), "A_D": (0.27, 12, 0.23, 0.18, 0.16, 9)}
    for sel, w in want.items():
        v = v3[sel]
        got = (round(v["frac_median_pct"], 2), v["frac_met"], round(v["raw_q_median_pct"], 2), round(v["cqr_median_pct"], 2),
               round(v["mondrian_cqr_median_pct"], 2), v["frac_beats_best_quantile"])
        check(f"reproduces_recorded_selection_{sel}", LIT(f"recorded_selection_{sel}", got, w), got)
    xs = {}
    for sname_, pols in (("quantile", QP), ("conformal", CP), ("all", POLICIES)):
        a = df[(df.selection == "b_xfit_0.001") & df.policy.isin(pols)].set_index(["fold", "scenario", "policy"])
        b = df[(df.selection == "A_D") & df.policy.isin(pols)].set_index(["fold", "scenario", "policy"])
        dd = (a.saving_pct - b.saving_pct)
        k = int((dd > 0).sum())
        ya, yb = a.groupby(level="fold").saving_L.sum(), b.groupby(level="fold").saving_L.sum()
        yd = 100 * (ya - yb) / a.groupby(level="fold").fuel_handbook.sum()
        ky = int((dd.groupby(level="fold").median() > 0).sum())
        ds = (a.saving_L - b.saving_L); dn = (a.starts - b.starts); fh = a.fuel_handbook
        cross = (ds / dn).where(dn > 0)
        pm = lambda L: float((100 * (ds - L * dn) / fh).median())  # noqa: E731
        lo, hi = 0.0, 200.0
        if pm(lo) > 0 and pm(hi) < 0:
            for _ in range(60):
                mid = 0.5 * (lo + hi)
                lo, hi = (mid, hi) if pm(mid) > 0 else (lo, mid)
            pm_cross = 0.5 * (lo + hi)
        else:
            pm_cross = float("nan")
        xs[sname_] = dict(static_saves_more=k, n=len(dd), median_difference_points=float(dd.median()),
                          years_static_ahead=ky, n_years=int(dd.index.get_level_values("fold").nunique()),
                          p_two_sided_years=sign_p_two(ky, int(dd.index.get_level_values("fold").nunique())),
                          year_difference_points={int(y): float(v) for y, v in yd.items()},
                          median_year_difference_points=float(yd.median()),
                          net_all_cases_dn_positive=bool((dn > 0).all()),
                          net_cases_static_ahead_at_high_L=int(((ds - su_litres("high") * dn) > 0).sum()),
                          net_min_case_crossover_L=float(cross.min()), net_every_case_ahead_below_L=int(np.floor(cross.min())),
                          net_paired_median_crossover_L=pm_cross,
                          net_paired_median_difference_at_high_L=pm(su_litres("high")))
    check("starts_xh_starts_more_than_AD_in_every_quantile_case", xs["quantile"]["net_all_cases_dn_positive"])
    check("starts_every_quantile_case_ahead_at_8.5L", xs["quantile"]["net_cases_static_ahead_at_high_L"] == xs["quantile"]["n"],
          f"({xs['quantile']['net_cases_static_ahead_at_high_L']} of {xs['quantile']['n']})")
    check("starts_paired_median_crossover_above_min_case_crossover",
          xs["quantile"]["net_paired_median_crossover_L"] > xs["quantile"]["net_min_case_crossover_L"] > su_litres("high"),
          f"(median {xs['quantile']['net_paired_median_crossover_L']:.2f} L, min case {xs['quantile']['net_min_case_crossover_L']:.2f} L)")
    signs["xfit_half_vs_AD"] = xs
    json.dump(signs, open(out / "signtests.json", "w"), indent=1, default=float)

    # ---- events: per test year, peak against online capacity and the largest online unit (A-R, A-D)
    ev = pd.read_csv(inp("WP61/events.csv"))
    erows = []
    for (proto, pol), g in ev.groupby(["protocol", "policy"]):
        per = g.groupby(["fold", "scenario"]).size().reindex(pd.MultiIndex.from_product([FOLDS, SCEN]), fill_value=0)
        erows.append(dict(protocol=proto, policy=pol, events_per_year_median=float(per.median()), events_per_year_max=int(per.max()),
                          peak_kW_median=float(g.peak_kW.median()), peak_share_online_median=float(g.peak_share_online.median()),
                          peak_share_online_max=float(g.peak_share_online.max()), peak_share_largest_max=float(g.peak_share_largest.max()),
                          above_largest=int((g.peak_share_largest > 1).sum()), n_events=len(g)))
    for proto in ("A-R", "A-D"):
        for sname_, pols in (("conformal", CP), ("quantile", QP)):
            g = ev[(ev.protocol == proto) & ev.policy.isin(pols)]
            per = g.groupby(["fold", "scenario", "policy"]).size().reindex(
                pd.MultiIndex.from_product([FOLDS, SCEN, pols]), fill_value=0)
            erows.append(dict(protocol=proto, policy=f"set:{sname_}", events_per_year_median=float(per.median()),
                              events_per_year_max=int(per.max()), peak_kW_median=float(g.peak_kW.median()),
                              peak_share_online_median=float(g.peak_share_online.median()),
                              peak_share_online_max=float(g.peak_share_online.max()),
                              peak_share_largest_max=float(g.peak_share_largest.max()),
                              above_largest=int((g.peak_share_largest > 1).sum()), n_events=len(g)))
    evd = pd.DataFrame(erows)
    evd.to_csv(out / "events.csv", index=False)
    e = evd.set_index(["protocol", "policy"])
    check("events_reproduce", LIT("events_per_year_median_AD_AR_and_AD_peak_kW",
                                       (e.loc[("A-D", "set:conformal"), "events_per_year_median"], e.loc[("A-R", "set:conformal"), "events_per_year_median"],
                                        round(e.loc[("A-D", "set:conformal"), "peak_kW_median"])), (12.0, 27.5, 85)),
          (e.loc[("A-D", "set:conformal"), "events_per_year_median"], e.loc[("A-R", "set:conformal"), "events_per_year_median"]))

    # ---- paired DR2 / DR3 contrasts at fixed levels (main, 15 min), fuel and UE, 1- and 7-day blocks
    pairs = [("DR2", "best", "raw_q"), ("DR2", "cqr", "raw_q"), ("DR2", "mondrian_cqr", "raw_q"),
             ("DR3", "mondrian_cqr", "cqr"), ("DR3", "mondrian", "icp"), ("DR3", "mondrian", "nicp_var")]
    FIX = [0.99, 0.995, 0.998, 0.999]
    prs = []
    for yr in FOLDS:
        z = np.load(R / f"S6E/f{yr}_h3_daily.npz")
        sj = json.load(open(inp(f"S6E/f{yr}_h3_select.json")))["selections"]["main"]["policies"]
        best = "cqr" if sj["cqr"]["fuel_at_target_H2"] <= sj["mondrian_cqr"]["fuel_at_target_H2"] else "mondrian_cqr"
        pol_r = np.asarray(z["rows_policy"]).astype(str)
        par_r = np.asarray(z["rows_param"]).astype(float)
        rix = {(p, round(float(x), 6)): i for i, (p, x) in enumerate(zip(pol_r, par_r))}
        nd = z["main__fuel_L"].shape[1]
        for dr, a, b in pairs:
            a_ = best if a == "best" else a
            levels = [("fixed", t) for t in FIX] + [("AR_of_second", float(sj[b]["A_R"]["param"]))]
            for kind, th in levels:
                ia, ib = rix.get((a_, round(th, 6))), rix.get((b, round(th, 6)))
                if ia is None or ib is None:
                    continue
                df_ = z["main__fuel_L"][ia].astype(float) - z["main__fuel_L"][ib].astype(float)
                du = z["main__ue_kWh"][ia].astype(float) - z["main__ue_kWh"][ib].astype(float)
                r = dict(fold=yr, contrast=dr, first=a if a != "best" else f"best={best}", second=b, level_kind=kind, theta=th,
                         d_fuel_L=float(df_.sum()), d_ue_kWh=float(du.sum()))
                for L in (1, 7):
                    lo, hi = ci(Cn[nd][L] @ df_)
                    ulo, uhi = ci(Cn[nd][L] @ du)
                    r.update({f"fuel_ci{L}_lo": lo, f"fuel_ci{L}_hi": hi, f"ue_ci{L}_lo": ulo, f"ue_ci{L}_hi": uhi})
                prs.append(r)
    prd = pd.DataFrame(prs)
    prd.to_csv(out / "paired_dr.csv", index=False)
    summ = (prd.assign(below7=prd.fuel_ci7_hi < 0, above7=prd.fuel_ci7_lo > 0, below1=prd.fuel_ci1_hi < 0, above1=prd.fuel_ci1_lo > 0,
                       ue_below7=prd.ue_ci7_hi < 0, ue_above7=prd.ue_ci7_lo > 0)
            .groupby(["contrast", "first", "second", "level_kind", "theta"], dropna=False)
            [["below7", "above7", "below1", "above1", "ue_below7", "ue_above7"]].sum().reset_index())
    summ.to_csv(out / "paired_dr_summary.csv", index=False)
    check("paired_dr_rows", len(prd) >= 6 * 6 * 4, len(prd))

    # ---- traces
    tr = {}
    # (a) days at the handbook level under A-D, per policy (S51 primary, both shapes)
    dl = ad.groupby("policy").days_at_limit.median()
    tr["days_at_limit_median_by_policy"] = {p: float(dl[p]) for p in POLICIES}
    tr["days_at_limit_median_quantile_range"] = [float(dl[QP].min()), float(dl[QP].max())]
    tr["days_at_limit_median_conformal_pooled"] = float(ad[ad.policy.isin(CP)].days_at_limit.median())
    tr["days_at_limit_frac_pv_max"] = int(ad[ad.policy == "frac_pv"].days_at_limit.max())
    check("trace_247_254_and_277.5", LIT("days_at_limit_quantile_range_and_conformal_pooled",
                                         [tr["days_at_limit_median_quantile_range"], tr["days_at_limit_median_conformal_pooled"]],
                                         [[247.0, 254.0], 277.5]), tr["days_at_limit_median_quantile_range"])
    # (b) the A-D quantile-policy saving minimum (fold 2024, shape A)
    q = df[(df.selection == "A_D") & df.policy.isin(QP)]
    mn = q.loc[q.saving_pct.idxmin()]
    tr["AD_quantile_min"] = dict(fold=int(mn.fold), scenario=mn.scenario, policy=mn.policy, saving_pct=float(mn.saving_pct),
                                 negative_cases=q[q.saving_pct < 0][["fold", "scenario", "policy", "saving_pct"]].to_dict("records"))
    check("trace_minus_0.02_is_fold2024_main", LIT("AD_quantile_min_fold_scenario", [int(mn.fold), mn.scenario], [2024, "main"]),
          tr["AD_quantile_min"])
    # (c) committed diesel capacity with the battery (15 min, shape A, A-R): handbook and quantile policies
    cap = dict(handbook=[], quantile=[])
    for yr in FOLDS:
        tot = pd.read_parquet(R / f"S6E/f{yr}_h3_totals.parquet")
        sj = json.load(open(R / f"S6E/f{yr}_h3_select.json"))["selections"]
        te = tot[tot.set == "test"]
        on = lambda sc, pol, par=None: float(te[(te.scenario == sc) & (te.policy == pol) & (
            True if par is None else np.isclose(te.param, par))].online_kW_h.iloc[0])
        cap["handbook"].append(100 * (1 - on("battery", "handbook") / on("main", "handbook")))
        for pol in QP:
            pm, pb = float(sj["main"]["policies"][pol]["A_R"]["param"]), float(sj["battery"]["policies"][pol]["A_R"]["param"])
            cap["quantile"].append(100 * (1 - on("battery", pol, pb) / on("main", pol, pm)))
    tr["committed_capacity_reduction_pct"] = dict(handbook_median=float(np.median(cap["handbook"])),
                                                  quantile_median=float(np.median(cap["quantile"])),
                                                  handbook_range=[float(min(cap["handbook"])), float(max(cap["handbook"]))],
                                                  quantile_range=[float(min(cap["quantile"])), float(max(cap["quantile"]))])
    check("trace_3.1_and_3.5", LIT("committed_capacity_reduction_pct_handbook_quantile",
                                   (round(tr["committed_capacity_reduction_pct"]["handbook_median"], 1),
                                    round(tr["committed_capacity_reduction_pct"]["quantile_median"], 1)), (3.1, 3.5)),
          tr["committed_capacity_reduction_pct"])
    b6 = pd.read_csv(inp("S61B/ad_cells.csv"))
    b6 = b6[b6.primary & b6.scenario.isin(["battery_day", "batteryB_day"]) & b6.policy.isin(QP)].copy()
    f0 = []
    for r in b6.itertuples():
        sj = json.load(open(R / f"S6E/f{r.fold}_h3_select.json"))["selections"]
        f0.append(sj["main" if r.scenario == "battery_day" else "shapeB"]["anchors"]["handbook"]["fuel_L"])
    b6["sav"] = 100 * (1 - b6.fuel_L / np.array(f0))
    tr["battery_day_AD_quantile_combined_median_pct"] = float(b6.sav.median())
    tr["battery_day_AD_quantile_cells"] = len(b6)
    tr["battery_day_AD_basis"] = "S61B/ad_cells.csv primary, battery_day and batteryB_day, quantile policies, 36 cells"
    check("trace_0.93", LIT("battery_day_AD_quantile_combined_median_pct", round(tr["battery_day_AD_quantile_combined_median_pct"], 2), 0.93)
          and len(b6) == 36,
          tr["battery_day_AD_quantile_combined_median_pct"])
    # (e): time-of-day, fold 2024, 15 min, transitional: S7 file stores PICP rounded to 4 places
    c7 = pd.read_csv(inp("S7/coverage_ci.csv"))
    s46 = pd.read_csv(inp("S46/coverage.csv"))
    a = c7[(c7.site == "yulara") & (c7.fold == 2024) & (c7.horizon_min == 15) & (c7.method == "tod") & (c7.variant == "static")
           & (c7.nominal == 0.9) & (c7.scope == "transitional")].iloc[0]
    b = s46[(s46.definition == "base") & (s46.fold == 2024) & (s46.horizon_min == 15) & (s46.method == "tod") & (s46.scope == "transitional")].iloc[0]
    tr["picp_double_rounding"] = dict(S7_PICP_stored=float(a.PICP), S7_n=int(a.n), S46_picp=float(b.picp), S46_n=int(b.n),
                          explanation="same scored set; S4.1 rounds a 4-place stored value (double rounding)")
    check("trace_picp_same_rows_double_rounding", int(a.n) == int(b.n) and abs(a.PICP - b.picp) < 5e-5
          and round(float(a.PICP), 4) == float(a.PICP), tr["picp_double_rounding"])
    # same class: how many S4.1 cells (S7 file) differ at 3 places from the unrounded S46 value
    m = c7[(c7.site == "yulara") & (c7.fold == 2024) & (c7.variant == "static") & (c7.nominal == 0.9)].merge(
        s46[(s46.definition == "base") & (s46.fold == 2024)], on=["horizon_min", "method", "scope"], suffixes=("", "_46"))
    r3 = lambda x: np.floor(np.asarray(x, float) * 1000 + 0.5) / 1000
    diff = m[r3(m.PICP) != r3(m.picp)]
    tr["picp_same_class_cells"] = dict(cells_compared=len(m), cells_differing_at_3_places=len(diff),
                                     cells=diff[["horizon_min", "method", "scope", "PICP", "picp"]].to_dict("records"))
    dip = pd.read_csv(inp("WP59C/icp_dip.csv")).set_index("fold")
    same_year = []
    for yr in FOLDS[:-1]:
        same_year.append(dict(year=yr, as_test=float(dip.loc[yr, "test_share_transitional"]),
                              as_calibration=float(dip.loc[yr + 1, "calib_share_transitional"])))
    tr["calib_vs_test_same_year"] = dict(pairs=same_year, max_abs_diff=float(max(abs(p["as_test"] - p["as_calibration"]) for p in same_year)),
                          explanation="labels of the same year come from two fold caches (test rows of fold Y, calibration rows "
                                      "of fold Y+1); row sets are compared in S63 (needs the S2 caches)")
    tr["transitional_share_test"] = dict(test_transitional_mean_5min=float(dip.test_share_transitional.mean()),
                          basis="WP59C icp_dip is 5 min (S5 obs h1); main text 33.3% is pooled at 15 min; S2.1 33.4% is a mean")
    # (g): projected-ACI widening computed on unrounded PINAW
    iv = pd.read_csv(inp("S5/intervals.csv"))
    g = iv[(iv.site == "yulara") & (iv.fold == 2024) & (iv.horizon_min == 60) & (iv.method == "aci") & (iv.nominal == 0.9)
           & (iv.scope == "all")].set_index("variant")
    w = 100 * (g.loc["delayed_projected", "PINAW"] / g.loc["anticipative", "PINAW"] - 1)
    w3 = 100 * (round(g.loc["delayed_projected", "PINAW"], 3) / round(g.loc["anticipative", "PINAW"], 3) - 1)
    ww = 100 * (g.loc["delayed_projected", "width_Wm2"] / g.loc["anticipative", "width_Wm2"] - 1)
    tr["aci_widening"] = dict(widening_pct_stored_PINAW=float(w), widening_pct_from_3_place_PINAW=float(w3), widening_pct_Wm2=float(ww))
    check("trace_aci_widening", round(w, 1) == 9.9 and round(w3, 1) == 10.2 and round(ww, 1) == 9.9, tr["aci_widening"])
    json.dump(tr, open(out / "traces.json", "w"), indent=1, default=float)

    ok = all(c["ok"] for c in checks.values())
    outs = {p.name: md5(p) for p in sorted(out.glob("*")) if p.name != "checks.json"}
    json.dump(dict(checks=checks, inputs=inputs, outputs=outs, code={"ce1_wp63.py": md5(Path(__file__))}),
              open(out / "checks.json", "w"), indent=1)
    (R / "_runs").mkdir(exist_ok=True)
    json.dump(dict(stage="WP63", ok=ok, finished=time.strftime("%Y-%m-%dT%H:%M:%S"), host="Linux",
                   n_checks=len(checks), n_failed=sum(not c["ok"] for c in checks.values()), outputs=outs,
                   code={"ce1_wp63.py": md5(Path(__file__))},
                   env=dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__)),
              open(R / "_runs" / "WP63.json", "w"), indent=1)
    print(f"{'ALL CHECKS PASS' if ok else 'CHECK FAILED'} ({len(checks)} checks, {sum(not c['ok'] for c in checks.values())} failed)")
    return ok


# ================================================================ self-test
def selftest(verbose=True):
    res = {}

    def ok(name, cond, msg=""):
        res[name] = bool(cond)
        if verbose:
            print(f"  [{'PASS' if cond else 'FAIL'}] {name} {msg}")

    ok("sign_upper_6_of_6", abs(sign_p_upper(6, 6) - 1 / 64) < 1e-15)
    ok("sign_two_sided_symmetric", abs(sign_p_two(9, 12) - sign_p_two(3, 12)) < 1e-15 and abs(sign_p_two(6, 12) - 1.0) < 1e-15)
    ok("breakeven_simple", abs(breakeven([10.0], [2.0], [100.0]) - 5.0) < 1e-6)
    ok("breakeven_no_extra_starts", breakeven([10.0], [0.0], [100.0]) == float("inf"))
    ok("breakeven_no_saving", breakeven([-1.0], [5.0], [100.0]) == 0.0)
    C = block_counts(20, 50, 7)
    ok("block_counts_rows_sum_to_n", np.allclose(C.sum(axis=1), 20))
    ok("block1_is_iid_draws", np.array_equal(block_counts(20, 50, 1), draw_counts(20, 50)))
    ok("su_central_1.708", abs(su_litres("central") - 1.7083333333333335) < 1e-12)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    st = selftest()
    if not all(st.values()):
        sys.exit(1)
    if a.selftest:
        sys.exit(0)
    sys.exit(0 if analyse(Path(a.root)) else 1)


if __name__ == "__main__":
    main()
