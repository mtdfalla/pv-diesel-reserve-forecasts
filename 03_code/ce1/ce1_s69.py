#!/usr/bin/env python3
"""Stage S69: the plain, two-term N-1 and additive N-1 constructions, and the persistence
fraction, for every selection rule (rule D-136).

Usage (from the repository root unless the path starts at ce1/, then from 03_code):
    python3 03_code/ce1/ce1_s69.py --selftest
"""
from __future__ import annotations

import argparse
import gc
import pickle
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "utils"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import ce1_adaptive as A  # noqa: E402
import ce1_dispatch as D  # noqa: E402
import ce1_folds as FC  # noqa: E402
import ce1_load as LM  # noqa: E402
import ce1_s6 as S6  # noqa: E402
import ce1_s63 as S63  # noqa: E402
import ce1_wp56 as W56  # noqa: E402
import ce1_wp57 as W57  # noqa: E402
import ce1_wp67_sim as SIM  # noqa: E402

H = 3
FOLDS = [2019, 2020, 2021, 2022, 2023, 2024]
TGT = LM.UE_TARGET
HALF = TGT / 2.0
POLS = ["icp", "mondrian", "cqr", "mondrian_cqr", "raw_q", "tod_q", "nicp_var", "frac_pv"]
CP = ["icp", "mondrian", "cqr", "mondrian_cqr"]
QP = ["raw_q", "cqr", "mondrian_cqr"]
SETS = {"conformal": CP, "quantile": QP}
STATIC = {"A_R": ("T2", TGT), "b_xfit_0.002": ("X", TGT), "c_prior_0.002": ("P", TGT),
          "a_half_H2": ("T2", HALF), "b_xfit_0.001": ("X", HALF), "c_prior_0.001": ("P", HALF)}
LABEL = {"A_R": "A-R", "b_xfit_0.002": "XF", "c_prior_0.002": "PF", "a_half_H2": "H2",
         "b_xfit_0.001": "XH", "c_prior_0.001": "PH", "A_D": "A-D"}
SHAPES = ("A", "B")
NAME = {("plain", "A"): "main", ("plain", "B"): "shapeB", ("s56", "A"): "cont", ("s56", "B"): "contB",
        ("n1a", "A"): "n1a", ("n1a", "B"): "n1aB"}
PFR = "pfr"
CONSTRS = ("plain", "s56", "n1a")  # S69; S69b (D-137) runs ("n1a") with PFR_IN = {"n1a"}
PFR_IN = {"plain"}
REF69 = None  # S69b: S69 outputs the n1a rows must reproduce
EQUIV = 0.1  # D-134 item 6, percentage points
TOTK = ("fuel_L", "ue_kWh", "load_kWh", "ue_share", "starts", "curt_kWh", "online_kW_h", "fallback_steps", "scored_steps")


def md5(p: Path) -> str:
    h = hashlib.md5()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def log(m):
    print(time.strftime("%H:%M:%S"), f"[S69] {m}", flush=True)


def scen(constr, shape):
    if constr == "plain":
        return S6.SCENARIOS["main" if shape == "A" else "shapeB"]
    return W56.SCEN56["cont" if shape == "A" else "contB"]


# ================================================================ N1A
def kmin_n1a(L, pc, ratings):
    """Smallest feasible online capacity under N1A for rows pc (K, n) of counted PV (NaN: 0)."""
    cap, capn1, small = W56.combo_table(ratings)
    pcz = np.nan_to_num(pc)
    need = L[None, :] - pcz
    k1 = np.maximum(need, small)
    out = np.full(pc.shape, cap[-1])
    feas = np.zeros(pc.shape, bool)
    for j in range(len(cap) - 1, -1, -1):
        m = (cap[j] >= k1 - 1e-9) & (capn1[j] >= need - 1e-9)
        out = np.where(m, cap[j], out)
        feas |= m
    return out, feas


def effective_n1a(L, C, ratings, chunk=64):
    K, n = C.shape
    Ce = np.empty((K, n))
    infeas = np.zeros((K, n), bool)
    for a in range(0, K, chunk):
        km, f = kmin_n1a(L, C[a:a + chunk], ratings)
        Ce[a:a + chunk] = L[None, :] - km
        infeas[a:a + chunk] = ~f
    return Ce, infeas


def transform(constr, L, C, det_row, ratings):
    if constr == "plain":
        return C, np.zeros(C.shape, bool)
    if constr == "s56":
        Ce, _, inf_ = W56.effective_counted(L, C, det_row, ratings, True)
        return Ce, inf_
    return effective_n1a(L, C, ratings)


# ================================================================ PFR rows
def pfr_matrix(ev_idx, geo, m, grid, h=H):
    """(51, len(grid)) counted PV of the persistence fraction, placed at the target steps."""
    idx = pd.DatetimeIndex(ev_idx)
    ghi = geo["ghi"].reindex(idx).values.astype(float)
    pv = S6.PVAt(idx, m)(ghi)
    M = np.vstack([(1.0 - float(r)) * pv for r in D.FRAC_GRID])
    return S6.place(grid, idx + pd.Timedelta(minutes=5 * h), M), ghi


# ================================================================ simulation helpers
def tot_table(r, rows, extra=None):
    t = pd.DataFrame({k: np.asarray(r["total"][k], float) for k in TOTK if k in r["total"]})
    t.insert(0, "param", [x[1] for x in rows])
    t.insert(0, "policy", [x[0] for x in rows])
    if extra:
        for k, v in extra.items():
            t[k] = v
    return t


def cmp_tot(a: pd.DataFrame, b: pd.DataFrame, keys=("fuel_L", "ue_kWh", "starts")):
    """Worst relative difference between two totals tables on the same (policy, param) rows."""
    a = a.reset_index(drop=True)
    b = b.reset_index(drop=True)
    assert len(a) == len(b), (len(a), len(b))
    assert (a.policy.astype(str).values == b.policy.astype(str).values).all()
    pa, pb = a.param.to_numpy(float), b.param.to_numpy(float)
    assert np.array_equal(np.isnan(pa), np.isnan(pb)) and np.allclose(pa[~np.isnan(pa)], pb[~np.isnan(pb)], atol=1e-9)
    w = 0.0
    for k in keys:
        x, y = a[k].to_numpy(float), b[k].to_numpy(float)
        w = max(w, float(np.max(np.abs(x - y) / np.maximum(np.abs(y), 1.0))))
    return w


# ================================================================ one fold
def run_fold(yr, R, out, geo, m, unit96, scales, rows6, check, store):
    t0 = time.time()
    rows = rows6 + [(PFR, float(r)) for r in D.FRAC_GRID]
    n6 = len(rows6)
    hb = [i for i, r in enumerate(rows) if r[0] == "handbook"][0]
    det = [i for i, r in enumerate(rows) if r[0] == "deterministic"][0]
    orc = [i for i, r in enumerate(rows) if r[0] == "oracle"][0]
    rix = {(p, round(float(x), 6)): i for i, (p, x) in enumerate(rows) if p not in S6.ANCHORS}
    pol_ix = {}
    for pol in A.POLICIES + [PFR]:
        ix = [i for i, r in enumerate(rows) if r[0] == pol]
        lv = np.array([rows[i][1] for i in ix])
        o = np.argsort(lv)
        pol_ix[pol] = (lv[o], np.array(ix)[o])

    cache = FC.load_cache_ext(R / "S2" / f"yulara_f{yr}_h3.parquet", R / "S2X" / f"yulara_f{yr}_h3.parquet")
    prep = S6.prepare(cache, H, yr, geo, m, rows6)
    calib = S6.as_dict(cache[cache["split"] == "calib"])
    test = S6.as_dict(cache[cache["split"] == "test"])
    H1, H2d = D.split_halves(calib)
    # PFR rows on H2 (issue rows of H2), test, H1x
    for part, ev in (("H2", H2d), ("test", test)):
        M, ghi_issue = pfr_matrix(ev["idx"], geo, m, prep[part]["grid"])
        prep[part]["C"] = np.vstack([prep[part]["C"], M])
        # measured GHI at issue equals the cache target GHI of the row whose target is the issue time
        tg = pd.Series(ev["y"], index=pd.DatetimeIndex(ev["idx"]) + pd.Timedelta(minutes=5 * H))
        both = tg.reindex(pd.DatetimeIndex(ev["idx"]))
        ok = np.isfinite(both.values) & np.isfinite(ghi_issue)
        check(f"pfr_issue_ghi_is_measured_f{yr}_{part}", ok.sum() > 1000 and np.allclose(both.values[ok], ghi_issue[ok], atol=1e-6),
              f"({int(ok.sum())} rows compared, {int(np.isnan(ghi_issue).sum())} issue times without measured GHI)")
        store.setdefault("pfr_missing_issue", []).append(dict(fold=yr, part=part, issue_rows=len(ghi_issue),
                                                              no_measured_ghi=int(np.isnan(ghi_issue).sum())))
    cy = yr - 1
    tgt = pd.DatetimeIndex(H1["idx"]) + pd.Timedelta(minutes=5 * H)
    gridx = S6.year_grid(f"{cy}-01-01", f"{cy}-06-30 23:55")
    pavx = LM.available_pv_on_grid(gridx, geo["ghi"].reindex(gridx).values, m)
    pv_of = S6.PVAt(tgt, m)
    Cx = S6.place(gridx, tgt, S6.counted_matrix(rows6, H2d, H1, pv_of, pv_of(H1["y"])))
    Mx, _ = pfr_matrix(H1["idx"], geo, m, gridx)
    Cx = np.vstack([Cx, Mx])
    dayx = LM.day_ids(gridx)
    parts = {"H2": (prep["H2"]["grid"], prep["H2"]["pav"], prep["H2"]["C"], prep["H2"]["day"]),
             "test": (prep["test"]["grid"], prep["test"]["pav"], prep["test"]["C"], prep["test"]["day"]),
             "H1x": (gridx, pavx, Cx, dayx)}
    grid, pav, day = parts["test"][0], parts["test"][1], parts["test"][3]
    te_d = S6.as_dict(cache[cache["split"] == "test"])
    cdays = S63.complete_days_fc(te_d["idx"], geo["is_day"].astype(bool), H, 0.95)
    gd = pd.DatetimeIndex(grid).normalize().unique().values.astype("datetime64[D]")
    keep = np.array([d_ in cdays for d_ in gd])

    s6e = pd.read_parquet(R / "S6E" / f"f{yr}_h3_totals.parquet")
    s57 = pd.read_parquet(R / "S57" / f"f{yr}_h3_H1x.parquet")
    s56 = pd.read_parquet(R / "S56" / f"f{yr}_h3_totals.parquet")
    s51 = pd.read_csv(R / "S51" / "cells.csv")
    ad56 = pd.read_csv(R / "S56" / "ad_cells.csv", float_precision="round_trip")
    cases = pd.read_csv(R / "WP63" / "cases.csv", float_precision="round_trip")
    simc = pd.read_csv(R / "WP67" / "sim" / f"f{yr}_cases.csv", float_precision="round_trip")
    simn = pd.read_csv(R / "WP67" / "sim" / f"f{yr}_n1.csv", float_precision="round_trip")

    case_rows, ev_rows, anchor_rows = [], [], []
    for constr in CONSTRS:
        for shape in SHAPES:
            nm = NAME[(constr, shape)]
            sc = scen(constr, shape)
            tabs, res_test, Ce_test, inf_test, L_test, rat_test = {}, None, None, None, None, None
            for part, (g_, pv_, C_, d_) in parts.items():
                L = S6.load_for(g_, unit96, scales, sc)
                ratings = LM.fleet_for_peak(L.max())
                sim_kw = dict(ratings=ratings, min_load=sc["min_load"], fuel=sc["fuel"], battery=False)
                rws = rows if constr in PFR_IN else rows6
                Cuse = C_ if constr in PFR_IN else C_[:n6]
                Ce, inf_ = transform(constr, L, Cuse, Cuse[det], ratings)
                r = LM.simulate(L, pv_, Ce, d_, H, **sim_kw)
                fb = W56.fallback_counts(Cuse, pv_)
                r["total"]["fallback_steps"] = fb.astype(float)
                tabs[part] = tot_table(r, rws, dict(infeasible_steps=inf_[:, np.isfinite(pv_)].sum(axis=1)))
                if part == "test":
                    res_test, Ce_test, inf_test, L_test, rat_test = r, Ce, inf_, L, ratings
                check(f"handbook_zero_ue_f{yr}_{nm}_{part}", float(r["total"]["ue_kWh"][hb]) == 0.0)
                if constr == "plain" and part in ("H2", "test"):
                    ref = s6e[(s6e.scenario == nm) & (s6e.set == part)]
                    check(f"GATE_plain_reproduces_S6E_f{yr}_{nm}_{part}", (w := cmp_tot(tabs[part].iloc[:n6], ref)) < 1e-9, f"(worst {w:.1e})")
                if constr == "plain" and part == "H1x":
                    ref = s57[s57.scenario == nm]
                    check(f"GATE_plain_reproduces_S57_H1x_f{yr}_{nm}", (w := cmp_tot(tabs[part].iloc[:n6], ref)) < 1e-9, f"(worst {w:.1e})")
                if constr == "s56" and part in ("H2", "test"):
                    ref = s56[(s56.scenario == nm) & (s56.set == part)]
                    check(f"GATE_s56_reproduces_S56_f{yr}_{nm}_{part}", (w := cmp_tot(tabs[part], ref)) < 1e-9, f"(worst {w:.1e})")
                    if part == "test":
                        check(f"GATE_s56_infeasible_equals_S56_f{yr}_{nm}",
                              np.array_equal(tabs[part].infeasible_steps.to_numpy(int), ref.n1_infeasible_steps.to_numpy(int)))
                if REF69 is not None and constr == "n1a":
                    t69 = REF69["totals"]
                    ref = t69[(t69.fold == yr) & (t69.scenario == nm) & (t69.set == part)]
                    check(f"GATE_S69b_n1a_totals_equal_S69_f{yr}_{nm}_{part}", (w := cmp_tot(tabs[part].iloc[:n6], ref, keys=TOTK[:5])) == 0.0, f"(worst {w:.1e})")
                if part != "test":
                    del Ce, inf_, r
                gc.collect()
            for part, t in tabs.items():
                t2 = t.copy()
                t2.insert(0, "set", part)
                t2.insert(0, "construction", constr)
                t2.insert(0, "scenario", nm)
                t2.insert(0, "fold", yr)
                store.setdefault("totals", []).append(t2)
            store.setdefault("test_tabs", {})[(yr, constr, shape)] = tabs["test"]
            T2, TX, TT = tabs["H2"], tabs["H1x"], tabs["test"]
            hb_fuel = float(TT.fuel_L.iloc[hb])
            anchor_rows.append(dict(fold=yr, construction=constr, shape=shape, scenario=nm, handbook_fuel_L=hb_fuel,
                                    oracle_fuel_L=float(TT.fuel_L.iloc[orc]), handbook_starts=float(TT.starts.iloc[hb]),
                                    handbook_infeasible_steps=int(TT.infeasible_steps.iloc[hb]),
                                    oracle_ceiling_pct=100 * (1 - float(TT.fuel_L.iloc[orc]) / hb_fuel)))
            # ---- A-D from this construction's own test record, theta0 = its A-R level
            pols = POLS + ([PFR] if constr in PFR_IN else [])
            sim_kw = dict(ratings=rat_test, min_load=sc["min_load"], fuel=sc["fuel"], battery=False)
            daily = res_test["daily"]
            load_d = daily["load_kWh"][hb]
            abar = A.h2_mean_daily_load(prep["H2"], unit96, scales, sc)
            pol_C, rec_ue, specs, ar_lv = {}, {}, [], {}
            for pol in pols:
                lv, ixo = pol_ix[pol]
                pol_C[pol] = (lv, Ce_test[ixo])
                rec_ue[pol] = daily["ue_kWh"][ixo]
                ar_lv[pol], _ = W57.select(T2, pol, TGT)
                specs.append((pol, A.GAMMA, A.KAPPA, float(ar_lv[pol]), float(lv.min())))
            rp, idx_ad, _, passes, _ = A.run_paths(L_test, pav, day, H, sim_kw, pol_C, specs, rec_ue, load_d, abar)
            check(f"ad_converged_f{yr}_{nm}", 1 <= passes < 400 and bool(np.isfinite(rp["total"]["fuel_L"]).all()), f"({passes} passes)")
            for k, (pol, *_r) in enumerate(specs):
                if pol == PFR:
                    continue
                if constr == "plain":
                    ref = s51[(s51.fold == yr) & (s51.h == H) & (s51.scenario == nm) & (s51.policy == pol) & s51.primary.astype(bool)]
                    check(f"GATE_plain_AD_reproduces_S51_f{yr}_{nm}_{pol}",
                          len(ref) == 1 and abs(float(rp["total"]["fuel_L"][k]) / float(ref.fuel_L.iloc[0]) - 1) < 1e-7
                          and abs(float(rp["total"]["ue_kWh"][k]) - float(ref.ue_kWh.iloc[0])) < 1e-3)
                if constr == "s56":
                    ref = ad56[(ad56.fold == yr) & (ad56.scenario == nm) & (ad56.policy == pol) & ad56.primary.astype(bool)]
                    check(f"GATE_s56_AD_reproduces_S56_f{yr}_{nm}_{pol}",
                          len(ref) == 1 and abs(float(rp["total"]["fuel_L"][k]) / float(ref.fuel_L.iloc[0]) - 1) < 1e-9)
            # ---- selections
            prior = [store["test_tabs"][(y, constr, shape)] for y in FOLDS if y < yr]
            meta = [("handbook", "handbook", np.nan)]
            Crec = [Ce_test[hb]]
            Corig = [parts["test"][2][hb]]
            for pol in pols:
                X = W57.pooled([T2[T2.policy == pol].reset_index(drop=True), TX[TX.policy == pol].reset_index(drop=True)])
                Pp = W57.pooled([t[t.policy == pol].reset_index(drop=True) for t in prior]) if prior else T2[T2.policy == pol].reset_index(drop=True)
                tab_of = {"T2": T2, "X": X, "P": Pp}
                for sel, (tb, tg) in STATIC.items():
                    p_, _ = W57.select(tab_of[tb], pol, tg)
                    i = rix[(pol, round(float(p_), 6))]
                    meta.append((pol, sel, float(p_)))
                    Crec.append(Ce_test[i])
                    Corig.append(parts["test"][2][i])
                    if constr == "plain" and pol != PFR:
                        ref = cases[(cases.fold == yr) & (cases.scenario == nm) & (cases.policy == pol) & (cases.selection == sel)]
                        check(f"GATE_plain_selection_equals_WP63_f{yr}_{nm}_{pol}_{sel}", len(ref) == 1 and abs(float(ref.param.iloc[0]) - p_) < 1e-9)
                    if constr == "s56":
                        ref = simn[(simn.scenario == nm) & (simn.policy == pol) & (simn.selection == sel)]
                        check(f"GATE_s56_selection_equals_WP67_n1_f{yr}_{nm}_{pol}_{sel}", len(ref) == 1 and abs(float(ref.param.iloc[0]) - p_) < 1e-9)
                k = [kk for kk, s in enumerate(specs) if s[0] == pol][0]
                lv, ixo = pol_ix[pol]
                Crec.append(A.path_row(Ce_test[ixo], idx_ad[k], day))
                Corig.append(A.path_row(parts["test"][2][ixo], idx_ad[k], day))
                meta.append((pol, "A_D", float(np.mean(lv[idx_ad[k]]))))
            Crec = np.vstack(Crec)
            Corig = np.vstack(Corig)
            rr = LM.simulate(L_test, pav, Crec, day, H, record=True, **sim_kw)
            rr["_L"] = L_test
            a_f, b_f = LM.FUEL[sc["fuel"]]
            fuelstep = (a_f * rr["rec"]["KU"] + b_f * rr["rec"]["D"]) * LM.DT
            scored = np.isfinite(pav)
            cov = np.isfinite(Corig) & scored[None, :]
            ratings_a = np.asarray(rat_test, float)
            evs = SIM.events_detail(rr["rec"]["ue"], day, rr["rec"]["KU"], rr["rec_on"], ratings_a)
            # N-1 of this construction holds on every recorded step where a set exists
            nfeas = np.zeros(len(meta), int)
            if constr != "plain":
                lg = np.where(rr["rec_on"], ratings_a[None, None, :], 0).max(axis=2)
                if constr == "n1a":
                    need = L_test[None, :] - np.nan_to_num(Corig)
                    _, feas = kmin_n1a(L_test, Corig, rat_test)
                else:
                    need = np.broadcast_to((L_test - np.nan_to_num(parts["test"][2][det]))[None, :], Corig.shape)
                    _, _, feas = W56.k_min(L_test, Corig, parts["test"][2][det], rat_test, True)
                viol = (rr["rec"]["KU"] - lg < need - 1e-6) & scored[None, :] & feas
                nfeas = (~feas & scored[None, :]).sum(axis=1)
                check(f"n1_holds_on_recorded_steps_f{yr}_{nm}", int(viol.sum()) == 0, f"({int(viol.sum())} violations)")
            hbf = float(rr["total"]["fuel_L"][0])
            check(f"record_handbook_equals_totals_f{yr}_{nm}", abs(hbf / hb_fuel - 1) < 1e-12)
            ts = pd.DatetimeIndex(grid)
            for j, (pol, sel, par) in enumerate(meta):
                if pol == "handbook":
                    continue
                rec = SIM.case_record(rr, j, evs[j], keep, cov, 0, fuelstep)
                rec["infeasible_steps"] = int(nfeas[j])
                case_rows.append(dict(fold=yr, construction=constr, shape=shape, scenario=nm, policy=pol, selection=sel,
                                      rule=LABEL[sel], param=par, f_registered=(1 - par) if pol == PFR and sel != "A_D" else np.nan,
                                      fuel_handbook=hbf, saving_pct=100 * (1 - rec["fuel_L"] / hbf),
                                      meets=bool(rec["ue_share"] <= TGT + 1e-15), **rec))
                if constr == "plain" and pol != PFR:
                    ref = simc[(simc.scenario == nm) & (simc.policy == pol) & (simc.selection == sel)]
                    okk = len(ref) == 1 and abs(rec["fuel_L"] / float(ref.fuel_L.iloc[0]) - 1) < 1e-9 \
                        and abs(rec["ue_share"] - float(ref.ue_share.iloc[0])) <= 1e-9 * max(float(ref.ue_share.iloc[0]), 1e-9) \
                        and rec["starts"] == float(ref.starts.iloc[0]) and rec["events"] == int(ref.events.iloc[0]) \
                        and abs(rec["ue_share_covered"] - float(ref.ue_share_covered.iloc[0])) <= 1e-9 * max(float(ref.ue_share_covered.iloc[0]), 1e-9)
                    check(f"GATE_plain_record_equals_WP67_sim_f{yr}_{nm}_{pol}_{sel}", okk)
                if constr == "s56":
                    ref = simn[(simn.scenario == nm) & (simn.policy == pol) & (simn.selection == sel)]
                    okk = len(ref) == 1 and abs(rec["fuel_L"] / float(ref.fuel_L.iloc[0]) - 1) < 1e-9 \
                        and rec["starts"] == float(ref.starts.iloc[0]) and rec["events"] == int(ref.events.iloc[0])
                    check(f"GATE_s56_record_equals_WP67_n1_f{yr}_{nm}_{pol}_{sel}", okk)
                if REF69 is not None and constr == "n1a" and pol != PFR:
                    c69 = REF69["cases"]
                    ref = c69[(c69.fold == yr) & (c69.scenario == nm) & (c69.policy == pol) & (c69.selection == sel)]
                    okk = len(ref) == 1 and all(float(ref[k].iloc[0]) == float(case_rows[-1][k]) for k in
                                                ("param", "fuel_L", "ue_share", "starts", "events", "fuel_handbook"))
                    check(f"GATE_S69b_n1a_record_equals_S69_f{yr}_{nm}_{pol}_{sel}", okk)
                for e, st in zip(evs[j], _event_starts(rr["rec"]["ue"][j], day)):
                    ev_rows.append(dict(fold=yr, construction=constr, shape=shape, policy=pol, selection=sel, steps=e[0],
                                        energy_kWh=e[1], peak_kW=e[2], online_kW=e[3], largest_online_kW=e[4],
                                        start=str(ts[st])))
            log(f"fold {yr} {nm}: done ({time.time() - t0:.0f} s, A-D {passes} passes)")
            del rr, fuelstep, Crec, Corig, Ce_test, res_test, inf_test, cov, evs
            gc.collect()
    # descriptive per fold (computed here, read only by --desc)
    P = prep["test"]
    dayt = np.isfinite(P["pav"]) & (P["pav"] > 0)
    nofc = np.isnan(P["C"][det]) & dayt
    store.setdefault("fallback_pv", []).append(dict(fold=yr, daytime_pv_kWh=float(np.nansum(P["pav"][dayt]) * LM.DT),
                                                    fallback_pv_kWh=float(np.nansum(P["pav"][nofc]) * LM.DT),
                                                    daytime_steps=int(dayt.sum()), fallback_steps=int(nofc.sum())))
    for split in ("calib", "test"):
        c = cache[cache["split"] == split]
        mo = pd.DatetimeIndex(c.index).month
        yy = pd.DatetimeIndex(c.index).year
        for y_ in np.unique(yy):
            s = (yy == y_)
            store.setdefault("winter", []).append(dict(fold=yr, split=split, year=int(y_), issue_times=int(s.sum()),
                                                       winter_issue_times=int((s & np.isin(mo, (6, 7, 8))).sum()),
                                                       winter_transitional=int((s & np.isin(mo, (6, 7, 8)) & (c.g.values == "transitional")).sum()),
                                                       transitional=int((s & (c.g.values == "transitional")).sum())))
    store.setdefault("mondrian", []).extend(mondrian_counts(yr, calib, cases))
    return case_rows, ev_rows, anchor_rows


def _event_starts(ue_row, day):
    pos = ue_row > 1e-9
    return np.flatnonzero(pos & ~np.r_[False, pos[:-1] & (day[1:] == day[:-1])])


def mondrian_counts(yr, calib, cases):
    out = []
    g = np.asarray(calib["g"]).astype(str)
    for pol in ("mondrian", "mondrian_cqr"):
        c = cases[(cases.fold == yr) & (cases.policy == pol) & (cases.selection != "A_D") & (cases.selection != "AW_clip")]
        for th in sorted(set(c.param.astype(float))):
            if th >= 1.0:
                continue
            if pol == "mondrian":
                s = calib["y"] - calib["p"]
                lvl = 1 - th
            else:
                tk = D.tau_key(th)
                s = calib["q"][tk] - calib["y"]
                lvl = th
            for grp in ("clear", "transitional", "cloudy"):
                sg = s[g == grp]
                own = len(sg) > 30
                use = sg if own else s
                q = np.quantile(use, lvl)
                # scores on the failing side of the selected quantile: errors below the lower offset
                # (mondrian), conformity scores above the adjustment (mondrian_cqr)
                tail = int((use < q).sum()) if pol == "mondrian" else int((use > q).sum())
                rules = sorted(set(c[np.isclose(c.param.astype(float), th)].selection.map(LABEL).dropna()))
                out.append(dict(fold=yr, policy=pol, theta=th, group=grp, n_group=int(len(sg)), uses_own_scores=bool(own),
                                n_used=int(len(use)), tail_count=tail, expected_tail=float(len(use) * th), rules=" ".join(rules)))
    return out


# ================================================================ self-test
def selftest(verbose=True):
    res = {}

    def ok(name, cond, msg=""):
        res[name] = bool(cond)
        if verbose:
            print(f"  [{'PASS' if cond else 'FAIL'}] {name} {msg}")

    cap, capn1, small = W56.combo_table(LM.UNITS_BASE)
    rng = np.random.default_rng(7)
    n = 3000
    L = rng.uniform(1500, 6400, n)
    pc = np.vstack([np.where(rng.random(n) < 0.05, np.nan, rng.uniform(0, 1800, n) * f) for f in (0.0, 0.5, 1.0)])
    km, feas = kmin_n1a(L, pc, LM.UNITS_BASE)
    bf = np.full(pc.shape, cap[-1]); bff = np.zeros(pc.shape, bool)
    for k in range(pc.shape[0]):
        for i in range(n):
            p = 0.0 if np.isnan(pc[k, i]) else pc[k, i]
            c = [cap[j] for j in range(len(cap)) if cap[j] >= max(L[i] - p, small) - 1e-9 and capn1[j] >= L[i] - p - 1e-9]
            if c:
                bf[k, i] = min(c); bff[k, i] = True
    ok("n1a_k_min_equals_brute_force", np.array_equal(km, bf) and np.array_equal(feas, bff))
    # N1A with the point forecast as every row's counted PV equals S56's k_min on those rows
    ph = pc[2]
    k56, _, f56 = W56.k_min(L, ph[None, :], ph, LM.UNITS_BASE, True)
    k1a, f1a = kmin_n1a(L, ph[None, :], LM.UNITS_BASE)
    ok("n1a_equals_s56_when_counted_is_point", np.array_equal(k56, k1a) and np.array_equal(f56, f1a))
    # N1A is never below the plain requirement and never below S56 when counted <= point
    k0, _, _ = W56.k_min(L, pc, ph, LM.UNITS_BASE, False)
    ok("n1a_not_below_plain", (km >= k0).all())
    low = np.vstack([np.minimum(np.nan_to_num(pc[k]), np.nan_to_num(ph)) for k in range(3)])
    kl, _ = kmin_n1a(L, low, LM.UNITS_BASE)
    ks, _, _ = W56.k_min(L, low, ph, LM.UNITS_BASE, True)
    ok("n1a_not_below_s56_when_counted_le_point", (kl >= ks).all())
    # simulate requests k_min (toy) and N-1 holds after minimum up time
    grid, Lt, pav, cs = LM._toy(1)
    dd = LM.day_ids(grid)
    C = LM._counted(pav, cs, np.random.default_rng(5))
    Ce, inf_ = effective_n1a(Lt, C, LM.UNITS_BASE)
    r = LM.simulate(Lt, pav, Ce, dd, 3, record=True)
    km_t, feas_t = kmin_n1a(Lt, C, LM.UNITS_BASE)
    sc = np.isfinite(pav)
    ok("online_capacity_at_least_k_min", (r["rec"]["KU"][:, sc] >= km_t[:, sc] - 1e-6).all())
    fleet = np.asarray(LM.UNITS_BASE, float)
    lg = np.where(r["rec_on"], fleet[None, None, :], 0).max(axis=2)
    viol = (r["rec"]["KU"] - lg < (Lt[None, :] - np.nan_to_num(C)) - 1e-6) & sc[None, :] & feas_t
    ok("n1a_holds_after_min_up_time", int(viol.sum()) == 0, f"({int(viol.sum())} violations)")
    # PFR parametrisation: r = 1 counts nothing (handbook), r = 0 counts the persistence PV
    pv = np.array([0.0, 100.0, np.nan, 50.0])
    M = np.vstack([(1.0 - float(x)) * pv for x in D.FRAC_GRID])
    ok("pfr_r1_is_handbook_r0_is_persistence", np.allclose(np.nan_to_num(M[-1]), 0) and np.allclose(M[0], pv, equal_nan=True)
       and np.isnan(M[:, 2]).all())
    ok("frac_grid_symmetric", np.allclose(np.sort(1 - D.FRAC_GRID), D.FRAC_GRID))
    ok("conservative_param_is_max", LM.conservative_param(PFR, D.FRAC_GRID) == 1.0)
    return res


# ================================================================ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--fold", type=int, help="run one fold (folds run in order: PF and PH pool earlier folds)")
    ap.add_argument("--assemble", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--s69b", action="store_true", help="D-137: persistence fraction under N1A, requires S69")
    a = ap.parse_args()
    global CONSTRS, PFR_IN, REF69
    st = selftest()
    if a.selftest or not (a.fold or a.assemble):
        sys.exit(0 if all(st.values()) else 1)
    if not all(st.values()):
        sys.exit("self-test failed")
    root = Path(a.root)
    R = root / "04_results" / "ce1"
    work = root / ("_s69b_work" if a.s69b else "_s69_work")
    work.mkdir(parents=True, exist_ok=True)
    out = R / ("S69b" if a.s69b else "S69")
    if a.s69b:
        CONSTRS, PFR_IN = ("n1a",), {"n1a"}
        REF69 = dict(totals=pd.read_parquet(R / "S69" / "totals.parquet"),
                     cases=pd.read_csv(R / "S69" / "cases.csv", float_precision="round_trip"))
    if a.fold:
        t0 = time.time()
        checks = {f"selftest_{k}": dict(ok=v, msg="") for k, v in st.items()}

        def check(name, cond, msg=""):
            checks[name] = dict(ok=bool(cond), msg=str(msg))
            if not cond:
                log(f"FAIL  {name}  {msg}")
                if name.startswith("CHECK"):
                    (work / f"f{a.fold}_gate_fail.json").write_text(json.dumps(checks, indent=1))
                    sys.exit(f"GATE FAILED: {name} {msg}")

        geo, m, unit96, scales, _ = S6.load_inputs(R / "S1", R / "S1M", R / "S1L")
        D.extend_theta_low()
        rows6 = S6.row_table()
        store = {"test_tabs": {}}
        for y in FOLDS:
            if y < a.fold:
                pk = pickle.load(open(work / f"f{y}.pkl", "rb"))
                store["test_tabs"].update(pk["test_tabs"])
        c, e, an = run_fold(a.fold, R, out, geo, m, unit96, scales, rows6, check, store)
        own = {k: v for k, v in store["test_tabs"].items() if k[0] == a.fold}
        rec = dict(fold=a.fold, cases=c, events=e, anchors=an, test_tabs=own, checks=checks,
                   totals=store["totals"], **{k: store[k] for k in ("fallback_pv", "winter", "mondrian", "pfr_missing_issue")},
                   code=md5(Path(__file__)), seconds=time.time() - t0)
        pickle.dump(rec, open(work / f"f{a.fold}.pkl", "wb"))
        log(f"fold {a.fold}: {sum(v['ok'] for v in checks.values())} of {len(checks)} checks pass ({time.time() - t0:.0f} s)")
        sys.exit(0 if all(v["ok"] for v in checks.values()) else 1)
    recs = [pickle.load(open(work / f"f{y}.pkl", "rb")) for y in FOLDS]
    code = md5(Path(__file__))
    assert all(r["code"] == code for r in recs), "a fold was run with different code"
    checks = {}
    for r in recs:
        checks.update(r["checks"])
    gate = [k for k in checks if k.startswith("CHECK")]
    allok = all(v["ok"] for v in checks.values())
    log(f"gate: {sum(checks[k]['ok'] for k in gate)} of {len(gate)} pass; all checks {sum(v['ok'] for v in checks.values())} of {len(checks)}")
    if not allok:
        sys.exit("CHECK FAILED")
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([x for r in recs for x in r["cases"]]).to_csv(out / "cases.csv", index=False)
    pd.DataFrame([x for r in recs for x in r["events"]]).to_csv(out / "events.csv", index=False)
    pd.DataFrame([x for r in recs for x in r["anchors"]]).to_csv(out / "anchors.csv", index=False)
    pd.concat([t for r in recs for t in r["totals"]], ignore_index=True).to_parquet(out / "totals.parquet")
    for k in ("fallback_pv", "winter", "mondrian", "pfr_missing_issue"):
        pd.DataFrame([x for r in recs for x in r[k]]).to_csv(out / f"desc_{k}.csv", index=False)
    outputs = {p.name: md5(p) for p in sorted(out.glob("*")) if p.is_file()}
    detail = dict(stage="S69b" if a.s69b else "S69", ok=allok, seconds=round(sum(r["seconds"] for r in recs), 1), n_checks=len(checks),
                  n_gate=len(gate), checks=checks, outputs=outputs, code={"ce1_s69.py": code},
                  env=dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__))
    (R / "_runs").mkdir(exist_ok=True)
    (R / "_runs" / ("S69b.detail.json" if a.s69b else "S69.detail.json")).write_text(json.dumps(detail, indent=1), encoding="utf-8")
    log(f"ALL CHECKS PASS ({len(checks)} checks, {len(gate)} gate checks)")


if __name__ == "__main__":
    main()
