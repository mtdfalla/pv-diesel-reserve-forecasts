"""Start-up fuel, unserved-energy steps and events, and the target sweep, from existing outputs.
Rebuilds the stored Protocol A-D paths of stage S51 and asserts that daily fuel and UE equal
the S51 record before anything is read.

Usage (from the repository root unless the path starts at ce1/, then from 03_code):
    python3 03_code/ce1/ce1_wp55.py --paths
    python3 03_code/ce1/ce1_wp55.py --analyse
"""
from __future__ import annotations

import argparse
import hashlib
import json
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
import ce1_adaptive as A  # noqa: E402
import ce1_dispatch as D  # noqa: E402
import ce1_folds as FC  # noqa: E402
import ce1_load as LM  # noqa: E402
import ce1_s6 as S6  # noqa: E402
import ce1_literal as LT  # noqa: E402  (D-164 phase-aware literal checks)

H = 3
SCEN = ("main", "shapeB")


def md5(p: Path) -> str:
    return hashlib.md5(Path(p).read_bytes()).hexdigest()


def log(m):
    print(time.strftime("%H:%M:%S"), m, flush=True)


def events_of(ue_step: np.ndarray, day: np.ndarray):
    """Per row: (UE events, days with UE). An event is a maximal run of consecutive UE steps
    on the same day (steps are contiguous on the 5-min grid).
    """
    pos = ue_step > 1e-9
    start = pos.copy()
    start[:, 1:] &= ~(pos[:, :-1] & (day[1:] == day[:-1])[None, :])
    ev = start.sum(axis=1)
    nd = int(day.max()) + 1
    ue_days = np.array([np.unique(day[pos[k]]).size for k in range(pos.shape[0])])
    ev_daily = np.stack([np.bincount(day, weights=start[k].astype(float), minlength=nd) for k in range(pos.shape[0])])
    return ev, ue_days, ev_daily


def stage_paths(root: Path, only=None):
    R = root / "04_results" / "ce1"
    out = R / "WP55"
    out.mkdir(parents=True, exist_ok=True)
    checks, inputs = {}, {}

    def check(name, cond, msg=""):
        checks[name] = dict(ok=bool(cond), msg=msg)
        log(f"{'PASS' if cond else 'FAIL'}  {name}  {msg}")

    geo, m, unit96, scales, _ = S6.load_inputs(R / "S1", R / "S1M", R / "S1L")
    D.extend_theta_low()  # S51 ran with --ext (D-092)
    rows = S6.row_table()
    rows_out = []
    folds = [only] if only else A.FOLDS
    for yr in folds:
        t1 = time.time()
        fz = R / "S51" / f"f{yr}_h{H}_paths.npz"
        inputs[str(fz.relative_to(R))] = md5(fz)
        pz = np.load(fz)
        specs = [s.split("|") for s in pz["specs"].astype(str)]
        specs = [(p, float(g), float(k)) for p, g, k in specs]
        cache = FC.load_cache_ext(R / "S2" / f"yulara_f{yr}_h{H}.parquet", R / "S2X" / f"yulara_f{yr}_h{H}.parquet")
        prep = S6.prepare(cache, H, yr, geo, m, rows)
        P = prep["test"]
        grid, pav, day = P["grid"], P["pav"], P["day"]
        pol_C = {}
        for pol in A.POLICIES:
            ix = [i for i, r in enumerate(rows) if r[0] == pol]
            lv = np.array([rows[i][1] for i in ix])
            o = np.argsort(lv)
            pol_C[pol] = (lv[o], P["C"][np.array(ix)[o]])
        rec_sel = json.load(open(R / "S6E" / f"f{yr}_h{H}_select.json"))["selections"]
        inputs[f"S6E/f{yr}_h{H}_select.json"] = md5(R / "S6E" / f"f{yr}_h{H}_select.json")
        daily_out = {}
        for sname in SCEN:
            sc = S6.SCENARIOS[sname]
            L = S6.load_for(grid, unit96, scales, sc)
            sim_kw = dict(ratings=LM.fleet_for_peak(L.max()), min_load=sc["min_load"], fuel=sc["fuel"],
                          battery=sc["battery"])
            lev = pz[f"{sname}__level"].astype(np.float64)  # (K, nd) float32 in the record
            idx = np.empty(lev.shape, np.int64)
            for k, (pol, g, kp) in enumerate(specs):
                levels = pol_C[pol][0]
                j = np.abs(levels[None, :].astype(np.float32).astype(np.float64) - lev[k][:, None]).argmin(axis=1)
                assert np.array_equal(levels[j].astype(np.float32), pz[f"{sname}__level"][k]), f"level lookup {pol}"
                idx[k] = j
            C = np.vstack([A.path_row(pol_C[specs[k][0]][1], idx[k], day) for k in range(len(specs))])
            r = LM.simulate(L, pav, C, day, H, **sim_kw)
            same_f = np.array_equal(r["daily"]["fuel_L"].astype(np.float32), pz[f"{sname}__fuel_L"])
            same_u = np.array_equal(r["daily"]["ue_kWh"].astype(np.float32), pz[f"{sname}__ue_kWh"])
            if not (same_f and same_u):
                df = float(np.max(np.abs(r["daily"]["fuel_L"] - pz[f"{sname}__fuel_L"])))
                du = float(np.max(np.abs(r["daily"]["ue_kWh"] - pz[f"{sname}__ue_kWh"])))
                check(f"reproduces_S51_f{yr}_{sname}", False, f"(max |d fuel| {df:.3g} L/day, |d UE| {du:.3g} kWh/day)")
            else:
                check(f"reproduces_S51_f{yr}_{sname}", True, f"({len(specs)} paths, daily fuel and UE bit for bit)")
            # step-level UE for primary and constant rows
            sel = [k for k, (p, g, kp) in enumerate(specs) if (g == A.GAMMA and kp == A.KAPPA) or g == 0.0]
            rr = LM.simulate(L, pav, C[sel], day, H, record=True, **sim_kw)
            check(f"record_rows_equal_f{yr}_{sname}",
                  np.array_equal(rr["daily"]["fuel_L"], r["daily"]["fuel_L"][sel]))
            ev, ued, evd = events_of(rr["rec"]["ue"], day)
            evmap = {k: (ev[i], ued[i]) for i, k in enumerate(sel)}
            # anchors from the S6E record (handbook, deterministic, oracle) for starts
            anc = rec_sel[sname]["anchors"]
            for k, (pol, g, kp) in enumerate(specs):
                t = {kk: float(v[k]) for kk, v in r["total"].items()}
                row = dict(fold=yr, h=H, scenario=sname, policy=pol, gamma=g, kappa=kp,
                           primary=bool(g == A.GAMMA and kp == A.KAPPA), constant=bool(g == 0.0), **t,
                           ue_days=int((r["daily"]["ue_kWh"][k] > 1e-9).sum()),
                           events=int(evmap[k][0]) if k in evmap else -1,
                           handbook_fuel_L=anc["handbook"]["fuel_L"], handbook_starts=anc["handbook"]["starts"],
                           oracle_fuel_L=anc["oracle"]["fuel_L"], oracle_starts=anc["oracle"]["starts"],
                           det_fuel_L=anc["deterministic"]["fuel_L"], det_starts=anc["deterministic"]["starts"])
                rows_out.append(row)
            if sname == "main":
                ar = [k for k in sel if specs[k][1] == 0.0]
                # the constant rows must equal the S6E A-R record totals (a second, independent anchor)
                okar = all(abs(r["total"]["starts"][k] - rec_sel[sname]["policies"][specs[k][0]]["A_R"]["test"]["starts"]) < 0.5
                           and abs(r["total"]["fuel_L"][k] - rec_sel[sname]["policies"][specs[k][0]]["A_R"]["test"]["fuel_L"]) < 1e-3
                           for k in ar)
                check(f"constant_equals_S6E_AR_starts_f{yr}", okar)
            daily_out[f"{sname}__starts"] = r["daily"]["starts"].astype(np.int32)
            daily_out[f"{sname}__ue_steps"] = r["daily"]["ue_steps"].astype(np.int32)
            daily_out[f"{sname}__events_sel"] = evd.astype(np.int32)
            daily_out[f"{sname}__sel"] = np.array(sel)
        np.savez_compressed(out / f"ad_daily_{yr}.npz", specs=pz["specs"], **daily_out)
        log(f"fold {yr}: {time.time() - t1:.0f} s")
    df = pd.DataFrame(rows_out)
    name = "ad_totals.csv" if not only else f"ad_totals_{only}.csv"
    df.to_csv(out / name, index=False)
    json.dump(dict(checks=checks, inputs=inputs, code={"ce1_wp55.py": md5(Path(__file__))}),
              open(out / ("checks.json" if not only else f"checks_{only}.json"), "w"), indent=1)
    return all(c["ok"] for c in checks.values())


# ================================================================ stage B
NOLOAD_L_PER_H_PER_KW = LM.FUEL["cat_c32"][0]  # 0.0205 L/h per rated kW (model's no-load term)
# Start-up fuel per start = no-load running each start adds outside the committed steps x the
# unit's no-load rate. Central: 5 min cool-down (Caterpillar genset O&M manual M0112246-02,
# Nov 2020, Engine Stopping: "Allow the engine to run for five minutes to cool") on a 1 MW
# unit. Low: half of it. High: twice it (cool-down plus an equal warm-up) on the 2.5 MW unit.
SU_CASES = {"low": (2.5, 1000.0), "central": (5.0, 1000.0), "high": (10.0, 2500.0)}
QP = ["raw_q", "cqr", "mondrian_cqr"]
CPP = ["icp", "mondrian", "cqr", "mondrian_cqr"]
TGT = LM.UE_TARGET
SWEEP = [0.001, 0.0015, 0.002, 0.003, 0.005, 0.0075, 0.01]  # percent of load
BLOCKS = (1, 3, 7, 14)


def su_litres(case):
    mins, kw = SU_CASES[case]
    return mins / 60.0 * NOLOAD_L_PER_H_PER_KW * kw


def block_counts(nd, B, L, seed=A.SEED):
    """(B, nd) day counts of a circular block bootstrap with block length L (L = 1 reproduces
    ce1_decision.draw_counts exactly).
    """
    if L == 1:
        return A.draw_counts(nd, B)
    rng = np.random.default_rng(seed)
    nb = int(np.ceil(nd / L))
    st = rng.integers(0, nd, size=(B, nb))
    idx = ((st[:, :, None] + np.arange(L)[None, None, :]) % nd).reshape(B, -1)[:, :nd]
    C = np.zeros((B, nd))
    np.add.at(C, (np.repeat(np.arange(B), nd), idx.ravel()), 1.0)
    return C


def sign_p(k, n):
    """One-sided exact sign test P(X >= k), X ~ Bin(n, 1/2)."""
    from math import comb
    return sum(comb(n, i) for i in range(k, n + 1)) / 2 ** n


def stage_analyse(root: Path) -> bool:
    R = root / "04_results" / "ce1"
    out = R / "WP55"
    ad = pd.read_csv(out / "ad_totals.csv")
    cells = pd.read_csv(R / "S51" / "cells.csv")
    res, checks = {}, {}
    LIT = LT.Literal(out, "ce1_wp55.py")

    def check(name, cond, msg=""):
        checks[name] = dict(ok=bool(cond), msg=msg)
        log(f"{'PASS' if cond else 'FAIL'}  {name}  {msg}")

    pr = ad[ad.primary]
    cs = ad[ad.constant]
    c3 = cells[(cells.h == H) & cells.primary & cells.scenario.isin(SCEN)]
    mg = pr.merge(c3, on=["fold", "scenario", "policy"], suffixes=("", "_s51"))
    check("stageA_totals_equal_S51_cells", len(mg) == 96 and np.allclose(mg.fuel_L, mg.fuel_L_s51, rtol=0, atol=1e-3)
          and np.allclose(mg.ue_kWh, mg.ue_kWh_s51, rtol=0, atol=1e-6))
    q = pr[pr.policy.isin(QP)].copy()
    q["sav_pct"] = (q.handbook_fuel_L - q.fuel_L) / q.handbook_fuel_L * 100
    check("median_reproduced", LIT("median_pct", round(q.sav_pct.median(), 2), 0.18), f"({q.sav_pct.median():.4f}%)")

    # ---- PR-1 start-up fuel
    q["d_starts"] = q.starts - q.handbook_starts
    pr1 = dict(cells=len(q), saving_median_pct=q.sav_pct.median(),
               d_starts_median=float(q.d_starts.median()), d_starts_min=float(q.d_starts.min()),
               d_starts_max=float(q.d_starts.max()))
    rows = []
    for case in SU_CASES:
        Ls = su_litres(case)
        net = (q.handbook_fuel_L - q.fuel_L - Ls * q.d_starts) / q.handbook_fuel_L * 100
        pr1[f"L_per_start_{case}"] = Ls
        pr1[f"net_median_pct_{case}"] = float(net.median())
        pr1[f"net_min_pct_{case}"] = float(net.min())
        pr1[f"net_max_pct_{case}"] = float(net.max())
        pr1[f"cells_net_le0_{case}"] = int((net <= 0).sum())
        rows.append(q.assign(case=case, L_per_start=Ls, net_pct=net))
    lo, hi = 0.0, 500.0
    for _ in range(80):
        mid = (lo + hi) / 2
        med = ((q.handbook_fuel_L - q.fuel_L - mid * q.d_starts) / q.handbook_fuel_L).median()
        lo, hi = (mid, hi) if med > 0 else (lo, mid)
    pr1["breakeven_L_per_start"] = lo
    pr1["breakeven_min_1MW"] = lo / (NOLOAD_L_PER_H_PER_KW * 1000) * 60
    pr1["breakeven_min_2p5MW"] = lo / (NOLOAD_L_PER_H_PER_KW * 2500) * 60
    pr1["PR1_fires"] = bool(pr1["net_median_pct_central"] <= 0)
    pd.concat(rows).to_csv(out / "pr1_cells.csv", index=False)
    res["PR1"] = pr1
    qa = cs[cs.policy.isin(QP) & (cs.scenario == "main")].copy()
    qa["sav_pct"] = (qa.handbook_fuel_L - qa.fuel_L) / qa.handbook_fuel_L * 100
    check("range_reproduced", LIT("range_pct", (round(qa.sav_pct.min(), 2), round(qa.sav_pct.max(), 2)), (0.34, 0.79)),
          f"({qa.sav_pct.min():.3f}-{qa.sav_pct.max():.3f}%)")
    qa["d_starts"] = qa.starts - qa.handbook_starts
    ar1 = dict(cells=len(qa), d_starts_min=float(qa.d_starts.min()), d_starts_max=float(qa.d_starts.max()))
    for case in SU_CASES:
        net = (qa.handbook_fuel_L - qa.fuel_L - su_litres(case) * qa.d_starts) / qa.handbook_fuel_L * 100
        ar1[f"net_min_pct_{case}"], ar1[f"net_max_pct_{case}"] = float(net.min()), float(net.max())
        ar1[f"net_median_pct_{case}"] = float(net.median())
    res["PR1_AR"] = ar1

    # ---- starts per policy (main, 15 min): handbook, deterministic, oracle, A-R, A-D
    st = []
    for sname in SCEN:
        for pol in A.POLICIES:
            a_r = cs[(cs.scenario == sname) & (cs.policy == pol)]
            a_d = pr[(pr.scenario == sname) & (pr.policy == pol)]
            st.append(dict(scenario=sname, policy=pol, AR_starts_median=a_r.starts.median(),
                           AD_starts_median=a_d.starts.median(), AR_starts_min=a_r.starts.min(),
                           AR_starts_max=a_r.starts.max(), AD_starts_min=a_d.starts.min(), AD_starts_max=a_d.starts.max()))
        a0 = pr[pr.scenario == sname].drop_duplicates("fold")
        st.append(dict(scenario=sname, policy="handbook", AR_starts_median=a0.handbook_starts.median(),
                       AR_starts_min=a0.handbook_starts.min(), AR_starts_max=a0.handbook_starts.max()))
        st.append(dict(scenario=sname, policy="deterministic", AR_starts_median=a0.det_starts.median(),
                       AR_starts_min=a0.det_starts.min(), AR_starts_max=a0.det_starts.max()))
        st.append(dict(scenario=sname, policy="oracle", AR_starts_median=a0.oracle_starts.median(),
                       AR_starts_min=a0.oracle_starts.min(), AR_starts_max=a0.oracle_starts.max()))
    pd.DataFrame(st).to_csv(out / "starts_summary.csv", index=False)

    # ---- UE events, UE days, UE steps per policy and year (A-R = constant rows, A-D = primary)
    ev = pd.concat([cs.assign(protocol="A-R"), pr.assign(protocol="A-D")])[
        ["fold", "scenario", "policy", "protocol", "ue_kWh", "ue_share", "ue_steps", "events", "ue_days", "load_kWh"]]
    ev.to_csv(out / "ue_events.csv", index=False)
    evc = ev[ev.policy.isin(CPP)]
    res["events_conformal"] = {p: dict(events_median=float(g.events.median()), events_min=int(g.events.min()),
                                       events_max=int(g.events.max()), ue_days_median=float(g.ue_days.median()),
                                       ue_days_max=int(g.ue_days.max()), ue_steps_median=float(g.ue_steps.median()))
                               for p, g in evc.groupby("protocol")}
    evq = ev[ev.policy.isin(QP)]
    res["events_quantile"] = {p: dict(events_median=float(g.events.median()), events_min=int(g.events.min()),
                                      events_max=int(g.events.max()), ue_days_median=float(g.ue_days.median()))
                              for p, g in evq.groupby("protocol")}
    check("events_le_steps [definition identity]", bool((ev.events <= ev.ue_steps).all() and (ev.ue_days <= ev.events).all()))

    # ---- target sweep as a curve (the fixed selections evaluated against other targets)
    sw = []
    for t in SWEEP:
        for p, g in ev.groupby("protocol"):
            gc = g[g.policy.isin(CPP)]
            sw.append(dict(target_pct=t, protocol=p, met_conformal=int((gc.ue_share <= t / 100 + 1e-15).sum()),
                           n_conformal=len(gc), met_all=int((g.ue_share <= t / 100 + 1e-15).sum()), n_all=len(g)))
    sw = pd.DataFrame(sw)
    sw.to_csv(out / "target_sweep.csv", index=False)
    s2 = sw[sw.target_pct == 0.002].set_index("protocol")
    check("sweep_reproduces", LIT("sweep_met_AD_AR_conformal_all",
                                       (int(s2.loc["A-D", "met_conformal"]), int(s2.loc["A-R", "met_conformal"]),
                                        int(s2.loc["A-D", "met_all"]), int(s2.loc["A-R", "met_all"])), (48, 25, 96, 58)))
    res["target_sweep"] = sw.to_dict("records")

    # ---- sign tests over the six test years (fold is the unit; cells within a fold are dependent)
    per = []
    for yr in A.FOLDS:
        g = ev[(ev.fold == yr) & ev.policy.isin(CPP)]
        a_r, a_d = g[g.protocol == "A-R"], g[g.protocol == "A-D"]
        m = a_r.merge(a_d, on=["scenario", "policy"], suffixes=("_AR", "_AD"))
        per.append(dict(fold=yr, met_AR=int((m.ue_share_AR <= TGT + 1e-15).sum()),
                        met_AD=int((m.ue_share_AD <= TGT + 1e-15).sum()), n=len(m),
                        ue_AD_lower=int((m.ue_kWh_AD < m.ue_kWh_AR).sum()),
                        events_AD_fewer=int((m.events_AD < m.events_AR).sum()),
                        median_ue_share_AR=float(m.ue_share_AR.median()), median_ue_share_AD=float(m.ue_share_AD.median()),
                        median_events_AR=float(m.events_AR.median()), median_events_AD=float(m.events_AD.median())))
    per = pd.DataFrame(per)
    per.to_csv(out / "sign_by_year.csv", index=False)
    k_met = int((per.met_AD > per.met_AR).sum())
    n_met = int((per.met_AD != per.met_AR).sum())
    k_ue = int((per.median_ue_share_AD < per.median_ue_share_AR).sum())
    n_ue = int((per.median_ue_share_AD != per.median_ue_share_AR).sum())
    res["sign_test"] = dict(years_more_met_AD=k_met, years_differ_met=n_met, p_met=sign_p(k_met, n_met) if n_met else np.nan,
                            years_lower_median_ue_AD=k_ue, years_differ_ue=n_ue, p_ue=sign_p(k_ue, n_ue) if n_ue else np.nan,
                            fuel_years_AD_above_AR=int(sum(
                                (pr[(pr.fold == y) & pr.policy.isin(QP)].fuel_L.median()
                                 > cs[(cs.fold == y) & cs.policy.isin(QP)].fuel_L.median()) for y in A.FOLDS)))

    # ---- handbook fallback and skipped steps (15 min, test years; identical for every policy)
    fb = []
    for yr in A.FOLDS:
        s = json.load(open(R / "S6E" / f"f{yr}_h{H}_select.json"))["selections"]["main"]
        hb = s["anchors"]["handbook"]
        fb.append(dict(fold=yr, fallback_steps=int(hb["fallback_steps"]), scored_steps=int(hb["scored_steps"]),
                       missing_steps_test=int(s["missing_steps"]["test"]),
                       oracle_fallback=int(s["anchors"]["oracle"]["fallback_steps"])))
        one = ad[(ad.fold == yr) & (ad.scenario == "main")]
        check(f"fallback_same_all_policies_f{yr}", bool((one.fallback_steps == hb["fallback_steps"]).all()))
    fb = pd.DataFrame(fb)
    fb.to_csv(out / "fallback_missing.csv", index=False)
    res["fallback"] = dict(fallback_min=int(fb.fallback_steps.min()), fallback_max=int(fb.fallback_steps.max()),
                           missing_min=int(fb.missing_steps_test.min()), missing_max=int(fb.missing_steps_test.max()),
                           fallback_share_min=float((fb.fallback_steps / fb.scored_steps).min()),
                           fallback_share_max=float((fb.fallback_steps / fb.scored_steps).max()))

    # ---- kappa grid: conformal cells meeting the target, 15 min, main and shapeB
    kg = cells[(cells.h == H) & cells.scenario.isin(SCEN) & cells.policy.isin(CPP) & (cells.gamma > 0)]
    kt = (kg.assign(met=kg.ue_share <= TGT + 1e-15).groupby(["gamma", "kappa"]).met.agg(["sum", "count"]).reset_index())
    kt.to_csv(out / "kappa_grid_met.csv", index=False)
    k1 = kt[(kt.gamma == 0.005) & (kt.kappa == 1.0)].iloc[0]
    check("kappa1_gamma0005_is_14_of_48", LIT("kappa1_gamma0005_met", int(k1["sum"]), 14) and int(k1["count"]) == 48, f"({int(k1['sum'])} of {int(k1['count'])})")
    res["kappa_grid"] = kt.to_dict("records")

    # ---- battery additivity under A-D at matched reliability (15 min, quantile policies)
    ba = []
    for shp, sn, sb in (("A", "main", "battery"), ("B", "shapeB", "batteryB")):
        for yr in A.FOLDS:
            for pol in QP:
                n0 = cells[(cells.fold == yr) & (cells.h == H) & (cells.scenario == sn) & (cells.policy == pol) & cells.primary].iloc[0]
                b0 = cells[(cells.fold == yr) & (cells.h == H) & (cells.scenario == sb) & (cells.policy == pol) & cells.primary].iloc[0]
                hb_nb, hb_wb = n0.fuel_handbook, b0.fuel_handbook
                ba.append(dict(shape=shp, fold=yr, policy=pol,
                               battery_alone_pct=(hb_nb - hb_wb) / hb_nb * 100,
                               forecast_alone_pct=(hb_nb - n0.fuel_L) / hb_nb * 100,
                               combined_pct=(hb_nb - b0.fuel_L) / hb_nb * 100,
                               met_nb=bool(n0.ue_share <= TGT + 1e-15), met_wb=bool(b0.ue_share <= TGT + 1e-15)))
    ba = pd.DataFrame(ba)
    ba["interaction_pct"] = ba.combined_pct - ba.battery_alone_pct - ba.forecast_alone_pct
    ba.to_csv(out / "battery_additivity_AD.csv", index=False)
    both = ba[ba.met_nb & ba.met_wb]
    res["battery_AD"] = dict(cells=len(ba), cells_both_met=len(both),
                             battery_alone_median=float(ba.battery_alone_pct.median()),
                             forecast_alone_median=float(ba.forecast_alone_pct.median()),
                             combined_median=float(ba.combined_pct.median()),
                             interaction_median=float(ba.interaction_pct.median()),
                             interaction_min=float(ba.interaction_pct.min()), interaction_max=float(ba.interaction_pct.max()),
                             cells_interaction_negative=int((ba.interaction_pct < 0).sum()))

    # ---- block-length sensitivity of the A-D minus A-R and A-D minus handbook fuel intervals
    boot = pd.read_csv(R / "S51" / "boot.csv")
    bl = []
    for yr in A.FOLDS:
        pz = np.load(R / "S51" / f"f{yr}_h{H}_paths.npz")
        specs = [(p, float(g), float(k)) for p, g, k in (s.split("|") for s in pz["specs"].astype(str))]
        z = np.load(R / "S6E" / f"f{yr}_h{H}_daily.npz")
        rmap, amap = A.record_rows(z)
        for sname in SCEN:
            nd = pz[f"{sname}__fuel_L"].shape[1]
            hb = z[f"{sname}__fuel_L"][amap["handbook"]].astype(float)
            for L in BLOCKS:
                Cn = block_counts(nd, A.B_DEF, L)
                for k, (pol, g, kp) in enumerate(specs):
                    if not (g == A.GAMMA and kp == A.KAPPA) or pol not in QP:
                        continue
                    row = cells[(cells.fold == yr) & (cells.h == H) & (cells.scenario == sname) & (cells.policy == pol) & cells.primary].iloc[0]
                    ar = z[f"{sname}__fuel_L"][rmap[(pol, round(row.theta0, 6))]].astype(float)
                    f = pz[f"{sname}__fuel_L"][k].astype(float)
                    c1, c2 = A.ci(Cn @ (f - ar)), A.ci(Cn @ (f - hb))
                    bl.append(dict(fold=yr, scenario=sname, policy=pol, block=L, d_AR=float((f - ar).sum()),
                                   ci_AR_lo=c1[0], ci_AR_hi=c1[1], d_hb=float((f - hb).sum()), ci_hb_lo=c2[0], ci_hb_hi=c2[1]))
    bl = pd.DataFrame(bl)
    bl.to_csv(out / "block_sensitivity.csv", index=False)
    # L = 1 must reproduce the S51 intervals
    b1 = bl[bl.block == 1].merge(boot[boot.h == H], on=["fold", "scenario", "policy"])
    lo_s51 = b1.d_fuel_AR_ci.map(lambda s: float(s.strip("()").split(",")[0]))
    check("block1_reproduces_S51_boot", len(b1) == 36 and np.allclose(b1.ci_AR_lo, lo_s51, rtol=1e-9, atol=1e-6))
    res["block"] = {int(L): dict(AR_above_zero=int((g.ci_AR_lo > 0).sum()), hb_below_zero=int((g.ci_hb_hi < 0).sum()),
                                 hb_above_zero=int((g.ci_hb_lo > 0).sum()), n=len(g))
                    for L, g in bl.groupby("block")}
    check("block1_AR_above_zero_is_30_of_36", LIT("block1_AR_above_zero", res["block"][1]["AR_above_zero"], 30), f"({res['block'][1]['AR_above_zero']})")

    json.dump(dict(results=res, checks=checks, code={"ce1_wp55.py": md5(Path(__file__))},
                   inputs={"WP55/ad_totals.csv": md5(out / "ad_totals.csv"), "S51/cells.csv": md5(R / "S51" / "cells.csv"),
                           "S51/boot.csv": md5(R / "S51" / "boot.csv")}),
              open(out / "results.json", "w"), indent=1, default=float)
    return all(c["ok"] for c in checks.values())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--paths", action="store_true")
    ap.add_argument("--analyse", action="store_true")
    ap.add_argument("--only", type=int, default=None)
    a = ap.parse_args()
    root = Path(a.root)
    if a.paths:
        ok = stage_paths(root, a.only)
        sys.exit(0 if ok else 1)
    if a.analyse:
        ok = stage_analyse(root)
        sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
