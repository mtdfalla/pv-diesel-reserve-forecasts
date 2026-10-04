#!/usr/bin/env python3
"""The held-back test year 2025: dispatch of every selection rule at 15 min, both load shapes, with
nothing fitted or tuned on 2025 (rule D-129).

Usage (from the repository root unless the path starts at ce1/, then from 03_code):
    python3 03_code/ce1/ce1_wp67_f2025.py --fold 2024
"""
from __future__ import annotations

import argparse
import hashlib
import json
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
import ce1_wp61aw as AW  # noqa: E402
import ce1_wp67_sim as SIM  # noqa: E402

H = 3
PRIOR = [2019, 2020, 2021, 2022, 2023, 2024]
TGT, EPS = LM.UE_TARGET, 1e-15
HALF = TGT / 2.0
STATIC = SIM.STATIC
POLS = SIM.POLS
N1 = SIM.N1
CHUNK = 160
QP = ["raw_q", "cqr", "mondrian_cqr"]
CP = ["icp", "mondrian", "cqr", "mondrian_cqr"]


def md5(p):
    return SIM.md5(Path(p))


log = SIM.log


def totals_df(r, rows, sname, part):
    t = pd.DataFrame({k: r["total"][k] for k in S6.TOT_KEYS})
    t.insert(0, "param", [x[1] for x in rows])
    t.insert(0, "policy", [x[0] for x in rows])
    t.insert(0, "set", part)
    t.insert(0, "scenario", sname)
    return t


def sim_chunked(L, pav, C, day, sim_kw):
    """ce1_load.simulate over row chunks (rows are independent, the S6X subset identity)."""
    tots, daily = {}, {}
    for a in range(0, C.shape[0], CHUNK):
        r = LM.simulate(L, pav, C[a:a + CHUNK], day, H, **sim_kw)
        for k, v in r["total"].items():
            tots.setdefault(k, []).append(np.asarray(v))
        for k, v in r["daily"].items():
            daily.setdefault(k, []).append(np.asarray(v))
    return dict(total={k: np.concatenate(v) for k, v in tots.items()},
                daily={k: np.vstack(v) for k, v in daily.items()})


def run(yr: int, R: Path, out: Path, validate: bool) -> bool:
    checks = {}

    def check(name, cond, msg=""):
        checks[name] = dict(ok=bool(cond), msg=str(msg))
        if not cond:
            log(f"FAIL  {name}  {msg}")

    t0 = time.time()
    geo, m, unit96, scales, _ = S6.load_inputs(R / "S1E", R / "S1M", R / "S1L")
    D.extend_theta_low()
    rows = S6.row_table()
    rix = {(p, round(float(x), 6)): i for i, (p, x) in enumerate(rows) if p not in S6.ANCHORS}
    hb = [i for i, r in enumerate(rows) if r[0] == "handbook"][0]
    det = [i for i, r in enumerate(rows) if r[0] == "deterministic"][0]
    if validate:
        cache = FC.load_cache_ext(R / "S2" / f"yulara_f{yr}_h3.parquet", R / "S2X" / f"yulara_f{yr}_h3.parquet")
    else:
        cache = FC.load_cache_ext(R / "S2E" / "f2025" / f"yulara_f{yr}_h3.parquet",
                                  R / "S2E" / "f2025x" / f"yulara_f{yr}_h3.parquet")
    prep = S6.prepare(cache, H, yr, geo, m, rows)
    P = prep["test"]
    grid, pav, day = P["grid"], P["pav"], P["day"]
    # H1x (S57 construction)
    calib = S6.as_dict(cache[cache["split"] == "calib"])
    H1, H2d = D.split_halves(calib)
    cy = yr - 1
    tgt = pd.DatetimeIndex(H1["idx"]) + pd.Timedelta(minutes=5 * H)
    gridx = S6.year_grid(f"{cy}-01-01", f"{cy}-06-30 23:55")
    pavx = LM.available_pv_on_grid(gridx, geo["ghi"].reindex(gridx).values, m)
    pv_of = S6.PVAt(tgt, m)
    pav_t = pv_of(H1["y"])
    check("H1x_target_basis", bool(np.allclose(pav_t, pavx[gridx.get_indexer(tgt)], atol=1e-9, equal_nan=True)))
    Cx = S6.place(gridx, tgt, S6.counted_matrix(rows, H2d, H1, pv_of, pav_t))
    dayx = LM.day_ids(gridx)
    # complete days
    te = S6.as_dict(cache[cache["split"] == "test"])
    cdays = S63.complete_days_fc(te["idx"], geo["is_day"].astype(bool), H, 0.95)
    gd = pd.DatetimeIndex(grid).normalize().unique().values.astype("datetime64[D]")
    keep = np.array([d_ in cdays for d_ in gd])
    if validate:
        s63cd = pd.read_csv(R / "S63" / "complete_days_dispatch.csv")
        check("complete_days_equal_S63", int(keep.sum()) == int(s63cd[s63cd.fold == yr].complete_days.iloc[0]))
        s6e = pd.read_parquet(R / "S6E" / f"f{yr}_h3_totals.parquet")
        s57 = pd.read_parquet(R / "S57" / f"f{yr}_h3_H1x.parquet")
        wcases = pd.read_csv(R / "WP67" / "sim" / f"f{yr}_cases.csv", float_precision="round_trip")
        wn1 = pd.read_csv(R / "WP67" / "sim" / f"f{yr}_n1.csv", float_precision="round_trip")
        pz = np.load(R / "S51" / f"f{yr}_h3_paths.npz")
        specs51 = [(p, float(g), float(k)) for p, g, k in (s.split("|") for s in pz["specs"].astype(str))]
    prior6 = {y: pd.read_parquet(R / "S6E" / f"f{y}_h3_totals.parquet") for y in PRIOR if y < yr}
    prior56 = {y: pd.read_parquet(R / "S56" / f"f{y}_h3_totals.parquet") for y in PRIOR if y < yr}
    pol_C = {}
    for pol in A.POLICIES:
        ix = [i for i, r in enumerate(rows) if r[0] == pol]
        lv = np.array([rows[i][1] for i in ix])
        o = np.argsort(lv)
        pol_C[pol] = (lv[o], np.array(ix)[o])

    def cmp_tot(a, b, name):
        mm = a.merge(b, on=["policy", "param"], suffixes=("", "_ref"))
        worst = max(float(np.max(np.abs(mm[k] - mm[k + "_ref"]) / np.maximum(np.abs(mm[k + "_ref"]), 1.0)))
                    for k in ("fuel_L", "ue_kWh", "starts"))
        check(name, len(mm) == len(a) == len(b) and worst < 1e-9, f"({len(mm)}/{len(a)}/{len(b)} rows, worst {worst:.1e})")

    recs, evrows, selrows, totals_out = [], [], [], []
    # ===================================================== main and shape B
    for sname in ("main", "shapeB"):
        sc = S6.SCENARIOS[sname]
        tabs = {}
        for part, (G, PAV, C, DAY) in {"H2": (prep["H2"]["grid"], prep["H2"]["pav"], prep["H2"]["C"], prep["H2"]["day"]),
                                       "H1x": (gridx, pavx, Cx, dayx), "test": (grid, pav, P["C"], day)}.items():
            L_ = S6.load_for(G, unit96, scales, sc)
            r = sim_chunked(L_, PAV, C, DAY, dict(ratings=LM.fleet_for_peak(L_.max()), min_load=sc["min_load"],
                                                  fuel=sc["fuel"], battery=sc["battery"]))
            tabs[part] = totals_df(r, rows, sname, part)
            check(f"handbook_zero_ue_{sname}_{part}", float(r["total"]["ue_kWh"][hb]) == 0.0)
            if part == "test":
                rtest = r
            if validate:
                if part == "H1x":
                    cmp_tot(tabs[part], s57[s57.scenario == sname], f"H1x_equals_S57_{sname}")
                else:
                    cmp_tot(tabs[part], s6e[(s6e.scenario == sname) & (s6e.set == part)], f"{part}_equals_S6E_{sname}")
            log(f"fold {yr} {sname} {part} simulated ({time.time() - t0:.0f} s)")
        totals_out += [tabs["H2"], tabs["H1x"], tabs["test"]]
        T2, TX, TT = (tabs[k].reset_index(drop=True) for k in ("H2", "H1x", "test"))
        prior = [p_[(p_.scenario == sname) & (p_.set == "test")].reset_index(drop=True) for p_ in prior6.values()]
        L = S6.load_for(grid, unit96, scales, sc)
        ratings = np.asarray(LM.fleet_for_peak(L.max()), float)
        sim_kw = dict(ratings=list(ratings), min_load=sc["min_load"], fuel=sc["fuel"], battery=sc["battery"])
        # daily record as S6E stores it (float32), for the controller's initial lookup
        ue32 = rtest["daily"]["ue_kWh"].astype(np.float32).astype(float)
        load_d = rtest["daily"]["load_kWh"][hb].astype(np.float32).astype(float)
        abar = A.h2_mean_daily_load(prep["H2"], unit96, scales, sc)
        sel_static = {}
        for pol in POLS:
            X = W57.pooled([T2[T2.policy == pol].reset_index(drop=True), TX[TX.policy == pol].reset_index(drop=True)])
            Pp = W57.pooled([t[t.policy == pol].reset_index(drop=True) for t in prior]) if prior else T2[T2.policy == pol].reset_index(drop=True)
            for sel, (tab, tg) in {"A_R": (T2, TGT), "b_xfit_0.002": (X, TGT), "c_prior_0.002": (Pp, TGT),
                                   "a_half_H2": (T2, HALF), "b_xfit_0.001": (X, HALF), "c_prior_0.001": (Pp, HALF)}.items():
                p_, met = W57.select(tab, pol, tg)
                sel_static[(pol, sel)] = float(p_)
                selrows.append(dict(fold=yr, scenario=sname, policy=pol, selection=sel, param=float(p_), met_on_selection=met))
        rec_ue, specs_ad, specs_cl, pc_full = {}, [], [], {}
        for pol in A.POLICIES:
            levels, ixo = pol_C[pol]
            pc_full[pol] = (levels, P["C"][ixo])
            rec_ue[pol] = ue32[ixo]
            t0_ = sel_static[(pol, "A_R")]
            specs_ad.append((pol, A.GAMMA, A.KAPPA, t0_, float(levels.min())))
            specs_cl.append((pol, A.GAMMA, A.KAPPA, t0_, (float(levels.min()), 1.0)))
        rad, idx_ad, _, passes_ad, _ = A.run_paths(L, pav, day, H, sim_kw, pc_full, specs_ad, rec_ue, load_d, abar)
        c0, c1 = A.control, A.control_lookup
        A.control, A.control_lookup = AW.control_clip, AW.control_lookup_clip
        try:
            rcl, idx_cl, _, passes_cl, _ = A.run_paths(L, pav, day, H, sim_kw, pc_full, specs_cl, rec_ue, load_d, abar)
        finally:
            A.control, A.control_lookup = c0, c1
        if validate:
            for k, (pol, g, kp, _, _) in enumerate(specs_ad):
                kk = specs51.index((pol, g, kp))
                check(f"AD_equals_S51_{sname}_{pol}",
                      np.array_equal(rad["daily"]["fuel_L"][k].astype(np.float32), pz[f"{sname}__fuel_L"][kk])
                      and np.array_equal(rad["daily"]["ue_kWh"][k].astype(np.float32), pz[f"{sname}__ue_kWh"][kk]))
        Crows, meta = [P["C"][hb]], [("handbook", "handbook", np.nan)]
        for pol in POLS:
            for sel in STATIC:
                Crows.append(P["C"][rix[(pol, round(sel_static[(pol, sel)], 6))]])
                meta.append((pol, sel, sel_static[(pol, sel)]))
            k = [i for i, s in enumerate(specs_ad) if s[0] == pol][0]
            Crows.append(A.path_row(P["C"][pol_C[pol][1]], idx_ad[k], day))
            meta.append((pol, "A_D", float(np.mean(pol_C[pol][0][idx_ad[k]]))))
            Crows.append(A.path_row(P["C"][pol_C[pol][1]], idx_cl[k], day))
            meta.append((pol, "AW_clip", float(np.mean(pol_C[pol][0][idx_cl[k]]))))
        C = np.vstack(Crows)
        rr = LM.simulate(L, pav, C, day, H, record=True, **sim_kw)
        rr["_L"] = L
        a_f, b_f = LM.FUEL[sc["fuel"]]
        fuelstep = (a_f * rr["rec"]["KU"] + b_f * rr["rec"]["D"]) * LM.DT
        check(f"fuelstep_sums_{sname}", np.allclose(np.nansum(fuelstep, axis=1), rr["total"]["fuel_L"], rtol=1e-9))
        cov = np.isfinite(C) & np.isfinite(pav)[None, :]
        evs = SIM.events_detail(rr["rec"]["ue"], day, rr["rec"]["KU"], rr["rec_on"], ratings)
        hb_fuel = float(rr["total"]["fuel_L"][0])
        for jrow, (pol, sel, par) in enumerate(meta):
            if pol == "handbook":
                continue
            rec = SIM.case_record(rr, jrow, evs[jrow], keep, cov, 0, fuelstep)
            row = dict(fold=yr, scenario=sname, policy=pol, selection=sel, param=par, fuel_handbook=hb_fuel, **rec)
            row["meets"] = bool(rec["ue_share"] <= TGT + EPS)
            row["saving_pct"] = 100 * (1 - rec["fuel_L"] / hb_fuel)
            recs.append(row)
            if validate:
                w = wcases[(wcases.scenario == sname) & (wcases.policy == pol) & (wcases.selection == sel)]
                ok_ = len(w) == 1 and abs(rec["fuel_L"] / w.fuel_L.iloc[0] - 1) < 1e-9 and \
                    abs(rec["ue_kWh"] - w.ue_kWh.iloc[0]) <= 1e-9 * max(w.ue_kWh.iloc[0], 1.0) and \
                    rec["starts"] == w.starts.iloc[0] and len(evs[jrow]) == w.events.iloc[0] and \
                    abs(rec["ue_share_complete"] - w.ue_share_complete.iloc[0]) <= 1e-12 + 1e-9 * abs(w.ue_share_complete.iloc[0])
                if sel in STATIC:
                    ok_ = ok_ and abs(par - w.param.iloc[0]) < 1e-9
                check(f"record_equals_WP67sim_{sname}_{pol}_{sel}", ok_,
                      (rec["fuel_L"], None if len(w) != 1 else w.fuel_L.iloc[0]))
            for e in evs[jrow]:
                evrows.append(dict(fold=yr, scenario=sname, policy=pol, selection=sel, steps=e[0], energy_kWh=e[1],
                                   peak_kW=e[2], online_kW=e[3], largest_online_kW=e[4]))
        log(f"fold {yr} {sname}: records done, A-D {passes_ad} passes, CL {passes_cl} passes ({time.time() - t0:.0f} s)")
        del rr, fuelstep, rtest

    # ===================================================== N-1 (descriptive for 2025)
    n1rows = []
    for sname, base in N1.items():
        sc = W56.SCEN56[sname]
        tabs = {}
        Ce_test = binds_test = None
        for part, (G, PAV, C, DAY) in {"H2": (prep["H2"]["grid"], prep["H2"]["pav"], prep["H2"]["C"], prep["H2"]["day"]),
                                       "H1x": (gridx, pavx, Cx, dayx), "test": (grid, pav, P["C"], day)}.items():
            L_ = S6.load_for(G, unit96, scales, sc)
            rat = LM.fleet_for_peak(L_.max())
            Ce, binds, _ = W56.effective_counted(L_, C, C[det], rat, True)
            r = sim_chunked(L_, PAV, Ce, DAY, dict(ratings=rat, min_load=sc["min_load"], fuel=sc["fuel"], battery=False))
            tabs[part] = totals_df(r, rows, sname, part)
            if part == "test":
                Ce_test, binds_test, rt = Ce, binds, r
            else:
                del Ce, binds
            if validate and part != "H1x":
                s56 = pd.read_parquet(R / "S56" / f"f{yr}_h3_totals.parquet")
                cmp_tot(tabs[part], s56[(s56.scenario == sname) & (s56.set == part)][["policy", "param", "fuel_L", "ue_kWh", "starts"]],
                        f"n1_{part}_equals_S56_{sname}")
        T2, TX = tabs["H2"].reset_index(drop=True), tabs["H1x"].reset_index(drop=True)
        prior = [p_[(p_.scenario == sname) & (p_.set == "test")].reset_index(drop=True) for p_ in prior56.values()]
        L = S6.load_for(grid, unit96, scales, sc)
        ratings = np.asarray(LM.fleet_for_peak(L.max()), float)
        sim_kw = dict(ratings=list(ratings), min_load=sc["min_load"], fuel=sc["fuel"], battery=False)
        load_d = rt["daily"]["load_kWh"][hb]
        abar = A.h2_mean_daily_load(prep["H2"], unit96, scales, sc)
        pc56, ru56, sp56 = {}, {}, []
        selAR = {pol: W57.select(T2, pol, TGT)[0] for pol in A.POLICIES}
        for pol in A.POLICIES:
            lv, ixo = pol_C[pol]
            pc56[pol] = (lv, Ce_test[ixo])
            ru56[pol] = rt["daily"]["ue_kWh"][ixo]
            sp56.append((pol, A.GAMMA, A.KAPPA, float(selAR[pol]), float(lv.min())))
        rp, idx56, _, _, _ = A.run_paths(L, pav, day, H, sim_kw, pc56, sp56, ru56, load_d, abar)
        Crows, meta = [Ce_test[hb]], [("handbook", "handbook", np.nan, np.nan)]
        dayt = np.isfinite(pav) & (pav > 0)
        for pol in POLS:
            X = W57.pooled([T2[T2.policy == pol].reset_index(drop=True), TX[TX.policy == pol].reset_index(drop=True)])
            Pp = W57.pooled([t[t.policy == pol].reset_index(drop=True) for t in prior]) if prior else T2[T2.policy == pol].reset_index(drop=True)
            for sel, (tab, tg) in {"A_R": (T2, TGT), "b_xfit_0.002": (X, TGT), "c_prior_0.002": (Pp, TGT),
                                   "a_half_H2": (T2, HALF), "b_xfit_0.001": (X, HALF), "c_prior_0.001": (Pp, HALF)}.items():
                p_, _ = W57.select(tab, pol, tg)
                i = rix[(pol, round(float(p_), 6))]
                Crows.append(Ce_test[i])
                meta.append((pol, sel, float(p_), float(binds_test[i][dayt].mean())))
            k = [kk for kk, s in enumerate(sp56) if s[0] == pol][0]
            Crows.append(A.path_row(Ce_test[pol_C[pol][1]], idx56[k], day))
            bpath = A.path_row(binds_test[pol_C[pol][1]].astype(float), idx56[k], day)
            meta.append((pol, "A_D", float(np.mean(pol_C[pol][0][idx56[k]])), float(np.nanmean(bpath[dayt]))))
        C = np.vstack(Crows)
        rr = LM.simulate(L, pav, C, day, H, record=True, **sim_kw)
        evs = SIM.events_detail(rr["rec"]["ue"], day, rr["rec"]["KU"], rr["rec_on"], ratings)
        hbf = float(rr["total"]["fuel_L"][0])
        for jrow, (pol, sel, par, bshare) in enumerate(meta):
            if pol == "handbook":
                continue
            evj = evs[jrow]
            row = dict(fold=yr, scenario=sname, base=base, policy=pol, selection=sel, param=par,
                       fuel_L=float(rr["total"]["fuel_L"][jrow]), fuel_handbook=hbf,
                       ue_share=float(rr["total"]["ue_share"][jrow]), starts=float(rr["total"]["starts"][jrow]),
                       events=len(evj), event_peak_max_kW=max([e[2] for e in evj], default=0.0), pv_binds_share=bshare)
            n1rows.append(row)
            if validate:
                w = wn1[(wn1.scenario == sname) & (wn1.policy == pol) & (wn1.selection == sel)]
                check(f"n1_row_equals_WP67sim_{sname}_{pol}_{sel}",
                      len(w) == 1 and abs(row["fuel_L"] / w.fuel_L.iloc[0] - 1) < 1e-9 and row["starts"] == w.starts.iloc[0]
                      and abs(row["ue_share"] - w.ue_share.iloc[0]) <= 1e-12 and row["events"] == w.events.iloc[0],
                      (row["fuel_L"], None if len(w) != 1 else w.fuel_L.iloc[0]))
        log(f"fold {yr} {sname}: N-1 done ({time.time() - t0:.0f} s)")
        del rt, Ce_test, binds_test, rr

    out.mkdir(parents=True, exist_ok=True)
    tag = f"f{yr}"
    pd.DataFrame(recs).to_csv(out / f"{tag}_cases.csv", index=False)
    pd.DataFrame(evrows).to_csv(out / f"{tag}_events.csv", index=False)
    pd.DataFrame(n1rows).to_csv(out / f"{tag}_n1.csv", index=False)
    pd.DataFrame(selrows).to_csv(out / f"{tag}_selections.csv", index=False)
    pd.concat(totals_out, ignore_index=True).to_parquet(out / f"{tag}_h3_totals.parquet")
    ok = all(c["ok"] for c in checks.values())
    inputs = {}
    for p in ([R / "S2" / f"yulara_f{yr}_h3.parquet", R / "S2X" / f"yulara_f{yr}_h3.parquet"] if validate else
              [R / "S2E" / "f2025" / "yulara_f2025_h3.parquet", R / "S2E" / "f2025x" / "yulara_f2025_h3.parquet"]) + \
            [R / "S1E" / "yulara_geo_5min.parquet", R / "S1M" / "available_pv_map.npz", R / "S1L" / "load_profile.npz",
             R / "S1L" / "load_scales.json"] + [R / "S6E" / f"f{y}_h3_totals.parquet" for y in prior6] + \
            [R / "S56" / f"f{y}_h3_totals.parquet" for y in prior56]:
        inputs[str(p.relative_to(R))] = md5(p)
    (out / f"{tag}_checks.json").write_text(json.dumps(dict(ok=ok, n=len(checks), checks=checks, seconds=time.time() - t0,
                                                            inputs=inputs, code={"ce1_wp67_f2025.py": md5(Path(__file__))}),
                                                       indent=1), encoding="utf-8")
    log(f"fold {yr}: {sum(c['ok'] for c in checks.values())} of {len(checks)} checks pass ({time.time() - t0:.0f} s)")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--fold", type=int, required=True)
    a = ap.parse_args()
    R = Path(a.root) / "04_results" / "ce1"
    out = R / "WP67" / "f2025"
    if a.fold == 2025:
        v = out / "validate_f2024" / "f2024_checks.json"
        if not (v.exists() and json.load(open(v))["ok"]):
            sys.exit("fold 2025 refused: the fold-2024 validation has not passed")
        sys.exit(0 if run(2025, R, out, False) else 1)
    sys.exit(0 if run(a.fold, R, out / "validate_f2024", True) else 1)


if __name__ == "__main__":
    main()
