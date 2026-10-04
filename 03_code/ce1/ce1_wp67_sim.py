#!/usr/bin/env python3
"""Re-simulation of every selection rule at 15 min with the step record, from the stored inputs and
the unchanged dispatch, for the D-129 step 2 items.

Usage (from the repository root unless the path starts at ce1/, then from 03_code):
    python3 03_code/ce1/ce1_wp67_sim.py --fold 2024
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

H = 3
FOLDS = [2019, 2020, 2021, 2022, 2023, 2024]
TGT, EPS = LM.UE_TARGET, 1e-15
HALF = TGT / 2.0
STATIC = ["A_R", "b_xfit_0.002", "c_prior_0.002", "a_half_H2", "b_xfit_0.001", "c_prior_0.001"]
POLS = ["icp", "mondrian", "cqr", "mondrian_cqr", "raw_q", "tod_q", "nicp_var", "frac_pv"]
N1 = {"cont": "main", "contB": "shapeB"}


def md5(p: Path) -> str:
    h = hashlib.md5()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def log(m):
    print(time.strftime("%H:%M:%S"), m, flush=True)


def events_detail(ue, day, KU, on, ratings):
    """Per row: list of (steps, energy kWh, peak kW, online kW at peak, largest online unit kW at peak)."""
    out = []
    for j in range(ue.shape[0]):
        pos = ue[j] > 1e-9
        st = np.flatnonzero(pos & ~np.r_[False, pos[:-1] & (day[1:] == day[:-1])])
        evs = []
        for s0 in st:
            s1 = s0
            while s1 + 1 < len(pos) and pos[s1 + 1] and day[s1 + 1] == day[s1]:
                s1 += 1
            seg = ue[j, s0:s1 + 1]
            pk = s0 + int(np.argmax(seg))
            o = on[j, pk]
            evs.append((int(s1 - s0 + 1), float(seg.sum() * LM.DT), float(seg.max()), float(KU[j, pk]),
                        float(ratings[o].max()) if o.any() else 0.0))
        out.append(evs)
    return out


def case_record(rr, j, evs, keep_days, cov, hb_j, fuelstep):
    """Totals, events, complete-day and covered-step figures of row j of a record run."""
    t = rr["total"]
    d = rr["daily"]
    peaks = [e[2] for e in evs]
    shares = [e[2] / e[3] for e in evs if e[3] > 0]
    ue_c = float(d["ue_kWh"][j][keep_days].sum())
    ld_c = float(d["load_kWh"][j][keep_days].sum())
    f_c = float(d["fuel_L"][j][keep_days].sum())
    fh_c = float(d["fuel_L"][hb_j][keep_days].sum())
    ue_cov = float(np.nansum(rr["rec"]["ue"][j][cov[j]]) * LM.DT)
    ld_cov = float(np.nansum(rr["_L"][cov[j]]) * LM.DT)
    f_cov = float(np.nansum(fuelstep[j][cov[j]]))
    fh_cov = float(np.nansum(fuelstep[hb_j][cov[j]]))
    return dict(fuel_L=float(t["fuel_L"][j]), ue_kWh=float(t["ue_kWh"][j]), load_kWh=float(t["load_kWh"][j]),
                ue_share=float(t["ue_share"][j]), starts=float(t["starts"][j]),
                events=len(evs), event_peak_max_kW=max(peaks) if peaks else 0.0,
                event_peak_median_kW=float(np.median(peaks)) if peaks else 0.0,
                event_share_online_max=max(shares) if shares else 0.0,
                event_share_largest_max=max([e[2] / e[4] for e in evs if e[4] > 0], default=0.0),
                ue_share_complete=ue_c / ld_c if ld_c > 0 else np.nan, saving_complete_pct=100 * (1 - f_c / fh_c),
                ue_share_covered=ue_cov / ld_cov if ld_cov > 0 else np.nan, saving_covered_pct=100 * (1 - f_cov / fh_cov),
                covered_steps=int(cov[j].sum()))


def run_fold(yr: int, R: Path, out: Path) -> bool:
    checks = {}

    def check(name, cond, msg=""):
        checks[name] = dict(ok=bool(cond), msg=str(msg))
        if not cond:
            log(f"FAIL  {name}  {msg}")

    t0 = time.time()
    geo, m, unit96, scales, _ = S6.load_inputs(R / "S1", R / "S1M", R / "S1L")
    D.extend_theta_low()
    rows = S6.row_table()
    rix = {(p, round(float(x), 6)): i for i, (p, x) in enumerate(rows) if p not in S6.ANCHORS}
    hb = [i for i, r in enumerate(rows) if r[0] == "handbook"][0]
    det = [i for i, r in enumerate(rows) if r[0] == "deterministic"][0]
    orc = [i for i, r in enumerate(rows) if r[0] == "oracle"][0]
    cache = FC.load_cache_ext(R / "S2" / f"yulara_f{yr}_h3.parquet", R / "S2X" / f"yulara_f{yr}_h3.parquet")
    prep = S6.prepare(cache, H, yr, geo, m, rows)
    P = prep["test"]
    grid, pav, day = P["grid"], P["pav"], P["day"]
    # complete days (S63 item 7 set)
    te = S6.as_dict(cache[cache["split"] == "test"])
    cdays = S63.complete_days_fc(te["idx"], geo["is_day"].astype(bool), H, 0.95)
    gd = pd.DatetimeIndex(grid).normalize().unique().values.astype("datetime64[D]")
    keep = np.array([d_ in cdays for d_ in gd])
    s63cd = pd.read_csv(R / "S63" / "complete_days_dispatch.csv")
    check("complete_days_equal_S63", int(keep.sum()) == int(s63cd[s63cd.fold == yr].complete_days.iloc[0]),
          (int(keep.sum()), int(s63cd[s63cd.fold == yr].complete_days.iloc[0])))
    cases = pd.read_csv(R / "WP63" / "cases.csv", float_precision="round_trip")
    cases = cases[cases.fold == yr]
    aw = pd.read_csv(R / "AW" / "cells.csv", float_precision="round_trip")
    aw = aw[(aw.fold == yr) & np.isclose(aw.clip_h.astype(float), 1.0)]
    pz = np.load(R / "S51" / f"f{yr}_h3_paths.npz")
    specs = [(p, float(g), float(k)) for p, g, k in (s.split("|") for s in pz["specs"].astype(str))]
    z = np.load(R / "S6E" / f"f{yr}_h3_daily.npz")
    sel_e = json.load(open(R / "S6E" / f"f{yr}_h3_select.json"))["selections"]
    pol_C = {}
    for pol in A.POLICIES:
        ix = [i for i, r in enumerate(rows) if r[0] == pol]
        lv = np.array([rows[i][1] for i in ix])
        o = np.argsort(lv)
        pol_C[pol] = (lv[o], np.array(ix)[o])
    recs, evrows = [], []

    # ============================================================ main and shape B (items 4, 5, 13)
    for sname in ("main", "shapeB"):
        sc = S6.SCENARIOS[sname]
        L = S6.load_for(grid, unit96, scales, sc)
        ratings = np.asarray(LM.fleet_for_peak(L.max()), float)
        sim_kw = dict(ratings=list(ratings), min_load=sc["min_load"], fuel=sc["fuel"], battery=sc["battery"])
        # --- CL paths (anti-windup h = 1.0) with the wp61aw controller
        rmap, amap = A.record_rows(z)
        load_d = z[f"{sname}__load_kWh"][amap["handbook"]].astype(float)
        abar = A.h2_mean_daily_load(prep["H2"], unit96, scales, sc)
        rec_ue, cspecs, pc_full = {}, [], {}
        for pol in A.POLICIES:
            levels, ixo = pol_C[pol]
            pc_full[pol] = (levels, P["C"][ixo])
            rrx = np.array([rmap[(pol, round(float(x), 6))] for x in levels])
            rec_ue[pol] = z[f"{sname}__ue_kWh"][rrx].astype(float)
            t0_ = float(sel_e[sname]["policies"][pol]["A_R"]["param"])
            cspecs.append((pol, A.GAMMA, A.KAPPA, t0_, (float(levels.min()), 1.0)))
        c0, c1 = A.control, A.control_lookup
        A.control, A.control_lookup = AW.control_clip, AW.control_lookup_clip
        try:
            rcl, idx_cl, _, passes, _ = A.run_paths(L, pav, day, H, sim_kw, pc_full, cspecs, rec_ue, load_d, abar)
        finally:
            A.control, A.control_lookup = c0, c1
        # --- rows of the record run
        Crows, meta = [P["C"][hb]], [("handbook", "handbook", np.nan)]
        for pol in POLS:
            cc = cases[(cases.scenario == sname) & (cases.policy == pol)]
            for sel in STATIC:
                r_ = cc[cc.selection == sel].iloc[0]
                Crows.append(P["C"][rix[(pol, round(float(r_.param), 6))]])
                meta.append((pol, sel, float(r_.param)))
            k = specs.index((pol, A.GAMMA, A.KAPPA))
            lev = pz[f"{sname}__level"][k].astype(np.float64)
            levels = pol_C[pol][0]
            j = np.abs(levels[None, :].astype(np.float32).astype(np.float64) - lev[:, None]).argmin(axis=1)
            check(f"AD_level_map_f{yr}_{sname}_{pol}", np.array_equal(levels[j].astype(np.float32), pz[f"{sname}__level"][k]))
            Crows.append(A.path_row(P["C"][pol_C[pol][1]], j, day))
            meta.append((pol, "A_D", np.nan))
            kc = [i for i, s in enumerate(cspecs) if s[0] == pol][0]
            Crows.append(A.path_row(P["C"][pol_C[pol][1]], idx_cl[kc], day))
            meta.append((pol, "AW_clip", np.nan))
        C = np.vstack(Crows)
        rr = LM.simulate(L, pav, C, day, H, record=True, **sim_kw)
        rr["_L"] = L
        a_f, b_f = LM.FUEL[sc["fuel"]]
        fuelstep = (a_f * rr["rec"]["KU"] + b_f * rr["rec"]["D"]) * LM.DT
        check(f"fuelstep_sums_f{yr}_{sname}", np.allclose(np.nansum(fuelstep, axis=1), rr["total"]["fuel_L"], rtol=1e-9))
        scored = np.isfinite(pav)
        cov = np.isfinite(C) & scored[None, :]
        evs = events_detail(rr["rec"]["ue"], day, rr["rec"]["KU"], rr["rec_on"], ratings)
        hb_fuel = float(rr["total"]["fuel_L"][0])
        for jrow, (pol, sel, par) in enumerate(meta):
            if pol == "handbook":
                check(f"handbook_zero_ue_f{yr}_{sname}", float(rr["total"]["ue_kWh"][0]) == 0.0)
                continue
            rec = case_record(rr, jrow, evs[jrow], keep, cov, 0, fuelstep)
            if sel in STATIC:
                ref = cases[(cases.scenario == sname) & (cases.policy == pol) & (cases.selection == sel)].iloc[0]
                check(f"static_reproduces_WP63_f{yr}_{sname}_{pol}_{sel}",
                      abs(rec["fuel_L"] / ref.fuel_L - 1) < 1e-9 and abs(rec["ue_share"] - ref.ue_share) <= 1e-9 * max(ref.ue_share, 1e-9)
                      and rec["starts"] == ref.starts, (rec["fuel_L"], ref.fuel_L, rec["starts"], ref.starts))
            elif sel == "A_D":
                k = specs.index((pol, A.GAMMA, A.KAPPA))
                check(f"AD_reproduces_S51_f{yr}_{sname}_{pol}",
                      np.array_equal(rr["daily"]["fuel_L"][jrow].astype(np.float32), pz[f"{sname}__fuel_L"][k])
                      and np.array_equal(rr["daily"]["ue_kWh"][jrow].astype(np.float32), pz[f"{sname}__ue_kWh"][k]))
                ref = cases[(cases.scenario == sname) & (cases.policy == pol) & (cases.selection == "A_D")].iloc[0]
                check(f"AD_reproduces_WP63_f{yr}_{sname}_{pol}", abs(rec["fuel_L"] / ref.fuel_L - 1) < 1e-9 and rec["starts"] == ref.starts,
                      (rec["fuel_L"], ref.fuel_L, rec["starts"], ref.starts))
            else:
                ref = aw[(aw.scenario == sname) & (aw.policy == pol)].iloc[0]
                check(f"CL_reproduces_AW_f{yr}_{sname}_{pol}",
                      abs(rec["fuel_L"] / ref.fuel_L - 1) < 1e-9 and abs(rec["ue_share"] - ref.ue_share) <= 1e-9 * max(ref.ue_share, 1e-9),
                      (rec["fuel_L"], ref.fuel_L, rec["ue_share"], ref.ue_share))
            recs.append(dict(fold=yr, scenario=sname, policy=pol, selection=sel, param=par, fuel_handbook=hb_fuel, **rec))
            for e in evs[jrow]:
                evrows.append(dict(fold=yr, scenario=sname, policy=pol, selection=sel, steps=e[0], energy_kWh=e[1],
                                   peak_kW=e[2], online_kW=e[3], largest_online_kW=e[4]))
        log(f"fold {yr} {sname}: {len(meta)} rows, CL {passes} passes ({time.time() - t0:.0f} s)")
        del rr, fuelstep

    # ============================================================ N-1
    s56 = pd.read_parquet(R / "S56" / f"f{yr}_h3_totals.parquet")
    sel56 = json.load(open(R / "S56" / f"f{yr}_h3_select.json"))
    ad56 = pd.read_csv(R / "S56" / "ad_cells.csv", float_precision="round_trip")
    prior56 = {y: pd.read_parquet(R / "S56" / f"f{y}_h3_totals.parquet") for y in FOLDS if y < yr}
    # H1x record (S64 construction) under N-1
    calib = S6.as_dict(cache[cache["split"] == "calib"])
    H1, H2d = D.split_halves(calib)
    cy = yr - 1
    tgt = pd.DatetimeIndex(H1["idx"]) + pd.Timedelta(minutes=5 * H)
    gridx = S6.year_grid(f"{cy}-01-01", f"{cy}-06-30 23:55")
    pavx = LM.available_pv_on_grid(gridx, geo["ghi"].reindex(gridx).values, m)
    pv_of = S6.PVAt(tgt, m)
    Cx = S6.place(gridx, tgt, S6.counted_matrix(rows, H2d, H1, pv_of, pv_of(H1["y"])))
    dayx = LM.day_ids(gridx)
    n1rows = []
    for sname, base in N1.items():
        sc = W56.SCEN56[sname]
        # H1x under N-1
        Lx = S6.load_for(gridx, unit96, scales, sc)
        ratx = LM.fleet_for_peak(Lx.max())
        Cex, _, _ = W56.effective_counted(Lx, Cx, Cx[det], ratx, True)
        rx = LM.simulate(Lx, pavx, Cex, dayx, H, ratings=ratx, min_load=sc["min_load"], fuel=sc["fuel"], battery=False)
        tx = pd.DataFrame({k: rx["total"][k] for k in ("fuel_L", "ue_kWh", "load_kWh", "ue_share", "starts")})
        tx.insert(0, "param", [x[1] for x in rows])
        tx.insert(0, "policy", [x[0] for x in rows])
        check(f"n1_H1x_handbook_zero_ue_f{yr}_{sname}", float(rx["total"]["ue_kWh"][hb]) == 0.0)
        T2 = s56[(s56.scenario == sname) & (s56.set == "H2")].reset_index(drop=True)
        TT = s56[(s56.scenario == sname) & (s56.set == "test")].reset_index(drop=True)
        prior = [p_[(p_.scenario == sname) & (p_.set == "test")].reset_index(drop=True) for p_ in prior56.values()]
        # test year under N-1, every row (for the A-D record and the selected rows)
        L = S6.load_for(grid, unit96, scales, sc)
        ratings = np.asarray(LM.fleet_for_peak(L.max()), float)
        sim_kw = dict(ratings=list(ratings), min_load=sc["min_load"], fuel=sc["fuel"], battery=False)
        Ce, binds, _ = W56.effective_counted(L, P["C"], P["C"][det], list(ratings), True)
        rt = LM.simulate(L, pav, Ce, day, H, **sim_kw)
        worst = max(float(np.max(np.abs(rt["total"][k] - TT[k].to_numpy(float)) / np.maximum(np.abs(TT[k].to_numpy(float)), 1.0)))
                    for k in ("fuel_L", "ue_kWh", "starts"))
        check(f"n1_test_reproduces_S56_f{yr}_{sname}", worst < 1e-9, f"(worst {worst:.1e})")
        # A-D under N-1 from the scenario's own record (S56 ad_block construction)
        load_d = rt["daily"]["load_kWh"][hb]
        abar = A.h2_mean_daily_load(prep["H2"], unit96, scales, sc)
        pc56, ru56, sp56 = {}, {}, []
        for pol in A.POLICIES:
            lv, ixo = pol_C[pol]
            pc56[pol] = (lv, Ce[ixo])
            ru56[pol] = rt["daily"]["ue_kWh"][ixo]
            sp56.append((pol, A.GAMMA, A.KAPPA, float(sel56[sname]["policies"][pol]["A_R"]["param"]), float(lv.min())))
        rp, idx56, _, _, _ = A.run_paths(L, pav, day, H, sim_kw, pc56, sp56, ru56, load_d, abar)
        # selections under N-1
        Crows, meta = [Ce[hb]], [("handbook", "handbook", np.nan, np.nan)]
        dayt = np.isfinite(pav) & (pav > 0)
        for pol in POLS:
            X = W57.pooled([T2[T2.policy == pol].reset_index(drop=True), tx[tx.policy == pol].reset_index(drop=True)])
            Pp = W57.pooled([t[t.policy == pol].reset_index(drop=True) for t in prior]) if prior else T2[T2.policy == pol].reset_index(drop=True)
            for sel, (tab, tg) in {"A_R": (T2, TGT), "b_xfit_0.002": (X, TGT), "c_prior_0.002": (Pp, TGT),
                                   "a_half_H2": (T2, HALF), "b_xfit_0.001": (X, HALF), "c_prior_0.001": (Pp, HALF)}.items():
                p_, met = W57.select(tab, pol, tg)
                if sel == "A_R":
                    check(f"n1_AR_equals_S56_f{yr}_{sname}_{pol}", abs(p_ - float(sel56[sname]["policies"][pol]["A_R"]["param"])) < 1e-9)
                i = rix[(pol, round(float(p_), 6))]
                Crows.append(Ce[i])
                meta.append((pol, sel, float(p_), float(binds[i][dayt].mean())))
            k = [kk for kk, s in enumerate(sp56) if s[0] == pol][0]
            Crows.append(rp_row := A.path_row(Ce[pol_C[pol][1]], idx56[k], day))
            lv_path = pol_C[pol][0][idx56[k]]
            bpath = A.path_row(binds[pol_C[pol][1]].astype(float), idx56[k], day)
            meta.append((pol, "A_D", float(np.mean(lv_path)), float(np.nanmean(bpath[dayt]))))
            a_ = ad56[(ad56.fold == yr) & (ad56.scenario == sname) & (ad56.policy == pol) & ad56.primary.astype(bool)]
            check(f"n1_AD_equals_S56_f{yr}_{sname}_{pol}", len(a_) == 1 and abs(float(rp["total"]["fuel_L"][k]) / float(a_.fuel_L.iloc[0]) - 1) < 1e-9,
                  (float(rp["total"]["fuel_L"][k]), None if len(a_) != 1 else float(a_.fuel_L.iloc[0])))
        C = np.vstack(Crows)
        rr = LM.simulate(L, pav, C, day, H, record=True, **sim_kw)
        evs = events_detail(rr["rec"]["ue"], day, rr["rec"]["KU"], rr["rec_on"], ratings)
        hbf = float(rr["total"]["fuel_L"][0])
        check(f"n1_handbook_equals_S56_f{yr}_{sname}", abs(hbf / float(TT[TT.policy == "handbook"].fuel_L.iloc[0]) - 1) < 1e-9)
        for jrow, (pol, sel, par, bshare) in enumerate(meta):
            if pol == "handbook":
                continue
            evj = evs[jrow]
            n1rows.append(dict(fold=yr, scenario=sname, base=base, policy=pol, selection=sel, param=par,
                               fuel_L=float(rr["total"]["fuel_L"][jrow]), fuel_handbook=hbf,
                               ue_share=float(rr["total"]["ue_share"][jrow]), starts=float(rr["total"]["starts"][jrow]),
                               events=len(evj), event_peak_max_kW=max([e[2] for e in evj], default=0.0),
                               pv_binds_share=bshare))
        log(f"fold {yr} {sname}: N-1 done ({time.time() - t0:.0f} s)")
        del rt, Ce, binds, rr

    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(recs).to_csv(out / f"f{yr}_cases.csv", index=False)
    pd.DataFrame(evrows).to_csv(out / f"f{yr}_events.csv", index=False)
    pd.DataFrame(n1rows).to_csv(out / f"f{yr}_n1.csv", index=False)
    ok = all(c["ok"] for c in checks.values())
    (out / f"f{yr}_checks.json").write_text(json.dumps(dict(ok=ok, n=len(checks), checks=checks, seconds=time.time() - t0,
                                                              code={"ce1_wp67_sim.py": md5(Path(__file__))}), indent=1), encoding="utf-8")
    log(f"fold {yr}: {sum(c['ok'] for c in checks.values())} of {len(checks)} checks pass ({time.time() - t0:.0f} s)")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--fold", type=int)
    ap.add_argument("--all", action="store_true")
    a = ap.parse_args()
    R = Path(a.root) / "04_results" / "ce1"
    out = R / "WP67" / "sim"
    folds = FOLDS if a.all else [a.fold]
    ok = all(run_fold(y, R, out) for y in folds)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
