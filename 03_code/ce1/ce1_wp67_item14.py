#!/usr/bin/env python3
"""Rule D-129 item 14: bound on unserved energy when daytime steps without a measured GHI are
filled as total PV loss.

Usage (from the repository root unless the path starts at ce1/, then from 03_code):
    python3 03_code/ce1/ce1_wp67_item14.py
"""
from __future__ import annotations

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
import ce1_wp61aw as AW  # noqa: E402
import ce1_wp67_sim as SIM  # noqa: E402

H = 3
FOLDS = SIM.FOLDS
TGT, EPS = LM.UE_TARGET, 1e-15
STATIC, POLS = SIM.STATIC, SIM.POLS
QP = ["raw_q", "cqr", "mondrian_cqr"]
CP = ["icp", "mondrian", "cqr", "mondrian_cqr"]
log = SIM.log


def main(root="."):
    R = Path(root) / "04_results" / "ce1"
    out = R / "WP67" / "item14"
    out.mkdir(parents=True, exist_ok=True)
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
    recs, inputs = [], {}
    for yr in FOLDS:
        cache = FC.load_cache_ext(R / "S2" / f"yulara_f{yr}_h3.parquet", R / "S2X" / f"yulara_f{yr}_h3.parquet")
        miss = FC.load_cache_ext(R / "S2E" / "item14" / f"yulara_f{yr}_h3_miss.parquet",
                                 R / "S2E" / "item14x" / f"yulara_f{yr}_h3_miss.parquet")
        for p in (R / "S2E" / "item14" / f"yulara_f{yr}_h3_miss.parquet", R / "S2E" / "item14x" / f"yulara_f{yr}_h3_miss.parquet"):
            inputs[str(p.relative_to(R))] = SIM.md5(p)
        prep = S6.prepare(cache, H, yr, geo, m, rows)
        P = prep["test"]
        grid, pav, day = P["grid"], P["pav"], P["day"]
        calib = S6.as_dict(cache[cache["split"] == "calib"])
        mt = S6.as_dict(miss[miss["split"] == "test"])
        tgt = pd.DatetimeIndex(mt["idx"]) + pd.Timedelta(minutes=5 * H)
        pos = grid.get_indexer(tgt)
        check(f"f{yr}_miss_targets_on_grid_and_nan", (pos >= 0).all() and np.isnan(pav[pos]).all(),
              f"({len(pos)} rows)")
        pv_of = S6.PVAt(tgt, m)
        Cm = S6.counted_matrix(rows, calib, mt, pv_of, np.zeros(len(pos)))
        filled = np.isnan(pav)
        Cf = P["C"].copy()
        Cf[:, pos] = Cm
        pavf = np.where(filled, 0.0, pav)
        z = np.load(R / "S6E" / f"f{yr}_h3_daily.npz")
        sel_e = json.load(open(R / "S6E" / f"f{yr}_h3_select.json"))["selections"]
        cases = pd.read_csv(R / "WP63" / "cases.csv", float_precision="round_trip")
        cases = cases[cases.fold == yr]
        wcases = pd.read_csv(R / "WP67" / "sim" / f"f{yr}_cases.csv", float_precision="round_trip")
        pz = np.load(R / "S51" / f"f{yr}_h3_paths.npz")
        specs = [(p, float(g), float(k)) for p, g, k in (s.split("|") for s in pz["specs"].astype(str))]
        pol_C = {}
        for pol in A.POLICIES:
            ix = [i for i, r in enumerate(rows) if r[0] == pol]
            lv = np.array([rows[i][1] for i in ix])
            o = np.argsort(lv)
            pol_C[pol] = (lv[o], np.array(ix)[o])
        for sname in ("main", "shapeB"):
            sc = S6.SCENARIOS[sname]
            L = S6.load_for(grid, unit96, scales, sc)
            ratings = np.asarray(LM.fleet_for_peak(L.max()), float)
            sim_kw = dict(ratings=list(ratings), min_load=sc["min_load"], fuel=sc["fuel"], battery=sc["battery"])
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
                _, idx_cl, _, _, _ = A.run_paths(L, pav, day, H, sim_kw, pc_full, cspecs, rec_ue, load_d, abar)
            finally:
                A.control, A.control_lookup = c0, c1
            Ca, Cb, meta = [P["C"][hb]], [Cf[hb]], [("handbook", "handbook")]
            for pol in POLS:
                cc = cases[(cases.scenario == sname) & (cases.policy == pol)]
                for sel in STATIC:
                    i = rix[(pol, round(float(cc[cc.selection == sel].iloc[0].param), 6))]
                    Ca.append(P["C"][i])
                    Cb.append(Cf[i])
                    meta.append((pol, sel))
                    check(f"f{yr}_{sname}_{pol}_{sel}_filled_counted_finite", np.isfinite(Cm[i]).all())
                k = specs.index((pol, A.GAMMA, A.KAPPA))
                lev = pz[f"{sname}__level"][k].astype(np.float64)
                levels = pol_C[pol][0]
                j = np.abs(levels[None, :].astype(np.float32).astype(np.float64) - lev[:, None]).argmin(axis=1)
                Ca.append(A.path_row(P["C"][pol_C[pol][1]], j, day))
                Cb.append(A.path_row(Cf[pol_C[pol][1]], j, day))
                meta.append((pol, "A_D"))
                kc = [i for i, s in enumerate(cspecs) if s[0] == pol][0]
                Ca.append(A.path_row(P["C"][pol_C[pol][1]], idx_cl[kc], day))
                Cb.append(A.path_row(Cf[pol_C[pol][1]], idx_cl[kc], day))
                meta.append((pol, "AW_clip"))
            ra = LM.simulate(L, pav, np.vstack(Ca), day, H, **sim_kw)
            rb = LM.simulate(L, pavf, np.vstack(Cb), day, H, record=True, **sim_kw)
            check(f"f{yr}_{sname}_handbook_zero_ue_filled", float(rb["total"]["ue_kWh"][0]) == 0.0)
            fl_load = float(np.sum(L[filled]) * LM.DT)
            day_mask = filled
            for jrow, (pol, sel) in enumerate(meta):
                if pol == "handbook":
                    continue
                w = wcases[(wcases.scenario == sname) & (wcases.policy == pol) & (wcases.selection == sel)].iloc[0]
                check(f"f{yr}_{sname}_{pol}_{sel}_asrun_equals_WP67sim",
                      abs(ra["total"]["fuel_L"][jrow] / w.fuel_L - 1) < 1e-9 and ra["total"]["starts"][jrow] == w.starts
                      and abs(ra["total"]["ue_kWh"][jrow] - w.ue_kWh) <= 1e-9 * max(w.ue_kWh, 1.0))
                ue_s, ld_s = float(ra["total"]["ue_kWh"][jrow]), float(ra["total"]["load_kWh"][jrow])
                ue_f = float(np.nansum(rb["rec"]["ue"][jrow][day_mask]) * LM.DT)
                share_b = (ue_s + ue_f) / (ld_s + fl_load)
                recs.append(dict(fold=yr, scenario=sname, policy=pol, selection=sel,
                                 filled_steps=int(filled.sum()), filled_with_forecast=int(np.isfinite(Cb[jrow][filled]).sum()),
                                 filled_load_kWh=fl_load, ue_scored_kWh=ue_s, load_scored_kWh=ld_s, ue_filled_kWh=ue_f,
                                 ue_share_primary=ue_s / ld_s, ue_share_bound=share_b,
                                 meets_primary=bool(ue_s / ld_s <= TGT + EPS), meets_bound=bool(share_b <= TGT + EPS),
                                 ue_share_filled_run=float(rb["total"]["ue_kWh"][jrow]) / float(rb["total"]["load_kWh"][jrow]),
                                 meets_filled_run=bool(float(rb["total"]["ue_kWh"][jrow]) / float(rb["total"]["load_kWh"][jrow]) <= TGT + EPS)))
            log(f"fold {yr} {sname} done ({time.time() - t0:.0f} s)")
            del rb
    df = pd.DataFrame(recs)
    df.to_csv(out / "cases.csv", index=False)
    summ = []
    for sel in STATIC + ["A_D", "AW_clip"]:
        for sname_, pols in (("conformal", CP), ("quantile", QP)):
            d = df[(df.selection == sel) & df.policy.isin(pols)]
            summ.append(dict(selection=sel, set=sname_, n=len(d), met_primary=int(d.meets_primary.sum()),
                             met_bound=int(d.meets_bound.sum()), met_filled_run=int(d.meets_filled_run.sum()),
                             ue_share_bound_max=float(d.ue_share_bound.max())))
    S = pd.DataFrame(summ)
    S.to_csv(out / "summary.csv", index=False)
    xh = S[(S.selection == "b_xfit_0.001") & (S.set == "quantile")].iloc[0]
    reading = dict(rule="D-129 item 14: XH quantile met count under the bound below 34 of 36 -> main text, else one Supplement sentence",
                   xh_quantile_met_primary=int(xh.met_primary), xh_quantile_met_bound=int(xh.met_bound),
                   fires=bool(int(xh.met_bound) < 34))
    ok = all(c["ok"] for c in checks.values())
    (out / "checks.json").write_text(json.dumps(dict(ok=ok, n=len(checks), reading=reading, checks=checks, inputs=inputs,
                                                     seconds=time.time() - t0, code={"ce1_wp67_item14.py": SIM.md5(Path(__file__))}),
                                                indent=1), encoding="utf-8")
    log(f"{sum(c['ok'] for c in checks.values())} of {len(checks)} checks pass")
    print(S.to_string(index=False))
    print(json.dumps(reading))
    return ok


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
