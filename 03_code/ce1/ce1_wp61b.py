"""Stage S61B: battery reserve credit limited to the available PV measured at issue time,
R_b' = min(R_b, P_av(t_issue)), 0 at night, for the battery scenarios of stage S6.

Usage (from the repository root unless the path starts at ce1/, then from 03_code):
    python3 03_code/ce1/ce1_wp61b.py --root .
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

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import ce1_dispatch as D  # noqa: E402
import ce1_load as LM  # noqa: E402

FOLDS = [2019, 2020, 2021, 2022, 2023, 2024]
H = 3
SCEN = {
    "battery_day":  dict(shape="A", scale="primary", min_load=LM.MIN_LOAD, battery=True, fuel="cat_c32", base="battery"),
    "batteryB_day": dict(shape="B", scale="primary", min_load=LM.MIN_LOAD, battery=True, fuel="cat_c32", base="batteryB"),
}


def md5(p: Path) -> str:
    h = hashlib.md5()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def log(m):
    print(time.strftime("%H:%M:%S"), f"[S61B] {m}", flush=True)


def issue_cap(pav, h):
    """Available PV measured at the issue time of each target step (0 where unknown or before the first issue)."""
    cap = np.zeros(len(pav))
    cap[h:] = np.nan_to_num(pav[:-h], nan=0.0)
    return cap


def selftest():
    res = {}
    grid, Lt, pav, cs = LM._toy(1)
    dd = LM.day_ids(grid)
    rng = np.random.default_rng(5)
    C = LM._counted(pav, cs, rng)
    a = LM.simulate(Lt, pav, C, dd, 3, battery=True)["total"]
    b = LM.simulate(Lt, pav, C, dd, 3, battery=True, credit_cap=np.full(len(pav), np.inf))["total"]
    res["infinite_cap_equals_no_cap"] = all(np.array_equal(a[k], b[k]) for k in ("fuel_L", "ue_kWh", "starts", "dis_kWh"))
    z = LM.simulate(Lt, pav, C, dd, 3, battery=True, credit_cap=np.zeros(len(pav)), record=True)
    nb = LM.simulate(Lt, pav, C, dd, 3, battery=False, record=True)
    # with a zero cap the committed capacity equals the no-battery commitment on every step
    res["zero_cap_commits_as_no_battery"] = np.array_equal(z["rec"]["KU"], nb["rec"]["KU"], equal_nan=True)
    cp = issue_cap(np.array([1.0, np.nan, 3.0, 4.0, 5.0]), 2)
    res["issue_cap_shift"] = np.array_equal(cp, np.array([0.0, 0.0, 1.0, 0.0, 3.0]))
    return res


def ad_block(yr, sname, sc, prep, rows, rtest, s, unit96, scales, check, hb, orc, cap):
    import ce1_adaptive as A
    import ce1_s6 as S6
    P = prep["test"]
    L = S6.load_for(P["grid"], unit96, scales, sc)
    sim_kw = dict(ratings=LM.fleet_for_peak(L.max()), min_load=sc["min_load"], fuel=sc["fuel"],
                  battery=sc["battery"], credit_cap=cap)
    daily, totT = rtest["daily"], rtest["total"]
    load_d = daily["load_kWh"][hb]
    abar = A.h2_mean_daily_load(prep["H2"], unit96, scales, sc)
    pol_C, rec_ue, specs = {}, {}, []
    for pol in A.POLICIES:
        ix = [i for i, r in enumerate(rows) if r[0] == pol]
        lv = np.array([rows[i][1] for i in ix])
        o = np.argsort(lv)
        ixo = np.array(ix)[o]
        pol_C[pol] = (lv[o], P["C"][ixo])
        rec_ue[pol] = daily["ue_kWh"][ixo]
        t0 = float(s["policies"][pol]["A_R"]["param"])
        specs += [(pol, 0.0, A.KAPPA, t0, float(lv.min())), (pol, A.GAMMA, A.KAPPA, t0, float(lv.min()))]
    max_pass = 400
    rp, idx, lat, passes, work = A.run_paths(L, P["pav"], P["day"], H, sim_kw, pol_C, specs, rec_ue, load_d, abar,
                                             max_pass=max_pass)
    # run_paths raises when the fixed point is not reached; this records that it was, and in how many passes
    check(f"ad_fixed_point_f{yr}_{sname}", 1 <= passes < max_pass and all(np.isfinite(rp["total"]["fuel_L"])),
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


def run_fold(yr, R, geo, m, unit96, scales, rows, check):
    import ce1_folds as FC
    import ce1_s6 as S6
    cache = FC.load_cache_ext(R / "S2" / f"yulara_f{yr}_h{H}.parquet", R / "S2X" / f"yulara_f{yr}_h{H}.parquet")
    prep = S6.prepare(cache, H, yr, geo, m, rows)
    hb = [i for i, r in enumerate(rows) if r[0] == "handbook"][0]
    orc = [i for i, r in enumerate(rows) if r[0] == "oracle"][0]
    s6e = pd.read_parquet(R / "S6E" / f"f{yr}_h{H}_totals.parquet")
    s51 = pd.read_csv(R / "S51" / "cells.csv", float_precision="round_trip")
    tables, sel, cells = [], {}, []
    pr = [i for i, r in enumerate(rows) if (r[0], r[1]) in set(S6.PROBES) or r[0] in S6.ANCHORS]
    for sname, sc in SCEN.items():
        res = {}
        for part in ("H2", "test"):
            P = prep[part]
            L = S6.load_for(P["grid"], unit96, scales, sc)
            kw = dict(ratings=LM.fleet_for_peak(L.max()), min_load=sc["min_load"], fuel=sc["fuel"], battery=True)
            cap = issue_cap(P["pav"], H)
            if part == "test":
                # cap off: the machinery reproduces S6E's battery scenario on the probe and anchor rows
                t0 = LM.simulate(L, P["pav"], P["C"][pr], P["day"], H, **kw)["total"]
                ref = s6e[(s6e.scenario == sc["base"]) & (s6e.set == "test")]
                worst = 0.0
                for j, i in enumerate(pr):
                    pol, par = rows[i]
                    rr = ref[(ref.policy == pol) & ((ref.param.isna() & (par != par)) | np.isclose(ref.param, par if par == par else -9))]
                    for k in ("fuel_L", "ue_kWh", "starts"):
                        worst = max(worst, abs(float(t0[k][j]) - float(rr[k].iloc[0])) / max(abs(float(rr[k].iloc[0])), 1.0))
                check(f"cap_off_reproduces_S6E_{sc['base']}_f{yr}", worst < 1e-9, f"({len(pr)} rows, worst {worst:.2e})")
            r = LM.simulate(L, P["pav"], P["C"], P["day"], H, credit_cap=cap, **kw)
            tot = r["total"]
            tot["fallback_steps"] = np.asarray(tot["fallback_steps"], float)
            res[part] = r
            t = pd.DataFrame({k: tot[k] for k in S6.TOT_KEYS})
            t.insert(0, "param", [x[1] for x in rows])
            t.insert(0, "policy", [x[0] for x in rows])
            t.insert(0, "set", part)
            t.insert(0, "scenario", sname)
            t["missing_steps"] = r["missing_steps"]
            tables.append(t)
            check(f"handbook_zero_ue_f{yr}_{sname}_{part}", float(tot["ue_kWh"][hb]) == 0.0, float(tot["ue_kWh"][hb]))
            check(f"oracle_fuel_le_handbook_f{yr}_{sname}_{part}", float(tot["fuel_L"][orc]) <= float(tot["fuel_L"][hb]) + 1e-6)
            if part == "test":
                # the cap is applied: committed capacity >= load - counted PV - capped credit on every scored step
                # (the same invariant fails on the uncapped run, where the credit covers load at night)
                rr = LM.simulate(L, P["pav"], P["C"][pr], P["day"], H, credit_cap=cap, record=True, **kw)
                ru = LM.simulate(L, P["pav"], P["C"][pr], P["day"], H, record=True, **kw)
                need = np.maximum(L[None, :] - np.nan_to_num(P["C"][pr]) - cap[None, :], 0.0)
                scm = np.isfinite(P["pav"])[None, :]
                v_cap = int(((rr["rec"]["KU"] < need - 1e-6) & scm).sum())
                v_unc = int(((ru["rec"]["KU"] < need - 1e-6) & scm).sum())
                check(f"cap_applied_on_every_scored_step_f{yr}_{sname}", v_cap == 0 and v_unc > 0,
                      f"(violations capped {v_cap}, uncapped {v_unc})")
        totH2, totT = res["H2"]["total"], res["test"]["total"]
        s = dict(scenario={k: v for k, v in sc.items()}, credit_cap="available PV at issue time",
                 anchors={a: {k: float(totT[k][i]) for k in S6.TOT_KEYS} for i, a in enumerate(S6.ANCHORS)},
                 policies={p: S6.select_policy(totH2, totT, rows, p) for p in D.ALL_POLICIES})
        sel[sname] = s
        cells += ad_block(yr, sname, sc, prep, rows, res["test"], s, unit96, scales, check, hb, orc, issue_cap(prep["test"]["pav"], H))
        log(f"fold {yr} {sname}: handbook {float(totT['fuel_L'][hb]):,.0f} L")
    # A-D wiring with the cap off reproduces the S51 battery cells (primary)
    sc = dict(SCEN["battery_day"])
    P = prep["test"]
    L = S6.load_for(P["grid"], unit96, scales, sc)
    kw = dict(ratings=LM.fleet_for_peak(L.max()), min_load=sc["min_load"], fuel=sc["fuel"], battery=True)
    r = LM.simulate(L, P["pav"], P["C"], P["day"], H, **kw)
    selb = json.load(open(R / "S6E" / f"f{yr}_h{H}_select.json"))["selections"]["battery"]
    c = pd.DataFrame(ad_block(yr, "battery_ref", sc, prep, rows, r, selb, unit96, scales, check, hb, orc, None))
    ref = s51[(s51.fold == yr) & (s51.h == H) & (s51.scenario == "battery") & s51.primary]
    mm = c[c.primary].merge(ref, on="policy", suffixes=("", "_s51"))
    worst = float(np.max(np.abs(mm.fuel_L / mm.fuel_L_s51 - 1))) if len(mm) else np.inf
    worst_ue = float(np.max(np.abs(mm.ue_kWh - mm.ue_kWh_s51))) if len(mm) else np.inf
    check(f"ad_wiring_reproduces_S51_battery_f{yr}", len(mm) == int(c.primary.sum()) and len(mm) > 0 and worst < 1e-7 and worst_ue < 1e-3,
          f"({len(mm)} policies, worst relative fuel {worst:.1e}, worst UE {worst_ue:.1e} kWh)")
    return pd.concat(tables, ignore_index=True), sel, cells


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(HERE.parents[1]))
    ap.add_argument("--only", type=int, default=None)
    a = ap.parse_args()
    import ce1_s6 as S6
    R = Path(a.root) / "04_results" / "ce1"
    out = R / "S61B"
    runs = R / "_runs"
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    started = time.strftime("%Y-%m-%dT%H:%M:%S")
    checks, inputs, outputs = {}, {}, {}

    def check(name, cond, msg=""):
        checks[name] = dict(ok=bool(cond), msg=str(msg))
        log(f"{'PASS' if cond else 'FAIL'}  {name}  {msg}")

    for k, v in selftest().items():
        check(f"selftest_{k}", v)
    geo, m, unit96, scales, _ = S6.load_inputs(R / "S1", R / "S1M", R / "S1L")
    D.extend_theta_low()
    rows = S6.row_table()
    all_cells = []
    for yr in (FOLDS if a.only is None else [a.only]):
        for nm in (f"S2/yulara_f{yr}_h{H}.parquet", f"S2X/yulara_f{yr}_h{H}.parquet", f"S6E/f{yr}_h{H}_totals.parquet",
                   f"S6E/f{yr}_h{H}_select.json", "S51/cells.csv"):
            inputs[nm] = md5(R / nm)
        f_tot, f_sel, f_cells = out / f"f{yr}_h{H}_totals.parquet", out / f"f{yr}_h{H}_select.json", out / f"f{yr}_h{H}_cells.csv"
        t1 = time.time()
        tab, sel, cells = run_fold(yr, R, geo, m, unit96, scales, rows, check)
        tab.to_parquet(f_tot)
        json.dump(sel, open(f_sel, "w"), indent=1, default=float)
        pd.DataFrame(cells).to_csv(f_cells, index=False)
        all_cells.append(pd.DataFrame(cells))
        for f in (f_tot, f_sel, f_cells):
            outputs[f.name] = md5(f)
        log(f"fold {yr}: {time.time() - t1:.0f} s")
    ac = pd.concat(all_cells, ignore_index=True)
    ac.to_csv(out / "ad_cells.csv", index=False)
    outputs["ad_cells.csv"] = md5(out / "ad_cells.csv")
    ok = all(c["ok"] for c in checks.values())
    env = dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__)
    detail = dict(stage="S61B", ok=ok, seconds=round(time.time() - t0, 1), checks=checks, inputs=inputs, outputs=outputs,
                  code={p: md5(HERE / p) for p in ("ce1_wp61b.py", "ce1_load.py", "ce1_adaptive.py", "ce1_s6.py")}, env=env)
    json.dump(detail, open(runs / "S61B.detail.json", "w"), indent=1)
    json.dump(dict(stage="S61B", ok=ok, error="", started=started, finished=time.strftime("%Y-%m-%dT%H:%M:%S"),
                   minutes=round((time.time() - t0) / 60, 1), host=platform.node() + "", smoke=False,
                   n_checks=len(checks)), open(runs / "S61B.json", "w"), indent=1)
    log(f"{'ALL CHECKS PASS' if ok else 'CHECK FAILED'} ({len(checks)} checks, {time.time() - t0:.0f} s)")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
