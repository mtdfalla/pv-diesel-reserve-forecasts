"""Matched static baselines for Protocol A-D (rule D-112): half allowance, cross-fitted halves of
the calibration year, and all earlier test years. Stage S57 simulates the first half of every
calibration year with scores fitted on its second half.

Usage (from the repository root unless the path starts at ce1/, then from 03_code):
    python ce1/ce1_wp57.py --stage S57 [--smoke]
    python3 03_code/ce1/ce1_wp57.py --analyse --root .
    python3 03_code/ce1/ce1_wp57.py --selftest
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
SCEN = ("main", "shapeB")
CP = ["icp", "mondrian", "cqr", "mondrian_cqr"]
POLICIES = list(D.ALL_POLICIES)
TGT = LM.UE_TARGET  # 0.002 %
HALF = TGT / 2.0  # 0.001 %
PR2_NEED = 40
EPS = 1e-15


def md5(p: Path) -> str:
    h = hashlib.md5()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def log(m):
    print(time.strftime("%H:%M:%S"), f"[S57] {m}", flush=True)


# ================================================================ selection helpers
def pooled(tabs):
    """Sum fuel, UE and load over several totals tables with identical (policy, param) rows."""
    key = ["policy", "param"]
    base = tabs[0][key].reset_index(drop=True)
    for t in tabs[1:]:
        assert t[key].reset_index(drop=True).equals(base), "row sets differ between pooled tables"
    out = base.copy()
    for k in ("fuel_L", "ue_kWh", "load_kWh"):
        out[k] = sum(t[k].to_numpy(float) for t in tabs)
    out["ue_share"] = np.divide(out["ue_kWh"], out["load_kWh"], out=np.zeros(len(out)), where=out["load_kWh"] > 0)
    return out


def select(tab: pd.DataFrame, pol: str, target: float):
    """ce1_load.select_AR on one policy's rows of a totals table."""
    t = tab[tab.policy == pol]
    params = t["param"].to_numpy(float)
    p, met = LM.select_AR(dict(ue_share=t["ue_share"].to_numpy(float), fuel_L=t["fuel_L"].to_numpy(float)),
                          params, pol, target=target)
    return p, bool(met)


def at(tab: pd.DataFrame, pol: str, param: float) -> dict:
    r = tab[(tab.policy == pol) & (np.isclose(tab.param.to_numpy(float), param, atol=1e-9))]
    assert len(r) == 1, (pol, param, len(r))
    return r.iloc[0].to_dict()


def stage_s57(root: Path, smoke=False):
    import ce1_folds as FC
    import ce1_s6 as S6
    R = root / "04_results" / "ce1"
    out = R / ("S57_smoke" if smoke else "S57")
    runs = R / "_runs"
    out.mkdir(parents=True, exist_ok=True)
    runs.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    checks, inputs, outputs = {}, {}, {}

    def check(name, cond, msg=""):
        checks[name] = dict(ok=bool(cond), msg=msg)
        log(f"{'PASS' if cond else 'FAIL'}  {name}  {msg}")

    geo, m, unit96, scales, _ = S6.load_inputs(R / "S1", R / "S1M", R / "S1L")
    D.extend_theta_low()  # the S6E set (D-092)
    rows = S6.row_table()
    folds = [2024] if smoke else FOLDS
    for yr in folds:
        f_out = out / f"f{yr}_h{H}_H1x.parquet"
        if f_out.exists() and not smoke:
            log(f"fold {yr}: exists, skipped")
            outputs[f_out.name] = md5(f_out)
            continue
        t1 = time.time()
        s2 = R / "S2" / f"yulara_f{yr}_h{H}.parquet"
        s2x = R / "S2X" / f"yulara_f{yr}_h{H}.parquet"
        inputs[f"S2/{s2.name}"] = md5(s2)
        inputs[f"S2X/{s2x.name}"] = md5(s2x)
        cache = FC.load_cache_ext(s2, s2x)
        calib = S6.as_dict(cache[cache["split"] == "calib"])
        H1, H2 = D.split_halves(calib)
        lead = pd.Timedelta(minutes=5 * H)
        cy = yr - 1
        parts = {"H1x": (H2, H1, (f"{cy}-01-01", f"{cy}-06-30 23:55")),
                 "H2probe": (H1, H2, (f"{cy}-07-01", f"{cy}-12-31 23:55"))}
        s6e = pd.read_parquet(R / "S6E" / f"f{yr}_h{H}_totals.parquet")
        inputs[f"S6E/f{yr}_h{H}_totals.parquet"] = md5(R / "S6E" / f"f{yr}_h{H}_totals.parquet")
        tabs = []
        for part, (fit, ev, (a, b)) in parts.items():
            use_rows = rows if part == "H1x" else [r for r in rows if r[0] in S6.ANCHORS or
                                                   (r[0], r[1]) in set(S6.PROBES) or r[1] == S6.THETA_LIMIT]
            if smoke and part == "H1x":
                use_rows = [r for r in rows if r[0] in S6.ANCHORS or r[0] in ("icp", "mondrian_cqr")]
            tgt = pd.DatetimeIndex(ev["idx"]) + lead
            grid = S6.year_grid(a, b)
            pav = LM.available_pv_on_grid(grid, geo["ghi"].reindex(grid).values, m)
            pv_of = S6.PVAt(tgt, m)
            pav_t = pv_of(ev["y"])
            pos = grid.get_indexer(tgt)
            check(f"target_basis_f{yr}_{part}", bool(np.allclose(pav_t, pav[pos], atol=1e-9, equal_nan=True)))
            C = S6.place(grid, tgt, S6.counted_matrix(use_rows, fit, ev, pv_of, pav_t))
            day = LM.day_ids(grid)
            for sname in SCEN:
                sc = S6.SCENARIOS[sname]
                L = S6.load_for(grid, unit96, scales, sc)
                r = LM.simulate(L, pav, C, day, H, ratings=LM.fleet_for_peak(L.max()), min_load=sc["min_load"],
                                fuel=sc["fuel"], battery=sc["battery"])
                t = pd.DataFrame({k: r["total"][k] for k in S6.TOT_KEYS})
                t.insert(0, "param", [x[1] for x in use_rows])
                t.insert(0, "policy", [x[0] for x in use_rows])
                t.insert(0, "set", part)
                t.insert(0, "scenario", sname)
                t["missing_steps"] = r["missing_steps"]
                if part == "H2probe":
                    ref = s6e[(s6e.scenario == sname) & (s6e.set == "H2")]
                    mm = t.merge(ref, on=["policy", "param"], suffixes=("", "_ref"))
                    worst = max(float(np.max(np.abs(mm[k] - mm[k + "_ref"]) / np.maximum(np.abs(mm[k + "_ref"]), 1.0)))
                                for k in ("fuel_L", "ue_kWh", "starts"))
                    check(f"probe_reproduces_S6E_H2_f{yr}_{sname}", len(mm) == len(t) and worst < 1e-9,
                          f"({len(mm)} rows, worst relative difference {worst:.2e})")
                else:
                    anc = t.set_index("policy").loc["handbook"]
                    check(f"handbook_zero_ue_H1x_f{yr}_{sname}", float(anc["ue_kWh"]) == 0.0)
                    tabs.append(t)
        pd.concat(tabs, ignore_index=True).to_parquet(f_out)
        outputs[f_out.name] = md5(f_out)
        log(f"fold {yr}: {time.time() - t1:.0f} s")
    ok = all(c["ok"] for c in checks.values())
    detail = dict(simulator=LM.sim_defaults(), stage="S57", ok=ok, smoke=smoke, seconds=round(time.time() - t0, 1), checks=checks,
                  inputs=inputs, outputs=outputs, code={"ce1_wp57.py": md5(Path(__file__))},
                  env=dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__))
    json.dump(detail, open(runs / ("S57_smoke.detail.json" if smoke else "S57.detail.json"), "w"), indent=1)
    log(f"{'ALL CHECKS PASS' if ok else 'CHECK FAILED'} ({len(checks)} checks, {time.time() - t0:.0f} s)")
    return ok


def analyse(root: Path):
    R = root / "04_results" / "ce1"
    out = R / "WP57"
    out.mkdir(parents=True, exist_ok=True)
    checks, inputs = {}, {}

    def check(name, cond, msg=""):
        checks[name] = dict(ok=bool(cond), msg=msg)
        print(f"{'PASS' if cond else 'FAIL'}  {name}  {msg}", flush=True)

    tot, sel = {}, {}
    for yr in FOLDS:
        p = R / "S6E" / f"f{yr}_h{H}_totals.parquet"
        inputs[f"S6E/{p.name}"] = md5(p)
        tot[yr] = pd.read_parquet(p)
        q = R / "S6E" / f"f{yr}_h{H}_select.json"
        inputs[f"S6E/{q.name}"] = md5(q)
        sel[yr] = json.load(open(q))["selections"]
    s57_ok = False
    st = R / "_runs" / "S57.json"
    if st.exists():
        s57_ok = bool(json.load(open(st)).get("ok")) and bool(json.load(open(R / "_runs" / "S57.detail.json")).get("ok"))
    cells = pd.read_csv(R / "S51" / "cells.csv")
    inputs["S51/cells.csv"] = md5(R / "S51" / "cells.csv")
    ad = cells[(cells.h == H) & cells.primary & cells.scenario.isin(SCEN)]

    def T(yr, sname, part):
        t = tot[yr]
        return t[(t.scenario == sname) & (t.set == part)].reset_index(drop=True)

    recs = []
    for yr in FOLDS:
        for sname in SCEN:
            H2, TE = T(yr, sname, "H2"), T(yr, sname, "test")
            hb = TE[TE.policy == "handbook"].iloc[0].to_dict()
            prior = [T(y, sname, "test") for y in FOLDS if y < yr]
            H1x = None
            if s57_ok:
                z = pd.read_parquet(R / "S57" / f"f{yr}_h{H}_H1x.parquet")
                H1x = z[(z.scenario == sname)].reset_index(drop=True)
            for pol in POLICIES:
                # anchor: the stored A-R selection is reproduced from the totals
                pAR, _ = select(H2, pol, TGT)
                check(f"reproduces_S6E_AR_f{yr}_{sname}_{pol}",
                      abs(pAR - float(sel[yr][sname]["policies"][pol]["A_R"]["param"])) < 1e-9)
                variants = {"A_R": (H2, TGT), "a_half_H2": (H2, HALF)}
                if prior:
                    P = pooled([t[t.policy.isin([pol])] for t in prior])
                    variants["c_prior_0.002"] = (P, TGT)
                    variants["c_prior_0.001"] = (P, HALF)
                else:  # fold 2019: H2 of 2018 (standard A-R)
                    variants["c_prior_0.002"] = (H2, TGT)
                    variants["c_prior_0.001"] = (H2, HALF)
                if H1x is not None:
                    X = pooled([H2[H2.policy == pol].reset_index(drop=True), H1x[H1x.policy == pol].reset_index(drop=True)])
                    variants["b_xfit_0.002"] = (X, TGT)
                    variants["b_xfit_0.001"] = (X, HALF)
                for vname, (tab, tg) in variants.items():
                    p, met = select(tab, pol, tg)
                    r = at(TE, pol, p)
                    recs.append(dict(fold=yr, scenario=sname, policy=pol, cp=pol in CP, selection=vname,
                                     target_sel=tg, param=p, met_on_selection=met, fuel_L=r["fuel_L"],
                                     ue_share=r["ue_share"], ue_kWh=r["ue_kWh"], starts=r["starts"],
                                     meets=bool(r["ue_share"] <= TGT + EPS), fuel_handbook=hb["fuel_L"],
                                     saving_pct=100 * (1 - r["fuel_L"] / hb["fuel_L"])))
                a = ad[(ad.fold == yr) & (ad.scenario == sname) & (ad.policy == pol)]
                if len(a) == 1:
                    a = a.iloc[0]
                    recs.append(dict(fold=yr, scenario=sname, policy=pol, cp=pol in CP, selection="A_D",
                                     target_sel=np.nan, param=np.nan, met_on_selection=np.nan, fuel_L=a.fuel_L,
                                     ue_share=a.ue_share, ue_kWh=a.ue_kWh, starts=np.nan,
                                     meets=bool(a.ue_share <= TGT + EPS), fuel_handbook=a.fuel_handbook,
                                     saving_pct=100 * (1 - a.fuel_L / a.fuel_handbook)))
                    check(f"S51_AR_equals_S6E_f{yr}_{sname}_{pol}",
                          abs(a.fuel_AR / float(sel[yr][sname]["policies"][pol]["A_R"]["test"]["fuel_L"]) - 1) < 1e-7,
                          "(S51 sums float32 daily values: relative tolerance 1e-7)")
    df = pd.DataFrame(recs)
    df.to_csv(out / "cases.csv", index=False)
    cpd = df[df.cp]
    summ = cpd.groupby("selection").agg(n=("meets", "size"), meets=("meets", "sum"),
                                        saving_median_pct=("saving_pct", "median"),
                                        saving_min_pct=("saving_pct", "min"),
                                        ue_share_median=("ue_share", "median")).reset_index()
    summ.to_csv(out / "selections.csv", index=False)
    print(summ.to_string(index=False))
    n_a = int(summ.set_index("selection").loc["a_half_H2", "meets"])
    ext = {v: int(summ.set_index("selection").loc[v, "meets"]) for v in ("b_xfit_0.001", "c_prior_0.001")
           if v in set(summ.selection)}
    fired = n_a >= PR2_NEED
    ext_fired = {k: v >= PR2_NEED for k, v in ext.items()}
    # matched comparison: A-D against each static selection, same case (fold, shape, policy)
    piv = cpd.pivot_table(index=["fold", "scenario", "policy"], columns="selection", values="fuel_L")
    pm = cpd.pivot_table(index=["fold", "scenario", "policy"], columns="selection", values="meets")
    comp = {}
    for v in piv.columns:
        if v == "A_D":
            continue
        both = (pm[v] == 1) & (pm["A_D"] == 1)
        d = 100 * (piv[v] - piv["A_D"]) / piv[v]  # % fuel A-D saves against the static rule
        comp[v] = dict(cases_both_meet=int(both.sum()),
                       ad_fuel_saving_vs_static_median_pct_when_both_meet=float(d[both].median()) if both.any() else None,
                       ad_fuel_saving_vs_static_median_pct_all=float(d.median()),
                       ad_uses_less_fuel_cases=int((d > 0).sum()), n=int(d.notna().sum()))
    pr2 = dict(rule="D-110 PR-2, operational D-112", selection="a_half_H2 (A-R on H2 at 0.001%)",
               meets_of_48=n_a, need=PR2_NEED, fired=bool(fired), extension_counts=ext, extension_fired=ext_fired,
               s57_available=s57_ok, ad_vs_static=comp)
    json.dump(pr2, open(out / "pr2.json", "w"), indent=1, default=float)
    check("case_count_48", int((cpd.selection == "a_half_H2").sum()) == 48)
    ok = all(c["ok"] for c in checks.values())
    outs = {p.name: md5(p) for p in sorted(out.glob("*")) if p.name != "checks.json"}
    json.dump(dict(checks=checks, inputs=inputs, outputs=outs, code={"ce1_wp57.py": md5(Path(__file__))}),
              open(out / "checks.json", "w"), indent=1)
    json.dump(dict(stage="WP57", ok=ok, finished=time.strftime("%Y-%m-%dT%H:%M:%S"), host="Linux",
                   s57_included=s57_ok, n_checks=len(checks), n_failed=sum(not c["ok"] for c in checks.values()),
                   outputs=outs, code={"ce1_wp57.py": md5(Path(__file__))},
                   env=dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__)),
              open(R / "_runs" / "WP57.json", "w"), indent=1)
    print(f"PR-2 fired: {fired} ({n_a} of 48 meet 0.002% under (a)); extension {ext}")
    print(json.dumps(comp, indent=1, default=float))
    print(f"{'ALL CHECKS PASS' if ok else 'CHECK FAILED'} ({len(checks)} checks)")
    return ok


# ================================================================ self-test
def selftest(verbose=True):
    res = {}

    def ok(name, cond, msg=""):
        res[name] = bool(cond)
        if verbose:
            print(f"  [{'PASS' if cond else 'FAIL'}] {name} {msg}")

    t1 = pd.DataFrame(dict(policy=["x"] * 3, param=[0.9, 0.95, 1.0], fuel_L=[10.0, 11.0, 12.0],
                           ue_kWh=[3.0, 1.0, 0.0], load_kWh=[1e5] * 3))
    t2 = t1.assign(ue_kWh=[0.0, 0.0, 0.0])
    P = pooled([t1, t2])
    ok("pooled_share", np.allclose(P.ue_share, [1.5e-5, 0.5e-5, 0.0]))
    ok("select_least_fuel_meeting", select(P.assign(ue_share=P.ue_share), "x", 1e-5)[0] == 0.95)
    ok("select_conservative_when_none_meets", select(P, "x", -1.0) == (1.0, False))
    bad = t1.assign(param=[0.9, 0.96, 1.0])
    try:
        pooled([t1, bad])
        ok("pooled_refuses_different_rows", False)
    except AssertionError:
        ok("pooled_refuses_different_rows", True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--analyse", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--root", default=None)
    a = ap.parse_args()
    st = selftest()
    if not all(st.values()):
        sys.exit(1)
    if a.selftest:
        sys.exit(0)
    if a.stage == "S57":
        root = Path(a.root) if a.root else HERE.parents[1]
        sys.exit(0 if stage_s57(root, a.smoke) else 1)
    if a.analyse:
        sys.exit(0 if analyse(Path(a.root or ".")) else 1)
    ap.error("nothing to do")


if __name__ == "__main__":
    main()
