"""Reading of stage S56 under D-112 and D-113: oracle saving, A-R and A-D savings, target met,
and the share of steps where the PV term binds.

Usage (from the repository root unless the path starts at ce1/, then from 03_code):
    python3 03_code/ce1/ce1_wp56_read.py
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path.cwd()
R = ROOT / "04_results" / "ce1"
FOLDS = [2019, 2020, 2021, 2022, 2023, 2024]
TGT = 0.002 / 100
QP = ["raw_q", "cqr", "mondrian_cqr"]
CP = ["icp", "mondrian", "cqr", "mondrian_cqr"]
BASE = {"cont": "main", "contB": "shapeB", "nl2": "main", "nl2eq": "main", "min30": "main", "min50": "main"}
SHAPE = {"cont": "A", "contB": "B", "nl2": "A", "nl2eq": "A", "min30": "A", "min50": "A", "main": "A", "shapeB": "B"}


def md5(p):
    return hashlib.md5(Path(p).read_bytes()).hexdigest()


def main():
    out = R / "WP56"
    out.mkdir(parents=True, exist_ok=True)
    checks, inputs = {}, {}

    def check(n, c, m=""):
        checks[n] = dict(ok=bool(c), msg=m)
        print(f"{'PASS' if c else 'FAIL'}  {n}  {m}", flush=True)

    st = json.load(open(R / "_runs" / "S56.detail.json"))
    check("S56_stamp_ok", st["ok"] and not st.get("smoke"), f"({len(st['checks'])} stage checks)")
    for f, h in st["outputs"].items():
        ok = md5(R / "S56" / f) == h
        if not ok:
            check(f"S56_output_md5_{f}", False)
    check("S56_outputs_match_stamp", all(md5(R / "S56" / f) == h for f, h in st["outputs"].items()),
          f"({len(st['outputs'])} files)")
    inputs["S56/ad_cells.csv"] = md5(R / "S56" / "ad_cells.csv")
    ad = pd.read_csv(R / "S56" / "ad_cells.csv")
    s51 = pd.read_csv(R / "S51" / "cells.csv")
    s51 = s51[(s51.h == 3) & s51.primary & s51.scenario.isin(("main", "shapeB"))]
    rows = []
    for yr in FOLDS:
        sel = json.load(open(R / "S56" / f"f{yr}_h3_select.json"))
        tot = pd.read_parquet(R / "S56" / f"f{yr}_h3_totals.parquet")
        base_sel = json.load(open(R / "S6E" / f"f{yr}_h3_select.json"))["selections"]
        for sname, s in sel.items():
            hb = s["anchors"]["handbook"]["fuel_L"]
            orc = s["anchors"]["oracle"]["fuel_L"]
            T = tot[(tot.scenario == sname) & (tot.set == "test")]
            for pol in CP + ["raw_q"]:
                ar = s["policies"][pol]["A_R"]
                a = ad[(ad.fold == yr) & (ad.scenario == sname) & (ad.policy == pol) & ad.primary].iloc[0]
                trow = T[(T.policy == pol) & np.isclose(T.param, ar["param"])].iloc[0]
                b = base_sel[BASE[sname]]
                bhb = b["anchors"]["handbook"]["fuel_L"]
                bb = s51[(s51.fold == yr) & (s51.scenario == BASE[sname]) & (s51.policy == pol)].iloc[0]
                rows.append(dict(fold=yr, scenario=sname, base=BASE[sname], policy=pol,
                                 ceiling_pct=100 * (1 - orc / hb),
                                 base_ceiling_pct=100 * (1 - b["anchors"]["oracle"]["fuel_L"] / bhb),
                                 AR_param=ar["param"], AR_saving_pct=100 * (1 - ar["test"]["fuel_L"] / hb),
                                 AR_meets=ar["test"]["ue_share"] <= TGT + 1e-15,
                                 AD_saving_pct=100 * (1 - a.fuel_L / a.fuel_handbook),
                                 AD_meets=a.ue_share <= TGT + 1e-15,
                                 base_AD_saving_pct=100 * (1 - bb.fuel_L / bb.fuel_handbook),
                                 base_AD_meets=bb.ue_share <= TGT + 1e-15,
                                 pv_binds_share_handbook=float(T[T.policy == "handbook"].pv_binds_share.iloc[0]),
                                 pv_binds_share_AR=float(trow.pv_binds_share),
                                 n1_infeasible_steps=int(T[T.policy == "handbook"].n1_infeasible_steps.iloc[0])))
                check(f"handbook_fuel_consistent_f{yr}_{sname}_{pol}", abs(a.fuel_handbook / hb - 1) < 1e-12)
    c = pd.DataFrame(rows)
    c.to_csv(out / "cells.csv", index=False)
    summ = []
    for sname in ("cont", "contB", "nl2", "nl2eq", "min30", "min50"):
        g = c[c.scenario == sname]
        q, p = g[g.policy.isin(QP)], g[g.policy.isin(CP)]
        summ.append(dict(scenario=sname, base=BASE[sname],
                         ceiling_median=g.groupby("fold").ceiling_pct.first().median(),
                         base_ceiling_median=g.groupby("fold").base_ceiling_pct.first().median(),
                         AD_saving_median_quantile=q.AD_saving_pct.median(),
                         base_AD_saving_median_quantile=q.base_AD_saving_pct.median(),
                         AD_saving_median_conformal=p.AD_saving_pct.median(),
                         base_AD_saving_median_conformal=p.base_AD_saving_pct.median(),
                         AD_meets_conformal=int(p.AD_meets.sum()), n_conformal=len(p),
                         AR_meets_conformal=int(p.AR_meets.sum()),
                         AR_saving_median_quantile=q.AR_saving_pct.median(),
                         AR_param_median_conformal=p.AR_param.median(),
                         pv_binds_handbook_median=g.groupby("fold").pv_binds_share_handbook.first().median(),
                         pv_binds_AR_median=p.pv_binds_share_AR.median(),
                         n1_infeasible_max=int(g.n1_infeasible_steps.max())))
    S = pd.DataFrame(summ)
    S.to_csv(out / "scenario_summary.csv", index=False)
    print(S.round(3).to_string(index=False))
    pair = c[c.scenario.isin(("cont", "contB"))]
    red = dict(
        per_scenario_quantile={r.scenario: r.AD_saving_median_quantile for r in S.itertuples()},
        per_scenario_conformal={r.scenario: r.AD_saving_median_conformal for r in S.itertuples()},
        contingency_both_shapes_quantile=float(pair[pair.policy.isin(QP)].AD_saving_pct.median()),
        contingency_both_shapes_conformal=float(pair[pair.policy.isin(CP)].AD_saving_pct.median()))
    vals = list(red["per_scenario_quantile"].values()) + list(red["per_scenario_conformal"].values()) + \
        [red["contingency_both_shapes_quantile"], red["contingency_both_shapes_conformal"]]
    red["fires"] = bool(min(vals) <= 0)
    red["rule"] = "D-112 WP-56 reading, scope clarified in D-113 (quantile or conformal; fires if either)"
    json.dump(red, open(out / "reading.json", "w"), indent=1, default=float)
    print(json.dumps(red, indent=1, default=float))
    ok = all(v["ok"] for v in checks.values())
    outs = {p.name: md5(p) for p in sorted(out.glob("*")) if p.name != "checks.json"}
    json.dump(dict(checks=checks, inputs=inputs, outputs=outs, code={"ce1_wp56_read.py": md5(Path(__file__))}),
              open(out / "checks.json", "w"), indent=1)
    print(f"{'ALL CHECKS PASS' if ok else 'CHECK FAILED'} ({len(checks)} checks)")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
