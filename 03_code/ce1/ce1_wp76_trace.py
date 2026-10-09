#!/usr/bin/env python3
"""Complete days and forecast-covered savings in fold 2025, and event-energy values, from existing
files. Writes 04_results/ce1/WP76/.

Usage (from the repository root unless the path starts at ce1/, then from 03_code):
    python3 03_code/ce1/ce1_wp76_trace.py
"""
from __future__ import annotations

import glob
import json
import statistics as st
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path.cwd()
sys.path.insert(0, str(ROOT / "03_code" / "ce1"))
R = ROOT / "04_results" / "ce1"
OUT = R / "WP76"
QP = ("raw_q", "cqr", "mondrian_cqr")
TGT = 2e-5
CHECKS: dict = {}
import ce1_literal as LT
LIT = LT.Literal(OUT, "ce1_wp76_trace.py")


def check(name, cond, msg=""):
    CHECKS[name] = dict(ok=bool(cond), msg=str(msg))
    print(("PASS  " if cond else "FAIL  ") + name + "  " + str(msg), flush=True)
    return bool(cond)


def t1():
    import ce1_s6 as S6
    import ce1_s63 as S63
    import ce1_folds as FC
    geo, *_ = S6.load_inputs(R / "S1E", R / "S1M", R / "S1L")
    is_day = geo["is_day"].astype(bool)
    ref = pd.read_csv(R / "S63" / "complete_days_dispatch.csv").groupby("fold").agg(cd=("complete_days", "first"),
                                                                                    days=("days", "first"))
    rows, ok = [], True
    for yr in range(2019, 2026):
        p = R / "S2" / f"yulara_f{yr}_h3.parquet" if yr < 2025 else R / "S2E" / "f2025" / "yulara_f2025_h3.parquet"
        cache = FC.load_cache(p)
        te = S6.as_dict(cache[cache["split"] == "test"])
        cd = S63.complete_days_fc(te["idx"], is_day, 3, 0.95)
        n = sum(1 for d in cd if str(d)[:4] == str(yr))
        ndays = 366 if yr % 4 == 0 else 365
        rows.append(dict(fold=yr, complete_days=n, complete_days_all=len(cd), days=ndays, share=n / ndays))
        if yr < 2025:
            ok &= check(f"T1_gate_fold{yr}", n == int(ref.loc[yr, "cd"]) and len(cd) == n, (n, int(ref.loc[yr, "cd"]), len(cd)))
    return pd.DataFrame(rows), ok


def red(c):
    x = c[(c.selection == "b_xfit_0.001") & (c.policy.isin(QP))]
    return dict(cases=len(x), met_covered=int((x.ue_share_covered <= TGT + 1e-12).sum()),
                median_saving_covered=float(x.saving_covered_pct.median()),
                met_complete=int((x.ue_share_complete <= TGT + 1e-12).sum()),
                median_saving_complete=float(x.saving_complete_pct.median()),
                met_all=int((x.ue_share <= TGT + 1e-12).sum()),
                median_saving_all=float((100 * (1 - x.fuel_L / x.fuel_handbook)).median()))  # sim files carry no meets/saving_pct


def t2():
    six = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(str(R / "WP67" / "sim" / "f20*_cases.csv")))])
    a = red(six)
    ref = pd.read_csv(R / "WP67" / "covered_steps.csv").set_index(["rule", "set"]).loc[("XH", "quantile")]
    ok = check("T2_gate_six_folds", a["cases"] == 36 and a["met_covered"] == int(ref.met_covered)
               and abs(a["median_saving_covered"] - float(ref.median_saving_covered)) < 1e-9,
               (a["cases"], a["met_covered"], a["median_saving_covered"], int(ref.met_covered), float(ref.median_saving_covered)))
    b = red(pd.read_csv(R / "WP67" / "f2025" / "f2025_cases.csv"))
    check("T2_2025_six_cases", b["cases"] == 6, b["cases"])
    return pd.DataFrame([dict(period="six_folds", **a), dict(period="2025", **b)]), ok


def t3():
    e = pd.read_csv(R / "WP67" / "sim" / "f2024_events.csv")
    import ce1_wp74_trace as W74  # the peak event, selected by the same rule
    ev_rows = W74.r1p1_rows(e, LIT)
    ev = ev_rows.iloc[0]
    sel = {"A-R": "A_R", "XF": "b_xfit_0.002", "PF": "c_prior_0.002", "H2": "a_half_H2", "XH": "b_xfit_0.001",
           "PH": "c_prior_0.001", "A-D": "A_D"}
    c = pd.read_csv(R / "S69" / "cases.csv")
    shp = "B" if ev.scenario == "shapeB" else "A"  # the event's load shape (B in v2.0.0)
    x = c[(c.construction == "plain") & (c.fold == 2024) & (c["shape"] == shp) & (c.policy.isin(QP))].copy()
    x = x[[sel.get(r) == s for r, s in zip(x.rule, x.selection)]]
    check("T3_21_cases", len(x) == 21, len(x))  # the quantile cases of the seven rules
    carry = set(zip(ev_rows.policy, ev_rows.selection))  # only the cases that carry the event
    x = x[[(p, s) in carry for p, s in zip(x.policy, x.selection)]].copy()
    check("T3_cases_carrying_event", len(x) >= 1 and "XH" in set(x.rule), f"{len(x)} of 21")
    x["allow_per_event_kWh"] = TGT * x.load_kWh / x.events
    x["ratio"] = float(ev.energy_kWh) / x.allow_per_event_kWh
    g = x.groupby("rule").ratio.agg(["count", "min", "max"]).reset_index()
    xh = g.set_index("rule").loc["XH"]
    ok = check("T3_gate_XH", LIT("XH_ratio_min_max", [round(xh["min"], 1), round(xh["max"], 1)], [3.0, 3.6]), (xh["min"], xh["max"]))
    return x[["rule", "policy", "selection", "load_kWh", "events", "allow_per_event_kWh", "ratio"]], g, ev, ok


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cd, ok1 = t1()
    if ok1:
        cd.to_csv(OUT / "complete_days.csv", index=False)
    print(cd.to_string())
    cov, ok2 = t2()
    if ok2:
        cov.to_csv(OUT / "covered_2025.csv", index=False)
    print(cov.to_string())
    x, g, ev, ok3 = t3()
    if ok3:
        x.to_csv(OUT / "event_ratio.csv", index=False)
    print(g.to_string(), "\nall rules:", x.ratio.min(), x.ratio.max())
    ok4 = check("T4_gate_online_1000", LIT("peak_event_online_kW", float(ev.online_kW), 1000.0), ev.online_kW)
    t4 = dict(peak_kW=float(ev.peak_kW), online_kW=float(ev.online_kW), share_online=float(ev.peak_kW) / float(ev.online_kW))
    for k in ev.index:
        if "largest" in k:
            t4[k] = ev[k] if not isinstance(ev[k], (np.floating, np.integer)) else float(ev[k])
    print("T4", t4)
    json.dump(dict(checks=CHECKS, t4=t4 if ok4 else None, all_rules=[float(x.ratio.min()), float(x.ratio.max())]),
              open(OUT / "checks.json", "w"), indent=1, default=str)
    print("ALL PASS" if all(v["ok"] for v in CHECKS.values()) else "SOME FAIL")


if __name__ == "__main__":
    main()
