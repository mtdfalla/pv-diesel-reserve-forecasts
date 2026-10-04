#!/usr/bin/env python3
"""Readings of the held-back fold 2025 (rule D-129). Writes 04_results/ce1/WP67/f2025/.

Usage (from the repository root unless the path starts at ce1/, then from 03_code):
    python3 03_code/ce1/ce1_wp67_f2025_read.py
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

R = Path(".") / "04_results" / "ce1"
OUT = R / "WP67" / "f2025"
TGT, EPS, LIMIT, LEVEL_TOL, SAV_MIN = 2e-5, 1e-15, 1.0, 1e-6, 0.01
QP = ["raw_q", "cqr", "mondrian_cqr"]
CP = ["icp", "mondrian", "cqr", "mondrian_cqr"]
RULES = [("A_R", "A-R"), ("b_xfit_0.002", "XF"), ("c_prior_0.002", "PF"), ("a_half_H2", "H2"),
         ("b_xfit_0.001", "XH"), ("c_prior_0.001", "PH"), ("A_D", "A-D"), ("AW_clip", "CL")]


def md5(p):
    return hashlib.md5(open(p, "rb").read()).hexdigest()


def cp_lower(k, n, a=0.05):
    return 0.0 if k == 0 else float(stats.beta.ppf(a, k, n - k + 1))


def main():
    d = pd.read_csv(OUT / "f2025_cases.csv", float_precision="round_trip")
    chk = json.load(open(OUT / "f2025_checks.json"))
    val = json.load(open(OUT / "validate_f2024" / "f2024_checks.json"))
    assert chk["ok"] and val["ok"], "fold 2025 or its fold-2024 validation did not pass"
    assert len(d) == 2 * 8 * 8, len(d)
    d["meets_chk"] = d.ue_share <= TGT + EPS
    assert (d.meets == d.meets_chk).all()
    # ---- C1
    c1 = d[(d.selection == "b_xfit_0.001") & d.policy.isin(QP)]
    assert len(c1) == 6
    c1_ok = bool(c1.meets.all() and (c1.saving_pct >= SAV_MIN).all())
    # ---- C2
    ad = d[(d.selection == "A_D") & d.policy.isin(QP)]
    assert len(ad) == 6
    xh_med, ad_med = float(c1.saving_pct.median()), float(ad.saving_pct.median())
    c2_ok = xh_med > ad_med
    # ---- year-level (D-129 general rule 4)
    yl = pd.read_csv(R / "WP67" / "year_level.csv")
    six = yl[(yl.selection == "b_xfit_0.001") & (yl.set == "quantile")].iloc[0]
    k7 = int(six.years_met) + int(c1.meets.all())
    # ---- C4
    ar = d[d.selection == "A_R"]
    c4 = dict(conformal_met=int(ar[ar.policy.isin(CP)].meets.sum()), conformal_cases=int(ar.policy.isin(CP).sum()),
              quantile_met=int(ar[ar.policy.isin(QP)].meets.sum()), quantile_cases=int(ar.policy.isin(QP).sum()))
    # ---- C3, lag
    v = json.load(open(R / "WP67" / "c3lag" / "verdict.json"))
    assert v["ok"]
    # ---- every rule, both sets (descriptive), with the item-3 Met split
    rows = []
    for sel, lab in RULES:
        for sname, pols in (("conformal", CP), ("quantile", QP)):
            c = d[(d.selection == sel) & d.policy.isin(pols)]
            met = c[c.meets]
            at_lim = met.param >= LIMIT - LEVEL_TOL
            hb = met[(met.saving_pct < SAV_MIN) | at_lim]
            rows.append(dict(rule=lab, selection=sel, set=sname, cases=len(c), met=int(c.meets.sum()),
                             met_with_saving=len(met) - len(hb), met_handbook_level=len(hb),
                             saving_median_pct=float(c.saving_pct.median()), saving_min_pct=float(c.saving_pct.min()),
                             saving_max_pct=float(c.saving_pct.max()), events_median=float(c.events.median()),
                             event_peak_max_kW=float(c.event_peak_max_kW.max()), starts_median=float(c.starts.median()),
                             met_complete_days=int((c.ue_share_complete <= TGT + EPS).sum())))
    T = pd.DataFrame(rows)
    T.to_csv(OUT / "f2025_rules.csv", index=False)
    n1 = pd.read_csv(OUT / "f2025_n1.csv")
    n1["meets"] = n1.ue_share <= TGT + EPS
    n1["saving_pct"] = 100 * (1 - n1.fuel_L / n1.fuel_handbook)
    N1 = n1[n1.policy.isin(CP + QP)].groupby("selection").agg(cases=("meets", "size"), met=("meets", "sum"),
                                                              saving_median_pct=("saving_pct", "median"),
                                                              pv_binds_share_max=("pv_binds_share", "max")).reset_index()
    N1.to_csv(OUT / "f2025_n1_summary.csv", index=False)
    reading = dict(
        scope="pop=Yulara fold 2025 (test 2025, calibration 2024, training 2016-2023), h=15 min, shape=both, "
              "selection rules frozen (D-129), reported as a separate confirmatory row, never pooled",
        C1=dict(rule="XH, quantile policies, both shapes, 6 cases: all meet 0.002% and each saves >= 0.01% of handbook fuel",
                met=int(c1.meets.sum()), cases=6, saving_min_pct=float(c1.saving_pct.min()),
                saving_median_pct=xh_med, saving_max_pct=float(c1.saving_pct.max()),
                ue_share_max=float(c1.ue_share.max()), confirmed=c1_ok),
        C2=dict(rule="XH median saving over the C1 cases exceeds A-D's over the same cases",
                xh_median_pct=xh_med, ad_median_pct=ad_med, confirmed=bool(c2_ok)),
        year_level=dict(rule="D-129 general rule 4, XH quantile set", six_fold_years_met=int(six.years_met),
                        years_met=k7, years=7, cp_lower_95_one_sided=round(cp_lower(k7, 7), 4)),
        C3=v["c3"], C4=dict(rule="A-R met counts 2025, DR1 not re-evaluated", **c4),
        lag=v["lag"],
        context=dict(note="every static rule and A-D meet the target in every 2025 case of both sets",
                     all_static_and_AD_met=bool(d[d.selection != "AW_clip"].meets.all())),
        inputs={p: md5(R / p) for p in ("WP67/f2025/f2025_cases.csv", "WP67/f2025/f2025_n1.csv", "WP67/year_level.csv",
                                        "WP67/c3lag/verdict.json")},
        code={"ce1_wp67_f2025_read.py": md5(Path(__file__))})
    json.dump(reading, open(OUT / "reading.json", "w"), indent=1, default=float)
    print(T.to_string(index=False))
    print(N1.to_string(index=False))
    print(json.dumps({k: reading[k] for k in ("C1", "C2", "year_level", "C3", "C4", "context")}, indent=1, default=float))


if __name__ == "__main__":
    sys.exit(main())
