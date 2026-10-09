#!/usr/bin/env python3
"""Analyses from existing outputs under the rules of D-129, one subcommand per item. Each writes to
04_results/ce1/WP67/. Items with csv or json inputs use the standard library only; items that
read parquet need pandas and pyarrow.

Usage (from the repository root unless the path starts at ce1/, then from 03_code):
    python3 03_code/ce1/ce1_wp67.py c1
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import ce1_literal as LT

ROOT = Path.cwd()
R = ROOT / "04_results" / "ce1"
OUT = R / "WP67"
FOLDS = [2019, 2020, 2021, 2022, 2023, 2024]


def headline_cells():
    """The S63 item 7 headline cells in the order ce1_s63.py lists them, before and after deduplication."""
    head = [("yulara", 2024, 5, "icp", "transitional"), ("yulara", 2024, 5, "mondrian_cqr", "transitional"),
            ("yulara", 2024, 5, "icp", "clear"), ("yulara", 2024, 5, "icp", "cloudy")]
    head += [("yulara", y, 5, m, "transitional") for y in FOLDS for m in ("icp", "mondrian", "mondrian_cqr")]
    head += [("yulara", 2024, h, "icp", "all") for h in (5, 15, 30, 60)]
    head += [("yulara", 2024, h, m, "transitional") for h in (5, 15, 30, 60) for m in ("icp", "tod", "mondrian_cqr")]
    return head, list(dict.fromkeys(head))


def c1():
    rows = list(csv.DictReader(open(R / "S63" / "complete_days_coverage.csv", newline="", encoding="utf-8")))
    idx = {(r["site"], int(r["fold"]), int(r["horizon_min"]), r["method"], r["scope"]): r for r in rows}
    raw, uniq = headline_cells()
    out = []
    for k in uniq:
        r = idx[k]
        ch = float(r["picp_complete"]) - float(r["picp"])
        out.append(dict(site=k[0], fold=k[1], horizon_min=k[2], method=k[3], scope=k[4], picp=r["picp"],
                        picp_complete=r["picp_complete"], change=ch, over_0_01=abs(ch) > 0.01))
    over = [o for o in out if o["over_0_01"]]
    stored = json.loads((R / "S63" / "reading.json").read_text(encoding="utf-8"))["7_complete_days"]
    # ce1_s63.py counts each cell once (D-129). The v2.0.0 S63 record predates that fix and stores
    # the as-run 38; an S63 record from the fixed code stores the 34 distinct cells. Either is the same set of cells.
    assert len(raw) == 38 and stored["headline_cells"] in (38, len(uniq)), ("as-run list has 38 entries", stored["headline_cells"])
    assert len(uniq) == 34, len(uniq)
    assert len(over) == len(stored["cells_over_0_01"]) == 9, (len(over), len(stored["cells_over_0_01"]))
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "complete_days_headline.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)
    rec = dict(item="D-129 step 1 C1", listed_as_run=len(raw), distinct_cells=len(uniq),
               duplicates=len(raw) - len(uniq), cells_over_0_01=len(over),
               max_abs_change=max(abs(o["change"]) for o in out),
               scope="Yulara, 90% nominal, the headline coverage cells of S63 item 7 (fold 2024 at 5-60 min and "
                     "six folds at 5 min, transitional), 1-day blocks n/a",
               printed="9 of 34")
    (OUT / "complete_days_headline.json").write_text(json.dumps(rec, indent=1), encoding="utf-8")
    print(json.dumps(rec, indent=1))


# ---------------------------------------------------------------- step 2, shared definitions
TGT, EPS, LIMIT, LEVEL_TOL = 2e-5, 1e-15, 1.0, 1e-6
SAV_MIN = 0.01  # D-129 item 3: a saving of at least 0.01 % of handbook fuel
CP = ["icp", "mondrian", "cqr", "mondrian_cqr"]
QP = ["raw_q", "cqr", "mondrian_cqr"]
SETS = {"conformal": CP, "quantile": QP}
RULES = [("A_R", "A-R"), ("b_xfit_0.002", "XF"), ("c_prior_0.002", "PF"), ("a_half_H2", "H2"),
         ("b_xfit_0.001", "XH"), ("c_prior_0.001", "PH"), ("A_D", "A-D"), ("AW_clip", "CL")]
PRINTED_YRS = {"A-R": (1, 3), "XF": (1, 3), "PF": (0, 3), "H2": (3, 5), "XH": (6, 6), "PH": (5, 6),
               "A-D": (6, 6), "CL": (0, 0)}
PRINTED_MET = {"A-R": (25, 21), "XF": (34, 21), "PF": (26, 22), "H2": (42, 33), "XH": (48, 36), "PH": (47, 36),
               "A-D": (48, 36), "CL": (0, 0)}


def _binom_sf(k, n, p):
    """P(X >= k) for X ~ Binomial(n, p), standard library."""
    from math import comb
    return sum(comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(k, n + 1))


def cp_lower(k, n, alpha=0.05):
    """One-sided (1 - alpha) Clopper-Pearson lower bound for a proportion k of n (D-129 general rule 4)."""
    if k == 0:
        return 0.0
    lo, hi = 0.0, 1.0
    for _ in range(200):
        mid = (lo + hi) / 2
        if _binom_sf(k, n, mid) < alpha:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def sign_p(k, n):
    """One-sided exact sign test: P(X >= k | n, 1/2)."""
    return _binom_sf(k, n, 0.5)


def read_cases():
    rows = list(csv.DictReader(open(R / "WP63" / "cases.csv", newline="", encoding="utf-8")))
    for r in rows:
        r["fold"] = int(r["fold"])
        r["meets"] = r["meets"] == "True"
        r["saving_pct"] = float(r["saving_pct"])
        r["param"] = float(r["param"]) if r["param"] not in ("", "nan") else float("nan")
        r["days_departing"] = int(float(r["days_departing"]))
    assert len(rows) == 768, len(rows)
    return rows


def _median(x):
    x = sorted(x)
    n = len(x)
    return x[n // 2] if n % 2 else (x[n // 2 - 1] + x[n // 2]) / 2


def item1():
    """D-129 step 2 item 1: years met of 6 with the one-sided Clopper-Pearson bound, sign test of the per-year median saving."""
    cases = read_cases()
    LIT = LT.Literal(OUT, "ce1_wp67.py item1")
    out = []
    for sel, lab in RULES:
        for si, (sname, pols) in enumerate(SETS.items()):
            yrs_met, pos, meds = 0, 0, {}
            n_cases = met_cases = 0
            for y in FOLDS:
                c = [r for r in cases if r["selection"] == sel and r["policy"] in pols and r["fold"] == y]
                assert len(c) == 2 * len(pols), (sel, sname, y, len(c))
                n_cases += len(c)
                met_cases += sum(r["meets"] for r in c)
                yrs_met += all(r["meets"] for r in c)
                m = _median([r["saving_pct"] for r in c])
                meds[y] = m
                pos += m > 0
            assert LIT(f"years_met_{lab}_{sname}", yrs_met, PRINTED_YRS[lab][si]), (lab, sname, yrs_met, PRINTED_YRS[lab][si])
            assert LIT(f"cases_met_{lab}_{sname}", met_cases, PRINTED_MET[lab][si]), (lab, sname, met_cases, PRINTED_MET[lab][si])
            out.append(dict(rule=lab, selection=sel, set=sname, cases=n_cases, cases_met=met_cases,
                            years_met=yrs_met, years=len(FOLDS), cp_lower_95=round(cp_lower(yrs_met, 6), 4),
                            years_positive_median_saving=pos, sign_p_one_sided=sign_p(pos, 6),
                            **{f"median_saving_{y}": meds[y] for y in FOLDS}))
    assert abs(cp_lower(6, 6) - 0.6070) < 5e-5, cp_lower(6, 6)
    OUT.mkdir(parents=True, exist_ok=True)
    _write(out, "year_level.csv")
    rec = dict(item="D-129 step 2 item 1",
               scope="pop=Table 3 rules x {conformal 4 policies, quantile 3 policies}, h=15 min, shape=both, folds=2019-2024, "
                     "unit=test year (a year is met when every case of the set meets 0.002 %), CP one-sided 95 %, "
                     "sign test one-sided exact on the per-year median saving, no multiplicity adjustment",
               cp_6_of_6=round(cp_lower(6, 6), 4), cp_5_of_6=round(cp_lower(5, 6), 4), sign_p_6_of_6=sign_p(6, 6),
               rows={f"{o['rule']}|{o['set']}": dict(years_met=o["years_met"], cp=o["cp_lower_95"],
                                                       pos=o["years_positive_median_saving"], p=o["sign_p_one_sided"])
                     for o in out})
    (OUT / "year_level.json").write_text(json.dumps(rec, indent=1), encoding="utf-8")
    print(json.dumps(rec, indent=1))


def item3():
    """D-129 step 2 item 3: met with a saving of at least 0.01 %, and met at the handbook level
    (saving below 0.01 % or the level at the limit on every day).
    """
    cases = read_cases()
    LIT = LT.Literal(OUT, "ce1_wp67.py item3")
    out = []
    for sel, lab in RULES:
        for si, (sname, pols) in enumerate(SETS.items()):
            c = [r for r in cases if r["selection"] == sel and r["policy"] in pols]
            met = [r for r in c if r["meets"]]
            at_lim = lambda r: r["days_departing"] == 0 or (r["param"] == r["param"] and r["param"] >= LIMIT - LEVEL_TOL)  # noqa: E731
            hb = [r for r in met if r["saving_pct"] < SAV_MIN or at_lim(r)]
            sav = [r for r in met if not (r["saving_pct"] < SAV_MIN or at_lim(r))]
            assert len(hb) + len(sav) == len(met) and LIT(f"cases_met_{lab}_{sname}", len(met), PRINTED_MET[lab][si])
            assert all(r["saving_pct"] >= SAV_MIN for r in sav)
            as_run = sum(r["meets"] and r["saving_pct"] > 0.005 for r in c)  # WP-63 'met_nonzero' (prints as 0.01)
            out.append(dict(rule=lab, selection=sel, set=sname, cases=len(c), met=len(met), met_with_saving=len(sav),
                            met_handbook_level=len(hb), met_at_limit_every_day=sum(at_lim(r) for r in met),
                            met_saving_below_0_01_not_at_limit=sum((r["saving_pct"] < SAV_MIN) and not at_lim(r) for r in met),
                            wp63_met_nonzero_gt_0_005=as_run))
    _write(out, "met_split.csv")
    rec = dict(item="D-129 step 2 item 3",
               scope="pop=Table 3 rules x {conformal 4, quantile 3}, h=15 min, shape=both, folds=2019-2024, criterion "
                     "saving >= 0.01 % of handbook fuel (D-129 literal), handbook level = saving < 0.01 % or level at "
                     "the limit on every day",
               rows={f"{o['rule']}|{o['set']}": {k: o[k] for k in ("met", "met_with_saving", "met_handbook_level",
                                                                  "met_at_limit_every_day", "wp63_met_nonzero_gt_0_005")}
                     for o in out})
    (OUT / "met_split.json").write_text(json.dumps(rec, indent=1), encoding="utf-8")
    print(json.dumps(rec, indent=1))


def item2():
    v = json.loads((R / "S7E" / "verdicts.json").read_text(encoding="utf-8"))
    v7d = json.loads((R / "S7D" / "verdicts.json").read_text(encoding="utf-8"))
    assert v == v7d, "S7E must reproduce S7D (D-097 context)"
    LIT = LT.Literal(OUT, "ce1_wp67.py item2")  # outcome strings computed from the result files
    s63 = json.loads((R / "S63" / "reading.json").read_text(encoding="utf-8"))
    s64 = json.loads((R / "S64" / "reading.json").read_text(encoding="utf-8"))
    sets = {(r["selection"], r["set"]): r for r in csv.DictReader(open(R / "WP63" / "table4_sets.csv", encoding="utf-8"))}
    dr1_max = max(p["folds_passing_both_shapes"] for p in v["DR1"].values())
    assert LIT("DR1_max_folds_and_any_holds", [dr1_max, any(p["holds"] for p in v["DR1"].values())], [3, False])
    assert LIT("DR2_folds_gain_and_credited", [v["DR2"]["folds_gain"], bool(v["DR2"]["credited"])], [0, False])
    assert LIT("DR3_folds_gain", [v[k]["folds_gain"] for k in ("DR3_mcqr_vs_cqr", "DR3_mondrian_vs_icp", "DR3_mondrian_vs_nicp_var")], [0, 0, 0])
    assert LIT("DR3_folds_cost", [v[k]["folds_cost"] for k in ("DR3_mcqr_vs_cqr", "DR3_mondrian_vs_icp", "DR3_mondrian_vs_nicp_var")], [0, 1, 6])
    assert LIT("DR4_outcome_and_max_folds", [v["DR4"]["outcome"], max(v["DR4"]["folds_passing"].values())], ["DROPPED", 0])
    assert LIT("DR5_top_agree", v["DR5"]["top_agree"], 0) and v["DR5"]["n"] == 24
    assert LIT("PR2_H2_conformal_met", int(sets[("a_half_H2", "conformal")]["met"]), 42)  # PR-2 threshold 40 of 48
    assert LIT("AD_AR_conformal_met", [int(sets[("A_D", "conformal")]["met"]), int(sets[("A_R", "conformal")]["met"])], [48, 25])
    import ast
    o1 = ast.literal_eval(s63["1_order_stats"]) if isinstance(s63["1_order_stats"], str) else s63["1_order_stats"]
    assert o1["met_interpolated"] == o1["met_order_stat"] and LIT("S63_order_stats_met", o1["met_order_stat"], 25)
    s5 = ast.literal_eval(s63["5_persistence"]) if isinstance(s63["5_persistence"], str) else s63["5_persistence"]
    assert s5["folds_negative"] == 0
    s7 = ast.literal_eval(s63["7_complete_days"]) if isinstance(s63["7_complete_days"], str) else s63["7_complete_days"]
    assert len(s7["cells_over_0_01"]) == 9
    b7 = ast.literal_eval(s63["8_blocks7"]) if isinstance(s63["8_blocks7"], str) else s63["8_blocks7"]
    assert b7["cells"] == 1008 and b7["status_changes"] == 65, (b7["cells"], b7["status_changes"])
    hold = s64["3_hold"]
    assert LIT("S64_hold_met_conformal_and_triggers",
               [hold["met_conformal"], bool(hold["controller_to_supplement"]), bool(hold["pid_trigger_satisfied"])], [44, False, False])
    # outcomes computed from the verdict and reading files
    NM = {"icp": "marginal ICP", "mondrian": "Mondrian", "cqr": "CQR", "mondrian_cqr": "Mondrian-CQR", "raw_q": "the raw quantile",
          "tod_q": "the time-of-day quantile baseline", "nicp_var": "VN-ICP", "frac_pv": "the fixed fraction of PV"}

    def joinw(xs):
        xs = [str(x) for x in xs]
        return xs[0] if len(xs) == 1 else ", ".join(xs[:-1]) + " and " + xs[-1]
    hold1 = [p for p, x in v["DR1"].items() if x["holds"]]
    rest1 = max(x["folds_passing_both_shapes"] for x in v["DR1"].values() if not x["holds"])
    if hold1:
        k1 = sorted({v["DR1"][p]["folds_passing_both_shapes"] for p in hold1})
        dr1_out = (f"PASS for {joinw(NM[p] for p in hold1)} only ({joinw(k1)} of 6 folds), "
                   f"FAIL for every other policy (at most {rest1} of 6 folds)")
    else:
        dr1_out = f"FAIL: no policy passes (at most {rest1} of 6 folds)"
    assert not v["DR2"]["credited"]
    dr2_out = f"FAIL: {v['DR2']['folds_gain']} of 6 folds (no calibration gain in fuel credited)"

    def dr3_out(k):
        g, c_ = v[k]["folds_gain"], v[k]["folds_cost"]
        assert not (g >= 4 and c_ >= 4), k
        return f"costs fuel: {c_} of 6 folds" if c_ >= 4 else (f"regime gain: {g} of 6 folds" if g >= 4 else f"neither: gain {g}, cost {c_} of 6")
    fp4 = set(v["DR4"]["folds_passing"].values())
    assert v["DR4"]["outcome"] == "DROPPED" and len(fp4) == 1, v["DR4"]
    dr4_out = f"FAIL ({fp4.pop()} folds for every policy)"
    assert v["DR5"]["differs_any"]
    dr5_out = f"reported: rankings differ, top policy agrees in {v['DR5']['top_agree']} of {v['DR5']['n']} fold-horizons"
    import statistics
    pr1 = statistics.median(float(r_["net_pct"]) for r_ in csv.DictReader(open(R / "WP55" / "pr1_cells.csv", encoding="utf-8"))
                            if r_["case"] == "central")
    assert pr1 > 0, pr1
    pr1_out = f"NOT FIRED: {pr1:.2f}% net central"
    pr2m = int(sets[("a_half_H2", "conformal")]["met"])
    assert pr2m >= 40, pr2m
    p3 = json.loads((R / "WP58" / "pr3.json").read_text(encoding="utf-8"))
    assert p3["fired"]
    pr3_out = (f"FIRED: ICP {p3['icp_folds_ratio_gt_2']} of 6, Mondrian at most 2 in {p3['mondrian_folds_ratio_le_2']}, "
               f"M-CQR in {p3['mondrian_cqr_folds_ratio_le_2']}")
    assert not o1["main_text"]
    s631_out = (f"Supplement sentence: {o1['met_order_stat']} of 48 both ways, "
                f"{o1['median_saving_order_stat'] - o1['median_saving_interpolated']:+.3f} points")
    mono = list(csv.DictReader(open(R / "S63" / "monotone_rearranged.csv", encoding="utf-8")))
    mq = [r_ for r_ in mono if r_["policy"] in ("cqr", "mondrian_cqr", "raw_q")]
    mc = [r_ for r_ in mono if r_["policy"] in ("icp", "mondrian", "cqr", "mondrian_cqr")]
    assert (len(mq), len(mc)) == (36, 48), (len(mq), len(mc))
    s636_out = (f"FIRED: sorted variant reported (quantile {sum(r_['meets_rearranged'] == 'True' for r_ in mq)} of 36, "
                f"conformal {sum(r_['meets_rearranged'] == 'True' for r_ in mc)} of 48)")
    hc = s7["headline_cells"]
    s637_out = f"FIRED: {len(s7['cells_over_0_01'])} of 34 distinct cells" + (f" ({hc} listed as run)" if hc != 34 else "")
    sp_ = s64["1_spill"]
    pq = sp_["per_quantile_policy"]
    assert not sp_["trigger_fires_any_policy"]
    s641_out = f"NOT FIRED (XH {pq['raw_q']['static_met']}, {pq['cqr']['static_met']}, {pq['mondrian_cqr']['static_met']} of 12)"
    assert not hold["controller_to_supplement"] and not hold["pid_trigger_satisfied"]
    s643_out = f"neither: {hold['met_conformal']} of 48"
    t = [
     ("DR1", "D-091 (WP-44 spec 5)", "2026-09-28", "15 min, A-R, test year: a fold passes for policy p when p's level uses less fuel than the handbook rule with UE share at most 0.002 %, main and shape B; holds for p in at least 4 of 6 folds", dr1_out, "S7E/verdicts.json DR1"),
     ("DR1-D", "D-094", "2026-09-28", "DR1 applied unchanged to A-D", "PASS for 4 of 4 conformal policies (claim rule (a) applies)", "D-094 result, S51/S51S"),
     ("DR2", "D-091", "2026-09-28", "F(best of CQR, M-CQR on H2) - F(raw_q) upper 95 % end below zero in at least 4 of 6 folds (main, 15 min)", dr2_out, "S7E/verdicts.json DR2"),
     ("DR3 M-CQR vs CQR", "D-091", "2026-09-28", "regime gain if the interval lies below zero in 4 of 6 folds; costs fuel if it lies above zero in 4 of 6", dr3_out("DR3_mcqr_vs_cqr"), "S7E/verdicts.json"),
     ("DR3 Mondrian vs ICP", "D-091", "2026-09-28", "as above", dr3_out("DR3_mondrian_vs_icp"), "S7E/verdicts.json"),
     ("DR3 Mondrian vs VN-ICP", "D-091", "2026-09-28", "as above", dr3_out("DR3_mondrian_vs_nicp_var"), "S7E/verdicts.json"),
     ("DR4", "D-091", "2026-09-28", "battery raises the A-R level: kept if each conformal policy passes in 4 of 6 folds, dropped if none does", dr4_out, "S7E/verdicts.json DR4"),
     ("DR5", "D-091", "2026-09-28", "proxy ranking against load-model ranking, Kendall tau and top agreement; load model governs either way", dr5_out, "S7E/verdicts.json DR5"),
     ("DR6", "D-091", "2026-09-28", "each conformal policy against frac_pv and tod_q, absolute differences with intervals (A-R and A-C)", "reported (descriptive, no pass criterion)", "S7E, Supplement"),
     ("PR-1", "D-110", "2026-09-30", "if the A-D median quantile saving at 15 min is at or below zero at the central start-up fuel, reframe title and abstract", pr1_out, "WP55/pr1_cells.csv (rule D-110, first reading D-111)"),
     ("PR-2", "D-110 (operational D-112)", "2026-09-30", "if A-R at half the allowance meets the target in at least 40 of 48 conformal cases, credit the margin", f"FIRED: {pr2m} of 48", "WP63/table4_sets.csv"),
     ("PR-3", "D-110 (operational D-112)", "2026-09-30", "ICP issue-time regime exceedance ratio above 2 in at least 4 of 6 folds and at most 2 for Mondrian or M-CQR", pr3_out, "WP58/pr3.json (first reading D-113)"),
     ("S63-1 order statistics", "D-122", "2026-09-30", "main text if met changes by more than 2 of 48 or median saving by more than 0.05 points", s631_out, "S63/reading.json"),
     ("S63-4 lower tail", "D-122", "2026-09-30", "reliable in the lower tail only if Kupiec does not reject in 2 of 3 regimes in 4 of 6 folds at every theta up to 0.99", "FAIL for every method (0 of 7)", "S63/reading.json, D-124"),
     ("S63-5 persistence reference", "D-122", "2026-09-30", "no CRPS superiority claim if M-CQR transitional skill is negative in 4 of 6 folds", "NOT FIRED: positive in 6 of 6", "S63/reading.json"),
     ("S63-6 monotonicity", "D-122", "2026-09-30", "if any policy exceeds 0.1 % of steps, a sorted variant is reported", s636_out, "S63/monotone_rearranged.csv (first reading D-124)"),
     ("S63-7 complete days", "D-122 (D-123)", "2026-09-30", "a headline coverage value changing by more than 0.01 is given with its restricted value", s637_out, "WP67/complete_days_headline.json"),
     ("S63-8 7-day blocks", "D-122", "2026-09-30", "where the class against nominal changes, cite the 7-day interval", "FIRED where the class changes (65 of 1,008 cells)", "S63/reading.json"),
     ("S64-1 spill-matched scale", "D-122", "2026-09-30", "D-121 trigger if XH meets fewer than 10 of 12 per quantile policy while A-D meets at least 11 of 12", s641_out, "S64/reading.json (first reading D-124)"),
     ("S64-2 commitment constraints", "D-122", "2026-09-30", "(i) abstract conditional if the leading rule's median quantile saving changes sign; (ii) start-up net rule", "NOT FIRED: no sign change", "D-124"),
     ("S64-3 hold comparator", "D-122", "2026-09-30", "controller to Supplement if hold meets at least 46 of 48 and saves at least A-D; PID trigger if fewer than 42 of 48", s643_out, "S64/reading.json"),
    ]
    rows = [dict(rule=a, registered_in=b, registered_date=c, wording=d, outcome=e, evidence=f) for a, b, c, d, e, f in t]
    assert len(rows) == 21
    OUT.mkdir(parents=True, exist_ok=True)
    _write(rows, "registered_rules.csv")
    rec = dict(item="D-129 step 2 item 2", n_rules=len(rows),
               fail_or_not_credited=[r["rule"] for r in rows if r["outcome"].startswith("FAIL")],
               fired=[r["rule"] for r in rows if r["outcome"].startswith("FIRED")],
               main_text_sentence_basis=f"DR4 fails, DR5 rankings differ (top agreement {v['DR5']['top_agree']} of {v['DR5']['n']}), DR6 descriptive",
               scope="registered rules of D-091, D-094, D-110/D-112 and D-122 with outcomes from the verdict and reading files")
    (OUT / "registered_rules.json").write_text(json.dumps(rec, indent=1), encoding="utf-8")
    print(json.dumps(rec, indent=1))


def item6():
    import numpy as np
    import pandas as pd
    sys.path.insert(0, str(ROOT / "03_code" / "ce1"))
    import ce1_wp57 as W57
    import ce1_load as LM
    TG, HALF = LM.UE_TARGET, LM.UE_TARGET / 2.0
    assert TG == TGT
    pols = ["icp", "mondrian", "cqr", "mondrian_cqr", "raw_q"]
    tot = {y: pd.read_parquet(R / "S64" / f"f{y}_totals.parquet") for y in FOLDS}
    s64 = pd.read_csv(R / "S64" / "cases.csv", float_precision="round_trip")
    base = pd.read_csv(R / "WP63" / "cases.csv")
    SU_C, SU_H = 1.708, None
    rows, nchk = [], 0
    for cname in ("min_down", "deadband"):
        for sname in ("main", "shapeB"):
            sc = f"{sname}_{cname}"
            for yr in FOLDS:
                T = {p: tot[yr][(tot[yr].scenario == sc) & (tot[yr].set == p)].reset_index(drop=True) for p in ("H1x", "H2", "test")}
                te = T["test"]
                hb = te[te.policy == "handbook"].iloc[0]
                prior = [tot[y][(tot[y].scenario == sc) & (tot[y].set == "test")].reset_index(drop=True) for y in FOLDS if y < yr]
                for pol in pols:
                    X = W57.pooled([T["H2"][T["H2"].policy == pol].reset_index(drop=True),
                                    T["H1x"][T["H1x"].policy == pol].reset_index(drop=True)])
                    Pp = (W57.pooled([t[t.policy == pol].reset_index(drop=True) for t in prior]) if prior
                          else T["H2"][T["H2"].policy == pol].reset_index(drop=True))
                    for sel, (tab, tg) in {"A_R": (T["H2"], TG), "b_xfit_0.002": (X, TG), "c_prior_0.002": (Pp, TG),
                                           "a_half_H2": (T["H2"], HALF), "b_xfit_0.001": (X, HALF),
                                           "c_prior_0.001": (Pp, HALF)}.items():
                        p_, met = W57.select(tab, pol, tg)
                        rr = W57.at(te, pol, p_)
                        if sel in ("A_R", "b_xfit_0.001"):
                            ref = s64[(s64.item == cname) & (s64.scenario == sname) & (s64.fold == yr) & (s64.policy == pol) & (s64.selection == sel)]
                            assert len(ref) == 1 and np.isclose(ref.param.iloc[0], p_, atol=1e-12) \
                                and abs(float(ref.fuel_L.iloc[0]) / float(rr["fuel_L"]) - 1) < 1e-12 \
                                and abs(float(ref.ue_share.iloc[0]) - float(rr["ue_share"])) <= 1e-12 * max(float(rr["ue_share"]), 1e-9), \
                                (cname, sname, yr, pol, sel)
                            nchk += 1
                        rows.append(dict(constraint=cname, scenario=sname, fold=yr, policy=pol, selection=sel, param=p_,
                                         fuel_L=float(rr["fuel_L"]), ue_share=float(rr["ue_share"]), starts=float(rr["starts"]),
                                         events=float(rr["events"]), fuel_handbook=float(hb.fuel_L), handbook_starts=float(hb.starts)))
                    ref = s64[(s64.item == cname) & (s64.scenario == sname) & (s64.fold == yr) & (s64.policy == pol) & (s64.selection == "A_D")]
                    assert len(ref) == 1
                    r0 = ref.iloc[0]
                    rows.append(dict(constraint=cname, scenario=sname, fold=yr, policy=pol, selection="A_D", param=float("nan"),
                                     fuel_L=float(r0.fuel_L), ue_share=float(r0.ue_share), starts=float(r0.starts),
                                     events=float(r0.events) if r0.events == r0.events else float("nan"),
                                     fuel_handbook=float(hb.fuel_L), handbook_starts=float(hb.starts)))
    assert nchk == 2 * 2 * 6 * 5 * 2, nchk
    df = pd.DataFrame(rows)
    df["meets"] = df.ue_share <= TGT + EPS
    df["saving_pct"] = 100 * (1 - df.fuel_L / df.fuel_handbook)
    df["d_starts"] = df.starts - df.handbook_starts
    su = {"central": 0.0205 * 1000 * 5 / 60, "high": 0.0205 * 2500 * 10 / 60}
    for k, v in su.items():
        df[f"net_pct_{k}"] = 100 * (df.fuel_handbook - df.fuel_L - v * df.d_starts) / df.fuel_handbook
    df.to_csv(OUT / "constraints_cases.csv", index=False)
    summ = []
    for cname in ("min_down", "deadband"):
        for sel, lab in RULES:
            if sel == "AW_clip":
                continue
            for sname_, pl in SETS.items():
                g = df[(df.constraint == cname) & (df.selection == sel) & df.policy.isin(pl)]
                b = base[(base.selection == sel) & base.policy.isin(pl)]
                assert len(g) == 12 * len(pl) == len(b)
                summ.append(dict(constraint=cname, rule=lab, set=sname_, n=len(g), met=int(g.meets.sum()),
                                 met_unconstrained=int(b.meets.sum()),
                                 med_saving=float(g.saving_pct.median()), med_saving_unconstrained=float(b.saving_pct.median()),
                                 med_net_central=float(g.net_pct_central.median()), med_net_high=float(g.net_pct_high.median()),
                                 med_d_starts=float(g.d_starts.median()), med_events=float(g.events.median())))
    S = pd.DataFrame(summ)
    S.to_csv(OUT / "constraints_summary.csv", index=False)
    x = S[(S.rule == "XH") & (S.set == "quantile")].set_index("constraint")
    # D-122 S64 item 2 rules applied to XH as written: (i) sign change of the median quantile saving
    rule_i = {c: bool(np.sign(x.loc[c, "med_saving"]) != np.sign(x.loc[c, "med_saving_unconstrained"])) for c in x.index}
    rule_ii = {c: bool(x.loc[c, "med_net_high"] > 0) for c in x.index}
    rd = json.loads((R / "S64" / "reading.json").read_text(encoding="utf-8"))["2_constraints"]
    for c in ("min_down", "deadband"):
        assert abs(rd[c]["b_xfit_0.001"]["med_saving_quantile"] - x.loc[c, "med_saving"]) < 1e-12
        assert int(rd[c]["b_xfit_0.001"]["met_quantile"]) == int(x.loc[c, "met"])
    rec = dict(item="D-129 step 2 item 6", reproduced_s64_cells=nchk,
               scope="pop=Table 3 rules except CL x {conformal 4, quantile 3}, h=15 min, shape=both, folds=2019-2024, "
                     "constraint alone (min down 1 h, deadband 3 steps), levels reselected under it, PF/PH pool earlier "
                     "constrained test years (2019: H2), start-up central 1.71 L, high 8.54 L",
               xh_rule_i_sign_change=rule_i, xh_rule_ii_net_high_positive=rule_ii,
               rows={f"{r['constraint']}|{r['rule']}|{r['set']}": dict(met=r["met"], unc=r["met_unconstrained"],
                     sav=round(r["med_saving"], 4), sav_unc=round(r["med_saving_unconstrained"], 4),
                     net_c=round(r["med_net_central"], 4), ev=r["med_events"]) for r in summ})
    (OUT / "constraints.json").write_text(json.dumps(rec, indent=1), encoding="utf-8")
    print(json.dumps(rec, indent=1))


def item10():
    """D-129 step 2 item 10: fold 2024, 90 % two-sided coverage at 5 and 15 min by carried-input status
    (S2 column in_imp), per issue-time regime, and the Mondrian fallback share (regime groups with at most
    min_grp calibration rows use the marginal scores). Reading rule: a difference above 0.02 between carried
    and uncarried rows for any method is stated in the Limitations.
    """
    import numpy as np
    import pandas as pd
    out, flags = [], []
    for h, hm in ((1, 5), (3, 15)):
        z = np.load(R / "S5" / f"yulara_f2024_h{h}_obs.npz", allow_pickle=True)
        js = json.loads((R / "S5" / f"yulara_f2024_h{h}.json").read_text(encoding="utf-8"))
        c = pd.read_parquet(R / "S2" / f"yulara_f2024_h{h}.parquet")
        te, ca = c[c.split == "test"], c[c.split == "calib"]
        assert len(te) == len(z["y"]) == js["info"]["n_test"]
        assert np.array_equal(te.index.values.astype("datetime64[ns]").astype(np.int64), z["ts_ns"].astype(np.int64))
        assert np.allclose(te.y.values, z["y"].astype(float), atol=1e-3)
        imp = te.in_imp.values > 0.5
        reg = te.g.astype(str).values
        code = z["reg_it"].astype(int)
        pairs = set(zip(code.tolist(), reg.tolist()))
        assert len(pairs) == 3 and len({a for a, _ in pairs}) == 3, pairs  # codes map one to one onto the labels
        min_grp = js["settings"]["min_grp"]
        cal_n = ca.g.value_counts()
        fb = [g for g in ("clear", "transitional", "cloudy") if cal_n.get(g, 0) <= min_grp]
        fb_share = float(np.isin(reg, fb).mean())
        for meth in js["settings"]["static"]:
            cv = z[f"cov__{meth}|static|0.9"].astype(bool)
            for scope in ("all", "clear", "transitional", "cloudy"):
                m = np.ones(len(cv), bool) if scope == "all" else reg == scope
                a, b = cv[m & imp], cv[m & ~imp]
                d = float(a.mean() - b.mean()) if len(a) and len(b) else float("nan")
                out.append(dict(horizon_min=hm, method=meth, scope=scope, n_carried=int(len(a)), n_uncarried=int(len(b)),
                                picp_carried=float(a.mean()) if len(a) else float("nan"),
                                picp_uncarried=float(b.mean()) if len(b) else float("nan"), difference=d,
                                over_0_02=bool(abs(d) > 0.02) if d == d else False))
        flags.append(dict(horizon_min=hm, carried_share=float(imp.mean()), mondrian_fallback_groups=fb,
                          mondrian_fallback_share=fb_share, calib_regime_counts={k: int(v) for k, v in cal_n.items()}))
    _write(out, "coverage_carried.csv")
    over_all = [o for o in out if o["over_0_02"] and o["scope"] == "all"]
    over_reg = [o for o in out if o["over_0_02"] and o["scope"] != "all"]
    rec = dict(item="D-129 step 2 item 10",
               scope="pop=Yulara fold 2024 test rows, h=5 and 15 min, 90 % two-sided, static methods, by in_imp and issue-time regime",
               per_horizon=flags, cells=len(out),
               over_0_02_all_scope=[(o["horizon_min"], o["method"], round(o["difference"], 4)) for o in over_all],
               over_0_02_regime_scope=[(o["horizon_min"], o["method"], o["scope"], round(o["difference"], 4), o["n_carried"]) for o in over_reg],
               reading="stated in the Limitations" if (over_all or over_reg) else "no difference above 0.02")
    (OUT / "coverage_carried.json").write_text(json.dumps(rec, indent=1), encoding="utf-8")
    print(json.dumps(rec, indent=1))


def item11():
    """D-129 step 2 item 11: does the S1M map output fall as GHI rises at a fixed time of day?
    The map has no temperature input (GHI and the sun position only). At every daytime step of the six test
    years the map is evaluated on a GHI grid 0..1500 W/m2 in 5 W/m2 steps. A step counts when any consecutive
    pair falls by more than 1e-9 kW. Reported as a count, zero or not, with the largest fall.
    Speed: the sun position and extraterrestrial irradiance are computed once per timestamp and broadcast
    over the GHI grid through the same pvlib calls as ce1_pvmap.array_poa; a sample of points must equal
    ce1_pvmap.pv_available_from_ghi exactly before any count is kept.
    """
    import numpy as np
    import pandas as pd
    sys.path.insert(0, str(ROOT / "03_code" / "ce1"))
    import ce1_pvmap as PM
    import pvlib
    m = PM.load_map(R / "S1M" / "available_pv_map.npz")
    gg = np.arange(0.0, 1500.0 + 1e-9, 5.0)
    rng = np.random.default_rng(42)
    n_day = n_bad = 0
    worst, worst_at = 0.0, None
    per_year, checked = {}, 0
    for y in FOLDS:
        g = pd.date_range(f"{y}-01-01", f"{y}-12-31 23:55", freq="5min")
        t = g.tz_localize(PM.SITE["tz"], ambiguous="NaT", nonexistent="NaT")
        sp = pvlib.solarposition.get_solarposition(t, PM.SITE["lat"], PM.SITE["lon"], PM.SITE["alt"])
        zen_all = np.asarray(sp["apparent_zenith"])
        day = zen_all < 90.0
        zen = zen_all[day]
        azi = np.asarray(sp["azimuth"])[day]
        ext = np.asarray(pvlib.irradiance.get_extra_radiation(t))[day]
        Z, AZ, EX = (np.repeat(v[:, None], len(gg), axis=1) for v in (zen, azi, ext))
        G = np.repeat(gg[None, :], len(zen), axis=0)
        e = pvlib.irradiance.erbs(G.ravel(), Z.ravel(), EX.ravel())
        poa = pvlib.irradiance.get_total_irradiance(m["plane_tilt"], m["plane_az"], Z.ravel(), AZ.ravel(),
                                                    np.asarray(e["dni"]), G.ravel(), np.asarray(e["dhi"]),
                                                    dni_extra=EX.ravel(), model="haydavies", albedo=PM.ALBEDO)["poa_global"]
        poa = np.clip(np.nan_to_num(np.asarray(poa, float)), 0, None)
        poa = np.where(G.ravel() > 0, poa, 0.0)
        pv = PM.pv_available(poa, m).reshape(G.shape)
        # exactness check against the map's own function on sampled (step, GHI) points
        gd = g[day]
        si = rng.integers(0, len(gd), 200)
        gi = rng.integers(0, len(gg), 200)
        ref = PM.pv_available_from_ghi(gg[gi], gd[si], m)
        assert np.array_equal(ref, pv[si, gi]), float(np.max(np.abs(ref - pv[si, gi])))
        checked += 200
        dpv = np.diff(pv, axis=1)
        bad = (dpv < -1e-9).any(axis=1)
        mn = float(dpv.min())
        if mn < worst:
            worst, worst_at = mn, str(gd[int(np.argmin(dpv.min(axis=1)))])
        per_year[y] = dict(daytime_steps=int(day.sum()), steps_with_fall=int(bad.sum()))
        n_day += int(day.sum())
        n_bad += int(bad.sum())
    rec = dict(item="D-129 step 2 item 11",
               scope="S1M map, every daytime step of 2019-2024, GHI grid 0-1500 W/m2 step 5, no temperature input in the map",
               exactness_points_checked=checked, daytime_steps=n_day, steps_with_fall=n_bad, share=n_bad / n_day,
               largest_fall_kW=worst, largest_fall_at=worst_at, per_year=per_year)
    (OUT / "pav_monotonicity.json").write_text(json.dumps(rec, indent=1), encoding="utf-8")
    print(json.dumps(rec, indent=1))


def items_sim():
    """Aggregate the re-simulation (ce1_wp67_sim.py, every fold's checks must pass) into items 4, 5, 7 and 13."""
    import pandas as pd
    sim = R / "WP67" / "sim"
    for y in FOLDS:
        c = json.loads((sim / f"f{y}_checks.json").read_text(encoding="utf-8"))
        assert c["ok"] and c["n"] == 203, (y, c["n"])
    C = pd.concat([pd.read_csv(sim / f"f{y}_cases.csv") for y in FOLDS], ignore_index=True)
    E = pd.concat([pd.read_csv(sim / f"f{y}_events.csv") for y in FOLDS], ignore_index=True)
    N = pd.concat([pd.read_csv(sim / f"f{y}_n1.csv") for y in FOLDS], ignore_index=True)
    assert len(C) == 6 * 2 * 8 * 8 and len(N) == 6 * 2 * 8 * 7
    C["meets"] = C.ue_share <= TGT + EPS
    C["meets_complete"] = C.ue_share_complete <= TGT + EPS
    C["saving_pct"] = 100 * (1 - C.fuel_L / C.fuel_handbook)
    N["meets"] = N.ue_share <= TGT + EPS
    N["saving_pct"] = 100 * (1 - N.fuel_L / N.fuel_handbook)
    base = list(csv.DictReader(open(R / "WP63" / "cases.csv", encoding="utf-8")))
    LIT = LT.Literal(OUT, "ce1_wp67.py items_sim")
    out4, out5, out7, out13 = [], [], [], []
    for sel, lab in RULES:
        for si, (sname, pols) in enumerate(SETS.items()):
            g = C[(C.selection == sel) & C.policy.isin(pols)]
            assert len(g) == 12 * len(pols) and LIT(f"sim_cases_met_{lab}_{sname}", int(g.meets.sum()), PRINTED_MET[lab][si]), \
                (lab, sname, int(g.meets.sum()))
            e = E[(E.selection == sel) & E.policy.isin(pols)]
            e_share = e.peak_kW / e.online_kW
            out4.append(dict(rule=lab, set=sname, cases=len(g), events_per_year_median=float(g.events.median()),
                             events_per_year_max=int(g.events.max()), events_total=len(e),
                             largest_event_kW=float(e.peak_kW.max()) if len(e) else 0.0,
                             largest_event_share_online=float(e_share.max()) if len(e) else 0.0,
                             median_event_peak_kW=float(e.peak_kW.median()) if len(e) else 0.0,
                             events_above_largest_unit=int((e.peak_kW > e.largest_online_kW).sum())))
            d = int(g.meets_complete.sum()) - int(g.meets.sum())
            out5.append(dict(rule=lab, set=sname, met=int(g.meets.sum()), met_complete_days=int(g.meets_complete.sum()),
                             change=d, reading_main_text=bool(abs(d) > 2),
                             median_saving=float(g.saving_pct.median()), median_saving_complete=float(g.saving_complete_pct.median())))
            out13.append(dict(rule=lab, set=sname, met_covered=int((g.ue_share_covered <= TGT + EPS).sum()),
                              median_saving_covered=float(g.saving_covered_pct.median()), median_saving=float(g.saving_pct.median()),
                              median_ue_share_covered=float(g.ue_share_covered.median())))
            if sel != "AW_clip":
                for scen in ("cont", "contB"):
                    pass
                n = N[(N.selection == sel) & N.policy.isin(pols)]
                assert len(n) == 12 * len(pols)
                out7.append(dict(rule=lab, set=sname, cases=len(n), met=int(n.meets.sum()), median_saving=float(n.saving_pct.median()),
                                 min_saving=float(n.saving_pct.min()), events_per_year_median=float(n.events.median()),
                                 largest_event_kW=float(n.event_peak_max_kW.max()), starts_median=float(n.starts.median()),
                                 pv_binds_share_median=float(n.pv_binds_share.median()), pv_binds_share_max=float(n.pv_binds_share.max())))
    ev63 = {(r["protocol"], r["policy"]): r for r in csv.DictReader(open(R / "WP63" / "events.csv", encoding="utf-8"))}
    for proto, sel in (("A-R", "A_R"), ("A-D", "A_D")):
        for sname in SETS:
            mine = [o for o in out4 if o["rule"] == proto and o["set"] == sname][0]
            assert abs(mine["events_per_year_median"] - float(ev63[(proto, f"set:{sname}")]["events_per_year_median"])) < 1e-9, (proto, sname)
            assert abs(mine["largest_event_share_online"] - float(ev63[(proto, f"set:{sname}")]["peak_share_online_max"])) < 1e-9
    for name, rows in (("events_per_rule", out4), ("complete_days_rules", out5), ("n1_rules", out7), ("covered_steps", out13)):
        _write(rows, name + ".csv")
    rec = dict(item="D-129 step 2 items 4, 5, 7, 13 from ce1_wp67_sim.py (1,218 checks pass)",
               scope="pop=Table 3 rules x {conformal 4, quantile 3}, h=15 min, shape=both, folds=2019-2024; item 5 complete days "
                     "(S63 item 7 set); item 7 online reserve for the largest unit (S56 cont, contB), N-1 handbook as reference; "
                     "item 13 steps with a forecast row only",
               item4={f"{o['rule']}|{o['set']}": dict(ev_med=o["events_per_year_median"], ev_max=o["events_per_year_max"],
                                                    peak_kW=round(o["largest_event_kW"], 1), share=round(o["largest_event_share_online"], 3))
                      for o in out4},
               item5={f"{o['rule']}|{o['set']}": dict(met=o["met"], met_cd=o["met_complete_days"], main=o["reading_main_text"]) for o in out5},
               item5_main_text=[f"{o['rule']}|{o['set']}" for o in out5 if o["reading_main_text"]],
               item7={f"{o['rule']}|{o['set']}": dict(met=o["met"], sav=round(o["median_saving"], 4), ev=o["events_per_year_median"],
                                                    binds=round(o["pv_binds_share_median"], 4)) for o in out7},
               item13={f"{o['rule']}|{o['set']}": dict(met_cov=o["met_covered"], sav_cov=round(o["median_saving_covered"], 4),
                                                     sav=round(o["median_saving"], 4)) for o in out13})
    (OUT / "sim_items.json").write_text(json.dumps(rec, indent=1), encoding="utf-8")
    print(json.dumps(rec, indent=1))


MONEY_SOURCES = {
    "vcr": dict(source="AER, Values of Customer Reliability 2025 Annual Adjustment Summary (Dec 2025), p. 1-2, "
                       "https://www.aer.gov.au/industry/registers/resources/reviews/values-customer-reliability-2024/2025-vcr-annual-adjustment",
                read="2026-10-01", effective="2025-12-18", unit="AUD2025 per kWh",
                nt_residential=31.86, commercial=35.70, commercial_large=32.06, commercial_small_medium=54.64, nem_residential=43.06),
    "diesel": dict(source="Australian Institute of Petroleum, terminal gate prices, https://www.aip.com.au/pricing/terminal-gate-prices",
                   read="2026-10-01", date="2026-09-30", darwin_diesel_c_per_L_incl_gst=273.0,
                   credit_source="ATO, fuel tax credit rates 1 July 2026 to 30 June 2027, all other business uses, from 3 August 2026",
                   fuel_tax_credit_c_per_L=53.7, registered_price_AUD_per_L=1.68, registered_range=(1.20, 2.20),
                   note="net = TGP / 1.1 - credit; Darwin terminal, freight to Yulara not included"),
    "tender": dict(source="Territory Generation tender YUO061-19A 'Yulara Hybrid Power Station Project', published 16 January 2020, "
                          "closing 28 January 2020, scope 'supply and installation of a new 4 MW / 2 MWh BESS', "
                          "https://www.australiantenders.com.au/tenders/404759/yulara-hybrid-power-station-project/",
                   read="2026-10-01"),
    "start_sync": dict(source="Caterpillar C32 genset datasheet TSS-EM0745-01 (09/01/2016) lists NFPA 110 among its codes but gives "
                              "no start or load-acceptance time", read="2026-10-01", status="NOT READ AT A PRIMARY SOURCE, carried"),
}


def item12():
    import pandas as pd
    ms = MONEY_SOURCES
    net = ms["diesel"]["darwin_diesel_c_per_L_incl_gst"] / 1.1 - ms["diesel"]["fuel_tax_credit_c_per_L"]
    assert abs(net - 194.48) < 0.01, net
    lo_p, hi_p = ms["diesel"]["registered_price_AUD_per_L"], net / 100
    vcr_lo, vcr_hi = ms["vcr"]["nt_residential"], ms["vcr"]["commercial"]
    C = pd.concat([pd.read_csv(R / "WP67" / "sim" / f"f{y}_cases.csv") for y in FOLDS], ignore_index=True)
    C["saving_kL"] = (C.fuel_handbook - C.fuel_L) / 1000
    out = []
    for sel, lab in RULES:
        g = C[(C.selection == sel) & C.policy.isin(QP)]
        kl = float(g.saving_kL.median())
        ue = float(g.ue_kWh.median())
        out.append(dict(rule=lab, set="quantile", saving_kL_median=kl, saving_AUD_lo=kl * 1000 * lo_p, saving_AUD_hi=kl * 1000 * hi_p,
                        ue_kWh_median=ue, ue_cost_AUD_lo=ue * vcr_lo, ue_cost_AUD_hi=ue * vcr_hi))
    _write(out, "money_frame.csv")
    xh = [o for o in out if o["rule"] == "XH"][0]
    assert LT.Literal(OUT, "ce1_wp67.py item12")("XH_quantile_saving_kL_median", xh["saving_kL_median"], 11.212420506531839,
                                                 equal=abs(xh["saving_kL_median"] - 11.212420506531839) < 0.05), \
        xh["saving_kL_median"]
    rec = dict(item="D-129 step 2 item 12", sources=ms, net_price_c_per_L=round(net, 2),
               price_range_AUD_per_L=[lo_p, hi_p], vcr_range_AUD_per_kWh=[vcr_lo, vcr_hi],
               scope="pop=quantile set (raw q, CQR, M-CQR), h=15 min, shape=both, folds=2019-2024, median case; fuel price from "
                     "the registered 1.68 AUD/L to the Darwin terminal price net of GST and credit on 30 Sep 2026",
               rows={o["rule"]: dict(kL=round(o["saving_kL_median"], 2), aud=[round(o["saving_AUD_lo"]), round(o["saving_AUD_hi"])],
                                     ue_kWh=round(o["ue_kWh_median"], 1), ue_aud=[round(o["ue_cost_AUD_lo"]), round(o["ue_cost_AUD_hi"])])
                     for o in out})
    (OUT / "money_frame.json").write_text(json.dumps(rec, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in rec.items() if k != "sources"}, indent=1))


def _write(rows, name):
    with open(OUT / name, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    cmds = {"c1": c1, "item1": item1, "item3": item3, "item2": item2, "item6": item6, "item10": item10, "item11": item11, "items_sim": items_sim, "item12": item12}
    if len(sys.argv) < 2 or sys.argv[1] not in cmds:
        print(__doc__)
        sys.exit(2)
    cmds[sys.argv[1]]()
