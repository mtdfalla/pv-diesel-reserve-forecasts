#!/usr/bin/env python3
"""Readings and descriptive items of stage S69 (D-136), from 04_results/ce1/S69.

Usage (from the repository root unless the path starts at ce1/, then from 03_code):
    python3 03_code/ce1/ce1_s69_read.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "utils"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import ce1_literal as LT  # noqa: E402  (D-164 phase-aware literal checks)

R = Path("04_results/ce1")
S = R / "S69"
TGT = 2e-5
EQUIV = 0.1
CP = ["icp", "mondrian", "cqr", "mondrian_cqr"]
QP = ["raw_q", "cqr", "mondrian_cqr"]
SETS = {"conformal": CP, "quantile": QP}
RULES = ["A-R", "XF", "PF", "H2", "XH", "PH", "A-D"]
FOLDS = [2019, 2020, 2021, 2022, 2023, 2024]
SEASON = {12: "summer", 1: "summer", 2: "summer", 3: "autumn", 4: "autumn", 5: "autumn",
          6: "winter", 7: "winter", 8: "winter", 9: "spring", 10: "spring", 11: "spring"}


def med(x):
    return float(np.median(np.asarray(x, float))) if len(x) else float("nan")


def rule_table(c, constr, pols_sets):
    out = []
    for rule in RULES:
        for sname, pols in pols_sets.items():
            x = c[(c.construction == constr) & (c.rule == rule) & c.policy.isin(pols)]
            yrs = sum(bool(x[x.fold == y].meets.all()) for y in FOLDS) if len(x) else 0
            out.append(dict(construction=constr, rule=rule, set=sname, cases=len(x), met=int(x.meets.sum()), years_met=yrs,
                            median_saving_pct=med(x.saving_pct), min_saving_pct=float(x.saving_pct.min()) if len(x) else np.nan,
                            events_per_year_median=med(x.events), largest_event_kW=float(x.event_peak_max_kW.max()) if len(x) else np.nan,
                            starts_median=med(x.starts), infeasible_steps_max=int(x.infeasible_steps.max()) if len(x) else 0,
                            ue_share_median=med(x.ue_share)))
    return out


def main():
    c = pd.read_csv(S / "cases.csv", float_precision="round_trip")
    an = pd.read_csv(S / "anchors.csv", float_precision="round_trip")
    rec = {}
    # ---------------------------------------------------------------- consistency of the record
    n_plain = 10 if (c.policy == "spf").any() else 9  # D-162: the v2.1 S69 adds the SPF rows to plain
    assert n_plain == 10 or LT.registered(), "a v2.1 S69 record must carry the SPF rows (D-162)"
    assert len(c) == 6 * (2 * n_plain * 7 + 4 * 8 * 7), len(c)  # plain 9 policies incl. pfr (10 with spf); s56 and n1a 8
    assert c[c.construction == "plain"].groupby(["fold", "shape"]).size().eq(7 * n_plain).all()
    # ---------------------------------------------------------------- rule tables per construction
    tabs = []
    for constr in ("plain", "s56", "n1a"):
        tabs += rule_table(c, constr, SETS)
    tabs += rule_table(c, "plain", {"pfr": ["pfr"]})
    if n_plain == 10:
        tabs += rule_table(c, "plain", {"spf": ["spf"]})  # D-162 rows
    for q in QP:
        tabs += rule_table(c, "plain", {f"policy_{q}": [q]})
    T = pd.DataFrame(tabs)
    T.to_csv(S / "rules_by_construction.csv", index=False)
    printed = {"A-R": (25, 21), "XF": (34, 21), "PF": (26, 22), "H2": (42, 33), "XH": (48, 36), "PH": (47, 36), "A-D": (48, 36)}
    LIT = LT.Literal(S, "ce1_s69_read.py")
    same = True
    for rule, (mc, mq) in printed.items():
        a = T[(T.construction == "plain") & (T.rule == rule)]
        got = (int(a[a.set == "conformal"].met.iloc[0]), int(a[a.set == "quantile"].met.iloc[0]))
        same &= got == (mc, mq)
        assert LIT(f"table3_plain_met_{rule}", got, (mc, mq)), rule
    rec["table3_plain_reproduced"] = bool(same)

    # ---------------------------------------------------------------- N1A-1
    xq = c[(c.construction == "n1a") & (c.rule == "XH") & c.policy.isin(QP)]
    assert len(xq) == 36
    m1 = med(xq.saving_pct)
    fires = m1 >= EQUIV
    rec["N1A-1"] = dict(cases=36, median_saving_pct_vs_n1a_handbook=m1, threshold=EQUIV, fires=bool(fires),
                        min_saving_pct=float(xq.saving_pct.min()), max_saving_pct=float(xq.saving_pct.max()),
                        cases_saving_ge_0_1=int((xq.saving_pct >= EQUIV).sum()), cases_negative=int((xq.saving_pct < 0).sum()),
                        consequence=("article: the forecast keeps a fuel role under an additive "
                                     "N-1 reserve; 'rarely sets the reserve' scoped to the two-term construction") if fires else
                        "the sentence stays with the construction named and the N1A result beside it")
    # same statistic for the other constructions, for context
    for constr in ("plain", "s56"):
        x = c[(c.construction == constr) & (c.rule == "XH") & c.policy.isin(QP)]
        rec["N1A-1"][f"context_median_saving_{constr}"] = med(x.saving_pct)
    # ---------------------------------------------------------------- N1A-2
    n1 = T[T.construction == "n1a"]
    rec["N1A-2"] = dict(met={f"{r.rule}|{r.set}": f"{r.met} of {r.cases}" for r in n1.itertuples()})
    xh_q = n1[(n1.rule == "XH") & (n1.set == "quantile")].iloc[0]
    rec["N1A-2"]["xh_quantile_met"] = int(xh_q.met)
    rec["N1A-2"]["shortfall_finding"] = bool(xh_q.met < 36)
    # anchors in kL/yr
    A_ = an.pivot_table(index=["fold", "shape"], columns="construction", values="handbook_fuel_L")
    rec["anchors_kL_per_yr"] = {f"{k[0]}|{k[1]}": {cn: round(float(v) / 1000, 3) for cn, v in row.items()} for k, row in A_.iterrows()}
    rec["anchors_median_kL"] = {cn: float(np.median(A_[cn]) / 1000) for cn in A_.columns}
    rec["anchors_median_increase_pct"] = {cn: float(np.median(100 * (A_[cn] / A_["plain"] - 1))) for cn in ("s56", "n1a")}
    rec["n1a_handbook_infeasible_steps_max"] = int(an[an.construction == "n1a"].handbook_infeasible_steps.max())

    # ---------------------------------------------------------------- PFR-1
    p = c[(c.construction == "plain") & (c.rule == "XH")]
    pf = p[p.policy == "pfr"]
    assert len(pf) == 12
    per = []
    for q in QP:
        x = p[p.policy == q]
        assert len(x) == 12
        per.append(dict(policy=q, met=int(x.meets.sum()), median_saving_pct=med(x.saving_pct)))
    best = sorted(per, key=lambda d: (-d["met"], -d["median_saving_pct"]))[0]
    pfr_met, pfr_med = int(pf.meets.sum()), med(pf.saving_pct)
    match = pfr_met >= best["met"] and abs(pfr_med - best["median_saving_pct"]) <= EQUIV
    rec["PFR-1"] = dict(pfr_met=pfr_met, pfr_median_saving_pct=pfr_med, pfr_levels_f=sorted(set((1 - pf.param).round(4))),
                        quantile_policies=per, best=best, difference_pct=pfr_med - best["median_saving_pct"],
                        fires=bool(match),
                        consequence=("abstract and Conclusions: a margin on a persistence rule matches the probabilistic forecast "
                                     "under XH") if match else "3.4 states the quantile policies' lead over the persistence fraction with its count")
    rec["PFR_all_rules"] = {r.rule: dict(met=f"{r.met} of {r.cases}", median_saving_pct=r.median_saving_pct)
                            for r in T[(T.construction == "plain") & (T.set == "pfr")].itertuples()}
    fx = c[(c.construction == "plain") & (c.rule == "XH") & (c.policy == "frac_pv")]
    rec["PFR_context_fixed_fraction_XH"] = dict(met=int(fx.meets.sum()), median_saving_pct=med(fx.saving_pct))
    (S / "readings.json").write_text(json.dumps(rec, indent=1, default=float), encoding="utf-8")
    print(json.dumps({k: rec[k] for k in ("N1A-1", "N1A-2", "PFR-1", "anchors_median_kL", "anchors_median_increase_pct")}, indent=1, default=float))
    desc(c)


def desc(c):
    import ce1_adaptive as A
    import ce1_load as LM
    import ce1_wp57 as W57
    tot = pd.read_parquet(S / "totals.parquet")
    cases63 = pd.read_csv(R / "WP63" / "cases.csv", float_precision="round_trip")
    # ---- D1: static cross-fitted rule at 0.25, 0.5, 0.75, 1.0 of the allowance
    rows = []
    for y in FOLDS:
        for shape, nm in (("A", "main"), ("B", "shapeB")):
            t = tot[(tot.fold == y) & (tot.construction == "plain") & (tot.scenario == nm)]
            T2, TX, TT = (t[t.set == s].reset_index(drop=True) for s in ("H2", "H1x", "test"))
            hb = float(TT[TT.policy == "handbook"].fuel_L.iloc[0])
            for pol in sorted(set(CP + QP + ["frac_pv", "pfr"])):
                X = W57.pooled([T2[T2.policy == pol].reset_index(drop=True), TX[TX.policy == pol].reset_index(drop=True)])
                for frac in (0.25, 0.5, 0.75, 1.0):
                    p_, _ = W57.select(X, pol, frac * TGT)
                    r = W57.at(TT, pol, p_)
                    if frac in (0.5, 1.0) and pol != "pfr":
                        sel = "b_xfit_0.001" if frac == 0.5 else "b_xfit_0.002"
                        ref = cases63[(cases63.fold == y) & (cases63.scenario == nm) & (cases63.policy == pol) & (cases63.selection == sel)]
                        assert abs(float(ref.param.iloc[0]) - p_) < 1e-9, (y, nm, pol, frac)
                    rows.append(dict(fold=y, shape=shape, policy=pol, allowance=frac, param=p_, ue_share=r["ue_share"],
                                     meets=bool(r["ue_share"] <= TGT + 1e-15), saving_pct=100 * (1 - r["fuel_L"] / hb)))
    D1 = pd.DataFrame(rows)
    D1.to_csv(S / "desc_xfit_allowance_cases.csv", index=False)
    summ = []
    for frac in (0.25, 0.5, 0.75, 1.0):
        for sname, pols in SETS.items():
            x = D1[(D1.allowance == frac) & D1.policy.isin(pols)]
            summ.append(dict(allowance=frac, set=sname, cases=len(x), met=int(x.meets.sum()),
                             years_met=sum(bool(x[x.fold == y].meets.all()) for y in FOLDS), median_saving_pct=med(x.saving_pct)))
    pd.DataFrame(summ).to_csv(S / "desc_xfit_allowance.csv", index=False)
    # ---- D2: A-R UE by austral season of the test year
    rows = []
    for y in FOLDS:
        z = np.load(R / "S6E" / f"f{y}_h3_daily.npz")
        rmap, amap = A.record_rows(z)
        days = pd.date_range(f"{y}-01-01", f"{y}-12-31", freq="D")
        seas = np.array([SEASON[d.month] for d in days])
        for nm, shape in (("main", "A"), ("shapeB", "B")):
            ue_all, ld_all = z[f"{nm}__ue_kWh"], z[f"{nm}__load_kWh"]
            assert ue_all.shape[1] == len(days)
            for pol in sorted(set(CP + QP)):
                p_ = float(cases63[(cases63.fold == y) & (cases63.scenario == nm) & (cases63.policy == pol) & (cases63.selection == "A_R")].param.iloc[0])
                i = rmap[(pol, round(p_, 6))]
                for s_ in ("summer", "autumn", "winter", "spring"):
                    mk = seas == s_
                    rows.append(dict(fold=y, shape=shape, policy=pol, season=s_, ue_kWh=float(ue_all[i][mk].sum()),
                                     load_kWh=float(ld_all[i][mk].sum())))
    D2 = pd.DataFrame(rows)
    D2.to_csv(S / "desc_ar_season_cases.csv", index=False)
    summ = []
    for sname, pols in SETS.items():
        x = D2[D2.policy.isin(pols)]
        tot_ue = x.ue_kWh.sum()
        for s_ in ("summer", "autumn", "winter", "spring"):
            xs = x[x.season == s_]
            per_case = xs.ue_kWh / xs.load_kWh
            summ.append(dict(set=sname, season=s_, share_of_ar_ue_energy=float(xs.ue_kWh.sum() / tot_ue),
                             pooled_ue_share=float(xs.ue_kWh.sum() / xs.load_kWh.sum()),
                             cases_over_target_in_season=int((per_case > TGT + 1e-15).sum()), cases=len(xs)))
    pd.DataFrame(summ).to_csv(S / "desc_ar_season.csv", index=False)
    rows = []
    for rule in RULES:
        for sname, pols in SETS.items():
            x = c[(c.construction == "plain") & (c.rule == rule) & c.policy.isin(pols)]
            rows.append(dict(rule=rule, set=sname, cases=len(x), met_covered=int((x.ue_share_covered <= TGT + 1e-15).sum()),
                             median_saving_covered=med(x.saving_covered_pct), median_saving=med(x.saving_pct),
                             median_ue_share_covered=med(x.ue_share_covered)))
    D3 = pd.DataFrame(rows)
    ref = pd.read_csv(R / "WP67" / "covered_steps.csv")
    mm = D3.merge(ref, on=["rule", "set"], suffixes=("", "_wp67"))
    assert len(mm) == len(D3) and (mm.met_covered == mm.met_covered_wp67).all() and np.allclose(mm.median_saving_covered, mm.median_saving_covered_wp67)
    D3.to_csv(S / "desc_covered_every_rule.csv", index=False)
    pd.DataFrame([t for t in (rule_table(c, "s56", SETS) + rule_table(c, "n1a", SETS))]).to_csv(S / "desc_n1_rules.csv", index=False)
    # ---- D6, D7, D8 summaries
    mo = pd.read_csv(S / "desc_mondrian.csv")
    mo.groupby(["policy", "group"]).agg(n_group_min=("n_group", "min"), n_group_max=("n_group", "max"),
                                        tail_min=("tail_count", "min"), tail_max=("tail_count", "max"),
                                        own_scores_all=("uses_own_scores", "all")).reset_index().to_csv(S / "desc_mondrian_summary.csv", index=False)
    w = pd.read_csv(S / "desc_winter.csv")
    w[w.split == "test"][["fold", "year", "issue_times", "winter_issue_times", "winter_transitional", "transitional"]].to_csv(S / "desc_winter_test.csv", index=False)
    fb = pd.read_csv(S / "desc_fallback_pv.csv")
    fb["fallback_pv_share"] = fb.fallback_pv_kWh / fb.daytime_pv_kWh
    fb.to_csv(S / "desc_fallback_pv.csv", index=False)
    # ---- D4: the largest event of the conformal and quantile policies, its step, gap and carried input
    import ce1_folds as FC
    import ce1_s6 as S6
    E = pd.read_csv(S / "events.csv")
    E = E[(E.construction == "plain") & E.policy.isin(sorted(set(CP + QP)))]
    big = E.sort_values("peak_kW", ascending=False).iloc[0]
    same = E[(E.fold == big.fold) & (E["shape"] == big["shape"]) & (E.start == big.start)]
    cases_fs = c[(c.construction == "plain") & (c.fold == big.fold) & (c["shape"] == big["shape"]) & c.policy.isin(sorted(set(CP + QP)))]
    geo, m, unit96, scales, _ = S6.load_inputs(R / "S1", R / "S1M", R / "S1L")
    cache = FC.load_cache_ext(R / "S2" / f"yulara_f{int(big.fold)}_h3.parquet", R / "S2X" / f"yulara_f{int(big.fold)}_h3.parquet")
    st0 = pd.Timestamp(big.start)
    win = pd.date_range(st0 - pd.Timedelta(minutes=30), st0 + pd.Timedelta(minutes=5 * int(big.steps) + 15), freq="5min")
    issue = [t - pd.Timedelta(minutes=15) for t in win]
    rows = []
    for t, it in zip(win, issue):
        rows.append(dict(t=str(t), ghi=None if not np.isfinite(geo.ghi.get(t, np.nan)) else float(geo.ghi[t]),
                         forecast_row=bool(it in cache.index), issue=str(it),
                         in_imp=float(cache.in_imp[it]) if it in cache.index else None,
                         regime_at_issue=str(cache.g[it]) if it in cache.index else None,
                         point_forecast=float(cache.p[it]) if it in cache.index else None))
    lev = dict(fold=int(big.fold), shape=str(big["shape"]), start=str(st0), steps=int(big.steps), peak_kW=float(big.peak_kW),
               online_kW=float(big.online_kW), largest_online_kW=float(big.largest_online_kW),
               cases_with_this_event=int(same[["policy", "selection"]].drop_duplicates().shape[0]),
               cases_in_fold_shape=int(len(cases_fs)), rules=sorted(set(same.selection)),
               policies=sorted(set(same.policy)), window=rows)
    (S / "desc_largest_event_cp_qp.json").write_text(json.dumps(lev, indent=1), encoding="utf-8")
    print("descriptive tables written")


if __name__ == "__main__" and "--b" not in sys.argv:
    main()


def read_b():
    """D-137 reading PFR-N1A, from 04_results/ce1/S69b (n1a rows gated on S69) and the S69 n1a totals."""
    Sb = R / "S69b"
    c = pd.read_csv(Sb / "cases.csv", float_precision="round_trip")
    an = pd.read_csv(Sb / "anchors.csv", float_precision="round_trip")
    p = c[(c.construction == "n1a") & (c.rule == "XH")]
    pf = p[p.policy == "pfr"]
    assert len(pf) == 12
    per = []
    for q in QP:
        x = p[p.policy == q]
        assert len(x) == 12
        per.append(dict(policy=q, met=int(x.meets.sum()), median_saving_pct=med(x.saving_pct)))
    best = sorted(per, key=lambda d: (-d["met"], -d["median_saving_pct"]))[0]
    pfr_met, pfr_med = int(pf.meets.sum()), med(pf.saving_pct)
    match = pfr_met >= best["met"] and abs(pfr_med - best["median_saving_pct"]) <= EQUIV
    # deterministic anchor under N1A (point forecast counted in full), no reading rule
    tot = pd.read_parquet(Sb / "totals.parquet")
    t = tot[(tot.set == "test") & tot.policy.isin(["handbook", "deterministic"])]
    dd = []
    for (y, nm), g in t.groupby(["fold", "scenario"]):
        hb = float(g[g.policy == "handbook"].fuel_L.iloc[0]); de = g[g.policy == "deterministic"].iloc[0]
        dd.append(dict(fold=y, scenario=nm, saving_pct=100 * (1 - float(de.fuel_L) / hb), ue_share=float(de.ue_share),
                       meets=bool(float(de.ue_share) <= TGT + 1e-15)))
    dd = pd.DataFrame(dd)
    dd.to_csv(Sb / "desc_point_anchor.csv", index=False)
    rec = dict(reading="PFR-N1A (D-137)", pfr_met=pfr_met, pfr_median_saving_pct=pfr_med,
               pfr_min_saving_pct=float(pf.saving_pct.min()), pfr_max_saving_pct=float(pf.saving_pct.max()),
               pfr_levels_f=sorted(set((1 - pf.param).round(4))), quantile_policies=per, best=best,
               difference_pct=pfr_med - best["median_saving_pct"], fires=bool(match),
               consequence=("the N1A sentence says this fuel role does not need a probabilistic forecast (a persistence "
                            "estimate of PV does as well)") if match else "the N1A sentence states the quantile policies' lead under N1A with its count",
               deterministic_anchor=dict(met=int(dd.meets.sum()), cases=len(dd), median_saving_pct=med(dd.saving_pct),
                                         min_saving_pct=float(dd.saving_pct.min())),
               pfr_all_rules={r: dict(met=int(c[(c.rule == r) & (c.policy == "pfr")].meets.sum()),
                                      median_saving_pct=med(c[(c.rule == r) & (c.policy == "pfr")].saving_pct)) for r in RULES})
    (Sb / "readings.json").write_text(json.dumps(rec, indent=1, default=float), encoding="utf-8")
    print(json.dumps(rec, indent=1, default=float))


if __name__ == "__main__" and "--b" in sys.argv:
    read_b()
