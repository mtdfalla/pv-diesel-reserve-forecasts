"""Selection-rule tables, load-scale comparison and issue-time battery credit, read from existing
outputs. Writes 04_results/ce1/WP61/.

Usage (from the repository root unless the path starts at ce1/, then from 03_code):
    python3 03_code/ce1/ce1_wp61read.py --root .
"""
import argparse, hashlib, json, sys
from pathlib import Path

import numpy as np
import pandas as pd

import ce1_literal as LT

FOLDS = [2019, 2020, 2021, 2022, 2023, 2024]
CP = ["icp", "mondrian", "cqr", "mondrian_cqr"]
QP = ["raw_q", "cqr", "mondrian_cqr"]
TGT = 2e-5
SEL_ORDER = [("A_R", "Year ahead (A-R), full allowance"),
             ("b_xfit_0.002", "Static, cross-fitted on the calibration year, full allowance"),
             ("a_half_H2", "Year ahead, half allowance"),
             ("b_xfit_0.001", "Static, cross-fitted on the calibration year, half allowance"),
             ("c_prior_0.001", "Static, all earlier test years, half allowance"),
             ("A_D", "Daily controller (A-D), half allowance"),
             ("AW_1.0", "Daily controller, latent level clipped at the limit")]


def md5(p):
    return hashlib.md5(Path(p).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(Path(__file__).resolve().parents[2]))
    a = ap.parse_args()
    R = Path(a.root) / "04_results" / "ce1"
    out = R / "WP61"
    out.mkdir(parents=True, exist_ok=True)
    checks, inputs, rd = {}, {}, {}
    LIT = LT.Literal(out, "ce1_wp61read.py")

    def check(name, ok, detail=""):
        checks[name] = dict(ok=bool(ok), detail=str(detail))
        print(("ok    " if ok else "FAIL  ") + name + ("" if ok else "  " + str(detail)))

    def inp(rel):
        inputs[rel] = md5(R / rel)
        return R / rel

    # ---------------------------------------------------------------- 1. selection table
    c = pd.read_csv(inp("WP57/cases.csv"))
    c = c[c.cp]
    aw = pd.read_csv(inp("AW/cells.csv"), float_precision="round_trip")
    aw = aw[(aw.clip_h == 1.0) & aw.policy.isin(CP)].copy()
    aw["selection"], aw["meets"] = "AW_1.0", aw.ue_share <= TGT
    x = pd.concat([c[["fold", "scenario", "policy", "selection", "meets", "saving_pct"]],
                   aw[["fold", "scenario", "policy", "selection", "meets", "saving_pct"]]], ignore_index=True)
    check("every selection has 48 conformal cases", (x.groupby("selection").size() == 48).all(), x.groupby("selection").size().to_dict())
    # the 'meets' flag of WP57 is recomputed here from the UE share (target 0.002%), not trusted
    cc = c.assign(m2=c.ue_share <= TGT)
    check("WP57 meets flag equals ue_share <= 0.002%", (cc.m2 == cc.meets).all())
    rows = []
    for key, label in SEL_ORDER:
        g = x[x.selection == key]
        r = dict(selection=key, label=label, met=int(g.meets.sum()), n=len(g))
        for yr in FOLDS:
            r[f"met_{yr}"] = int(g[g.fold == yr].meets.sum())
        for p in CP:
            r[f"median_{p}"] = float(g[g.policy == p].saving_pct.median())
        r["median_all"] = float(g.saving_pct.median())
        rows.append(r)
    T = pd.DataFrame(rows)
    T.to_csv(out / "selection.csv", index=False)
    t = T.set_index("selection")
    check("A-R, A-D and cross-fitted half met counts (v2.0.0: 25, 48 and 48 of 48, as recorded)",
          LIT("selection_met_AR_AD_xfit_half", (int(t.loc["A_R", "met"]), int(t.loc["A_D", "met"]), int(t.loc["b_xfit_0.001", "met"])),
              (25, 48, 48)))
    # paired: static cross-fitted half allowance against A-D, per case
    p1 = c[c.selection == "b_xfit_0.001"].set_index(["fold", "scenario", "policy"]).saving_pct
    p2 = c[c.selection == "A_D"].set_index(["fold", "scenario", "policy"]).saving_pct
    d = (p1 - p2.reindex(p1.index))
    rd["xfit_vs_AD_static_more"] = int((d > 0).sum())
    rd["xfit_vs_AD_median_diff_by_policy"] = d.groupby(level="policy").median().round(4).to_dict()
    yb = {k: [int(t.loc[k, f"met_{yr}"]) for yr in FOLDS] for k in ("A_D", "b_xfit_0.001")}
    rd["xfit_vs_AD_ties_by_year_met"] = ("8 of 8 each in every year for both" if all(m == 8 for v_ in yb.values() for m in v_) else yb)
    # fold 2019 of the all-earlier-years rule falls back to the calibration-year half (no earlier test year)
    s19 = c[(c.fold == 2019) & c.selection.isin(["a_half_H2", "c_prior_0.001"])].pivot_table(
        index=["scenario", "policy"], columns="selection", values="param")
    check("fold 2019: the all-earlier-years rule equals the calibration-year half rule", (s19["a_half_H2"] == s19["c_prior_0.001"]).all())

    # ---------------------------------------------------------------- 2. load scales, deterministic, oracle
    ls = []
    for yr in FOLDS:
        S = json.load(open(inp(f"S6E/f{yr}_h3_select.json")))["selections"]
        for sc in ("main", "shapeB", "peak6MW", "spill"):
            anc = S[sc]["anchors"]
            hb = anc["handbook"]["fuel_L"]
            for pol, v in S[sc]["policies"].items():
                t_ = v["A_R"]["test"]
                ls.append(dict(fold=yr, scenario=sc, policy=pol, level=v["A_R"]["param"], met=t_["ue_share"] <= TGT,
                               saving_pct=100 * (1 - t_["fuel_L"] / hb), ceiling_pct=100 * (1 - anc["oracle"]["fuel_L"] / hb),
                               handbook_ML=hb / 1e6, det_saving_pct=100 * (1 - anc["deterministic"]["fuel_L"] / hb),
                               det_ue_share_pct=100 * anc["deterministic"]["ue_share"], pv_share_load=anc["handbook"]["pv_share_of_load"],
                               curt_share_handbook=anc["handbook"]["curt_share"]))
    L = pd.DataFrame(ls)
    L.to_csv(out / "load_scales.csv", index=False)
    summ = {}
    for sc, g in L.groupby("scenario"):
        cp_, q_ = g[g.policy.isin(CP)], g[g.policy.isin(QP)]
        summ[sc] = dict(cp_met=int(cp_.met.sum()), cp_n=len(cp_), q_median=float(q_.saving_pct.median()),
                        q_min=float(q_.saving_pct.min()), q_max=float(q_.saving_pct.max()),
                        ceil_min=float(g.ceiling_pct.min()), ceil_max=float(g.ceiling_pct.max()),
                        det_sav_min=float(g.det_saving_pct.min()), det_sav_max=float(g.det_saving_pct.max()),
                        det_ue_min_pct=float(g.det_ue_share_pct.min()), det_ue_max_pct=float(g.det_ue_share_pct.max()),
                        pv_share_min=float(g.pv_share_load.min()), pv_share_max=float(g.pv_share_load.max()),
                        curt_min=float(g.curt_share_handbook.min()), curt_max=float(g.curt_share_handbook.max()))
    rd["load_scales"] = summ
    check("main and shapeB A-R conformal met counts (v2.0.0: 14 and 11 of 24)",
          LIT("AR_conformal_met_main_shapeB", (int(summ["main"]["cp_met"]), int(summ["shapeB"]["cp_met"])), (14, 11)))

    # ---------------------------------------------------------------- 3. forecast-alone median and interaction
    w = pd.read_csv(inp("WP52/cells.csv"))
    q = w[(w.h_min == 15) & (w["shape"] == "A") & w.policy.isin(QP)]
    check("WP52 quantile cells at 15 min, shape A: 18", len(q) == 18, len(q))
    rd["decomp_plain_median"] = {k: float(q[k].median()) for k in ("battery_alone_pct", "forecast_alone_pct", "combined_pct", "interaction_pct")}
    rd["decomp_cell_range"] = {k: [float(q[k].min()), float(q[k].max())] for k in ("combined_pct", "interaction_pct")}
    yi = q.groupby("fold").interaction_pct.median()
    rd["interaction_year_median_range"] = [float(yi.min()), float(yi.max())]
    rd["years_AR_quantile_miss_all_no_battery"] = [int(y) for y, g in q.groupby("fold") if not g.target_met_nb.any()]

    d6 = pd.read_csv(inp("S7E/dr6.csv"))
    f6 = d6[(d6.h == 3) & (d6.scenario == "main") & (d6.protocol == "F")]
    r6 = {}
    for (pol, base), g in f6.groupby(["policy", "baseline"]):
        ci = g.d_fuel_L_ci.apply(lambda s: tuple(float(v) for v in s.strip("()").split(",")))
        r6[f"{pol}|{base}"] = dict(median_kL=float(g.d_fuel_L.median() / 1000), below_zero=int(sum(c_[1] < 0 for c_ in ci)))
    rd["frontier_vs_operator_rules"] = r6

    # ---------------------------------------------------------------- 5. WP-56 per cell
    s56 = pd.read_csv(inp("S56/ad_cells.csv"))
    s56 = s56[s56.primary & (s56.scenario != "main_ref")].copy()
    s56["sav"] = 100 * (1 - s56.fuel_L / s56.fuel_handbook)
    q56 = s56[s56.policy.isin(QP)]
    rd["s56_quantile_negative_cells_max_per_scenario"] = int(q56.groupby("scenario").apply(lambda g: int((g.sav < 0).sum())).max())
    rd["s56_quantile_median_min"] = float(q56.groupby("scenario").sav.median().min())
    rd["s56_conformal_met_all"] = bool((s56[s56.policy.isin(CP)].ue_share <= TGT).all())

    # ---------------------------------------------------------------- 6. S61B battery with the credit limited to issue-time PV
    if (R / "S61B" / "ad_cells.csv").exists():
        st = json.load(open(inp("_runs/S61B.json")))
        check("S61B stamp ok", st.get("ok") is True, st)
        brow = []
        for yr in FOLDS:
            S = json.load(open(inp(f"S6E/f{yr}_h3_select.json")))["selections"]
            B = json.load(open(inp(f"S61B/f{yr}_h3_select.json")))
            for shape, base_nb, base_wb, day in (("A", "main", "battery", "battery_day"), ("B", "shapeB", "batteryB", "batteryB_day")):
                F0 = S[base_nb]["anchors"]["handbook"]["fuel_L"]
                for pol in ["handbook"] + QP:
                    if pol == "handbook":
                        nb, wb, dy = F0, S[base_wb]["anchors"]["handbook"]["fuel_L"], B[day]["anchors"]["handbook"]["fuel_L"]
                        m_nb = m_wb = m_dy = True
                        l_nb = l_wb = l_dy = np.nan
                    else:
                        a_nb, a_wb, a_dy = S[base_nb]["policies"][pol]["A_R"], S[base_wb]["policies"][pol]["A_R"], B[day]["policies"][pol]["A_R"]
                        nb, wb, dy = a_nb["test"]["fuel_L"], a_wb["test"]["fuel_L"], a_dy["test"]["fuel_L"]
                        m_nb, m_wb, m_dy = (a["test"]["ue_share"] <= TGT for a in (a_nb, a_wb, a_dy))
                        l_nb, l_wb, l_dy = a_nb["param"], a_wb["param"], a_dy["param"]
                    brow.append(dict(fold=yr, shape=shape, policy=pol, protocol="A-R",
                                     combined_d088=100 * (1 - wb / F0), combined_day=100 * (1 - dy / F0), alone_forecast=100 * (1 - nb / F0),
                                     met_nb=m_nb, met_d088=m_wb, met_day=m_dy, level_nb=l_nb, level_d088=l_wb, level_day=l_dy))
        Bt = pd.DataFrame(brow)
        hbA = Bt[(Bt.policy == "handbook")]
        rd["battery_alone"] = dict(d088_median=float(hbA[hbA["shape"] == "A"].combined_d088.median()),
                                   day_median=float(hbA[hbA["shape"] == "A"].combined_day.median()),
                                   day_min=float(hbA[hbA["shape"] == "A"].combined_day.min()),
                                   day_max=float(hbA[hbA["shape"] == "A"].combined_day.max()),
                                   day_median_B=float(hbA[hbA["shape"] == "B"].combined_day.median()),
                                   d088_median_B=float(hbA[hbA["shape"] == "B"].combined_d088.median()))
        qA = Bt[(Bt.policy.isin(QP)) & (Bt["shape"] == "A")]
        rd["battery_AR_quantile_A"] = dict(combined_d088_median=float(qA.combined_d088.median()), combined_day_median=float(qA.combined_day.median()),
                                           met_nb=int(qA.met_nb.sum()), met_d088=int(qA.met_d088.sum()), met_day=int(qA.met_day.sum()), n=len(qA))
        qq = Bt[Bt.policy.isin(QP + ["icp", "mondrian"]) & (Bt.policy != "handbook")]
        lv = Bt[Bt.policy.isin(QP)]
        rd["battery_level_direction_quantile_15min"] = dict(lower=int((lv.level_day < lv.level_nb).sum()), equal=int((lv.level_day == lv.level_nb).sum()),
                                                           higher=int((lv.level_day > lv.level_nb).sum()), n=len(lv))
        ad = pd.read_csv(inp("S61B/ad_cells.csv"))
        ad = ad[ad.primary & ad.scenario.isin(["battery_day", "batteryB_day"])].copy()
        ad["F0"] = [json.load(open(R / "S6E" / f"f{r.fold}_h3_select.json"))["selections"]["main" if r.scenario == "battery_day" else "shapeB"]["anchors"]["handbook"]["fuel_L"]
                    for r in ad.itertuples()]
        ad["sav_vs_nb"] = 100 * (1 - ad.fuel_L / ad.F0)
        adq = ad[ad.policy.isin(QP)]
        rd["battery_AD_day"] = dict(q_met=int((adq.ue_share <= TGT).sum()), q_n=len(adq), q_combined_median=float(adq.sav_vs_nb.median()),
                                    cp_met=int((ad[ad.policy.isin(CP)].ue_share <= TGT).sum()), cp_n=int(ad.policy.isin(CP).sum()))
        Bt.to_csv(out / "battery_day.csv", index=False)
    # ---------------------------------------------------------------- 7. events
    if (R / "WP61" / "events.csv").exists():
        E = pd.read_csv(inp("WP61/events.csv"))
        e = E[E.policy.isin(CP)]
        ev = {}
        for pr, g in e.groupby("protocol"):
            ev[pr] = dict(events=len(g), peak_kW_median=float(g.peak_kW.median()), peak_kW_p90=float(g.peak_kW.quantile(0.9)),
                          peak_kW_max=float(g.peak_kW.max()), steps_median=float(g.steps.median()), steps_max=int(g.steps.max()),
                          share_largest_median=float(g.peak_share_largest.median()), above_largest=int((g.peak_share_largest > 1).sum()),
                          one_step_share=float((g.steps == 1).mean()), start_hour_median=float(g.start_hour.median()))
        rd["events_conformal"] = ev
    (out / "reading.json").write_text(json.dumps(rd, indent=1, default=float), encoding="utf-8")
    ok = all(v["ok"] for v in checks.values())
    outs = [p for p in ("selection.csv", "load_scales.csv", "battery_day.csv", "reading.json") if (out / p).exists()]
    (out / "read_checks.json").write_text(json.dumps(dict(ok=ok, checks=checks, inputs_md5=inputs,
                                                          outputs_md5={p: md5(out / p) for p in outs}), indent=1), encoding="utf-8")
    print(json.dumps(rd, indent=1, default=float)[:6000])
    print("WP61-READ PASS" if ok else "WP61-READ FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
