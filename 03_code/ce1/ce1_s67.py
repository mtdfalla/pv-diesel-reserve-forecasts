"""Stage S67: PV doubled and tripled (with and without the largest-unit reserve), the sorted
rerun and the 2022 load scale (rules D-129, D-131).

Usage (from the repository root unless the path starts at ce1/, then from 03_code):
    python3 03_code/ce1/ce1_s67.py --fold 2024 [--only pv2,pv3,...]
    python3 03_code/ce1/ce1_s67.py --analyse
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
TGT = LM.UE_TARGET
HALF = TGT / 2.0
EPS = 1e-15
GAMMA, KAPPA = 0.005, 0.5
POLICIES = list(D.ALL_POLICIES)
QSET = ["raw_q", "cqr", "mondrian_cqr"]
CSET = ["icp", "mondrian", "cqr", "mondrian_cqr"]
SHAPES = {"A": "main", "B": "shapeB"}
GROUPS = {"pv2": dict(pv=2.0), "pv3": dict(pv=3.0), "n1pv2": dict(pv=2.0, n1=True), "n1pv3": dict(pv=3.0, n1=True),
          "l22": dict(scale="y2022"), "srt": dict(sorted=True)}
RULES = ["A_R", "a_half_H2", "b_xfit_0.002", "b_xfit_0.001", "c_prior_0.002", "c_prior_0.001", "A_D", "AW_clip"]
LABEL = {"A_R": "A-R", "a_half_H2": "H2", "b_xfit_0.002": "XF", "b_xfit_0.001": "XH", "c_prior_0.002": "PF",
         "c_prior_0.001": "PH", "A_D": "A-D", "AW_clip": "CL"}


def md5(p: Path) -> str:
    h = hashlib.md5()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def log(m):
    print(time.strftime("%H:%M:%S"), m, flush=True)


def scale_2022(geo, m, unit96):
    """The primary-scale rule of ce1_s6.load_inputs applied to a given year (shape B = shape A)."""
    import ce1_s6 as S6

    def sc(y):
        g = S6.year_grid(f"{y}-01-01", f"{y}-12-31 23:55")
        pav = LM.available_pv_on_grid(g, geo["ghi"].reindex(g).values, m)
        return LM.scale_energy_share(LM.load_on(g, unit96, "A"), pav)
    return sc(2022), sc(2023)


def run_fold(R: Path, yr: int, only=None) -> bool:
    import ce1_adaptive as A
    import ce1_folds as FC
    import ce1_s6 as S6
    import ce1_s63 as S63
    import ce1_s64 as S64
    import ce1_wp56 as W56
    import ce1_wp57 as W57
    out = R / "S67"
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    groups = [g for g in GROUPS if (only is None or g in only)]
    geo, m, unit96, scales, _ = S6.load_inputs(R / "S1", R / "S1M", R / "S1L")
    s22, s23 = scale_2022(geo, m, unit96)
    scales22 = {shp: dict(primary=s22) for shp in ("A", "B")}  # shape B = shape A scale (D-084), as load_inputs
    D.extend_theta_low()
    rows = S6.row_table()
    hb = [i for i, r in enumerate(rows) if r[0] == "handbook"][0]
    det = [i for i, r in enumerate(rows) if r[0] == "deterministic"][0]
    orc = [i for i, r in enumerate(rows) if r[0] == "oracle"][0]
    cache = FC.load_cache_ext(R / "S2" / f"yulara_f{yr}_h3.parquet", R / "S2X" / f"yulara_f{yr}_h3.parquet")
    tot_e = pd.read_parquet(R / "S6E" / f"f{yr}_h3_totals.parquet")
    prep = S6.prepare(cache, H, yr, geo, m, rows)
    calib = S6.as_dict(cache[cache["split"] == "calib"])
    H1, H2d = D.split_halves(calib)
    cy = yr - 1
    tgt = pd.DatetimeIndex(H1["idx"]) + pd.Timedelta(minutes=5 * H)
    gridx = S6.year_grid(f"{cy}-01-01", f"{cy}-06-30 23:55")
    pavx = LM.available_pv_on_grid(gridx, geo["ghi"].reindex(gridx).values, m)
    pv_of = S6.PVAt(tgt, m)
    prep["H1x"] = dict(grid=gridx, pav=pavx, day=LM.day_ids(gridx),
                       C=S6.place(gridx, tgt, S6.counted_matrix(rows, H2d, H1, pv_of, pv_of(H1["y"]))))
    PARTS = ("H1x", "H2", "test")
    log(f"fold {yr}: counted PV ready ({time.time() - t0:.0f} s, {len(rows)} rows)")

    def pol_levels(pol):
        ix = [i for i, r in enumerate(rows) if r[0] == pol]
        lv = np.array([rows[i][1] for i in ix])
        o = np.argsort(lv)
        return lv[o], np.array(ix)[o]

    for g in groups:
        tg = time.time()
        checks = {}

        def check(name, cond, msg=""):
            checks[name] = dict(ok=bool(cond), msg=str(msg))
            log(f"{'PASS' if cond else 'FAIL'}  {name}  {msg}")

        G = GROUPS[g]
        f = G.get("pv", 1.0)
        tables, cells = [], []
        if g == "l22":
            check("l22_shape_B_scale_is_shape_A", scales["B"]["primary"] == scales["A"]["primary"])
            check("l22_rule_on_2023_reproduces_primary", abs(s23 / scales["A"]["primary"] - 1) < 1e-12,
                  (s23, scales["A"]["primary"], s22))
        if g == "pv2":
            # factor 1 with this code reproduces S6E main on H2 and test, every row
            for part in ("H2", "test"):
                P = prep[part]
                sc = S6.SCENARIOS["main"]
                L = S6.load_for(P["grid"], unit96, scales, sc)
                r1 = LM.simulate(L, P["pav"] * 1.0, P["C"] * 1.0, P["day"], H, ratings=LM.fleet_for_peak(L.max()),
                                 min_load=sc["min_load"], fuel=sc["fuel"], battery=False)
                t = S64.tab_of(r1, rows, part, "main")
                ref = tot_e[(tot_e.scenario == "main") & (tot_e.set == part)]
                mm = t.merge(ref, on=["policy", "param"], suffixes=("", "_ref"))
                worst = max(float(np.max(np.abs(mm[k] - mm[k + "_ref"]) / np.maximum(np.abs(mm[k + "_ref"]), 1.0)))
                            for k in ("fuel_L", "ue_kWh", "starts"))
                check(f"factor1_reproduces_S6E_main_f{yr}_{part}", len(mm) == len(t) and worst < 1e-9, f"(worst {worst:.1e})")
        for shp, base in SHAPES.items():
            sname = f"{g}{shp}"
            sc = dict(S6.SCENARIOS[base])
            res, Cpart = {}, {}
            for part in PARTS:
                P = prep[part]
                if G.get("scale") == "y2022":
                    L = S6.load_for(P["grid"], unit96, scales22, sc)
                else:
                    L = S6.load_for(P["grid"], unit96, scales, sc)
                ratings = list(np.asarray(LM.fleet_for_peak(L.max()), float))
                pav = P["pav"] * f
                C = P["C"] * f if f != 1.0 else (P["C"].copy() if G.get("sorted") else P["C"])
                if G.get("sorted"):
                    for pol in POLICIES:
                        lv, ix = pol_levels(pol)
                        Cs = S63.rearrange(C[ix])
                        if pol == "frac_pv":
                            check(f"srt_frac_pv_unchanged_f{yr}_{shp}_{part}", np.array_equal(Cs, C[ix], equal_nan=True))
                        C[ix] = Cs
                binds, n1kw = None, {}
                if G.get("n1"):
                    Ce_t, binds, infeas = W56.effective_counted(L, C, C[det], ratings, True)
                    del infeas
                    # v2.1 (D-161): the construction inside simulate; registered v2.0.0: the transform
                    C, _, n1kw = (Ce_t, None, {}) if W56.n1_registered() else W56.n1_setup("s56", L, C, C[det], ratings)
                    del Ce_t
                res[part] = LM.simulate(L, pav, C, P["day"], H, ratings=ratings, min_load=sc["min_load"], fuel=sc["fuel"],
                                        battery=False, **n1kw)
                t = S64.tab_of(res[part], rows, part, sname)
                if binds is not None:
                    dayt = np.isfinite(pav) & (pav > 0)
                    t["pv_binds_share"] = binds[:, dayt].mean(axis=1)
                if part == "test":
                    Cpart[part] = (L, pav, C, ratings, binds, n1kw)  # only the test record feeds A-D and CL
                else:
                    del C, binds
                    import gc
                    gc.collect()
                tables.append(t)
                if not G.get("n1"):
                    check(f"handbook_zero_ue_{sname}_f{yr}_{part}", float(res[part]["total"]["ue_kWh"][hb]) == 0.0)
                check(f"oracle_zero_ue_{sname}_f{yr}_{part}", float(res[part]["total"]["ue_kWh"][orc]) < 1e-9,
                      float(res[part]["total"]["ue_kWh"][orc]))
            T = {p_: tables[-3 + i] for i, p_ in enumerate(PARTS)}
            hbT = float(res["test"]["total"]["fuel_L"][hb])
            hbS = float(res["test"]["total"]["starts"][hb])
            for pol in POLICIES:
                X = W57.pooled([T["H2"][T["H2"].policy == pol].reset_index(drop=True),
                                T["H1x"][T["H1x"].policy == pol].reset_index(drop=True)])
                for vname, (tab, tgv) in {"A_R": (T["H2"], TGT), "a_half_H2": (T["H2"], HALF), "b_xfit_0.002": (X, TGT),
                                          "b_xfit_0.001": (X, HALF)}.items():
                    p_, met = W57.select(tab, pol, tgv)
                    rr = W57.at(T["test"], pol, p_)
                    cells.append(dict(fold=yr, scenario=sname, group=g, shape=shp, policy=pol, selection=vname, param=p_,
                                      met_on_selection=met, fuel_L=rr["fuel_L"], ue_kWh=rr["ue_kWh"], ue_share=rr["ue_share"],
                                      starts=rr["starts"], fuel_handbook=hbT, handbook_starts=hbS,
                                      pv_binds_share=rr.get("pv_binds_share", np.nan)))
            # A-D and CL from the scenario's own daily record (S64 ad_block construction)
            L, pav, Ct, ratings, binds, n1kw = Cpart["test"]
            P = prep["test"]
            if G.get("scale") == "y2022":
                abar = A.h2_mean_daily_load(prep["H2"], unit96, scales22, sc)
            else:
                abar = A.h2_mean_daily_load(prep["H2"], unit96, scales, sc)
            load_d = res["test"]["daily"]["load_kWh"][hb]
            specs, meta = [], []
            kinds = (("A_D", np.inf),) if G.get("n1") else (("A_D", np.inf), ("AW_clip", 1.0))
            for pol in POLICIES:
                lv, ix = pol_levels(pol)
                rec_ue = res["test"]["daily"]["ue_kWh"][ix]
                sub = T["H2"][T["H2"].policy == pol].sort_values("param")
                t0_, _ = LM.select_AR(dict(ue_share=sub.ue_share.to_numpy(float), fuel_L=sub.fuel_L.to_numpy(float)), lv, pol)
                lo = float(lv.min())
                for kind, clip in kinds:
                    init = S64.ad_levels(lambda d, i, R_=rec_ue: R_[i, d], load_d, t0_, GAMMA, KAPPA, abar, lv, lo, clip)
                    ctrl = (lambda ue, ld, lv_=lv, t0__=t0_, lo_=lo, cl=clip:
                            S64.ad_levels(lambda d, i, u=ue: u[d], load_d, t0__, GAMMA, KAPPA, abar, lv_, lo_, cl))
                    specs.append((Ct[ix], init, ctrl))
                    meta.append((pol, kind, t0_, ix))
            r, idxp, passes, work = S64.fixed_point(LM.simulate, dict(ratings=ratings, min_load=sc["min_load"], fuel=sc["fuel"],
                                                                      battery=False, **n1kw), L, pav, P["day"], H, specs)
            check(f"fixed_point_f{yr}_{sname}", 1 <= passes < 400, f"({passes} passes, {work} row-years)")
            dayt = np.isfinite(pav) & (pav > 0)
            for k, (pol, kind, t0_, ix) in enumerate(meta):
                lv, _ = pol_levels(pol)
                lvp = lv[idxp[k]]
                bsh = np.nan
                if binds is not None:
                    bsh = float(np.nanmean(A.path_row(binds[ix].astype(float), idxp[k], P["day"])[dayt]))
                cells.append(dict(fold=yr, scenario=sname, group=g, shape=shp, policy=pol, selection=kind, param=float(np.mean(lvp)),
                                  met_on_selection=np.nan, fuel_L=float(r["total"]["fuel_L"][k]), ue_kWh=float(r["total"]["ue_kWh"][k]),
                                  ue_share=float(r["total"]["ue_share"][k]), starts=float(r["total"]["starts"][k]),
                                  fuel_handbook=hbT, handbook_starts=hbS, pv_binds_share=bsh, theta0=t0_,
                                  days_at_limit=int((lvp >= 1.0 - 1e-9).sum()), passes=passes))
            log(f"fold {yr} {sname}: done ({time.time() - tg:.0f} s)")
            del res, Cpart
        tt = pd.concat(tables, ignore_index=True)
        tt.to_parquet(out / f"f{yr}_{g}_totals.parquet", index=False)
        pd.DataFrame(cells).to_csv(out / f"f{yr}_{g}_cells.csv", index=False)
        ok = all(c["ok"] for c in checks.values())
        (out / f"f{yr}_{g}_checks.json").write_text(json.dumps(dict(
            ok=ok, n=len(checks), checks=checks, seconds=time.time() - tg, scale_2022_mean_kW=s22,
            code={p.name: md5(p) for p in (Path(__file__), HERE / "ce1_s64.py", HERE / "ce1_load.py", HERE / "ce1_s6.py",
                                            HERE / "ce1_wp56.py", HERE / "ce1_wp57.py", HERE / "ce1_s63.py")},
            env=dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__)), indent=1))
        log(f"fold {yr} group {g}: {sum(c['ok'] for c in checks.values())} of {len(checks)} checks ({time.time() - tg:.0f} s)")
        if not ok:
            return False
    return True


def n1_factor1_check(R: Path, yr: int) -> bool:
    """N-1 at factor 1 with this code's path reproduces the S56 cont / contB test totals."""
    import ce1_folds as FC
    import ce1_s6 as S6
    import ce1_s64 as S64
    import ce1_wp56 as W56
    geo, m, unit96, scales, _ = S6.load_inputs(R / "S1", R / "S1M", R / "S1L")
    D.extend_theta_low()
    rows = S6.row_table()
    det = [i for i, r in enumerate(rows) if r[0] == "deterministic"][0]
    cache = FC.load_cache_ext(R / "S2" / f"yulara_f{yr}_h3.parquet", R / "S2X" / f"yulara_f{yr}_h3.parquet")
    P = S6.prepare(cache, H, yr, geo, m, rows)["test"]
    s56 = pd.read_parquet(R / "S56" / f"f{yr}_h3_totals.parquet")
    ok = True
    for name, base in (("cont", "main"), ("contB", "shapeB")):
        sc = S6.SCENARIOS[base]
        L = S6.load_for(P["grid"], unit96, scales, sc)
        ratings = list(np.asarray(LM.fleet_for_peak(L.max()), float))
        C, _, n1kw = W56.n1_setup("s56", L, P["C"] * 1.0, P["C"][det] * 1.0, ratings)  # N-1 construction s56
        r = LM.simulate(L, P["pav"] * 1.0, C, P["day"], H, ratings=ratings, min_load=sc["min_load"], fuel=sc["fuel"], battery=False,
                        **n1kw)
        t = S64.tab_of(r, rows, "test", name)
        ref = s56[(s56.scenario == name) & (s56.set == "test")]
        mm = t.merge(ref, on=["policy", "param"], suffixes=("", "_ref"))
        worst = max(float(np.max(np.abs(mm[k] - mm[k + "_ref"]) / np.maximum(np.abs(mm[k + "_ref"]), 1.0)))
                    for k in ("fuel_L", "ue_kWh", "starts"))
        good = len(mm) == len(t) and worst < 1e-9
        log(f"{'PASS' if good else 'FAIL'}  n1_factor1_reproduces_S56_{name}_f{yr}  (worst {worst:.1e})")
        ok &= good
    (R / "S67" / f"f{yr}_n1factor1_checks.json").write_text(json.dumps(dict(ok=bool(ok)), indent=1))
    return ok


def analyse(R: Path) -> bool:
    import ce1_wp57 as W57
    out = R / "S67"
    cells, tots = [], {}
    for g in GROUPS:
        for yr in FOLDS:
            cells.append(pd.read_csv(out / f"f{yr}_{g}_cells.csv"))
            tots[(g, yr)] = pd.read_parquet(out / f"f{yr}_{g}_totals.parquet")
    c = pd.concat(cells, ignore_index=True)
    # PF and PH: pooled test tables of all earlier folds of the same scenario (2019: H2)
    prior_rows = []
    for g in GROUPS:
        for shp in SHAPES:
            sname = f"{g}{shp}"
            for yr in FOLDS:
                T = tots[(g, yr)]
                T2 = T[(T.scenario == sname) & (T.set == "H2")]
                TT = T[(T.scenario == sname) & (T.set == "test")]
                prior = [tots[(g, y)][(tots[(g, y)].scenario == sname) & (tots[(g, y)].set == "test")] for y in FOLDS if y < yr]
                hbT = float(TT[TT.policy == "handbook"].fuel_L.iloc[0])
                hbS = float(TT[TT.policy == "handbook"].starts.iloc[0])
                for pol in POLICIES:
                    Pp = (W57.pooled([p_[p_.policy == pol].reset_index(drop=True) for p_ in prior]) if prior
                          else T2[T2.policy == pol].reset_index(drop=True))
                    for vname, tg in (("c_prior_0.002", TGT), ("c_prior_0.001", HALF)):
                        p_, met = W57.select(Pp, pol, tg)
                        rr = W57.at(TT, pol, p_)
                        prior_rows.append(dict(fold=yr, scenario=sname, group=g, shape=shp, policy=pol, selection=vname, param=p_,
                                               met_on_selection=met, fuel_L=rr["fuel_L"], ue_kWh=rr["ue_kWh"], ue_share=rr["ue_share"],
                                               starts=rr["starts"], fuel_handbook=hbT, handbook_starts=hbS,
                                               pv_binds_share=rr.get("pv_binds_share", np.nan)))
    c = pd.concat([c, pd.DataFrame(prior_rows)], ignore_index=True)
    c["rule"] = c.selection.map(LABEL)
    c["met"] = c.ue_share <= TGT + EPS
    c["saving_pct"] = 100 * (1 - c.fuel_L / c.fuel_handbook)
    c.to_csv(out / "cases.csv", index=False)
    # end-to-end check: rearrangement leaves frac_pv unchanged, so every srt frac_pv rule must equal WP-63
    w = pd.read_csv(R / "WP63" / "cases.csv", float_precision="round_trip")
    w = w[w.policy == "frac_pv"].assign(shape=lambda d: d.scenario.map({"main": "A", "shapeB": "B"}))
    s = c[(c.group == "srt") & (c.policy == "frac_pv")]
    mm = s.merge(w, on=["fold", "shape", "selection"], suffixes=("", "_w"))
    worst = float(np.max(np.abs(mm.fuel_L / mm.fuel_L_w - 1)))
    okr = len(mm) == 6 * 2 * len(RULES) and worst < 1e-9 and bool((mm.starts == mm.starts_w).all())
    print(f"{'PASS' if okr else 'FAIL'}  srt_frac_pv_reproduces_WP63_every_rule  ({len(mm)} cases, worst {worst:.1e})")
    chk = {"srt_frac_pv_reproduces_WP63_every_rule": dict(ok=okr, n=len(mm), worst=worst)}
    for g in GROUPS:
        for yr in FOLDS:
            j = json.loads((out / f"f{yr}_{g}_checks.json").read_text())
            chk[f"stage_f{yr}_{g}"] = dict(ok=bool(j["ok"]), n=j["n"])
    for yr in FOLDS:
        j = json.loads((out / f"f{yr}_n1factor1_checks.json").read_text())
        chk[f"n1factor1_f{yr}"] = dict(ok=bool(j["ok"]))
    (out / "checks.json").write_text(json.dumps(dict(ok=all(v["ok"] for v in chk.values()), checks=chk), indent=1))
    if not okr:
        return False
    # anchors per scenario: handbook, oracle ceiling, PV energy share, binding share
    anc = []
    for (g, yr), T in tots.items():
        for sname in T.scenario.unique():
            TT = T[(T.scenario == sname) & (T.set == "test")]
            h_ = TT[TT.policy == "handbook"].iloc[0]
            o_ = TT[TT.policy == "oracle"].iloc[0]
            anc.append(dict(group=g, fold=yr, scenario=sname, fuel_handbook=h_.fuel_L, ceiling_pct=100 * (1 - o_.fuel_L / h_.fuel_L),
                            pv_energy_share=float(h_.pav_kWh / h_.load_kWh), handbook_curt_share=float(h_.curt_share),
                            handbook_pv_binds=float(h_.get("pv_binds_share", np.nan)),
                            oracle_pv_binds=float(o_.get("pv_binds_share", np.nan))))
    anc = pd.DataFrame(anc)
    anc.to_csv(out / "anchors.csv", index=False)

    def xh(g):
        s = c[(c.group == g) & (c.selection == "b_xfit_0.001") & c.policy.isin(QSET)]
        assert len(s) == 36, (g, len(s))
        return int(s.met.sum()), float(s.saving_pct.median())

    rd = {}
    m2, s2 = xh("pv2")
    m3, s3 = xh("pv3")
    rd["1_pv"] = dict(xh_quantile_met_pv2=m2, xh_quantile_median_saving_pv2=s2, xh_quantile_met_pv3=m3,
                      xh_quantile_median_saving_pv3=s3,
                      rule_i_conclusions_state_growth=bool(m2 >= 34 and m3 >= 34),
                      rule_ii_reselect_at_higher_penetration=bool(m2 < 30 or m3 < 30),
                      rule_iv_title_changes=bool(s2 < 0 or s3 < 0),
                      n1_handbook_pv_binds={g: float(anc[anc.group == g].handbook_pv_binds.max()) for g in ("n1pv2", "n1pv3")})
    ms, ss = xh("srt")
    base = pd.read_csv(R / "WP63" / "cases.csv")
    b = base[(base.selection == "b_xfit_0.001") & base.policy.isin(QSET) & base.scenario.isin(["main", "shapeB"])]
    b_met = int((b.ue_share <= TGT + EPS).sum())
    b_sav = float((100 * (1 - b.fuel_L / b.fuel_handbook)).median()) if "fuel_handbook" in b else float("nan")
    rd["2_sorted"] = dict(xh_quantile_met=ms, xh_quantile_median_saving=ss, as_run_met=b_met, as_run_median_saving=b_sav,
                          abstract_carries_both=bool(ms < 34 or abs(ss - b_sav) > 0.05))
    ml, sl = xh("l22")
    rd["4_load2022"] = dict(xh_quantile_met=ml, xh_quantile_median_saving=sl, main_text=bool(ml < 34 or sl < 0))
    (out / "reading.json").write_text(json.dumps(rd, indent=1))
    print(json.dumps(rd, indent=1))
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--fold", type=int)
    ap.add_argument("--only")
    ap.add_argument("--n1check", action="store_true")
    ap.add_argument("--analyse", action="store_true")
    a = ap.parse_args()
    R = Path(a.root) / "04_results" / "ce1"
    if a.analyse:
        sys.exit(0 if analyse(R) else 1)
    if a.n1check:
        sys.exit(0 if n1_factor1_check(R, a.fold) else 1)
    only = a.only.split(",") if a.only else None
    sys.exit(0 if run_fold(R, a.fold, only) else 1)


if __name__ == "__main__":
    main()
