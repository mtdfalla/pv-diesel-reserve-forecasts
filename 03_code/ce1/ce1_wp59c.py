"""Seasonal coverage of the static methods and the regime mix behind marginal ICP's transitional
coverage in test year 2020, from the S2 caches and S5 records.

Usage (from the repository root unless the path starts at ce1/, then from 03_code):
    python3 03_code/ce1/ce1_wp59c.py --root .
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

FOLDS = range(2019, 2025)
METHODS = ["icp", "mondrian", "mondrian_cqr"]
REG = {0: "clear", 1: "transitional", 2: "cloudy"}
SEASON = {12: "DJF", 1: "DJF", 2: "DJF", 3: "MAM", 4: "MAM", 5: "MAM",
          6: "JJA", 7: "JJA", 8: "JJA", 9: "SON", 10: "SON", 11: "SON"}
ALPHA = 0.1


def md5(p: Path) -> str:
    return hashlib.md5(p.read_bytes()).hexdigest()


def split_quantile(s: np.ndarray, alpha: float = ALPHA) -> float:
    """Finite-sample split-conformal quantile, rank ceil((n+1)(1-alpha))."""
    s = np.sort(s)
    k = int(np.ceil((len(s) + 1) * (1 - alpha))) - 1
    return float(s[min(k, len(s) - 1)])


def weighted_quantile(s: np.ndarray, w: np.ndarray, alpha: float = ALPHA) -> float:
    o = np.argsort(s)
    c = np.cumsum(w[o]) / w.sum()
    return float(s[o][np.searchsorted(c, 1 - alpha)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(Path(__file__).resolve().parents[2]))
    a = ap.parse_args()
    R = Path(a.root) / "04_results" / "ce1"
    out = R / "WP59C"
    out.mkdir(parents=True, exist_ok=True)
    checks, inputs = {}, {}

    def check(name, ok, detail=""):
        checks[name] = dict(ok=bool(ok), detail=str(detail))
        print(("ok    " if ok else "FAIL  ") + name + ("" if ok else "  " + str(detail)))

    iv = pd.read_csv(R / "S5" / "intervals.csv")
    inputs["S5/intervals.csv"] = md5(R / "S5" / "intervals.csv")
    seas, dip = [], []
    for fo in FOLDS:
        p2 = R / "S2" / f"yulara_f{fo}_h1.parquet"
        p5 = R / "S5" / f"yulara_f{fo}_h1_obs.npz"
        inputs[str(p2.relative_to(R))] = md5(p2)
        inputs[str(p5.relative_to(R))] = md5(p5)
        d = pd.read_parquet(p2)
        c, t = d[d["split"] == "calib"], d[d["split"] == "test"]
        o = np.load(p5)
        ts = pd.DatetimeIndex(pd.to_datetime(o["ts_ns"]))
        tt = t.loc[ts]
        reg = pd.Series(o["reg_it"]).map(REG).values
        check(f"f{fo} S5 records align with the S2 test rows (regime, forecast)",
              (reg == tt["g"].values).all() and np.allclose(tt["p"].values, o["p"], atol=1e-3),
              np.abs(tt["p"].values - o["p"]).max())
        cal_first = c.index.min().year
        check(f"f{fo} calibration year is {fo - 1}", cal_first == fo - 1 and c.index.max().year == fo - 1, cal_first)

        # --- seasonal and monthly coverage
        mon = ts.month
        for m in METHODS:
            cov = o[f"cov__{m}|static|0.9"].astype(float)
            for scope, msk in (("all", np.ones(len(cov), bool)), ("transitional", reg == "transitional")):
                s_ = pd.Series(cov[msk])
                for k, g in s_.groupby(np.asarray(mon[msk])):
                    seas.append(dict(fold=fo, method=m, scope=scope, month=int(k), season=SEASON[int(k)],
                                     n=int(len(g)), covered=int(g.sum()), picp=float(g.mean())))
            # reproduction of S5 overall and transitional PICP
            for scope, msk in (("all", np.ones(len(cov), bool)), ("transitional", reg == "transitional")):
                ref = iv[(iv.site == "yulara") & (iv.fold == fo) & (iv.horizon_min == 5) & (iv.method == m)
                         & (iv.variant == "static") & (iv.nominal == 0.9) & (iv.scope == scope)].PICP.iloc[0]
                # aggregation identity only (intervals.csv is aggregated from the same flags); the coverage
                # itself is tested independently for marginal ICP below (reproduced per observation)
                check(f"f{fo} {m} {scope} S5 flags aggregate to intervals.csv PICP ({ref}) [aggregation identity]",
                      round(cov[msk].mean(), 4) == round(ref, 4), cov[msk].mean())

        # --- marginal ICP decomposition
        sc = np.abs(c["y"].values - c["p"].values)
        st = np.abs(tt["y"].values - tt["p"].values)
        q = split_quantile(sc)
        cov_icp = o["cov__icp|static|0.9"]
        mine = (st <= q)
        check(f"f{fo} reproduced ICP coverage equals S5 per observation", (mine == cov_icp).mean() > 0.9995,
              (mine == cov_icp).mean())
        check(f"f{fo} reproduced ICP width 2q equals S5 maximum width", abs(2 * q - float(o["wid__icp|static|0.9"].max())) < 1e-2,
              (2 * q, float(o["wid__icp|static|0.9"].max())))
        gc, gt = c["g"].values, reg
        mix_c = pd.Series(gc).value_counts(normalize=True)
        mix_t = pd.Series(gt).value_counts(normalize=True)
        w = pd.Series(gc).map(mix_t / mix_c).values
        q_rw = weighted_quantile(sc, w)
        trc, trt = gc == "transitional", gt == "transitional"
        dip.append(dict(
            fold=fo, calib_year=fo - 1,
            calib_share_transitional=float(trc.mean()), calib_share_cloudy=float((gc == "cloudy").mean()),
            test_share_transitional=float(trt.mean()), test_share_cloudy=float((gt == "cloudy").mean()),
            icp_q=q, icp_q_reweighted=q_rw,
            cov_transitional_test=float(mine[trt].mean()),
            cov_transitional_calib=float((sc[trc] <= q).mean()),
            cov_transitional_reweighted=float((st[trt] <= q_rw).mean()),
            q90_abs_err_transitional_calib=split_quantile(sc[trc]),
            q90_abs_err_transitional_test=split_quantile(st[trt]),
            mae_transitional_calib=float(sc[trc].mean()), mae_transitional_test=float(st[trt].mean())))

    S = pd.DataFrame(seas)
    D = pd.DataFrame(dip)
    S.to_csv(out / "seasonal.csv", index=False)
    D.to_csv(out / "icp_dip.csv", index=False)

    # ---  missingness against weather
    miss_rows = []
    for site, folds in (("yulara", FOLDS), ("asp", [2024])):
        g1 = R / "S1" / f"{site}_geo_5min.parquet"
        inputs[str(g1.relative_to(R))] = md5(g1)
        geo = pd.read_parquet(g1)
        for fo in folds:
            p2 = R / "S2" / f"{site}_f{fo}_h1.parquet"
            inputs.setdefault(str(p2.relative_to(R)), md5(p2))
            s2 = pd.read_parquet(p2, columns=["split", "g"])
            lab_src = s2[s2["split"] == "test"]["g"]
            y = geo.loc[str(fo)]
            day = y["is_day"].values.astype(bool)
            miss = y["ghi"].isna().values & day
            lab = lab_src.reindex(y.index).ffill(limit=12).fillna("unlabelled").values
            check(f"{site} f{fo} every scored test issue time is a daytime step of the S1 grid",
                  bool(y["is_day"].reindex(lab_src.index).fillna(False).all()))
            mon = y.index.month
            for k in range(1, 13):
                m_ = day & (mon == k)
                miss_rows.append(dict(site=site, fold=fo, label=f"month{k:02d}", daytime_steps=int(m_.sum()), missing=int((miss & m_).sum()),
                                      share=float((miss & m_).sum() / max(m_.sum(), 1))))
            for L in ("clear", "transitional", "cloudy", "unlabelled", "all"):
                m_ = day & ((lab == L) if L != "all" else True)
                miss_rows.append(dict(site=site, fold=fo, label=L, daytime_steps=int(m_.sum()), missing=int((miss & m_).sum()),
                                      share=float((miss & m_).sum() / max(m_.sum(), 1))))
    Mi = pd.DataFrame(miss_rows)
    Mi.to_csv(out / "missing.csv", index=False)
    lab3 = Mi[Mi.label.isin(["clear", "transitional", "cloudy"])].pivot_table(index=["site", "fold"], columns="label", values="share")
    for (site_, fo_), g_ in Mi[Mi.label.isin(["clear", "transitional", "cloudy"])].groupby(["site", "fold"]):
        n_s2 = int((pd.read_parquet(R / "S2" / f"{site_}_f{fo_}_h1.parquet", columns=["split"])["split"] == "test").sum())
        check(f"{site_} f{fo_} labelled daytime steps cover the scored S2 test issue times",
              int(g_.daytime_steps.sum()) >= n_s2, (int(g_.daytime_steps.sum()), n_s2))
    r_ = lab3.max(axis=1) / lab3.min(axis=1)
    check("trigger (a regime's missing share differs from another's by more than a factor of two) fires",
          bool((r_ > 2).any()), r_.round(2).to_dict())
    check("labelled missing shares at most 6.9% in any regime, test year and site", float(lab3.values.max()) <= 0.069 + 5e-5,
          float(lab3.values.max()))

    for fo in FOLDS:
        n_iv = int(iv[(iv.site == "yulara") & (iv.fold == fo) & (iv.horizon_min == 5) & (iv.method == "icp")
                      & (iv.variant == "static") & (iv.nominal == 0.9) & (iv.scope == "all")].n.iloc[0])
        check(f"f{fo} seasonal counts equal the S5 intervals.csv n [aggregation identity]",
              int(S[(S.scope == "all") & (S.method == "icp") & (S.fold == fo)].n.sum()) == n_iv, n_iv)
    z = D.set_index("fold")
    check("2020 has the lowest ICP transitional coverage (0.660)", z.cov_transitional_test.idxmin() == 2020
          and round(z.loc[2020, "cov_transitional_test"], 3) == 0.660)
    check("its calibration year 2019 has the lowest transitional and cloudy shares of the six calibration years",
          z.calib_share_transitional.idxmin() == 2020 and z.calib_share_cloudy.idxmin() == 2020)
    check("2020 has the smallest ICP quantile", z.icp_q.idxmin() == 2020)
    check("2020's transitional error quantile lies inside the range of the other test years",
          z.drop(2020).q90_abs_err_transitional_test.min() <= z.loc[2020, "q90_abs_err_transitional_test"]
          <= z.drop(2020).q90_abs_err_transitional_test.max())
    check("reweighting raises 2020's coverage but it stays lowest among the reweighted years (explains part)",
          z.loc[2020, "cov_transitional_reweighted"] > z.loc[2020, "cov_transitional_test"]
          and z.cov_transitional_reweighted.idxmin() == 2020,
          z.cov_transitional_reweighted.round(3).to_dict())
    ok = all(v["ok"] for v in checks.values())
    (out / "checks.json").write_text(json.dumps(dict(ok=ok, checks=checks, inputs_md5=inputs,
                                                    outputs_md5={p.name: md5(p) for p in (out / "seasonal.csv", out / "icp_dip.csv", out / "missing.csv")}),
                                               indent=1), encoding="utf-8")
    print(f"{sum(v['ok'] for v in checks.values())} of {len(checks)} checks pass; " + ("WP59C PASS" if ok else "WP59C FAIL"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
