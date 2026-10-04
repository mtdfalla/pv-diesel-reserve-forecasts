"""Stage S7: statistical evidence from the S4 and S5 outputs (rules D-089). Diebold-Mariano tests
with Newey-West variance and the Harvey-Leybourne-Newbold correction, paired day-block
bootstrap intervals (B = 10,000, seed 42), Holm adjustment within each family, coverage
intervals and per-day RMSE models.

Usage (from the repository root unless the path starts at ce1/, then from 03_code):
    python ce1/ce1_stats.py --stage S7 [--smoke] [--B 10000]
    python ce1/ce1_stats.py --selftest
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
CODE = HERE.parent
for p in (HERE, CODE / "utils", CODE / "evaluation"):
    sys.path.insert(0, str(p))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

B_DEF = 10000
SEED = 42
REGIMES = ["clear", "transitional", "cloudy"]
STATIC = ["icp", "icp_norm", "mondrian", "cqr", "mondrian_cqr", "raw_q", "tod", "nicp_var"]
CLASSICAL = ["persistence", "smart_persistence", "linear_ar", "gbm"]
DAY_CLEAR_MEAN, DAY_CLEAR_STD, DAY_OVERCAST = 0.85, 0.12, 0.50


def log(m):
    print(f"[S7] {m}", flush=True)


def md5(p: Path) -> str:
    h = hashlib.md5()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ------------------------------------------------------------------ tests (v1 ports)
def dm_test(d, hac_lag, h):
    d = np.asarray(d, float); n = len(d); dbar = d.mean()
    x = d - dbar
    var = float(np.dot(x, x)) / n
    L = int(max(hac_lag, 0))
    for k in range(1, L + 1):
        ck = float(np.dot(x[k:], x[:-k])) / n
        var += 2.0 * (1.0 - k / (L + 1.0)) * ck
    if var <= 0 or n < 3:
        return np.nan, np.nan
    hh = int(max(h, 1))
    DM = dbar / np.sqrt(var / n)
    corr = (n + 1 - 2 * hh + hh * (hh - 1) / n) / n
    if corr <= 0:
        return float(DM), float(2 * stats.t.cdf(-abs(DM), df=n - 1))
    DM *= np.sqrt(corr)
    return float(DM), float(2 * stats.t.cdf(-abs(DM), df=n - 1))


class DayBoot:

    def __init__(self, day_code, n_days, B, seed=SEED):
        rng = np.random.default_rng(seed)
        self.idx = rng.integers(0, n_days, size=(B, n_days))
        self.day_code = np.asarray(day_code)
        self.nd = n_days

    def _per_day(self, v, mask):
        dc = self.day_code[mask]
        S = np.bincount(dc, weights=np.asarray(v, float)[mask], minlength=self.nd)
        C = np.bincount(dc, minlength=self.nd).astype(float)
        return S, C

    def mean_ci(self, v, mask, alpha=0.05):
        S, C = self._per_day(v, mask)
        num = S[self.idx].sum(axis=1); den = C[self.idx].sum(axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            b = num / den
        b = b[np.isfinite(b)]
        pt = float(S.sum() / C.sum()) if C.sum() > 0 else np.nan
        return pt, float(np.percentile(b, 100 * alpha / 2)), float(np.percentile(b, 100 * (1 - alpha / 2))), b

    def rmse_diff(self, e_a, e_b, mask, alpha=0.05):
        """RMSE_a - RMSE_b and (RMSE_a / RMSE_b - 1) with day-block intervals."""
        Sa, C = self._per_day(np.asarray(e_a, float) ** 2, mask)
        Sb, _ = self._per_day(np.asarray(e_b, float) ** 2, mask)
        den = C[self.idx].sum(axis=1)
        ra = np.sqrt(Sa[self.idx].sum(axis=1) / den); rb = np.sqrt(Sb[self.idx].sum(axis=1) / den)
        d, r = ra - rb, ra / rb - 1
        pa, pb = np.sqrt(Sa.sum() / C.sum()), np.sqrt(Sb.sum() / C.sum())
        q = lambda x: (float(np.percentile(x, 100 * alpha / 2)), float(np.percentile(x, 100 * (1 - alpha / 2))))
        return dict(rmse_a=float(pa), rmse_b=float(pb), d=float(pa - pb), d_ci=q(d),
                    rel=float(pa / pb - 1), rel_ci=q(r))


def holm(p):
    p = np.asarray(p, float)
    out = np.full(len(p), np.nan)
    ok = np.isfinite(p)
    idx = np.argsort(p[ok]); m = ok.sum()
    adj = np.maximum.accumulate(np.minimum(1.0, (m - np.arange(m)) * p[ok][idx]))
    tmp = np.empty(m); tmp[idx] = adj
    out[ok] = tmp
    return out


def day_codes(ts):
    days = pd.DatetimeIndex(ts).normalize().values
    u, code = np.unique(days, return_inverse=True)
    return code.astype(np.int32), len(u)


# ------------------------------------------------------------------ (a) point
def point_tests(df: pd.DataFrame, site, fold, h, B):
    code, nd = day_codes(df.index)
    hac_day = int(np.median(np.bincount(code)))
    boot = DayBoot(code, nd, B)
    y = df["y"].values
    rows = []
    others = [c for c in df.columns if c not in ("y", "g", "g_t", "gbm")]
    for m in others:
        ok = np.isfinite(df[m].values) & np.isfinite(df["gbm"].values)
        e_g, e_m = df["gbm"].values - y, df[m].values - y
        d = e_g ** 2 - e_m ** 2
        dm1, p1 = dm_test(d[ok], h - 1, h)
        dm2, p2 = dm_test(d[ok], hac_day, h)
        r = boot.rmse_diff(e_g, e_m, ok)
        rows.append(dict(site=site, fold=fold, horizon_min=h * 5, model_a="gbm", model_b=m, n=int(ok.sum()),
                         days=nd, hac_day=hac_day, rmse_gbm=round(r["rmse_a"], 4), rmse_other=round(r["rmse_b"], 4),
                         d_rmse=round(r["d"], 4), d_rmse_ci_lo=round(r["d_ci"][0], 4), d_rmse_ci_hi=round(r["d_ci"][1], 4),
                         rel_pct=round(100 * r["rel"], 3), rel_pct_ci_lo=round(100 * r["rel_ci"][0], 3),
                         rel_pct_ci_hi=round(100 * r["rel_ci"][1], 3),
                         dm_stat_hac_day=round(dm2, 3), dm_p_hac_h=p1, dm_p_hac_day=p2,
                         bootstrap_excludes_zero=bool((r["d_ci"][0] > 0) == (r["d_ci"][1] > 0))))
    hp = holm([r["dm_p_hac_day"] for r in rows])
    for r, q in zip(rows, hp):
        r["dm_p_hac_day_holm"] = q
    return rows


# ------------------------------------------------------------------ (b) intervals
def interval_stats(z, site, fold, hmin, B):
    h = hmin // 5
    code = z["day_code"]; nd = int(code.max()) + 1
    hac_day = int(np.median(np.bincount(code)))
    boot = DayBoot(code, nd, B)
    n = len(code)
    scopes = [("all", np.ones(n, bool))] + [(r, z["reg_it"] == i) for i, r in enumerate(REGIMES)] \
        + [(f"tt_{r}", z["reg_tt"] == i) for i, r in enumerate(REGIMES)]
    crps = {k[len("crps__"):]: z[k].astype(float) for k in z.files if k.startswith("crps__")}
    base = dict(site=site, fold=fold, horizon_min=hmin)
    rs, rc = [], []
    for scope, mask in scopes:
        if mask.sum() < 30:
            continue
        for key, v in crps.items():
            p, lo, hi, _ = boot.mean_ci(v, mask)
            rs.append(dict(base, scope=scope, kind="CRPS_level", method_a=key, method_b="", n=int(mask.sum()),
                           value=round(p, 4), ci_lo=round(lo, 4), ci_hi=round(hi, 4)))
        keys = [f"{m}|static" for m in STATIC if f"{m}|static" in crps]
        pairs = [(keys[i], keys[j]) for i in range(len(keys)) for j in range(i + 1, len(keys))]
        pairs += [(f"{m}|rearranged", f"{m}|static") for m in ("cqr", "mondrian_cqr") if f"{m}|rearranged" in crps]
        for A, Bk in pairs:
            dv = crps[A] - crps[Bk]
            dm1, p1 = dm_test(dv[mask], h - 1, h)
            dm2, p2 = dm_test(dv[mask], hac_day, h)
            _, lo, hi, _ = boot.mean_ci(dv, mask)
            rs.append(dict(base, scope=scope, kind="CRPS_diff", method_a=A, method_b=Bk, n=int(mask.sum()),
                           value=round(float(dv[mask].mean()), 4), ci_lo=round(lo, 4), ci_hi=round(hi, 4),
                           dm_stat_hac_day=round(dm2, 3), dm_p_hac_h=p1, dm_p_hac_day=p2,
                           significant_95=bool((lo > 0) == (hi > 0))))
    covs = {k[len("cov__"):]: z[k] for k in z.files if k.startswith("cov__")}
    for key, v in covs.items():
        m, variant, cov = key.split("|"); cov = float(cov)
        for scope, mask in scopes:
            if mask.sum() < 30:
                continue
            p, lo, hi, _ = boot.mean_ci(v.astype(float), mask)
            rc.append(dict(base, method=m, variant=variant, nominal=cov, scope=scope, n=int(mask.sum()),
                           PICP=round(p, 4), PICP_ci_lo=round(lo, 4), PICP_ci_hi=round(hi, 4),
                           ACE=round(p - cov, 4), covers_nominal=bool(lo <= cov <= hi)))
    return rs, rc


# ------------------------------------------------------------------ (c) per-day RMSE
def day_class_series(geo: pd.DataFrame, fold: int):
    import ce1_causal as K
    cal, _ = K.fit_scalar(geo, K.fold(fold)["train_max"])
    b = K.build_base(geo, cal, "locf6")
    b = b[b["is_day"] & b["kt"].notna() & (b["year"] == fold)]
    g = b.groupby(b.index.normalize())["kt"].agg(["mean", "std"])
    cls = np.where((g["mean"] >= DAY_CLEAR_MEAN) & (g["std"] < DAY_CLEAR_STD), "clear",
                   np.where(g["mean"] < DAY_OVERCAST, "overcast", "mixed"))
    return pd.Series(cls, index=g.index)


def perday_records(df: pd.DataFrame, dclass: pd.Series, fold, h):
    rows = []
    dn = df.index.normalize()
    dc = dclass.reindex(dn).values
    for m in CLASSICAL:
        e2 = (df[m].values - df["y"].values) ** 2
        g = pd.DataFrame(dict(day=dn, dc=dc, e2=e2)).groupby(["day", "dc"])["e2"].agg(["mean", "size"]).reset_index()
        g = g[g["size"] >= 10]
        for r in g.itertuples():
            rows.append(dict(year=fold, date=str(r.day.date()), day_class=r.dc, model=m, horizon_min=h * 5,
                             n=int(r.size), RMSE=float(np.sqrt(r.mean))))
    return rows


def perday_models(d: pd.DataFrame):
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    rows = []
    for h, g in d.groupby("horizon_min"):
        g = g.copy()
        g["day"] = g.year.astype(str) + "_" + g.date.astype(str)
        f = "RMSE ~ C(model)*C(day_class) + C(year)"
        ols = smf.ols(f, data=g).fit()
        a_ols = sm.stats.anova_lm(ols, typ=2)
        mix = smf.mixedlm(f, data=g, groups=g["day"]).fit(reml=True, method="lbfgs")
        var_day = float(np.asarray(mix.cov_re).ravel()[0])
        icc = var_day / (var_day + float(mix.scale))
        # the fixed-effect design is the same formula in both fits (mixed-model params put the
        # fixed effects first); newer statsmodels keep design_info only on the OLS data object
        dat = ols.model.data  # statsmodels < 0.15: design_info; 0.15: model_spec
        di = getattr(dat, "design_info", None) or getattr(dat, "model_spec")
        slices = di.term_name_slices
        assert list(mix.fe_params.index) == list(ols.params.index)
        clu = ols.get_robustcov_results(cov_type="cluster", groups=g["day"].astype("category").cat.codes)

        def wald(res, cols):
            R = np.zeros((len(cols), len(res.params)))
            for r_, c_ in enumerate(cols):
                R[r_, c_] = 1.0
            w = res.wald_test(R, use_f=False, scalar=True)
            return float(np.asarray(w.statistic).ravel()[0]), float(np.asarray(w.pvalue).ravel()[0])

        for term, label in [("C(model)", "model"), ("C(day_class)", "day_class"),
                            ("C(model):C(day_class)", "model:day_class"), ("C(year)", "year")]:
            if term not in a_ols.index or term not in slices:
                continue
            cols = list(range(slices[term].start, slices[term].stop))
            chi_m, p_m = wald(mix, cols)
            chi_c, p_c = wald(clu, cols)
            rows.append(dict(horizon_min=int(h), term=label,
                             level="within-day" if label in ("model", "model:day_class") else "between-day",
                             F_ols=round(float(a_ols.loc[term, "F"]), 4), p_ols=float(a_ols.loc[term, "PR(>F)"]),
                             chi2_mixed_day_RE=round(chi_m, 4), p_mixed_day_RE=p_m,
                             chi2_cluster_by_day=round(chi_c, 4), p_cluster_by_day=p_c,
                             n_obs=int(len(g)), n_days=int(g["day"].nunique()),
                             var_day=round(var_day, 3), var_resid=round(float(mix.scale), 3), icc_day=round(icc, 4)))
    return rows


# ------------------------------------------------------------------ self-tests
def selftest(verbose=True):
    import metrics as M
    res = {}

    def ok(name, cond, msg=""):
        res[name] = bool(cond)
        if verbose:
            print(f"  [{'PASS' if cond else 'FAIL'}] {name} {msg}", flush=True)

    rng = np.random.default_rng(0)
    y = rng.normal(500, 100, 3000); p1 = y + rng.normal(0, 30, 3000); p2 = y + rng.normal(0, 32, 3000)
    d = (y - p1) ** 2 - (y - p2) ** 2
    a = dm_test(d, 0, 1); b = M.diebold_mariano(y, p1, p2, h=1)
    ok("dm_lag0_h1_equals_R1_metrics_module", np.allclose(a, b, rtol=1e-10), f"({a[0]:.4f} vs {b[0]:.4f})")
    ok("dm_bartlett_variance_nonnegative", np.isfinite(dm_test(np.r_[np.ones(50), -np.ones(50)], 60, 3)[0]))
    # bootstrap: point equals the plain mean; iid width close to the normal theory width
    code = np.repeat(np.arange(300), 20)
    v = rng.normal(0, 1, len(code))
    bt = DayBoot(code, 300, 4000)
    pt, lo, hi, _ = bt.mean_ci(v, np.ones(len(v), bool))
    ok("boot_point_is_mean", np.isclose(pt, v.mean()))
    wth = 2 * 1.96 / np.sqrt(len(v))
    ok("boot_width_iid_matches_theory", 0.8 < (hi - lo) / wth < 1.2, f"(ratio {(hi - lo) / wth:.3f})")
    # day blocks: a shared day effect widens the interval relative to iid
    vd = v + np.repeat(rng.normal(0, 1, 300), 20)
    _, lo2, hi2, _ = bt.mean_ci(vd, np.ones(len(v), bool))
    ok("boot_respects_day_blocks", (hi2 - lo2) > 3 * (hi - lo), f"({(hi2 - lo2) / (hi - lo):.1f}x)")
    ea, eb = rng.normal(0, 10, len(code)), rng.normal(0, 11, len(code))
    r = bt.rmse_diff(ea, eb, np.ones(len(code), bool))
    ok("rmse_diff_point", np.isclose(r["d"], np.sqrt(np.mean(ea ** 2)) - np.sqrt(np.mean(eb ** 2)))
       and r["d_ci"][0] < r["d"] < r["d_ci"][1])
    ok("holm_known_example", np.allclose(holm([0.01, 0.04, 0.03, 0.5]), [0.04, 0.09, 0.09, 0.5]))
    # point_tests end to end
    idx = pd.date_range("2024-01-01 07:00", periods=len(code), freq="5min")
    idx = pd.DatetimeIndex([pd.Timestamp("2024-01-01") + pd.Timedelta(days=int(c)) + pd.Timedelta(minutes=5 * (i % 20))
                            for i, c in enumerate(code)])
    df = pd.DataFrame(dict(y=500 + 0 * ea, gbm=500 + ea, persistence=500 + 3 * eb, g="clear"), index=idx)
    rows = point_tests(df, "syn", 2024, 3, 2000)
    ok("point_tests_detect_large_difference", rows[0]["d_rmse"] < 0 and rows[0]["bootstrap_excludes_zero"]
       and rows[0]["dm_p_hac_day"] < 1e-6)
    # mixed model on synthetic per-day data with a model effect
    recs = []
    for dday in range(120):
        de = rng.normal(0, 10)
        for m, eff in (("gbm", 0.0), ("persistence", 8.0), ("smart_persistence", 4.0)):
            recs.append(dict(year=2020 + dday % 3, date=f"d{dday}", day_class=["clear", "mixed"][dday % 2], model=m,
                             horizon_min=5, n=50, RMSE=50 + eff + de + rng.normal(0, 2)))
    try:
        mm = pd.DataFrame(perday_models(pd.DataFrame(recs)))
        mrow = mm[mm.term == "model"].iloc[0]
        ok("mixed_model_finds_model_effect", mrow.p_mixed_day_RE < 1e-6 and 0.5 < mrow.icc_day < 1,
           f"(ICC {mrow.icc_day:.2f})")
    except ModuleNotFoundError:
        ok("mixed_model_finds_model_effect", False, "(statsmodels missing)")
    return res


# ------------------------------------------------------------------ stage
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="S7")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--B", type=int, default=B_DEF)
    ap.add_argument("--s1", default=None)
    ap.add_argument("--s4", default=None)
    ap.add_argument("--s5", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--allow-unverified", action="store_true", help="test run on unverified outputs only")
    a = ap.parse_args()
    if a.selftest:
        r = selftest()
        sys.exit(0 if all(r.values()) else 1)
    import config as C
    t0 = time.time()
    R = C.RESULTS / "ce1"
    s1 = Path(a.s1) if a.s1 else R / "S1"
    s4 = Path(a.s4) if a.s4 else R / "S4"
    s5 = Path(a.s5) if a.s5 else R / "S5"
    out = Path(a.out) if a.out else R / (a.stage + ("_smoke" if a.smoke else ""))
    runs = R / "_runs" if not a.out else out / "_runs"
    out.mkdir(parents=True, exist_ok=True); runs.mkdir(parents=True, exist_ok=True)
    B = 500 if a.smoke else a.B
    checks = {}

    def check(name, cond, msg=""):
        checks[name] = dict(ok=bool(cond), msg=msg)
        log(f"{'PASS' if cond else 'FAIL'}  {name}  {msg}")

    st = selftest(verbose=False)
    check("selftests", all(st.values()), f"({sum(st.values())}/{len(st)}"
          + ("" if all(st.values()) else "; failed " + ", ".join(k for k, v in st.items() if not v)) + ")")
    if not a.allow_unverified:
        for s in ("S4", "S5"):
            stamp = R / "_runs" / f"{s}.json"
            v = stamp.exists() and json.load(open(stamp)).get("ok", False)
            check(f"input_stage_{s}_verified", v)
    inputs = {}
    # (a)
    pt_rows = []
    pfiles = sorted(s4.glob("*_point.parquet"))
    if a.smoke:
        pfiles = pfiles[:1]
    for pf in pfiles:
        site, f, hh = pf.stem.replace("_point", "").split("_")
        fold, h = int(f[1:]), int(hh[1:])
        inputs[pf.name] = md5(pf)
        df = pd.read_parquet(pf)
        for dp in sorted(s4.glob(pf.name.replace("_point.parquet", "_deep_*.parquet"))):
            inputs[dp.name] = md5(dp)
            b = dp.stem.split("_")[-1]
            dd = pd.read_parquet(dp)
            for c in dd.columns:
                df[f"{c}_{b}"] = dd[c].values
        pt_rows += point_tests(df, site, fold, h, B)
        log(f"(a) {pf.stem}: {len(df.columns) - 4} comparisons")
    P = pd.DataFrame(pt_rows)
    P.to_csv(out / "point_tests.csv", index=False)
    # (b)
    rs, rc = [], []
    zfiles = sorted(s5.glob("*_f2024_h*_obs.npz"))
    if a.smoke:
        zfiles = zfiles[:1]
    for zf in zfiles:
        site, f, hh = zf.stem.replace("_obs", "").split("_")
        inputs[zf.name] = md5(zf)
        z = np.load(zf)
        x, y = interval_stats(z, site, int(f[1:]), int(hh[1:]) * 5, B)
        rs += x; rc += y
        log(f"(b) {zf.stem}: {len(x)} CRPS rows, {len(y)} coverage rows")
    pd.DataFrame(rs).to_csv(out / "interval_crps.csv", index=False)
    pd.DataFrame(rc).to_csv(out / "coverage_ci.csv", index=False)
    # (c)
    pdr = []
    geo = pd.read_parquet(s1 / "yulara_geo_5min.parquet")
    inputs["yulara_geo_5min.parquet"] = md5(s1 / "yulara_geo_5min.parquet")
    yfiles = sorted(s4.glob("yulara_f*_point.parquet"))
    dcs = {}
    for pf in yfiles:
        _, f, hh = pf.stem.replace("_point", "").split("_")
        fold, h = int(f[1:]), int(hh[1:])
        if fold not in dcs:
            dcs[fold] = day_class_series(geo, fold)
        pdr += perday_records(pd.read_parquet(pf), dcs[fold], fold, h)
    PD = pd.DataFrame(pdr)
    PD.to_csv(out / "perday_rmse.csv", index=False)
    MM = pd.DataFrame(perday_models(PD)) if len(PD) and PD.year.nunique() > 1 else pd.DataFrame()
    MM.to_csv(out / "perday_models.csv", index=False)
    log(f"(c) {len(PD)} per-day records, {len(MM)} model terms")
    # checks
    check("point_tests_every_file", len(pfiles) > 0 and len(P) >= len(pfiles) * 3, f"({len(pfiles)} files, {len(P)} rows)")
    check("interval_stats_every_file", len(zfiles) > 0 and len(rs) > 0 and len(rc) > 0, f"({len(zfiles)} files)")
    check("finite_point_tests", bool(len(P)) and np.isfinite(P[["d_rmse", "d_rmse_ci_lo", "d_rmse_ci_hi"]].values).all())
    check("perday_models_fitted", a.smoke or (len(MM) > 0), f"({len(MM)} rows)")
    ok = all(c["ok"] for c in checks.values())
    detail = dict(stage=a.stage, smoke=a.smoke, ok=ok, B=B, seconds=round(time.time() - t0, 1), checks=checks,
                  outputs={p.name: md5(p) for p in sorted(out.glob("*")) if p.is_file()}, inputs=inputs,
                  code={p: md5(HERE / p) for p in ("ce1_stats.py", "ce1_causal.py", "../evaluation/metrics.py")},
                  versions=dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__,
                                statsmodels=__import__("statsmodels").__version__), host=platform.node())
    json.dump(detail, open(runs / f"{a.stage}{'_smoke' if a.smoke else ''}.detail.json", "w"), indent=1)
    log(f"{'ALL CHECKS PASSED' if ok else 'SOME CHECKS FAILED'} in {time.time() - t0:.0f}s")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
