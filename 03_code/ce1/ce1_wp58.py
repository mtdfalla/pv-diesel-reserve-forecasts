"""One-sided exceedance of the lower bounds the decision uses, by regime, and the complete-history
persistence ensemble as probabilistic reference (rule D-112).

Usage (from the repository root unless the path starts at ce1/, then from 03_code):
    python3 03_code/ce1/ce1_wp58.py [--selftest]
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
import ce1_folds as FC  # noqa: E402

FOLDS = [2019, 2020, 2021, 2022, 2023, 2024]
HORIZONS = [1, 3, 6, 12]
H_TAIL = 3
POLS = ["icp", "mondrian", "cqr", "mondrian_cqr", "raw_q"]
THETAS = [0.95, 0.96, 0.97, 0.98, 0.99, 0.995]
REGS = ["clear", "transitional", "cloudy"]
MIN_ROWS = 500  # D-112: regimes with at least 500 test rows
RATIO_LIMIT = 2.0
FOLDS_NEEDED = 4
PROXY_TAUS = [0.025, 0.05, 0.10, 0.5, 0.90, 0.95, 0.975]
KT_MAX = 1.5
S5_METHODS = ["icp", "icp_norm", "mondrian", "cqr", "mondrian_cqr", "raw_q", "tod", "nicp_var"]
REG_CODE = None  # filled from the data (S5 stores int codes)


def md5(p: Path) -> str:
    return hashlib.md5(Path(p).read_bytes()).hexdigest()


def log(m):
    print(time.strftime("%H:%M:%S"), m, flush=True)


# ---------------------------------------------------------------- 1. exceedance
def exceed_table(calib, test, levels_by_pol, g_it, g_tt):
    """Rows: policy, theta, kind (grid / A_R_main / A_R_shapeB), scope, regime, n, exceed."""
    out = []
    y = test["y"]
    for pol, lv in levels_by_pol.items():
        for theta, kind in lv:
            if theta >= 1.0:  # the limit level counts no PV: bound 0
                b = np.zeros(len(y))
            else:
                b = D.counted_ghi(pol, theta, calib, test)
            e = y < b
            out.append(dict(policy=pol, theta=theta, kind=kind, scope="all", regime="all",
                            n=len(y), exceed=float(e.mean())))
            for scope, gg in (("issue", g_it), ("target", g_tt)):
                for r in REGS:
                    m = gg == r
                    out.append(dict(policy=pol, theta=theta, kind=kind, scope=scope, regime=r,
                                    n=int(m.sum()), exceed=float(e[m].mean()) if m.any() else np.nan))
    return out


def regime_ratio(rows_pol):
    """Largest / smallest exceedance across issue-time regimes with >= MIN_ROWS rows."""
    v = [r["exceed"] for r in rows_pol if r["scope"] == "issue" and r["n"] >= MIN_ROWS]
    if len(v) < 2:
        return np.nan
    lo, hi = min(v), max(v)
    if lo == 0:
        return np.inf if hi > 0 else 1.0
    return hi / lo


def apply_pr3(ex: pd.DataFrame) -> dict:
    a = ex[ex.kind == "A_R_main"]
    per = {}
    for pol in ("icp", "mondrian", "mondrian_cqr", "cqr", "raw_q"):
        per[pol] = {}
        for yr in FOLDS:
            rr = a[(a.policy == pol) & (a.fold == yr)].to_dict("records")
            per[pol][yr] = dict(ratio=regime_ratio(rr), theta=rr[0]["theta"] if rr else None)
    cnt = lambda pol, cond: sum(1 for yr in FOLDS if cond(per[pol][yr]["ratio"]))  # noqa: E731
    icp_gt = cnt("icp", lambda x: x > RATIO_LIMIT)
    mon_le = cnt("mondrian", lambda x: x <= RATIO_LIMIT)
    mcq_le = cnt("mondrian_cqr", lambda x: x <= RATIO_LIMIT)
    fired = icp_gt >= FOLDS_NEEDED and mon_le >= FOLDS_NEEDED and mcq_le >= FOLDS_NEEDED
    js = lambda x: None if x is None or (isinstance(x, float) and np.isnan(x)) else ("inf" if x == np.inf else x)  # noqa: E731
    return dict(rule="D-110 PR-3, operational D-112", icp_folds_ratio_gt_2=icp_gt,
                mondrian_folds_ratio_le_2=mon_le, mondrian_cqr_folds_ratio_le_2=mcq_le, fired=bool(fired),
                per_fold={p: {str(y): dict(ratio=js(v["ratio"]), theta=v["theta"]) for y, v in d.items()}
                          for p, d in per.items()})


# ---------------------------------------------------------------- 2. CH-PeEn
def chpeen_quantiles(geo: pd.DataFrame, scalar: float, test_year: int, target_ts, gcs_target):
    """Quantiles (dict tau -> array) of the complete-history persistence ensemble at the
    target times. History: every geo row before 1 January of the test year.
    """
    hist = geo[geo.index < pd.Timestamp(f"{test_year}-01-01")]
    cs = hist["ghi_cs_raw"].values * scalar
    ok = (cs > 0) & np.isfinite(hist["ghi"].values)
    kt = np.clip(hist["ghi"].values[ok] / cs[ok], 0.0, KT_MAX)
    ts = hist.index[ok]
    slot = (ts.hour * 60 + ts.minute) // 5
    qs = np.full((288, len(PROXY_TAUS)), np.nan)
    n_mem = np.zeros(288, int)
    for s in range(288):
        v = kt[slot == s]
        n_mem[s] = len(v)
        if len(v):
            qs[s] = np.quantile(v, PROXY_TAUS)
    tt = pd.DatetimeIndex(target_ts)
    ts_slot = (tt.hour * 60 + tt.minute) // 5
    Q = qs[ts_slot] * np.asarray(gcs_target, float)[:, None]
    return {t: Q[:, i] for i, t in enumerate(PROXY_TAUS)}, n_mem[ts_slot], hist.index.max()


def crps_obs(y, qd):
    """ce1_conformal.crps_obs (seven-level WIS-form proxy), restated here so this module does
    not import the v1 conformal package; equality is checked against the S5 record.
    """
    taus = sorted(qd)
    return 2.0 * np.mean(np.column_stack([np.maximum(t * (y - qd[t]), (t - 1) * (y - qd[t]))
                                          for t in taus]), axis=1)


# ---------------------------------------------------------------- self-test
def selftest(verbose=True):
    res = {}

    def ok(name, cond, msg=""):
        res[name] = bool(cond)
        if verbose:
            print(f"  [{'PASS' if cond else 'FAIL'}] {name} {msg}")

    rng = np.random.default_rng(0)
    y = rng.uniform(0, 1000, 500)
    p = y + rng.normal(0, 50, 500)
    qd = {t: p for t in PROXY_TAUS}
    ok("proxy_degenerate_equals_2MAE", np.isclose(crps_obs(y, qd).mean(), np.mean(np.abs(y - p))),
       "(seven equal quantiles: 2 x mean pinball = mean |e|)")
    rows = [dict(scope="issue", n=1000, exceed=0.01), dict(scope="issue", n=1000, exceed=0.03),
            dict(scope="issue", n=100, exceed=0.5)]
    ok("ratio_ignores_small_regimes", regime_ratio(rows) == 3.0)
    rows0 = [dict(scope="issue", n=1000, exceed=0.0), dict(scope="issue", n=1000, exceed=0.01)]
    ok("ratio_zero_min_is_inf", regime_ratio(rows0) == np.inf)
    # the statistic is largest / smallest over the three issue-time regimes: a cloudy
    # regime with the largest exceedance drives it even when clear and transitional are balanced
    rows3 = [dict(scope="issue", n=1000, exceed=0.010), dict(scope="issue", n=1000, exceed=0.012),
             dict(scope="issue", n=600, exceed=0.030)]
    ok("ratio_over_three_regimes_cloudy_drives", np.isclose(regime_ratio(rows3), 3.0))
    # PR-3 counting on a constructed table
    recs = []
    for yr in FOLDS:
        for pol, ex in (("icp", (0.002, 0.02)), ("mondrian", (0.01, 0.012)), ("mondrian_cqr", (0.01, 0.015)),
                        ("cqr", (0.01, 0.01)), ("raw_q", (0.01, 0.01))):
            for r, e in zip(("clear", "transitional"), ex):
                recs.append(dict(fold=yr, policy=pol, kind="A_R_main", theta=0.99, scope="issue", regime=r,
                                 n=1000, exceed=e))
    pr = apply_pr3(pd.DataFrame(recs))
    ok("pr3_fires_on_constructed_case", pr["fired"] and pr["icp_folds_ratio_gt_2"] == 6)
    recs2 = [dict(r, exceed=0.01) if r["policy"] == "icp" else r for r in recs]
    ok("pr3_silent_without_icp_spread", not apply_pr3(pd.DataFrame(recs2))["fired"])
    return res


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    st = selftest()
    if a.selftest:
        sys.exit(0 if all(st.values()) else 1)
    root = Path(a.root)
    R = root / "04_results" / "ce1"
    out = R / "WP58"
    out.mkdir(parents=True, exist_ok=True)
    checks, inputs = {f"selftest_{k}": dict(ok=v, msg="") for k, v in st.items()}, {}

    def check(name, cond, msg=""):
        checks[name] = dict(ok=bool(cond), msg=msg)
        log(f"{'PASS' if cond else 'FAIL'}  {name}  {msg}")

    t0 = time.time()
    geo = pd.read_parquet(R / "S1" / "yulara_geo_5min.parquet")
    scal = json.load(open(R / "S1" / "scalars.json"))["yulara"]
    inputs["S1/yulara_geo_5min.parquet"] = md5(R / "S1" / "yulara_geo_5min.parquet")
    inputs["S1/scalars.json"] = md5(R / "S1" / "scalars.json")

    # regime codes of the S5 records, read once from fold 2019 at 15 min against the cache
    c0 = FC.load_cache(R / "S2" / f"yulara_f{FOLDS[0]}_h{H_TAIL}.parquet")
    t0c = c0[c0.split == "test"]
    z0 = np.load(R / "S5" / f"yulara_f{FOLDS[0]}_h{H_TAIL}_obs.npz")
    code_map = {r: int(np.unique(z0["reg_it"][(t0c["g"] == r).values])[0]) for r in REGS}
    check("s5_regime_code_map", len(set(code_map.values())) == 3, str(code_map))
    ex_rows, cal_rows, crps_rows = [], [], []
    for yr in FOLDS:
        for h in HORIZONS:
            s2 = R / "S2" / f"yulara_f{yr}_h{h}.parquet"
            s2x = R / "S2X" / f"yulara_f{yr}_h{h}.parquet"
            have_cache = h == H_TAIL
            if have_cache and not s2.exists():
                raise FileNotFoundError(s2)
            z5p = R / "S5" / f"yulara_f{yr}_h{h}_obs.npz"
            z5 = np.load(z5p)
            inputs[str(z5p.relative_to(R))] = md5(z5p)
            ts = pd.DatetimeIndex(pd.to_datetime(z5["ts_ns"]))
            tgt = ts + pd.Timedelta(minutes=5 * h)
            sc = float(scal[str(yr)]["scalar"])
            gcs_t = geo["ghi_cs_raw"].reindex(tgt).values * sc
            y5 = z5["y"].astype(float)
            if have_cache:
                inputs[str(s2.relative_to(R))] = md5(s2)
                c = FC.load_cache_ext(s2, s2x) if s2x.exists() else FC.load_cache(s2)
                ca, te = c[c.split == "calib"], c[c.split == "test"]
                check(f"s5_rows_equal_cache_f{yr}_h{h}", len(te) == len(ts) and (te.index == ts).all()
                      and np.allclose(te["y"].values.astype(np.float32), z5["y"]))
                check(f"gcs_equals_raw_x_scalar_f{yr}_h{h}", np.allclose(te["gcs"].values, gcs_t, rtol=1e-9))
                # 3. calibration rows and days per regime
                for r in REGS:
                    m = (ca["g"] == r).values
                    cal_rows.append(dict(fold=yr, h=h, regime=r, rows=int(m.sum()),
                                         days=int(pd.DatetimeIndex(ca.index[m]).normalize().nunique())))
                if h == H_TAIL:
                    cal, test = FC_as_dict(ca), FC_as_dict(te)
                    sel = json.load(open(R / "S6E" / f"f{yr}_h{h}_select.json"))["selections"]
                    inputs[f"S6E/f{yr}_h{h}_select.json"] = md5(R / "S6E" / f"f{yr}_h{h}_select.json")
                    lv = {}
                    for pol in POLS:
                        lst = [(t, "grid") for t in THETAS]
                        lst.append((float(sel["main"]["policies"][pol]["A_R"]["param"]), "A_R_main"))
                        lst.append((float(sel["shapeB"]["policies"][pol]["A_R"]["param"]), "A_R_shapeB"))
                        lv[pol] = lst
                    rows = exceed_table(cal, test, lv, te["g"].values, te["g_t"].values)
                    ex_rows += [dict(fold=yr, h=h, **r) for r in rows]
                    b1 = D.counted_ghi("mondrian_cqr", 0.97, cal, test)
                    tk = D.tau_key(0.97)
                    s_c = pd.Series(cal["q"][tk] - cal["y"])
                    adj = s_c.groupby(pd.Series(cal["g"])).quantile(0.97)
                    cnt = pd.Series(cal["g"]).value_counts()
                    adj = adj.where(cnt.reindex(adj.index) > 30, s_c.quantile(0.97))
                    b2 = np.clip(test["q"][tk] - pd.Series(test["g"]).map(adj).values, 0, None)
                    check(f"probe_bound_equals_eq4_f{yr}", np.isfinite(b1).all() and np.allclose(b1, b2, rtol=0, atol=1e-9),
                          float(np.nanmax(np.abs(b1 - b2))))
                    # probe: regime codes of S5 match the cache's regimes
                    codes = {}
                    for r in REGS:
                        m = (te["g"] == r).values
                        u = np.unique(z5["reg_it"][m])
                        codes[r] = u.tolist()
                    # the proxy function reproduces the S5 per-observation raw_q CRPS
                    qraw = {t: te[f"q{t:.6f}"].values.astype(float) for t in PROXY_TAUS if t != 0.5}
                    qraw[0.5] = te["p"].values.astype(float)
                    cr_raw = crps_obs(te["y"].values.astype(float), qraw)
                    check(f"proxy_reproduces_s5_raw_q_f{yr}",
                          np.allclose(cr_raw, z5["crps__raw_q|static"].astype(float), atol=2e-3, rtol=1e-5),
                          f"(max |d| {np.max(np.abs(cr_raw - z5['crps__raw_q|static'])):.2e})")
                    check(f"s5_regime_codes_consistent_f{yr}",
                          all(v == [code_map[r]] for r, v in codes.items()), str(codes))
            # 2. CH-PeEn on the S5 rows
            Q, nmem, last = chpeen_quantiles(geo, sc, yr, tgt, gcs_t)
            check(f"chpeen_history_before_test_f{yr}_h{h}", last < pd.Timestamp(f"{yr}-01-01"),
                  f"(last history row {last}, min members {int(nmem.min())})")
            cr_ref = crps_obs(y5, Q)
            reg_it = z5["reg_it"]
            scopes = {"all": np.ones(len(y5), bool)}
            for r in REGS:
                scopes[r] = reg_it == code_map[r]
            for sname, m in scopes.items():
                base = dict(fold=yr, h=h, horizon_min=5 * h, scope=sname, n=int(m.sum()),
                            CRPS_chpeen=float(cr_ref[m].mean()))
                for meth in S5_METHODS:
                    k = f"crps__{meth}|static"
                    if k in z5:
                        v = float(z5[k][m].astype(float).mean())
                        crps_rows.append(dict(base, method=meth, CRPS=v, skill=1 - v / base["CRPS_chpeen"]))
            log(f"fold {yr} h {h}: CH-PeEn CRPS {cr_ref.mean():.2f}  ({time.time() - t0:.0f} s)")
    ex = pd.DataFrame(ex_rows)
    ex.to_csv(out / "exceedance.csv", index=False)
    pd.DataFrame(cal_rows).to_csv(out / "calib_days.csv", index=False)
    cr = pd.DataFrame(crps_rows)
    cr.to_csv(out / "chpeen_crps.csv", index=False)
    # cross-check: the S5 aggregate CRPS equals the mean of its per-observation record
    s5 = pd.read_csv(R / "S5" / "crps.csv")
    s5 = s5[(s5.site == "yulara") & (s5.variant == "static") & (s5.scope == "all")]
    mm = cr[cr.scope == "all"].merge(s5, left_on=["fold", "horizon_min", "method"],
                                     right_on=["fold", "horizon_min", "method"], suffixes=("", "_s5"))
    check("s5_crps_aggregate_reproduced", len(mm) > 0 and np.allclose(mm["CRPS"], mm["CRPS_s5"], atol=6e-4),
          f"({len(mm)} rows, max |d| {np.max(np.abs(mm['CRPS'] - mm['CRPS_s5'])):.4f})")
    pr3 = apply_pr3(ex)
    json.dump(pr3, open(out / "pr3.json", "w"), indent=1)
    log(f"PR-3 fired: {pr3['fired']}  (ICP ratio > 2 in {pr3['icp_folds_ratio_gt_2']} folds, "
        f"Mondrian <= 2 in {pr3['mondrian_folds_ratio_le_2']}, Mondrian-CQR <= 2 in {pr3['mondrian_cqr_folds_ratio_le_2']})")
    outs = {p.name: md5(p) for p in sorted(out.glob("*")) if p.suffix in (".csv", ".json") and p.name != "checks.json"}
    ok_all = all(c["ok"] for c in checks.values())
    json.dump(dict(checks=checks, inputs=inputs, outputs=outs, code={"ce1_wp58.py": md5(Path(__file__))}),
              open(out / "checks.json", "w"), indent=1)
    (R / "_runs").mkdir(exist_ok=True)
    json.dump(dict(stage="WP58", ok=ok_all, finished=time.strftime("%Y-%m-%dT%H:%M:%S"),
                   seconds=round(time.time() - t0, 1), host="Linux",
                   env=dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__),
                   n_checks=len(checks), n_failed=sum(not c["ok"] for c in checks.values()),
                   outputs=outs, code={"ce1_wp58.py": md5(Path(__file__))}),
              open(R / "_runs" / "WP58.json", "w"), indent=1)
    log(f"{'ALL CHECKS PASS' if ok_all else 'CHECK FAILED'} ({len(checks)} checks)")
    sys.exit(0 if ok_all else 1)


def FC_as_dict(c: pd.DataFrame) -> dict:
    """Same dict as ce1_s6.as_dict (kept local so this module does not import pvlib)."""
    return dict(idx=c.index.values, y=c["y"].values.astype(float), p=c["p"].values.astype(float),
                g=c["g"].values, v=c["v"].values.astype(float), gcs=c["gcs"].values.astype(float),
                q={round(float(c_[1:]), 6): c[c_].values.astype(float) for c_ in c.columns if c_.startswith("q")})


if __name__ == "__main__":
    main()
