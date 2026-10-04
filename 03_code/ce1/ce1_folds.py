"""Stage S2: per-fold point and quantile caches for both sites.

For every site, fold and horizon, fits on the fold's training years only the GBM point model
and one GBM quantile model per level that the interval and decision layers need, and stores
their predictions for the calibration and test years. Folds: test year Y, calibration year
Y-1, training years up to Y-2. Yulara runs the folds 2019 to 2024, DKASC Alice Springs fold
2024. Horizons 1, 3, 6 and 12 steps (5, 15, 30, 60 min). HistGradientBoostingRegressor (150
iterations, learning rate 0.08, 31 leaves, patience 10, seed 42) on the clear-sky index, with
early stopping on the last 10% of the training rows in time order.

Usage (from the repository root unless the path starts at ce1/, then from 03_code):
    python ce1/ce1_folds.py --stage S2 [--smoke] [--only yulara:2024:3]
    python ce1/ce1_folds.py --selftest
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
import ce1_causal as K  # noqa: E402
import ce1_dispatch as D  # noqa: E402

HORIZONS = [1, 3, 6, 12]
TAUS_INTERVAL = [0.025, 0.05, 0.10, 0.90, 0.95, 0.975]
TAUS = sorted(set(D.taus_needed()) | set(TAUS_INTERVAL))
GBM = dict(max_iter=150, learning_rate=0.08, max_leaf_nodes=31, n_iter_no_change=10,
           random_state=42)
VAL_FRAC = 0.10
KT_MAX = K.KT_MAX
FOLDS = {"yulara": K.SITES["yulara"]["fold_tests"], "asp": K.SITES["asp"]["fold_tests"]}


def qcol(tau: float) -> str:
    return f"q{tau:.6f}"


def md5(p: Path) -> str:
    h = hashlib.md5()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def log(m):
    print(f"[S2] {m}", flush=True)


# ---------------------------------------------------------------- frames
def fold_frames(geo: pd.DataFrame, test_year: int, h: int):
    """The fold's supervised frame (identical rows to ce1_causal.fold_frame) plus v and g_t."""
    f = K.fold(test_year)
    cal, nband = K.fit_scalar(geo, f["train_max"])
    base = K.build_base(geo, cal, "locf6")
    d = K.make_xy(base, h).sort_index()
    v = base["kt_in"].rolling(K.VAR_WIN, min_periods=K.VAR_WIN // 2).std()
    g_t = base["regime"].shift(-h)
    d = d.assign(v=v.reindex(d.index).values, g_t=g_t.reindex(d.index).values)
    tr, ca, te = K.split(d, f)
    return dict(fold=f, scalar=cal, n_band=nband, train=tr, calib=ca, test=te)


def chrono_split(tr: pd.DataFrame, frac=VAL_FRAC):
    """Fit rows and the chronological early-stopping block (last frac of rows in time order)."""
    assert tr.index.is_monotonic_increasing
    cut = int(round(len(tr) * (1 - frac)))
    return tr.iloc[:cut], tr.iloc[cut:]


def gbm(tau=None, max_iter=None):
    from sklearn.ensemble import HistGradientBoostingRegressor as HGB
    kw = dict(GBM)
    if max_iter:
        kw["max_iter"] = max_iter
    if tau is not None:
        kw.update(loss="quantile", quantile=float(tau))
    return HGB(early_stopping=True, **kw)


def to_ghi(kt, cs):
    return np.clip(np.clip(kt, 0, KT_MAX) * cs, 0, None)


def fit_cache(fr: dict, taus=TAUS, max_iter=None, sub=None, seed=0, point=True):
    """Fit point and quantile models on the fold's training rows; return the cache frame
    (calibration and test rows) and the fit record.
    """
    tr = fr["train"]
    if sub:
        rng = np.random.default_rng(seed)
        tr = tr.iloc[np.sort(rng.choice(len(tr), int(len(tr) * sub), replace=False))]
    fit, val = chrono_split(tr)
    F = K.FEATURES
    Xf, yf, Xv, yv = fit[F].values, fit["y_kt"].values, val[F].values, val["y_kt"].values
    ev = pd.concat([fr["calib"].assign(split="calib"), fr["test"].assign(split="test")])
    Xe, cse = ev[F].values, ev["y_ghi_cs"].values
    out = pd.DataFrame(index=ev.index)
    out.index.name = "ts"
    out["split"] = ev["split"].values
    out["y"] = ev["y_ghi"].values
    out["y_kt"] = ev["y_kt"].values
    out["gcs"] = cse
    out["g"] = ev["base_regime"].values
    out["g_t"] = ev["g_t"].values
    out["v"] = ev["v"].values
    out["in_imp"] = ev["in_imp"].values
    iters = {}
    t0 = time.time()
    if point:
        m = gbm(max_iter=max_iter).fit(Xf, yf, X_val=Xv, y_val=yv)
        out["p"] = to_ghi(m.predict(Xe), cse)
        iters["point"] = int(m.n_iter_)
    else:  # [D-092] S2X: p is taken from S2
        out["p"] = np.nan
    for tau in taus:
        m = gbm(tau, max_iter=max_iter).fit(Xf, yf, X_val=Xv, y_val=yv)
        out[qcol(tau)] = to_ghi(m.predict(Xe), cse)
        iters[qcol(tau)] = int(m.n_iter_)
    Q = out[[qcol(t) for t in taus]].values.astype(float)
    dq = np.diff(Q, axis=1)  # adjacent levels, ascending tau
    worst = np.maximum(-dq.min(axis=1), 0.0)  # largest adjacent crossing, W/m2
    te = out["split"] == "test"
    lo, hi = qcol(0.05), qcol(0.95)
    crossing = dict(share_adjacent_pairs_crossed_by_more_than_1Wm2=float((dq < -1.0).mean()),
                    adjacent_crossing_Wm2_p50=float(np.median(worst)),
                    adjacent_crossing_Wm2_p95=float(np.quantile(worst, 0.95)))
    if lo in out and hi in out:  # S2X (D-092) fits neither level
        crossing = dict(share_rows_q05_above_q95=float((out[lo] > out[hi]).mean()), **crossing)
    rec = dict(n_train=int(len(tr)), n_fit=int(len(fit)), n_val=int(len(val)),
               val_first=str(val.index[0]), fit_last=str(fit.index[-1]),
               n_calib=int((out["split"] == "calib").sum()), n_test=int(te.sum()),
               scalar=fr["scalar"], n_band=fr["n_band"], iterations=iters,
               test_rmse_point=float(np.sqrt(np.mean((out.loc[te, "y"] - out.loc[te, "p"]) ** 2))),
               crossing=crossing, seconds=round(time.time() - t0, 1),
               taus=list(map(float, taus)), gbm=dict(GBM, max_iter=max_iter or GBM["max_iter"]),
               val_frac=VAL_FRAC, subsample=sub)
    return out, rec


def settings_key(taus, max_iter, sub):
    return dict(taus=list(map(float, taus)), gbm=dict(GBM, max_iter=max_iter or GBM["max_iter"]),
                val_frac=VAL_FRAC, subsample=sub)


def encode_q(out: pd.DataFrame) -> pd.DataFrame:
    qc = sorted([c for c in out.columns if c.startswith("q")], key=lambda c: float(c[1:]))
    k = np.round(out[qc].values.astype(float) * 100.0).astype(np.int64)
    k[:, 1:] = np.diff(k, axis=1)
    assert np.abs(k).max() < 2**31
    enc = out.drop(columns=qc).copy()
    for i, c in enumerate(qc):
        enc["d" + c] = k[:, i].astype(np.int32)
    return enc


def decode_q(enc: pd.DataFrame) -> pd.DataFrame:
    dc = sorted([c for c in enc.columns if c.startswith("dq")], key=lambda c: float(c[2:]))
    k = np.cumsum(enc[dc].values.astype(np.int64), axis=1)
    out = enc.drop(columns=dc).copy()
    for i, c in enumerate(dc):
        out[c[1:]] = k[:, i] / 100.0
    return out


def quantized(out: pd.DataFrame) -> pd.DataFrame:
    """What the cache holds: out with quantile columns rounded to 0.01 W/m2 (float64)."""
    o = out.copy()
    for c in [c for c in o.columns if c.startswith("q")]:
        o[c] = np.round(o[c].values.astype(float) * 100.0) / 100.0
    return o


def write(out: pd.DataFrame, rec: dict, path: Path):
    import pyarrow as pa
    import pyarrow.parquet as pq
    t = pa.Table.from_pandas(encode_q(out), preserve_index=True)
    pq.write_table(t, path, compression="zstd", compression_level=9)
    json.dump(dict(rec, storage="quantiles as int32 hundredths of W/m2, delta-coded across tau"),
              open(path.with_suffix(".json"), "w"), indent=1)


def load_cache(path: Path) -> pd.DataFrame:
    """Cache frame with quantile columns decoded to float64 GHI (W/m2, 0.01 resolution)."""
    return decode_q(pd.read_parquet(path))


EXT_KEYS = ("split", "y")
P_TOL = 1e-6


def load_cache_ext(path: Path, path_x: Path) -> pd.DataFrame:
    """D-092: the S2 cache joined with the extra quantile columns of stage S2X (same site,
    fold and horizon). Rows, split, measured GHI and point forecast must be identical;
    the extra file contributes only quantile columns S2 does not have.
    """
    a, b = load_cache(path), load_cache(path_x)
    assert a.index.equals(b.index), f"S2X rows differ from S2 ({path_x.name})"
    for k in EXT_KEYS:
        assert np.array_equal(a[k].values, b[k].values, equal_nan=k != "split"), \
            f"S2X column {k} differs from S2 ({path_x.name})"
    assert np.allclose(a["p"].values, b["p"].values, rtol=0, atol=P_TOL), f"S2X point forecast differs ({path_x.name})"
    qa = {c for c in a.columns if c.startswith("q")}
    qb = [c for c in b.columns if c.startswith("q")]
    assert qb and not (qa & set(qb)), "S2X must add only new quantile levels"
    out = pd.concat([a, b[qb]], axis=1)
    qc = sorted([c for c in out.columns if c.startswith("q")], key=lambda c: float(c[1:]))
    return out[[c for c in out.columns if not c.startswith("q")] + qc]


# ---------------------------------------------------------------- self-tests
def selftest(geo_real: pd.DataFrame | None = None, verbose=True) -> dict:
    res = {}

    def rep(name, ok, msg=""):
        res[name] = bool(ok)
        if verbose:
            print(f"{'PASS' if ok else 'FAIL'}  {name}  {msg}", flush=True)

    rep("taus_cover_dispatch_and_interval",
        set(D.taus_needed()) <= set(TAUS) and set(TAUS_INTERVAL) <= set(TAUS)
        and len(TAUS) == len(set(qcol(t) for t in TAUS)), f"({len(TAUS)} levels)")
    keys = {round(float(qcol(t)[1:]), 6) for t in TAUS}
    rep("dispatch_tau_keys_resolve_from_column_names",
        {D.tau_key(t) for t in D.theta_set(D.RATIOS)} <= keys,
        "(fractile levels such as 1/11 keep 6 decimals; test run caught 4-decimal names)")
    if geo_real is None:
        geo = K._synthetic_geo(n_days=40)
        geo.index = pd.date_range("2021-11-01", periods=len(geo), freq="5min")
        yr = 2022
    else:
        geo = geo_real
        yr = 2024
    h = 3
    if geo_real is not None:
        fr = fold_frames(geo, yr, h)
        ref = K.fold_frame(geo, yr, h)
        same = all(fr[s].index.equals(ref[s].index) and np.array_equal(fr[s]["y_ghi"].values,
                   ref[s]["y_ghi"].values) for s in ("train", "calib", "test"))
        rep("rows_equal_ce1_causal_fold_frame", same and fr["scalar"] == ref["scalar"],
            f"(train {len(fr['train'])}, calib {len(fr['calib'])}, test {len(fr['test'])})")
        # v: trailing, causal. Truncate the record after a sample of issue times and recompute.
        base_full = K.build_base(geo, fr["scalar"])
        v_full = base_full["kt_in"].rolling(K.VAR_WIN, min_periods=K.VAR_WIN // 2).std()
        rng = np.random.default_rng(1)
        cand = fr["test"].index[rng.choice(len(fr["test"]), 25, replace=False)]
        ok_c = True
        for t in cand:
            b = K.build_base(geo.loc[:t].iloc[-400:], fr["scalar"])
            vt = b["kt_in"].rolling(K.VAR_WIN, min_periods=K.VAR_WIN // 2).std().iloc[-1]
            ok_c &= bool(np.isclose(vt, fr["test"].loc[t, "v"], equal_nan=True)
                         and np.isclose(v_full.loc[t], vt, equal_nan=True))
        rep("v_is_causal_truncation", ok_c, "(25 test issue times, record cut at t)")
        rep("v_defined_on_every_row", all(fr[s]["v"].notna().all() for s in ("calib", "test")))
        gt = base_full["regime"].shift(-h)
        rep("g_t_is_regime_at_target", all((fr[s]["g_t"].values == gt.reindex(fr[s].index).values).all()
                                           for s in ("calib", "test")))
        tr = fr["train"]
    else:
        f = K.fold(yr)
        base = K.build_base(geo, 1.0)
        d = K.make_xy(base, h).sort_index()
        d = d.assign(v=base["kt_in"].rolling(12, min_periods=6).std().reindex(d.index).values,
                     g_t=base["regime"].shift(-h).reindex(d.index).values)
        tr, ca, te = d[d.index < "2021-12-01"], d[(d.index >= "2021-12-01") & (d.index < "2021-12-06")], \
            d[d.index >= "2021-12-06"]
        tr = tr.assign(year=2020); ca = ca.assign(year=2021); te = te.assign(year=2022)
        fr = dict(fold=f, scalar=1.0, n_band=0, train=tr, calib=ca, test=te)
    fit, val = chrono_split(tr)
    rep("early_stopping_block_is_chronological", fit.index.max() < val.index.min()
        and abs(len(val) - VAL_FRAC * len(tr)) <= 1, f"(fit ends {fit.index.max()}, block starts {val.index.min()})")
    taus = [0.05, 0.5, 0.95]
    out, rec = fit_cache(fr, taus=taus, max_iter=30, sub=0.2 if geo_real is not None else None)
    rep("columns_present", {"split", "y", "y_kt", "gcs", "p", "g", "g_t", "v", "in_imp"} <= set(out.columns)
        and all(qcol(t) in out.columns for t in taus))
    rep("ghi_nonnegative_and_bounded", (out["p"] >= 0).all()
        and (out["p"] <= KT_MAX * out["gcs"] + 1e-6).all()
        and all((out[qcol(t)] >= 0).all() for t in taus))
    rep("y_is_measured_target", np.allclose(out["y"].values, np.r_[fr["calib"]["y_ghi"].values,
                                                                    fr["test"]["y_ghi"].values]))
    # Refit determinism: the same data give the same predictions
    out2, _ = fit_cache(fr, taus=taus, max_iter=30, sub=0.2 if geo_real is not None else None)
    rep("refit_is_deterministic", np.array_equal(out["p"].values, out2["p"].values)
        and np.array_equal(out[qcol(0.05)].values, out2[qcol(0.05)].values))
    # Calibration/test rows never enter the fit: perturbing their targets changes nothing
    fr3 = dict(fr, calib=fr["calib"].assign(y_kt=fr["calib"]["y_kt"] * 0.5),
               test=fr["test"].assign(y_kt=0.0))
    out3, _ = fit_cache(fr3, taus=taus, max_iter=30, sub=0.2 if geo_real is not None else None)
    rep("no_calib_or_test_target_enters_fit", np.array_equal(out["p"].values, out3["p"].values))
    # Parquet round trip
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "x.parquet"
        write(out, rec, p)
        back = load_cache(p)
        qz = quantized(out)
        rep("parquet_round_trip_exact", back.index.equals(out.index)
            and list(back.columns) == list(qz.columns)
            and all(np.array_equal(back[c].values, qz[c].values) for c in qz.columns
                    if c not in ("g", "g_t", "split"))
            and (back["g"].values == out["g"].values).all()
            and float(np.max(np.abs(back[qcol(0.05)].values - out[qcol(0.05)].values))) <= 0.005 + 1e-9,
            "(quantiles held to 0.01 W/m2, everything else exact)")
    return res


# ---------------------------------------------------------------- stage
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="S2")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--only", default=None, help="site:year:h[,site:year:h] (test run)")
    ap.add_argument("--s1", default=None, help="S1 output folder (default 04_results/ce1/S1)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if a.selftest:
        r = selftest()
        sys.exit(0 if all(r.values()) else 1)
    import config as C
    t0 = time.time()
    s1 = Path(a.s1) if a.s1 else C.RESULTS / "ce1" / "S1"
    out = Path(a.out) if a.out else C.RESULTS / "ce1" / (a.stage + ("_smoke" if a.smoke else ""))
    runs = C.RESULTS / "ce1" / "_runs" if not a.out else out / "_runs"
    out.mkdir(parents=True, exist_ok=True); runs.mkdir(parents=True, exist_ok=True)
    checks = {}

    def check(name, cond, msg=""):
        checks[name] = dict(ok=bool(cond), msg=msg)
        log(f"{'PASS' if cond else 'FAIL'}  {name}  {msg}")

    scalars = json.load(open(s1 / "scalars.json"))
    geos = {s: pd.read_parquet(s1 / f"{s}_geo_5min.parquet") for s in FOLDS}
    st = selftest(geos["yulara"], verbose=False)
    check("selftests", all(st.values()), f"({sum(st.values())}/{len(st)}"
          + ("" if all(st.values()) else "; failed " + ", ".join(k for k, v in st.items() if not v)) + ")")
    jobs = [(s, y, h) for s in FOLDS for y in FOLDS[s] for h in HORIZONS]
    taus = TAUS
    if a.stage == "S2X":  # [D-092] extra levels, Yulara only
        taus = D.taus_low()
        jobs = [j for j in jobs if j[0] == "yulara"]
        s2_dir = C.RESULTS / "ce1" / ("S2_smoke" if a.smoke else "S2")
    if a.smoke:
        jobs = [("yulara", 2024, 3)]
    if a.only:
        jobs = [(p.split(":")[0], int(p.split(":")[1]), int(p.split(":")[2])) for p in a.only.split(",")]
    max_iter, sub = (40, 0.2) if a.smoke else (None, None)
    key = settings_key(taus, max_iter, sub)
    summary = []
    for site, yr, h in jobs:
        path = out / f"{site}_f{yr}_h{h}.parquet"
        if path.exists() and path.with_suffix(".json").exists():
            rec = json.load(open(path.with_suffix(".json")))
            if all(rec.get(k) == v for k, v in key.items()):
                log(f"{site} fold {yr} h={h}: cached, skipped")
                summary.append(dict(site=site, fold=yr, h=h, **{k: rec[k] for k in
                               ("n_calib", "n_test", "test_rmse_point", "seconds")}))
                continue
        fr = fold_frames(geos[site], yr, h)
        sc = scalars.get(site, {}).get(str(yr), {}).get("scalar")
        check(f"scalar_matches_S1_{site}_{yr}_h{h}", sc is not None and abs(sc - fr["scalar"]) < 1e-12,
              f"({fr['scalar']:.6f} vs S1 {sc})")
        cache, rec = fit_cache(fr, taus=taus, max_iter=max_iter, sub=sub, point=a.stage != "S2X")
        if a.stage == "S2X":
            base = load_cache(s2_dir / path.name)
            same = base.index.equals(cache.index) and all(
                np.array_equal(base[k].values, cache[k].values, equal_nan=k != "split") for k in EXT_KEYS)
            check(f"rows_and_y_equal_S2_{site}_{yr}_h{h}", same,
                  "(the fold's rows, split and measured GHI equal the stored S2 cache; p copied from S2)")
            ql = base[qcol(0.25)].values
            rec["crossing_q0.255_below_S2_q0.25_by_more_than_1Wm2"] = float((cache[qcol(0.255)].values
                                                                            < ql - 1.0).mean())
            if not same:
                continue
            cache["p"] = base["p"].values
            te_ = (cache["split"] == "test").values
            rec["test_rmse_point"] = float(np.sqrt(np.mean((cache["y"].values[te_] - cache["p"].values[te_]) ** 2)))
            rec["point"] = "copied from S2 (not refitted) [D-092]"
        write(cache, rec, path)
        log(f"{site} fold {yr} h={h}: calib {rec['n_calib']}, test {rec['n_test']}, point RMSE "
            f"{rec['test_rmse_point']:.2f} W/m2, q0.05 > q0.95 on {rec['crossing'].get('share_rows_q05_above_q95', float('nan')):.4f} of rows, "
            f"{rec['seconds']:.0f}s, {path.stat().st_size / 1e6:.1f} MB")
        summary.append(dict(site=site, fold=yr, h=h, **{k: rec[k] for k in
                       ("n_calib", "n_test", "test_rmse_point", "seconds")}))
    sm = pd.DataFrame(summary)
    sm.to_csv(out / "caches_summary.csv", index=False)
    done = {(r.site, r.fold, r.h) for r in sm.itertuples()}
    check("every_job_cached", all(j in done for j in jobs), f"({len(done)}/{len(jobs)})")
    check("rows_present", bool(len(sm)) and (sm["n_calib"] > 1000).all() and (sm["n_test"] > 1000).all()
          if not a.smoke else bool(len(sm)))
    check("point_rmse_plausible", bool(len(sm)) and sm["test_rmse_point"].between(20, 250).all(),
          f"(range {sm['test_rmse_point'].min():.1f}-{sm['test_rmse_point'].max():.1f} W/m2)")
    ok = all(c["ok"] for c in checks.values())
    detail = dict(stage=a.stage, smoke=a.smoke, ok=ok, seconds=round(time.time() - t0, 1), checks=checks,
                  jobs=len(jobs), outputs={p.name: md5(p) for p in sorted(out.glob("*")) if p.is_file()},
                  inputs={p.name: md5(p) for p in sorted(s1.glob("*_geo_5min.parquet"))} | {"scalars.json": md5(s1 / "scalars.json")},
                  code={p.name: md5(p) for p in (Path(__file__), HERE / "ce1_causal.py", HERE / "ce1_dispatch.py")},
                  versions=dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__,
                                sklearn=__import__("sklearn").__version__), host=platform.node())
    json.dump(detail, open(runs / f"{a.stage}{'_smoke' if a.smoke else ''}.detail.json", "w"), indent=1)
    log(f"{'ALL CHECKS PASSED' if ok else 'SOME CHECKS FAILED'} in {time.time() - t0:.0f}s")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
