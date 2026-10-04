"""Stage S4: point-forecast benchmark on the causal folds. Persistence, smart persistence, a
ridge autoregression and the GBM on every site, fold and horizon, and compact GRU and GRU-TCN
baselines on Yulara fold 2024 at two training budgets, all on the rows of the S2 cache of each
fold. Rules: D-089.

Usage (from the repository root unless the path starts at ce1/, then from 03_code):
    python ce1/ce1_point.py --stage S4 [--smoke] [--only yulara:2024:3] [--no-deep]
    python ce1/ce1_point.py --selftest
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
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
import ce1_causal as K  # noqa: E402
import metrics as M  # noqa: E402  (v1, unchanged)

HORIZONS = [1, 3, 6, 12]
FOLDS = {"yulara": [2019, 2020, 2021, 2022, 2023, 2024], "asp": [2024]}
CLASSICAL = ["persistence", "smart_persistence", "linear_ar", "gbm"]
DEEP_JOBS = [("yulara", 2024)]
BUDGETS = {"A": dict(epochs=15, subsample=150000), "C": dict(epochs=30, subsample=0)}
RIDGE_ALPHA = 1.0
W = 12
SEED = 42
REGIMES = ["clear", "transitional", "cloudy"]
SETTINGS = dict(classical=CLASSICAL, budgets=BUDGETS, ridge_alpha=RIDGE_ALPHA, window=W, seed=SEED,
                features=K.FEATURES, version=1)


def log(m):
    print(f"[S4] {m}", flush=True)


def md5(p: Path) -> str:
    h = hashlib.md5()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def to_ghi(kt, cs):
    return np.clip(np.clip(kt, 0, K.KT_MAX) * cs, 0, None)


# ------------------------------------------------------------------ frames
def fold_data(geo: pd.DataFrame, test_year: int, h: int):
    """Base frame and the fold's supervised split (rows identical to ce1_causal.fold_frame)."""
    f = K.fold(test_year)
    cal, nband = K.fit_scalar(geo, f["train_max"])
    base = K.build_base(geo, cal, "locf6")
    d = K.make_xy(base, h).sort_index()
    tr, ca, te = K.split(d, f)
    return dict(base=base, train=tr, calib=ca, test=te, scalar=cal)


def windows(base: pd.DataFrame, rows: pd.DatetimeIndex, w=W):
    """Issue-time windows kt_in(t-w+1..t) for the given issue times, shape (n, w, 1)."""
    pos = base.index.get_indexer(rows)
    assert (pos >= 0).all()
    kin = base["kt_in"].values.astype("float32")
    J = pos[:, None] + np.arange(-w + 1, 1)[None, :]
    assert (J >= 0).all()
    X = kin[J]
    return X[..., None]


# ------------------------------------------------------------------ models
def classical(fr: dict, gbm_p=None):
    tr, te = fr["train"], fr["test"]
    cs = te["y_ghi_cs"].values
    out = pd.DataFrame(index=te.index)
    out.index.name = "ts"
    out["y"] = te["y_ghi"].values
    out["g"] = te["base_regime"].values
    out["persistence"] = te["base_ghi"].values
    out["smart_persistence"] = to_ghi(te["base_kt"].values, cs)
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    ar = make_pipeline(StandardScaler(), Ridge(alpha=RIDGE_ALPHA)).fit(tr[K.FEATURES].values, tr["y_kt"].values)
    out["linear_ar"] = to_ghi(ar.predict(te[K.FEATURES].values), cs)
    if gbm_p is not None:
        out["gbm"] = gbm_p
    return out


def load_deep_module():
    src = CODE / "models" / "deep_gru_tcn.py"
    spec = importlib.util.spec_from_file_location("deep_gru_tcn_ref", src)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod, md5(src)


def deep(fr: dict, budget: dict, ref=None):
    """GRU and GRU-TCN predictions on the fold's test rows (GHI, W/m2) and a fit record."""
    import torch
    if ref is None:
        ref, _ = load_deep_module()
    torch.set_num_threads(max(1, os.cpu_count() or 2))
    Xtr = windows(fr["base"], fr["train"].index)
    ytr = fr["train"]["y_kt"].values.astype("float32")
    ok = ~np.isnan(Xtr).any(axis=(1, 2))
    n_nan = int((~ok).sum())
    Xtr, ytr = Xtr[ok], ytr[ok]
    if budget["subsample"] and len(Xtr) > budget["subsample"]:
        sel = np.random.RandomState(SEED).choice(len(Xtr), budget["subsample"], replace=False)
        Xtr, ytr = Xtr[sel], ytr[sel]
    Xte = windows(fr["base"], fr["test"].index)
    okt = ~np.isnan(Xte).any(axis=(1, 2))  # rows whose 12-step window is complete
    cs = fr["test"]["y_ghi_cs"].values
    out, rec = {}, dict(n_train=int(len(Xtr)), train_windows_dropped_nan=n_nan,
                        test_rows_without_complete_window=int((~okt).sum()), seconds={})
    for name, Net in (("gru", ref.GRUNet), ("gru_tcn", ref.GRUTCN)):
        t0 = time.time()
        torch.manual_seed(SEED)  # the network's initial weights
        pred = ref.train_eval(Net(), Xtr, ytr, Xte[okt], budget["epochs"], seed=SEED)
        full = np.full(len(Xte), np.nan)
        full[okt] = to_ghi(pred, cs[okt])
        out[name] = full
        rec["seconds"][name] = round(time.time() - t0, 1)
    return out, rec


def metrics_rows(df: pd.DataFrame, models, base: dict):
    """support 'own': every row where the model has a prediction (all S2 rows for the
    classical models; rows with a complete 12-step window for the deep models).
    support 'common': rows where every model of the job has a prediction.
    """
    rows = []
    models = [m for m in models if m in df]
    have = np.column_stack([np.isfinite(df[m].values) for m in models])
    common = have.all(axis=1)
    for scope in ["all"] + REGIMES:
        sm = np.ones(len(df), bool) if scope == "all" else (df["g"].values == scope)
        for support in ("own", "common"):
            if support == "common" and common.all():
                continue  # identical to 'own' for every model
            for j, mod in enumerate(models):
                m = sm & (have[:, j] if support == "own" else common)
                if m.sum() < 30:
                    continue
                y = df["y"].values[m]
                ref = df["smart_persistence"].values[m] if "smart_persistence" in df else None
                r = M.all_metrics(y, df[mod].values[m], ref)
                rows.append(dict(base, model=mod, scope=scope, support=support,
                                 **{k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}))
    return rows


# ------------------------------------------------------------------ self-tests
def selftest(geo_real=None, verbose=True):
    res = {}

    def ok(name, cond, msg=""):
        res[name] = bool(cond)
        if verbose:
            print(f"  [{'PASS' if cond else 'FAIL'}] {name} {msg}", flush=True)

    if geo_real is None:
        geo = K._synthetic_geo(n_days=40)
        geo.index = pd.date_range("2021-11-01", periods=len(geo), freq="5min")
        geo = geo.copy()
        base = K.build_base(geo, 1.0, "locf6")
        d = K.make_xy(base, 3).sort_index()
        fr = dict(base=base, train=d[d.index < "2021-12-01"], calib=d.iloc[:0], test=d[d.index >= "2021-12-01"],
                  scalar=1.0)
    else:
        geo = geo_real
        fr = fold_data(geo_real, 2024, 3)
        base = fr["base"]
    te = fr["test"]
    # 1. windows are causal and agree with the GBM lag features
    X = windows(base, te.index)
    agree = all(np.allclose(X[:, W - 1 - L, 0], te[f"kt_l{L}"].values, equal_nan=True)
                for L in K.LAGS if L < W)
    ok("window_matches_lag_features", agree, "(lags 0-6 and 9)")
    trunc_ok = True
    rng = np.random.default_rng(0)
    for t in te.index[rng.choice(len(te), 10, replace=False)]:
        b2 = K.build_base(geo.loc[:t].iloc[-400:], fr["scalar"], "locf6")  # record cut at t
        x2 = windows(b2, pd.DatetimeIndex([t]))
        trunc_ok &= bool(np.allclose(x2[0], X[te.index.get_loc(t)], equal_nan=True))
    ok("window_uses_no_value_after_issue_time", trunc_ok, "(record cut at t and rebuilt, 10 issue times)")
    inc = np.isnan(X).any(axis=(1, 2)).mean()
    ok("incomplete_windows_rare", inc < 0.005, f"(share {inc:.5f}: an input run beyond the D-066 limit inside t-11..t)")
    # 2. persistence definitions
    cl = classical(fr)
    ok("smart_persistence_definition", np.allclose(cl["smart_persistence"].values,
       np.clip(np.clip(te["kt_in"].values, 0, 1.5) * te["y_ghi_cs"].values, 0, None)))
    meas = te["imp"].values == 0
    ok("persistence_is_measured_ghi_when_not_imputed",
       np.allclose(cl["persistence"].values[meas], te["kt"].values[meas] * te["ghi_cs"].values[meas]))
    ok("ridge_deterministic", np.array_equal(classical(fr)["linear_ar"].values, cl["linear_ar"].values))
    # 3. test targets never enter the ridge fit
    fr2 = dict(fr, test=fr["test"].assign(y_kt=0.0, y_ghi=0.0))
    ok("no_test_target_in_ridge", np.array_equal(classical(fr2)["linear_ar"].values, cl["linear_ar"].values))
    # 4. metrics identical to the v1 module
    cl["gbm"] = cl["linear_ar"] * 1.01
    mr = metrics_rows(cl, CLASSICAL, dict(site="x"))
    r = [x for x in mr if x["model"] == "linear_ar" and x["scope"] == "all"][0]
    ok("metrics_equal_R1_module", np.isclose(r["RMSE"], round(M.rmse(cl["y"].values, cl["linear_ar"].values), 4)))
    # 5. deep models: deterministic under the seed (tiny budget)
    try:
        import torch  # noqa: F401
        small = dict(fr, train=fr["train"].iloc[:3000], test=fr["test"].iloc[:500])
        a1, _ = deep(small, dict(epochs=1, subsample=0))
        a2, _ = deep(small, dict(epochs=1, subsample=0))
        ok("deep_deterministic_under_seed", all(np.array_equal(a1[k], a2[k]) for k in a1))
        ok("deep_outputs_bounded", all((a1[k] >= 0).all() and (a1[k] <= 1.5 * small["test"]["y_ghi_cs"].values + 1e-3).all()
                                       for k in a1))
    except ModuleNotFoundError:
        ok("deep_deterministic_under_seed", False, "(torch missing)")
    return res


# ------------------------------------------------------------------ stage
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="S4")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--only", default=None)
    ap.add_argument("--no-deep", action="store_true")
    ap.add_argument("--s1", default=None)
    ap.add_argument("--s2", default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if a.selftest:
        r = selftest()
        sys.exit(0 if all(r.values()) else 1)
    import config as C
    import ce1_folds as F
    t0 = time.time()
    s1 = Path(a.s1) if a.s1 else C.RESULTS / "ce1" / "S1"
    s2 = Path(a.s2) if a.s2 else C.RESULTS / "ce1" / "S2"
    out = Path(a.out) if a.out else C.RESULTS / "ce1" / (a.stage + ("_smoke" if a.smoke else ""))
    runs = C.RESULTS / "ce1" / "_runs" if not a.out else out / "_runs"
    out.mkdir(parents=True, exist_ok=True); runs.mkdir(parents=True, exist_ok=True)
    checks = {}

    def check(name, cond, msg=""):
        checks[name] = dict(ok=bool(cond), msg=msg)
        log(f"{'PASS' if cond else 'FAIL'}  {name}  {msg}")

    geos = {s: pd.read_parquet(s1 / f"{s}_geo_5min.parquet") for s in FOLDS}
    st = selftest(geos["yulara"], verbose=False)
    check("selftests", all(st.values()), f"({sum(st.values())}/{len(st)}"
          + ("" if all(st.values()) else "; failed " + ", ".join(k for k, v in st.items() if not v)) + ")")
    jobs = [(s, y, h) for s in FOLDS for y in FOLDS[s] for h in HORIZONS]
    budgets = dict(BUDGETS)
    if a.smoke:
        jobs = [("yulara", 2024, 3)]
        budgets = {"A": dict(epochs=2, subsample=20000)}
    if a.only:
        jobs = [(p.split(":")[0], int(p.split(":")[1]), int(p.split(":")[2])) for p in a.only.split(",")]
    settings = dict(SETTINGS, budgets=budgets)
    ref, ref_md5 = (None, None)
    inputs, rows, fitrec = {}, [], []
    rows_equal, deep_done, nan_windows = [], [], []
    for site, yr, h in jobs:
        stem = f"{site}_f{yr}_h{h}"
        src = s2 / f"{stem}.parquet"
        inputs[src.name] = md5(src)
        cache = F.load_cache(src)
        ctest = cache[cache["split"] == "test"]
        fr = None
        pp = out / f"{stem}_point.parquet"
        pj = pp.with_suffix(".json")
        done = pp.exists() and pj.exists() and json.load(open(pj)).get("settings") == settings \
            and json.load(open(pj)).get("input_md5") == inputs[src.name]
        if not done:
            t1 = time.time()
            fr = fold_data(geos[site], yr, h)
            same = fr["test"].index.equals(ctest.index) and np.allclose(fr["test"]["y_ghi"].values, ctest["y"].values)
            cl = classical(fr, gbm_p=ctest["p"].values)
            cl["g_t"] = ctest["g_t"].values
            cl.to_parquet(pp)
            rec = dict(settings=settings, input_md5=inputs[src.name], rows_equal_cache=bool(same),
                       n_train=int(len(fr["train"])), n_test=int(len(fr["test"])), seconds=round(time.time() - t1, 1))
            json.dump(rec, open(pj, "w"), indent=1)
            log(f"{stem}: classical done ({rec['n_test']} test rows, {rec['seconds']:.0f}s)")
        rec = json.load(open(pj))
        rows_equal.append(rec["rows_equal_cache"])
        df = pd.read_parquet(pp)
        base = dict(site=site, fold=yr, horizon_min=h * 5)
        models = list(CLASSICAL)
        if (site, yr) in DEEP_JOBS and not a.no_deep:
            for b, cfg in budgets.items():
                dp = out / f"{stem}_deep_{b}.parquet"
                dj = dp.with_suffix(".json")
                ddone = dp.exists() and dj.exists() and json.load(open(dj)).get("budget") == cfg \
                    and json.load(open(dj)).get("input_md5") == inputs[src.name]
                if not ddone:
                    if ref is None:
                        ref, ref_md5 = load_deep_module()
                    if fr is None:
                        fr = fold_data(geos[site], yr, h)
                    t1 = time.time()
                    preds, drec = deep(fr, cfg, ref)
                    pd.DataFrame(preds, index=df.index).to_parquet(dp)
                    drec.update(budget=cfg, input_md5=inputs[src.name], module_md5=ref_md5,
                                torch=__import__("torch").__version__, total_seconds=round(time.time() - t1, 1))
                    json.dump(drec, open(dj, "w"), indent=1)
                    log(f"{stem}: deep budget {b} done ({drec['n_train']} windows, {drec['total_seconds']:.0f}s)")
                drec = json.load(open(dj))
                nan_windows.append(drec["train_windows_dropped_nan"] + drec["test_rows_without_complete_window"])
                dd = pd.read_parquet(dp)
                for k in dd.columns:
                    df[f"{k}_{b}"] = dd[k].values
                    models.append(f"{k}_{b}")
                deep_done.append((site, yr, h, b))
        rows += metrics_rows(df, models, base)
        fitrec.append(dict(base, n_test=rec["n_test"], n_train=rec["n_train"]))
    Mt = pd.DataFrame(rows)
    Mt.to_csv(out / "metrics.csv", index=False)
    check("every_classical_job_done", len(fitrec) == len(jobs), f"({len(fitrec)}/{len(jobs)})")
    check("rows_equal_S2_cache_every_job", all(rows_equal))
    want_deep = [(s, y, h, b) for (s, y, h) in jobs if (s, y) in DEEP_JOBS for b in budgets] if not a.no_deep else []
    check("every_deep_job_done", all(j in deep_done for j in want_deep), f"({len(deep_done)}/{len(want_deep)})")
    check("incomplete_windows_rare", sum(nan_windows) <= 0.005 * max(1, sum(r["n_test"] + r["n_train"] for r in fitrec
                                                                          if (r["site"], r["fold"]) in DEEP_JOBS)),
          f"(train and test rows without a complete 12-step window: {sum(nan_windows)})")
    al = Mt[(Mt.scope == "all") & (Mt.support == "own")] if len(Mt) else Mt
    gb = al[al.model == "gbm"]
    check("gbm_rmse_equals_S2_record", True if a.smoke else bool(len(gb)) and all(
        abs(r.RMSE - json.load(open(s2 / f"{r.site}_f{r.fold}_h{r.horizon_min // 5}.json"))["test_rmse_point"]) < 1e-3
        for r in gb.itertuples()), "(RMSE recomputed from the cache equals the S2 fit record)")
    check("rmse_plausible", bool(len(al)) and al.RMSE.between(15, 300).all(),
          f"(range {al.RMSE.min():.1f}-{al.RMSE.max():.1f} W/m2; diagnostic bound)")
    ok = all(c["ok"] for c in checks.values())
    detail = dict(stage=a.stage, smoke=a.smoke, ok=ok, seconds=round(time.time() - t0, 1), checks=checks,
                  jobs=len(jobs), settings=settings,
                  outputs={p.name: md5(p) for p in sorted(out.glob("*")) if p.is_file()},
                  inputs=inputs | {p.name: md5(p) for p in sorted(s1.glob("*_geo_5min.parquet"))},
                  code={p: md5(HERE / p) for p in ("ce1_point.py", "ce1_causal.py", "ce1_folds.py",
                                                   "../models/deep_gru_tcn.py", "../evaluation/metrics.py")},
                  versions=dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__,
                                sklearn=__import__("sklearn").__version__,
                                torch=(__import__("torch").__version__ if ref is not None else None)),
                  host=platform.node())
    json.dump(detail, open(runs / f"{a.stage}{'_smoke' if a.smoke else ''}.detail.json", "w"), indent=1)
    log(f"{'ALL CHECKS PASSED' if ok else 'SOME CHECKS FAILED'} in {time.time() - t0:.0f}s")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
