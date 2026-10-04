"""Runner stages S1E, S2E and S5E: the record extended to 2025, the item-14 refits, the DKASC
one-step lag and the caches and intervals of fold 2025 (rules D-129, D-131).

Usage (from the repository root unless the path starts at ce1/, then from 03_code):
    powershell -ExecutionPolicy Bypass -File "<repository folder>/03_code/ce1/ce1_run.ps1" -Stage S1E,S2E,S5E
    python ce1/ce1_wp67_pc.py --selftest
    python ce1/ce1_wp67_pc.py --stage S2E --only item14:2019:3 --out .
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import platform
import subprocess
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
import ce1_folds as F  # noqa: E402

NEW_YEAR = 2025
ITEM14_FOLDS = [2019, 2020, 2021, 2022, 2023, 2024]
ITEM14_H = 3
NEW_H = [1, 3]
GEO_COLS = ["ghi", "zenith", "ghi_cs_raw", "is_day"]
INPUT_COLS = ["split", "y", "y_kt", "gcs", "g", "g_t", "v", "in_imp"]
LAG_COLS = [f"kt_l{L}" for L in K.LAGS] + ["kt_rmean", "kt_rstd", "in_imp", "imp", "kt_in", "regime"]
CODE_FILES = ["ce1_wp67_pc.py", "ce1_causal.py", "ce1_sensor.py", "ce1_folds.py", "ce1_dispatch.py",
              "ce1_conformal.py", "ce1_aci.py", "../conformal/conformal.py",
              "../conformal/conformal_adaptive.py", "../r1/r1_conformal_delayed.py", "../utils/config.py"]


def md5(p: Path) -> str:
    h = hashlib.md5()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def log(stage, m):
    print(f"[{stage}] {m}", flush=True)


# ====================================================================== constructions
def years_with_new(site: str) -> list[int]:
    return list(K.SITES[site]["years"]) + [NEW_YEAR]


def lag_inputs(base: pd.DataFrame) -> pd.DataFrame:
    """R5: every input-derived column at t takes its value at t - 5 min. The base frame is
    on the regular 5-min grid (asserted by build_base), so one row is one step.
    """
    b = base.copy()
    for c in LAG_COLS:
        b[c] = base[c].shift(1)
    # the lagged regime of the first grid step is undefined; build_base would call it night
    b["regime"] = b["regime"].fillna("night")
    return b


def make_xy_ext(base: pd.DataFrame, h: int):
    """R3: (rows make_xy keeps, rows dropped for a missing target alone). Same assignments
    as ce1_causal.make_xy; the first frame is asserted equal to it by the caller.
    """
    d = base.copy()
    d["y_kt"] = d["kt"].shift(-h)
    d["y_ghi"] = d["ghi"].shift(-h)
    d["y_ghi_cs"] = d["ghi_cs"].shift(-h)
    d["y_is_day"] = d["is_day"].shift(-h)
    d["base_kt"] = d["kt_in"]
    d["base_ghi"] = d["kt_in"] * d["ghi_cs"]
    d["base_regime"] = d["regime"]
    need_in = K.FEATURES + ["y_ghi_cs", "base_ghi", "base_kt"]
    common = (d["is_day"] & d["y_is_day"].astype("boolean").fillna(False).astype(bool)
              & d[need_in].notna().all(axis=1))
    has_y = d[["y_kt", "y_ghi"]].notna().all(axis=1)
    return d[common & has_y], d[common & ~has_y]


def frames(geo: pd.DataFrame, test_year: int, h: int, lag: bool = False, base=None, cal=None):
    """ce1_folds.fold_frames plus the missing-target rows; lag=True applies R5."""
    f = K.fold(test_year)
    if cal is None:
        cal, nband = K.fit_scalar(geo, f["train_max"])
    else:
        nband = None
    if base is None:
        base = K.build_base(geo, cal, "locf6")
        if lag:
            base = lag_inputs(base)
    d_std, d_miss = make_xy_ext(base, h)
    d_std, d_miss = d_std.sort_index(), d_miss.sort_index()
    v = base["kt_in"].rolling(K.VAR_WIN, min_periods=K.VAR_WIN // 2).std()
    g_t = base["regime"].shift(-h)
    d_std = d_std.assign(v=v.reindex(d_std.index).values, g_t=g_t.reindex(d_std.index).values)
    d_miss = d_miss.assign(v=v.reindex(d_miss.index).values, g_t=g_t.reindex(d_miss.index).values)
    tr, ca, te = K.split(d_std, f)
    _, cam, tem = K.split(d_miss, f)
    fr = dict(fold=f, scalar=cal, n_band=nband, train=tr, calib=ca, test=te)
    return fr, dict(calib=cam, test=tem), base


def crossing(out: pd.DataFrame, taus) -> dict:
    """ce1_folds.fit_cache's crossing record, recomputed on a row subset."""
    Q = out[[F.qcol(t) for t in taus]].values.astype(float)
    dq = np.diff(Q, axis=1)
    worst = np.maximum(-dq.min(axis=1), 0.0) if len(Q) else np.array([0.0])
    c = dict(share_adjacent_pairs_crossed_by_more_than_1Wm2=float((dq < -1.0).mean()) if len(Q) else 0.0,
             adjacent_crossing_Wm2_p50=float(np.median(worst)),
             adjacent_crossing_Wm2_p95=float(np.quantile(worst, 0.95)))
    lo, hi = F.qcol(0.05), F.qcol(0.95)
    if lo in out and hi in out:
        c = dict(share_rows_q05_above_q95=float((out[lo] > out[hi]).mean()) if len(out) else 0.0, **c)
    return c


def fit_both(fr: dict, miss: dict, taus, point=True, max_iter=None, sub=None):
    """One fit on the unchanged training rows; predictions on the measured-target rows and
    on the missing-target rows. Returns (std cache, std record, miss cache, miss record).
    """
    allf = dict(fr, calib=pd.concat([fr["calib"], miss["calib"]]).sort_index(),
                test=pd.concat([fr["test"], miss["test"]]).sort_index())
    out, rec = F.fit_cache(allf, taus=taus, max_iter=max_iter, sub=sub, point=point)
    has_y = out["y"].notna().values
    std, ms = out[has_y].copy(), out[~has_y].copy()
    ref_idx = pd.concat([fr["calib"], fr["test"]]).sort_index().index
    assert std.index.equals(ref_idx), "measured-target rows differ from the fold frame"
    te = (std["split"] == "test").values
    rs = dict(rec, n_calib=int((std["split"] == "calib").sum()), n_test=int(te.sum()),
              test_rmse_point=float(np.sqrt(np.mean((std["y"].values[te] - std["p"].values[te]) ** 2)))
              if point else float("nan"), crossing=crossing(std, taus))
    rm = dict(rec, n_calib=int((ms["split"] == "calib").sum()), n_test=int((ms["split"] == "test").sum()),
              test_rmse_point=None, crossing=crossing(ms, taus),
              rows="issue times with t and t+h daytime, every input present (D-066), target missing (R3)")
    return std, rs, ms, rm


def compare(new: pd.DataFrame, old: pd.DataFrame, cols=None) -> dict:
    """Exact comparison as stored. Returns {'ok': bool, 'diffs': {col: msg}}."""
    diffs = {}
    if not new.index.equals(old.index):
        return dict(ok=False, diffs={"index": f"new {len(new)} rows vs old {len(old)}"})
    cols = cols or list(old.columns)
    for c in cols:
        if c not in new.columns:
            diffs[c] = "missing in new"; continue
        a, b = new[c].values, old[c].values
        if a.dtype.kind in "fiub" and b.dtype.kind in "fiub":
            if not np.array_equal(a.astype(float), b.astype(float), equal_nan=True):
                dd = np.abs(a.astype(float) - b.astype(float))
                diffs[c] = f"max |diff| {np.nanmax(dd):.3g}, cells {int(np.nansum(dd > 0))}"
        else:
            sa = pd.Series(a, dtype=object).map(lambda x: "<NA>" if pd.isna(x) else str(x)).values
            sb = pd.Series(b, dtype=object).map(lambda x: "<NA>" if pd.isna(x) else str(x)).values
            if not (sa == sb).all():
                diffs[c] = f"cells {int((sa != sb).sum())}"
    return dict(ok=not diffs, diffs=diffs)


# ====================================================================== self-tests
def selftest(verbose=True) -> dict:
    res = {}

    def ok(name, cond, msg=""):
        res[name] = bool(cond)
        if verbose:
            print(f"  [{'PASS' if cond else 'FAIL'}] {name} {msg}", flush=True)

    geo = K._synthetic_geo(n_days=40, seed=4)
    geo.index = pd.date_range("2021-11-01", periods=len(geo), freq="5min")
    rng = np.random.default_rng(11)
    day = np.where(geo["is_day"].values)[0]
    for s in rng.choice(day, 40, replace=False):  # gaps of 1..10 steps
        geo.iloc[s:s + int(rng.integers(1, 11)), 0] = np.nan
    base = K.build_base(geo, 1.0)
    for h in (1, 3):
        std, ms = make_xy_ext(base, h)
        ok(f"std_rows_equal_make_xy_h{h}", std.equals(K.make_xy(base, h)))
        ok(f"miss_rows_have_no_target_h{h}", len(ms) > 0 and ms["y_ghi"].isna().all()
           and ms[K.FEATURES].notna().all().all() and ms["is_day"].all(), f"({len(ms)} rows)")
        ok(f"row_sets_disjoint_h{h}", not std.index.intersection(ms.index).size)
        # a missing-target row becomes a make_xy row once its target step is measured: filling
        # target steps can only add inputs at later issue times, never remove one
        G = geo.copy()
        pos = geo.index.get_indexer(ms.index) + h
        G.iloc[pos, G.columns.get_loc("ghi")] = 500.0
        full = K.make_xy(K.build_base(G, 1.0), h)
        ok(f"miss_rows_are_target_only_drops_h{h}", set(ms.index) <= set(full.index),
           "(every missing-target row is kept by make_xy once its target is supplied)")
    lb = lag_inputs(base)
    inner = base.index[1:]
    prev = base.index[:-1]
    ok("lag_inputs_take_previous_step", all(np.array_equal(lb.loc[inner, c].values, base.loc[prev, c].values)
                                            if c == "regime" else
                                            np.array_equal(lb.loc[inner, c].values, base.loc[prev, c].values, equal_nan=True)
                                            for c in LAG_COLS))
    keep = [c for c in base.columns if c not in LAG_COLS]
    ok("lag_keeps_targets_and_clock", all(np.array_equal(lb[c].values, base[c].values, equal_nan=True)
                                          if base[c].dtype.kind in "fiub" else (lb[c].values == base[c].values).all()
                                          for c in keep), f"({', '.join(keep)})")
    xl = K.make_xy(lb, 3)
    ok("lag_newest_input_is_h_plus_1_before_target",
       np.allclose(xl["kt_l0"].values, base["kt_in"].shift(1).reindex(xl.index).values, equal_nan=True)
       and np.allclose(xl["y_kt"].values, base["kt"].shift(-3).reindex(xl.index).values, equal_nan=True))
    # fold frames on the synthetic record: train 2021, calib Dec 2021 split by hand
    d_std, d_miss = make_xy_ext(base, 3)
    v = base["kt_in"].rolling(12, min_periods=6).std()
    gt = base["regime"].shift(-3)
    d_std = d_std.sort_index().assign(v=v.reindex(d_std.sort_index().index).values,
                                      g_t=gt.reindex(d_std.sort_index().index).values)
    d_miss = d_miss.sort_index().assign(v=v.reindex(d_miss.sort_index().index).values,
                                        g_t=gt.reindex(d_miss.sort_index().index).values)
    cut1, cut2 = pd.Timestamp("2021-11-26"), pd.Timestamp("2021-12-04")
    fr = dict(fold=dict(test=2022, calib=2021, train_max=2020), scalar=1.0, n_band=0,
              train=d_std[d_std.index < cut1], calib=d_std[(d_std.index >= cut1) & (d_std.index < cut2)],
              test=d_std[d_std.index >= cut2])
    miss = dict(calib=d_miss[(d_miss.index >= cut1) & (d_miss.index < cut2)], test=d_miss[d_miss.index >= cut2])
    taus = [0.05, 0.5, 0.95]
    std, rs, ms, rm = fit_both(fr, miss, taus, max_iter=25)
    ref, rref = F.fit_cache(fr, taus=taus, max_iter=25)
    ok("one_fit_reproduces_std_only_fit", compare(F.quantized(std), F.quantized(ref))["ok"]
       and rs["iterations"] == rref["iterations"] and rs["n_test"] == rref["n_test"]
       and abs(rs["test_rmse_point"] - rref["test_rmse_point"]) < 1e-12)
    ok("miss_rows_predicted", len(ms) == len(miss["calib"]) + len(miss["test"]) and ms["p"].notna().all()
       and ms["y"].isna().all(), f"({len(ms)} rows)")
    bad = F.quantized(std).copy()
    bad.iloc[5, bad.columns.get_loc(F.qcol(0.5))] += 0.01
    ok("compare_catches_one_hundredth", not compare(bad, F.quantized(ref))["ok"])
    bad2 = std.copy(); bad2.iloc[3, bad2.columns.get_loc("g")] = "x"
    ok("compare_catches_label", not compare(bad2, std)["ok"])
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "m.parquet"
        F.write(ms, rm, p)
        back = F.load_cache(p)
        ok("miss_cache_round_trip", compare(back, F.quantized(ms))["ok"])
    return res


# ====================================================================== stage S1E
def close(a: np.ndarray, b: np.ndarray, tol: float):
    """(same missing pattern, bit-exact, max |diff|) for two float arrays."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    pat = bool(np.array_equal(np.isnan(a), np.isnan(b)))
    exact = bool(np.array_equal(a, b, equal_nan=True))
    m = ~np.isnan(a) & ~np.isnan(b)
    dmax = float(np.max(np.abs(a[m] - b[m]))) if m.any() else 0.0
    return pat and dmax <= tol, exact, dmax


def stage_s1e(a, C):
    st = "S1E"
    TOL = 1e-9
    s1 = C.RESULTS / "ce1" / "S1"
    out = Path(a.out) if a.out else C.RESULTS / "ce1" / (st + ("_smoke" if a.smoke else ""))
    out.mkdir(parents=True, exist_ok=True)
    checks, exact = {}, {}

    def check(name, cond, msg=""):
        checks[name] = dict(ok=bool(cond), msg=msg)
        log(st, f"{'PASS' if cond else 'FAIL'}  {name}  {msg}")

    r = selftest(verbose=False)
    check("selftests", all(r.values()), f"({sum(r.values())}/{len(r)})"
          + ("" if all(r.values()) else " failed: " + ", ".join(k for k, v in r.items() if not v)))
    old_scal = json.load(open(s1 / "scalars.json"))
    old_counts = pd.read_csv(s1 / "retained_counts.csv")
    scalars, counts = {}, []
    for site in ("yulara", "asp"):
        t1 = time.time()
        grid = K.load_unfilled_raw(site, years=years_with_new(site))
        sens = {k: grid.attrs.pop(k) for k in ("sensor_frame", "sensor_days", "p1_grid", "ghr_grid")
                if k in grid.attrs}
        geo = K.add_geometry(grid, site)
        log(st, f"{site}: grid {len(grid)} rows {grid.index.min()} .. {grid.index.max()} ({time.time() - t1:.0f}s)")
        old = pd.read_parquet(s1 / f"{site}_geo_5min.parquet")
        head = geo.loc[geo.index <= old.index.max()]
        same_idx = head.index.equals(old.index)
        msgs, good = [], same_idx
        if same_idx:
            good &= bool(np.array_equal(head["is_day"].values, old["is_day"].values))
            for c in ("ghi", "zenith", "ghi_cs_raw"):
                g_, e_, d_ = close(head[c].values, old[c].values, TOL)
                good &= g_
                exact[f"{site}_geo_{c}"] = e_
                msgs.append(f"{c} {'bit-exact' if e_ else f'max |diff| {d_:.1e}'}")
        check(f"{site}_geo_reproduces_S1", good,
              f"({len(old)} rows {old.index.min()} .. {old.index.max()}, index and missing pattern exact, is_day "
              f"exact, " + ", ".join(msgs) + ")" if same_idx else f"(index differs: {len(head)} vs {len(old)})")
        tail = geo.loc[geo.index > old.index.max()]
        yrs = set(tail.index.year)
        late = tail.index[tail.index.year > NEW_YEAR]
        check(f"{site}_new_rows_after_S1", len(tail) > 0 and yrs <= {NEW_YEAR, NEW_YEAR + 1}
              and tail.index.min() == old.index.max() + pd.Timedelta("5min")
              and (len(late) == 0 or late.max() < pd.Timestamp(f"{NEW_YEAR + 1}-01-02")),
              f"({len(tail)} rows {tail.index.min()} .. {tail.index.max()}, of which {len(late)} on "
              f"{NEW_YEAR + 1}-01-01 (in no fold); measured GHI on {int(tail['ghi'].notna().sum())})")
        rec = pd.concat([old[GEO_COLS], tail[GEO_COLS].astype(old[GEO_COLS].dtypes.to_dict())])
        reg = np.diff(rec.index.values) == np.timedelta64(5, "m")
        check(f"{site}_record_regular", bool(reg.all()), f"(irregular steps {int((~reg).sum())})")
        rec.to_parquet(out / f"{site}_geo_5min.parquet")
        if site == "yulara":
            days = sens["sensor_days"]
            buf = io.StringIO(); days.to_csv(buf); buf.seek(0)
            nd = pd.read_csv(buf, index_col=0, parse_dates=True)
            od = pd.read_csv(s1 / "yulara_sensor_days.csv", index_col=0, parse_dates=True)
            hd = nd.loc[nd.index <= od.index.max()]
            good = hd.index.equals(od.index) and all((hd[c] == od[c]).all() for c in ("exchanged", "transition"))
            ms = []
            for c in ("q", "factor"):
                g_, e_, d_ = close(hd[c].values, od[c].values, TOL) if good else (False, False, np.nan)
                good &= g_; exact[f"sensor_days_{c}"] = e_
                ms.append(f"{c} {'bit-exact' if e_ else f'max |diff| {d_:.1e}'}")
            check("yulara_sensor_days_reproduce_S1", good, f"({len(od)} days, flags exact, " + ", ".join(ms) + ")")
            buf.seek(0)
            lines = buf.getvalue().splitlines()
            old_txt = (s1 / "yulara_sensor_days.csv").read_text().splitlines()
            later = [ln for ln in lines[1:] if pd.Timestamp(ln.split(",")[0]) > od.index.max()]
            (out / "yulara_sensor_days.csv").write_text("\n".join(old_txt + later) + "\n")
            d25 = nd[nd.index.year == NEW_YEAR]
            pc = C.RESULTS / "ce1" / "WP67" / "precheck2025" / "days_2025.csv"
            pdays = pd.read_csv(pc, index_col=0, parse_dates=True)
            good = pdays.index.equals(d25.index) and all((d25[c] == pdays[c]).all() for c in ("exchanged", "transition"))
            ms = []
            for c in ("q", "factor"):
                g_, e_, d_ = close(d25[c].values, pdays[c].values, TOL) if good else (False, False, np.nan)
                good &= g_; exact[f"precheck_{c}"] = e_
                ms.append(f"{c} {'bit-exact' if e_ else f'max |diff| {d_:.1e}'}")
            check("yulara_2025_days_equal_precheck", good, f"({len(d25)} days, flags exact, " + ", ".join(ms) + ")")
            check("yulara_2025_no_exchanged_no_transition", (~d25["exchanged"]).all() and (~d25["transition"]).all(),
                  f"(factor {d25['factor'].min():.3f}-{d25['factor'].max():.3f})")
        # the rebuild's scalars on the rebuilt rows, against S1 (the record itself keeps the S1 rows)
        for Y in K.SITES[site]["fold_tests"]:
            cal, nb = K.fit_scalar(geo, K.fold(Y)["train_max"])
            o = old_scal[site][str(Y)]
            e_ = cal == o["scalar"]
            exact[f"rebuild_scalar_{site}_{Y}"] = e_
            rel = abs(cal / o["scalar"] - 1)
            check(f"rebuild_scalar_{site}_f{Y}_reproduces_S1", nb == o["n_band"] and rel <= 1e-12,
                  f"(band {nb} exact, scalar " + ("bit-exact" if e_ else f"rel diff {rel:.1e}") + ")")
        # scalars and counts on the S1E record (S1 rows + the new year)
        scalars[site] = {}
        for Y in list(K.SITES[site]["fold_tests"]) + [NEW_YEAR]:
            f = K.fold(Y)
            cal, nb = K.fit_scalar(rec, f["train_max"])
            scalars[site][str(Y)] = dict(train_max=f["train_max"], calib=f["calib"], test=Y, scalar=cal, n_band=nb)
            if str(Y) in old_scal.get(site, {}):
                o = old_scal[site][str(Y)]
                check(f"{site}_scalar_f{Y}_on_record_equals_S1", o["scalar"] == cal and o["n_band"] == nb,
                      f"({cal!r}, band {nb})")
                if "scalar_R1_channel" in o:
                    scalars[site][str(Y)]["scalar_R1_channel"] = o["scalar_R1_channel"]
            for fill in ("locf6", "none"):
                base = K.build_base(rec, cal, fill)
                for h in F.HORIZONS:
                    x = K.make_xy(base, h)
                    tr, ca, te = K.split(x, f)
                    counts.append(dict(site=site, fold_test=Y, h_min=5 * h, frame="ce1_" + fill, train=len(tr),
                                       calib=len(ca), test=len(te), test_carried=int(te["in_imp"].sum())))
            log(st, f"{site} fold {Y}: scalar {cal:.6f} (band {nb})")
    cdf = pd.DataFrame(counts)
    oc = old_counts[old_counts.frame.isin(["ce1_locf6", "ce1_none"])].copy()
    m = oc.merge(cdf, on=["site", "fold_test", "h_min", "frame"], how="left", suffixes=("_s1", "_new"))
    eq = all((m[f"{c}_s1"].astype(float) == m[f"{c}_new"].astype(float)).all()
             for c in ("train", "calib", "test", "test_carried"))
    check("retained_counts_reproduce_S1", eq and len(m) == len(oc), f"({len(oc)} rows)")
    cdf.to_csv(out / "retained_counts.csv", index=False)
    json.dump(scalars, open(out / "scalars.json", "w"), indent=1)
    json.dump(dict(tolerance=TOL, bit_exact=exact, all_bit_exact=all(exact.values())),
              open(out / "reproduction.json", "w"), indent=1)
    log(st, f"bit-exact on every compared value: {all(exact.values())} "
            f"({sum(exact.values())}/{len(exact)})")
    return checks, out, dict(s1_inputs={p.name: md5(p) for p in sorted(s1.glob("*")) if p.is_file()},
                             raw_2025={"Yulara_2025.csv": md5(C.DATA_RAW / "Yulara_2025.csv"),
                                       "Alice_Springs_2025.csv": md5(C.DATA_DKASC / "raw" / "Alice_Springs_2025.csv")})


# ====================================================================== stage S2E
def job_list(only=None):
    jobs = []
    for Y in ITEM14_FOLDS:
        jobs += [("item14", "yulara", Y, ITEM14_H, "S2"), ("item14", "yulara", Y, ITEM14_H, "S2X")]
    for h in NEW_H:
        jobs.append(("lag", "asp", 2024, h, "S2"))
    for h in NEW_H:
        jobs += [("f2025", "yulara", NEW_YEAR, h, "S2"), ("f2025", "yulara", NEW_YEAR, h, "S2X"),
                 ("f2025", "asp", NEW_YEAR, h, "S2")]
    if only:
        sel = [tuple(p.split(":")) for p in only.split(",")]
        jobs = [j for j in jobs if any(j[0] == s[0] and str(j[2]) == s[1] and str(j[3]) == s[2]
                                       and (len(s) < 4 or j[4] == s[3]) for s in sel)]
    return jobs


def job_paths(out: Path, kind, site, Y, h, lv):
    stem = f"{site}_f{Y}_h{h}"
    x = "x" if lv == "S2X" else ""
    if kind == "item14":
        return dict(miss=out / f"item14{x}" / f"{stem}_miss.parquet")
    if kind == "lag":
        return dict(std=out / "lag" / f"{stem}.parquet")
    return dict(std=out / f"f2025{x}" / f"{stem}.parquet", miss=out / f"f2025{x}_miss" / f"{stem}_miss.parquet")


def stage_s2e(a, C):
    st = "S2E"
    R = C.RESULTS / "ce1"
    s1e = Path(a.s1) if a.s1 else R / "S1E"
    out = Path(a.out) if a.out else R / (st + ("_smoke" if a.smoke else ""))
    out.mkdir(parents=True, exist_ok=True)
    (out / "jobs").mkdir(exist_ok=True)
    checks = {}
    smoke = a.smoke

    def check(name, cond, msg=""):
        checks[name] = dict(ok=bool(cond), msg=msg)
        log(st, f"{'PASS' if cond else 'FAIL'}  {name}  {msg}")

    def finish():
        return checks, out, dict(s1e={p.name: md5(p) for p in sorted(s1e.glob("*")) if p.is_file()})

    if not a.s1:
        stp, det = R / "_runs" / "S1E.json", R / "_runs" / "S1E.detail.json"
        okd = det.exists() and json.load(open(det)).get("ok") is True
        oks = stp.exists() and json.load(open(stp)).get("ok") is True
        check("S1E_record_ok", okd and oks, "(S2E uses the S1E record only after S1E passed every check)")
        if not (okd and oks):
            return finish()
    r = selftest(verbose=False)
    check("selftests", all(r.values()), f"({sum(r.values())}/{len(r)})")
    if not all(r.values()):
        return finish()
    geos = {s: pd.read_parquet(s1e / f"{s}_geo_5min.parquet") for s in ("yulara", "asp")}
    scal = json.load(open(s1e / "scalars.json"))
    s2, s2x = R / "S2", R / "S2X"
    # R4 (b): the level lists
    js2 = sorted(s2.glob("*.json")); js2x = sorted(s2x.glob("*.json"))
    js2 = [p for p in js2 if p.stem.count("_f") == 1]; js2x = [p for p in js2x if p.stem.count("_f") == 1]
    lv_ok = all(json.load(open(p))["taus"] == list(map(float, F.TAUS)) for p in js2)
    lvx_ok = all(json.load(open(p))["taus"] == list(map(float, D.taus_low())) for p in js2x)
    check("levels_equal_S2_and_S2X", lv_ok and lvx_ok and len(js2) >= 1,
          f"({len(F.TAUS)} S2 levels over {len(js2)} caches, {len(D.taus_low())} S2X levels over {len(js2x)})")
    if not (lv_ok and lvx_ok):
        return finish()
    # R4 (a): inputs of every existing S2 cache from the S1E record
    if not a.skip_preflight:
        bad = []
        n = 0
        for site, folds in (("yulara", K.SITES["yulara"]["fold_tests"]), ("asp", K.SITES["asp"]["fold_tests"])):
            for Y in folds:
                f = K.fold(Y)
                cal = scal[site][str(Y)]["scalar"]
                base = K.build_base(geos[site], cal, "locf6")
                for h in F.HORIZONS:
                    p = s2 / f"{site}_f{Y}_h{h}.parquet"
                    if not p.exists():
                        continue
                    fr, _, _ = frames(geos[site], Y, h, base=base, cal=cal)
                    ev = pd.concat([fr["calib"].assign(split="calib"), fr["test"].assign(split="test")])
                    new = pd.DataFrame({"split": ev["split"].values, "y": ev["y_ghi"].values, "y_kt": ev["y_kt"].values,
                                        "gcs": ev["y_ghi_cs"].values, "g": ev["base_regime"].values,
                                        "g_t": ev["g_t"].values, "v": ev["v"].values, "in_imp": ev["in_imp"].values},
                                       index=ev.index)
                    old = pd.read_parquet(p, columns=INPUT_COLS)
                    c = compare(new, old, INPUT_COLS)
                    n += 1
                    if not c["ok"]:
                        bad.append(f"{p.name}: {c['diffs']}")
                log(st, f"preflight {site} fold {Y} done")
        check("preflight_inputs_of_every_S2_cache_reproduce", n > 0 and not bad,
              f"({n} caches)" if not bad else "; ".join(bad[:5]))
        if bad:
            return finish()
    max_iter, sub = (40, 0.2) if smoke else (None, None)
    jobs = job_list(a.only)
    findings = {}
    summary = []
    base_cache = {}
    for kind, site, Y, h, lv in jobs:
        name = f"{kind}_{site}_f{Y}_h{h}_{lv}"
        jpath = out / "jobs" / f"{name}.json"
        paths = job_paths(out, kind, site, Y, h, lv)
        if jpath.exists():
            jr = json.load(open(jpath))
            if jr.get("ok") and bool(jr.get("smoke")) == bool(smoke) and all(
                    p.exists() and md5(p) == jr["outputs"][k] for k, p in paths.items()):
                log(st, f"{name}: done earlier, skipped")
                summary.append(jr["summary"])
                if jr.get("reproduction"):
                    findings[name] = jr["reproduction"]
                checks[name] = dict(ok=True, msg=jr["msg"] + " (earlier run)")
                continue
        t1 = time.time()
        geo = geos[site]
        cal = scal[site][str(Y)]["scalar"]
        key = (site, Y, kind == "lag")
        if key not in base_cache:
            base_cache.clear()
            b = K.build_base(geo, cal, "locf6")
            base_cache[key] = lag_inputs(b) if kind == "lag" else b
        fr, miss, _ = frames(geo, Y, h, base=base_cache[key], cal=cal)
        fr["n_band"] = scal[site][str(Y)]["n_band"]
        if kind == "lag":
            miss = dict(calib=miss["calib"].iloc[:0], test=miss["test"].iloc[:0])
        taus = F.TAUS if lv == "S2" else D.taus_low()
        std, rs, ms, rm = fit_both(fr, miss, taus, point=(lv == "S2"), max_iter=max_iter, sub=sub)
        jc, msg = {}, ""
        if lv == "S2X":
            # p from the S2 cache of the same rows (D-092)
            if kind == "item14":
                src_std = F.load_cache(s2 / f"{site}_f{Y}_h{h}.parquet")
                src_miss = F.load_cache(job_paths(out, kind, site, Y, h, "S2")["miss"])
            else:
                src_std = F.load_cache(job_paths(out, kind, site, Y, h, "S2")["std"])
                src_miss = F.load_cache(job_paths(out, kind, site, Y, h, "S2")["miss"])
            jc["s2x_rows_equal_s2"] = compare(std, src_std, list(F.EXT_KEYS))["ok"] and \
                compare(ms, src_miss, list(F.EXT_KEYS))["ok"]
            std["p"] = src_std["p"].values
            ms["p"] = src_miss["p"].values
            te_ = (std["split"] == "test").values
            rs["test_rmse_point"] = float(np.sqrt(np.mean((std["y"].values[te_] - std["p"].values[te_]) ** 2)))
            rs["point"] = rm["point"] = "copied from S2 (not refitted) [D-092]"
        if kind == "item14":
            ref_p = (s2 if lv == "S2" else s2x) / f"{site}_f{Y}_h{h}.parquet"
            ref = F.load_cache(ref_p)
            refj = json.load(open(ref_p.with_suffix(".json")))
            c = compare(F.quantized(std), ref)
            it_ok = rs["iterations"] == refj["iterations"]
            if smoke:
                msg = f"(smoke: reproduction not applicable, {len(ms)} missing-target rows)"
            else:
                qz = F.quantized(std)
                per = {}
                for col in [k for k in ref.columns if k.startswith("q") or k == "p"]:
                    a_, b_ = qz[col].values.astype(float), ref[col].values.astype(float)
                    d_ = np.abs(a_ - b_)
                    if np.nanmax(d_) > 0:
                        per[col] = dict(share_cells=float(np.mean(d_ > 0)), median_abs=float(np.median(d_)),
                                        p99_abs=float(np.quantile(d_, 0.99)), max_abs=float(np.max(d_)))
                it_diff = {k: [v, refj["iterations"].get(k)] for k, v in rs["iterations"].items()
                           if refj["iterations"].get(k) != v}
                exact_rep = c["ok"] and it_ok
                findings[name] = dict(exact=exact_rep, other_columns_equal=all(
                    k.startswith("q") or k == "p" for k in c["diffs"]), levels_differing=per,
                    iterations_differing=it_diff, n_models=len(rs["iterations"]))
                jc["non_forecast_columns_equal"] = findings[name]["other_columns_equal"]
                msg = (f"({len(std)} measured rows reproduce {ref_p.parent.name}/{ref_p.name} exactly, "
                       f"{len(rs['iterations'])} models with the recorded iterations; {len(ms)} missing-target rows)"
                       if exact_rep else
                       f"(refit NOT exact: {len(per)} forecast columns differ, early stopping differs in "
                       f"{len(it_diff)} of {len(rs['iterations'])} models; recorded finding R4 (c); "
                       f"{len(ms)} missing-target rows)")
        if kind in ("item14", "f2025"):
            jc["miss_rows_have_forecasts"] = bool(len(ms) == 0 or ms[[F.qcol(t) for t in taus]].notna().all().all())
        if kind == "lag":
            ref = F.load_cache(s2 / f"{site}_f{Y}_h{h}.parquet")
            jc["lag_rows_plausible"] = abs(len(std) / len(ref) - 1) < 0.02
            msg = (f"(lagged inputs: calib {rs['n_calib']}, test {rs['n_test']} vs as-run "
                   f"{int((ref['split'] == 'calib').sum())}, {int((ref['split'] == 'test').sum())})")
        if kind == "f2025":
            jc["rows_present"] = rs["n_calib"] > 1000 and rs["n_test"] > 1000
            msg = f"(calib {rs['n_calib']}, test {rs['n_test']}, missing-target rows {len(ms)})"
        if lv == "S2" and not smoke:
            jc["point_rmse_plausible"] = 20 <= rs["test_rmse_point"] <= 250
        outputs = {}
        if "std" in paths:
            paths["std"].parent.mkdir(parents=True, exist_ok=True)
            F.write(std, dict(rs, wp67=kind, smoke=smoke), paths["std"])
            outputs["std"] = md5(paths["std"])
        if "miss" in paths:
            paths["miss"].parent.mkdir(parents=True, exist_ok=True)
            F.write(ms, dict(rm, wp67=kind, smoke=smoke), paths["miss"])
            outputs["miss"] = md5(paths["miss"])
        jok = all(jc.values())
        summ = dict(job=name, kind=kind, site=site, fold=Y, h=h, levels=lv, n_calib=rs["n_calib"],
                    n_test=rs["n_test"], n_miss_calib=rm["n_calib"], n_miss_test=rm["n_test"],
                    test_rmse_point=rs["test_rmse_point"], seconds=round(time.time() - t1, 1), ok=jok)
        summary.append(summ)
        json.dump(dict(ok=jok, smoke=smoke, checks=jc, msg=msg, outputs=outputs, summary=summ,
                       reproduction=findings.get(name)),
                  open(jpath, "w"), indent=1, default=str)
        check(name, jok, msg + ("" if jok else f" checks {jc}"))
        log(st, f"{name}: {summ['seconds']:.0f}s")
    pd.DataFrame(summary).to_csv(out / "jobs_summary.csv", index=False)
    json.dump(findings, open(out / "item14_reproduction.json", "w"), indent=1)
    if findings:
        n_ex = sum(v["exact"] for v in findings.values())
        log(st, f"FINDING R4 (c): {n_ex} of {len(findings)} item-14 refits reproduce their cache exactly")
    done = {s["job"] for s in summary}
    check("every_job_done", all(f"{k}_{s}_f{y}_h{h}_{l}" in done for k, s, y, h, l in jobs),
          f"({len(done)}/{len(jobs)})")
    return finish()


# ====================================================================== stage S5E
S5E_RUNS = [("repro", "S2", "asp:2024:1"),
            ("lag", "S2E/lag", "asp:2024:1,asp:2024:3"),
            ("f2025", "S2E/f2025", "yulara:2025:1,yulara:2025:3,asp:2025:1,asp:2025:3")]


def stage_s5e(a, C):
    st = "S5E"
    R = C.RESULTS / "ce1"
    out = Path(a.out) if a.out else R / st
    out.mkdir(parents=True, exist_ok=True)
    checks = {}

    def check(name, cond, msg=""):
        checks[name] = dict(ok=bool(cond), msg=msg)
        log(st, f"{'PASS' if cond else 'FAIL'}  {name}  {msg}")

    if not a.out:
        stp = R / "_runs" / "S2E.json"
        oks = stp.exists() and json.load(open(stp)).get("ok") is True
        check("S2E_record_ok", oks)
        if not oks:
            return checks, out, {}
    s2root = Path(a.s2root) if a.s2root else R
    runs = [r for r in S5E_RUNS if not a.only or r[0] in a.only.split(",")]
    for tag, src, only in runs:
        o = out / tag
        cmd = [sys.executable, str(HERE / "ce1_conformal.py"), "--stage", "S5E", "--s2", str(s2root / src),
               "--out", str(o), "--only", only]
        log(st, "running " + " ".join(cmd[1:]))
        rc = subprocess.call(cmd)
        check(f"conformal_{tag}_exit0", rc == 0, f"(exit {rc})")
        if rc != 0:
            break
        if tag == "repro":
            for job in only.split(","):
                site, Y, h = job.split(":")
                stem = f"{site}_f{Y}_h{h}"
                new, old = json.load(open(o / f"{stem}.json")), json.load(open(R / "S5" / f"{stem}.json"))
                keys = ("intervals", "crps", "reliability", "calib_size", "bound_cap")
                same = all(new.get(k) == old.get(k) for k in keys)
                check(f"S5_rerun_reproduces_{stem}", same and new.get("input_md5") == old.get("input_md5"),
                      f"({sum(len(old.get(k, [])) for k in keys)} records equal)" if same else
                      "differs in " + ", ".join(k for k in keys if new.get(k) != old.get(k)))
    return checks, out, {}


# ====================================================================== main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default=None)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--only", default=None, help="S2E: kind:fold:h[:levels],...  S5E: repro,lag,f2025")
    ap.add_argument("--out", default=None, help="test run: output folder (stamps go to <out>/_runs)")
    ap.add_argument("--s1", default=None, help="S2E test run: S1E folder")
    ap.add_argument("--s2root", default=None, help="S5E test run: folder holding S2 and S2E")
    ap.add_argument("--skip-preflight", action="store_true", help="test run only")
    a = ap.parse_args()
    if a.selftest:
        r = selftest()
        sys.exit(0 if all(r.values()) else 1)
    assert a.stage in ("S1E", "S2E", "S5E"), "--stage S1E, S2E or S5E"
    if not a.out:
        assert not a.skip_preflight and not a.only, "--only and --skip-preflight are for test run with --out"
    import config as C
    t0 = time.time()
    fn = dict(S1E=stage_s1e, S2E=stage_s2e, S5E=stage_s5e)[a.stage]
    checks, out, extra = fn(a, C)
    runs = (C.RESULTS / "ce1" / "_runs") if not a.out else (out / "_runs")
    runs.mkdir(parents=True, exist_ok=True)
    ok = bool(checks) and all(c["ok"] for c in checks.values())
    detail = dict(stage=a.stage, smoke=a.smoke, ok=ok, seconds=round(time.time() - t0, 1), checks=checks,
                  outputs={str(p.relative_to(out)).replace("\\", "/"): md5(p) for p in sorted(out.rglob("*"))
                           if p.is_file() and "_runs" not in p.parts},
                  code={p: md5(HERE / p) for p in CODE_FILES},
                  inputs=extra,
                  versions=dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__,
                                sklearn=__import__("sklearn").__version__, pvlib=__import__("pvlib").__version__),
                  host=platform.node(), dry_run=bool(a.out))
    json.dump(detail, open(runs / f"{a.stage}{'_smoke' if a.smoke else ''}.detail.json", "w"), indent=1, default=str)
    log(a.stage, f"{'ALL CHECKS PASSED' if ok else 'SOME CHECKS FAILED'} in {time.time() - t0:.0f}s")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
