"""Stage S1: causal preprocessing for both sites. Rebuilds the unfilled GHI grid from the raw
files, adds solar geometry, applies the Yulara sensor record (ce1_sensor.py), fits the
clear-sky scalar of every fold on its training years and counts retained samples per fold,
horizon and split.

Usage (from the repository root unless the path starts at ce1/, then from 03_code):
    python ce1/ce1_preprocess.py --stage S1 [--smoke]
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
import config as C  # noqa: E402

HORIZONS = [1, 3, 6, 12]


def md5(p: Path) -> str:
    h = hashlib.md5()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def log(m):
    print(f"[S1] {m}", flush=True)


def r1_legacy_counts(site: str, h_list) -> list[dict]:
    import datasets as D
    if site == "yulara":
        clean = pd.read_parquet(C.DATA_CLEAN / "yulara_clean_5min.parquet")
        reg = pd.read_parquet(C.DATA_REGIME / "yulara_regimes_5min.parquet")
        fl = pd.read_parquet(C.DATA_CLEAN / "yulara_quality_flags.parquet")
        df = pd.DataFrame(index=clean.index)
        df["ghi"] = clean["ghi"]
        for c in ["ghi_cs", "kt", "zenith", "is_day", "regime"]:
            df[c] = reg[c]
        df["ghi_imputed"] = fl["ghi_imputed"]
    else:
        reg = pd.read_parquet(C.DATA_DKASC / "regime_labels" / "asp_regimes_5min.parquet")
        df = pd.DataFrame(index=reg.index)
        for c in ["ghi", "ghi_cs", "kt", "zenith", "is_day", "regime", "ghi_imputed"]:
            df[c] = reg[c]
    df["year"] = df.index.year
    hod = df.index.hour + df.index.minute / 60
    df["hod_sin"] = np.sin(2 * np.pi * hod / 24); df["hod_cos"] = np.cos(2 * np.pi * hod / 24)
    doy = df.index.dayofyear
    df["doy_sin"] = np.sin(2 * np.pi * doy / 365); df["doy_cos"] = np.cos(2 * np.pi * doy / 365)
    df["cosz"] = np.cos(np.radians(df["zenith"].clip(0, 90)))
    for L in D.LAGS:
        df[f"kt_l{L}"] = df["kt"].shift(L)
    df["kt_rmean"] = df["kt"].rolling(D.ROLL, min_periods=D.ROLL).mean()
    df["kt_rstd"] = df["kt"].rolling(D.ROLL, min_periods=D.ROLL).std()
    rows = []
    f = K.fold(2024)
    for h in h_list:
        x = D.make_xy(df, h)
        tr, ca, te = K.split(x, f)
        rows.append(dict(site=site, fold_test=2024, h_min=5 * h, frame="r1_legacy",
                         train=len(tr), calib=len(ca), test=len(te), test_carried=np.nan))
    return rows, df


def sensor_checks(sens, geo, out, check):
    """WP-47 / D-083 checks of the Yulara horizontal sensor record on the full raw data."""
    import ce1_sensor as SENS
    days = sens["sensor_days"]
    p1, ghr = sens["p1_grid"], sens["ghr_grid"]
    days.to_csv(out / "yulara_sensor_days.csv")
    st = SENS.selftest(verbose=False)
    check("sensor_selftests", all(st.values()), f"({sum(st.values())}/{len(st)})")
    # (a) the calendar rule against the observed channel ratio, every classifiable day
    elev = 90 - geo["zenith"]
    r = (p1 / ghr).reindex(geo.index).where((elev > 10) & (p1.reindex(geo.index) > 30)
                                            & (ghr.reindex(geo.index) > 0.5))
    dr = r.groupby(geo.index.normalize()).median().dropna()
    obs = dr > 4
    rule = pd.Series(SENS.exchanged(dr.index + pd.Timedelta("12h")), index=dr.index)
    mism = int((obs != rule).sum())
    check("sensor_exchange_rule_matches_every_day", mism == 0,
          f"({len(dr)} classifiable days, {mism} mismatches)")
    # (b)-(c) clear-sky index of the record: one sensor, horizontal, stable
    kt = geo["ghi"] / geo["ghi_cs_raw"].where(geo["ghi_cs_raw"] > 300)
    var = kt.rolling(12, min_periods=6).std()
    clear = kt.between(0.7, 1.4) & (var < 0.03) & (elev > 15)
    ex = pd.Series(SENS.exchanged(geo.index), index=geo.index)
    pre = geo.index.year <= 2022
    s_ex = float(kt[clear & ex & pre].median()); s_lab = float(kt[clear & ~ex & pre].median())
    check("sensor_same_scale_both_channels_2016_2022", abs(s_ex / s_lab - 1) < 0.015,
          f"(clear-sky index {s_ex:.4f} on Pyranometer_1 days vs {s_lab:.4f} on GHR days)")
    k = kt[clear]
    mon = k.groupby([k.index.year, k.index.month]).median()
    rms = np.log(mon).groupby(level=0).std(ddof=0)
    kp = (p1.reindex(geo.index) / geo["ghi_cs_raw"].where(geo["ghi_cs_raw"] > 300))[clear]
    rms_p1 = np.log(kp.groupby([kp.index.year, kp.index.month]).median()).groupby(level=0).std(ddof=0)
    check("sensor_seasonal_rms_le_0.04_every_year", bool((rms <= 0.04).all()),
          "(" + ", ".join(f"{y} {v:.3f}" for y, v in rms.items()) + "; v1 channel "
          + ", ".join(f"{y} {v:.3f}" for y, v in rms_p1.items()) + ")")
    # (d) the monitor: silent before 2024, catches the April 2024 step
    corr = days[days["factor"] != 1]
    before = int((corr.index.year < 2024).sum())
    f24 = corr["factor"][corr.index.year == 2024]
    check("sensor_monitor_silent_2016_2023", before == 0, f"(days corrected before 2024: {before})")
    check("sensor_2024_step_caught", len(f24) > 200 and f24.between(1.10, 1.30).all()
          and f24.index.min() < pd.Timestamp("2024-05-01"),
          f"(first {f24.index.min().date() if len(f24) else None}, {len(f24)} days, "
          f"factor {f24.min():.3f}-{f24.max():.3f})" if len(f24) else "(none)")
    tr = days.index[days["transition"]]
    check("sensor_transition_days_few", len(tr) <= 30,
          f"({len(tr)}: " + ", ".join(str(t.date()) for t in tr) + ")")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="S1")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    t0 = time.time()
    out = C.RESULTS / "ce1" / ("S1_smoke" if a.smoke else "S1")
    out.mkdir(parents=True, exist_ok=True)
    runs = C.RESULTS / "ce1" / "_runs"
    runs.mkdir(parents=True, exist_ok=True)
    checks, scalars, counts, tests = {}, {}, [], {}

    def check(name, cond, msg=""):
        checks[name] = dict(ok=bool(cond), msg=msg)
        log(f"{'PASS' if cond else 'FAIL'}  {name}  {msg}")

    for site in ("yulara", "asp"):
        log(f"=== {site} ===")
        ts = time.time()
        sens = {}
        if a.smoke:
            grid = K.load_unfilled_from_cleaned(site)
            conflicts = []
        else:
            grid = K.load_unfilled_raw(site)
            conflicts = grid.attrs.get("dup_conflict_ts", [])
            # the sensor bookkeeping frames leave the grid at once: pandas copies attrs onto
            # every derived frame and to_parquet cannot serialize frames held there
            sens = {k: grid.attrs.pop(k) for k in ("sensor_frame", "sensor_days", "p1_grid",
                                                   "ghr_grid") if k in grid.attrs}
            rec = K.load_unfilled_from_cleaned(site)
            # the v1 cross-check runs on the v1 channel (Pyranometer_1 for Yulara), D-083
            leg = sens["p1_grid"] if site == "yulara" else grid["ghi"]
            j = grid.index.intersection(rec.index)
            a_, b_ = leg.reindex(j), rec["ghi"].reindex(j)
            mism = ~((a_ == b_) | (a_.isna() & b_.isna()))
            geo_j = K.add_geometry(pd.DataFrame({"ghi": a_[mism]}), site) if mism.any() else None
            n_day = int(geo_j["is_day"].sum()) if geo_j is not None else 0
            unexplained = 0
            if geo_j is not None:
                dayts = set(str(t) for t in geo_j.index[geo_j["is_day"].values])
                unexplained = len(dayts - set(conflicts))
            check(f"{site}_raw_equals_R1_unfilled", unexplained == 0,
                  f"(cells {len(j)}, mismatches {int(mism.sum())}, daytime {n_day}, daytime "
                  f"not at a conflicting duplicate {unexplained}; conflicting duplicate "
                  f"timestamps {len(conflicts)})")
            check(f"{site}_grid_span_equals_R1", grid.index.min() == rec.index.min()
                  and grid.index.max() == rec.index.max(),
                  f"({grid.index.min()} .. {grid.index.max()})")
        log(f"grid {len(grid)} rows, GHI missing {int(grid['ghi'].isna().sum())} "
            f"({time.time() - ts:.0f}s)")
        geo = K.add_geometry(grid, site)
        # geometry against v1
        regp = (C.DATA_REGIME / "yulara_regimes_5min.parquet" if site == "yulara"
                else C.DATA_DKASC / "regime_labels" / "asp_regimes_5min.parquet")
        r1reg = pd.read_parquet(regp, columns=["zenith", "is_day"])
        jj = geo.index.intersection(r1reg.index)
        dz = float(np.nanmax(np.abs(geo.loc[jj, "zenith"].values - r1reg.loc[jj, "zenith"].values)))
        dd = int((geo.loc[jj, "is_day"].values != r1reg.loc[jj, "is_day"].values).sum())
        check(f"{site}_geometry_equals_R1", dz < 1e-6 and dd == 0,
              f"(max |zenith diff| {dz:.2e} deg, is_day disagreements {dd})")
        gp = out / f"{site}_geo_5min.parquet"
        geo[["ghi", "zenith", "ghi_cs_raw", "is_day"]].to_parquet(gp)
        if site == "yulara" and not a.smoke:
            sensor_checks(sens, geo, out, check)

        scalars[site] = {}
        for Y in K.SITES[site]["fold_tests"]:
            f = K.fold(Y)
            cal, nb = K.fit_scalar(geo, f["train_max"])
            scalars[site][str(Y)] = dict(train_max=f["train_max"], calib=f["calib"], test=Y,
                                         scalar=cal, n_band=nb)
            for fill in ("locf6", "none"):
                base = K.build_base(geo, cal, fill)
                for h in HORIZONS:
                    x = K.make_xy(base, h)
                    tr, ca, te = K.split(x, f)
                    counts.append(dict(site=site, fold_test=Y, h_min=5 * h,
                                       frame="ce1_" + fill, train=len(tr), calib=len(ca),
                                       test=len(te), test_carried=int(te["in_imp"].sum())))
            log(f"fold {Y}: scalar {cal:.6f} (band {nb}) counts done")
        prim = scalars[site]["2024"]["scalar"]
        r1_cal = 1.0507 if site == "yulara" else 1.0907
        if site == "yulara" and not a.smoke:
            # the machinery reproduces v1 on v1's channel; the v2 basis is recorded beside it
            gl = geo.copy(); gl["ghi"] = sens["p1_grid"].reindex(gl.index).values
            leg_cal, _ = K.fit_scalar(gl, K.fold(2024)["train_max"])
            scalars[site]["2024"]["scalar_R1_channel"] = leg_cal
            check(f"{site}_R1_channel_scalar_close_to_R1", abs(leg_cal - r1_cal) < 0.002,
                  f"({leg_cal:.6f} vs R1 {r1_cal}; CE1 horizontal-sensor basis {prim:.6f})")
        else:
            check(f"{site}_primary_scalar_close_to_R1", abs(prim - r1_cal) < 0.002,
                  f"({prim:.6f} vs R1 {r1_cal}, measured values only vs R1 interpolated)")
        lrows, _ = r1_legacy_counts(site, HORIZONS)
        counts.extend(lrows)

        # self-tests on this site's real data (constructed-data tests run once, on yulara)
        log("self-tests")
        res = K.selftest(geo, n_trunc=300, verbose=True)
        for k, v in res.items():
            tests[f"{site}:{k}"] = v
        check(f"{site}_selftests", all(res.values()),
              f"({sum(res.values())}/{len(res)} passed)")

    cdf = pd.DataFrame(counts)
    cdf.to_csv(out / "retained_counts.csv", index=False)
    json.dump(scalars, open(out / "scalars.json", "w"), indent=1)
    json.dump(tests, open(out / "selftest.json", "w"), indent=1)
    piv = cdf[cdf.frame.isin(["ce1_locf6", "ce1_none"])].pivot_table(
        index=["site", "fold_test", "h_min"], columns="frame", values="test")
    check("d066_keeps_at_least_nofill", bool((piv["ce1_locf6"] >= piv["ce1_none"]).all()),
          f"(min ratio locf6/none {float((piv['ce1_locf6'] / piv['ce1_none']).min()):.4f})")

    ok = all(c["ok"] for c in checks.values())
    detail = dict(stage=a.stage, smoke=a.smoke, ok=ok, seconds=round(time.time() - t0, 1),
                  checks=checks,
                  outputs={p.name: md5(p) for p in sorted(out.glob("*")) if p.is_file()},
                  code={p.name: md5(p) for p in (HERE / "ce1_causal.py", HERE / "ce1_sensor.py",
                                                Path(__file__))},
                  versions=dict(python=platform.python_version(), numpy=np.__version__,
                                pandas=pd.__version__, pvlib=__import__("pvlib").__version__),
                  host=platform.node())
    dname = f"{a.stage}{'_smoke' if a.smoke else ''}.detail.json"
    json.dump(detail, open(runs / dname, "w"), indent=1)
    print(cdf[(cdf.fold_test == 2024)].to_string(index=False))
    log(f"{'ALL CHECKS PASSED' if ok else 'SOME CHECKS FAILED'} in {time.time() - t0:.0f}s")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
