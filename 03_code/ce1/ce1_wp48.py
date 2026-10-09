"""Stage S48: fold-2024 target sensitivities (rule D-090). Two alternative Yulara GHI records, a
static correction factor (SF) and the transposed tilted sensor (B), replace the primary record
in the calibration and test years of fold 2024.

Usage (from the repository root unless the path starts at ce1/, then from 03_code):
    python ce1/ce1_wp48.py --stage S48 [--smoke]
    python ce1/ce1_wp48.py --stage S48 --root . --hs 3 --years 2022,2023,2024
    python ce1/ce1_wp48.py --selftest
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
import subprocess
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

VARIANTS = ("SF", "B")
HORIZONS = [1, 3, 6, 12]
FOLD = 2024
SF_FROM, SF_TO = pd.Timestamp("2024-04-13"), pd.Timestamp("2024-12-31")
B_FROM = pd.Timestamp("2023-01-02")
B_SCALE = 1.047 / 1.035
ELEV_MIN = 5.0
B_RMSE_BAND = (25.0, 35.0)  # D-090: audit value 29.5 W/m2 against true H in 2023
S6_SCEN = "main,shapeB,peak6MW,spill,minload10,fuel_su,arrays"
CODE_FILES = ("ce1_wp48.py", "ce1_s6.py", "ce1_folds.py", "ce1_conformal.py", "ce1_aci.py",
              "ce1_sensor.py", "ce1_causal.py", "ce1_load.py", "ce1_dispatch.py", "ce1_pvmap.py")


def md5(p: Path) -> str:
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def log(m):
    print(f"[S48] {m}", flush=True)


# ---------------------------------------------------------------- records
def read_days(path: Path) -> pd.DataFrame:
    d = pd.read_csv(path, index_col=0, parse_dates=True)
    d.index = pd.DatetimeIndex(d.index).normalize()
    return d


def static_factor(days: pd.DataFrame) -> tuple[float, int]:
    w = days.loc[SF_FROM:SF_TO, "q"].dropna()
    return float(np.median(w.values)), int(len(w))


def build_sf(geo_e: pd.DataFrame, days: pd.DataFrame):
    f_sf, n = static_factor(days)
    dates = geo_e.index.normalize()
    f_e = days["factor"].reindex(dates).values
    win = np.asarray((dates >= SF_FROM) & (dates <= SF_TO))
    g = geo_e.copy()
    g.loc[win, "ghi"] = geo_e["ghi"].values[win] * f_e[win] / f_sf
    info = dict(f_sf=f_sf, n_counting_days=n, window=[str(SF_FROM.date()), str(SF_TO.date())],
                e_factor_min=float(np.nanmin(f_e[win])), e_factor_max=float(np.nanmax(f_e[win])),
                e_factor_not_1_everywhere_in_window=bool((f_e[win] != 1.0).all()))
    return g, win, info


def solar_apparent(index: pd.DatetimeIndex) -> pd.DataFrame:
    import pvlib
    import ce1_sensor as SENS
    loc = pvlib.location.Location(SENS.LAT, SENS.LON, SENS.TZ, SENS.ALT_M)
    t = index.tz_localize(SENS.TZ)
    sp = loc.get_solarposition(t)
    return pd.DataFrame({"app_zen": sp["apparent_zenith"].values, "azi": sp["azimuth"].values,
                         "app_elev": sp["apparent_elevation"].values}, index=index)


def gti_to_ghi(T: pd.Series, sp: pd.DataFrame) -> pd.Series:
    """gti_dirint on T's plane, the WP-47 audit configuration, in monthly chunks."""
    import pvlib
    import ce1_sensor as SENS
    tilt, az = SENS.PLANE_T
    aoi = pvlib.irradiance.aoi(tilt, az, sp["app_zen"], sp["azi"])
    outs = []
    for _, ss in T.groupby([T.index.year, T.index.month]):
        tzz = ss.index.tz_localize(SENS.TZ)
        r = pvlib.irradiance.gti_dirint(
            pd.Series(ss.values * B_SCALE, index=tzz), pd.Series(aoi.loc[ss.index].values, index=tzz),
            pd.Series(sp.loc[ss.index, "app_zen"].values, index=tzz),
            pd.Series(sp.loc[ss.index, "azi"].values, index=tzz), tzz, tilt, az,
            albedo=SENS.ALBEDO, model="perez", max_iterations=30)
        outs.append(pd.Series(r["ghi"].values, index=ss.index))
    return pd.concat(outs) if outs else pd.Series(dtype=float)


def build_b(geo_e: pd.DataFrame, p1_grid: pd.Series):
    import ce1_sensor as SENS
    idx = geo_e.index
    win = np.asarray(idx >= B_FROM)
    lab = win & ~SENS.exchanged(idx)
    T = p1_grid.reindex(idx).where(lab)
    sp = solar_apparent(idx[win])
    hi = pd.Series(False, index=idx)
    hi.loc[sp.index] = sp["app_elev"].values > ELEV_MIN
    rows = hi.values & T.notna().values
    gb = gti_to_ghi(T[rows], sp.loc[idx[rows]])
    new = pd.Series(np.nan, index=idx)
    new.loc[gb.index] = gb.values
    ghi = geo_e["ghi"].copy()
    repl = win & hi.values  # above 5 degrees in the window: transposed T
    ghi[repl] = new[repl].values
    ghi[geo_e["ghi"].isna().values] = np.nan  # E's missing steps stay missing (D-090)
    g = geo_e.copy()
    g["ghi"] = ghi.values
    sel23 = np.asarray(idx.year == 2023) & repl & geo_e["ghi"].notna().values & ghi.notna().values
    rmse23 = float(np.sqrt(np.mean((ghi.values[sel23] - geo_e["ghi"].values[sel23]) ** 2)))
    info = dict(window_from=str(B_FROM.date()), scale=B_SCALE, plane=list(SENS.PLANE_T),
                elev_min=ELEV_MIN, n_replaced=int(repl.sum()),
                n_lost_vs_e=int((geo_e["ghi"].notna().values & ghi.isna().values).sum()),
                n_label_state_in_window=int(lab.sum()), n_window=int(win.sum()),
                rmse_2023_vs_e=rmse23, n_rmse_2023=int(sel23.sum()))
    return g, win, info


def seasonal_rms(geo: pd.DataFrame) -> pd.Series:
    """ce1_preprocess.sensor_checks (c): monthly median clear-sky index, log std per year."""
    elev = 90 - geo["zenith"]
    kt = geo["ghi"] / geo["ghi_cs_raw"].where(geo["ghi_cs_raw"] > 300)
    var = kt.rolling(12, min_periods=6).std()
    clear = kt.between(0.7, 1.4) & (var < 0.03) & (elev > 15)
    k = kt[clear]
    mon = k.groupby([k.index.year, k.index.month]).median()
    return np.log(mon).groupby(level=0).std(ddof=0)


def same(a: np.ndarray, b: np.ndarray) -> bool:
    return bool(((a == b) | (np.isnan(a) & np.isnan(b))).all())


# ---------------------------------------------------------------- S6 wrapper
def s6_wrapper(argv):
    """Runs ce1_s6.main with the variant record as dispatch truth and E's load scales."""
    import ce1_s6 as S6
    vs1 = Path(argv[0])
    orig = S6.load_inputs

    def patched(s1, s1m, s1l):
        geo_e, m, unit96, scales, rec = orig(s1, s1m, s1l)  # s1 = E: scales from E
        geo_v = pd.read_parquet(vs1 / "yulara_geo_5min.parquet")
        S6.log(f"[D-090] dispatch truth from {vs1}; load scales from E ({s1})")
        return geo_v, m, unit96, scales, rec

    S6.load_inputs = patched
    sys.argv = ["ce1_s6.py"] + list(argv[1:])
    S6.main()


# ---------------------------------------------------------------- self-tests
def selftest(verbose=True) -> dict:
    res = {}

    def rep(name, ok, msg=""):
        res[name] = bool(ok)
        if verbose:
            print(f"{'PASS' if ok else 'FAIL'}  {name}  {msg}", flush=True)

    idx = pd.date_range("2024-04-10", "2024-04-20 23:55", freq="5min")
    rng = np.random.default_rng(1)
    geo = pd.DataFrame({"ghi": rng.uniform(0, 900, len(idx)), "zenith": 40.0, "ghi_cs_raw": 900.0,
                        "is_day": True}, index=idx)
    geo.iloc[5, 0] = np.nan
    days = pd.DataFrame({"q": [np.nan, 1.18, np.nan, 1.19, 1.17, np.nan, 1.20, 1.18, 1.21, np.nan, 1.19],
                         "factor": [1.0, 1.0, 1.0, 1.165, 1.18, 1.18, 1.19, 1.19, 1.19, 1.19, 1.19],
                         "transition": False}, index=pd.date_range("2024-04-10", periods=11, freq="D"))
    g, win, info = build_sf(geo, days)
    dates = idx.normalize()
    rep("sf_constant_is_median_of_window_q", np.isclose(info["f_sf"], np.median([1.19, 1.17, 1.20, 1.18, 1.21, 1.19]))
        and info["n_counting_days"] == 6, f"({info['f_sf']:.4f})")
    rep("sf_outside_window_identical", same(g["ghi"].values[~win], geo["ghi"].values[~win])
        and (dates[win] >= SF_FROM).all() and (dates[~win] < SF_FROM).all())
    H = geo["ghi"] * days["factor"].reindex(dates).values  # raw H before E's correction
    rep("sf_record_equals_H_over_constant", np.allclose(g["ghi"].values[win], (H / info["f_sf"]).values[win],
                                                        equal_nan=True))
    # gti_dirint on a clear, horizontal-equivalent case: the transposition inverts its forward model
    import pvlib
    import ce1_sensor as SENS
    t = pd.date_range("2023-06-15 07:00", "2023-06-15 17:00", freq="5min")
    sp = solar_apparent(t)
    loc = pvlib.location.Location(SENS.LAT, SENS.LON, SENS.TZ, SENS.ALT_M)
    cs = loc.get_clearsky(t.tz_localize(SENS.TZ), model="ineichen")
    tilt, az = SENS.PLANE_T
    poa = pvlib.irradiance.get_total_irradiance(
        tilt, az, sp["app_zen"].values, sp["azi"].values, cs["dni"].values, cs["ghi"].values,
        cs["dhi"].values, dni_extra=pvlib.irradiance.get_extra_radiation(t.tz_localize(SENS.TZ)).values,
        model="perez", albedo=SENS.ALBEDO)["poa_global"]
    m = sp["app_elev"].values > 20
    Tser = pd.Series(np.asarray(poa)[m] / B_SCALE, index=t[m])
    gb = gti_to_ghi(Tser, sp.loc[t[m]])
    err = np.nanmax(np.abs(gb.values - cs["ghi"].values[m]) / cs["ghi"].values[m])
    rep("gti_inverts_clear_sky_forward_model", err < 0.03, f"(max relative error {err:.4f})")
    # build_b on a synthetic record: window, elevation rule, E's missing kept
    idx2 = pd.date_range("2022-12-31", "2023-01-03 23:55", freq="5min")
    sp2 = solar_apparent(idx2)
    csg = loc.get_clearsky(idx2.tz_localize(SENS.TZ), model="ineichen")["ghi"].values
    geo2 = pd.DataFrame({"ghi": csg.copy(), "zenith": 90 - sp2["app_elev"].values, "ghi_cs_raw": csg,
                         "is_day": csg > 0}, index=idx2)
    geo2.iloc[300, 0] = np.nan  # a missing step inside the window
    poa2 = pvlib.irradiance.get_total_irradiance(
        tilt, az, sp2["app_zen"].values, sp2["azi"].values, csg * 0 + loc.get_clearsky(
            idx2.tz_localize(SENS.TZ), model="ineichen")["dni"].values, csg,
        loc.get_clearsky(idx2.tz_localize(SENS.TZ), model="ineichen")["dhi"].values,
        dni_extra=pvlib.irradiance.get_extra_radiation(idx2.tz_localize(SENS.TZ)).values,
        model="perez", albedo=SENS.ALBEDO)["poa_global"]
    p1 = pd.Series(np.asarray(poa2) / B_SCALE, index=idx2)
    gB, winB, infB = build_b(geo2, p1)
    before = np.asarray(idx2 < B_FROM)
    low = np.asarray(sp2["app_elev"].values <= ELEV_MIN)
    rep("b_identical_before_window_and_at_low_sun", same(gB["ghi"].values[before], geo2["ghi"].values[before])
        and same(gB["ghi"].values[winB & low], geo2["ghi"].values[winB & low]))
    rep("b_keeps_E_missing", np.isnan(gB["ghi"].values[300]) and infB["n_lost_vs_e"] >= 0)
    aoi2 = pvlib.irradiance.aoi(tilt, az, sp2["app_zen"], sp2["azi"]).values
    hi = winB & ~low & np.asarray(csg > 200) & (aoi2 < 75)  # well-posed geometry (not grazing)
    e2 = np.nanmax(np.abs(gB["ghi"].values[hi] - csg[hi]) / csg[hi])
    rep("b_recovers_clear_sky_ghi", e2 < 0.03, f"(max relative error {e2:.4f}, above 200 W/m2 and aoi < 75 deg)")
    return res


# ---------------------------------------------------------------- stage
def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--s6-wrapper":
        return s6_wrapper(sys.argv[2:])
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="S48")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--root", default=None, help="output root (test run); default 04_results/ce1/S48")
    ap.add_argument("--hs", default=None, help="comma list of horizons in steps (test run)")
    ap.add_argument("--years", default=None, help="raw years to load (test run; default all)")
    ap.add_argument("--variants", default=",".join(VARIANTS))
    a = ap.parse_args()
    if a.selftest:
        r = selftest()
        sys.exit(0 if all(r.values()) else 1)
    import config as C
    import ce1_causal as K
    t0 = time.time()
    R = C.RESULTS / "ce1"
    # S48D: the dispatch substeps only on the kept records, S2 and S5 of S48, into S48/<v>/
    dispatch_only = a.stage == "S48D"
    root = Path(a.root) if a.root else R / (("S48" if dispatch_only else a.stage) + ("_smoke" if a.smoke else ""))
    runs = R / "_runs" if not a.root else root / "_runs"
    root.mkdir(parents=True, exist_ok=True); runs.mkdir(parents=True, exist_ok=True)
    hs = [int(x) for x in a.hs.split(",")] if a.hs else ([3] if a.smoke else HORIZONS)
    variants = [v for v in a.variants.split(",") if v]
    checks = {}

    def check(name, cond, msg=""):
        checks[name] = dict(ok=bool(cond), msg=msg)
        log(f"{'PASS' if cond else 'FAIL'}  {name}  {msg}")

    st = selftest(verbose=False)
    check("selftests", all(st.values()), f"({sum(st.values())}/{len(st)}"
          + ("" if all(st.values()) else "; failed " + ", ".join(k for k, v in st.items() if not v)) + ")")
    s1 = R / "S1"
    stamp = R / "_runs" / "S1.json"
    if not a.root:
        check("input_stage_S1_completed", stamp.exists() and json.load(open(stamp)).get("ok", False))
    geo_e = pd.read_parquet(s1 / "yulara_geo_5min.parquet")
    days = read_days(s1 / "yulara_sensor_days.csv")
    scal = json.load(open(s1 / "scalars.json"))
    inputs = {n: md5(s1 / n) for n in ("yulara_geo_5min.parquet", "yulara_sensor_days.csv", "scalars.json",
                                       "asp_geo_5min.parquet")}
    records = {}
    # ---- records
    need = [v for v in variants if not (root / v / "S1" / "record_info.json").exists()
            or json.load(open(root / v / "S1" / "record_info.json")).get("inputs") != inputs]
    p1 = None
    if "B" in need:
        years = [int(y) for y in a.years.split(",")] if a.years else None
        tl = time.time()
        grid = K.load_unfilled_raw("yulara", years=years)
        p1 = grid.attrs.pop("p1_grid")
        for k in ("sensor_frame", "sensor_days", "ghr_grid"):
            grid.attrs.pop(k, None)
        j = geo_e.index.intersection(grid.index)
        j = j[j >= (pd.Timestamp("2023-01-01") if years else j.min())]
        ra, rb = grid["ghi"].reindex(j).values, geo_e["ghi"].reindex(j).values
        nn = ~np.isnan(rb)
        rel = np.abs(ra[nn] - rb[nn]) / np.maximum(np.abs(rb[nn]), 1e-9)
        check("raw_rebuild_equals_S1_record", np.array_equal(np.isnan(ra), np.isnan(rb)) and rel.max() <= 1e-12,
              f"({len(j)} steps from {j.min()}, same missing steps, {int((rel == 0).sum())} of {int(nn.sum())} "
              f"bit-exact, max relative difference {rel.max():.1e} on factor-corrected steps; "
              f"{time.time() - tl:.0f}s)")
    for v in variants:
        d1 = root / v / "S1"
        d1.mkdir(parents=True, exist_ok=True)
        if v not in need:
            info = json.load(open(d1 / "record_info.json"))
            log(f"record {v}: built earlier, kept")
        else:
            tb = time.time()
            if v == "SF":
                g, win, info = build_sf(geo_e, days)
            else:
                g, win, info = build_b(geo_e, p1)
            g[["ghi", "zenith", "ghi_cs_raw", "is_day"]].to_parquet(d1 / "yulara_geo_5min.parquet")
            for n in ("asp_geo_5min.parquet", "scalars.json"):
                shutil.copy2(s1 / n, d1 / n)
            f = K.fold(FOLD)
            sc, _ = K.fit_scalar(g, f["train_max"])
            rms = seasonal_rms(g)
            info.update(inputs=inputs, seconds=round(time.time() - tb, 1),
                        identical_outside_window=same(g["ghi"].values[~win], geo_e["ghi"].values[~win]),
                        scalar_2024=sc, scalar_2024_S1=scal["yulara"]["2024"]["scalar"],
                        seasonal_rms={str(k): float(x) for k, x in rms.items()},
                        n_valid_2023=int(g["ghi"][g.index.year == 2023].notna().sum()),
                        n_valid_2024=int(g["ghi"][g.index.year == 2024].notna().sum()))
            json.dump(info, open(d1 / "record_info.json", "w"), indent=1, default=float)
            log(f"record {v}: built in {info['seconds']:.0f}s")
        records[v] = info
        check(f"{v}_identical_to_E_outside_window", info["identical_outside_window"])
        check(f"{v}_scalar_2024_equals_S1", abs(info["scalar_2024"] - info["scalar_2024_S1"]) < 1e-12,
              f"({info['scalar_2024']:.6f})")
        r = info["seasonal_rms"]
        check(f"{v}_seasonal_rms_le_0.04_2023_2024", all(r.get(y, 1) <= 0.04 for y in ("2023", "2024")),
              f"(2023 {r.get('2023', float('nan')):.3f}, 2024 {r.get('2024', float('nan')):.3f})")
        if v == "SF":
            check("SF_factor_within_E_range", info["e_factor_min"] <= info["f_sf"] <= info["e_factor_max"]
                  and info["e_factor_not_1_everywhere_in_window"],
                  f"(f_SF {info['f_sf']:.4f} from {info['n_counting_days']} counting days; E "
                  f"{info['e_factor_min']:.3f}-{info['e_factor_max']:.3f})")
        else:
            check("B_rmse_2023_vs_E_in_band", B_RMSE_BAND[0] <= info["rmse_2023_vs_e"] <= B_RMSE_BAND[1],
                  f"({info['rmse_2023_vs_e']:.1f} W/m2 on {info['n_rmse_2023']} steps; "
                  f"{info['n_lost_vs_e']} steps valid in E lost)")
    if not all(c["ok"] for c in checks.values()):
        log("a record check failed: no stage is run on the variants (D-090)")
    else:
        # ---- stages on each variant
        py = sys.executable
        only = ",".join(f"yulara:{FOLD}:{h}" for h in hs)
        only6 = ",".join(f"{FOLD}:{h}" for h in hs)
        sm = ["--smoke"] if a.smoke else []
        for v in variants:
            d = root / v
            plan = [("S2", [py, "ce1/ce1_folds.py", "--stage", "S2", "--only", only, "--s1", str(d / "S1"),
                            "--out", str(d / "S2")] + sm),
                    ("S5", [py, "ce1/ce1_conformal.py", "--stage", "S5", "--only", only, "--s2", str(d / "S2"),
                            "--out", str(d / "S5")] + sm),
                    ("S6", [py, "ce1/ce1_wp48.py", "--s6-wrapper", str(d / "S1"), "--stage", "S6", "--only", only6,
                            "--s2", str(d / "S2"), "--out", str(d / "S6"), "--scenarios", S6_SCEN] + sm),
                    ("S6B", [py, "ce1/ce1_wp48.py", "--s6-wrapper", str(d / "S1"), "--stage", "S6B", "--only",
                             only6, "--s2", str(d / "S2"), "--out", str(d / "S6B")] + sm)]
            if dispatch_only:
                for name in ("S2", "S5"):
                    det0 = d / name / "_runs" / f"{name}{'_smoke' if a.smoke else ''}.detail.json"
                    check(f"{v}_{name}_kept_ok", det0.exists() and json.load(open(det0)).get("ok", False))
                plan = [x for x in plan if x[0] in ("S6", "S6B")]
            for name, cmd in plan:
                ts = time.time()
                log(f"=== {v} {name} ===")
                rc = subprocess.run(cmd, cwd=str(CODE)).returncode
                det = d / name / "_runs" / f"{name}{'_smoke' if a.smoke else ''}.detail.json"
                ok = rc == 0 and det.exists() and json.load(open(det)).get("ok", False)
                check(f"{v}_{name}_ok", ok, f"(exit {rc}, {time.time() - ts:.0f}s)")
                if not ok:
                    break
    ok = all(c["ok"] for c in checks.values())
    outs = {str(p.relative_to(root)).replace("\\", "/"): md5(p) for p in sorted(root.rglob("*")) if p.is_file()}
    detail = dict(stage=a.stage, smoke=a.smoke, ok=ok, seconds=round(time.time() - t0, 1), checks=checks,
                  variants=variants, horizons=hs, records=records, outputs=outs, inputs=inputs,
                  code={n: md5(HERE / n) for n in CODE_FILES},
                  versions=dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__,
                                pvlib=__import__("pvlib").__version__), host=platform.node())
    json.dump(detail, open(runs / f"{a.stage}{'_smoke' if a.smoke else ''}.detail.json", "w"), indent=1,
              default=float)
    log(f"{'ALL CHECKS PASSED' if ok else 'SOME CHECKS FAILED'} in {time.time() - t0:.0f}s")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
