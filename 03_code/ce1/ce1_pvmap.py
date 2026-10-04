"""Stage S1M: available-PV plant model for Yulara. Maps measured horizontal irradiance to the PV
the plant can produce, fitted on the uncurtailed arrays in 2023 and capped at the largest
binned median output of the fit. Rules: D-074, D-079.

Usage (from the repository root unless the path starts at ce1/, then from 03_code):
    python ce1/ce1_pvmap.py --stage S1M
    python3 03_code/ce1/ce1_pvmap.py --selftest
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
CODE = HERE.parent
sys.path.insert(0, str(CODE / "utils"))
import config as C

RATED = {"DG": 1058.4, "LD": 327.6, "SS": 226.8, "SD": 106.7, "CA": 105.9}
RATED_TOTAL = round(sum(RATED.values()), 1)  # 1825.4 kW
# numeric column prefix -> array; the prefix is the same under both header styles
# (2016-2022 "5_1_Active_Power", 2023+ "5_DG_PV1_DB_DG_M1A_Active_Power")
PREFIX = {"8": "CA", "5": "DG", "11": "LD", "10": "SS", "7": "SD1", "9": "SD2", "6": "SD3",
          "3050": "TOTAL", "3051": "SD_TOTAL"}
GHI_COL = C.COLS["ghi"]
GHI_BOUNDS = (-50.0, 1500.0)
POWER_LO = -5.0
BAND = (900.0, 1000.0)
CV_MAX, P5P95_MIN = 0.20, 0.60
BIN_W, MIN_BIN = 25.0, 20
FIT_YEAR, CHECK_YEAR = 2023, 2024
YEARS = (2023, 2024)
ARENA_SPILL = 0.28
TILTS = tuple(range(0, 36, 5))
AZIMUTHS = (330, 345, 0, 15, 30)
ALBEDO = 0.25
SITE = dict(lat=-25.2406, lon=130.9889, alt=492, tz="Australia/Darwin")


def md5(p):
    return hashlib.md5(Path(p).read_bytes()).hexdigest()


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


# ---------------------------------------------------------------- data
def rename_columns(cols):
    """Map raw column names to short names (both header styles)."""
    out = {}
    for c in cols:
        if c == C.TS:
            out[c] = "ts"
        elif c == GHI_COL:
            out[c] = "ghi"
        elif c.endswith("_Active_Power") and c.split("_")[0] in PREFIX:
            out[c] = PREFIX[c.split("_")[0]]
    return out


def clean_frame(raw: pd.DataFrame) -> pd.DataFrame:
    """Stable first-in-file-order dedupe, physical bounds, SD = sum of its 3 subsystems."""
    d = raw.rename(columns=rename_columns(raw.columns))
    d["ts"] = pd.to_datetime(d["ts"], errors="coerce")
    d = d.dropna(subset=["ts"]).sort_values("ts", kind="stable")
    d = d.drop_duplicates("ts", keep="first").set_index("ts")
    keep = ["ghi"] + [v for v in PREFIX.values() if v in d.columns]
    d = d[keep].astype(float)
    d["ghi"] = d["ghi"].where(d["ghi"].between(*GHI_BOUNDS))
    ub = dict(RATED, SD1=106.7, SD2=106.7, SD3=106.7, SD_TOTAL=106.7, TOTAL=RATED_TOTAL)
    for c in keep[1:]:
        d[c] = d[c].where(d[c].between(POWER_LO, 1.25 * ub[c]))
    d["SD"] = d[["SD1", "SD2", "SD3"]].sum(axis=1, min_count=3)
    return d


def load_years(years=YEARS, raw_dir=None) -> pd.DataFrame:
    """Power channels from the raw CSVs; GHI replaced by the horizontal sensor record."""
    import ce1_causal as K
    raw_dir = Path(raw_dir or C.DATA_RAW)
    frames = []
    for y in years:
        hdr = pd.read_csv(raw_dir / f"Yulara_{y}.csv", nrows=0).columns
        use = [c for c in hdr if c in rename_columns(hdr)]
        f = pd.read_csv(raw_dir / f"Yulara_{y}.csv", usecols=use)
        # parse per file: the timestamp format differs between yearly files, and a single
        # to_datetime over the concatenation would infer one format and drop the other year
        f[C.TS] = pd.to_datetime(f[C.TS], errors="coerce")
        frames.append(f)
    d = clean_frame(pd.concat(frames, ignore_index=True))
    d = d[d.index.year.isin(years)]
    hg = K.load_unfilled_raw("yulara", raw_dir=raw_dir, years=list(years))
    hg.attrs.clear()  # bookkeeping frames are not needed here
    d["ghi_p1"] = d["ghi"]
    d["ghi"] = hg["ghi"].reindex(d.index)
    missing = [y for y in years if (d.index.year == y).sum() < 50000]
    if missing:
        raise RuntimeError(f"too few parsed rows for {missing}")
    return d


def channel_check(raw_dir=None, years=range(2016, 2025)) -> pd.DataFrame:
    """Night-time median of the site-total channel per year (PV output must be ~0 at
    night). Documents why only 2023-2024 are usable. Reads two columns per file.
    """
    raw_dir = Path(raw_dir or C.DATA_RAW)
    rows = []
    for y in years:
        hdr = pd.read_csv(raw_dir / f"Yulara_{y}.csv", nrows=0).columns
        tot = [c for c in hdr if c.startswith("3050_") and c.endswith("_Active_Power")][0]
        d = pd.read_csv(raw_dir / f"Yulara_{y}.csv", usecols=[GHI_COL, tot])
        g, t = d[GHI_COL], d[tot].where(d[tot].abs() < 8e4)
        night = g.between(-50, -1)
        rows.append(dict(year=y, total_night_median_kW=round(float(t[night].median()), 1),
                         total_day_median_kW=round(float(t[g > 500].median()), 1)))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- curtailment test
def curtailment_test(d: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for y in sorted(set(d.index.year)):
        x = d[(d.index.year == y) & (d["ghi"] >= BAND[0]) & (d["ghi"] < BAND[1])]
        for a, r in RATED.items():
            v = x[a].dropna() / r
            q5, q50, q95 = v.quantile([0.05, 0.5, 0.95])
            cv = float(v.std() / v.mean())
            rows.append(dict(year=y, array=a, rated_kW=r, n=int(len(v)), median_frac=round(q50, 4),
                             p5_frac=round(q5, 4), p95_frac=round(q95, 4), cv=round(cv, 4),
                             p5_p95=round(q5 / q95, 4),
                             uncurtailed=bool(cv <= CV_MAX and q5 / q95 >= P5P95_MIN)))
    return pd.DataFrame(rows)


def select_arrays(ct: pd.DataFrame) -> list[str]:
    ok = ct.groupby("array")["uncurtailed"].all()
    return [a for a in RATED if ok.get(a, False)]


# ---------------------------------------------------------------- plane irradiance
def array_poa(ghi, index, tilt, azimuth) -> np.ndarray:
    """Irradiance on the arrays' plane from GHI (Erbs decomposition, Hay-Davies transposition).
    index: naive local standard time. No fitted parameter except the plane.
    """
    import pvlib
    idx = pd.DatetimeIndex(index)
    t = idx.tz_localize(SITE["tz"], ambiguous="NaT", nonexistent="NaT")
    sp = pvlib.solarposition.get_solarposition(t, SITE["lat"], SITE["lon"], SITE["alt"])
    g = np.asarray(ghi, float)
    gz = np.nan_to_num(np.clip(g, 0, None))
    ext = np.asarray(pvlib.irradiance.get_extra_radiation(t))
    zen = np.asarray(sp["apparent_zenith"])
    e = pvlib.irradiance.erbs(gz, zen, ext)
    poa = pvlib.irradiance.get_total_irradiance(
        tilt, azimuth, zen, np.asarray(sp["azimuth"]), np.asarray(e["dni"]), gz,
        np.asarray(e["dhi"]), dni_extra=ext, model="haydavies", albedo=ALBEDO)["poa_global"]
    poa = np.clip(np.nan_to_num(np.asarray(poa, float)), 0, None)
    return np.where(np.isnan(g), np.nan, np.where(gz > 0, poa, 0.0))


def _pooled_r2(d, arrays, year, col):
    m = fit_map(d, arrays, year, col)
    x = d[(d.index.year == year) & (d[col] > 0)].dropna(subset=[col] + list(arrays))
    y = x[list(arrays)].sum(axis=1).values / sum(RATED[a] for a in arrays)
    return _stats(y, specific(x[col].values, m))["R2"]


def fit_plane(d: pd.DataFrame, arrays, year=FIT_YEAR) -> tuple[int, int]:
    """Plane maximizing the pooled in-sample R2 of the map on the fit year only."""
    x = d[d.index.year == year].copy()
    best = None
    for tilt in TILTS:
        for az in AZIMUTHS:
            x["irr"] = array_poa(x["ghi"].values, x.index, tilt, az)
            r2 = _pooled_r2(x, arrays, year, "irr")
            if best is None or r2 > best[0]:
                best = (r2, tilt, az)
    return best[1], best[2]


# ---------------------------------------------------------------- map
def fit_map(d: pd.DataFrame, arrays, year=FIT_YEAR, col="irr") -> dict:
    x = d[(d.index.year == year) & (d[col] > 0)].dropna(subset=[col] + list(arrays))
    s = x[list(arrays)].sum(axis=1) / sum(RATED[a] for a in arrays)
    b = np.floor(x[col].values / BIN_W).astype(int)
    grp = pd.Series(s.values).groupby(b)
    med, cnt = grp.median(), grp.size()
    med = med[cnt >= MIN_BIN]
    centers = (med.index.values + 0.5) * BIN_W
    vals = np.maximum.accumulate(med.values)
    return dict(centers=centers, vals=vals, cap=float(vals.max()), fit_year=int(year),
                n_fit=int(len(x)), n_bins=int(len(centers)), arrays=list(arrays),
                rated_total=RATED_TOTAL)


def specific(irr, m) -> np.ndarray:
    """Available specific power (kW per kW rated) at plane irradiance irr."""
    g = np.asarray(irr, float)
    return np.clip(np.interp(g, np.r_[0.0, m["centers"]], np.r_[0.0, m["vals"]], right=m["cap"]),
                   0.0, m["cap"])


def pv_available(irr, m) -> np.ndarray:
    """Available plant PV (kW) at plane irradiance irr."""
    return specific(irr, m) * m["rated_total"]


def pv_available_from_ghi(ghi, index, m) -> np.ndarray:
    """Available plant PV (kW) at GHI and time. The plant model of the v2 decision layer:
    a GHI forecast (or quantile) at a target time maps to PV through this function.
    """
    return pv_available(array_poa(ghi, index, m["plane_tilt"], m["plane_az"]), m)


def load_map(path=None) -> dict:
    path = Path(path or (C.RESULTS / "ce1" / "S1M" / "available_pv_map.npz"))
    z = np.load(path)
    return dict(centers=z["centers"], vals=z["vals"], cap=float(z["cap"]),
                rated_total=float(z["rated_total"]), plane_tilt=float(z["plane_tilt"]),
                plane_az=float(z["plane_az"]))


def _stats(y, yh):
    r = y - yh
    return dict(n=int(len(y)), R2=round(float(1 - np.sum(r ** 2) / np.sum((y - y.mean()) ** 2)), 4),
                RMSE_frac=round(float(np.sqrt(np.mean(r ** 2))), 4), MBE_frac=round(float(r.mean()), 4))


def validate(d, m, year) -> dict:
    t = d[(d.index.year == year) & (d["ghi"] > 0)]
    out = {}
    for a, r in RATED.items():
        x = t.dropna(subset=["irr", a])
        y, yh = x[a].values / r, specific(x["irr"].values, m)
        out[a] = _stats(y, yh)
        if a == "DG":
            lo = x["ghi"].values < 300
            out["DG_lowG_MBE_frac"] = round(float((y[lo] - yh[lo]).mean()), 4)
    sel = m["arrays"]
    x = t.dropna(subset=["irr"] + sel)
    out["pooled_selected"] = _stats(x[sel].sum(axis=1).values / sum(RATED[a] for a in sel),
                                    specific(x["irr"].values, m))
    tt = d[d.index.year == year].dropna(subset=["irr", "TOTAL"])
    avail = pv_available(tt["irr"].clip(lower=0).values, m)
    out["site_total_spill_share"] = round(float(1 - tt["TOTAL"].clip(lower=0).sum() / avail.sum()), 4)
    dg = d[d.index.year == year].dropna(subset=["irr", "DG"])
    out["DG_spill_share"] = round(float(1 - dg["DG"].clip(lower=0).sum()
                                        / (specific(dg["irr"].clip(lower=0).values, m) * RATED["DG"]).sum()), 4)
    return out


# ---------------------------------------------------------------- self-tests
def selftest(verbose=True) -> dict:
    res = {}

    def ok(name, cond, msg=""):
        res[name] = bool(cond)
        if verbose:
            print(f"  [{'PASS' if cond else 'FAIL'}] {name} {msg}")

    old = ["timestamp", GHI_COL, "3050_Total_Site_PV_Generation_Active_Power", "5_1_Active_Power",
           "10_2_Active_Power", "7_3_A_Active_Power", "11_4_Active_Power", "8_5_Active_Power"]
    new = ["timestamp", GHI_COL, "5_DG_PV1_DB_DG_M1A_Active_Power", "11_LD_PV1_DB_LD_1A_Active_Power",
           "10_SS_PV1_DB_SS_1A_Active_Power", "8_CA_PV1_DB_CA_1A_Active_Power"]
    ro, rn = rename_columns(old), rename_columns(new)
    ok("prefix_map_both_header_styles", ro[old[3]] == "DG" == rn[new[2]] and ro[old[6]] == "LD" == rn[new[3]]
       and ro[old[4]] == "SS" == rn[new[4]] and ro[old[7]] == "CA" == rn[new[5]])
    ok("rated_total_1825.4", abs(RATED_TOTAL - 1825.4) < 1e-9)
    # synthetic plant: every array produces k * G / 1000 * rated, clipped at 0.8; DG curtailed
    rng = np.random.default_rng(0)
    rows = []
    for y in (2023, 2024):
        ts = pd.date_range(f"{y}-01-01 06:00", periods=20000, freq="5min")
        g = rng.uniform(1, 1100, len(ts))
        s = np.minimum(0.75 * g / 1000, 0.8)
        rec = {"ts": ts.astype(str), GHI_COL: g}
        for a, r in RATED.items():
            v = s * r * (1 + 0.03 * rng.standard_normal(len(ts)))
            if a == "DG":
                v = v * rng.uniform(0.05, 1.0, len(ts))  # load-proportional curtailment
            if a == "SD":
                for k, sub in zip(("7_SD_x_Active_Power", "9_SD_x_Active_Power", "6_SD_x_Active_Power"),
                                  (22.6, 38.3, 45.8)):
                    rec[k] = v * sub / r
                continue
            key = {"DG": "5_DG_x_Active_Power", "LD": "11_LD_x_Active_Power", "SS": "10_SS_x_Active_Power",
                   "CA": "8_CA_x_Active_Power"}[a]
            rec[key] = v
        rec["3050_T_Active_Power"] = sum(rec[k] for k in rec if k.endswith("Active_Power"))
        rows.append(pd.DataFrame(rec))
    raw = pd.concat(rows, ignore_index=True)
    raw.loc[5, "11_LD_x_Active_Power"] = 89980.3  # sentinel below 9e4
    dup = raw.iloc[[10]].copy(); dup[GHI_COL] = -999.0
    raw = pd.concat([raw.iloc[:11], dup, raw.iloc[11:]], ignore_index=True)
    d = clean_frame(raw)
    d["irr"] = d["ghi"]  # the map machinery on a known plane irradiance
    ok("sentinel_removed", np.isnan(d["LD"].iloc[5]))
    ok("duplicate_keeps_first_in_file_order", d["ghi"].iloc[10] != -999.0 and d.index.is_unique)
    ct = curtailment_test(d)
    sel = select_arrays(ct)
    ok("curtailed_array_rejected", "DG" not in sel and set(sel) == {"LD", "SS", "SD", "CA"}, f"({sel})")
    m = fit_map(d, sel, 2023)
    ok("map_monotone", bool(np.all(np.diff(m["vals"]) >= 0)))
    gtest = np.array([200.0, 500.0, 800.0])
    ok("map_recovers_true_curve", np.allclose(specific(gtest, m), 0.75 * gtest / 1000, atol=0.01),
       f"({np.round(specific(gtest, m), 4).tolist()})")
    ok("map_zero_at_zero_and_capped", specific(0.0, m) == 0.0 and specific(5000.0, m) == m["cap"])
    v = validate(d, m, 2024)
    ok("heldout_R2_selected_high", all(v[a]["R2"] > 0.95 for a in sel))
    ok("curtailed_array_shows_spill", v["DG_spill_share"] > 0.3, f"({v['DG_spill_share']})")
    # transposition: horizontal plane returns GHI; monotone in GHI at fixed time; zero at zero
    ts = pd.date_range("2023-06-15 07:00", "2023-06-15 17:55", freq="5min")
    gh = np.full(len(ts), 500.0)
    flat = array_poa(gh, ts, 0, 0)
    day = np.asarray((ts.hour >= 10) & (ts.hour < 15))  # physical GHI for 500 W/m2
    ok("poa_flat_plane_equals_ghi", np.allclose(flat[day], gh[day], rtol=1e-3))
    ts1 = pd.DatetimeIndex([pd.Timestamp("2023-06-15 11:00")] * 200 + [pd.Timestamp("2023-12-15 15:00")] * 200)
    gg = np.r_[np.linspace(0, 1100, 200), np.linspace(0, 1100, 200)]
    p = array_poa(gg, ts1, 15, 0)
    mono = np.all(np.diff(p[:200]) >= -1e-6) and np.all(np.diff(p[200:]) >= -1e-6)
    ok("poa_monotone_in_ghi", mono and p[0] == 0.0 and p[200] == 0.0)
    ok("poa_nan_stays_nan", np.isnan(array_poa(np.array([np.nan]), ts1[:1], 15, 0))[0])
    return res


# ---------------------------------------------------------------- stage
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="S1M")
    ap.add_argument("--out", default=None)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        r = selftest()
        sys.exit(0 if all(r.values()) else 1)
    t0 = time.time()
    out = Path(a.out) if a.out else C.RESULTS / "ce1" / a.stage
    runs = (out / "_runs") if a.out else C.RESULTS / "ce1" / "_runs"
    out.mkdir(parents=True, exist_ok=True); runs.mkdir(parents=True, exist_ok=True)
    checks = {}

    def check(name, cond, msg=""):
        checks[name] = dict(ok=bool(cond), msg=msg)
        log(f"{'PASS' if cond else 'FAIL'}  {name}  {msg}")

    st = selftest(verbose=False)
    check("selftests", all(st.values()), f"({sum(st.values())}/{len(st)})")
    ch = channel_check()
    ch.to_csv(out / "channel_check.csv", index=False)
    bad = ch[ch.total_night_median_kW.abs() > 5].year.tolist()
    check("site_total_physical_only_2023_2024", bad == list(range(2016, 2023)),
          f"(night-time site total far from 0 in {bad})")
    d = load_years()
    log(f"loaded {len(d)} rows 2023-2024")
    # array selection and plane: the curtailment band is on the irradiance the arrays
    # receive; start from the GHI band, fit the plane on the selection, re-test on the plane
    # band, and repeat until the selection is unchanged
    sel = select_arrays(curtailment_test(d)); path = [list(sel)]
    for _ in range(5):
        tilt, az = fit_plane(d, sel, FIT_YEAR)
        dp = d.copy(); dp["ghi"] = array_poa(d["ghi"].values, d.index, tilt, az)
        ct = curtailment_test(dp)
        new = select_arrays(ct)
        path.append(list(new))
        if new == sel:
            break
        sel = new
    ct.to_csv(out / "curtailment_test.csv", index=False)
    check("selection_and_plane_converged", path[-1] == path[-2],
          f"(selections {path}, plane tilt {tilt} azimuth {az})")
    check("selected_arrays", len(sel) >= 2 and "DG" not in sel, f"({sel})")
    w = d[(d.index.year == 2023) & d.index.month.isin([5, 6, 7]) & (d["ghi"] > 300)]
    ratio = float((w["ghi_p1"] / w["ghi"]).median())
    check("ghi_basis_is_horizontal_record", ratio > 1.1,
          f"(winter 2023 median Pyranometer_1 / record {ratio:.3f} over {len(w)} steps: the "
          f"tilted channel reads high in winter and is no longer the basis)")
    check("array_plane_fitted_on_fit_year", tilt in TILTS and az in AZIMUTHS,
          f"(tilt {tilt}, azimuth {az}, fitted on {FIT_YEAR} only)")
    d["irr"] = array_poa(d["ghi"].values, d.index, tilt, az)
    m = fit_map(d, sel, FIT_YEAR)
    m_swap = fit_map(d, sel, CHECK_YEAR)
    m["plane_tilt"], m["plane_az"] = tilt, az
    v = validate(d, m, CHECK_YEAR)
    v_swap = validate(d, m_swap, FIT_YEAR)
    check("map_monotone", bool(np.all(np.diff(m["vals"]) >= 0)))
    check("heldout_R2_each_selected_array", all(v[x]["R2"] >= 0.85 for x in sel)
          and all(v_swap[x]["R2"] >= 0.85 for x in sel),
          "(" + ", ".join(f"{x} {v[x]['R2']}/{v_swap[x]['R2']}" for x in sel) + ")")
    check("DG_matches_map_at_low_GHI", abs(v["DG_lowG_MBE_frac"]) < 0.02,
          f"(mean bias {v['DG_lowG_MBE_frac']} of rating at GHI < 300)")
    check("implied_spill_near_ARENA_28pct", abs(v["site_total_spill_share"] - ARENA_SPILL) < 0.07,
          f"(2024 site-total spill {v['site_total_spill_share']})")
    np.savez(out / "available_pv_map.npz", centers=m["centers"], vals=m["vals"], cap=m["cap"],
             rated_total=m["rated_total"], fit_year=m["fit_year"], plane_tilt=tilt, plane_az=az)
    fitj = dict(method="GHI (horizontal sensor record) to array-plane irradiance (Erbs, "
                       "Hay-Davies, plane fitted on the fit year), pooled specific power of "
                       "uncurtailed arrays, 25 W/m2 binned median, running maximum, x rated total",
                plane=dict(tilt=tilt, azimuth=az), selection_rule=dict(band_Wm2=BAND, cv_max=CV_MAX,
                                                                           p5_p95_min=P5P95_MIN),
                arrays=sel, rated_kW=RATED, rated_total_kW=RATED_TOTAL, fit_year=m["fit_year"],
                n_fit=m["n_fit"], n_bins=m["n_bins"], cap_specific=round(m["cap"], 4),
                cap_plant_kW=round(m["cap"] * RATED_TOTAL, 1),
                validation={f"{FIT_YEAR}->{CHECK_YEAR}": v, f"{CHECK_YEAR}->{FIT_YEAR}": v_swap},
                swap_cap_specific=round(m_swap["cap"], 4))
    json.dump(fitj, open(out / "pvmap_fit.json", "w"), indent=1)
    ok = all(c["ok"] for c in checks.values())
    detail = dict(stage=a.stage, smoke=False, ok=ok, seconds=round(time.time() - t0, 1), checks=checks,
                  outputs={p.name: md5(p) for p in sorted(out.glob("*")) if p.is_file()},
                  code={p.name: md5(p) for p in (Path(__file__), HERE / "ce1_sensor.py",
                                                  HERE / "ce1_causal.py")},
                  versions=dict(python=platform.python_version(), numpy=np.__version__,
                                pandas=pd.__version__), host=platform.node())
    json.dump(detail, open(runs / f"{a.stage}.detail.json", "w"), indent=1)
    print(ct.to_string(index=False))
    print(json.dumps(fitj["validation"], indent=1))
    log(f"{'ALL CHECKS PASSED' if ok else 'SOME CHECKS FAILED'} in {time.time() - t0:.0f}s")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
