"""Strictly causal preprocessing with a fold-specific clear-sky scalar.

Builds the modelling frame from the unfilled 5-min GHI grid: no interpolation of any kind,
targets never imputed, an input value missing at issue time carried forward only within the
gap limit (D-066), and the clear-sky scalar fitted on each fold's training years only. The
frame keeps the column names of the v1 code (kt_l*, kt_rmean, kt_rstd, cosz, hod_*, doy_*,
y_kt, y_ghi, y_ghi_cs, base_ghi, base_kt, base_regime, year).

Usage (from the repository root unless the path starts at ce1/, then from 03_code):
    python ce1/ce1_causal.py --selftest
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE / "utils"))
import config as C

GAP_LIMIT = 6  # steps (30 min) an input may be carried forward, D-066
LAGS = [0, 1, 2, 3, 4, 5, 6, 9, 12]
ROLL = 6
IMP_WIN = max(LAGS) + 1  # 13 input steps t-12..t
KT_CLEAR, KT_OVERCAST, VAR_WIN, VAR_STABLE, CS_DAY_MIN = 0.85, 0.50, 12, 0.10, 20.0
KT_MAX = 1.5

FEATURES_R1 = [f"kt_l{L}" for L in LAGS] + ["kt_rmean", "kt_rstd", "cosz",
                                            "hod_sin", "hod_cos", "doy_sin", "doy_cos"]
FEATURES = FEATURES_R1 + ["in_imp"]

SITES = {
    "yulara": dict(name="Yulara", latitude=-25.2406, longitude=130.9889, altitude=492,
                   tz="Australia/Darwin", years=list(range(2016, 2025)),
                   bounds=(-50.0, 1500.0), fold_tests=[2019, 2020, 2021, 2022, 2023, 2024]),
    "asp": dict(name="DKASC_AliceSprings", latitude=-23.7621, longitude=133.8745, altitude=545,
                tz="Australia/Darwin", years=[2020, 2021, 2022, 2023, 2024],
                bounds=(-50.0, 1600.0), fold_tests=[2024]),
}
ASP_GHI = "101_DKA_WeatherStation_Global_Horizontal_Radiation"


def fold(test_year: int) -> dict:
    return dict(test=test_year, calib=test_year - 1, train_max=test_year - 2)


# --------------------------------------------------------------------------------------
# 1. Unfilled measured grid
# --------------------------------------------------------------------------------------
def _finish_grid(ghi: pd.Series, lo: float, hi: float) -> pd.DataFrame:
    ghi = ghi.mask(ghi.abs() >= C.SENTINEL_ABS)
    ghi = ghi.mask((ghi < lo) | (ghi > hi))
    ghi = ghi.where(~(ghi < 0), 0.0)  # night offset, as in v1
    out = pd.DataFrame({"ghi": ghi.astype(float)})
    out.index.name = "ts"
    return out


def load_unfilled_raw(site: str, raw_dir: Path | None = None, years=None,
                      basis: str = "h") -> pd.DataFrame:
    """Rebuild the measured GHI grid from the raw CSVs with NO gap filling.
    Yulara follows p2_clean.py steps 1-5 and 7 exactly (same parse, dedupe and grid); DKASC
    follows dkasc_prepare.py (grid from the first midnight to the last midnight).
    Yulara basis (D-083): "h" (default) is the horizontal sensor record of ce1_sensor.py
    (channel assignment, causal step correction, transition days missing); "p1" is the v1
    channel Pyranometer_1 alone, kept only for the cross-check against the frozen v1 files.
    """
    S = SITES[site]
    years = years or S["years"]
    if site == "yulara":
        import ce1_sensor as SENS
        assert basis in ("h", "p1")
        raw_dir = Path(raw_dir or C.DATA_RAW)
        col = C.COLS["ghi"]
        assert col == SENS.P1_COL, "config GHI column is not Pyranometer_1"
        cols = (col, SENS.GHR_COL)
        frames = []
        for y in years:
            df = pd.read_csv(raw_dir / f"Yulara_{y}.csv",
                             usecols=lambda c: c in (C.TS,) + cols)
            df["ts"] = pd.to_datetime(df[C.TS], errors="coerce")
            df = df.dropna(subset=["ts"]).drop(columns=[C.TS])
            frames.append(df)
        raw = pd.concat(frames, ignore_index=True)
        # Duplicate timestamps (yearly files overlap, and a few repeat inside a file) keep the
        # FIRST row in file order. v1 used an unstable sort, so among conflicting duplicates
        # its choice depended on the sort algorithm; the stable sort makes v2 deterministic.
        dups = raw[raw.duplicated("ts", keep=False)]
        conflicts = dups.groupby("ts")[col].nunique(dropna=False)
        raw = raw.sort_values("ts", kind="stable").drop_duplicates(subset="ts", keep="first")
        raw = raw.set_index("ts")
        idx = pd.date_range(raw.index.min(), raw.index.max(), freq=C.FREQ)
        p1 = _finish_grid(raw[col].reindex(idx), *S["bounds"])
        if basis == "p1":
            out = p1
        else:
            ghr = _finish_grid(raw[SENS.GHR_COL].reindex(idx), *S["bounds"])
            rec, days = SENS.build_record(p1["ghi"], ghr["ghi"])
            out = pd.DataFrame({"ghi": rec["ghi"].astype(float)})
            out.index.name = "ts"
            out.attrs["sensor_frame"] = rec
            out.attrs["sensor_days"] = days
            out.attrs["p1_grid"] = p1["ghi"]
            out.attrs["ghr_grid"] = ghr["ghi"]
        out.attrs["dup_conflict_ts"] = [str(t) for t in conflicts[conflicts > 1].index]
        return out
    raw_dir = Path(raw_dir or (C.DATA_DKASC / "raw"))
    frames = [pd.read_csv(raw_dir / f"Alice_Springs_{y}.csv", usecols=["timestamp", ASP_GHI])
              for y in years]
    raw = pd.concat(frames, ignore_index=True)
    raw["timestamp"] = pd.to_datetime(raw["timestamp"])
    dups = raw[raw.duplicated("timestamp", keep=False)]
    conflicts = dups.groupby("timestamp")[ASP_GHI].nunique(dropna=False)
    raw = raw.drop_duplicates("timestamp").set_index("timestamp").sort_index(kind="stable")
    idx = pd.date_range(raw.index.min().floor("D"), raw.index.max().ceil("D"), freq="5min",
                        inclusive="left")
    out = _finish_grid(raw[ASP_GHI].reindex(idx), *S["bounds"])
    out.attrs["dup_conflict_ts"] = [str(t) for t in conflicts[conflicts > 1].index]
    return out


def load_unfilled_from_cleaned(site: str, clean_path=None, flags_path=None) -> pd.DataFrame:
    """Reconstruct the unfilled grid from the frozen v1 cleaned files by blanking every cell
    the v1 cleaner filled. Used for development and as a cross-check of the raw path.
    """
    if site == "yulara":
        c = pd.read_parquet(clean_path or (C.DATA_CLEAN / "yulara_clean_5min.parquet"))
        f = pd.read_parquet(flags_path or (C.DATA_CLEAN / "yulara_quality_flags.parquet"))
        ghi = c["ghi"].mask(f["ghi_imputed"].reindex(c.index).fillna(False).astype(bool))
    else:
        c = pd.read_parquet(clean_path or (C.DATA_DKASC / "cleaned" / "asp_clean_5min.parquet"))
        ghi = c["ghi"].mask(c["ghi_imputed"].astype(bool))
    out = pd.DataFrame({"ghi": ghi.astype(float)})
    out.index.name = "ts"
    return out


# --------------------------------------------------------------------------------------
# 2. Solar geometry (deterministic, no fitted parameter)
# --------------------------------------------------------------------------------------
def add_geometry(grid: pd.DataFrame, site: str) -> pd.DataFrame:
    import pvlib
    S = SITES[site]
    loc = pvlib.location.Location(S["latitude"], S["longitude"], tz=S["tz"],
                                  altitude=S["altitude"], name=S["name"])
    idx = grid.index.tz_localize(S["tz"], ambiguous="NaT", nonexistent="NaT")
    ok = np.asarray(idx.notna())
    g = grid.loc[ok].copy()
    idx = idx[ok]
    g["zenith"] = loc.get_solarposition(idx)["zenith"].values
    g["ghi_cs_raw"] = loc.get_clearsky(idx, model="ineichen")["ghi"].values
    g["is_day"] = g["ghi_cs_raw"].values > CS_DAY_MIN
    return g


# --------------------------------------------------------------------------------------
# 3. Fold-specific clear-sky scalar (training years only)
# --------------------------------------------------------------------------------------
def fit_scalar(geo: pd.DataFrame, train_max: int) -> tuple[float, int]:
    """Median raw clear-sky index over stable near-envelope daytime points of the training
    years (<= train_max). Same band as v1 (0.85..1.15, trailing 1-h std < 0.05), computed
    on measured values only.
    """
    day = geo["is_day"].values
    cs = geo["ghi_cs_raw"].values
    kt_raw = pd.Series(np.where(day, geo["ghi"].values / np.where(cs > 0, cs, np.nan), np.nan),
                       index=geo.index)
    var = kt_raw.rolling(VAR_WIN, min_periods=VAR_WIN // 2).std()
    band = (day & kt_raw.between(0.85, 1.15).values & (var.values < 0.05)
            & (geo.index.year <= train_max))
    return float(np.nanmedian(kt_raw.values[band])), int(band.sum())


# --------------------------------------------------------------------------------------
# 4. Issue-time base frame and supervised frame
# --------------------------------------------------------------------------------------
def build_base(geo: pd.DataFrame, cal: float, fill: str = "locf6") -> pd.DataFrame:
    """Issue-time features on the regular 5-min grid. fill='locf6' is D-066; fill='none'
    is the no-fill alternative used for the retained-count comparison.
    """
    assert fill in ("locf6", "none")
    idx = geo.index
    if len(idx) > 1:
        assert (np.diff(idx.values) == np.timedelta64(5, "m")).all(), "grid must be regular"
    day = geo["is_day"].values
    ghi_cs = geo["ghi_cs_raw"].values * cal
    kt = np.where(day, geo["ghi"].values / np.where(ghi_cs > 0, ghi_cs, np.nan), np.nan)
    kt = pd.Series(np.clip(kt, 0, KT_MAX), index=idx)  # measured kt, NaN if missing
    if fill == "locf6":
        kin = kt.ffill(limit=GAP_LIMIT).where(day)  # causal: uses steps <= s only
    else:
        kin = kt.copy()
    imp = (kt.isna() & kin.notna()).astype(float)

    df = pd.DataFrame(index=idx)
    df["ghi"] = geo["ghi"].values  # measured (targets)
    df["ghi_cs"] = ghi_cs
    df["kt"] = kt.values  # measured (targets)
    df["kt_in"] = kin.values  # issue-time input series
    df["imp"] = imp.values
    df["zenith"] = geo["zenith"].values
    df["is_day"] = day
    df["year"] = idx.year
    hod = idx.hour + idx.minute / 60
    df["hod_sin"] = np.sin(2 * np.pi * hod / 24); df["hod_cos"] = np.cos(2 * np.pi * hod / 24)
    doy = idx.dayofyear
    df["doy_sin"] = np.sin(2 * np.pi * doy / 365); df["doy_cos"] = np.cos(2 * np.pi * doy / 365)
    df["cosz"] = np.cos(np.radians(df["zenith"].clip(0, 90)))
    for L in LAGS:
        df[f"kt_l{L}"] = kin.shift(L).values
    df["kt_rmean"] = kin.rolling(ROLL, min_periods=ROLL).mean().values
    df["kt_rstd"] = kin.rolling(ROLL, min_periods=ROLL).std().values
    df["in_imp"] = imp.rolling(IMP_WIN, min_periods=1).max().values
    # regime at issue time, from the issue-time series (trailing window, fixed thresholds)
    kvar = kin.rolling(VAR_WIN, min_periods=VAR_WIN // 2).std().values
    kv = kin.values
    d = day & ~np.isnan(kv)
    reg = np.full(len(df), "night", dtype=object)
    trans = d & (kvar >= VAR_STABLE)
    clear = d & ~trans & (kv >= KT_CLEAR)
    cloudy = d & ~trans & (kv < KT_OVERCAST)
    mid = d & ~trans & ~clear & ~cloudy
    reg[trans] = "transitional"; reg[clear] = "clear"; reg[cloudy] = "cloudy"
    reg[mid] = "transitional"
    df["regime"] = reg
    return df


def make_xy(base: pd.DataFrame, h: int) -> pd.DataFrame:
    """Supervised frame for horizon h steps. Targets are measured values at t+h (never
    imputed); inputs are the issue-time series. Row kept iff t and t+h are daytime and every
    feature and target is present.
    """
    d = base.copy()
    d["y_kt"] = d["kt"].shift(-h)
    d["y_ghi"] = d["ghi"].shift(-h)
    d["y_ghi_cs"] = d["ghi_cs"].shift(-h)
    d["y_is_day"] = d["is_day"].shift(-h)
    d["base_kt"] = d["kt_in"]
    d["base_ghi"] = d["kt_in"] * d["ghi_cs"]
    d["base_regime"] = d["regime"]
    need = FEATURES + ["y_kt", "y_ghi", "y_ghi_cs", "base_ghi", "base_kt"]
    valid = (d["is_day"] & d["y_is_day"].astype("boolean").fillna(False).astype(bool)
             & d[need].notna().all(axis=1))
    return d[valid]


def split(d: pd.DataFrame, f: dict):
    return (d[d["year"] <= f["train_max"]], d[d["year"] == f["calib"]],
            d[d["year"] == f["test"]])


def fold_frame(geo: pd.DataFrame, test_year: int, h: int, fill: str = "locf6"):
    """Convenience driver: scalar fitted on the fold's training years, then the split."""
    f = fold(test_year)
    cal, nband = fit_scalar(geo, f["train_max"])
    d = make_xy(build_base(geo, cal, fill), h).sort_index()
    tr, ca, te = split(d, f)
    return dict(fold=f, scalar=cal, n_band=nband, train=tr, calib=ca, test=te)


# --------------------------------------------------------------------------------------
# 5. Self-tests
# --------------------------------------------------------------------------------------
def _synthetic_geo(n_days=3, seed=0):
    """Constructed daytime-like series on a regular grid: sinusoidal clear sky, noisy kt."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2030-01-01", periods=288 * n_days, freq="5min")
    hod = idx.hour + idx.minute / 60
    cs = np.clip(1000 * np.sin(np.pi * (hod - 6) / 12), 0, None)
    kt = np.clip(0.7 + 0.2 * rng.standard_normal(len(idx)), 0.05, 1.2)
    geo = pd.DataFrame({"ghi": cs * kt, "zenith": np.degrees(np.arccos(np.clip(cs / 1000, 0, 1))),
                        "ghi_cs_raw": cs, "is_day": cs > CS_DAY_MIN}, index=idx)
    return geo


def selftest(geo_real: pd.DataFrame | None = None, n_trunc: int = 150, verbose=True) -> dict:
    res = {}

    def ok(name, cond, msg=""):
        res[name] = bool(cond)
        if verbose:
            print(f"  [{'PASS' if cond else 'FAIL'}] {name} {msg}")
        return cond

    # T1 constructed gaps: boundary of the gap rule and targets never imputed
    g = _synthetic_geo()
    day_pos = np.where(g["is_day"].values)[0]
    s0 = day_pos[100]  # a midday position of day 1
    G = g.copy()
    G.iloc[s0:s0 + 8, G.columns.get_loc("ghi")] = np.nan  # an 8-step gap starting at s0
    b = build_base(G, 1.0, "locf6")
    last = b["kt"].iloc[s0 - 1]
    filled = b["kt_in"].iloc[s0:s0 + 8].values
    ok("gap_rule_first6_carried", np.allclose(filled[:6], last), f"(last obs {last:.4f})")
    ok("gap_rule_7th_8th_missing", np.isnan(filled[6:]).all())
    ok("gap_rule_flag", b["in_imp"].iloc[s0] == 1 and b["in_imp"].iloc[s0 + 5 + IMP_WIN - 1] == 1
       and b["in_imp"].iloc[s0 + 5 + IMP_WIN] == 0 and b["in_imp"].iloc[s0 - 1] == 0,
       "(flag on while any carried input is within t-12..t)")
    b0 = build_base(G, 1.0, "none")
    ok("nofill_leaves_gap", np.isnan(b0["kt_in"].iloc[s0:s0 + 8]).all())
    for h in (1, 3, 6, 12):
        xy = make_xy(b, h)
        tgt_pos = set(np.where(np.isnan(b["kt"].values))[0])
        rows = b.index.get_indexer(xy.index)
        ok(f"targets_never_imputed_h{h}", not any((r + h) in tgt_pos for r in rows))
    # the ongoing-gap case at issue time (h=3, so both targets s0+8, s0+9 are measured):
    # elapsed 6 missing steps -> kept on carried inputs, elapsed 7 -> dropped
    xy3 = make_xy(b, 3)
    ok("issue_time_elapsed6_kept", b.index[s0 + 5] in xy3.index)
    ok("issue_time_elapsed7_dropped", b.index[s0 + 6] not in xy3.index)
    # a short gap: interpolation would use the value after the gap, LOCF must not
    G2 = g.copy(); G2.iloc[s0:s0 + 3, 0] = np.nan
    G3 = G2.copy(); G3.iloc[s0 + 3, 0] = G3.iloc[s0 + 3, 0] * 0.1  # change the post-gap value
    ok("short_gap_ignores_post_gap_value",
       np.allclose(build_base(G2, 1.0)["kt_in"].iloc[s0:s0 + 3],
                   build_base(G3, 1.0)["kt_in"].iloc[s0:s0 + 3]))

    # T2 truncation test: features at t are unchanged when everything after t is removed
    def trunc_test(geo, cal, label, n, seed=1):
        rng = np.random.default_rng(seed)
        day = np.where(geo["is_day"].values)[0]
        day = day[(day > 600) & (day < len(geo) - 600)]
        cols = FEATURES + ["base_regime", "base_kt", "base_ghi"]
        bad = 0
        for T in rng.choice(day, size=min(n, len(day)), replace=False):
            win = geo.iloc[T - 600:T + 600].copy()
            cut = win.copy()
            cut.iloc[600 + 1:, cut.columns.get_loc("ghi")] = np.nan  # remove the future
            keep = FEATURES + ["regime", "kt_in"]
            a = build_base(win, cal).iloc[:601][keep]
            c = build_base(cut, cal).iloc[:601][keep]
            if not a.equals(c):
                bad += 1
        return bad

    # the detector itself: a two-sided interpolation (the v1 mechanism) must be caught
    _orig = globals()["build_base"]

    def _leaky(geo, cal):
        g2 = geo.copy(); g2["ghi"] = g2["ghi"].interpolate(limit=GAP_LIMIT, limit_area="inside")
        return _orig(g2, cal, "none")

    g_gappy = _synthetic_geo(n_days=8, seed=3)
    rng = np.random.default_rng(7)
    for start in rng.choice(len(g_gappy) - 20, size=60, replace=False):
        g_gappy.iloc[start:start + int(rng.integers(1, 15)), 0] = np.nan
    bad = trunc_test(g_gappy, 1.0, "constructed", 200)
    ok("no_lookahead_constructed_gaps", bad == 0, f"(violations {bad}/200)")
    globals()["build_base"] = lambda geo, cal, fill="locf6": _leaky(geo, cal)
    try:
        caught = trunc_test(g_gappy, 1.0, "mutant", 200)
    finally:
        globals()["build_base"] = _orig
    ok("detector_catches_two_sided_interpolation", caught > 0, f"(mutant flagged {caught}/200)")
    if geo_real is not None:
        bad = trunc_test(geo_real, 1.05, "real", n_trunc, seed=2)
        ok("no_lookahead_real_data", bad == 0, f"(violations {bad}/{n_trunc})")

        # T3 scalar depends only on the training years of the fold
        yrs = sorted(set(geo_real.index.year))
        tm = yrs[-3] if len(yrs) >= 3 else yrs[0]
        c1, _ = fit_scalar(geo_real, tm)
        P = geo_real.copy(); fut = P.index.year > tm
        P.loc[fut, "ghi"] = P.loc[fut, "ghi"] * 0.5
        c2, _ = fit_scalar(P, tm)
        ok("scalar_ignores_later_years", c1 == c2, f"(train_max {tm}: {c1:.6f} vs {c2:.6f})")
        Q = geo_real.copy(); past = Q.index.year <= tm
        Q.loc[past, "ghi"] = Q.loc[past, "ghi"] * 0.97
        c3, _ = fit_scalar(Q, tm)
        ok("scalar_responds_to_training_years", abs(c3 - c1) > 1e-4, f"({c1:.4f} -> {c3:.4f})")
    return res


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        r = selftest()
        sys.exit(0 if all(r.values()) else 1)
