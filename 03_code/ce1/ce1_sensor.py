"""The Yulara horizontal irradiance record (D-083). Two pyranometers sit behind the two
irradiance channels: a horizontal one and one tilted about 16 degrees toward azimuth 345
degrees. From 2016-07-04 to 2023-01-01, on day-of-year 116 to 1 January of each year, the
channels are exchanged and the tilted sensor is logged at about 0.084 of its scale. This module
rebuilds the horizontal record from the channel calendar and corrects the 2024 step causally.

Usage (from the repository root unless the path starts at ce1/, then from 03_code):
    python ce1/ce1_sensor.py --selftest
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

P1_COL = "3052_Environment_DG_Weather_Station_Pyranometer_1"
GHR_COL = "3052_Environment_DG_Weather_Station_Global_Horizontal_Radiation"

SWAP_FIRST = pd.Timestamp("2016-07-04")
SWAP_LAST = pd.Timestamp("2023-01-01")
SWAP_DOY = 116
PLANE_T = (16.0, 345.0)  # tilt, azimuth of T: fitted on label-state 2017-2022 only
ALBEDO = 0.25
NOON_DEG = 8.0
T_CLEAR = (0.95, 1.12)
H_RANGE = (0.8, 1.5)
MIN_COVER = 0.75
WIN, MIN_HIST, BAND, CLOSE_TOL = 7, 3, 0.05, 0.03

LAT, LON, ALT_M, TZ = -25.2406, 130.9889, 492, "Australia/Darwin"


def exchanged(index: pd.DatetimeIndex) -> np.ndarray:
    """True on days whose two irradiance channels are exchanged (see module docstring)."""
    D = index.normalize()
    doy = index.dayofyear
    return np.asarray((D >= SWAP_FIRST) & (D <= SWAP_LAST) & ((doy >= SWAP_DOY) | (doy == 1)))


def clear_sky_geometry(index: pd.DatetimeIndex) -> pd.DataFrame:
    """Solar elevation, clear-sky GHI and clear-sky irradiance on T's plane (no fitted data)."""
    import pvlib
    t = index.tz_localize(TZ, ambiguous="NaT", nonexistent="NaT")
    loc = pvlib.location.Location(LAT, LON, TZ, ALT_M)
    sp = loc.get_solarposition(t)
    cs = loc.get_clearsky(t, model="ineichen", solar_position=sp)
    poa = pvlib.irradiance.get_total_irradiance(
        PLANE_T[0], PLANE_T[1], sp["apparent_zenith"], sp["azimuth"], cs["dni"], cs["ghi"],
        cs["dhi"], dni_extra=pvlib.irradiance.get_extra_radiation(t), model="haydavies",
        albedo=ALBEDO)["poa_global"]
    return pd.DataFrame({"elev": sp["apparent_elevation"].values, "cs_ghi": cs["ghi"].values,
                         "cs_poa": poa.values}, index=index)


def daily_ratio(H: pd.Series, T: pd.Series, geo: pd.DataFrame, label_day: pd.Series) -> pd.Series:
    """q_D on label-state counting days (NaN elsewhere), indexed by day."""
    day = H.index.normalize()
    elev = pd.Series(geo["elev"].values, index=H.index)
    noon = (elev > elev.groupby(day).transform("max") - NOON_DEG).values
    h = (H / geo["cs_ghi"].where(geo["cs_ghi"] > 0))[noon]
    t = (T / geo["cs_poa"].where(geo["cs_poa"] > 0))[noon]
    d = day[noon]
    both = (h.notna() & t.notna()).groupby(d).mean()
    hm, tm = h.groupby(d).median(), t.groupby(d).median()
    ok = (tm.between(*T_CLEAR) & hm.between(*H_RANGE) & (both >= MIN_COVER)
          & label_day.reindex(hm.index).fillna(False).astype(bool))
    return (hm / tm).where(ok).reindex(label_day.index)


def step_factor(q: pd.Series, label_day: pd.Series) -> pd.Series:
    """Causal factor per day: median of the last WIN counting days before D, dead band BAND."""
    out, hist = [], []
    for d, v in q.items():
        m = float(np.median(hist[-WIN:])) if len(hist) >= MIN_HIST else 1.0
        out.append(m if (abs(m - 1) > BAND and bool(label_day[d])) else 1.0)
        if not np.isnan(v):
            hist.append(float(v))
    return pd.Series(out, index=q.index)


def transition_days(q: pd.Series, f: pd.Series) -> pd.Series:
    out, on = [], False
    for d, v in q.items():
        if not np.isnan(v):
            dev = abs(v / f[d] - 1)
            if dev > BAND:
                on = True
            elif dev <= CLOSE_TOL:
                on = False
        out.append(on)
    return pd.Series(out, index=q.index)


def build_record(p1: pd.Series, ghr: pd.Series, geo: pd.DataFrame | None = None):
    idx = p1.index
    ex = exchanged(idx)
    H = pd.Series(np.where(ex, p1.values, ghr.values), index=idx)
    T = pd.Series(np.where(ex, np.nan, p1.values), index=idx)  # full-scale T only
    geo = clear_sky_geometry(idx) if geo is None else geo
    days = idx.normalize()
    label_day = pd.Series(~ex, index=idx).groupby(days).first()
    q = daily_ratio(H, T, geo, label_day)
    f = step_factor(q, label_day)
    tr = transition_days(q, f)
    fs = f.reindex(days).values
    trs = tr.reindex(days).values.astype(bool)
    ghi = (H / fs).where(~trs)
    frame = pd.DataFrame({"ghi": ghi.values, "src": np.where(ex, "P1", "GHR"),
                          "h_factor": fs, "qc_transition": trs}, index=idx)
    table = pd.DataFrame({"exchanged": ~label_day, "q": q, "factor": f, "transition": tr})
    return frame, table


# --------------------------------------------------------------------------------------
def selftest(verbose=True) -> dict:
    res = {}

    def ok(name, cond, msg=""):
        res[name] = bool(cond)
        if verbose:
            print(f"  [{'PASS' if cond else 'FAIL'}] {name} {msg}")

    # S1 calendar rule at every documented boundary
    cases = {"2016-07-03": False, "2016-07-04": True, "2017-01-01": True, "2017-01-02": False,
             "2018-04-25": False, "2018-04-26": True, "2019-01-01": True, "2019-01-02": False,
             "2020-04-24": False, "2020-04-25": True, "2022-12-31": True, "2023-01-01": True,
             "2023-01-02": False, "2023-06-01": False, "2024-06-01": False, "2016-05-01": False}
    got = {k: bool(exchanged(pd.DatetimeIndex([pd.Timestamp(k) + pd.Timedelta("12h")]))[0])
           for k in cases}
    ok("exchange_rule_boundaries", got == cases,
       "" if got == cases else str({k: v for k, v in got.items() if v != cases[k]}))

    # S2 assignment: exchanged days take P1, others GHR, no scaling
    idx = pd.date_range("2018-04-24", "2018-04-28", freq="5min", inclusive="left")
    p1 = pd.Series(1.0, index=idx); ghr = pd.Series(2.0, index=idx)
    geo = pd.DataFrame({"elev": -1.0, "cs_ghi": 0.0, "cs_poa": 0.0}, index=idx)
    fr, _ = build_record(p1, ghr, geo)
    ex = exchanged(idx)
    ok("assignment_by_day", np.allclose(fr["ghi"][ex], 1.0) and np.allclose(fr["ghi"][~ex], 2.0))

    # S3-S6 monitor on constructed daily ratios (two label-state years)
    rng = np.random.default_rng(0)
    d = pd.date_range("2030-01-01", "2031-12-31", freq="D")
    lab = pd.Series(True, index=d)
    q0 = pd.Series(1.0 + 0.01 * rng.standard_normal(len(d)), index=d)
    q0[rng.random(len(d)) < 0.3] = np.nan  # cloudy days do not count
    f0 = step_factor(q0, lab)
    ok("no_trigger_without_step", (f0 == 1).all(), f"(days corrected {int((f0 != 1).sum())})")
    step = pd.Timestamp("2030-06-10")
    q1 = q0.copy(); q1[q1.index >= step] *= 1.18
    f1 = step_factor(q1, lab); t1 = transition_days(q1, f1)
    conv = f1[f1.index >= step]; first = conv[conv != 1].index.min()
    late = (f1[first:] / 1.18 - 1).abs().max()
    ok("step_detected_and_tracked", pd.notna(first) and late < 0.03,
       f"(first corrected {first.date() if pd.notna(first) else None}, max error after {late:.3f})")
    ok("transition_covers_step_day", bool(t1[step]) or np.isnan(q1[step]),
       f"(transition days {int(t1.sum())})")
    trans_span = t1[t1].index
    ok("transition_closes", len(trans_span) > 0 and trans_span.max() < first + pd.Timedelta("10D"),
       f"({trans_span.min().date()}..{trans_span.max().date()})" if len(trans_span) else "")
    # causality: factor on day D is unchanged when every day >= D is altered
    bad = 0
    for D in rng.choice(np.arange(20, len(d)), size=60, replace=False):
        q2 = q1.copy(); q2.iloc[D:] = q2.iloc[D:] * 0.7
        if step_factor(q2, lab).iloc[D] != f1.iloc[D]:
            bad += 1
    ok("factor_is_causal", bad == 0, f"(violations {bad}/60)")
    lab2 = lab.copy(); lab2[step:] = False
    f3 = step_factor(q1, lab2)
    ok("no_correction_on_exchanged_days", (f3[step:] == 1).all())
    return res


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        r = selftest()
        sys.exit(0 if all(r.values()) else 1)
