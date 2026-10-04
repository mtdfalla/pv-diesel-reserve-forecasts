"""Sensor-record check of the 2025 Yulara data (channel state and fitted planes) before any 2025
forecast (rules D-129, D-131).

Usage (from the repository root unless the path starts at ce1/, then from 03_code):
    python3 03_code/ce1/ce1_wp67_precheck2025.py
"""
from __future__ import annotations

import hashlib
import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "utils"))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = HERE.parents[1]
OUT = ROOT / "04_results" / "ce1" / "WP67" / "precheck2025"


def fit_plane(s: pd.Series, geo: pd.DataFrame, mask: pd.Series):
    import pvlib
    g = geo[mask]
    x = s[mask]
    key = [g.index.month, (g.index.hour * 60 + g.index.minute) // 60]
    y = np.log(x)
    best = []
    for b in range(0, 21):
        for az in range(270, 451, 15):
            p = pvlib.irradiance.get_total_irradiance(b, az % 360, g.apparent_zenith, g.azimuth, g.cs_dni, g.cs_ghi, g.cs_dhi,
                                                      dni_extra=g.dni_extra, model="haydavies", albedo=0.25)["poa_global"]
            lr = (y - np.log(p)).groupby(key).median()
            best.append((float(lr.std(ddof=0)), b, az % 360, float(np.exp(lr.mean()))))
    best.sort()
    return dict(n=int(mask.sum()), rms=best[0][0], tilt=best[0][1], azimuth=best[0][2], scale=best[0][3])


def main():
    import pvlib
    import ce1_causal as K
    import ce1_sensor as SENS
    OUT.mkdir(parents=True, exist_ok=True)
    years = list(range(2016, 2026))
    raw = ROOT / "02_data" / "Original Dataset"
    src = {f"Yulara_{y}.csv": hashlib.md5((raw / f"Yulara_{y}.csv").read_bytes()).hexdigest() for y in years}
    assert src["Yulara_2025.csv"] == "2421499169acb1cfe62bb94fa7aa66b3"
    grid = K.load_unfilled_raw("yulara", years=years)
    p1, ghr, frame, days = (grid.attrs["p1_grid"], grid.attrs["ghr_grid"], grid.attrs["sensor_frame"],
                            grid.attrs["sensor_days"])
    grid.attrs.clear()
    idx = p1.index
    t = idx.tz_localize(SENS.TZ, ambiguous="NaT", nonexistent="NaT")
    sp = pvlib.solarposition.get_solarposition(t, SENS.LAT, SENS.LON, SENS.ALT_M)
    cs = pvlib.location.Location(SENS.LAT, SENS.LON, SENS.TZ, SENS.ALT_M).get_clearsky(t, model="ineichen", solar_position=sp)
    geo = pd.DataFrame({"apparent_zenith": sp["apparent_zenith"].values, "azimuth": sp["azimuth"].values,
                        "apparent_elevation": sp["apparent_elevation"].values, "cs_ghi": cs["ghi"].values,
                        "cs_dni": cs["dni"].values, "cs_dhi": cs["dhi"].values,
                        "dni_extra": pvlib.irradiance.get_extra_radiation(t).values}, index=idx)
    elev = geo["apparent_elevation"]
    # 1. clear-day channel ratio, 2025
    r = (p1 / ghr).where((elev > 10) & (p1 > 30) & (ghr > 0.5))
    dr = r.groupby(idx.normalize()).median().dropna()
    d25 = dr[dr.index.year == 2025]
    obs = d25 > 4
    rule = pd.Series(SENS.exchanged(d25.index + pd.Timedelta("12h")), index=d25.index)
    a_ok = int(obs.sum()) == 0 and int((obs != rule).sum()) == 0
    # 2. planes
    Hc = frame["ghi"]
    rows = []
    for name, s in (("GHR_corrected", Hc), ("P1_raw", p1)):
        k = s / geo.cs_ghi.where(geo.cs_ghi > 300)
        v = k.rolling(12, min_periods=6).std()
        cm = k.between(0.6, 1.5) & (v < 0.03) & (elev > 15) & (s > 50)
        for y in (2023, 2024, 2025):
            res = fit_plane(s, geo, cm & (idx.year == y))
            rows.append(dict(channel=name, year=y, **res))
            print(rows[-1], flush=True)
    pl = pd.DataFrame(rows)
    g25 = pl[(pl.channel == "GHR_corrected") & (pl.year == 2025)].iloc[0]
    p25 = pl[(pl.channel == "P1_raw") & (pl.year == 2025)].iloc[0]
    b_ok = int(g25.tilt) <= 2
    c_ok = abs(int(p25.tilt) - 16) <= 3 and min(abs(int(p25.azimuth) - 345), 360 - abs(int(p25.azimuth) - 345)) <= 30
    # 3. monitor
    dd = days[days.index.year == 2025]
    cnt = dd.q.notna()
    dev = (dd.q[cnt] / dd.factor[cnt] - 1).abs()
    share_in = float((dev <= 0.05).mean()) if len(dev) else float("nan")
    open_end = bool(dd.transition.iloc[-1]) if len(dd) else True
    d_ok = share_in >= 0.95 and not open_end
    dd.to_csv(OUT / "days_2025.csv")
    pl.to_csv(OUT / "channel_planes.csv", index=False)
    # DKASC header and bounds
    import ce1_causal as K2
    asp = ROOT / "02_data" / "DKASC" / "raw"
    dk = {}
    if (asp / "Alice_Springs_2025.csv").exists():
        h24 = pd.read_csv(asp / "Alice_Springs_2024.csv", nrows=0).columns.tolist()
        h25 = pd.read_csv(asp / "Alice_Springs_2025.csv", nrows=0).columns.tolist()
        g = pd.read_csv(asp / "Alice_Springs_2025.csv", usecols=["timestamp", K2.ASP_GHI])[K2.ASP_GHI]
        dk = dict(header_identical=h24 == h25, ghi_min=float(g.min()), ghi_max=float(g.max()),
                  inside_bounds=bool(g.dropna().between(-50, 1600).all()))
    verdict = dict(
        rule="docs/REGISTERED_RULES.md (written before the first run)",
        a_no_exchanged_days=dict(ok=a_ok, classifiable_days_2025=int(len(d25)), observed_exchanged=int(obs.sum()),
                                 ratio_median=float(d25.median()), ratio_p99=float(d25.quantile(0.99))),
        b_ghr_horizontal=dict(ok=b_ok, tilt=int(g25.tilt), azimuth=int(g25.azimuth), rms=float(g25.rms), n=int(g25.n)),
        c_p1_tilted_plane=dict(ok=c_ok, tilt=int(p25.tilt), azimuth=int(p25.azimuth), rms=float(p25.rms)),
        d_monitor=dict(ok=d_ok, counting_days=int(cnt.sum()), share_within_5pct=share_in, transition_open_at_end=open_end,
                       factor_min=float(dd.factor.min()), factor_max=float(dd.factor.max()),
                       transition_days=int(dd.transition.sum())),
        covered=bool(a_ok and b_ok and c_ok and d_ok),
        dkasc=dk, sources=src, script_md5=hashlib.md5(Path(__file__).read_bytes()).hexdigest())
    (OUT / "verdict.json").write_text(json.dumps(verdict, indent=1, default=str))
    print(json.dumps({k: v for k, v in verdict.items() if k != "sources"}, indent=1, default=str))


if __name__ == "__main__":
    main()
