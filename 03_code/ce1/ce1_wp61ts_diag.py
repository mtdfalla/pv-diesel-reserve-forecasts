"""Post-hoc diagnostic of the timestamp test (not the registered rule): looser clear-day filter and
the sensor plane fitted jointly with the offset.

Usage (from the repository root unless the path starts at ce1/, then from 03_code):
    python3 03_code/ce1/ce1_wp61ts_diag.py --root .
"""
import argparse, json, sys
from pathlib import Path

import numpy as np
import pandas as pd
import pvlib

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ce1_wp61ts import SITES, md5  # noqa: E402

GRID_S = np.arange(-600, 601, 30)
TILTS = [0, 3, 6, 9]
AZS = [300, 315, 330, 345, 0, 15, 30, 45, 60]
ZEN_MAX = 80.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(Path(__file__).resolve().parents[2]))
    ap.add_argument("--max-days-per-year", type=int, default=12)
    a = ap.parse_args()
    R = Path(a.root) / "04_results" / "ce1"
    out = {}
    for site, S in SITES.items():
        loc = pvlib.location.Location(**S)
        geo = pd.read_parquet(R / "S1" / f"{site}_geo_5min.parquet")
        k = geo["ghi"].values / np.where(geo["ghi_cs_raw"].values > 0, geo["ghi_cs_raw"].values, np.nan)
        geo = geo.assign(k=k)
        days = []
        for day, d in geo.groupby(geo.index.normalize()):
            dm = d[d["zenith"] < ZEN_MAX]
            if len(dm) < 100 or dm["ghi"].isna().any():
                continue
            ratio = dm["ghi"].sum() / dm["ghi_cs_raw"].sum()
            if 0.90 <= ratio <= 1.10 and np.nanmax(np.abs(np.diff(dm["k"].values))) < 0.08:
                days.append(day)
        D = pd.Series(days)
        take = []
        for yr, g in D.groupby(D.dt.year):
            ix = np.unique(np.linspace(0, len(g) - 1, min(a.max_days_per_year, len(g))).round().astype(int))
            take += list(g.iloc[ix])
        # residual[plane, day, delta]
        res = np.full((len(TILTS) * len(AZS), len(take), len(GRID_S)), np.nan)
        for j, day in enumerate(take):
            d = geo.loc[day.strftime("%Y-%m-%d")]
            d = d[d["zenith"] < ZEN_MAX]
            g = d["ghi"].values.astype(float)
            idx0 = d.index.tz_localize(S["tz"])
            for m, dl in enumerate(GRID_S):
                idx = idx0 + pd.Timedelta(seconds=int(dl))
                sp = loc.get_solarposition(idx)
                cs = loc.get_clearsky(idx, model="ineichen", solar_position=sp)
                dni_extra = pvlib.irradiance.get_extra_radiation(idx)
                p = 0
                for t in TILTS:
                    for az in AZS:
                        poa = pvlib.irradiance.get_total_irradiance(t, az, sp["apparent_zenith"], sp["azimuth"],
                                                                    cs["dni"], cs["ghi"], cs["dhi"], dni_extra=dni_extra,
                                                                    model="haydavies", albedo=0.25)["poa_global"].values
                        s = float(np.dot(g, poa) / np.dot(poa, poa))
                        res[p, j, m] = float(np.sum((g - s * poa) ** 2))
                        p += 1
        best_delta = GRID_S[np.argmin(res, axis=2)]  # (plane, day)
        tot = np.nanmin(res, axis=2).sum(axis=1)  # total residual per plane
        pb = int(np.argmin(tot))
        planes = [(t, az) for t in TILTS for az in AZS]
        h0 = planes.index((0, 0))
        out[site] = dict(days=len(take), clear_days_found=len(days),
                         horizontal=dict(median_delta_min=float(np.median(best_delta[h0])) / 60,
                                         iqr_min=[float(np.percentile(best_delta[h0], q)) / 60 for q in (25, 75)]),
                         best_plane=dict(tilt=planes[pb][0], azimuth=planes[pb][1],
                                         median_delta_min=float(np.median(best_delta[pb])) / 60,
                                         iqr_min=[float(np.percentile(best_delta[pb], q)) / 60 for q in (25, 75)],
                                         residual_ratio_to_horizontal=float(tot[pb] / tot[h0])),
                         north_planes={f"{t}deg_az0": float(np.median(best_delta[planes.index((t, 0))])) / 60 for t in TILTS})
        print(site, json.dumps(out[site], indent=1), flush=True)
    o = R / "WP61" / "timestamp_diag.json"
    o.write_text(json.dumps(dict(label="POST-HOC diagnostic, not the D-118 rule", results=out), indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
