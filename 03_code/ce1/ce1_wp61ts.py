"""Timestamp-convention test of the DKASC 5-min irradiance values (rule D-118): the offset that
best aligns measured GHI with the clear-sky curve on clear days.

Usage (from the repository root unless the path starts at ce1/, then from 03_code):
    python3 03_code/ce1/ce1_wp61ts.py --root .
"""
import argparse, hashlib, json, sys
from pathlib import Path

import numpy as np
import pandas as pd
import pvlib

SITES = {"yulara": dict(latitude=-25.2406, longitude=130.9889, altitude=492, tz="Australia/Darwin"),
         "asp": dict(latitude=-23.7621, longitude=133.8745, altitude=545, tz="Australia/Darwin")}
GRID_S = np.arange(-600, 601, 15)  # seconds
ZEN_MAX = 80.0


def md5(p):
    return hashlib.md5(Path(p).read_bytes()).hexdigest()


def clearsky(loc, idx_local):
    return loc.get_clearsky(idx_local, model="ineichen")["ghi"].values


def selftest():
    """A synthetic clear day built from the model at t + d must return d, for d in {-150, 0, +150} s."""
    loc = pvlib.location.Location(**SITES["yulara"])
    idx = pd.date_range("2022-06-10 06:00", "2022-06-10 18:00", freq="5min").tz_localize("Australia/Darwin")
    out = {}
    for d in (-150, 0, 150):
        g = 0.97 * clearsky(loc, idx + pd.Timedelta(seconds=d))
        out[d] = fit_day(loc, idx, g)[0]
    ok = all(abs(out[d] - d) <= 15 for d in out)
    return ok, out


def fit_day(loc, idx, g):
    zen = loc.get_solarposition(idx)["zenith"].values
    m = (zen < ZEN_MAX) & np.isfinite(g)
    best = (None, np.inf, None)
    for d in GRID_S:
        cs = clearsky(loc, idx[m] + pd.Timedelta(seconds=int(d)))
        s = float(np.dot(g[m], cs) / np.dot(cs, cs))
        r = float(np.sum((g[m] - s * cs) ** 2))
        if r < best[1]:
            best = (int(d), r, s)
    return best[0], best[2], int(m.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(Path(__file__).resolve().parents[2]))
    ap.add_argument("--max-days-per-year", type=int, default=40)
    a = ap.parse_args()
    R = Path(a.root) / "04_results" / "ce1"
    out = R / "WP61"
    out.mkdir(parents=True, exist_ok=True)
    checks = {}

    def check(name, ok, detail=""):
        checks[name] = dict(ok=bool(ok), detail=str(detail))
        print(("ok    " if ok else "FAIL  ") + name + ("" if ok else "  " + str(detail)))

    st_ok, st = selftest()
    check("selftest: synthetic clear days shifted by -150, 0, +150 s are recovered within 15 s", st_ok, st)

    rows, inputs = [], {}
    for site, S in SITES.items():
        loc = pvlib.location.Location(**S)
        p = R / "S1" / f"{site}_geo_5min.parquet"
        inputs[str(p.relative_to(R))] = md5(p)
        geo = pd.read_parquet(p)
        z = geo["zenith"].values
        k = geo["ghi"].values / np.where(geo["ghi_cs_raw"].values > 0, geo["ghi_cs_raw"].values, np.nan)
        geo = geo.assign(k=k)
        samp = geo.iloc[::997]
        zz = loc.get_solarposition(samp.index.tz_localize(S["tz"]))["zenith"].values
        check(f"{site}: stored zenith equals pvlib at the stamp (tz {S['tz']})", np.nanmax(np.abs(zz - samp["zenith"].values)) < 1e-6,
              np.nanmax(np.abs(zz - samp["zenith"].values)))
        for day, d in geo.groupby(geo.index.normalize()):
            dm = d[d["zenith"] < ZEN_MAX]
            if len(dm) < 100 or dm["ghi"].isna().any():
                continue
            ratio = dm["ghi"].sum() / dm["ghi_cs_raw"].sum()
            if not (0.95 <= ratio <= 1.05) or np.nanmax(np.abs(np.diff(dm["k"].values))) >= 0.05:
                continue
            rows.append(dict(site=site, day=day.date().isoformat(), year=day.year, n=len(dm), ratio=ratio))
    D = pd.DataFrame(rows)
    sel = []
    for (site, yr), g in D.groupby(["site", "year"]):
        take = np.unique(np.linspace(0, len(g) - 1, min(a.max_days_per_year, len(g))).round().astype(int))
        sel.append(g.iloc[take])
    D = pd.concat(sel).reset_index(drop=True)
    res = []
    for site, S in SITES.items():
        loc = pvlib.location.Location(**S)
        geo = pd.read_parquet(R / "S1" / f"{site}_geo_5min.parquet")
        for _, r in D[D.site == site].iterrows():
            d = geo.loc[r.day]
            idx = d.index.tz_localize(S["tz"])
            dl, s, n = fit_day(loc, idx, d["ghi"].values.astype(float))
            res.append(dict(site=site, day=r.day, year=r.year, ratio=r.ratio, delta_s=dl, scale=s, n_fit=n))
    T = pd.DataFrame(res)
    T.to_csv(out / "timestamp_days.csv", index=False)
    summ = {}
    for site, g in T.groupby("site"):
        med = float(g.delta_s.median()) / 60
        q = g.delta_s.quantile([0.1, 0.25, 0.75, 0.9]).div(60).round(3).to_dict()
        if abs(med) <= 1.25:
            reading = "instantaneous or centred on the stamp"
        elif -3.75 <= med < -1.25:
            reading = "average over the 5 min ending at the stamp (causal)"
        elif 1.25 < med <= 3.75:
            reading = "average over the 5 min starting at the stamp (NOT causal)"
        else:
            reading = "inconclusive"
        by_year = g.groupby("year").delta_s.median().div(60).round(3).to_dict()
        summ[site] = dict(days=int(len(g)), median_delta_min=round(med, 3), quantiles_min=q, by_year_median_min=by_year,
                          at_grid_edge=int((g.delta_s.abs() >= 600).sum()), reading=reading)
        check(f"{site}: no day at the edge of the search grid", (g.delta_s.abs() < 600).all(), int((g.delta_s.abs() >= 600).sum()))
        check(f"{site}: at least 30 clear days", len(g) >= 30, len(g))
    ok = all(v["ok"] for v in checks.values())
    (out / "timestamp.json").write_text(json.dumps(dict(ok=ok, rule="D-118", summary=summ, checks=checks, inputs_md5=inputs,
                                                         outputs_md5={"timestamp_days.csv": md5(out / "timestamp_days.csv")}),
                                                    indent=1, default=str), encoding="utf-8")
    print(json.dumps(summ, indent=1, default=str))
    print("WP61-TS PASS" if ok else "WP61-TS FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
