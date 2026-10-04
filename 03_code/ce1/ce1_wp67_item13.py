#!/usr/bin/env python3
"""Rule D-129 item 13: binding share of the PV cap, the largest PV falls on the dispatch signal,
and related plant-model checks.

Usage (from the repository root unless the path starts at ce1/, then from 03_code):
    python3 03_code/ce1/ce1_wp67_item13.py [A|B|C|all]
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "utils"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = HERE.parents[1]
R = ROOT / "04_results" / "ce1"
OUT = R / "WP67" / "item13"
FOLDS = [2019, 2020, 2021, 2022, 2023, 2024]
H = 3
CHECKS = {}


def check(name, cond, msg=""):
    CHECKS[name] = dict(ok=bool(cond), msg=str(msg))
    print(("PASS  " if cond else "FAIL  ") + name + "  " + str(msg), flush=True)


def md5(p):
    return hashlib.md5(Path(p).read_bytes()).hexdigest()


def falls(x: np.ndarray, day: np.ndarray, lag: int):
    """x[t] - x[t+lag] where both finite and on the same day; returns (fall, start index)."""
    a, b = x[:-lag], x[lag:]
    ok = np.isfinite(a) & np.isfinite(b) & (day[:-lag] == day[lag:])
    i = np.flatnonzero(ok)
    return a[i] - b[i], i


# ============================================================ A
def part_A():
    import ce1_load as LM
    import ce1_pvmap as PM
    import ce1_s6 as S6
    m = PM.load_map(R / "S1M" / "available_pv_map.npz")
    cap_kw = m["cap"] * m["rated_total"]
    check("A_cap_is_1406.7_kW", abs(cap_kw - 1406.7) < 0.05, round(cap_kw, 2))
    geo = pd.read_parquet(R / "S1" / "yulara_geo_5min.parquet")
    rows = []
    for yr in FOLDS:
        grid = S6.year_grid(f"{yr}-01-01", f"{yr}-12-31 23:55")
        pav = LM.available_pv_on_grid(grid, geo["ghi"].reindex(grid).values, m)
        day = LM.day_ids(grid)
        sc = np.isfinite(pav) & (pav > 0)
        atcap = sc & (pav >= cap_kw * (1 - 1e-9))
        f, i = falls(pav, day, 3)
        thr = np.quantile(f, 0.999)
        top = f >= thr
        rows.append(dict(year=yr, scored_day_steps=int(sc.sum()), at_cap_steps=int(atcap.sum()),
                         at_cap_share=float(atcap.sum() / sc.sum()), fall15_p999_kW=float(thr),
                         top_falls=int(top.sum()), top_falls_from_cap=int(atcap[i[top]].sum()),
                         top_falls_from_cap_share=float(atcap[i[top]].mean()), fall15_max_kW=float(f.max()),
                         fall15_max_from_cap=bool(atcap[i[np.argmax(f)]])))
    a = pd.DataFrame(rows)
    tot = dict(year="pooled", scored_day_steps=int(a.scored_day_steps.sum()), at_cap_steps=int(a.at_cap_steps.sum()))
    tot["at_cap_share"] = tot["at_cap_steps"] / tot["scored_day_steps"]
    tot["top_falls"] = int(a.top_falls.sum()); tot["top_falls_from_cap"] = int(a.top_falls_from_cap.sum())
    tot["top_falls_from_cap_share"] = tot["top_falls_from_cap"] / tot["top_falls"]
    a = pd.concat([a, pd.DataFrame([tot])], ignore_index=True)
    # A3 measured arrays (WP-65 step set)
    import pvlib
    d = PM.load_years()
    full = pd.date_range(d.index.min().floor("D"), d.index.max().ceil("D"), freq="5min", inclusive="left")
    d = d.reindex(full)
    SEL = ("LD", "SS", "SD")
    rated = sum(PM.RATED[x] for x in SEL)
    check("A3_rated_661.1", abs(rated - 661.1) < 1e-6, rated)
    ghi = d["ghi"].to_numpy()
    mp = np.full(len(full), np.nan)
    k = np.isfinite(ghi)
    mp[k] = PM.pv_available_from_ghi(ghi[k], full[k], m) / PM.RATED_TOTAL
    ms = (d[list(SEL)].sum(axis=1, min_count=3) / rated).to_numpy()
    sp = pvlib.solarposition.get_solarposition(full.tz_localize(PM.SITE["tz"]), PM.SITE["lat"], PM.SITE["lon"],
                                               altitude=PM.SITE["alt"])
    up = sp["apparent_elevation"].to_numpy() > 5.0
    both = up & np.isfinite(mp) & np.isfinite(ms)
    capf = m["cap"]
    r3 = []
    for yr in (2023, 2024, None):
        yy = np.ones(len(full), bool) if yr is None else (full.year == yr)
        sel = both & yy
        at = sel & (mp >= capf * (1 - 1e-9))
        ratio = ms[at] / capf
        msx = np.where(sel, ms, np.nan)
        dd = LM.day_ids(full)
        f, i = falls(msx, dd, 3)
        thr = np.quantile(f, 0.999)
        top = f >= thr
        start = msx[i[top]]
        r3.append(dict(year="pooled" if yr is None else yr, steps=int(sel.sum()), map_at_cap_steps=int(at.sum()),
                       map_at_cap_share=float(at.sum() / sel.sum()), measured_over_cap_median=float(np.median(ratio)),
                       measured_over_cap_p99=float(np.quantile(ratio, 0.99)),
                       share_measured_above_1p05cap=float((ratio > 1.05).mean()),
                       measured_fall15_p999=float(thr), top_measured_falls=int(top.sum()),
                       top_falls_start_above_1p05cap=int((start > 1.05 * capf).sum()),
                       top_falls_start_above_1p05cap_share=float((start > 1.05 * capf).mean()),
                       top_falls_start_excess_median=float(np.median(np.maximum(start - capf, 0.0))),
                       top_falls_start_excess_max=float(np.max(np.maximum(start - capf, 0.0)))))
    a3 = pd.DataFrame(r3)
    pooled = a3[a3.year == "pooled"].iloc[0]
    verdict = ("cap does not remove the largest drops" if pooled.top_falls_start_above_1p05cap_share < 0.05
               else "cap removes part of the largest drops")
    return a, a3, dict(verdict_A3=verdict, cap_kW=cap_kw, cap_specific=capf)


# ============================================================ B
def part_B():
    import ce1_dispatch as D
    import ce1_folds as FC
    import ce1_load as LM
    import ce1_s6 as S6
    geo, m, unit96, scales, _ = S6.load_inputs(R / "S1", R / "S1M", R / "S1L")
    D.extend_theta_low()
    rows = S6.row_table()
    hb = [i for i, r in enumerate(rows) if r[0] == "handbook"][0]
    orc = [i for i, r in enumerate(rows) if r[0] == "oracle"][0]
    fb = pd.read_csv(R / "WP55" / "fallback_missing.csv").set_index("fold")
    out = []
    for yr in FOLDS:
        t0 = time.time()
        cache = FC.load_cache_ext(R / "S2" / f"yulara_f{yr}_h3.parquet", R / "S2X" / f"yulara_f{yr}_h3.parquet")
        prep = S6.prepare(cache, H, yr, geo, m, rows)
        P = prep["test"]
        pav, day, grid = P["pav"], P["day"], P["grid"]
        Cor = P["C"][orc]
        scored = np.isfinite(pav)
        cov = scored & np.isfinite(Cor)
        check(f"B_oracle_equals_pav_on_covered_f{yr}", np.array_equal(Cor[cov], pav[cov]))
        full = np.where(scored, pav, np.nan)
        tot = pd.read_parquet(R / "S6" / f"f{yr}_h3_totals.parquet")
        for sname in ("main", "shapeB"):
            sc = S6.SCENARIOS[sname]
            L = S6.load_for(grid, unit96, scales, sc)
            ratings = list(np.asarray(LM.fleet_for_peak(L.max()), float))
            C = np.vstack([P["C"][hb], Cor, full])
            rr = LM.simulate(L, pav, C, day, H, ratings=ratings, min_load=sc["min_load"], fuel=sc["fuel"], battery=sc["battery"])
            t = rr["total"]
            ref = tot[(tot.scenario == sname) & (tot.set == "test")]
            fh_ref = float(ref[ref.policy == "handbook"].fuel_L.iloc[0])
            fo_ref = float(ref[ref.policy == "oracle"].fuel_L.iloc[0])
            check(f"B_handbook_reproduces_S6_f{yr}_{sname}", abs(t["fuel_L"][0] / fh_ref - 1) < 1e-9, (t["fuel_L"][0], fh_ref))
            check(f"B_oracle_reproduces_S6_f{yr}_{sname}", abs(t["fuel_L"][1] / fo_ref - 1) < 1e-9, (t["fuel_L"][1], fo_ref))
            check(f"B_full_oracle_zero_ue_f{yr}_{sname}", float(t["ue_kWh"][2]) < 1e-9, t["ue_kWh"][2])
            nfb = int(t["fallback_steps"][1])
            check(f"B_fallback_equals_WP55_f{yr}_{sname}", nfb == int(fb.loc[yr, "fallback_steps"]) == int(t["fallback_steps"][0]),
                  (nfb, int(fb.loc[yr, "fallback_steps"])))
            check(f"B_full_oracle_no_fallback_f{yr}_{sname}", int(t["fallback_steps"][2]) == 0, t["fallback_steps"][2])
            fh, fo, ff = (float(x) for x in t["fuel_L"][:3])
            c_as, c_full = 100 * (fh - fo) / fh, 100 * (fh - ff) / fh
            out.append(dict(fold=yr, scenario=sname, fallback_steps=nfb, scored_steps=int(scored.sum()),
                            fuel_handbook_L=fh, fuel_oracle_L=fo, fuel_oracle_full_L=ff,
                            ceiling_as_run_pct=c_as, ceiling_full_pct=c_full, removed_pts=c_full - c_as,
                            removed_share=(c_full - c_as) / c_full if c_full > 0 else np.nan))
        print(f"fold {yr} done {time.time() - t0:.0f}s", flush=True)
    return pd.DataFrame(out)


# ============================================================ C
def part_C():
    import ce1_causal as K
    import ce1_sensor as SENS
    # C1 (ce1_preprocess.sensor_checks (a), unchanged construction)
    grid = K.load_unfilled_raw("yulara")
    p1, ghr = grid.attrs["p1_grid"], grid.attrs["ghr_grid"]
    grid.attrs.clear()
    geo = pd.read_parquet(R / "S1" / "yulara_geo_5min.parquet")
    elev = 90 - geo["zenith"]
    r = (p1 / ghr).reindex(geo.index).where((elev > 10) & (p1.reindex(geo.index) > 30) & (ghr.reindex(geo.index) > 0.5))
    dr = r.groupby(geo.index.normalize()).median().dropna()
    obs = dr > 4
    rule = pd.Series(SENS.exchanged(dr.index + pd.Timedelta("12h")), index=dr.index)
    mism = int((obs != rule).sum())
    s1 = json.load(open(R / "_runs" / "S1.detail.json"))["checks"]["sensor_exchange_rule_matches_every_day"]["msg"]
    check("C1_matches_S1_stamp", s1 == f"({len(dr)} classifiable days, {mism} mismatches)", (len(dr), mism, s1))
    c1 = pd.DataFrame(dict(day=dr.index.date, ratio_median=dr.values, observed_exchanged=obs.values, rule_exchanged=rule.values))
    # C2 (audit a15 construction)
    import pvlib
    import warnings
    warnings.filterwarnings("ignore")
    a = pd.read_parquet(R / "S1" / "asp_geo_5min.parquet")
    lat, lon, alt = -23.7621, 133.8745, 545
    t = a.index.tz_localize("Australia/Darwin")
    sp = pvlib.solarposition.get_solarposition(t, lat, lon, alt)
    cs = pvlib.location.Location(lat, lon, "Australia/Darwin", alt).get_clearsky(t, model="ineichen", solar_position=sp)
    for c in ["apparent_zenith", "azimuth", "apparent_elevation"]:
        a[c] = sp[c].values
    for c in ["ghi", "dni", "dhi"]:
        a["cs_" + c] = cs[c].values
    a["dni_extra"] = pvlib.irradiance.get_extra_radiation(t).values
    k = a.ghi / a.cs_ghi.where(a.cs_ghi > 300)
    v = k.rolling(12, min_periods=6).std()
    cm = k.between(0.6, 1.5) & (v < 0.03) & (a.apparent_elevation > 15) & (a.ghi > 50)
    cands = [(b, az) for b in range(0, 21, 1) for az in range(270, 451, 15)]
    rows = []
    for y in [2020, 2021, 2022, 2023, 2024]:
        s = a[cm & (a.index.year == y)]
        key = [s.index.month, (s.index.hour * 60 + s.index.minute) // 60]
        yy = np.log(s["ghi"])
        best = []
        for b, az in cands:
            p = pvlib.irradiance.get_total_irradiance(b, az % 360, s.apparent_zenith, s.azimuth, s.cs_dni, s.cs_ghi, s.cs_dhi,
                                                      dni_extra=s.dni_extra, model="haydavies", albedo=0.25)["poa_global"]
            lr = (yy - np.log(p)).groupby(key).median()
            best.append((float(lr.std(ddof=0)), b, az % 360, float(np.exp(lr.mean()))))
        best.sort()
        hz = [q for q in best if q[1] == 0][0]
        rows.append(dict(year=y, n=int(len(s)), best_rms=best[0][0], tilt_deg=best[0][1], azimuth_deg=best[0][2],
                         scale=best[0][3], horizontal_rms=hz[0]))
        print(rows[-1], flush=True)
    c2 = pd.DataFrame(rows)
    northish = all(min(az, 360 - az) <= 30 for az in c2.azimuth_deg)
    check("C2_tilt_4_to_6_north", int(c2.tilt_deg.min()) == 4 and int(c2.tilt_deg.max()) == 6 and northish,
          (c2.tilt_deg.tolist(), c2.azimuth_deg.tolist()))
    return c1, c2, dict(classifiable_days=len(dr), mismatches=mism)


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    OUT.mkdir(parents=True, exist_ok=True)
    meta = dict(script_md5=md5(__file__), rule="docs/REGISTERED_RULES.md (written before any number)")
    if which in ("A", "all"):
        a, a3, info = part_A()
        a.to_csv(OUT / "cap_binding.csv", index=False)
        a3.to_csv(OUT / "cap_measured.csv", index=False)
        meta["A"] = info
        print(a.to_string()); print(a3.to_string()); print(info)
    if which in ("B", "all"):
        b = part_B()
        b.to_csv(OUT / "fallback_ceiling.csv", index=False)
        print(b.to_string())
    if which in ("C", "all"):
        c1, c2, info = part_C()
        c1.to_csv(OUT / "yulara_sensor_rule_days.csv", index=False)
        c2.to_csv(OUT / "dkasc_tilt.csv", index=False)
        meta["C"] = info
    old = OUT / "item13_checks.json"
    prev = json.loads(old.read_text()) if old.exists() else {"checks": {}}
    prev["checks"].update(CHECKS)
    prev.update({k: v for k, v in meta.items()})
    prev["ok"] = all(c["ok"] for c in prev["checks"].values())
    old.write_text(json.dumps(prev, indent=1, default=str))
    print("ALL OK" if prev["ok"] else "SOME FAIL", len(prev["checks"]))


if __name__ == "__main__":
    main()
