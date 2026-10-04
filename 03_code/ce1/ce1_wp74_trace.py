#!/usr/bin/env python3
"""Paired lead of the quantile policies over the persistence fraction, unserved-energy event sizes,
the 2025 record and fold tables, and the DKASC sensor tilt 2020 to 2025, from existing outputs.
Writes 04_results/ce1/WP74/.

Usage (from the repository root unless the path starts at ce1/, then from 03_code):
    python3 03_code/ce1/ce1_wp74_trace.py
"""
from __future__ import annotations

import hashlib
import json
import statistics as st
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path.cwd()
R = ROOT / "04_results" / "ce1"
OUT = R / "WP74"
QP = ("raw_q", "cqr", "mondrian_cqr")
CHECKS: dict = {}


def check(name, cond, msg=""):
    CHECKS[name] = dict(ok=bool(cond), msg=str(msg))
    print(("PASS  " if cond else "FAIL  ") + name + "  " + str(msg), flush=True)
    return bool(cond)


def md5(p: Path) -> str:
    return hashlib.md5(p.read_bytes()).hexdigest()


# ------------------------------------------------------------------ T1
def t1():
    rows = []
    for tag, f, con in (("plain", R / "S69" / "cases.csv", "plain"), ("n1a", R / "S69b" / "cases.csv", "n1a")):
        c = pd.read_csv(f)
        x = c[(c.construction == con) & (c.rule == "XH") & (c.selection == "b_xfit_0.001")]
        p = x[x.policy == "pfr"].set_index(["fold", "shape"]).saving_pct
        assert len(p) == 12
        for pol in QP + ("pooled",):
            q = x[x.policy.isin(QP)] if pol == "pooled" else x[x.policy == pol]
            q = q.set_index(["fold", "shape"])
            d = q.saving_pct - p.reindex(q.index).values
            rows.append(dict(construction=tag, policy=pol, cases=len(d), paired_median=float(d.median()),
                             above_0p1=int((d > 0.1).sum()), q_median=float(q.saving_pct.median()),
                             pfr_median=float(p.median()), diff_of_medians=float(q.saving_pct.median() - p.median()),
                             pfr_met=int(x[x.policy == "pfr"].meets.sum()), q_met=int(q.meets.sum())))
    L = pd.DataFrame(rows)
    g = L.set_index(["construction", "policy"])
    check("T1_gate_N225_diff_of_medians_0.28", round(g.loc[("plain", "raw_q"), "diff_of_medians"], 2) == 0.28,
          g.loc[("plain", "raw_q"), "diff_of_medians"])
    check("T1_gate_N226_diff_of_medians_0.23", round(g.loc[("n1a", "mondrian_cqr"), "diff_of_medians"], 2) == 0.23,
          g.loc[("n1a", "mondrian_cqr"), "diff_of_medians"])
    return L


# ------------------------------------------------------------------ T2
def t2():
    e = pd.read_csv(R / "WP67" / "sim" / "f2024_events.csv")
    big = e[(e.peak_kW - 883.599078).abs() < 0.01]
    occ = big.groupby(["scenario", "policy"]).selection.apply(lambda s: ",".join(sorted(s))).reset_index()
    c = pd.read_csv(R / "WP67" / "sim" / "f2024_cases.csv")
    cs = c[(c.selection == "b_xfit_0.001") & (c.policy.isin(QP)) & (c.scenario == "shapeB")].copy()
    cs["allow_kWh"] = 2e-5 * cs.load_kWh
    cs["allow_per_event_kWh"] = cs.allow_kWh / cs.events
    ev = big.iloc[0]
    cs["event_kWh"] = float(ev.energy_kWh)
    cs["x_allow_per_event"] = cs.event_kWh / cs.allow_per_event_kWh
    cs["share_annual_allow"] = cs.event_kWh / cs.allow_kWh
    import glob
    per = []
    for fp in sorted(glob.glob(str(R / "WP67" / "sim" / "f20*_cases.csv"))):
        z = pd.read_csv(fp)
        z = z[(z.selection == "b_xfit_0.001") & (z.policy.isin(QP))]
        per += list(2e-5 * z.load_kWh / z.events)
    check("T2_gate_N233_25kWh", len(per) == 36 and round(st.median(per)) == 25, (len(per), st.median(per)))
    check("T2_event_steps_energy", int(ev.steps) == 2 and round(float(ev.energy_kWh), 1) == 77.9
          and float(ev.online_kW) == 1000.0, (ev.steps, ev.energy_kWh, ev.online_kW))
    cols = ["fold", "scenario", "policy", "selection", "load_kWh", "events", "ue_kWh", "allow_kWh", "allow_per_event_kWh",
            "event_kWh", "x_allow_per_event", "share_annual_allow"]
    return cs[cols], occ, ev


# ------------------------------------------------------------------ T3
def grid_numbers(geo: pd.DataFrame) -> dict:  # ce1_wp21_data.grid_numbers, unchanged
    day = geo["is_day"].astype(bool)
    valid = geo["ghi"].notna()
    g = geo.loc[day, "ghi"]
    rank = day[day].groupby(day[day].index.date).cumcount()
    first = (rank < 12).reindex(g.index).values
    e_all = float(g.fillna(0).sum())
    e_first = float(g[first].fillna(0).sum())
    return dict(slots=int(len(geo)), valid=int(valid.sum()), day_slots=int(day.sum()),
                day_valid=int((day & valid).sum()), first=str(geo.index.min()), last=str(geo.index.max()),
                first_hour_share=e_first / e_all)


def missing_rows(geo, s2path, site, fo):  # ce1_wp59c  block, unchanged
    s2 = pd.read_parquet(s2path, columns=["split", "g"])
    lab_src = s2[s2["split"] == "test"]["g"]
    y = geo.loc[str(fo)]
    day = y["is_day"].values.astype(bool)
    miss = y["ghi"].isna().values & day
    lab = lab_src.reindex(y.index).ffill(limit=12).fillna("unlabelled").values
    ok = bool(y["is_day"].reindex(lab_src.index).fillna(False).all())
    out = []
    for L in ("clear", "transitional", "cloudy", "unlabelled", "all"):
        m_ = day & ((lab == L) if L != "all" else True)
        out.append(dict(site=site, fold=fo, label=L, daytime_steps=int(m_.sum()), missing=int((miss & m_).sum()),
                        share=float((miss & m_).sum() / max(m_.sum(), 1))))
    return out, ok


def t3():
    dn = json.load(open(R / "WP21" / "data_numbers.json"))
    geo = {s: pd.read_parquet(R / "S1E" / f"{s}_geo_5min.parquet") for s in ("yulara", "asp")}
    rec, ok_all = [], True
    for s in ("yulara", "asp"):
        G = geo[s]
        old = dn[f"{s}_grid"]
        a = grid_numbers(G[G.index <= pd.Timestamp(old["last"])])
        same = all(a[k] == old[k] for k in ("slots", "valid", "day_slots", "day_valid", "first", "last")) \
            and abs(a["first_hour_share"] - old["first_hour_share"]) < 1e-12
        ok_all &= check(f"T3_gate_S1record_{s}_to2024", same, {k: (a[k], old[k]) for k in ("slots", "valid", "first_hour_share")})
        b = grid_numbers(G[G.index <= pd.Timestamp("2025-12-31 23:55")])
        rec.append(dict(site=s, period="to2024", **a))
        rec.append(dict(site=s, period="to2025", **b))
    # S1.3 folds
    s1 = json.load(open(R / "S1" / "scalars.json")); s1e = json.load(open(R / "S1E" / "scalars.json"))
    r1 = pd.read_csv(R / "S1" / "retained_counts.csv"); r1e = pd.read_csv(R / "S1E" / "retained_counts.csv")
    same = True
    for s in ("yulara", "asp"):
        for fo, v in s1[s].items():
            same &= all(s1e[s][fo][k] == v[k] for k in ("train_max", "calib", "scalar", "n_band"))
            a = r1[(r1.site == s) & (r1.fold_test == int(fo))].sort_values(["frame", "h_min"]).reset_index(drop=True)
            b = r1e[(r1e.site == s) & (r1e.fold_test == int(fo))].sort_values(["frame", "h_min"]).reset_index(drop=True)
            fr = sorted(set(a.frame) & set(b.frame))  # S1 also holds r1_legacy (fold 2024), S1E does not
            same &= {"ce1_locf6", "ce1_none"} <= set(fr)
            a = a[a.frame.isin(fr)].reset_index(drop=True); b = b[b.frame.isin(fr)].reset_index(drop=True)
            same &= bool((a[["frame", "h_min", "train", "calib", "test"]].values == b[["frame", "h_min", "train", "calib", "test"]].values).all())
    ok_all &= check("T3_gate_S1folds_S1E_equals_S1", same)
    fold = []
    for s in ("yulara", "asp"):
        v = s1e[s]["2025"]
        t = r1e[(r1e.site == s) & (r1e.fold_test == 2025) & (r1e.frame == "ce1_locf6")].set_index("h_min")
        fold.append(dict(site=s, test=2025, train=f"{2016 if s == 'yulara' else 2020}--{v['train_max']}", calib=v["calib"],
                         scalar=v["scalar"], n_band=v["n_band"], **{f"test_h{h}": int(t.loc[h, "test"]) for h in (5, 15, 30, 60)},
                         carried_pct=100 * float(t.loc[15, "test_carried"]) / float(t.loc[15, "test"])))
    # S1.1 missing
    w = pd.read_csv(R / "WP59C" / "missing.csv")
    gate = True
    for s in ("yulara", "asp"):
        rr, ok = missing_rows(geo[s], R / "S2" / f"{s}_f2024_h1.parquet", s, 2024)
        for r in rr:
            o = w[(w.site == s) & (w.fold == 2024) & (w.label == r["label"])].iloc[0]
            gate &= int(o.daytime_steps) == r["daytime_steps"] and int(o.missing) == r["missing"] and abs(float(o.share) - r["share"]) < 1e-12
        gate &= ok
    ok_all &= check("T3_gate_S1missing_2024_reproduces_WP59C", gate)
    mis = []
    for s in ("yulara", "asp"):
        rr, ok = missing_rows(geo[s], R / "S2E" / "f2025" / f"{s}_f2025_h1.parquet", s, 2025)
        check(f"T3_{s}_f2025_scored_issue_times_are_daytime", ok)
        mis += rr
    M = pd.DataFrame(mis)
    rc = pd.read_csv(R / "WP72" / "record_counts.csv")
    for s in ("yulara", "asp"):
        a = M[(M.site == s) & (M.label == "all")].iloc[0]
        b = rc[(rc.site == s) & (rc.period == "2025")].iloc[0]
        check(f"T3_{s}_2025_all_equals_record_counts", int(a.daytime_steps) == int(b.daytime_steps) and int(a.missing) == int(b.daytime_missing),
              (a.daytime_steps, a.missing, b.daytime_steps, b.daytime_missing))
    return (pd.DataFrame(rec), pd.DataFrame(fold), M) if ok_all else (None, None, None)


# ------------------------------------------------------------------ T4
def t4():
    import pvlib
    warnings.filterwarnings("ignore")
    a = pd.read_parquet(R / "S1E" / "asp_geo_5min.parquet")
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
    for y in [2020, 2021, 2022, 2023, 2024, 2025]:
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
    T = pd.DataFrame(rows)
    old = pd.read_csv(R / "WP67" / "item13" / "dkasc_tilt.csv")
    new = T[T.year <= 2024].reset_index(drop=True)
    g = all((new[c] == old[c]).all() for c in ("year", "n", "tilt_deg", "azimuth_deg")) and \
        all(np.allclose(new[c], old[c], rtol=1e-9, atol=0) for c in ("best_rms", "scale", "horizontal_rms"))
    if not check("T4_gate_2020_2024_reproduce_dkasc_tilt", g, new.to_dict("list")):
        return None, None
    y5 = T[T.year == 2025].iloc[0]
    north = min(y5.azimuth_deg, 360 - y5.azimuth_deg) <= 30
    holds = 4 <= y5.tilt_deg <= 6 and north and y5.best_rms <= 1.25 * old.best_rms.max()
    check("T4_reading_2025_sensor_state_holds", holds, (int(y5.tilt_deg), int(y5.azimuth_deg), y5.best_rms, old.best_rms.max()))
    return T, bool(holds)


# ------------------------------------------------------------------ T5
def main():
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    L = t1(); L.to_csv(OUT / "leads.csv", index=False); print(L.to_string())
    cs, occ, ev = t2(); cs.to_csv(OUT / "event_r1p1.csv", index=False); occ.to_csv(OUT / "event_r1p1_occurrence.csv", index=False)
    print(cs.to_string()); print(occ.to_string())
    rec, fold, mis = t3()
    if rec is not None:
        rec.to_csv(OUT / "s1_record_2025.csv", index=False); fold.to_csv(OUT / "s1_folds_2025.csv", index=False)
        mis.to_csv(OUT / "s1_missing_2025.csv", index=False)
        print(rec.to_string()); print(fold.to_string()); print(mis.to_string())
    T, holds = t4()
    if T is not None:
        T.to_csv(OUT / "dkasc_tilt_2020_2025.csv", index=False)
    meta = dict(script_md5=md5(Path(__file__)), run=time.strftime("%Y-%m-%d %H:%M:%S"), seconds=round(time.time() - t0),
                python=sys.version.split()[0], pandas=pd.__version__, numpy=np.__version__,
                inputs={str(p.relative_to(R)): md5(p) for p in (
                    R / "S69" / "cases.csv", R / "S69b" / "cases.csv", R / "WP67" / "sim" / "f2024_events.csv",
                    R / "WP67" / "sim" / "f2024_cases.csv", R / "S1E" / "yulara_geo_5min.parquet", R / "S1E" / "asp_geo_5min.parquet",
                    R / "S1E" / "scalars.json", R / "S1E" / "retained_counts.csv", R / "S2E" / "f2025" / "yulara_f2025_h1.parquet",
                    R / "S2E" / "f2025" / "asp_f2025_h1.parquet", R / "S2" / "yulara_f2024_h1.parquet", R / "S2" / "asp_f2024_h1.parquet",
                    R / "WP59C" / "missing.csv", R / "WP21" / "data_numbers.json", R / "WP67" / "item13" / "dkasc_tilt.csv")},
                t4_reading_holds=holds, checks=CHECKS,
                ok=all(c["ok"] for k, c in CHECKS.items() if k != "T4_reading_2025_sensor_state_holds"))
    (OUT / "checks.json").write_text(json.dumps(meta, indent=1, default=str))
    print("ALL GATES OK" if meta["ok"] else "SOME FAIL", len(CHECKS))


if __name__ == "__main__":
    main()
