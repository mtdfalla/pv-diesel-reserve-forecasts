#!/usr/bin/env python3
"""Stage AW: daily reserve controller with the latent level clipped from above at h (rule D-117).

    x_{d+1} = min(h, max(theta_lo, x_d + gamma (UE_d - a_d) / a_bar))

primary h = 1.0 (the limit level), descriptive h = 1.01 and 1.05, every other setting as the
S51 primary.

Usage (from the repository root unless the path starts at ce1/, then from 03_code):
    python3 03_code/ce1/ce1_wp61aw.py --root .
    python3 03_code/ce1/ce1_wp61aw.py --selftest
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "utils"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import ce1_adaptive as A  # noqa: E402
import ce1_dispatch as D  # noqa: E402
import ce1_folds as FC  # noqa: E402
import ce1_load as LM  # noqa: E402
import ce1_s6 as S6  # noqa: E402

H_CLIPS = (np.inf, 1.0, 1.01, 1.05)  # inf = S51 reproduction check, 1.0 primary (D-117)
H_PRIMARY_CLIP = 1.0
TGT = LM.UE_TARGET
CP = ["icp", "mondrian", "cqr", "mondrian_cqr"]
QP = ["quantile", "cqr", "mondrian_cqr"]
XFIT = {"cqr": None, "mondrian_cqr": None}  # filled from WP57/cases.csv at reading time


def md5(p: Path) -> str:
    h = hashlib.md5()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def log(m):
    print(f"[AW] {m}", flush=True)


# ------------------------------------------------------------------ clipped controller
# Specs carry the clip in the theta_lo slot as a tuple (lo, h). The wrappers unpack it, so
# ce1_adaptive.run_paths passes it through untouched.
def control_clip(ue, load, theta0, gamma, kappa, abar_load, levels, lo):
    lo, h = lo
    nd = len(ue)
    idx = np.empty(nd, np.int64)
    lat = np.empty(nd + 1)
    lat[0] = x = float(theta0)
    abar = kappa * TGT * abar_load
    for d in range(nd):
        idx[d] = A.level_index(levels, max(x, lo))
        if gamma:
            x = min(h, max(lo, x + gamma * (ue[d] - kappa * TGT * load[d]) / abar))
        lat[d + 1] = x
    return idx, lat


def control_lookup_clip(rec_ue, load, theta0, gamma, kappa, abar_load, levels, lo):
    lo, h = lo
    nd = rec_ue.shape[1]
    idx = np.empty(nd, np.int64)
    x = float(theta0)
    abar = kappa * TGT * abar_load
    for d in range(nd):
        i = A.level_index(levels, max(x, lo))
        idx[d] = i
        if gamma:
            x = min(h, max(lo, x + gamma * (rec_ue[i, d] - kappa * TGT * load[d]) / abar))
    return idx


def check_lookup_matches_s51_form():
    """The S51 control_lookup must have the form reproduced above (guard against drift)."""
    import inspect
    src = inspect.getsource(A.control_lookup)
    return "rec_ue[i, d]" in src and "max(lo, x + gamma" in src


# ------------------------------------------------------------------ one fold
def run_fold(yr, prep, rows, unit96, scales, rec, check):
    h = A.H_PRIMARY
    P = prep["test"]
    grid, pav, day = P["grid"], P["pav"], P["day"]
    pol_C = {}
    for pol in A.POLICIES:
        ix = [i for i, r in enumerate(rows) if r[0] == pol]
        lv = np.array([rows[i][1] for i in ix])
        o = np.argsort(lv)
        pol_C[pol] = (lv[o], P["C"][np.array(ix)[o]])
        fin = pol_C[pol][0] < 1.0 if pol != "frac_pv" else np.ones(len(o), bool)
        ok, bad = A.causal_ok(pol_C[pol][1][fin], grid, h)
        check(f"causal_f{yr}_{pol}", ok, f"({bad})")
    cells, lat_all = [], {}
    for sname in A.SCEN_PRIMARY:
        sc = S6.SCENARIOS[sname]
        z, sel = rec[sname]
        rmap, amap = A.record_rows(z)
        L = S6.load_for(grid, unit96, scales, sc)
        sim_kw = dict(ratings=LM.fleet_for_peak(L.max()), min_load=sc["min_load"], fuel=sc["fuel"],
                      battery=sc["battery"])
        abar_load = A.h2_mean_daily_load(prep["H2"], unit96, scales, sc)
        load_d = z[f"{sname}__load_kWh"][amap["handbook"]].astype(float)
        rec_ue, specs = {}, []
        for pol in A.POLICIES:
            levels = pol_C[pol][0]
            rix = np.array([rmap[(pol, round(float(x), 6))] for x in levels])
            rec_ue[pol] = z[f"{sname}__ue_kWh"][rix].astype(float)
            t0 = float(sel["policies"][pol]["A_R"]["param"])
            lo = float(levels.min())
            specs += [(pol, A.GAMMA, A.KAPPA, t0, (lo, hc)) for hc in H_CLIPS]
        r, idx, lat, passes, work = A.run_paths(L, pav, day, h, sim_kw, pol_C, specs, rec_ue, load_d, abar_load)
        check(f"converged_f{yr}_{sname}", 1 <= passes < 400 and bool(np.isfinite(r["total"]["fuel_L"]).all()),
              f"({passes} passes, {work} row-years)")
        dl = r["daily"]["load_kWh"]
        check(f"load_identity_f{yr}_{sname}", np.allclose(dl, load_d[None, :], rtol=1e-5, atol=1e-3))
        hb = amap["handbook"]
        hb_fuel = float(z[f"{sname}__fuel_L"][hb].astype(float).sum())
        tot = r["total"]
        for k, (pol, g, kp, t0, (lo, hc)) in enumerate(specs):
            levels = pol_C[pol][0]
            lv = levels[idx[k]]
            lim = lv >= 1.0 - A.LEVEL_TOL
            if np.isfinite(hc):
                check(f"latent_le_clip_f{yr}_{sname}_{pol}_h{hc}", float(lat[k].max()) <= hc + 1e-12)
            if lim.any():
                check(f"limit_zero_ue_f{yr}_{sname}_{pol}_h{hc}", float(np.max(r["daily"]["ue_kWh"][k][lim])) <= 1e-9)
            ue_d = r["daily"]["ue_kWh"][k]
            cells.append(dict(fold=yr, scenario=sname, policy=pol, clip_h=hc, theta0=t0,
                              fuel_L=float(tot["fuel_L"][k]), ue_kWh=float(tot["ue_kWh"][k]),
                              load_kWh=float(tot["load_kWh"][k]), ue_share=float(tot["ue_share"][k]),
                              fuel_handbook=hb_fuel,
                              saving_pct=100.0 * (hb_fuel - float(tot["fuel_L"][k])) / hb_fuel,
                              starts=float(tot["starts"][k]) if "starts" in tot else np.nan,
                              ue_days=int((ue_d > 1e-9).sum()), level_mean=float(lv.mean()),
                              days_at_limit=int(lim.sum()), level_changes=int((np.diff(idx[k]) != 0).sum()),
                              latent_max=float(lat[k].max()), passes=passes))
            lat_all[(sname, pol, hc)] = lat[k].astype(np.float32)
    return cells, lat_all


# ------------------------------------------------------------------ reading (D-117)
def reading(c: pd.DataFrame, xfit: pd.DataFrame) -> dict:
    p = c[np.isclose(c["clip_h"], H_PRIMARY_CLIP)]
    p = p.assign(meets=p.ue_share <= TGT + 1e-15)
    cp = p[p.policy.isin(CP)]
    xf = xfit[(xfit.selection == "b_xfit_0.001") & xfit.cp]
    xmed = xf.groupby("policy").saving_pct.median()
    med = cp.groupby("policy").saving_pct.median()
    cond_i = int(cp.meets.sum()) >= 47
    cond_ii = all(med[q] >= xmed[q] - 1e-12 for q in ("cqr", "mondrian_cqr"))
    q = p[p.policy.isin(QP)]
    return dict(n_cp=int(len(cp)), met_cp=int(cp.meets.sum()),
                met_cp_by_year={int(k): int(v) for k, v in cp.groupby("fold").meets.sum().items()},
                median_saving_cp=float(cp.saving_pct.median()),
                median_saving_by_policy={k: float(v) for k, v in med.items()},
                xfit_median_by_policy={k: float(xmed[k]) for k in CP},
                met_quantile=int(q.meets.sum()), n_quantile=int(len(q)),
                median_saving_quantile=float(q.saving_pct.median()),
                days_at_limit_median_quantile=float(q.days_at_limit.median()),
                cond_i=bool(cond_i), cond_ii=bool(cond_ii), fires=bool(cond_i and cond_ii))


def selftest(verbose=True):
    res = {}

    def ok(n, c):
        res[n] = bool(c)
        if verbose:
            print(("PASS " if c else "FAIL ") + n)
    ok("s51_lookup_form_unchanged", check_lookup_matches_s51_form())
    rng = np.random.default_rng(1)
    levels = np.array([0.9, 0.95, 0.99, 0.999, 1.0])
    ue = np.where(rng.random(200) < 0.05, 200.0, 0.0)
    load = np.full(200, 5e4)
    i0, l0 = A.control(ue, load, 0.95, 0.005, 0.5, 5e4, levels, 0.9)
    i1, l1 = control_clip(ue, load, 0.95, 0.005, 0.5, 5e4, levels, (0.9, np.inf))
    ok("inf_clip_equals_s51_control", np.array_equal(i0, i1) and np.array_equal(l0, l1))
    i2, l2 = control_clip(ue, load, 0.95, 0.005, 0.5, 5e4, levels, (0.9, 1.0))
    ok("clip_bounds_latent", l2.max() <= 1.0 + 1e-12 and l0.max() > 1.0)
    ok("clip_never_raises_level", np.all(levels[i2] <= levels[i0] + 1e-12))
    rec = np.tile(ue, (len(levels), 1))
    j0 = A.control_lookup(rec, load, 0.95, 0.005, 0.5, 5e4, levels, 0.9)
    j1 = control_lookup_clip(rec, load, 0.95, 0.005, 0.5, 5e4, levels, (0.9, np.inf))
    ok("inf_clip_equals_s51_lookup", np.array_equal(j0, j1))
    # a constructed reading: fires only when both conditions hold
    xf = pd.DataFrame(dict(selection="b_xfit_0.001", cp=True, policy=CP * 2, saving_pct=[0, 0, 0.26, 0.275] * 2))
    base = pd.DataFrame(dict(fold=np.repeat(range(12), 4), clip_h=1.0, policy=CP * 12, ue_share=1e-6,
                             saving_pct=0.3, days_at_limit=0))
    ok("reading_fires_on_constructed_pass", reading(base, xf)["fires"])
    b2 = base.copy(); b2.loc[b2.policy == "cqr", "saving_pct"] = 0.2
    ok("reading_silent_when_cqr_short", not reading(b2, xf)["fires"])
    b3 = base.copy(); b3.loc[:1, "ue_share"] = 1.0
    ok("reading_silent_when_two_miss", not reading(b3, xf)["fires"])
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--root", default=".")
    ap.add_argument("--out", default="AW")
    a = ap.parse_args()
    if a.selftest:
        r = selftest()
        sys.exit(0 if all(r.values()) else 1)
    t0 = time.time()
    R = Path(a.root) / "04_results" / "ce1"
    out, runs = R / a.out, R / "_runs"
    out.mkdir(parents=True, exist_ok=True); runs.mkdir(parents=True, exist_ok=True)
    checks = {}

    def check(name, cond, msg=""):
        checks[name] = dict(ok=bool(cond), msg=msg)
        if not cond or not name.startswith(("causal_", "limit_", "latent_")):
            log(f"{'PASS' if cond else 'FAIL'}  {name}  {msg}")

    st = selftest(verbose=False)
    check("selftests", all(st.values()), f"({sum(st.values())}/{len(st)})")
    # the clipped controller replaces S51's two functions for this run only
    A.control, A.control_lookup = control_clip, control_lookup_clip
    geo, m, unit96, scales, _ = S6.load_inputs(R / "S1", R / "S1M", R / "S1L")
    D.extend_theta_low()
    rows = S6.row_table()
    inputs, cells, lats = {}, [], {}
    for yr in A.FOLDS:
        t1 = time.time()
        rec = {}
        for sn in A.SCEN_PRIMARY:
            d = R / "S6E"
            fz, fs = d / f"f{yr}_h3_daily.npz", d / f"f{yr}_h3_select.json"
            inputs[str(fz.relative_to(R))] = md5(fz); inputs[str(fs.relative_to(R))] = md5(fs)
            rec[sn] = (np.load(fz), json.load(open(fs))["selections"][sn])
        cache = FC.load_cache_ext(R / "S2" / f"yulara_f{yr}_h3.parquet", R / "S2X" / f"yulara_f{yr}_h3.parquet")
        prep = S6.prepare(cache, 3, yr, geo, m, rows)
        c, la = run_fold(yr, prep, rows, unit96, scales, rec, check)
        cells += c
        lats.update({(yr,) + k: v for k, v in la.items()})
        log(f"fold {yr}: {len(c)} cells, {time.time() - t1:.0f} s")
    C = pd.DataFrame(cells)
    # h = inf reproduces the S51 primary cells
    s51 = pd.read_csv(R / "S51" / "cells.csv", float_precision="round_trip")  # exact float read-back
    s51 = s51[(s51.h == 3) & s51.primary & s51.scenario.isin(A.SCEN_PRIMARY)]
    inf = C[~np.isfinite(C["clip_h"])]
    mg = inf.merge(s51, on=["fold", "scenario", "policy"], suffixes=("", "_s51"))
    check("inf_clip_reproduces_s51_rows", len(mg) == len(inf) == len(s51), f"({len(mg)} of {len(s51)})")
    check("inf_clip_reproduces_s51_fuel", bool(np.array_equal(mg.fuel_L.values, mg.fuel_L_s51.values)),
          f"(max diff {np.max(np.abs(mg.fuel_L - mg.fuel_L_s51)):.3g} L)")
    check("inf_clip_reproduces_s51_ue", bool(np.array_equal(mg.ue_kWh.values, mg.ue_kWh_s51.values)),
          f"(max diff {np.max(np.abs(mg.ue_kWh - mg.ue_kWh_s51)):.3g} kWh)")
    C.to_csv(out / "cells.csv", index=False)
    np.savez_compressed(out / "latent.npz", **{f"f{k[0]}|{k[1]}|{k[2]}|{k[3]}": v for k, v in lats.items()})
    xfit = pd.read_csv(R / "WP57" / "cases.csv")
    rd = reading(C, xfit)
    json.dump(rd, open(out / "reading.json", "w"), indent=1)
    log(f"READING: {json.dumps(rd)}")
    ok = all(c["ok"] for c in checks.values())
    nfail = sum(not c["ok"] for c in checks.values())
    log(f"{len(checks)} checks, {nfail} failed; {time.time() - t0:.0f} s")
    outs = {p.name: md5(p) for p in sorted(out.glob("*")) if p.is_file()}
    json.dump(dict(stage=a.out, ok=ok, error="" if ok else f"{nfail} checks failed",
                   finished=time.strftime("%Y-%m-%dT%H:%M:%S"), seconds=round(time.time() - t0, 1),
                   inputs=inputs, outputs=outs,
                   code={"ce1_wp61aw.py": md5(Path(__file__)), "ce1_adaptive.py": md5(HERE / "ce1_adaptive.py")},
                   env=dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__,
                            host="Linux")),
              open(runs / f"{a.out}.json", "w"), indent=2)
    json.dump(dict(stage=a.out, checks=checks, n_checks=len(checks)), open(runs / f"{a.out}.detail.json", "w"),
              indent=1, default=float)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
