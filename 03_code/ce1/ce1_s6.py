"""Stage S6 and its variants (S6P, S6B, S6X, S6P2): multi-year load-based dispatch of every
reserve policy, in every fold, horizon and registered scenario, with the selection protocols
A-R and A-C. Scores for selection are fitted on the first half of the calibration year and
scored on its second half; the selected level is applied to the test year with scores fitted
on the whole calibration year. Rules: D-080, D-084.

Usage (from the repository root unless the path starts at ce1/, then from 03_code):
    python ce1/ce1_s6.py --stage S6 [--smoke] [--only 2024:3]
    python ce1/ce1_s6.py --selftest
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "utils"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import ce1_dispatch as D  # noqa: E402
import ce1_folds as FC  # noqa: E402
import ce1_load as LM  # noqa: E402
import ce1_pvmap as PM  # noqa: E402

FOLDS = [2019, 2020, 2021, 2022, 2023, 2024]
HORIZONS = [1, 3, 6, 12]
ANCHORS = ["handbook", "deterministic", "oracle"]
SCENARIOS = {
    "main":      dict(shape="A", scale="primary", min_load=LM.MIN_LOAD, battery=False, fuel="cat_c32"),
    "shapeB":    dict(shape="B", scale="primary", min_load=LM.MIN_LOAD, battery=False, fuel="cat_c32"),
    "peak6MW":   dict(shape="A", scale="peak6MW", min_load=LM.MIN_LOAD, battery=False, fuel="cat_c32"),
    "spill":     dict(shape="A", scale="spill_matched", min_load=LM.MIN_LOAD, battery=False, fuel="cat_c32"),
    "minload10": dict(shape="A", scale="primary", min_load=LM.MIN_LOAD_SENS, battery=False, fuel="cat_c32"),
    "battery":   dict(shape="A", scale="primary", min_load=LM.MIN_LOAD, battery=True, fuel="cat_c32"),
    "batteryB":  dict(shape="B", scale="primary", min_load=LM.MIN_LOAD, battery=True, fuel="cat_c32"),
    "fuel_su":   dict(shape="A", scale="primary", min_load=LM.MIN_LOAD, battery=False, fuel="skarstein_uhlen"),
}
ARRAYS_SCEN = dict(shape="A", scale="primary", min_load=LM.MIN_LOAD, battery=False, fuel="cat_c32",
                   realized="arrays")
ARRAYS_FOLDS = (2023, 2024)
UNCURTAILED = ("LD", "SS", "SD")
DAILY_SCEN = ("main", "shapeB", "battery", "batteryB")
DAILY_KEYS = ("fuel_L", "ue_kWh", "curt_kWh", "load_kWh")
TOT_KEYS = ("fuel_L", "co2_t", "ue_kWh", "ue_share", "ue_steps", "curt_kWh", "curt_share", "dump_kWh",
            "thermal_kWh", "pv_load_kWh", "pv_share_of_load", "chg_kWh", "dis_kWh", "pav_kWh",
            "load_kWh", "starts", "online_kW_h", "fallback_steps", "scored_steps")


def md5(p: Path) -> str:
    h = hashlib.md5()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def log(m):
    print(f"[S6] {m}", flush=True)


# ---------------------------------------------------------------- PV map at fixed times
class PVAt:
    """pv_available_from_ghi at a fixed set of timestamps, with the solar geometry computed
    once. Equal to ce1_pvmap.pv_available_from_ghi (self-test), only faster when many GHI
    vectors are mapped at the same times.
    """

    def __init__(self, index, m):
        import pvlib
        self.m = m
        idx = pd.DatetimeIndex(index)
        t = idx.tz_localize(PM.SITE["tz"], ambiguous="NaT", nonexistent="NaT")
        sp = pvlib.solarposition.get_solarposition(t, PM.SITE["lat"], PM.SITE["lon"], PM.SITE["alt"])
        self.zen = np.asarray(sp["apparent_zenith"])
        self.az = np.asarray(sp["azimuth"])
        self.ext = np.asarray(pvlib.irradiance.get_extra_radiation(t))

    def poa(self, ghi):
        import pvlib
        g = np.asarray(ghi, float)
        gz = np.nan_to_num(np.clip(g, 0, None))
        e = pvlib.irradiance.erbs(gz, self.zen, self.ext)
        poa = pvlib.irradiance.get_total_irradiance(
            self.m["plane_tilt"], self.m["plane_az"], self.zen, self.az, np.asarray(e["dni"]), gz,
            np.asarray(e["dhi"]), dni_extra=self.ext, model="haydavies", albedo=PM.ALBEDO)["poa_global"]
        poa = np.clip(np.nan_to_num(np.asarray(poa, float)), 0, None)
        return np.where(np.isnan(g), np.nan, np.where(gz > 0, poa, 0.0))

    def __call__(self, ghi):
        return PM.pv_available(self.poa(ghi), self.m)


class PVBinder:
    """Time-aware PV map for the proxy (ce1_dispatch): bound to the target times t + h of
    whichever row set a sweep evaluates (ce1_dispatch._bound).
    """

    def __init__(self, m, h):
        self.m, self.lead = m, pd.Timedelta(minutes=5 * h)

    def bind(self, ev):
        return PVAt(pd.DatetimeIndex(ev["idx"]) + self.lead, self.m)


# ---------------------------------------------------------------- proxy runs (stage S6P)
PROXY_MAIN = dict(mode="reset", batt="default", ratio=D.RATIO_DEF)


def proxy_configs(yr, h):
    """Proxy configurations: the main one in every fold and horizon; on the primary fold 2024
    the D-071/D-077 variants (carry, settle, strict at every horizon; small and large battery
    in every mode at 5 min) and the cost-ratio sensitivity.
    """
    cf = [dict(PROXY_MAIN)]
    if yr == 2024:
        cf += [dict(PROXY_MAIN, mode=md) for md in ("carry", "settle", "strict")]
        if h == 1:
            cf += [dict(PROXY_MAIN, mode=md, batt=b) for md in D.MODES for b in ("small", "large")]
        cf += [dict(PROXY_MAIN, ratio=r) for r in D.RATIOS if r != D.RATIO_DEF]
    return cf


def run_proxy(cache, h, m, cfgs):
    ca, te = as_dict(cache[cache["split"] == "calib"]), as_dict(cache[cache["split"] == "test"])
    rows, choices, surface = [], [], []
    for cf in cfgs:
        r = D.run_protocols(ca, te, PVBinder(m, h), mode=cf["mode"], ratio=cf["ratio"], batt=cf["batt"],
                            policies=D.ALL_POLICIES)
        rows += r["rows"]; choices += r["choices"]; surface += r["surface"]
    return pd.DataFrame(rows), choices, pd.DataFrame(surface)


# ---------------------------------------------------------------- cache to policy inputs
def as_dict(c: pd.DataFrame) -> dict:
    """ce1_dispatch input dict from cache rows (issue-time index)."""
    return dict(idx=c.index.values, y=c["y"].values.astype(float), p=c["p"].values.astype(float),
                g=c["g"].values, v=c["v"].values.astype(float), gcs=c["gcs"].values.astype(float),
                q={round(float(c_[1:]), 6): c[c_].values.astype(float) for c_ in c.columns
                   if c_.startswith("q")})


THETA_LIMIT = 1.0


def row_table(ratio=D.RATIO_DEF):
    """(policy, param) rows of every batch: the anchors, then every policy's parameter set.
    Every theta policy also carries the limit level theta = 1 (D-085): a one-sided lower
    bound at level 1 is zero PV, i.e. the handbook rule, so the set always holds a level that
    serves the load and every frontier reaches the reliability target.
    """
    rows = [(a, np.nan) for a in ANCHORS]
    for pol in D.ALL_POLICIES:
        rows += [(pol, float(x)) for x in D.param_grid(pol, ratio)]
        if pol != "frac_pv":
            rows.append((pol, THETA_LIMIT))
    return rows


PROBES = [("cqr", 0.9), ("icp", 0.95), ("mondrian_cqr", 0.8), ("frac_pv", 0.5)]  # [D-092] identity rows


def new_rows_low():
    """[D-092] The rows stage S6X simulates: every theta policy at every level of
    ce1_dispatch.THETA_LOW, then the probe rows.
    """
    return [(pol, float(t)) for pol in D.ALL_POLICIES if pol != "frac_pv" for t in D.THETA_LOW] + list(PROBES)


def counted_matrix(rows, fit, ev, pv_of, pav_target):
    """(K, n_ev) counted PV at the target steps of ev, one row per (policy, param)."""
    out = np.empty((len(rows), len(ev["y"])))
    for k, (pol, par) in enumerate(rows):
        if pol == "handbook":
            out[k] = 0.0
        elif pol == "deterministic":
            out[k] = pv_of(ev["p"])
        elif pol == "oracle":
            out[k] = pav_target
        elif pol != "frac_pv" and par == THETA_LIMIT:
            out[k] = 0.0  # the limit level: handbook rule
        else:
            out[k] = D.counted_pv(pol, par, fit, ev, pv_of)
    return out


def place(grid, target_idx, M):
    """Rows of M (K, n_ev) at the grid positions of their target steps; NaN elsewhere."""
    pos = pd.DatetimeIndex(grid).get_indexer(pd.DatetimeIndex(target_idx))
    assert (pos >= 0).all(), "a target step is outside the grid"
    out = np.full((M.shape[0], len(grid)), np.nan)
    out[:, pos] = M
    return out


def year_grid(start, end):
    return pd.date_range(pd.Timestamp(start), pd.Timestamp(end), freq="5min")


# ---------------------------------------------------------------- selection
def _slice(tot, ix):
    return {k: np.asarray(v)[ix] for k, v in tot.items()}


def select_policy(totH2, totT, rows, pol):
    """Protocol A-R, A-C, references, frontier value and the selection-only sensitivities
    for one policy, from the batch totals (the functions of ce1_load, unchanged).
    """
    ix = [i for i, r in enumerate(rows) if r[0] == pol]
    params = np.array([rows[i][1] for i in ix])
    sH, sT = _slice(totH2, ix), _slice(totT, ix)
    pick = lambda i: {k: float(sT[k][i]) for k in TOT_KEYS}  # noqa: E731
    idx = lambda p: int(np.flatnonzero(params == p)[0])  # noqa: E731
    pAR, met = LM.select_AR(sH, params, pol)
    pAC = LM.select_AC(sH, params)
    rAR, _ = LM.select_AR(sT, params, pol)
    rAC = LM.select_AC(sT, params)
    out = dict(A_R=dict(param=pAR, target_met_on_H2=bool(met), test=pick(idx(pAR))),
               A_C=dict(param=pAC, test=pick(idx(pAC))),
               reference_R=dict(param=rAR, test=pick(idx(rAR))),
               reference_C=dict(param=rAC, test=pick(idx(rAC))),
               fuel_at_target_test=LM.fuel_at_target(sT),
               fuel_at_target_H2=LM.fuel_at_target(sH))
    pS, metS = LM.select_AR(sH, params, pol, target=LM.UE_TARGET_SENS)
    out["A_R_target_0.003pct"] = dict(param=pS, target_met_on_H2=bool(metS), test=pick(idx(pS)))
    out["fuel_at_target_test_0.003pct"] = LM.fuel_at_target(sT, LM.UE_TARGET_SENS)
    for fp in LM.FUEL_PRICE_SENS:
        p_ = LM.select_AC(sH, params, price=fp)
        out[f"A_C_fuel{fp:.2f}"] = dict(param=p_, test=pick(idx(p_)))
    for vl in LM.VOLL_SENS:
        p_ = LM.select_AC(sH, params, voll=vl)
        out[f"A_C_voll{vl:.0f}"] = dict(param=p_, test=pick(idx(p_)))
    return out


# ---------------------------------------------------------------- one fold and horizon
def prepare(cache: pd.DataFrame, h: int, test_year: int, geo: pd.DataFrame, m: dict, rows,
            realized_arrays: pd.Series | None = None):
    """Counted-PV matrices and available PV on the H2 and test grids."""
    ca = cache[cache["split"] == "calib"]
    te = cache[cache["split"] == "test"]
    calib, test = as_dict(ca), as_dict(te)
    H1, H2 = D.split_halves(calib)
    lead = pd.Timedelta(minutes=5 * h)
    out = {}
    cy = test_year - 1
    for name, fit, ev, (a, b) in (("H2", H1, H2, (f"{cy}-07-01", f"{cy}-12-31 23:55")),
                                  ("test", calib, test, (f"{test_year}-01-01", f"{test_year}-12-31 23:55"))):
        tgt = pd.DatetimeIndex(ev["idx"]) + lead
        grid = year_grid(a, b)
        ghi = geo["ghi"].reindex(grid).values
        pav = LM.available_pv_on_grid(grid, ghi, m)
        pv_of = PVAt(tgt, m)
        pav_t = pv_of(ev["y"])
        pos = grid.get_indexer(tgt)
        assert np.allclose(pav_t, pav[pos], atol=1e-9, equal_nan=True), \
            "available PV at the forecast targets differs from the grid (target basis mismatch)"
        C = counted_matrix(rows, fit, ev, pv_of, pav_t)
        out[name] = dict(grid=grid, pav=pav, C=place(grid, tgt, C), day=LM.day_ids(grid),
                         n_rows=len(ev["y"]))
        if realized_arrays is not None and name == "test":
            ra = realized_arrays.reindex(grid).values
            out[name]["pav_arrays"] = np.where(np.isnan(pav), np.nan, np.where(pav == 0, 0.0, ra))
    return out


def load_for(grid, unit96, scales, sc):
    mk = scales[sc["shape"]][sc["scale"]]
    return mk * LM.load_on(grid, unit96, sc["shape"])


def run_fold_h(prep, rows, unit96, scales, h, scenarios, daily_scen=DAILY_SCEN):
    """Simulate every scenario on H2 and test; return totals table, selections, daily arrays."""
    tables, sel, daily = [], {}, {}
    pols = D.ALL_POLICIES
    for sname, sc in scenarios.items():
        res = {}
        for part in ("H2", "test"):
            P = prep[part]
            pav = P["pav"]
            if sc.get("realized") == "arrays":
                if part == "H2":
                    pass
                else:
                    pav = P["pav_arrays"]
            L = load_for(P["grid"], unit96, scales, sc)
            r = LM.simulate(L, pav, P["C"], P["day"], h, ratings=LM.fleet_for_peak(L.max()),
                            min_load=sc["min_load"], fuel=sc["fuel"], battery=sc["battery"])
            res[part] = r
            t = pd.DataFrame({k: r["total"][k] for k in TOT_KEYS})
            t.insert(0, "param", [x[1] for x in rows])
            t.insert(0, "policy", [x[0] for x in rows])
            t.insert(0, "set", part)
            t.insert(0, "scenario", sname)
            t["missing_steps"] = r["missing_steps"]
            tables.append(t)
        totH2, totT = res["H2"]["total"], res["test"]["total"]
        s = dict(scenario=sc, anchors={a: {k: float(totT[k][i]) for k in TOT_KEYS}
                                       for i, a in enumerate(ANCHORS)},
                 missing_steps=dict(H2=int(res["H2"]["missing_steps"]), test=int(res["test"]["missing_steps"])),
                 policies={p: select_policy(totH2, totT, rows, p) for p in pols})
        sel[sname] = s
        if sname in daily_scen:
            daily[sname] = {k: res["test"]["daily"][k].astype(np.float32) for k in DAILY_KEYS}
    return pd.concat(tables, ignore_index=True), sel, daily


def run_fold_h_rows(prep, rows, unit96, scales, h, scenarios, daily_scen=DAILY_SCEN):
    tables, daily = [], {}
    for sname, sc in scenarios.items():
        for part in ("H2", "test"):
            P = prep[part]
            pav = P["pav_arrays"] if (sc.get("realized") == "arrays" and part == "test") else P["pav"]
            L = load_for(P["grid"], unit96, scales, sc)
            r = LM.simulate(L, pav, P["C"], P["day"], h, ratings=LM.fleet_for_peak(L.max()),
                            min_load=sc["min_load"], fuel=sc["fuel"], battery=sc["battery"])
            t = pd.DataFrame({k: r["total"][k] for k in TOT_KEYS})
            t.insert(0, "param", [x[1] for x in rows])
            t.insert(0, "policy", [x[0] for x in rows])
            t.insert(0, "set", part)
            t.insert(0, "scenario", sname)
            t["missing_steps"] = r["missing_steps"]
            tables.append(t)
            if part == "test" and sname in daily_scen:
                daily[sname] = {k: r["daily"][k].astype(np.float32) for k in DAILY_KEYS}
    return pd.concat(tables, ignore_index=True), daily


def probe_identity(tab, s6_totals: pd.DataFrame, s6b_totals: pd.DataFrame):
    ref = pd.concat([s6_totals[~s6_totals["scenario"].isin(("battery", "batteryB"))], s6b_totals])
    key = ["scenario", "set", "policy", "param"]
    pr = tab[[tuple(x) in set(PROBES) for x in tab[["policy", "param"]].itertuples(index=False)]]
    m = pr.merge(ref, on=key, suffixes=("", "_ref"), how="left")
    worst = 0.0
    for k in ("fuel_L", "ue_kWh", "curt_kWh", "starts"):
        a, b = m[k].values.astype(float), m[k + "_ref"].values.astype(float)
        if np.isnan(b).any():
            return np.inf, len(m)
        worst = max(worst, float(np.max(np.abs(a - b) / np.maximum(np.abs(b), 1.0))))
    return worst, len(m)


# ---------------------------------------------------------------- inputs
def load_inputs(s1, s1m, s1l):
    geo = pd.read_parquet(Path(s1) / "yulara_geo_5min.parquet")
    m = PM.load_map(Path(s1m) / "available_pv_map.npz")
    z = np.load(Path(s1l) / "load_profile.npz")
    unit96 = z["unit96"]
    rec = json.load(open(Path(s1l) / "load_scales.json"))
    g23 = year_grid("2023-01-01", "2023-12-31 23:55")
    pav23 = LM.available_pv_on_grid(g23, geo["ghi"].reindex(g23).values, m)
    scales = {}
    for shp in ("A", "B"):
        ulA = LM.load_on(g23, unit96, "A")
        scales[shp] = dict(primary=LM.scale_energy_share(ulA, pav23), peak6MW=LM.scale_peak(ulA),
                           spill_matched=float(rec["scales"]["A"]["spill_matched"]["mean_kW"]))
    return geo, m, unit96, scales, rec


# ---------------------------------------------------------------- self-tests
def selftest(verbose=True) -> dict:
    res = {}

    def rep(name, ok, msg=""):
        res[name] = bool(ok)
        if verbose:
            print(f"{'PASS' if ok else 'FAIL'}  {name}  {msg}", flush=True)

    rng = np.random.default_rng(3)
    m = dict(centers=np.arange(12.5, 1250, 25.0), vals=np.minimum(np.arange(12.5, 1250, 25.0) / 1300, 0.77),
             cap=0.77, rated_total=1825.4, plane_tilt=15.0, plane_az=0.0)
    idx = pd.date_range("2023-03-01 06:00", periods=300, freq="5min")
    g = np.clip(900 * np.sin(np.linspace(0.1, 3.0, 300)) + rng.normal(0, 40, 300), 0, None)
    g[5] = np.nan
    a = PVAt(idx, m)(g)
    b = PM.pv_available_from_ghi(g, idx, m)
    rep("pv_at_equals_pvmap", np.array_equal(np.isnan(a), np.isnan(b)) and np.allclose(a, b, equal_nan=True, atol=0),
        f"(max diff {np.nanmax(np.abs(a - b)):.2e})")
    # selection from batch totals equals ce1_load.run_selection
    n_days = 6
    grid = pd.date_range("2030-01-01", periods=288 * n_days, freq="5min")
    day = LM.day_ids(grid)
    hod = np.asarray(grid.hour + grid.minute / 60)
    pav = np.clip(1400 * np.sin(np.pi * (hod - 6) / 12), 0, None) * np.repeat(rng.uniform(0.3, 1.0, n_days * 24), 12)
    pav[400:410] = np.nan
    L = 2200 + 600 * np.sin(np.pi * (hod - 3) / 12)
    params = np.round(np.linspace(0.0, 1.0, 11), 4)
    C = np.vstack([(1 - p) * 1.25 * np.nan_to_num(pav) for p in params])  # overcounts at small f
    C[:, 100:104] = np.nan
    ref = LM.run_selection(L, pav, C, day, L, pav, C, day, params, "frac_pv", 3)
    rows = [("frac_pv", float(p)) for p in params]
    tot = LM.simulate(L, pav, C, day, 3)["total"]
    s = select_policy(tot, tot, rows, "frac_pv")
    rep("selection_equals_run_selection", s["A_R"]["param"] == ref["A_R"]["param"]
        and s["A_C"]["param"] == ref["A_C"]["param"]
        and s["reference_R"]["param"] == ref["reference_R"]["param"]
        and np.isclose(s["fuel_at_target_test"], ref["fuel_at_target_test"], equal_nan=True)
        and np.isclose(s["A_R"]["test"]["fuel_L"], ref["A_R"]["test"]["fuel_L"])
        and 0.0 < s["A_R"]["param"] < 1.0,
        f"(A-R f = {s['A_R']['param']}, A-C f = {s['A_C']['param']})")
    # placement: rows land on their target steps, NaN elsewhere
    tgt = grid[[10, 50, 51, 700]]
    M = np.arange(8, dtype=float).reshape(2, 4)
    Pm = place(grid, tgt, M)
    rep("place_on_target_steps", np.array_equal(Pm[:, [10, 50, 51, 700]], M)
        and np.isnan(np.delete(Pm, [10, 50, 51, 700], axis=1)).all())
    # counted matrix: anchors and a policy row equal their definitions
    n = 400
    ev = dict(idx=pd.date_range("2023-08-01 08:00", periods=n, freq="5min").values,
              y=rng.uniform(100, 900, n), p=rng.uniform(100, 900, n), g=rng.choice(["clear", "cloudy"], n),
              v=rng.uniform(0, 0.2, n), gcs=rng.uniform(600, 1000, n),
              q={t: rng.uniform(0, 800, n) for t in D.taus_needed()})
    fit = {k: (v.copy() if k != "q" else {t: x.copy() for t, x in v.items()}) for k, v in ev.items()}
    pv = lambda x: np.clip(np.asarray(x, float), 0, None) * 1.5  # noqa: E731
    rws = [("handbook", np.nan), ("deterministic", np.nan), ("oracle", np.nan), ("cqr", 0.9), ("frac_pv", 0.5),
           ("mondrian", THETA_LIMIT), ("frac_pv", 1.0), ("icp", 0.999)]
    CM = counted_matrix(rws, fit, ev, pv, pv(ev["y"]))
    rep("counted_matrix_definitions", (CM[0] == 0).all() and np.allclose(CM[1], pv(ev["p"]))
        and np.allclose(CM[2], pv(ev["y"])) and np.allclose(CM[3], D.counted_pv("cqr", 0.9, fit, ev, pv))
        and np.allclose(CM[4], 0.5 * pv(ev["p"])) and (CM[5] == 0).all() and (CM[6] == 0).all()
        and np.allclose(CM[7], D.counted_pv("icp", 0.999, fit, ev, pv)),
        "(anchors, a CQR level, frac_pv, the theta = 1 limit equal to the handbook rule, a tail level)")
    # the row table holds every policy's full parameter set once
    rt = row_table()
    rep("row_table_complete", len(rt) == 3 + sum(len(D.param_grid(p)) + (p != "frac_pv") for p in D.ALL_POLICIES)
        and all((p, THETA_LIMIT) in rt for p in D.ALL_POLICIES if p != "frac_pv")
        and len(set(rt)) == len(rt) - 0, f"({len(rt)} rows per batch)")
    # frac_pv f = 1 and the handbook anchor are the same decision
    Ls = 2200 + 0 * pav
    C2 = np.vstack([np.zeros(len(grid)), 0.0 * np.nan_to_num(pav)])
    r2 = LM.simulate(Ls, pav, C2, day, 3)["total"]
    rep("frac_pv_1_equals_handbook", np.isclose(r2["fuel_L"][0], r2["fuel_L"][1]))
    # [D-092] rows are simulated independently: a subset batch equals the same rows of the full batch
    Cb = np.vstack([(1 - p) * 1.1 * np.nan_to_num(pav) for p in (0.1, 0.4, 0.7, 1.0)])
    ind = True
    for bat in (False, True):
        full = LM.simulate(L, pav, Cb, day, 3, battery=bat)["total"]
        sub = LM.simulate(L, pav, Cb[[1, 3]], day, 3, battery=bat)["total"]
        ind &= all(np.array_equal(np.asarray(full[k])[[1, 3]], np.asarray(sub[k])) for k in TOT_KEYS)
    rep("rows_simulated_independently", ind, "(subset batch = same rows of the full batch, with and without battery)")
    nr = new_rows_low()
    rep("new_rows_low_complete", len(nr) == 7 * len(D.THETA_LOW) + len(PROBES)
        and min(t for p_, t in nr if p_ != "frac_pv") == 0.5 and all(t < 0.75 for p_, t in nr[:-len(PROBES)]),
        f"({len(nr)} rows)")
    # PVBinder: the proxy sweep maps GHI at t + h exactly as PVAt does
    bnd = PVBinder(m, 3).bind(dict(idx=idx.values))
    rep("pv_binder_uses_target_times", np.allclose(bnd(g), PVAt(idx + pd.Timedelta(minutes=15), m)(g),
                                                   equal_nan=True))
    return res


# ---------------------------------------------------------------- stage
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="S6")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--only", default=None, help="year:h[,year:h]")
    ap.add_argument("--s2", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--raw-dir", default=None)
    ap.add_argument("--scenarios", default=None, help="comma list (default: all)")
    a = ap.parse_args()
    if a.selftest:
        r = selftest()
        sys.exit(0 if all(r.values()) else 1)
    import config as C
    t0 = time.time()
    R = C.RESULTS / "ce1"
    s2 = Path(a.s2) if a.s2 else R / ("S2_smoke" if a.smoke else "S2")
    out = Path(a.out) if a.out else R / (a.stage + ("_smoke" if a.smoke else ""))
    runs = R / "_runs" if not a.out else out / "_runs"
    out.mkdir(parents=True, exist_ok=True); runs.mkdir(parents=True, exist_ok=True)
    checks = {}

    def check(name, cond, msg=""):
        checks[name] = dict(ok=bool(cond), msg=msg)
        log(f"{'PASS' if cond else 'FAIL'}  {name}  {msg}")

    st = selftest(verbose=False)
    check("selftests", all(st.values()), f"({sum(st.values())}/{len(st)}"
          + ("" if all(st.values()) else "; failed " + ", ".join(k for k, v in st.items() if not v)) + ")")
    geo, m, unit96, scales, rec = load_inputs(R / "S1", R / "S1M", R / "S1L")
    for shp in ("A",):
        for k in ("primary", "peak6MW"):
            recd = rec["scales"][shp][k]["mean_kW"]
            check(f"scale_{shp}_{k}_matches_S1L", abs(scales[shp][k] - recd) <= 0.051,
                  f"({scales[shp][k]:.2f} kW recomputed vs {recd} recorded)")
    ext = a.stage in ("S6X", "S6P2")  # [D-092] extended theta set
    if ext:
        D.extend_theta_low()
        s2x = R / ("S2X_smoke" if a.smoke else "S2X")
        load = lambda yr, h: FC.load_cache_ext(s2 / f"yulara_f{yr}_h{h}.parquet",  # noqa: E731
                                                s2x / f"yulara_f{yr}_h{h}.parquet")
        check("theta_set_extended", float(D.theta_set().min()) == 0.5 and len(D.THETA_LOW) == 50,
              f"({len(D.theta_set())} levels from {D.theta_set().min()}) [D-092]")
    else:
        load = lambda yr, h: FC.load_cache(s2 / f"yulara_f{yr}_h{h}.parquet")  # noqa: E731
    rows = new_rows_low() if a.stage == "S6X" else row_table()
    jobs = [(y, h) for y in FOLDS for h in HORIZONS]
    if a.smoke:
        jobs = [(2024, 3)]
    if a.only:
        jobs = [(int(p.split(":")[0]), int(p.split(":")[1])) for p in a.only.split(",")]
    if a.stage.startswith("S6P"):
        return main_proxy(a, jobs, s2, out, runs, m, checks, check, t0, load)
    if a.stage == "S6X":
        return main_x(a, jobs, s2, out, runs, m, checks, check, t0, load, rows, geo, unit96, scales, R)
    scen = dict(SCENARIOS)
    if a.stage == "S6B" and not a.scenarios:
        a.scenarios = "battery,batteryB"  # [D-088] battery rerun only
    if a.scenarios:
        scen = {k: v for k, v in scen.items() if k in a.scenarios.split(",")}
    want_arrays = a.scenarios is None or "arrays" in a.scenarios.split(",")
    log(f"scenarios: {', '.join(scen)}{' + arrays' if want_arrays else ''}; battery reserve credit "
        f"rule {LM.CREDIT_RULE} [D-088]")
    arrays = None
    try:
        if not want_arrays:
            raise RuntimeError("not requested")
        d = PM.load_years(ARRAYS_FOLDS, raw_dir=a.raw_dir)
        arrays = d[list(UNCURTAILED)].sum(axis=1, min_count=3) / sum(PM.RATED[x] for x in UNCURTAILED) \
            * PM.RATED_TOTAL
        log(f"raw power channels found: OPP-014 scenario 'arrays' runs for folds {ARRAYS_FOLDS}")
    except Exception as e:
        log("scenario 'arrays' not requested" if not want_arrays else
            f"raw power channels not available ({type(e).__name__}); scenario 'arrays' skipped")
    summary = []
    for yr, h in jobs:
        stem = out / f"f{yr}_h{h}"
        if Path(f"{stem}_select.json").exists() and Path(f"{stem}_totals.parquet").exists():
            log(f"fold {yr} h={h}: done, skipped")
            summary.append(json.load(open(f"{stem}_select.json"))["summary"])
            continue
        t1 = time.time()
        cache = load(yr, h)
        use_arrays = arrays is not None and yr in ARRAYS_FOLDS
        prep = prepare(cache, h, yr, geo, m, rows, arrays if use_arrays else None)
        sc = dict(scen)
        if use_arrays:
            sc["arrays"] = ARRAYS_SCEN
        tab, sel, daily = run_fold_h(prep, rows, unit96, scales, h, sc)
        tab.to_parquet(f"{stem}_totals.parquet", compression="zstd")
        np.savez_compressed(f"{stem}_daily.npz", rows_policy=np.array([r[0] for r in rows]),
                            rows_param=np.array([r[1] for r in rows]),
                            **{f"{s}__{k}": v for s, dd in daily.items() for k, v in dd.items()})
        s0 = "main" if "main" in sel else next(iter(sel))
        mn = sel[s0]
        hb = tab[tab["policy"] == "handbook"]
        summ = dict(fold=yr, h=h, n_H2_rows=prep["H2"]["n_rows"], n_test_rows=prep["test"]["n_rows"],
                    summary_scenario=s0, credit_rule=LM.CREDIT_RULE,
                    handbook_ue_kWh_max_all_scenarios_sets=float(hb["ue_kWh"].max()),
                    missing_test=mn["missing_steps"]["test"],
                    fallback_test_handbook=int(tab.query(f"scenario=='{s0}' and set=='test' and policy=='handbook'")
                                               ["fallback_steps"].iloc[0]),
                    handbook_fuel_L=mn["anchors"]["handbook"]["fuel_L"],
                    handbook_ue_share=mn["anchors"]["handbook"]["ue_share"],
                    oracle_fuel_L=mn["anchors"]["oracle"]["fuel_L"], oracle_ue_kWh=mn["anchors"]["oracle"]["ue_kWh"],
                    seconds=round(time.time() - t1, 1))
        json.dump(dict(summary=summ, selections=sel, rows=len(rows)), open(f"{stem}_select.json", "w"),
                  indent=1, default=float)
        summary.append(summ)
        log(f"fold {yr} h={h}: {len(sc)} scenarios, handbook fuel {summ['handbook_fuel_L']:.0f} L, oracle "
            f"{summ['oracle_fuel_L']:.0f} L, {summ['seconds']:.0f}s")
    sm = pd.DataFrame(summary)
    sm.to_csv(out / "s6_summary.csv", index=False)
    check("every_job_done", len(sm) == len(jobs), f"({len(sm)}/{len(jobs)})")
    check("handbook_rule_meets_target_main", bool(len(sm)) and (sm["handbook_ue_share"] <= LM.UE_TARGET).all(),
          f"(the operator's rule serves the load in every fold and horizon, scenario "
          f"{sm['summary_scenario'].iloc[0] if 'summary_scenario' in sm else 'main'})")
    if a.stage == "S6B":
        check("handbook_battery_zero_UE_every_fold_horizon_set",
              bool(len(sm)) and (sm["handbook_ue_kWh_max_all_scenarios_sets"] == 0.0).all(),
              "(D-088: the corrected credit is deliverable; battery and batteryB, H2 and test)")
    check("oracle_no_ue_and_least_fuel", bool(len(sm)) and (sm["oracle_ue_kWh"] < 1e-6).all()
          and (sm["oracle_fuel_L"] <= sm["handbook_fuel_L"] + 1e-6).all())
    ok = all(c["ok"] for c in checks.values())
    detail = dict(stage=a.stage, smoke=a.smoke, ok=ok, seconds=round(time.time() - t0, 1), checks=checks,
                  jobs=len(jobs), scenarios=list(scen) + (["arrays"] if arrays is not None else []),
                  credit_rule=LM.CREDIT_RULE,
                  outputs={p.name: md5(p) for p in sorted(out.glob("*")) if p.is_file()},
                  inputs={p.name: md5(p) for p in sorted(s2.glob("yulara_*.parquet"))},
                  code={p.name: md5(p) for p in (Path(__file__), HERE / "ce1_load.py", HERE / "ce1_dispatch.py",
                                                  HERE / "ce1_folds.py", HERE / "ce1_pvmap.py")},
                  versions=dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__),
                  host=platform.node())
    json.dump(detail, open(runs / f"{a.stage}{'_smoke' if a.smoke else ''}.detail.json", "w"), indent=1,
              default=float)
    log(f"{'ALL CHECKS PASSED' if ok else 'SOME CHECKS FAILED'} in {time.time() - t0:.0f}s")
    sys.exit(0 if ok else 1)


def main_x(a, jobs, s2, out, runs, m, checks, check, t0, load, rows, geo, unit96, scales, R):
    scen = dict(SCENARIOS)
    arrays = None
    try:
        d = PM.load_years(ARRAYS_FOLDS, raw_dir=a.raw_dir)
        arrays = d[list(UNCURTAILED)].sum(axis=1, min_count=3) / sum(PM.RATED[x] for x in UNCURTAILED) \
            * PM.RATED_TOTAL
        log(f"raw power channels found: scenario 'arrays' runs for folds {ARRAYS_FOLDS}")
    except Exception as e:
        log(f"raw power channels not available ({type(e).__name__}); scenario 'arrays' skipped")
    log(f"S6X: {len(rows)} rows ({len(rows) - len(PROBES)} new, {len(PROBES)} probes), scenarios "
        f"{', '.join(scen)}{' + arrays' if arrays is not None else ''}, credit rule {LM.CREDIT_RULE}")
    summary = []
    for yr, h in jobs:
        stem = out / f"f{yr}_h{h}"
        if Path(f"{stem}_x.json").exists() and Path(f"{stem}_totals.parquet").exists():
            log(f"fold {yr} h={h}: done, skipped")
            summary.append(json.load(open(f"{stem}_x.json"))["summary"])
            continue
        t1 = time.time()
        cache = load(yr, h)
        use_arrays = arrays is not None and yr in ARRAYS_FOLDS
        prep = prepare(cache, h, yr, geo, m, rows, arrays if use_arrays else None)
        sc = dict(scen)
        if use_arrays:
            sc["arrays"] = ARRAYS_SCEN
        tab, daily = run_fold_h_rows(prep, rows, unit96, scales, h, sc)
        worst, n_probe = probe_identity(tab, pd.read_parquet(R / "S6" / f"f{yr}_h{h}_totals.parquet"),
                                        pd.read_parquet(R / "S6B" / f"f{yr}_h{h}_totals.parquet"))
        probe = [tuple(x) in set(PROBES) for x in tab[["policy", "param"]].itertuples(index=False)]
        new = tab[~np.array(probe)].reset_index(drop=True)
        new.to_parquet(f"{stem}_totals.parquet", compression="zstd")
        kp = np.array([r not in PROBES for r in rows])
        np.savez_compressed(f"{stem}_daily.npz", rows_policy=np.array([r[0] for r in rows])[kp],
                            rows_param=np.array([r[1] for r in rows])[kp],
                            **{f"{s_}__{k}": v[kp] for s_, dd in daily.items() for k, v in dd.items()})
        summ = dict(fold=yr, h=h, rows_new=int(kp.sum()), scenarios=len(sc), probe_rows_compared=n_probe,
                    probe_max_rel_diff=worst, seconds=round(time.time() - t1, 1))
        json.dump(dict(summary=summ), open(f"{stem}_x.json", "w"), indent=1, default=float)
        summary.append(summ)
        log(f"fold {yr} h={h}: {len(sc)} scenarios, probe max rel diff {worst:.1e} over {n_probe} rows, "
            f"{summ['seconds']:.0f}s")
    sm = pd.DataFrame(summary)
    sm.to_csv(out / "s6x_summary.csv", index=False)
    check("every_job_done", len(sm) == len(jobs), f"({len(sm)}/{len(jobs)})")
    check("probe_rows_equal_stored_record", bool(len(sm)) and (sm["probe_max_rel_diff"] <= 1e-9).all(),
          f"(max relative difference {sm['probe_max_rel_diff'].max():.1e}; S6 without battery, S6B with)")
    ok = all(c["ok"] for c in checks.values())
    detail = dict(stage=a.stage, smoke=a.smoke, ok=ok, seconds=round(time.time() - t0, 1), checks=checks,
                  jobs=len(jobs), credit_rule=LM.CREDIT_RULE,
                  scenarios=list(scen) + (["arrays"] if arrays is not None else []),
                  theta_low=[float(D.THETA_LOW.min()), float(D.THETA_LOW.max())],
                  outputs={p.name: md5(p) for p in sorted(out.glob("*")) if p.is_file()},
                  inputs={p.name: md5(p) for p in sorted(s2.glob("yulara_*.parquet"))}
                  | {"S2X/" + p.name: md5(p) for p in sorted((R / "S2X").glob("yulara_*.parquet"))},
                  code={p.name: md5(p) for p in (Path(__file__), HERE / "ce1_load.py", HERE / "ce1_dispatch.py",
                                                  HERE / "ce1_folds.py", HERE / "ce1_pvmap.py")},
                  versions=dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__),
                  host=platform.node())
    json.dump(detail, open(runs / f"{a.stage}{'_smoke' if a.smoke else ''}.detail.json", "w"), indent=1,
              default=float)
    log(f"{'ALL CHECKS PASSED' if ok else 'SOME CHECKS FAILED'} in {time.time() - t0:.0f}s")
    sys.exit(0 if ok else 1)


def main_proxy(a, jobs, s2, out, runs, m, checks, check, t0, load):
    summary = []
    for yr, h in jobs:
        stem = out / f"f{yr}_h{h}_proxy"
        cfgs = proxy_configs(yr, h)
        if Path(f"{stem}_choices.json").exists() and Path(f"{stem}_rows.parquet").exists():
            log(f"proxy fold {yr} h={h}: done, skipped")
            summary.append(json.load(open(f"{stem}_choices.json"))["summary"])
            continue
        t1 = time.time()
        cache = load(yr, h)
        rws, ch, sf = run_proxy(cache, h, m, cfgs)
        rws.drop(columns=[c for c in ("soc_lo", "soc_hi") if c in rws.columns]).to_parquet(
            f"{stem}_rows.parquet", compression="zstd")
        sf.to_parquet(f"{stem}_surface.parquet", compression="zstd")
        ref_ok = bool((rws.loc[rws["protocol"] != "anchor", "regret_signed"].dropna() >= -1e-9).all())
        lo = float(D.theta_set().min())
        edge = sum(int(c["policy"] != "frac_pv" and (abs(c["theta_ref"] - lo) < 1e-9 or abs(c["theta_A"] - lo) < 1e-9))
                   for c in ch)
        summ = dict(fold=yr, h=h, configs=len(cfgs), rows=len(rws), regret_nonnegative=ref_ok,
                    theta_low=lo, choices_at_lower_edge=edge, seconds=round(time.time() - t1, 1))
        json.dump(dict(summary=summ, configs=cfgs, choices=ch), open(f"{stem}_choices.json", "w"), indent=1,
                  default=float)
        summary.append(summ)
        log(f"proxy fold {yr} h={h}: {len(cfgs)} configurations, {summ['seconds']:.0f}s")
    sm = pd.DataFrame(summary)
    sm.to_csv(out / "s6p_summary.csv", index=False)
    check("every_job_done", len(sm) == len(jobs), f"({len(sm)}/{len(jobs)})")
    # an IDENTITY, not a test. The reference is the minimum over every level a
    # protocol can select, so the signed regret is non-negative by construction and this line cannot fail.
    # It guards the bookkeeping (a wrong merge or sign would break it), and says nothing about the protocols.
    check("identity_reference_is_a_minimum", bool(len(sm)) and sm["regret_nonnegative"].all(),
          "(identity by construction: the reference minimises over every selectable level)")
    if "choices_at_lower_edge" in sm:  # [D-092] reported, not a failure
        log(f"choices (reference or Protocol A) at the lower edge theta = {sm['theta_low'].min()}: "
            f"{int(sm['choices_at_lower_edge'].sum())} (reported as censored)")
    ok = all(c["ok"] for c in checks.values())
    detail = dict(stage=a.stage, smoke=a.smoke, ok=ok, seconds=round(time.time() - t0, 1), checks=checks,
                  jobs=len(jobs), outputs={p.name: md5(p) for p in sorted(out.glob("*")) if p.is_file()},
                  inputs={p.name: md5(p) for p in sorted(s2.glob("yulara_*.parquet"))}
                  | ({"S2X/" + p.name: md5(p) for p in sorted((s2.parent / "S2X").glob("yulara_*.parquet"))}
                     if a.stage == "S6P2" else {}),
                  code={p.name: md5(p) for p in (Path(__file__), HERE / "ce1_dispatch.py", HERE / "ce1_folds.py",
                                                  HERE / "ce1_pvmap.py")},
                  theta_set=dict(n=int(len(D.theta_set())), low=float(D.theta_set().min())),
                  versions=dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__),
                  host=platform.node())
    json.dump(detail, open(runs / f"{a.stage}{'_smoke' if a.smoke else ''}.detail.json", "w"), indent=1,
              default=float)
    log(f"{'ALL CHECKS PASSED' if ok else 'SOME CHECKS FAILED'} in {time.time() - t0:.0f}s")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
