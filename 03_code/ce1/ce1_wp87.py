#!/usr/bin/env python3
"""Dispatch simulator comparison. The fold loop of stage S69 (every static rule in the plain,
s56 and n1a constructions, A-D, the persistence and smart-persistence fractions) is run under each variant
of ce1_load.simulate, with every level reselected on calibration data under the simulator that
is scored: v20 (release 2.0.0 simulator, which must reproduce the S69 and S69b records), c02
(minimum feasible commitment), c03 (elapsed-time clocks over missing GHI), c23 (both, the
current simulator), f7a and f7b (seven-unit fleet, the two unrated sets assumed at 1.15 + 1.15 MW
and 0.8 + 1.5 MW). Outputs 04_results/ce1/WP87/<variant>/ and WP87/reading.json.

Usage (from the repository root unless the path starts at ce1/, then from 03_code):
    python3 03_code/ce1/ce1_wp87.py --selftest
    python3 03_code/ce1/ce1_wp87.py --variant c23 --fold 2019
    python3 03_code/ce1/ce1_wp87.py --variant c23 --assemble
    python3 03_code/ce1/ce1_wp87.py --read
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import platform
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "utils"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import ce1_dispatch as D  # noqa: E402
import ce1_load as LM  # noqa: E402
import ce1_s6 as S6  # noqa: E402
import ce1_s69 as S69  # noqa: E402
import ce1_wp56 as W56  # noqa: E402

VARIANTS = {
    "v20": dict(commit="greedy", gap="hold", extra=None, internal=False),
    "c02": dict(commit="minfeas", gap="hold", extra=None, internal=True),
    "c03": dict(commit="greedy", gap="elapsed", extra=None, internal=True),
    "c23": dict(commit="minfeas", gap="elapsed", extra=None, internal=True),
    "f7a": dict(commit="minfeas", gap="elapsed", extra=(1150.0, 1150.0), internal=True),
    "f7b": dict(commit="minfeas", gap="elapsed", extra=(800.0, 1500.0), internal=True),
}
FOLDS = S69.FOLDS
KEYS = ("fold", "scenario", "policy", "selection")
CMP = ("param", "fuel_L", "ue_share", "starts", "events")
SU_CENTRAL = 5.0 / 60.0 * 0.0205 * 1000.0  # 1.708 L per start, ce1_wp63.su_litres("central")

_orig_sim = LM.simulate
_orig_transform = S69.transform
_orig_fleet = LM.fleet_for_peak
_orig_combo = W56.combo_table
CTX = dict(variant=None, n1=None, phat={})


def md5(p: Path) -> str:
    h = hashlib.md5()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def log(m):
    print(time.strftime("%H:%M:%S"), f"[WP87] {m}", flush=True)


# ================================================================ variant plumbing
def _sim(L, P_av, P_cnt, day_id, h_steps, **kw):
    v = CTX["variant"]
    if v is not None and not v["internal"]:  # v20: the release 2.0.0 simulator (greedy, hold), passed explicitly
        kw.setdefault("commit", v["commit"])
        kw.setdefault("gap", v["gap"])
    if v is not None and v["internal"]:
        kw.setdefault("commit", v["commit"])
        kw.setdefault("gap", v["gap"])
        n = len(np.asarray(P_av))
        if CTX["n1"] is not None:
            kw.setdefault("n1", CTX["n1"])
            if CTX["n1"] == "s56":
                ph = CTX["phat"].get(n)
                assert ph is not None and len(ph) == n, ("no point forecast for this grid", n)
                kw.setdefault("p_hat", ph)
    return _orig_sim(L, P_av, P_cnt, day_id, h_steps, **kw)


def _infeasible(constr, L, C, det_row, ratings):
    """Steps where no unit set meets the construction (the whole fleet dominates every set in capacity and in
    capacity minus the largest unit, so feasibility is the whole fleet's).
    """
    r = np.asarray(ratings, float)
    tot, n1 = r.sum(), r.sum() - r.max()
    pcz = np.nan_to_num(C)
    k1 = np.maximum(L[None, :] - pcz, r.min())
    need = (L[None, :] - pcz) if constr == "n1a" else np.broadcast_to(L - np.nan_to_num(det_row), C.shape)
    return (k1 > tot + 1e-9) | (need > n1 + 1e-9)


def _transform(constr, L, C, det_row, ratings):
    v = CTX["variant"]
    if v is None or not v["internal"]:
        CTX["n1"] = None
        return _orig_transform(constr, L, C, det_row, ratings)
    if constr == "plain":
        CTX["n1"] = None
        return C, np.zeros(C.shape, bool), {}
    CTX["n1"] = constr
    if constr == "s56":
        n = len(L)
        if n in CTX["phat"]:
            assert np.array_equal(CTX["phat"][n], det_row, equal_nan=True), "two grids of one length"
        CTX["phat"][n] = np.asarray(det_row, float).copy()
    return C, _infeasible(constr, L, C, det_row, ratings), {}  # n1 and p_hat injected by _sim (CTX)


def _combo_noassert(ratings):
    fl = LM.Fleet(ratings)
    cls = np.asarray(fl.classes, float)
    largest = np.array([cls[np.flatnonzero(c)].max() for c in fl.counts])
    return fl.cap.astype(float), (fl.cap - largest).astype(float), float(fl.smallest)


def activate(name):
    v = VARIANTS[name]
    CTX.update(variant=v, n1=None, phat={})
    LM.simulate = _sim
    S69.transform = _transform
    if v["extra"]:
        ex = tuple(v["extra"])
        LM.fleet_for_peak = lambda peak_kw: tuple(LM.UNITS_BASE) + ex
        W56.combo_table = _combo_noassert  # feasibility checks only; k_min is not used under minfeas
    else:
        LM.fleet_for_peak = _orig_fleet
        W56.combo_table = _orig_combo
    S69.PFR_IN = {"plain", "n1a"}  # the persistence fraction under n1a carries the M2 lead (S69b)


# ================================================================ self-tests
def selftest(verbose=True):
    """Self-tests of the simulator variants. A call that passes no simulator argument runs the release 2.0.0 simulator."""
    with LM.registered_v200():
        return _selftest(verbose)


def _selftest(verbose=True):
    res = {}

    def ok(name, cond, msg=""):
        res[name] = bool(cond)
        if verbose:
            print(f"  [{'PASS' if cond else 'FAIL'}] {name} {msg}")

    import ce1_wp63 as W63
    ok("start_up_litres_equal_wp63_central", abs(SU_CENTRAL - W63.su_litres("central")) < 1e-12)
    # 1. worked example: forced 2.5 MW unit, requirement 1.9 MW
    vt = LM.variant_tables(LM.UNITS_BASE)
    m = int(vt.minfeas(np.array([1900.0]), None, np.array([1]), np.array([1]))[0])
    ok("example_minfeas_is_2500", m == 1 and vt.cap[m] == 2500.0, f"(mask {m}, {vt.cap[m]} kW)")
    fl = LM.Fleet()
    g = fl.choose(np.array([1900.0]), np.array([[True, False, False, False, False]]), np.array([[True, False, False, False, False]]))
    ok("example_registered_is_4500", float((g @ fl.r)[0]) == 4500.0)

    # 2. brute force of the minimum feasible choice, five and seven units, with and without N-1
    rng = np.random.default_rng(87)
    for lab, rat in (("5u", LM.UNITS_BASE), ("7a", LM.UNITS_BASE + (1150.0, 1150.0)), ("7b", LM.UNITS_BASE + (800.0, 1500.0))):
        vt = LM.variant_tables(rat)
        r = np.asarray(rat, float)
        U, M = len(r), 1 << len(rat)
        mb = ((np.arange(M)[:, None] >> np.arange(U)) & 1).astype(bool)
        cap = mb @ r
        cn1 = np.array([cap[x] - r[mb[x]].max() if x else -np.inf for x in range(M)])
        pop = mb.sum(axis=1)
        nt = 4000 if U == 5 else 6000
        bad = 0
        for _ in range(nt):
            o = int(rng.integers(M))
            f = o & int(rng.integers(M))
            kreq = float(rng.uniform(800, r.sum() + 600))
            need = None if rng.random() < 0.4 else float(rng.uniform(-500, r.sum() - r.max() + 400))
            best, key = M - 1, None
            for x in range(1, M):
                if (x & f) != f or cap[x] < kreq - 1e-9 or (need is not None and cn1[x] < need - 1e-9):
                    continue
                k = (cap[x], pop[x], bin(x & ~o).count("1"), x)
                if key is None or k < key:
                    best, key = x, k
            got = int(vt.minfeas(np.array([kreq]), None if need is None else np.array([need]), np.array([o]), np.array([f]))[0])
            bad += got != best
        ok(f"minfeas_equals_brute_force_{lab}", bad == 0, f"({nt} draws, {bad} differ)")
        # with no forced unit and no N-1, the least capacity equals the registered commitment's capacity
        if U == 5:
            same = True
            for kreq in np.linspace(500, 7000, 131):
                for o in range(M):
                    a = vt.cap[int(vt.minfeas(np.array([kreq]), None, np.array([o]), np.array([0]))[0])]
                    b = fl.cap_of[int(fl.choose_mask(np.array([kreq]), np.array([o]), np.array([0]))[0])]
                    same &= bool(a == b)
            ok("minfeas_equals_registered_without_forced_units", same)

    # 3. elapsed clocks: 12 missing steps between two scored steps
    pav = np.array([0.0] + [np.nan] * 12 + [0.0])
    a = _orig_sim(np.full(14, 1800.0), pav, np.zeros((1, 14)), np.zeros(14, dtype=int), 3, record=True)
    b = _orig_sim(np.full(14, 1800.0), pav, np.zeros((1, 14)), np.zeros(14, dtype=int), 3, record=True, gap="elapsed")
    ok("registered_holds_clocks", a["rec_up"][0, -1].tolist() == [0, 2, 2, 0, 0])
    ok("elapsed_advances_clocks", b["rec_up"][0, -1].tolist() == [0, 12, 12, 0, 0], str(b["rec_up"][0, -1].tolist()))
    ok("gap_starts_not_counted", float(a["total"]["starts"][0]) == float(b["total"]["starts"][0]) == 2.0)
    ok("gap_not_scored", int(a["total"]["scored_steps"][0]) == int(b["total"]["scored_steps"][0]) == 2)

    # 4. the self-tests of ce1_load and ce1_s69 (release 2.0.0 and current simulator)
    st = LM.selftest(verbose=False)
    ok("ce1_load_selftests_pass", all(st.values()), f"({sum(st.values())}/{len(st)})")
    st = S69.selftest(verbose=False)
    ok("ce1_s69_selftests_pass", all(st.values()), f"({sum(st.values())}/{len(st)})")

    # 5. internal N-1 with the registered commitment equals the S69 / S56 transforms (toy year with gaps)
    grid, Lt, pav, cs = LM._toy(3, n_days=20)
    dd = LM.day_ids(grid)
    C = LM._counted(pav, cs, np.random.default_rng(11), K=6)
    ph = C[-1].copy()
    Ce, _ = S69.effective_n1a(Lt, C, LM.UNITS_BASE)
    r1 = _orig_sim(Lt, pav, Ce, dd, 3, record=True)
    r2 = _orig_sim(Lt, pav, C, dd, 3, record=True, n1="n1a")
    same = all(np.array_equal(r1["total"][k], r2["total"][k]) for k in ("fuel_L", "ue_kWh", "starts")) and np.array_equal(r1["rec_on"], r2["rec_on"])
    ok("internal_n1a_registered_equals_s69_transform", same)
    Ce5, _, _ = W56.effective_counted(Lt, C, ph, LM.UNITS_BASE, True)
    r1 = _orig_sim(Lt, pav, Ce5, dd, 3, record=True)
    r2 = _orig_sim(Lt, pav, C, dd, 3, record=True, n1="s56", p_hat=ph)
    same = all(np.array_equal(r1["total"][k], r2["total"][k]) for k in ("fuel_L", "ue_kWh", "starts")) and np.array_equal(r1["rec_on"], r2["rec_on"])
    ok("internal_s56_registered_equals_s56_transform", same)

    # 6. under the minimum feasible commitment N-1 holds on every scored feasible step, and the online set
    #    contains every unit inside its minimum up time
    fleet = np.asarray(LM.UNITS_BASE, float)
    for constr in ("n1a", "s56"):
        for gp in ("hold", "elapsed"):
            kw = dict(n1=constr, p_hat=ph) if constr == "s56" else dict(n1=constr)
            r = _orig_sim(Lt, pav, C, dd, 3, record=True, commit="minfeas", gap=gp, **kw)
            lg = np.where(r["rec_on"], fleet[None, None, :], 0).max(axis=2)
            need = (Lt[None, :] - np.nan_to_num(C)) if constr == "n1a" else np.broadcast_to(Lt - np.nan_to_num(ph), C.shape)
            sc = np.isfinite(pav)
            feas = ~_infeasible(constr, Lt, C, ph, fleet)
            viol = (r["rec"]["KU"] - lg < need - 1e-6) & sc[None, :] & feas
            ok(f"minfeas_{constr}_{gp}_n1_holds", int(viol.sum()) == 0, f"({int(viol.sum())} violations)")
    r = _orig_sim(Lt, pav, C, dd, 3, record=True, commit="minfeas")
    on, up = r["rec_on"], r["rec_up"]
    sc = np.flatnonzero(np.isfinite(pav))
    kept = True
    for a_, b_ in zip(sc[:-1], sc[1:]):
        forced = on[:, a_, :] & (up[:, a_, :] < LM.MIN_UP)
        kept &= bool((on[:, b_, :] | ~forced).all())
    ok("minfeas_keeps_units_inside_min_up_time", kept)
    # 7. the plain run under minfeas never commits more capacity than the registered run on the same state
    #    (checked step by step through the lookup tables over the registered run's recorded states)
    rr = _orig_sim(Lt, pav, C, dd, 3, record=True)
    vt = LM.variant_tables(LM.UNITS_BASE)
    bits = (1 << np.arange(5)).astype(np.int64)
    worse = 0
    for s in sc[1:200 * 3]:
        prev = s - 1
        on_m = rr["rec_on"][:, prev, :] @ bits
        f_m = (rr["rec_on"][:, prev, :] & (rr["rec_up"][:, prev, :] < LM.MIN_UP)) @ bits
        pc = np.nan_to_num(C[:, s])
        kreq = np.maximum(Lt[s] - pc, 1000.0)
        a_ = vt.cap[vt.minfeas(kreq, None, on_m, f_m)]
        b_ = fl.cap_of[fl.choose_mask(kreq, on_m, f_m)]
        worse += int((a_ > b_ + 1e-9).sum())
    ok("minfeas_never_above_registered_on_same_state", worse == 0, f"({worse} cases)")
    return res


V200 = "ce1_v200"


class Overlay:
    """R / name: the v2.0.0 archive for an archived folder, else 04_results/ce1 (read-only use in run_fold)."""

    def __init__(self, ce1: Path, v200: Path):
        self.ce1, self.v200 = ce1, v200

    def __truediv__(self, name):
        a = self.v200 / name
        return a if a.exists() else self.ce1 / name


def results_root(name, root: Path):
    R = root / "04_results" / "ce1"
    A = root / "04_results" / V200
    if name == "v20" and A.exists():
        return Overlay(R, A), A
    return R, R


def run(name, fold, root: Path):
    R = root / "04_results" / "ce1"
    Rin, _ = results_root(name, root)
    work = root / "_wp87_work" / name
    work.mkdir(parents=True, exist_ok=True)
    activate(name)
    t0 = time.time()
    checks, diffs = {}, {}

    def check(cname, cond, msg=""):
        if cname.startswith("GATE") and name != "v20":
            diffs[cname] = dict(same=bool(cond), msg=str(msg))  # variants differ from the registered record by design
            return
        checks[cname] = dict(ok=bool(cond), msg=str(msg))
        if not cond:
            log(f"FAIL  {cname}  {msg}")
            (work / f"f{fold}_fail.json").write_text(json.dumps(checks, indent=1))
            sys.exit(f"CHECK FAILED: {cname} {msg}")

    geo, m, unit96, scales, _ = S6.load_inputs(R / "S1", R / "S1M", R / "S1L")
    D.extend_theta_low()
    rows6 = S6.row_table()
    store = {"test_tabs": {}}
    for y in FOLDS:
        if y < fold:
            pk = pickle.load(open(work / f"f{y}.pkl", "rb"))
            store["test_tabs"].update(pk["test_tabs"])
    mode = LM.registered_v200() if name == "v20" else LM.sim_mode(VARIANTS[name]["commit"], VARIANTS[name]["gap"], f"wp87-{name}")
    with mode:  # every simulate call of the fold, patched or not
        c, e, an = S69.run_fold(fold, Rin, work, geo, m, unit96, scales, rows6, check, store)
    own = {k: v for k, v in store["test_tabs"].items() if k[0] == fold}
    rec = dict(fold=fold, variant=name, cases=c, events=e, anchors=an, test_tabs=own, checks=checks, gate_diffs=diffs,
               totals=store["totals"], code=dict(wp87=md5(Path(__file__)), load=md5(HERE / "ce1_load.py"), s69=md5(HERE / "ce1_s69.py")),
               seconds=time.time() - t0)
    pickle.dump(rec, open(work / f"f{fold}.pkl", "wb"))
    log(f"{name} fold {fold}: {len(checks)} checks pass, {sum(not d['same'] for d in diffs.values())} of {len(diffs)} gate comparisons differ "
        f"({time.time() - t0:.0f} s)")


def assemble(name, root: Path):
    R = root / "04_results" / "ce1"
    work = root / "_wp87_work" / name
    recs = [pickle.load(open(work / f"f{y}.pkl", "rb")) for y in FOLDS]
    codes = {json.dumps(r["code"], sort_keys=True) for r in recs}
    assert len(codes) == 1, "a fold was run with different code"
    out = R / "WP87" / name
    out.mkdir(parents=True, exist_ok=True)
    cases = pd.DataFrame([x for r in recs for x in r["cases"]])
    cases.insert(0, "variant", name)
    cases.to_csv(out / "cases.csv", index=False)
    pd.DataFrame([x for r in recs for x in r["events"]]).to_csv(out / "events.csv", index=False)
    pd.DataFrame([x for r in recs for x in r["anchors"]]).to_csv(out / "anchors.csv", index=False)
    pd.concat([t for r in recs for t in r["totals"]], ignore_index=True).to_parquet(out / "totals.parquet")
    checks = {k: v for r in recs for k, v in r["checks"].items()}
    gate = {}
    if name == "v20":
        _, Rref = results_root(name, root)  # the release 2.0.0 records
        s69 = pd.read_csv(Rref / "S69" / "cases.csv", float_precision="round_trip")
        s69b = pd.read_csv(Rref / "S69b" / "cases.csv", float_precision="round_trip")
        ref = pd.concat([s69, s69b[s69b.policy == S69.PFR]], ignore_index=True)
        mm = ref.merge(cases, on=list(KEYS), suffixes=("_ref", ""), how="left")
        worst = 0.0
        for k in CMP:
            x, y = mm[k].to_numpy(float), mm[k + "_ref"].to_numpy(float)
            both = np.isfinite(x) | np.isfinite(y)
            d = np.abs(x[both] - y[both]) / np.maximum(np.abs(y[both]), 1e-12)
            worst = max(worst, float(np.nanmax(d)) if len(d) else 0.0)
            if np.isnan(d).any():
                worst = float("inf")
        gate = dict(ref_cases=len(ref), matched=int(mm.fuel_L.notna().sum()), worst_rel=worst,
                    ok=bool(len(ref) == int(mm.fuel_L.notna().sum()) and worst < 1e-9))
        log(f"GATE v20 reproduces S69 and S69b: {gate}")
    detail = dict(stage=f"WP87-{name}", variant=VARIANTS[name], ok=all(v["ok"] for v in checks.values()) and gate.get("ok", True),
                  seconds=round(sum(r["seconds"] for r in recs), 1), n_checks=len(checks), gate=gate,
                  gate_diffs_differ=sum(not d["same"] for r in recs for d in r["gate_diffs"].values()),
                  outputs={p.name: md5(p) for p in sorted(out.glob("*")) if p.is_file()}, code=recs[0]["code"],
                  env=dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__))
    (out / "checks.json").write_text(json.dumps(dict(detail, checks=checks), indent=1))
    log(f"{name}: assembled, ok={detail['ok']}")
    return detail["ok"]


# ================================================================ reading (D-159 section 5)
def breakeven(saving_L, d_starts, hb_L):
    import ce1_wp63 as W63
    return W63.breakeven(saving_L, d_starts, hb_L)


def summary(c: pd.DataFrame, an: pd.DataFrame) -> dict:
    QP, CP = S69.QP, S69.CP
    out = {}
    c = c.merge(an[["fold", "construction", "shape", "handbook_starts"]], on=["fold", "construction", "shape"], how="left")
    c["saving_L"] = c.fuel_handbook - c.fuel_L
    c["d_starts"] = c.starts - c.handbook_starts
    c["net17"] = 100 * (c.saving_L - SU_CENTRAL * c.d_starts) / c.fuel_handbook
    for constr in ("plain", "s56", "n1a"):
        x = c[c.construction == constr]
        for rule in ("A-R", "H2", "XF", "XH", "PF", "PH", "A-D"):
            for sname, pols in (("quantile", QP), ("conformal", CP)):
                s = x[(x.rule == rule) & x.policy.isin(pols)]
                if len(s):
                    out[f"{constr}|{rule}|{sname}|met"] = int(s.meets.sum())
                    out[f"{constr}|{rule}|{sname}|n"] = len(s)
                    out[f"{constr}|{rule}|{sname}|saving_median"] = float(s.saving_pct.median())
                    out[f"{constr}|{rule}|{sname}|net17_median"] = float(s.net17.median())
                    out[f"{constr}|{rule}|{sname}|breakeven_L"] = float(breakeven(s.saving_L, s.d_starts, s.fuel_handbook))
                    out[f"{constr}|{rule}|{sname}|starts_median"] = float(s.starts.median())
            p = x[(x.rule == rule) & (x.policy == S69.PFR)]
            if len(p):
                out[f"{constr}|{rule}|pfr|met"] = int(p.meets.sum())
                out[f"{constr}|{rule}|pfr|saving_median"] = float(p.saving_pct.median())
        a = an[an.construction == constr]
        out[f"{constr}|oracle_ceiling_median"] = float(a.oracle_ceiling_pct.median())
        out[f"{constr}|handbook_starts_median"] = float(a.handbook_starts.median())
    return out


def pfr_lead(c, constr="n1a", rule="XH"):
    """Median over (fold, shape) of the best quantile policy's saving minus the persistence fraction's (S69b PFR-1:
    best = most cases met under XH, ties by the larger median saving).
    """
    x = c[(c.construction == constr) & (c.rule == rule)]
    q = x[x.policy.isin(S69.QP)]
    agg = q.groupby("policy").agg(met=("meets", "sum"), med=("saving_pct", "median")).reset_index()
    best = agg.sort_values(["met", "med"], ascending=False).policy.iloc[0]
    b = x[x.policy == best].set_index(["fold", "shape"]).saving_pct
    p = x[x.policy == S69.PFR].set_index(["fold", "shape"]).saving_pct
    return best, float((b - p).median())


def read(root: Path):
    R = root / "04_results" / "ce1"
    W = R / "WP87"
    res = {}
    base = None
    for name in VARIANTS:
        f = W / name / "cases.csv"
        if not f.exists():
            continue
        c = pd.read_csv(f, float_precision="round_trip")
        an = pd.read_csv(W / name / "anchors.csv")
        s = summary(c, an)
        try:
            s["n1a|XH|pfr_lead"] = pfr_lead(c)[1]
            s["n1a|XH|pfr_best"] = pfr_lead(c)[0]
        except Exception as ex:  # noqa: BLE001
            s["n1a|XH|pfr_lead_error"] = str(ex)
        res[name] = s
        if name == "v20":
            base = s
    if base is None or "c23" not in res:
        print(json.dumps(res, indent=1, default=str))
        return res
    v, b = res["c23"], base
    d = lambda k: v[k] - b[k]  # noqa: E731
    M = {}
    M["M1"] = {k: d(k) for k in ("plain|XH|quantile|saving_median", "plain|A-R|quantile|saving_median", "plain|A-D|quantile|saving_median",
                                 "n1a|XH|quantile|saving_median", "s56|XH|quantile|saving_median")}
    M["M1_fires"] = any(abs(x) >= 0.10 for x in M["M1"].values())
    M["M2"] = dict(base=b["n1a|XH|pfr_lead"], c23=v["n1a|XH|pfr_lead"], delta=d("n1a|XH|pfr_lead"))
    M["M2_fires"] = (np.sign(v["n1a|XH|pfr_lead"]) != np.sign(b["n1a|XH|pfr_lead"])) or abs(d("n1a|XH|pfr_lead")) >= 0.10
    m3 = {}
    for rule in ("A-R", "H2", "XF", "XH", "PF", "PH", "A-D"):
        for sname in ("quantile", "conformal"):
            k = f"plain|{rule}|{sname}|met"
            m3[k] = dict(base=b[k], c23=v[k], delta=v[k] - b[k])
    all_met = {k: v[k] == v[k.replace("|met", "|n")] for k in ("plain|XH|quantile|met", "plain|PH|quantile|met", "plain|A-D|quantile|met",
                                                                 "plain|A-D|conformal|met")}
    M["M3"] = dict(counts=m3, every_case_met=all_met)
    M["M3_fires"] = any(abs(x["delta"]) >= 3 for x in m3.values()) or not all(all_met.values())
    M["M4"] = dict(gross=(b["plain|XH|quantile|saving_median"], v["plain|XH|quantile|saving_median"]),
                   net17=(b["plain|XH|quantile|net17_median"], v["plain|XH|quantile|net17_median"]))
    M["M4_fires"] = any(np.sign(x[0]) != np.sign(x[1]) for x in M["M4"].values())
    M["M5"] = dict(ceiling=(b["plain|oracle_ceiling_median"], v["plain|oracle_ceiling_median"]))
    M["M5_fires"] = abs(v["plain|oracle_ceiling_median"] - b["plain|oracle_ceiling_median"]) >= 0.10
    M["material"] = any(M[k] for k in ("M1_fires", "M2_fires", "M3_fires", "M4_fires", "M5_fires"))
    res["reading_D159"] = M
    W.mkdir(parents=True, exist_ok=True)
    (W / "reading.json").write_text(json.dumps(res, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
    print(json.dumps(M, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--variant", choices=list(VARIANTS))
    ap.add_argument("--fold", type=int)
    ap.add_argument("--assemble", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--read", action="store_true")
    a = ap.parse_args()
    root = Path(a.root)
    if a.selftest:
        st = selftest()
        sys.exit(0 if all(st.values()) else 1)
    if a.read:
        read(root)
        return
    if a.assemble:
        sys.exit(0 if assemble(a.variant, root) else 1)
    run(a.variant, a.fold, root)


if __name__ == "__main__":
    main()
