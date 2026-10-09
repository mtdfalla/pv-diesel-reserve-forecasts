"""Stage S64: dispatch items registered in D-122 and read by D-123 (spill-matched load scale,
commitment flexibility and related sensitivities), 15 min, six folds, both load shapes.

Usage (from the repository root unless the path starts at ce1/, then from 03_code):
    python3 03_code/ce1/ce1_s64.py --root . --fold 2024
    python3 03_code/ce1/ce1_s64.py --root . --analyse
    python3 03_code/ce1/ce1_s64.py --selftest
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
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
import ce1_load as LM  # noqa: E402

FOLDS = [2019, 2020, 2021, 2022, 2023, 2024]
H = 3
TGT = LM.UE_TARGET
HALF = TGT / 2.0
EPS = 1e-15
CP = ["icp", "mondrian", "cqr", "mondrian_cqr"]
QP = ["raw_q", "cqr", "mondrian_cqr"]
POLICIES = list(D.ALL_POLICIES)
CONSTR_POL = ["icp", "mondrian", "cqr", "mondrian_cqr", "raw_q"]
MIN_DOWN = 12
DEADBAND = 3
HOLD_N = (0, 1, 2, 3, 5, 7, 14, 30, 60, 90, 180)
EVENT_MAX_H2 = 5
GAMMA, KAPPA = 0.005, 0.5
NOLOAD_L_PER_H_PER_KW = 0.0205
SU = {"low": (2.5, 1000.0), "central": (5.0, 1000.0), "high": (10.0, 2500.0)}
PRIMARY = ("main", "shapeB")
SPILL = {"spillA": dict(shape="A", scale="spill_matched", min_load=LM.MIN_LOAD, battery=False, fuel="cat_c32"),
         "spillB": dict(shape="B", scale="spill_matched", min_load=LM.MIN_LOAD, battery=False, fuel="cat_c32")}
CONSTRAINTS = {"min_down": dict(min_down=MIN_DOWN, deadband=0), "deadband": dict(min_down=0, deadband=DEADBAND)}


def md5(p: Path) -> str:
    h = hashlib.md5()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def log(m):
    print(time.strftime("%H:%M:%S"), f"[S64] {m}", flush=True)


def su_litres(case):
    mins, kw = SU[case]
    return mins / 60.0 * NOLOAD_L_PER_H_PER_KW * kw


# ================================================================ fleet with availability
_FLEETS = {}


class FleetC(LM.Fleet):
    """ce1_load.Fleet plus a table over availability masks (minimum down time). first[a, ci]:
    the first combination at or after ci (sorted as Fleet: capacity, units, counts) whose per-class
    counts fit the available units, else the largest-capacity feasible one; table_av[c, on,
    forced, avail]: the new online mask, units chosen within a class as Fleet does (forced, then
    online, then lowest index) among available units.
    """

    def __init__(self, ratings):
        super().__init__(ratings)
        U, nm, nC = self.U, 1 << self.U, len(self.cap)
        assert U <= 5, "the availability table is built for at most five units"
        mbits = ((np.arange(nm)[:, None] >> np.arange(U)) & 1).astype(bool)
        avc = np.stack([mbits[:, mem].sum(axis=1) for mem in self.members], axis=1)
        feas = (self.counts[None, :, :] <= avc[:, None, :]).all(axis=-1)
        first = np.full((nm, nC), -1, np.int64)
        for a in range(nm):
            f = np.flatnonzero(feas[a])
            if not len(f):
                continue
            big = f[np.argmax(self.cap[f])]
            for ci in range(nC):
                g = f[f >= ci]
                first[a, ci] = g[0] if len(g) else big
        self.first = first
        ci, om, fm, am = np.meshgrid(np.arange(nC), np.arange(nm), np.arange(nm), np.arange(nm), indexing="ij")
        ci, om, fm, am = ci.ravel(), om.ravel(), fm.ravel(), am.ravel()
        fm = fm & om
        am = am | om
        new = self._choose_counts_av(self.counts[ci], mbits[om], mbits[fm], mbits[am])
        self.table_av = (new @ self.bits).reshape(nC, nm, nm, nm).astype(np.int64)

    def _choose_counts_av(self, cnt, on, forced, avail):
        new = forced.copy()
        onf = on & ~forced
        off = ~on & avail
        for j, mem in enumerate(self.members):
            o, x = onf[:, mem], off[:, mem]
            need = cnt[:, j] - forced[:, mem].sum(axis=1)
            take = o & ((np.cumsum(o, axis=1) - o) < need[:, None])
            need = need - o.sum(axis=1)
            take |= x & ((np.cumsum(x, axis=1) - x) < need[:, None])
            new[:, mem] |= take
        return new


def fleet_c(ratings):
    key = tuple(float(x) for x in ratings)
    if key not in _FLEETS:
        _FLEETS[key] = FleetC(key)
    return _FLEETS[key]


# ================================================================ the dispatch loop copy
def _minfeas_av(fl, kreq, on_m, forced_m, avail_m):
    """v2.1 minimum feasible commitment among available units: the least total rating at least kreq
    among sets x with forced <= x <= avail (ties: fewer units, fewer starts, lowest index); none: every available
    unit. avail_m = all units reproduces ce1_load.VariantTables.minfeas without N-1.
    """
    U = fl.U
    M = 1 << U
    x = np.arange(M, dtype=np.int64)
    capx = fl.cap_of
    popx = fl.pop
    valid = ((x[None, :] & forced_m[:, None]) == forced_m[:, None]) & ((x[None, :] & ~avail_m[:, None]) == 0) \
        & (capx[None, :] >= kreq[:, None] - 1e-9) & (x[None, :] > 0)
    starts = fl.pop[x[None, :] & ~on_m[:, None]]
    caprank = np.searchsorted(np.unique(capx), capx)
    key = ((caprank[None, :] * (U + 1) + popx[None, :]) * (U + 1) + starts) * M + x[None, :]
    key = np.where(valid, key, np.iinfo(np.int64).max)
    j = np.argmin(key, axis=1)
    return np.where(valid.any(axis=1), x[j], avail_m).astype(np.int64)


def simulate_c(L, P_av, P_cnt, day_id, h_steps, ratings=LM.UNITS_BASE, min_load=LM.MIN_LOAD, fuel="cat_c32",
               min_down=0, deadband=0, commit=None, gap=None):
    """ce1_load.simulate without the battery branch, plus the two commitment constraints and a
    UE event count. With min_down = deadband = 0 it must equal ce1_load.simulate (checked) under either
    simulator: commit and gap take ce1_load's defaults (v2.1: minimum feasible commitment, elapsed clocks).
    v2.1 port: under commit="minfeas" the choice is the least-capacity set containing the forced units
    among the available ones (none: every available unit). Under gap="elapsed" a step without measured GHI
    commits as a scored step with counted PV 0 (minimum down time and deadband included), the clocks and the
    stop times advance, and nothing is scored or counted there (starts, events).
    """
    commit = LM.DEFAULT_COMMIT if commit is None else commit
    gap = LM.DEFAULT_GAP if gap is None else gap
    assert commit in LM.COMMITS and gap in LM.GAPS, (commit, gap)
    fl = fleet_c(ratings)
    a_f, b_f = LM.FUEL[fuel]
    P_cnt = np.atleast_2d(np.asarray(P_cnt, float))
    K, n = P_cnt.shape
    L = np.asarray(L, float)
    L2 = L if L.ndim == 2 else None
    mload = np.broadcast_to(np.asarray(min_load, float), (K,)).copy()
    U = len(fl.r)
    nd = int(day_id.max()) + 1
    on = np.zeros((K, U), bool)
    on_m = np.zeros(K, np.int64)
    bitsU = ((np.arange(1 << U)[:, None] >> np.arange(U)) & 1).astype(bool)
    up = np.zeros((K, U), np.int16)
    last_stop = np.full((K, U), -(10 ** 9), np.int64)
    db = np.zeros(K, np.int64)
    keys = LM.DAY_KEYS + ("events",)
    nk = len(keys)
    day = np.zeros((nk, K, nd))
    acc = np.zeros((nk, K))
    (a_fuel, a_ue, a_curt, a_dump, a_th, a_pvl, a_chg, a_dis, a_pav, a_load, a_st, a_ues, a_onl,
     a_fb, a_sc, a_ev) = acc
    run_steps = np.zeros((K, U))
    missing = 0
    zK = np.zeros(K)
    prev_pos = np.zeros(K, bool)
    nC = len(fl.cap)
    cur_day = int(day_id[0]) if n else 0
    full = (1 << U) - 1

    def _commit(kreq, s, on_m, on, up, db, last_stop):
        forced_m = ((up < LM.MIN_UP) & on) @ fl.bits
        if commit == "minfeas":
            avail_m = (on_m | (((s - last_stop) >= min_down) @ fl.bits)) if min_down else np.full(K, full, np.int64)
            new_m = _minfeas_av(fl, kreq, on_m, forced_m, avail_m)
        else:
            ci = np.minimum(np.searchsorted(fl.cap, kreq - 1e-9, side="left"), nC - 1)
            if min_down:
                avail_m = on_m | (((s - last_stop) >= min_down) @ fl.bits)
                cj = fl.first[avail_m, ci]
                new_m = np.where(cj >= 0, fl.table_av[np.maximum(cj, 0), on_m, forced_m, avail_m], on_m)
            else:
                new_m = fl.table[ci, on_m, forced_m]
        if deadband:
            db = np.where(fl.cap_of[on_m] - kreq >= fl.smallest, db + 1, 0)
            block = ((on_m & ~new_m) != 0) & (db < deadband)
            new_m = np.where(block, new_m | on_m, new_m)
        if min_down:
            last_stop = np.where(bitsU[on_m & ~new_m], s, last_stop)
        return new_m, db, last_stop

    for s in range(n):
        d = int(day_id[s])
        if d != cur_day:
            day[:, :, cur_day] = acc
            acc[:] = 0.0
            cur_day = d
            prev_pos[:] = False
        pav = P_av[s]
        Ls = L2[:, s] if L2 is not None else L[s]
        if pav != pav:
            missing += 1
            prev_pos[:] = False
            if gap == "elapsed":  # v2.1: handbook commitment, clocks advance
                kreq = np.maximum(np.broadcast_to(np.asarray(Ls, float), (K,)) - 0.0, fl.smallest)
                new_m, db, last_stop = _commit(kreq, s, on_m, on, up, db, last_stop)
                new = bitsU[new_m]
                up = np.where(new, np.where(on, np.minimum(up + 1, LM.MIN_UP), 1), 0).astype(np.int16)
                on, on_m = new, new_m
            continue
        pc = P_cnt[:, s]
        fb = pc != pc
        if fb.any():
            pc = np.where(fb, 0.0, pc)
            if pav > 0:
                a_fb += fb
        kreq = np.maximum(Ls - pc, fl.smallest)
        new_m, db, last_stop = _commit(kreq, s, on_m, on, up, db, last_stop)
        a_st += fl.pop[new_m & ~on_m]
        new = bitsU[new_m]
        up = np.where(new, np.where(on, np.minimum(up + 1, LM.MIN_UP), 1), 0).astype(np.int16)
        on, on_m = new, new_m
        run_steps += on
        KU = fl.cap_of[new_m]
        MU = mload * KU
        net = Ls - pav
        c1 = net >= MU
        if c1.all():
            D_ = np.minimum(net, KU)
            ue = np.maximum(net - KU, 0.0)
            pv_load = pav
        else:
            D1 = np.minimum(net, KU)
            S_ = np.maximum(net - KU, 0.0)
            pv_load2 = np.minimum(np.maximum(Ls - MU, 0.0), pav)
            sur = pav - pv_load2
            D_ = np.where(c1, D1, MU)
            ue = np.where(c1, S_, 0.0)
            pv_load = np.where(c1, pav, pv_load2)
            curt = np.where(c1, 0.0, sur)
            dump = np.where(c1, 0.0, np.maximum(MU - Ls, 0.0))
            a_curt += curt
            a_dump += dump
        a_fuel += a_f * KU + b_f * D_
        a_ue += ue
        pos = ue > 1e-9
        a_ues += pos
        a_ev += pos & ~prev_pos
        prev_pos = pos
        a_th += D_
        a_pvl += pv_load
        a_pav += pav
        a_load += Ls
        a_onl += KU
        a_sc += 1
    if n:
        day[:, :, cur_day] = acc
    daily = {}
    for i, k in enumerate(keys):
        v = day[i]
        daily[k] = v if k in ("starts", "ue_steps", "fallback_steps", "scored_steps", "events") else v * LM.DT
    tot = {k: v.sum(axis=1) for k, v in daily.items()}
    tot["co2_t"] = tot["fuel_L"] * LM.CO2_KG_PER_L / 1000.0
    tot["ue_share"] = np.divide(tot["ue_kWh"], tot["load_kWh"], out=np.zeros(K), where=tot["load_kWh"] > 0)
    tot["curt_share"] = np.divide(tot["curt_kWh"], tot["pav_kWh"], out=np.zeros(K), where=tot["pav_kWh"] > 0)
    tot["pv_share_of_load"] = np.divide(tot["pv_load_kWh"] + tot["dis_kWh"], tot["load_kWh"], out=np.zeros(K),
                                        where=tot["load_kWh"] > 0)
    return dict(total=tot, daily=daily, run_h=run_steps * LM.DT, missing_steps=missing, ratings=fl.r)


def ref_constrained(L, P_av, pc, day_id, ratings, min_load, fuel, min_down, deadband, commit="greedy", gap="hold"):
    """Plain per-row reference of simulate_c (one policy), written independently: list-based unit
    choice, explicit availability and deadband counters. Returns fuel, UE, starts, events.
    commit="minfeas": enumerate every unit set within the available units that contains the forced units and pick
    the least (capacity, units, starts, mask index) meeting the requirement, else every available unit.
    gap="elapsed": a step without measured GHI commits with counted PV 0 and is not scored or counted.
    """
    r = list(map(float, ratings))
    U = len(r)
    a_f, b_f = LM.FUEL[fuel]
    classes = sorted(set(r))
    members = {c: [i for i in range(U) if r[i] == c] for c in classes}
    combos = []
    for cnt in itertools.product(*[range(len(members[c]) + 1) for c in classes]):
        if sum(cnt):
            combos.append((sum(c * k for c, k in zip(classes, cnt)), sum(cnt), cnt))
    combos.sort()
    small = min(r)
    on, up, last = [False] * U, [0] * U, [-(10 ** 9)] * U
    dbc, prev = 0, False
    fuel_l = ue_k = 0.0
    starts = events = 0
    for s in range(len(L)):
        if s > 0 and day_id[s] != day_id[s - 1]:
            prev = False
        gapstep = bool(np.isnan(P_av[s]))
        if gapstep:
            prev = False
            if gap == "hold":
                continue
        p = 0.0 if (gapstep or np.isnan(pc[s])) else pc[s]
        kreq = max(L[s] - p, small)
        avail = [on[u] or (s - last[u]) >= min_down if min_down else True for u in range(U)]
        forced = [on[u] and up[u] < LM.MIN_UP for u in range(U)]
        if commit == "minfeas":
            best = None
            for bits in itertools.product([False, True], repeat=U):
                if not any(bits) or any(bits[u] and not avail[u] for u in range(U)) or any(forced[u] and not bits[u] for u in range(U)):
                    continue
                cap = sum(r[u] for u in range(U) if bits[u])
                if cap < kreq - 1e-9:
                    continue
                key = (cap, sum(bits), sum(1 for u in range(U) if bits[u] and not on[u]), sum(1 << u for u in range(U) if bits[u]))
                if best is None or key < best[0]:
                    best = (key, list(bits))
            new = best[1] if best is not None else list(avail)
        else:
            feas = [c for c in combos if all(c[2][j] <= sum(1 for u in members[cl] if avail[u]) for j, cl in enumerate(classes))]
            big = [c for c in feas if c[0] >= kreq - 1e-9]
            cnt = big[0][2] if big else max(feas, key=lambda c: c[0])[2]
            new = [False] * U
            for j, cl in enumerate(classes):
                pref = sorted([u for u in members[cl] if avail[u]], key=lambda u: (-(forced[u] * 2 + on[u]), u))
                for u in pref[:cnt[j]]:
                    new[u] = True
            new = [new[u] or forced[u] for u in range(U)]
        if deadband:
            cap_in = sum(r[u] for u in range(U) if on[u])
            dbc = dbc + 1 if cap_in - kreq >= small else 0
            if any(on[u] and not new[u] for u in range(U)) and dbc < deadband:
                new = [new[u] or on[u] for u in range(U)]
        for u in range(U):
            if on[u] and not new[u]:
                last[u] = s
        if not gapstep:
            starts += sum(1 for u in range(U) if new[u] and not on[u])
        up = [(min(up[u] + 1, LM.MIN_UP) if on[u] else 1) if new[u] else 0 for u in range(U)]
        on = new
        if gapstep:
            continue
        KU = sum(r[u] for u in range(U) if on[u])
        MU = min_load * KU
        net = L[s] - P_av[s]
        if net >= MU:
            D_ = min(net, KU)
            ue = max(net - KU, 0.0)
        else:
            D_ = MU
            ue = 0.0
        fuel_l += (a_f * KU + b_f * D_) * LM.DT
        ue_k += ue * LM.DT
        pos = ue > 1e-9
        events += int(pos and not prev)
        prev = pos
    return fuel_l, ue_k, starts, events


# ================================================================ fixed-point paths
def fixed_point(simfn, sim_kw, L, pav, day, h, specs, max_pass=400):
    """Generic ce1_adaptive.run_paths: specs are (Cpol (n_levels, n), init_idx (nd), ctrl) where
    ctrl(ue_daily, load_daily) returns the day levels the path's own UE implies. A row whose
    recomputed levels equal its levels is frozen.
    """
    import ce1_adaptive as A
    nd = int(day.max()) + 1
    K = len(specs)
    idx = np.stack([np.asarray(s[1], np.int64) for s in specs]) if K else np.zeros((0, nd), np.int64)
    keys = None
    daily, total = {}, {}
    active = np.arange(K)
    work = 0
    for p in range(1, max_pass + 1):
        C = np.vstack([A.path_row(specs[k][0], idx[k], day) for k in active])
        r = simfn(L, pav, C, day, h, **sim_kw)
        work += len(active)
        if keys is None:
            keys = list(r["daily"].keys())
            daily = {k: np.zeros((K, nd)) for k in keys}
        keep = []
        for j, k in enumerate(active):
            new = specs[k][2](r["daily"]["ue_kWh"][j], r["daily"]["load_kWh"][j])
            if np.array_equal(new, idx[k]):
                for key in keys:
                    daily[key][k] = r["daily"][key][j]
                for key, v in r["total"].items():
                    total.setdefault(key, np.full(K, np.nan))[k] = v[j]
            else:
                idx[k] = new
                keep.append(k)
        if not keep:
            return dict(daily=daily, total=total), idx, p, work
        active = np.array(keep)
    raise RuntimeError(f"fixed point not reached in {max_pass} passes")


def hold_levels(ue_of, nd, i_ar, i_lim, N):
    """Day levels of the hold rule: the limit level on day d if any of days d-N..d-1 had UE."""
    idx = np.empty(nd, np.int64)
    last_ue = -(10 ** 9)
    for d in range(nd):
        idx[d] = i_lim if (N > 0 and d - last_ue <= N) else i_ar
        if ue_of(d, idx[d]) > 1e-9:
            last_ue = d
    return idx


def ad_levels(ue_of, load_d, theta0, gamma, kappa, abar_load, levels, lo, clip=np.inf):
    """ce1_adaptive.control / control_lookup (and the AW clip) in one sequential form."""
    import ce1_adaptive as A
    nd = len(load_d)
    idx = np.empty(nd, np.int64)
    x = float(theta0)
    abar = kappa * TGT * abar_load
    for d in range(nd):
        i = A.level_index(levels, max(x, lo))
        idx[d] = i
        if gamma:
            x = min(clip, max(lo, x + gamma * (ue_of(d, i) - kappa * TGT * load_d[d]) / abar))
    return idx


# ================================================================ selection helpers (ce1_wp57)
def tab_of(res, rows, part, scen):
    t = pd.DataFrame({k: np.asarray(v, float) for k, v in res["total"].items() if np.ndim(v) == 1})
    t.insert(0, "param", [x[1] for x in rows])
    t.insert(0, "policy", [x[0] for x in rows])
    t.insert(0, "set", part)
    t.insert(0, "scenario", scen)
    return t


# ================================================================ self-tests
def selftest(verbose=True):
    res = {}

    def ok(name, cond, msg=""):
        res[name] = bool(cond)
        if verbose:
            print(f"  [{'PASS' if cond else 'FAIL'}] {name} {msg}")

    grid, Lt, pav, cs = LM._toy(2)
    dd = LM.day_ids(grid)
    rng = np.random.default_rng(7)
    C = LM._counted(pav, cs, rng)
    for tag, mode in (("", LM.registered_v200()), ("_v21", LM.primary_v21())):
        with mode:
            a = LM.simulate(Lt, pav, C, dd, 3)
            b = simulate_c(Lt, pav, C, dd, 3)
        same = all(np.array_equal(a["total"][k], b["total"][k]) for k in LM.DAY_KEYS + ("co2_t", "ue_share", "curt_share"))
        same &= all(np.array_equal(a["daily"][k], b["daily"][k]) for k in LM.DAY_KEYS)
        ok(f"loop_copy_equals_simulate{tag}", same)
    # v2.1 availability choice equals ce1_load's minimum feasible table when every unit is available
    fl5 = fleet_c(LM.UNITS_BASE)
    vt = LM.variant_tables(LM.UNITS_BASE)
    rng2 = np.random.default_rng(64)
    o = rng2.integers(0, 32, 3000)
    f = o & rng2.integers(0, 32, 3000)
    kq = rng2.uniform(500, 7000, 3000)
    ok("v21_minfeas_av_all_available_equals_ce1_load", np.array_equal(
        _minfeas_av(fl5, kq, o, f, np.full(3000, 31, np.int64)), vt.minfeas(kq, None, o, f)))
    fl = fleet_c(LM.UNITS_BASE)
    full = (1 << fl.U) - 1
    nm = 1 << fl.U
    ok("first_all_available_is_identity", np.array_equal(fl.first[full], np.arange(len(fl.cap))))
    om, fm = np.meshgrid(np.arange(nm), np.arange(nm), indexing="ij")
    fm = fm & om
    ok("table_av_all_available_equals_table",
       all(np.array_equal(fl.table_av[c, om, fm, full], fl.table[c, om, fm]) for c in range(len(fl.cap))))
    # events equal ce1_wp55.events_of on the recorded UE steps
    from ce1_wp55 import events_of
    for tag, mode in (("", LM.registered_v200()), ("_v21", LM.primary_v21())):
        with mode:
            rr = LM.simulate(Lt, pav, C, dd, 3, record=True)
            b = simulate_c(Lt, pav, C, dd, 3)
        ev, _, _ = events_of(np.nan_to_num(rr["rec"]["ue"], nan=0.0), dd)
        ok(f"events_equal_wp55{tag}", np.array_equal(ev, b["total"]["events"]), f"({ev.tolist()})")
    # constrained loop against the plain reference, each constraint, every row of the toy batch
    good = True
    worst = 0.0
    for kw in ({"min_down": 12, "deadband": 0}, {"min_down": 0, "deadband": 3}, {"min_down": 0, "deadband": 0}):
        s_ = simulate_c(Lt, pav, C, dd, 3, commit="greedy", gap="hold", **kw)
        for k in range(C.shape[0]):
            f_, u_, st_, e_ = ref_constrained(Lt, pav, C[k], dd, LM.UNITS_BASE, LM.MIN_LOAD, "cat_c32", kw["min_down"], kw["deadband"])
            worst = max(worst, abs(f_ - s_["total"]["fuel_L"][k]) / f_, abs(u_ - s_["total"]["ue_kWh"][k]) / max(u_, 1.0))
            good &= int(st_) == int(s_["total"]["starts"][k]) and int(e_) == int(s_["total"]["events"][k])
    ok("constraints_equal_plain_reference", good and worst < 1e-9, f"(worst relative {worst:.1e})")
    good, worst = True, 0.0
    for kw in ({"min_down": 12, "deadband": 0}, {"min_down": 0, "deadband": 3}, {"min_down": 0, "deadband": 0}):
        s_ = simulate_c(Lt, pav, C, dd, 3, commit="minfeas", gap="elapsed", **kw)
        for k in range(C.shape[0]):
            f_, u_, st_, e_ = ref_constrained(Lt, pav, C[k], dd, LM.UNITS_BASE, LM.MIN_LOAD, "cat_c32", kw["min_down"], kw["deadband"],
                                              commit="minfeas", gap="elapsed")
            worst = max(worst, abs(f_ - s_["total"]["fuel_L"][k]) / f_, abs(u_ - s_["total"]["ue_kWh"][k]) / max(u_, 1.0))
            good &= int(st_) == int(s_["total"]["starts"][k]) and int(e_) == int(s_["total"]["events"][k])
    ok("v21_constraints_equal_plain_reference", good and worst < 1e-9, f"(worst relative {worst:.1e})")
    # the constraints change something on the toy (they are not inert) and never lower starts' floor
    m0 = simulate_c(Lt, pav, C, dd, 3)["total"]
    m1 = simulate_c(Lt, pav, C, dd, 3, min_down=12)["total"]
    m2 = simulate_c(Lt, pav, C, dd, 3, deadband=3)["total"]
    ok("constraints_bind_on_toy", (m1["starts"] != m0["starts"]).any() and (m2["starts"] <= m0["starts"]).all()
       and (m2["starts"] != m0["starts"]).any(), f"(starts {m0['starts'].tolist()} / {m1['starts'].tolist()} / {m2['starts'].tolist()})")
    # hold rule: N = 0 never holds; UE on day 2 with N = 3 holds days 3..5
    ue = np.zeros(10)
    ue[2] = 1.0
    f = lambda d, i: ue[d] if i == 0 else 0.0  # noqa: E731
    ok("hold_levels", np.array_equal(hold_levels(f, 10, 0, 1, 0), np.zeros(10, int))
       and np.array_equal(hold_levels(f, 10, 0, 1, 3), [0, 0, 0, 1, 1, 1, 0, 0, 0, 0]))
    # ad_levels equals ce1_adaptive.control_lookup / control
    import ce1_adaptive as A
    levels = np.array([0.9, 0.95, 0.99, 0.999, 1.0])
    rec = np.where(rng.random((5, 60)) < 0.1, 300.0, 0.0)
    rec[-1] = 0.0
    load = np.full(60, 5e4)
    i1 = A.control_lookup(rec, load, 0.95, 0.005, 0.5, 5e4, levels, 0.9)
    i2 = ad_levels(lambda d, i: rec[i, d], load, 0.95, 0.005, 0.5, 5e4, levels, 0.9)
    path_ue = rec[i1, np.arange(60)]
    j1, _ = A.control(path_ue, load, 0.95, 0.005, 0.5, 5e4, levels, 0.9)
    j2 = ad_levels(lambda d, i: path_ue[d], load, 0.95, 0.005, 0.5, 5e4, levels, 0.9)
    ok("ad_levels_equals_adaptive", np.array_equal(i1, i2) and np.array_equal(j1, j2))
    ok("su_high_8.54", abs(su_litres("high") - 8.541666666666666) < 1e-12)
    return res


# ================================================================ one fold
def run_fold(root: Path, yr: int, smoke=False) -> bool:
    import ce1_adaptive as A
    import ce1_folds as FC
    import ce1_s6 as S6
    from ce1_wp61b import issue_cap
    R = root / "04_results" / "ce1"
    out = R / "S64"
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    checks, inputs = {}, {}

    def check(name, cond, msg=""):
        checks[name] = dict(ok=bool(cond), msg=str(msg))
        log(f"{'PASS' if cond else 'FAIL'}  {name}  {msg}")

    def inp(rel):
        inputs[rel] = md5(R / rel)
        return R / rel

    for k, v in selftest(verbose=False).items():
        check(f"selftest_{k}", v)
    geo, m, unit96, scales, _ = S6.load_inputs(R / "S1", R / "S1M", R / "S1L")
    for rel in ("S1/yulara_geo_5min.parquet", "S1M/available_pv_map.npz", "S1L/load_profile.npz", "S1L/load_scales.json"):
        inp(rel)
    D.extend_theta_low()
    rows = S6.row_table()
    cache = FC.load_cache_ext(inp(f"S2/yulara_f{yr}_h3.parquet"), inp(f"S2X/yulara_f{yr}_h3.parquet"))
    tot_e = pd.read_parquet(inp(f"S6E/f{yr}_h3_totals.parquet"))
    sel_e = json.load(open(inp(f"S6E/f{yr}_h3_select.json")))["selections"]
    s57 = pd.read_parquet(inp(f"S57/f{yr}_h3_H1x.parquet"))
    if smoke:  # code path test only: a thinned level set
        s61s = json.load(open(R / "S61B" / f"f{yr}_h3_select.json"))
        keep = set()
        for pol in POLICIES:
            g = [r for r in rows if r[0] == pol]
            keep |= set(g[::12]) | {r for r in g if r[1] == 1.0}
            for sel_ in (sel_e["main"], sel_e["shapeB"], s61s["battery_day"], s61s["batteryB_day"]):
                keep |= {r for r in g if abs(r[1] - float(sel_["policies"][pol]["A_R"]["param"])) < 1e-9}
        rows = [r for r in rows if r[0] in S6.ANCHORS or r in keep]
        out = R / "S64_smoke"
        out.mkdir(parents=True, exist_ok=True)

    # ---------------- counted PV on H1x, H2 and the test year (S6 / S57 constructions)
    prep = S6.prepare(cache, H, yr, geo, m, rows)
    calib = S6.as_dict(cache[cache["split"] == "calib"])
    H1, H2d = D.split_halves(calib)
    cy = yr - 1
    tgt = pd.DatetimeIndex(H1["idx"]) + pd.Timedelta(minutes=5 * H)
    gridx = S6.year_grid(f"{cy}-01-01", f"{cy}-06-30 23:55")
    pavx = LM.available_pv_on_grid(gridx, geo["ghi"].reindex(gridx).values, m)
    pv_of = S6.PVAt(tgt, m)
    pav_t = pv_of(H1["y"])
    prep["H1x"] = dict(grid=gridx, pav=pavx, day=LM.day_ids(gridx),
                       C=S6.place(gridx, tgt, S6.counted_matrix(rows, H2d, H1, pv_of, pav_t)))
    log(f"fold {yr}: counted PV ready ({time.time() - t0:.0f} s, {len(rows)} rows)")
    hb = [i for i, r in enumerate(rows) if r[0] == "handbook"][0]

    def row_ix(pol, p_, rows_=rows):
        return [i for i, r_ in enumerate(rows_) if r_[0] == pol and abs(r_[1] - p_) < 1e-9][0]

    def pol_levels(P, pol, rows_=rows):
        ix = [i for i, r in enumerate(rows_) if r[0] == pol]
        lv = np.array([rows_[i][1] for i in ix])
        o = np.argsort(lv)
        return lv[o], np.array(ix)[o]

    def sc_kw(sc, L):
        return dict(ratings=LM.fleet_for_peak(L.max()), min_load=sc["min_load"], fuel=sc["fuel"])

    tables, cells, bat_rows = [], [], []

    def ad_block(sname, sc, simfn, sim_kw, rows_, Ctest, rec_daily, totT, H2tab, pols, tag, kinds=(("A_D", np.inf), ("AW_clip", 1.0)),
                 theta0=None):
        """A-D (primary) and AW (clip 1.0) paths for pols, from the record of every level."""
        P = prep["test"]
        L = S6.load_for(P["grid"], unit96, scales, sc)
        abar = A.h2_mean_daily_load(prep["H2"], unit96, scales, sc)
        load_d = rec_daily["load_kWh"][[i for i, r in enumerate(rows_) if r[0] == "handbook"][0]]
        specs, meta = [], []
        for pol in pols:
            lv, ix = pol_levels(P, pol, rows_)
            rec_ue = rec_daily["ue_kWh"][ix]
            if theta0 is not None:
                t0_ = float(theta0[pol])
            else:
                t0_, _ = LM.select_AR(dict(ue_share=H2tab[H2tab.policy == pol].sort_values("param").ue_share.to_numpy(float),
                                           fuel_L=H2tab[H2tab.policy == pol].sort_values("param").fuel_L.to_numpy(float)), lv, pol)
            lo = float(lv.min())
            for kind, clip in kinds:
                init = ad_levels(lambda d, i, R_=rec_ue: R_[i, d], load_d, t0_, GAMMA, KAPPA, abar, lv, lo, clip)
                ctrl = (lambda ue, ld, lv_=lv, t0__=t0_, lo_=lo, cl=clip:
                        ad_levels(lambda d, i, u=ue: u[d], load_d, t0__, GAMMA, KAPPA, abar, lv_, lo_, cl))
                specs.append((Ctest[ix], init, ctrl))
                meta.append((pol, kind, t0_))
        r, idxp, passes, work = fixed_point(simfn, sim_kw, L, P["pav"], P["day"], H, specs)
        check(f"fixed_point_{tag}_f{yr}_{sname}", 1 <= passes < 400, f"({passes} passes, {work} row-years)")
        hbf = float(totT["fuel_L"][[i for i, rr in enumerate(rows_) if rr[0] == "handbook"][0]])
        out_ = []
        for k, (pol, kind, t0_) in enumerate(meta):
            lv, _ = pol_levels(P, pol, rows_)
            lvp = lv[idxp[k]]
            out_.append(dict(fold=yr, scenario=sname, item=tag, policy=pol, selection=kind, param=np.nan, theta0=t0_,
                             fuel_L=float(r["total"]["fuel_L"][k]), ue_kWh=float(r["total"]["ue_kWh"][k]),
                             ue_share=float(r["total"]["ue_share"][k]), starts=float(r["total"]["starts"][k]),
                             events=float(r["total"].get("events", np.full(len(meta), np.nan))[k]),
                             fuel_handbook=hbf, handbook_starts=np.nan,
                             days_at_limit=int((lvp >= 1.0 - 1e-9).sum()), passes=passes))
        return out_, r, idxp, meta

    # ================================================================ item 1: spill-matched scale
    import ce1_wp57 as W57
    for sname, sc in SPILL.items():
        res = {}
        for part in ("H1x", "H2", "test"):
            P = prep[part]
            L = S6.load_for(P["grid"], unit96, scales, sc)
            res[part] = LM.simulate(L, P["pav"], P["C"], P["day"], H, **sc_kw(sc, L), battery=False)
            t = tab_of(res[part], rows, part, sname)
            tables.append(t)
            if sname == "spillA" and part in ("H2", "test"):
                ref = tot_e[(tot_e.scenario == "spill") & (tot_e.set == part)]
                mm = t.merge(ref, on=["policy", "param"], suffixes=("", "_ref"))
                worst = max(float(np.max(np.abs(mm[k] - mm[k + "_ref"]) / np.maximum(np.abs(mm[k + "_ref"]), 1.0)))
                            for k in ("fuel_L", "ue_kWh", "starts"))
                check(f"spillA_reproduces_S6E_spill_f{yr}_{part}", len(mm) == len(t) and worst < 1e-9, f"(worst {worst:.1e})")
            check(f"handbook_zero_ue_{sname}_f{yr}_{part}", float(res[part]["total"]["ue_kWh"][hb]) == 0.0)
        T = {p_: tables[-3 + i] for i, p_ in enumerate(("H1x", "H2", "test"))}
        hbT = float(res["test"]["total"]["fuel_L"][hb])
        hbS = float(res["test"]["total"]["starts"][hb])
        for pol in POLICIES:
            X = W57.pooled([T["H2"][T["H2"].policy == pol].reset_index(drop=True),
                            T["H1x"][T["H1x"].policy == pol].reset_index(drop=True)])
            for vname, (tab, tg) in {"A_R": (T["H2"], TGT), "a_half_H2": (T["H2"], HALF), "b_xfit_0.002": (X, TGT),
                                     "b_xfit_0.001": (X, HALF)}.items():
                p_, met = W57.select(tab, pol, tg)
                rr = W57.at(T["test"], pol, p_)
                cells.append(dict(fold=yr, scenario=sname, item="spill", policy=pol, selection=vname, param=p_,
                                  met_on_selection=met, fuel_L=rr["fuel_L"], ue_kWh=rr["ue_kWh"], ue_share=rr["ue_share"],
                                  starts=rr["starts"], events=np.nan, fuel_handbook=hbT, handbook_starts=hbS))
        ad, rsp, idxp, meta = ad_block(sname, sc, LM.simulate, dict(**sc_kw(sc, S6.load_for(prep["test"]["grid"], unit96, scales, sc)), battery=False),
                                       rows, prep["test"]["C"], res["test"]["daily"], res["test"]["total"], T["H2"], POLICIES, "spill")
        for c_ in ad:
            c_["handbook_starts"] = hbS
        cells += ad
        if sname == "spillA":
            # the generic fixed point equals ce1_adaptive.run_paths on the same inputs (A-D rows)
            P = prep["test"]
            L = S6.load_for(P["grid"], unit96, scales, sc)
            pol_C, rec_ue, specs = {}, {}, []
            for pol in ("cqr", "frac_pv"):
                lv, ix = pol_levels(P, pol)
                pol_C[pol] = (lv, P["C"][ix])
                rec_ue[pol] = res["test"]["daily"]["ue_kWh"][ix]
                t0_ = [m_["theta0"] for m_ in ad if m_["policy"] == pol and m_["selection"] == "A_D"][0]
                specs.append((pol, GAMMA, KAPPA, t0_, float(lv.min())))
            abar = A.h2_mean_daily_load(prep["H2"], unit96, scales, sc)
            load_d = res["test"]["daily"]["load_kWh"][hb]
            ra, _, _, _, _ = A.run_paths(L, P["pav"], P["day"], H, dict(**sc_kw(sc, L), battery=False), pol_C, specs, rec_ue,
                                         load_d, abar)
            mine = [m_ for m_ in ad if m_["selection"] == "A_D" and m_["policy"] in ("cqr", "frac_pv")]
            check(f"fixed_point_equals_run_paths_f{yr}",
                  all(abs(ra["total"]["fuel_L"][j] - mine[j]["fuel_L"]) < 1e-6 * mine[j]["fuel_L"]
                      and abs(ra["total"]["ue_kWh"][j] - mine[j]["ue_kWh"]) < 1e-6 for j in range(2)))
        log(f"fold {yr} {sname}: done ({time.time() - t0:.0f} s)")

    # ================================================================ item 4 (and the H2 identity): events on H2
    ev_tabs = {}
    for sname in PRIMARY:
        sc = S6.SCENARIOS[sname]
        P = prep["H2"]
        L = S6.load_for(P["grid"], unit96, scales, sc)
        r = simulate_c(L, P["pav"], P["C"], P["day"], H, **sc_kw(sc, L))
        t = tab_of(r, rows, "H2", sname + "_events")
        ref = tot_e[(tot_e.scenario == sname) & (tot_e.set == "H2")]
        mm = t.merge(ref, on=["policy", "param"], suffixes=("", "_ref"))
        worst = max(float(np.max(np.abs(mm[k] - mm[k + "_ref"]) / np.maximum(np.abs(mm[k + "_ref"]), 1.0)))
                    for k in ("fuel_L", "ue_kWh", "starts", "curt_kWh", "online_kW_h"))
        check(f"loop_copy_reproduces_S6E_H2_all_rows_f{yr}_{sname}", len(mm) == len(t) and worst < 1e-9,
              f"({len(mm)} rows, worst {worst:.1e})")
        ev_tabs[sname] = t
        tables.append(t)
    for sname in PRIMARY:
        sc = S6.SCENARIOS[sname]
        t = ev_tabs[sname]
        teT = tot_e[(tot_e.scenario == sname) & (tot_e.set == "test")]
        hbT = float(teT[teT.policy == "handbook"].fuel_L.iloc[0])
        hbS = float(teT[teT.policy == "handbook"].starts.iloc[0])
        chosen = []
        for pol in POLICIES:
            g = t[t.policy == pol].sort_values("param")
            okm = (g.ue_share.to_numpy() <= TGT + EPS) & (g.events.to_numpy() <= EVENT_MAX_H2)
            if okm.any():
                p_ = float(g.param.to_numpy()[okm][np.argmin(g.fuel_L.to_numpy()[okm])])
            else:
                p_ = float(g.param.max())
            p_ar = float(sel_e[sname]["policies"][pol]["A_R"]["param"])
            chosen += [(pol, p_, "A_R_events", bool(okm.any())), (pol, p_ar, "A_R", True)]
        P = prep["test"]
        L = S6.load_for(P["grid"], unit96, scales, sc)
        ix = [row_ix(pol, p_) for pol, p_, _, _ in chosen]
        r = simulate_c(L, P["pav"], P["C"][ix], P["day"], H, **sc_kw(sc, L))
        for j, (pol, p_, kind, met) in enumerate(chosen):
            ref = teT[(teT.policy == pol) & np.isclose(teT.param.astype(float), p_)].iloc[0]
            check(f"events_test_row_equals_S6E_f{yr}_{sname}_{pol}_{kind}", abs(r["total"]["fuel_L"][j] / ref.fuel_L - 1) < 1e-9)
            cells.append(dict(fold=yr, scenario=sname, item="events", policy=pol, selection=kind, param=p_, met_on_selection=met,
                              fuel_L=float(r["total"]["fuel_L"][j]), ue_kWh=float(r["total"]["ue_kWh"][j]),
                              ue_share=float(r["total"]["ue_share"][j]), starts=float(r["total"]["starts"][j]),
                              events=float(r["total"]["events"][j]), fuel_handbook=hbT, handbook_starts=hbS,
                              events_H2=float(t[(t.policy == pol) & np.isclose(t.param.astype(float), p_)].events.iloc[0])))
    log(f"fold {yr}: events done ({time.time() - t0:.0f} s)")

    # ================================================================ item 3: hold comparator
    for sname in PRIMARY:
        sc = S6.SCENARIOS[sname]
        teT = tot_e[(tot_e.scenario == sname) & (tot_e.set == "test")]
        hbT = float(teT[teT.policy == "handbook"].fuel_L.iloc[0])
        hbS = float(teT[teT.policy == "handbook"].starts.iloc[0])
        PH, PT = prep["H2"], prep["test"]
        LH = S6.load_for(PH["grid"], unit96, scales, sc)
        LT = S6.load_for(PT["grid"], unit96, scales, sc)
        kwH, kwT = sc_kw(sc, LH), sc_kw(sc, LT)
        Nsel = {}
        # H2: every N for every policy, at the A-R level
        for pol in POLICIES:
            p_ar = float(sel_e[sname]["policies"][pol]["A_R"]["param"])
            i_ar, i_lim = row_ix(pol, p_ar), row_ix(pol, 1.0)
            CH = PH["C"][[i_ar, i_lim]]
            base = LM.simulate(LH, PH["pav"], CH, PH["day"], H, **kwH, battery=False)["daily"]["ue_kWh"]
            nd = int(PH["day"].max()) + 1
            specs = []
            for N in HOLD_N:
                init = hold_levels(lambda d, i, b=base: b[i, d], nd, 0, 1, N)
                ctrl = lambda ue, ld, N_=N, nd_=nd: hold_levels(lambda d, i, u=ue: u[d], nd_, 0, 1, N_)  # noqa: E731
                specs.append((CH, init, ctrl))
            r, _, passes, _ = fixed_point(LM.simulate, dict(**kwH, battery=False), LH, PH["pav"], PH["day"], H, specs)
            ok_n = [N for j, N in enumerate(HOLD_N) if r["total"]["ue_share"][j] <= HALF + EPS]
            Nsel[pol] = (min(ok_n) if ok_n else HOLD_N[-1], bool(ok_n), i_ar, i_lim,
                         {int(N): float(r["total"]["ue_share"][j]) for j, N in enumerate(HOLD_N)})
            check(f"hold_N0_equals_AR_H2_f{yr}_{sname}_{pol}",
                  abs(r["total"]["fuel_L"][0] / float(tot_e[(tot_e.scenario == sname) & (tot_e.set == "H2") & (tot_e.policy == pol)
                                                       & np.isclose(tot_e.param.astype(float), p_ar)].fuel_L.iloc[0]) - 1) < 1e-9)
        specs, meta = [], []
        ndT = int(PT["day"].max()) + 1
        for pol in POLICIES:
            N, found, i_ar, i_lim, curve = Nsel[pol]
            CT = PT["C"][[i_ar, i_lim]]
            base = LM.simulate(LT, PT["pav"], CT, PT["day"], H, **kwT, battery=False)["daily"]["ue_kWh"]
            init = hold_levels(lambda d, i, b=base: b[i, d], ndT, 0, 1, N)
            ctrl = lambda ue, ld, N_=N: hold_levels(lambda d, i, u=ue: u[d], ndT, 0, 1, N_)  # noqa: E731
            specs.append((CT, init, ctrl))
            meta.append((pol, N, found, curve))
        r, idxp, passes, _ = fixed_point(LM.simulate, dict(**kwT, battery=False), LT, PT["pav"], PT["day"], H, specs)
        check(f"hold_fixed_point_f{yr}_{sname}", 1 <= passes < 400, f"({passes} passes)")
        for k, (pol, N, found, curve) in enumerate(meta):
            cells.append(dict(fold=yr, scenario=sname, item="hold", policy=pol, selection="hold", param=float(sel_e[sname]["policies"][pol]["A_R"]["param"]),
                              hold_N=N, hold_N_found=found, hold_H2_curve=json.dumps(curve),
                              fuel_L=float(r["total"]["fuel_L"][k]), ue_kWh=float(r["total"]["ue_kWh"][k]),
                              ue_share=float(r["total"]["ue_share"][k]), starts=float(r["total"]["starts"][k]),
                              events=np.nan, fuel_handbook=hbT, handbook_starts=hbS, days_at_limit=int(idxp[k].sum())))
    log(f"fold {yr}: hold done ({time.time() - t0:.0f} s)")

    # ================================================================ item 2: commitment constraints
    crow = [i for i, r_ in enumerate(rows) if r_[0] == "handbook" or r_[0] in CONSTR_POL]
    rows_c = [rows[i] for i in crow]
    hb_c = [i for i, r_ in enumerate(rows_c) if r_[0] == "handbook"][0]
    for cname, ckw in CONSTRAINTS.items():
        for sname in PRIMARY:
            sc = S6.SCENARIOS[sname]
            res = {}
            for part in ("H1x", "H2", "test"):
                P = prep[part]
                L = S6.load_for(P["grid"], unit96, scales, sc)
                res[part] = simulate_c(L, P["pav"], P["C"][crow], P["day"], H, **sc_kw(sc, L), **ckw)
                tables.append(tab_of(res[part], rows_c, part, f"{sname}_{cname}"))
            T = {p_: tables[-3 + i] for i, p_ in enumerate(("H1x", "H2", "test"))}
            hbT = float(res["test"]["total"]["fuel_L"][hb_c])
            hbS = float(res["test"]["total"]["starts"][hb_c])
            check(f"constrained_handbook_ue_{cname}_f{yr}_{sname}", True,
                  f"(handbook UE {float(res['test']['total']['ue_kWh'][hb_c]):.3f} kWh, starts {hbS:.0f})")
            for pol in CONSTR_POL:
                X = W57.pooled([T["H2"][T["H2"].policy == pol].reset_index(drop=True),
                                T["H1x"][T["H1x"].policy == pol].reset_index(drop=True)])
                for vname, (tab, tg) in {"A_R": (T["H2"], TGT), "b_xfit_0.001": (X, HALF)}.items():
                    p_, met = W57.select(tab, pol, tg)
                    rr = W57.at(T["test"], pol, p_)
                    cells.append(dict(fold=yr, scenario=sname, item=cname, policy=pol, selection=vname, param=p_,
                                      met_on_selection=met, fuel_L=rr["fuel_L"], ue_kWh=rr["ue_kWh"], ue_share=rr["ue_share"],
                                      starts=rr["starts"], events=rr["events"], fuel_handbook=hbT, handbook_starts=hbS))
            cells.append(dict(fold=yr, scenario=sname, item=cname, policy="handbook", selection="handbook", param=np.nan,
                              fuel_L=hbT, ue_kWh=float(res["test"]["total"]["ue_kWh"][hb_c]),
                              ue_share=float(res["test"]["total"]["ue_share"][hb_c]), starts=hbS,
                              events=float(res["test"]["total"]["events"][hb_c]), fuel_handbook=hbT, handbook_starts=hbS))
            P = prep["test"]
            L = S6.load_for(P["grid"], unit96, scales, sc)
            ad, _, _, _ = ad_block(sname, sc, simulate_c, dict(**sc_kw(sc, L), **ckw), rows_c, P["C"][crow],
                                   res["test"]["daily"], res["test"]["total"], T["H2"], CONSTR_POL, cname, kinds=(("A_D", np.inf),))
            for c_ in ad:
                c_["handbook_starts"] = hbS
            cells += [c_ for c_ in ad if c_["selection"] == "A_D"]
            log(f"fold {yr} {cname} {sname}: done ({time.time() - t0:.0f} s)")

    # ================================================================ item 5: battery energy (S61B scenarios)
    from ce1_wp61b import SCEN as BSCEN
    b61 = pd.read_parquet(inp(f"S61B/f{yr}_h3_totals.parquet"))
    s61ad = pd.read_csv(inp("S61B/ad_cells.csv"), float_precision="round_trip")
    s61 = json.load(open(inp(f"S61B/f{yr}_h3_select.json")))
    P = prep["test"]
    for sname, sc in BSCEN.items():
        L = S6.load_for(P["grid"], unit96, scales, sc)
        cap = issue_cap(P["pav"], H)
        pick = [("handbook", np.nan)] + [(pol, float(s61[sname]["policies"][pol]["A_R"]["param"])) for pol in POLICIES]
        ix = [hb] + [row_ix(pol, p_) for pol, p_ in pick[1:]]
        r = LM.simulate(L, P["pav"], P["C"][ix], P["day"], H, ratings=LM.fleet_for_peak(L.max()), min_load=sc["min_load"],
                        fuel=sc["fuel"], battery=True, credit_cap=cap, record=True)
        te = b61[(b61.scenario == sname) & (b61.set == "test")]
        sc_m = ~np.isnan(P["pav"])
        Eb = LM.BATT_E
        for j, (pol, p_) in enumerate(pick):
            ref = te[(te.policy == pol) & ((te.param.isna()) if pol == "handbook" else np.isclose(te.param.astype(float), p_))].iloc[0]
            check(f"battery_row_reproduces_S61B_f{yr}_{sname}_{pol}", abs(r["total"]["fuel_L"][j] / ref.fuel_L - 1) < 1e-9
                  and abs(r["total"]["dis_kWh"][j] - ref.dis_kWh) < 1e-6 * max(ref.dis_kWh, 1.0))
            E = r["rec"]["E"][j][sc_m]
            bat_rows.append(dict(fold=yr, scenario=sname, policy=pol, param=p_, selection="handbook" if pol == "handbook" else "A_R", chg_kWh=float(ref.chg_kWh), dis_kWh=float(ref.dis_kWh),
                                 throughput_kWh=float(ref.chg_kWh + ref.dis_kWh),
                                 full_cycles=float((ref.chg_kWh + ref.dis_kWh) / (2 * Eb)),
                                 soc_mean=float(E.mean() / Eb), soc_min=float(E.min() / Eb), soc_max=float(E.max() / Eb),
                                 share_at_floor=float(np.mean(E <= LM.E_LO * Eb + 1e-6)),
                                 share_at_or_below_band=float(np.mean(E <= LM.E_FLOOR * Eb + 1e-6)),
                                 share_full=float(np.mean(E >= LM.E_HI * Eb - 1e-6))))
        # A-D rows (S61B construction): the record of every level, the fixed point, then the SoC of the paths
        kwb = dict(ratings=LM.fleet_for_peak(L.max()), min_load=sc["min_load"], fuel=sc["fuel"], battery=True, credit_cap=cap)
        rf = LM.simulate(L, P["pav"], P["C"], P["day"], H, **kwb)
        t0map = {pol: float(s61[sname]["policies"][pol]["A_R"]["param"]) for pol in POLICIES}
        ad, rad, idxp, meta = ad_block(sname, sc, LM.simulate, kwb, rows, P["C"], rf["daily"], rf["total"], None, POLICIES,
                                       "battery", kinds=(("A_D", np.inf),), theta0=t0map)
        ref = s61ad[(s61ad.fold == yr) & (s61ad.scenario == sname) & s61ad.primary].set_index("policy")
        worst = max(abs(c_["fuel_L"] / float(ref.loc[c_["policy"], "fuel_L"]) - 1) for c_ in ad)
        worst_ue = max(abs(c_["ue_kWh"] - float(ref.loc[c_["policy"], "ue_kWh"])) for c_ in ad)
        check(f"battery_AD_reproduces_S61B_f{yr}_{sname}", worst < 1e-7 and worst_ue < 1e-3, f"(fuel {worst:.1e}, UE {worst_ue:.1e} kWh)")
        import ce1_adaptive as A
        Cp = []
        for k, (pol, kind, t0_) in enumerate(meta):
            lv, ixp = pol_levels(P, pol)
            Cp.append(A.path_row(P["C"][ixp], idxp[k], P["day"]))
        rp = LM.simulate(L, P["pav"], np.vstack(Cp), P["day"], H, record=True, **kwb)
        for k, (pol, kind, t0_) in enumerate(meta):
            E = rp["rec"]["E"][k][sc_m]
            ch, dis = float(rp["total"]["chg_kWh"][k]), float(rp["total"]["dis_kWh"][k])
            bat_rows.append(dict(fold=yr, scenario=sname, policy=pol, param=np.nan, selection="A_D", chg_kWh=ch, dis_kWh=dis,
                                 throughput_kWh=ch + dis, full_cycles=(ch + dis) / (2 * Eb),
                                 soc_mean=float(E.mean() / Eb), soc_min=float(E.min() / Eb), soc_max=float(E.max() / Eb),
                                 share_at_floor=float(np.mean(E <= LM.E_LO * Eb + 1e-6)),
                                 share_at_or_below_band=float(np.mean(E <= LM.E_FLOOR * Eb + 1e-6)),
                                 share_full=float(np.mean(E >= LM.E_HI * Eb - 1e-6))))
    log(f"fold {yr}: battery done ({time.time() - t0:.0f} s)")

    # ---------------- H1x reproduces S57 (shape A and B at the primary scale), a probe of rows
    for sname in PRIMARY:
        sc = S6.SCENARIOS[sname]
        P = prep["H1x"]
        L = S6.load_for(P["grid"], unit96, scales, sc)
        pr = [i for i, r_ in enumerate(rows) if r_[0] in S6.ANCHORS or (r_[0], r_[1]) in set(S6.PROBES)]
        r = LM.simulate(L, P["pav"], P["C"][pr], P["day"], H, **sc_kw(sc, L), battery=False)["total"]
        ref = s57[s57.scenario == sname]
        worst = 0.0
        for j, i in enumerate(pr):
            pol, par = rows[i]
            rr = ref[(ref.policy == pol) & ((ref.param.isna()) if par != par else np.isclose(ref.param.astype(float), par))]
            worst = max(worst, abs(r["fuel_L"][j] / float(rr.fuel_L.iloc[0]) - 1), abs(r["ue_kWh"][j] - float(rr.ue_kWh.iloc[0])))
        check(f"H1x_reproduces_S57_f{yr}_{sname}", worst < 1e-9, f"(worst {worst:.1e})")

    tab = pd.concat(tables, ignore_index=True)
    tab.to_parquet(out / f"f{yr}_totals.parquet")
    pd.DataFrame(cells).to_csv(out / f"f{yr}_cells.csv", index=False)
    pd.DataFrame(bat_rows).to_csv(out / f"f{yr}_battery.csv", index=False)
    ok = all(c_["ok"] for c_ in checks.values())
    json.dump(dict(fold=yr, ok=ok, seconds=round(time.time() - t0, 1), checks=checks, inputs=inputs,
                   outputs={p.name: md5(p) for p in (out / f"f{yr}_totals.parquet", out / f"f{yr}_cells.csv", out / f"f{yr}_battery.csv")},
                   code={p: md5(HERE / p) for p in ("ce1_s64.py", "ce1_load.py", "ce1_s6.py", "ce1_adaptive.py", "ce1_wp57.py",
                                                    "ce1_wp61b.py", "ce1_dispatch.py")}),
              open(out / f"f{yr}_checks.json", "w"), indent=1, default=float)
    log(f"fold {yr}: {'ALL CHECKS PASS' if ok else 'CHECK FAILED'} ({len(checks)} checks, {time.time() - t0:.0f} s)")
    return ok


# ================================================================ analysis and reading rules (D-122, D-123)
def analyse(root: Path) -> bool:
    import ce1_wp57 as W57
    R = root / "04_results" / "ce1"
    out = R / "S64"
    checks = {}

    def check(name, cond, msg=""):
        checks[name] = dict(ok=bool(cond), msg=str(msg))
        log(f"{'PASS' if cond else 'FAIL'}  {name}  {msg}")

    fold_ok = {}
    for yr in FOLDS:
        c = json.load(open(out / f"f{yr}_checks.json"))
        fold_ok[yr] = c["ok"]
        check(f"fold_{yr}_ok", c["ok"], f"({len(c['checks'])} checks)")
        for f, h in c["outputs"].items():
            check(f"fold_{yr}_{f}_unchanged", md5(out / f) == h)
    cells = pd.concat([pd.read_csv(out / f"f{yr}_cells.csv") for yr in FOLDS], ignore_index=True)
    tot = {yr: pd.read_parquet(out / f"f{yr}_totals.parquet") for yr in FOLDS}
    # c_prior at the spill scale: pooled test totals of every earlier fold (fold 2019: A-R on its H2)
    extra = []
    for yr in FOLDS:
        for sname in SPILL:
            te = tot[yr][(tot[yr].scenario == sname) & (tot[yr].set == "test")].reset_index(drop=True)
            h2 = tot[yr][(tot[yr].scenario == sname) & (tot[yr].set == "H2")].reset_index(drop=True)
            hbT = float(te[te.policy == "handbook"].fuel_L.iloc[0])
            hbS = float(te[te.policy == "handbook"].starts.iloc[0])
            prior = [tot[y][(tot[y].scenario == sname) & (tot[y].set == "test")].reset_index(drop=True) for y in FOLDS if y < yr]
            for pol in POLICIES:
                for tg, nm in ((TGT, "c_prior_0.002"), (HALF, "c_prior_0.001")):
                    tab = W57.pooled([t[t.policy == pol].reset_index(drop=True) for t in prior]) if prior else h2
                    p_, met = W57.select(tab, pol, tg)
                    rr = W57.at(te, pol, p_)
                    extra.append(dict(fold=yr, scenario=sname, item="spill", policy=pol, selection=nm, param=p_,
                                      met_on_selection=met, fuel_L=rr["fuel_L"], ue_kWh=rr["ue_kWh"], ue_share=rr["ue_share"],
                                      starts=rr["starts"], events=np.nan, fuel_handbook=hbT, handbook_starts=hbS))
    cells = pd.concat([cells, pd.DataFrame(extra)], ignore_index=True)
    cells["meets"] = cells.ue_share <= TGT + EPS
    cells["saving_pct"] = 100 * (1 - cells.fuel_L / cells.fuel_handbook)
    cells["d_starts"] = cells.starts - cells.handbook_starts
    for case in SU:
        cells[f"net_pct_{case}"] = 100 * (cells.fuel_handbook - cells.fuel_L - su_litres(case) * cells.d_starts) / cells.fuel_handbook
    cells.to_csv(out / "cases.csv", index=False)
    rd = {}
    sp = cells[cells.item == "spill"]
    summ = sp.groupby(["selection"]).apply(lambda g: pd.Series(dict(
        met_conformal=int(g[g.policy.isin(CP)].meets.sum()), met_quantile=int(g[g.policy.isin(QP)].meets.sum()),
        med_saving_conformal=float(g[g.policy.isin(CP)].saving_pct.median()),
        med_saving_quantile=float(g[g.policy.isin(QP)].saving_pct.median()),
        med_net_high_quantile=float(g[g.policy.isin(QP)].net_pct_high.median())))).reset_index()
    per_pol = {}
    fires = []
    for pol in QP:
        a = sp[(sp.selection == "b_xfit_0.001") & (sp.policy == pol)]
        b = sp[(sp.selection == "A_D") & (sp.policy == pol)]
        per_pol[pol] = dict(static_met=int(a.meets.sum()), ad_met=int(b.meets.sum()), n=len(a))
        fires.append(int(a.meets.sum()) < 10 and int(b.meets.sum()) >= 11)
    a = sp[(sp.selection == "b_xfit_0.001") & sp.policy.isin(QP)]
    b = sp[(sp.selection == "A_D") & sp.policy.isin(QP)]
    rd["1_spill"] = dict(summary=summ.to_dict("records"), per_quantile_policy=per_pol, trigger_fires_any_policy=bool(any(fires)),
                         pooled_static_met_of_36=int(a.meets.sum()), pooled_ad_met_of_36=int(b.meets.sum()),
                         rule="D-121 trigger fires if, at the spill-matched scale, the static cross-fitted half rule meets the target "
                              "in fewer than 10 of 12 cases while A-D meets it in at least 11 of 12 (per quantile policy, D-123)")
    # ---- item 2: commitment constraints
    r2 = {}
    base = pd.read_csv(R / "WP63" / "cases.csv")
    for cname in CONSTRAINTS:
        g = cells[cells.item == cname]
        d = {}
        for selname in ("A_R", "b_xfit_0.001", "A_D"):
            q = g[(g.selection == selname) & g.policy.isin(QP)]
            c_ = g[(g.selection == selname) & g.policy.isin(CP)]
            bq = base[(base.selection == selname) & base.policy.isin(QP)]
            d[selname] = dict(met_quantile=int(q.meets.sum()), n_quantile=len(q), med_saving_quantile=float(q.saving_pct.median()),
                              med_saving_quantile_unconstrained=float(bq.saving_pct.median()),
                              med_net_central_quantile=float(q.net_pct_central.median()),
                              med_net_high_quantile=float(q.net_pct_high.median()),
                              med_d_starts_quantile=float(q.d_starts.median()),
                              met_conformal=int(c_.meets.sum()), n_conformal=len(c_),
                              med_saving_conformal=float(c_.saving_pct.median()))
        hbrow = g[g.selection == "handbook"]
        sign_change = np.sign(d["b_xfit_0.001"]["med_saving_quantile"]) != np.sign(d["b_xfit_0.001"]["med_saving_quantile_unconstrained"])
        st_hi, ad_hi = d["b_xfit_0.001"]["med_net_high_quantile"], d["A_D"]["med_net_high_quantile"]
        d["handbook"] = dict(starts_median=float(hbrow.starts.median()), ue_kWh_max=float(hbrow.ue_kWh.max()))
        d["rule_i_saving_conditional"] = bool(sign_change)
        d["rule_ii_side_by_side"] = bool(st_hi <= 0 and ad_hi > 0)
        d["rule_ii_D121_stands"] = bool(st_hi > 0)
        r2[cname] = d
    rd["2_constraints"] = r2
    # ---- item 3: hold comparator
    ho = cells[cells.item == "hold"]
    adp = base[(base.selection == "A_D")]
    hc = ho[ho.policy.isin(CP)]
    rd["3_hold"] = dict(met_conformal=int(hc.meets.sum()), n_conformal=len(hc), med_saving_conformal=float(hc.saving_pct.median()),
                        ad_med_saving_conformal=float(adp[adp.policy.isin(CP)].saving_pct.median()),
                        met_quantile=int(ho[ho.policy.isin(QP)].meets.sum()),
                        med_saving_quantile=float(ho[ho.policy.isin(QP)].saving_pct.median()),
                        N_distribution={str(k): int(v) for k, v in ho.hold_N.value_counts().sort_index().items()},
                        N_not_found=int((~ho.hold_N_found.astype(bool)).sum()),
                        controller_to_supplement=bool(int(hc.meets.sum()) >= 46 and hc.saving_pct.median()
                                                      >= adp[adp.policy.isin(CP)].saving_pct.median()),
                        pid_trigger_satisfied=bool(int(hc.meets.sum()) < 42))
    # ---- item 4: event-count target
    ev = cells[cells.item == "events"]
    e1, e0 = ev[ev.selection == "A_R_events"], ev[ev.selection == "A_R"]
    rd["4_events"] = dict(met_conformal=int(e1[e1.policy.isin(CP)].meets.sum()), met_conformal_AR=int(e0[e0.policy.isin(CP)].meets.sum()),
                          met_quantile=int(e1[e1.policy.isin(QP)].meets.sum()), met_quantile_AR=int(e0[e0.policy.isin(QP)].meets.sum()),
                          events_per_year_median_conformal=float(e1[e1.policy.isin(CP)].events.median()),
                          events_per_year_median_conformal_AR=float(e0[e0.policy.isin(CP)].events.median()),
                          med_saving_conformal=float(e1[e1.policy.isin(CP)].saving_pct.median()),
                          med_saving_conformal_AR=float(e0[e0.policy.isin(CP)].saving_pct.median()),
                          med_saving_quantile=float(e1[e1.policy.isin(QP)].saving_pct.median()),
                          selections_not_found=int((~e1.met_on_selection.astype(bool)).sum()))
    # ---- item 5: battery
    bt = pd.concat([pd.read_csv(out / f"f{yr}_battery.csv") for yr in FOLDS], ignore_index=True)
    bt.to_csv(out / "battery.csv", index=False)
    rd["5_battery"] = dict(full_cycles_median_handbook=float(bt[bt.policy == "handbook"].full_cycles.median()),
                           full_cycles_median_quantile=float(bt[bt.policy.isin(QP)].full_cycles.median()),
                           soc_mean_median=float(bt.soc_mean.median()), share_at_floor_max=float(bt.share_at_floor.max()),
                           share_at_or_below_band_median=float(bt.share_at_or_below_band.median()))
    json.dump(rd, open(out / "reading.json", "w"), indent=1, default=float)
    check("spill_cases", len(sp) == 8 * 12 * 7 + 8 * 12, len(sp))
    check("hold_cases", len(ho) == 8 * 12)
    ok = all(c_["ok"] for c_ in checks.values())
    outs = {p.name: md5(p) for p in sorted(out.glob("*")) if p.is_file() and p.name not in ("checks.json",)}
    json.dump(dict(checks=checks, outputs=outs, code={"ce1_s64.py": md5(HERE / "ce1_s64.py")}), open(out / "checks.json", "w"), indent=1)
    allchecks = sum(len(json.load(open(out / f"f{yr}_checks.json"))["checks"]) for yr in FOLDS) + len(checks)
    env = dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__, pvlib=__import__("pvlib").__version__)
    json.dump(dict(stage="S64", ok=ok, error="", finished=time.strftime("%Y-%m-%dT%H:%M:%S"), host=platform.node() + "",
                   smoke=False, n_checks=allchecks, n_failed=sum(not c_["ok"] for c_ in checks.values()), env=env),
              open(R / "_runs" / "S64.json", "w"), indent=1)
    json.dump(dict(stage="S64", ok=ok, checks=checks, outputs=outs, env=env,
                   code={p: md5(HERE / p) for p in ("ce1_s64.py", "ce1_load.py", "ce1_s6.py", "ce1_adaptive.py", "ce1_wp57.py")}),
              open(R / "_runs" / "S64.detail.json", "w"), indent=1)
    log(f"{'ALL CHECKS PASS' if ok else 'CHECK FAILED'} ({allchecks} checks in total)")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--fold", type=int, default=None)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--analyse", action="store_true")
    a = ap.parse_args()
    st = selftest()
    if not all(st.values()):
        sys.exit(1)
    if a.selftest:
        sys.exit(0)
    if a.analyse:
        sys.exit(0 if analyse(Path(a.root)) else 1)
    sys.exit(0 if run_fold(Path(a.root), a.fold, a.smoke) else 1)


if __name__ == "__main__":
    main()
