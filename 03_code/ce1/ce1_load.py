"""Load-based dispatch model of the Yulara mini-grid.

1. Representative load: NREL ComStock End-Use Load Profiles, LargeHotel aggregate for
   Maricopa County AZ, TMY3 weather, moved from EST interval end to Phoenix local standard
   time interval start and shifted by 182 days to southern seasons.
2. Diesel fleet at the 5-min step: minimum feasible unit commitment (the set of least total
   rating that covers the requirement and keeps every unit inside its minimum up time), minimum
   loading, linear fuel curve, curtailment and unserved energy. A daytime step without measured
   GHI is not scored, while the commitment follows the handbook rule and the minimum-up clocks
   advance. The simulator of release 2.0.0 (greedy commitment, state held over missing GHI)
   stays available through the registered_v200 context manager.
3. Selection protocols and the frontier value at the reliability target.
Rules: D-080, D-084.

Usage (from the repository root unless the path starts at ce1/, then from 03_code):
    python ce1/ce1_load.py --stage S1L
    python ce1/ce1_load.py --selftest
"""
from __future__ import annotations

import argparse
import hashlib
import io
import itertools
import json
import platform
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
CODE = HERE.parent

DT = 5 / 60.0  # h per step
STEPS_PER_DAY = 288
UNITS_BASE = (2500.0, 1000.0, 1000.0, 1000.0, 1000.0)  # 2.5 MW base + four 1 MW sets
UNITS_EXTRA = (1150.0, 1150.0)
PEAK_EXTRA_KW = 6500.0
MIN_LOAD = 0.40  # handbook; sensitivity 0.10
MIN_LOAD_SENS = 0.10
FUEL = {"cat_c32": (0.0205, 0.230),  # L/h per kW rated, L/kWh (primary)
        "skarstein_uhlen": (0.08415, 0.246)}
MIN_UP = 12  # steps (1 h)
CO2_KG_PER_L = 2.68
BATT_P, BATT_E = 4000.0, 2000.0  # kW, kWh (YUO061-19A tender size)
ETA = 0.949
E_LO, E_HI, E_FLOOR = 0.10, 1.00, 0.50  # fractions of BATT_E
RESERVE_H = 0.5  # reserve credit: power sustainable 30 min
CREDIT_RULE = "A4"  # [D-088] reserve credit rule (A0 = stage S6 as run)
CREDIT_RULES = ("A0", "A1", "A2", "A4")
E0_FRAC = 0.50  # [D-084] initial battery energy
PV_SHARE = 0.15  # primary scale: PV = 15% of load energy
PEAK_SENS_KW = 6000.0
SHIFT_DAYS = 182
FLAT_WEIGHT = 0.5  # Shape B
UE_TARGET = 0.002 / 100  # NEM reliability standard (in force to 2028)
UE_TARGET_SENS = 0.003 / 100  # [D-084] RSSR 2026 recommendation from 2028
FUEL_PRICE = 1.68  # [D-084] AUD/L: 241.4/1.1 - 51.6 c/L
FUEL_PRICE_SENS = (1.20, 2.20)
VOLL = 34.39  # [D-084] AUD/kWh, AER 2024 VCR, commercial ($2024)
VOLL_SENS = (10.0, 100.0)
HEADLINE_H = 15

COMSTOCK_URL = ("https://oedi-data-lake.s3.amazonaws.com/nrel-pds-building-stock/"
                "end-use-load-profiles-for-us-building-stock/2021/comstock_tmy3_release_1/"
                "timeseries_aggregates/by_county/state=AZ/g0400130-largehotel.csv")
COMSTOCK_DOI = "10.25984/1876417"
FALLBACK_INDEX = ("https://openei.org/datasets/files/961/pub/COMMERCIAL_LOAD_DATA_E_PLUS_OUTPUT/"
                  "USA_AZ_Phoenix-Sky.Harbor.Intl.AP.722780_TMY3/")
FALLBACK_DOI = "10.25984/1788456"
EST_TO_PHOENIX_H = -2  # Phoenix MST = EST - 2 h (no DST in AZ)


def md5(p):
    return hashlib.md5(Path(p).read_bytes()).hexdigest()


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


# ================================================================ 1. load
def parse_comstock(text: str) -> np.ndarray:
    """ComStock aggregate CSV -> (365, 96) average kW per 15-min interval in Phoenix local
    standard time, interval START. Timestamps are EST interval END (ComStock FAQ).
    """
    d = pd.read_csv(io.StringIO(text))
    tcol = [c for c in d.columns if c.lower() in ("timestamp", "time", "timestamp_est")]
    ecol = [c for c in d.columns if c.startswith("out.electricity.total.energy_consumption")] or \
        [c for c in d.columns if "electricity" in c.lower() and "total" in c.lower()
         and "consumption" in c.lower() and "intensity" not in c.lower()]
    if not tcol or not ecol:
        raise ValueError(f"unexpected ComStock columns: {list(d.columns)[:12]}")
    ts = pd.to_datetime(d[tcol[0]])
    kwh = d[ecol[0]].astype(float).values
    if len(kwh) != 35040:
        raise ValueError(f"expected 35040 15-min rows, got {len(kwh)}")
    start_est = ts - pd.Timedelta(minutes=15)
    base = start_est.min()
    pos = ((start_est - base) // pd.Timedelta(minutes=15)).astype(int).values
    pos = (pos + EST_TO_PHOENIX_H * 4) % 35040
    if len(np.unique(pos)) != 35040:
        raise ValueError("timestamps do not cover the year exactly once")
    kw = np.empty(35040)
    kw[pos] = kwh * 4.0
    parse_comstock.columns = (tcol[0], ecol[0])
    return kw.reshape(365, 96)


def parse_doe_reference(text: str) -> np.ndarray:
    """DOE commercial reference building hourly CSV (EnergyPlus, TMY3 local standard time,
    hour-ending) -> (365, 96) kW, each hour held over its four 15-min intervals.
    """
    d = pd.read_csv(io.StringIO(text))
    col = [c for c in d.columns if c.startswith("Electricity:Facility")]
    if not col or len(d) != 8760:
        raise ValueError(f"unexpected reference file: {len(d)} rows, columns {list(d.columns)[:6]}")
    kw = d[col[0]].astype(float).values  # kW (hourly average, kWh per hour)
    return np.repeat(kw.reshape(365, 24), 4, axis=1)


def _get(url, timeout=120, tries=3):
    last = None
    for k in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "ce1-research/1.0"})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read()
        except Exception as e:  # noqa: BLE001  (network errors vary)
            last = e
            time.sleep(3 * (k + 1))
    raise RuntimeError(f"could not retrieve {url}: {last}")


def retrieve_profile(out: Path, offline: Path | None = None) -> dict:
    """Primary source first, fallback second. The raw bytes are kept beside the outputs."""
    if offline is not None:
        raw = Path(offline).read_bytes()
        prof = parse_comstock(raw.decode("utf-8"))
        src = dict(source="comstock_tmy3_maricopa_largehotel", url=COMSTOCK_URL, doi=COMSTOCK_DOI,
                   note=f"offline copy {Path(offline).name}", columns=parse_comstock.columns)
    else:
        try:
            raw = _get(COMSTOCK_URL)
            prof = parse_comstock(raw.decode("utf-8"))
            src = dict(source="comstock_tmy3_maricopa_largehotel", url=COMSTOCK_URL, doi=COMSTOCK_DOI,
                       columns=parse_comstock.columns)
        except Exception as e:  # noqa: BLE001
            log(f"primary source failed ({e}); trying the DOE reference fallback")
            idx = _get(FALLBACK_INDEX).decode("utf-8", "replace")
            names = sorted({n for n in __import__("re").findall(r'href="([^"]+\.csv)"', idx)
                            if "LargeHotel" in n})
            if not names:
                raise RuntimeError("fallback index lists no LargeHotel file") from e
            url = FALLBACK_INDEX + names[0]
            raw = _get(url)
            prof = parse_doe_reference(raw.decode("utf-8"))
            src = dict(source="doe_reference_largehotel_phoenix_tmy3", url=url, doi=FALLBACK_DOI,
                       primary_error=str(e))
    name = "load_source.csv"
    (out / name).write_bytes(raw)
    src.update(file=name, md5=hashlib.md5(raw).hexdigest(), bytes=len(raw),
               retrieved=time.strftime("%Y-%m-%dT%H:%M:%S"))
    return dict(profile=prof, source=src)


def unit_profile(prof96: np.ndarray) -> np.ndarray:
    """Normalize to an annual mean of 1 (the scales are mean load in kW)."""
    return prof96 / prof96.mean()


def au_day_noleap(index) -> np.ndarray:
    idx = pd.DatetimeIndex(index)
    d0 = idx.dayofyear.values - 1
    leap = idx.is_leap_year
    return np.where(leap & (d0 >= 59), d0 - 1, d0)


def load_on(index, unit96: np.ndarray, shape="A") -> np.ndarray:
    idx = pd.DatetimeIndex(index)
    us = (au_day_noleap(idx) - SHIFT_DAYS) % 365
    slot = (idx.hour.values * 60 + idx.minute.values) // 15
    a = unit96[us, slot]
    if shape == "A":
        return a
    if shape == "B":
        return FLAT_WEIGHT * 1.0 + (1 - FLAT_WEIGHT) * a
    raise ValueError(shape)


def scale_energy_share(unit_load, p_av, share=PV_SHARE) -> float:
    """Mean load (kW) such that available PV energy = share x load energy, summed over the
    steps where P_av is defined.
    """
    ok = ~np.isnan(p_av)
    return float(np.sum(p_av[ok]) / (share * np.sum(unit_load[ok])))


def scale_peak(unit_load, peak_kw=PEAK_SENS_KW) -> float:
    return float(peak_kw / np.max(unit_load))


def fleet_for_peak(peak_kw) -> tuple:
    return UNITS_BASE + (UNITS_EXTRA if peak_kw > PEAK_EXTRA_KW else ())


# ================================================================ 2. commitment and balance
class Fleet:

    def __init__(self, ratings=UNITS_BASE):
        self.r = np.asarray(ratings, float)
        self.classes = sorted(set(self.r.tolist()))
        self.members = [np.where(self.r == c)[0] for c in self.classes]
        rows = []
        for cnt in itertools.product(*[range(len(m) + 1) for m in self.members]):
            if sum(cnt) == 0:
                continue
            cap = sum(c * k for c, k in zip(self.classes, cnt))
            rows.append((cap, sum(cnt), cnt))
        rows.sort(key=lambda x: (x[0], x[1], x[2]))
        self.cap = np.array([x[0] for x in rows])
        self.nunits = np.array([x[1] for x in rows])
        self.counts = np.array([x[2] for x in rows])  # (n_combos, n_classes)
        self.smallest = float(self.r.min())
        # lookup table over (count vector, online mask, forced mask) built with choose,
        # so the step loop works on integer masks (same selection, far fewer array calls)
        U = len(self.r)
        self.U = U
        self.bits = (1 << np.arange(U)).astype(np.int64)
        nm = 1 << U
        masks = np.arange(nm)
        mbits = ((masks[:, None] >> np.arange(U)) & 1).astype(bool)
        self.cap_of = mbits @ self.r
        self.pop = mbits.sum(axis=1)
        ci, om, fm = np.meshgrid(np.arange(len(self.cap)), masks, masks, indexing="ij")
        ci, om, fm = ci.ravel(), om.ravel(), fm.ravel()
        fm = fm & om  # forced units are online
        new = self._choose_counts(self.counts[ci], mbits[om], mbits[fm])
        self.table = (new @ self.bits).reshape(len(self.cap), nm, nm).astype(np.int64)

    def choose_mask(self, kreq, on_m, forced_m):
        """Integer-mask form of choose: kreq (K), masks (K) -> new mask (K)."""
        ci = np.minimum(np.searchsorted(self.cap, kreq - 1e-9, side="left"), len(self.cap) - 1)
        return self.table[ci, on_m, forced_m]

    def choose(self, kreq, on, forced):
        """kreq (K), on/forced (K, U) bool -> new online set (K, U) bool. Within a rating
        class the units kept are those inside their minimum up time, then those already
        online, then the lowest index (identical units are interchangeable).
        """
        ci = np.searchsorted(self.cap, kreq - 1e-9, side="left")
        cnt = self.counts[np.minimum(ci, len(self.cap) - 1)]  # (K, n_classes)
        return self._choose_counts(cnt, on, forced)

    def _choose_counts(self, cnt, on, forced):
        new = forced.copy()
        onf = on & ~forced
        off = ~on
        for j, mem in enumerate(self.members):
            o, x = onf[:, mem], off[:, mem]
            need = cnt[:, j] - forced[:, mem].sum(axis=1)
            take = o & ((np.cumsum(o, axis=1) - o) < need[:, None])
            need = need - o.sum(axis=1)
            take |= x & ((np.cumsum(x, axis=1) - x) < need[:, None])
            new[:, mem] |= take
        return new


COMMITS = ("greedy", "minfeas")
GAPS = ("hold", "elapsed")
SIM_VERSION = "v2.1"
DEFAULT_COMMIT = "minfeas"
DEFAULT_GAP = "elapsed"
_VT = {}


class sim_mode:
    """Context manager: inside it, simulate's defaults are the given simulator."""

    def __init__(self, commit, gap, version):
        self.new = (commit, gap, version)

    def __enter__(self):
        global DEFAULT_COMMIT, DEFAULT_GAP, SIM_VERSION
        self._old = (DEFAULT_COMMIT, DEFAULT_GAP, SIM_VERSION)
        DEFAULT_COMMIT, DEFAULT_GAP, SIM_VERSION = self.new
        return self

    def __exit__(self, *exc):
        global DEFAULT_COMMIT, DEFAULT_GAP, SIM_VERSION
        DEFAULT_COMMIT, DEFAULT_GAP, SIM_VERSION = self._old
        return False


def registered_v200():
    """Inside it, simulate's defaults are the registered v2.0.0 simulator (greedy, hold)."""
    return sim_mode("greedy", "hold", "v2.0.0")


def primary_v21():
    """Inside it, simulate's defaults are the v2.1 primary simulator (minfeas, elapsed)."""
    return sim_mode("minfeas", "elapsed", "v2.1")


def sim_defaults() -> dict:
    return dict(version=SIM_VERSION, commit=DEFAULT_COMMIT, gap=DEFAULT_GAP)


class VariantTables:
    """Lookup tables for the minimum feasible commitment and the internal N-1 k_min of one fleet."""

    def __init__(self, ratings):
        r = np.asarray(ratings, float)
        U = len(r)
        M = 1 << U
        masks = np.arange(M)
        mb = ((masks[:, None] >> np.arange(U)) & 1).astype(bool)
        self.cap = mb @ r
        lg = np.where(mb, r[None, :], 0.0).max(axis=1)
        self.capn1 = np.where(masks > 0, self.cap - lg, -np.inf)
        pop = mb.sum(axis=1)
        self.caps_u = np.unique(self.cap[1:])
        self.n1_u = np.concatenate([[-np.inf], np.unique(self.capn1[1:])])
        nci, ncj = len(self.caps_u), len(self.n1_u)
        full = M - 1
        T = np.full((nci + 1, ncj + 1, M, M), full, dtype=np.uint8 if M <= 256 else np.int64)
        popc = np.array([bin(x).count("1") for x in range(M)])
        for o in range(M):
            starts = popc[masks & ~o]
            order = np.lexsort((masks, starts, pop, self.cap))  # cap, units, starts, index
            f = o
            while True:  # every forced subset f of o
                cand = order[(order & f) == f]
                cand = cand[cand > 0]
                fe = (self.cap[cand][None, None, :] >= self.caps_u[:, None, None] - 1e-9) & \
                     (self.capn1[cand][None, None, :] >= self.n1_u[None, :, None] - 1e-9)
                anyf = fe.any(axis=2)
                first = cand[np.argmax(fe, axis=2)]
                T[:nci, :ncj, o, f] = np.where(anyf, first, full)
                if f == 0:
                    break
                f = (f - 1) & o
        self.T = T
        # greedy N-1 k_min over the count vectors of Fleet (the order ce1_wp56.combo_table uses)
        fl = Fleet(ratings)
        cls = np.asarray(fl.classes, float)
        clg = np.array([cls[np.flatnonzero(c)].max() for c in fl.counts])
        self.ccap = fl.cap.astype(float)
        self.ccapn1 = (fl.cap - clg).astype(float)

    def minfeas(self, kreq, need, on_m, forced_m):
        ci = np.searchsorted(self.caps_u, kreq - 1e-9, side="left")
        cj = np.zeros(len(kreq), np.int64) if need is None else np.searchsorted(self.n1_u, need - 1e-9, side="left")
        return self.T[ci, cj, on_m, forced_m].astype(np.int64)

    def kmin_greedy(self, k1, k2):
        """As ce1_wp56.k_min / ce1_s69.kmin_n1a: least count-vector capacity with cap >= k1 and cap - largest >= k2,
        else the whole fleet.
        """
        ok = (self.ccap[None, :] >= k1[:, None] - 1e-9) & (self.ccapn1[None, :] >= k2[:, None] - 1e-9)
        j = np.argmax(ok, axis=1)
        return np.where(ok.any(axis=1), self.ccap[j], self.ccap[-1])


def variant_tables(ratings):
    key = tuple(float(x) for x in ratings)
    if key not in _VT:
        _VT[key] = VariantTables(key)
    return _VT[key]


DAY_KEYS = ("fuel_L", "ue_kWh", "curt_kWh", "dump_kWh", "thermal_kWh", "pv_load_kWh", "chg_kWh",
            "dis_kWh", "pav_kWh", "load_kWh", "starts", "ue_steps", "online_kW_h", "fallback_steps",
            "scored_steps")


def _credit(rule, E_issue, cring, e_lo, e_fl, Pb, h_steps):
    """Battery reserve credit (kW) for the target step, from issue-time information only.
    cring holds the credits of the h steps s-h..s-1 (all fixed at or before issue time).
    """
    if rule == "A0":  # D-084 rule 5 as run in stage S6
        usable, T = np.maximum(E_issue - e_lo, 0.0), RESERVE_H
    elif rule == "A1":  # D-087 candidate: spread over lead + 30 min
        usable, T = np.maximum(E_issue - e_lo, 0.0), RESERVE_H + h_steps * DT
    elif rule == "A2":  # D-087 candidate: protected band only
        usable, T = np.maximum(np.minimum(E_issue, e_fl) - e_lo, 0.0), RESERVE_H
    else:  # A4 [D-088]
        usable = np.maximum(np.minimum(E_issue, e_fl) - e_lo - cring.sum(axis=0) * (DT / ETA), 0.0)
        T = RESERVE_H
    return np.minimum(Pb, usable * (ETA / T))


def simulate(L, P_av, P_cnt, day_id, h_steps, ratings=UNITS_BASE, min_load=MIN_LOAD,
             fuel="cat_c32", battery=False, batt_p=BATT_P, batt_e=BATT_E, record=False,
             credit=CREDIT_RULE, credit_cap=None, commit=None, gap=None, n1=None, p_hat=None):
    """Load-based dispatch of K policies along a contiguous 5-min grid of target steps.

    L (n) or (K, n) load kW; P_av (n) available PV kW (NaN = not scored; with gap="hold" the state is held, with gap="elapsed" the
    commitment follows the handbook rule and the minimum-up clocks advance);
    P_cnt (K, n) counted PV for the target step, issued h steps earlier (NaN = no forecast
    row, handbook fallback); min_load scalar or (K); day_id (n) int from 0, non-decreasing;
    h_steps lead time in steps. Rows of a batch may differ in policy, level, load shape and
    scale, and minimum load; the fleet, fuel curve and battery are shared by the batch.
    credit_cap (n) optional upper limit of the battery reserve credit per target step, known
    at issue time (WP-61 sensitivity: the PV measured at issue time, so the battery counts only
    as reserve for PV and never in place of diesel capacity for load). None: unchanged (D-088).
    commit, gap: None takes the module defaults (v2.1: "minfeas", "elapsed"; see registered_v200 for v2.0.0).
    n1: None, "n1a" or "s56", an N-1 construction applied inside simulate from the counted PV; p_hat is the
    forecast that "s56" needs.
    Returns totals, per-day arrays (K, n_days) and, if record, per-step arrays (K, n).
    """
    fl = Fleet(ratings)
    commit = DEFAULT_COMMIT if commit is None else commit  # v2.1 defaults (module constants, D-161)
    gap = DEFAULT_GAP if gap is None else gap
    assert commit in COMMITS and gap in GAPS and n1 in (None, "n1a", "s56"), (commit, gap, n1)
    assert not (n1 and battery), "the N-1 constructions run without battery"
    variant = commit != "greedy" or gap != "hold" or n1 is not None
    vt = variant_tables(fl.r) if (commit == "minfeas" or n1) else None
    phz = np.nan_to_num(np.asarray(p_hat, float)) if n1 == "s56" else None
    a_f, b_f = FUEL[fuel]
    P_cnt = np.atleast_2d(np.asarray(P_cnt, float))
    K, n = P_cnt.shape
    L = np.asarray(L, float)
    L2 = L if L.ndim == 2 else None
    mload = np.broadcast_to(np.asarray(min_load, float), (K,)).copy()
    U = len(fl.r)
    nd = int(day_id.max()) + 1
    Pb = batt_p if battery else 0.0
    Eb = batt_e if battery else 0.0
    e_lo, e_hi, e_fl = E_LO * Eb, E_HI * Eb, E_FLOOR * Eb
    E = np.full(K, E0_FRAC * Eb)
    e_init = E.copy()
    ring = np.tile(E, (h_steps + 1, 1))
    assert credit in CREDIT_RULES, credit
    cring = np.zeros((max(h_steps, 1), K))  # [D-088] credits of steps s-h..s-1
    on = np.zeros((K, U), bool)
    on_m = np.zeros(K, np.int64)
    bitsU = ((np.arange(1 << U)[:, None] >> np.arange(U)) & 1).astype(bool)
    up = np.zeros((K, U), np.int16)
    nk = len(DAY_KEYS)
    day = np.zeros((nk, K, nd))
    acc = np.zeros((nk, K))
    (a_fuel, a_ue, a_curt, a_dump, a_th, a_pvl, a_chg, a_dis, a_pav, a_load, a_st, a_ues, a_onl,
     a_fb, a_sc) = acc
    run_steps = np.zeros((K, U))
    missing = 0
    zK = np.zeros(K)
    rec = {k: np.full((K, n), np.nan) for k in ("D", "KU", "MU", "ue", "curt", "dis", "chg",
                                                  "E", "pv_load", "dump")} if record else None
    rec_on = np.zeros((K, n, U), bool) if record else None
    rec_up = np.zeros((K, n, U), np.int16) if record else None
    cur_day = int(day_id[0]) if n else 0

    def _commit_step(Ls, pcz, rb, s):
        """Simulator variants: online mask for step s from counted PV pcz (NaN already 0) and battery credit rb."""
        Lk = np.broadcast_to(np.asarray(Ls, float), (K,))
        k1 = np.maximum(Lk - pcz - rb, fl.smallest)
        need = None
        if n1 == "n1a":
            need = Lk - pcz
        elif n1 == "s56":
            need = Lk - phz[s]
        forced = ((up < MIN_UP) & on) @ fl.bits
        if commit == "minfeas":
            return vt.minfeas(k1, need, on_m, forced)
        if need is not None:  # greedy N-1 as registered: request k_min
            km = vt.kmin_greedy(k1, need)
            k1 = np.maximum(Lk - (Lk - km), fl.smallest)
        return fl.choose_mask(k1, on_m, forced)

    for s in range(n):
        ring[s % (h_steps + 1)] = E  # energy at the start of step s
        d = int(day_id[s])
        if d != cur_day:
            day[:, :, cur_day] = acc
            acc[:] = 0.0
            cur_day = d
        pav = P_av[s]
        if pav != pav:  # NaN: not scored, state held
            missing += 1
            cring[s % cring.shape[0]] = 0.0  # no discharge, no credit
            if gap == "elapsed":
                Ls = L2[:, s] if L2 is not None else L[s]
                new_m = _commit_step(Ls, np.zeros(K), 0.0, s)
                new = bitsU[new_m]
                up = np.where(new, np.where(on, np.minimum(up + 1, MIN_UP), 1), 0).astype(np.int16)
                on, on_m = new, new_m
                if record:  # v2.1: the unit state of a gap step is recorded
                    rec_on[:, s, :] = on  # (the numeric records stay NaN: not scored)
                    rec_up[:, s, :] = up
            continue
        Ls = L2[:, s] if L2 is not None else L[s]
        pc = P_cnt[:, s]
        fb = pc != pc
        if fb.any():
            pc = np.where(fb, 0.0, pc)
            if pav > 0:
                a_fb += fb
        if battery:
            E_issue = ring[(s - h_steps) % (h_steps + 1)] if s >= h_steps else e_init
            rb = _credit(credit, E_issue, cring, e_lo, e_fl, Pb, h_steps)
            if credit_cap is not None:  # credit limited per target step
                rb = np.minimum(rb, credit_cap[s])
            cring[s % cring.shape[0]] = rb
            kreq = np.maximum(Ls - pc - rb, fl.smallest)
        else:
            rb = 0.0
            kreq = np.maximum(Ls - pc, fl.smallest)
        if variant:
            new_m = _commit_step(Ls, pc, rb, s)
        else:
            forced_m = ((up < MIN_UP) & on) @ fl.bits
            new_m = fl.choose_mask(kreq, on_m, forced_m)
        a_st += fl.pop[new_m & ~on_m]
        new = bitsU[new_m]
        up = np.where(new, np.where(on, np.minimum(up + 1, MIN_UP), 1), 0).astype(np.int16)
        on, on_m = new, new_m
        run_steps += on
        KU = fl.cap_of[new_m]
        MU = mload * KU
        net = Ls - pav
        c1 = net >= MU
        if c1.all():  # common daytime and night case
            D = np.minimum(net, KU)
            S = np.maximum(net - KU, 0.0)
            if battery:
                dis1 = np.minimum(np.minimum(S, Pb), np.maximum(E - e_lo, 0.0) * (ETA / DT))
                room = np.maximum(E - dis1 * (DT / ETA) - e_fl, 0.0) * (ETA / DT)
                x = np.minimum(np.minimum(np.maximum(D - MU, 0.0), Pb - dis1), room)
                D = D - x
                dis = dis1 + x
                ue = S - dis1
                E = E - dis * (DT / ETA)
            else:
                dis = zK
                ue = S
            pv_load, chg, curt, dump = pav, zK, zK, zK
        else:
            D1 = np.minimum(net, KU)
            S = np.maximum(net - KU, 0.0)
            if battery:
                dis1 = np.minimum(np.minimum(S, Pb), np.maximum(E - e_lo, 0.0) * (ETA / DT))
                room = np.maximum(E - dis1 * (DT / ETA) - e_fl, 0.0) * (ETA / DT)
                x = np.minimum(np.minimum(np.maximum(D1 - MU, 0.0), Pb - dis1), room)
            else:
                dis1 = x = zK
            pv_load2 = np.minimum(np.maximum(Ls - MU, 0.0), pav)
            sur = pav - pv_load2
            chg2 = np.minimum(np.minimum(sur, Pb), np.maximum(e_hi - E, 0.0) / (ETA * DT)) if battery else zK
            D = np.where(c1, D1 - x, MU)
            dis = np.where(c1, dis1 + x, 0.0)
            ue = np.where(c1, S - dis1, 0.0)
            pv_load = np.where(c1, pav, pv_load2)
            chg = np.where(c1, 0.0, chg2)
            curt = np.where(c1, 0.0, sur - chg2)
            dump = np.where(c1, 0.0, np.maximum(MU - Ls, 0.0))
            if battery:
                E = E - dis * (DT / ETA) + chg * (ETA * DT)
            a_curt += curt
            a_dump += dump
            a_chg += chg
        a_fuel += a_f * KU + b_f * D
        a_ue += ue
        a_ues += ue > 1e-9
        a_th += D
        a_pvl += pv_load
        a_dis += dis
        a_pav += pav
        a_load += Ls
        a_onl += KU
        a_sc += 1
        if record:
            for k, v in (("D", D), ("KU", KU), ("MU", MU), ("ue", ue), ("curt", curt), ("dis", dis),
                         ("chg", chg), ("E", E), ("pv_load", pv_load), ("dump", dump)):
                rec[k][:, s] = v
            rec_on[:, s, :] = on
            rec_up[:, s, :] = up
    if n:
        day[:, :, cur_day] = acc
    daily = {}
    for i, k in enumerate(DAY_KEYS):
        v = day[i]
        daily[k] = v if k in ("starts", "ue_steps", "fallback_steps", "scored_steps") else v * DT
    tot = {k: v.sum(axis=1) for k, v in daily.items()}
    tot["co2_t"] = tot["fuel_L"] * CO2_KG_PER_L / 1000.0
    tot["ue_share"] = np.divide(tot["ue_kWh"], tot["load_kWh"], out=np.zeros(K), where=tot["load_kWh"] > 0)
    tot["curt_share"] = np.divide(tot["curt_kWh"], tot["pav_kWh"], out=np.zeros(K), where=tot["pav_kWh"] > 0)
    tot["pv_share_of_load"] = np.divide(tot["pv_load_kWh"] + tot["dis_kWh"], tot["load_kWh"],
                                        out=np.zeros(K), where=tot["load_kWh"] > 0)
    out = dict(total=tot, daily=daily, run_h=run_steps * DT, missing_steps=missing, ratings=fl.r, E_end=E)
    if record:
        out.update(rec=rec, rec_on=rec_on, rec_up=rec_up)
    return out


def simulate_reference(L, P_av, P_cnt_1, idx, h_steps, ratings=UNITS_BASE, min_load=MIN_LOAD,
                       fuel="cat_c32", battery=False, batt_p=BATT_P, batt_e=BATT_E, credit=CREDIT_RULE):
    r = list(map(float, ratings))
    U = len(r)
    a_f, b_f = FUEL[fuel]
    Pb, Eb = (batt_p, batt_e) if battery else (0.0, 0.0)
    E = E0_FRAC * Eb
    ts = pd.DatetimeIndex(idx)
    val = {t: i for i, t in enumerate(ts)}
    grid = pd.date_range(ts[0], ts[-1], freq="5min")
    hist = {}
    cred = {}  # [D-088] credit issued per grid step
    on = [False] * U
    up = [0] * U
    small = min(r)
    classes = sorted(set(r))
    members = {c: [i for i in range(U) if r[i] == c] for c in classes}
    combos = []
    for cnt in itertools.product(*[range(len(members[c]) + 1) for c in classes]):
        if sum(cnt):
            combos.append((sum(c * k for c, k in zip(classes, cnt)), sum(cnt), cnt))
    combos.sort()
    out = {}
    for g_i, t in enumerate(grid):
        hist[g_i] = E
        i = val.get(t)
        if i is None or np.isnan(P_av[i]):
            continue
        E_issue = hist.get(g_i - h_steps, E0_FRAC * Eb) if g_i >= h_steps else E0_FRAC * Eb
        pc = P_cnt_1[i]
        pc = 0.0 if np.isnan(pc) else pc
        rb = 0.0
        if battery:
            e_lo_, e_fl_ = E_LO * Eb, E_FLOOR * Eb
            if credit == "A0":
                rb = min(Pb, max(E_issue - e_lo_, 0.0) * ETA / RESERVE_H)
            elif credit == "A1":
                rb = min(Pb, max(E_issue - e_lo_, 0.0) * ETA / (RESERVE_H + h_steps * DT))
            elif credit == "A2":
                rb = min(Pb, max(min(E_issue, e_fl_) - e_lo_, 0.0) * ETA / RESERVE_H)
            else:
                committed = sum(cred.get(g_i - j, 0.0) for j in range(1, max(h_steps, 1) + 1)) * DT / ETA
                rb = min(Pb, max(min(E_issue, e_fl_) - e_lo_ - committed, 0.0) * ETA / RESERVE_H)
        cred[g_i] = rb
        kreq = max(L[i] - pc - rb, small)
        cnt = next((c[2] for c in combos if c[0] >= kreq - 1e-9), combos[-1][2])
        forced = [on[u] and up[u] < MIN_UP for u in range(U)]
        new = [False] * U
        for j, c in enumerate(classes):
            mem = members[c]
            pref = sorted(mem, key=lambda u: (-(forced[u] * 2 + on[u]), u))
            for u in pref[:cnt[j]]:
                new[u] = True
        new = [new[u] or forced[u] for u in range(U)]
        starts = sum(1 for u in range(U) if new[u] and not on[u])
        up = [(up[u] + 1 if on[u] else 1) if new[u] else 0 for u in range(U)]
        on = new
        KU = sum(r[u] for u in range(U) if on[u])
        MU = min_load * KU
        net = L[i] - P_av[i]
        dis = chg = ue = curt = 0.0
        if net >= MU:
            D = min(net, KU)
            S = max(net - KU, 0.0)
            if battery:
                dis = min(S, Pb, max(E - E_LO * Eb, 0.0) * ETA / DT)
                room = max(E - dis * DT / ETA - E_FLOOR * Eb, 0.0) * ETA / DT
                x = min(max(D - MU, 0.0), Pb - dis, room)
                ue = S - dis
                D -= x
                dis += x
            else:
                ue = S
        else:
            D = MU
            pvl = min(max(L[i] - MU, 0.0), P_av[i])
            sur = P_av[i] - pvl
            if battery:
                chg = min(sur, Pb, max(E_HI * Eb - E, 0.0) / (ETA * DT))
            curt = sur - chg
        E = E - dis * DT / ETA + chg * ETA * DT
        key = t.normalize()
        o = out.setdefault(key, dict(fuel_L=0.0, ue_kWh=0.0, curt_kWh=0.0, starts=0))
        o["fuel_L"] += (a_f * KU + b_f * D) * DT
        o["ue_kWh"] += ue * DT
        o["curt_kWh"] += curt * DT
        o["starts"] += starts
    return out


# ================================================================ grid helpers
def to_grid(index, *series, start=None, end=None):
    """Place time-ordered series on the contiguous 5-min calendar grid (NaN where absent)."""
    idx = pd.DatetimeIndex(index)
    g = pd.date_range(start or idx[0].normalize(), end or (idx[-1].normalize() + pd.Timedelta(hours=23, minutes=55)),
                      freq="5min")
    out = [pd.Series(np.asarray(s, float), index=idx).reindex(g).values for s in series]
    return g, out


def day_ids(grid) -> np.ndarray:
    d = pd.DatetimeIndex(grid).normalize()
    return np.unique(d.values, return_inverse=True)[1]


def available_pv_on_grid(grid, ghi, m) -> np.ndarray:
    """Available PV (kW) on the grid from the measured GHI record and the S1M map. Night
    steps (sun below the horizon) are 0 even where GHI is missing; daytime NaN stays NaN.
    """
    import pvlib
    import ce1_pvmap as PM
    g = pd.DatetimeIndex(grid)
    pv = PM.pv_available_from_ghi(np.asarray(ghi, float), g, m)
    t = g.tz_localize(PM.SITE["tz"])
    zen = pvlib.solarposition.get_solarposition(t, PM.SITE["lat"], PM.SITE["lon"], PM.SITE["alt"])
    night = np.asarray(zen["apparent_zenith"]) >= 90.0
    return np.where(night, 0.0, pv)


def conservative_param(policy, grid):
    """The level that counts the least PV: highest theta, or f = 1 for frac_pv."""
    return float(np.max(grid))


def select_AR(tot_H2: dict, params, policy, target=UE_TARGET):
    """Least fuel among levels meeting the UE target on H2, else the most conservative."""
    ue, fuel = tot_H2["ue_share"], tot_H2["fuel_L"]
    ok = ue <= target + 1e-15
    if ok.any():
        i = np.flatnonzero(ok)[np.argmin(fuel[ok])]
        return float(params[i]), True
    return conservative_param(policy, params), False


def select_AC(tot_H2: dict, params, price=FUEL_PRICE, voll=VOLL):
    cost = price * tot_H2["fuel_L"] + voll * tot_H2["ue_kWh"]
    return float(params[int(np.argmin(cost))])


def frontier(tot: dict, params) -> list:
    return [dict(param=float(p), fuel_L=float(tot["fuel_L"][i]), ue_share=float(tot["ue_share"][i]),
                 ue_kWh=float(tot["ue_kWh"][i]), curt_share=float(tot["curt_share"][i]),
                 starts=float(tot["starts"][i])) for i, p in enumerate(params)]


def fuel_at_target(tot: dict, target=UE_TARGET) -> float:
    """Fuel interpolated on the fuel-versus-UE frontier at the target UE share (DR2): the
    lower envelope of the points, linear between the two points bracketing the target.
    NaN when the target is outside the range the levels reach.
    """
    ue, f = np.asarray(tot["ue_share"]), np.asarray(tot["fuel_L"])
    o = np.argsort(ue, kind="stable")
    ue, f = ue[o], np.minimum.accumulate(f[o])  # least fuel at UE <= x
    if target < ue[0] or target > ue[-1]:
        return float("nan")
    return float(np.interp(target, ue, f))


def run_selection(L_H2, Pav_H2, C_H2, day_H2, L_te, Pav_te, C_te, day_te, params, policy, h_steps,
                  **kw):
    """Protocol A-R, A-C and the test-tuned reference for one policy. C_*: (K, n) counted PV
    for every level in params (scores fitted on H1 for H2, on the calibration year for the
    test year). Returns the chosen levels, test outcomes at each, and both frontiers.
    """
    sH = simulate(L_H2, Pav_H2, C_H2, day_H2, h_steps, **kw)["total"]
    sT = simulate(L_te, Pav_te, C_te, day_te, h_steps, **kw)
    tT = sT["total"]
    pAR, met = select_AR(sH, params, policy)
    pAC = select_AC(sH, params)
    iAR, iAC = list(params).index(pAR), list(params).index(pAC)
    ref_AR, _ = select_AR(tT, params, policy)
    pick = lambda i: {k: float(v[i]) for k, v in tT.items()}  # noqa: E731
    return dict(policy=policy, A_R=dict(param=pAR, target_met_on_H2=met, test=pick(iAR)),
                A_C=dict(param=pAC, test=pick(iAC)),
                reference_R=dict(param=ref_AR, test=pick(list(params).index(ref_AR))),
                reference_C=dict(param=select_AC(tT, params)),
                frontier_H2=frontier(sH, params), frontier_test=frontier(tT, params),
                fuel_at_target_test=fuel_at_target(tT), daily_test=sT["daily"])


# ================================================================ self-tests
def _toy(seed=0, n_days=6, gaps=True, pv_scale=1600.0):
    rng = np.random.default_rng(seed)
    grid = pd.date_range("2023-06-01", periods=n_days * 288, freq="5min")
    hod = (grid.hour + grid.minute / 60).values
    cs = np.clip(np.sin(np.pi * (hod - 6.5) / 11.0), 0, None)
    cloud = np.clip(0.75 + 0.35 * rng.standard_normal(len(grid)), 0.05, 1.0)
    pav = pv_scale * cs * cloud
    L = 2600 + 900 * np.sin(np.pi * (hod - 8) / 14) ** 2 + 150 * rng.standard_normal(len(grid))
    L = np.clip(L, 1500, None)
    if gaps:
        day_mask = cs > 0
        pav[(rng.random(len(grid)) < 0.03) & day_mask] = np.nan
    return grid, L, pav, cs


def _counted(pav, cs, rng, K=5):
    base = np.nan_to_num(pav)
    C = np.vstack([np.clip(base * f + 200 * cs * rng.standard_normal(len(pav)), 0, None)
                   for f in np.linspace(0.3, 1.0, K)])
    C[:, rng.random(len(pav)) < 0.02] = np.nan  # issue-time gaps
    return C


def _selftest_registered(verbose=True) -> dict:
    """The registered self-tests (v2.0.0 simulator; run inside registered_v200 by selftest)."""
    res = {}

    def ok(name, cond, msg=""):
        res[name] = bool(cond)
        if verbose:
            print(f"  [{'PASS' if cond else 'FAIL'}] {name} {msg}")

    rng = np.random.default_rng(7)
    grid, L, pav, cs = _toy(1)
    dd = day_ids(grid)
    h = 3
    C = _counted(pav, cs, rng)
    handbook = np.zeros((1, len(grid)))
    oracle = np.nan_to_num(pav)[None, :]
    C_all = np.vstack([C, handbook, oracle])
    for batt in (False, True):
        r = simulate(L, pav, C_all, dd, h, battery=batt, record=True)
        R = r["rec"]
        sc = ~np.isnan(pav)
        bal = R["D"] + R["pv_load"] + R["dis"] + R["ue"] - R["dump"] - L[None, :]
        tag = "batt" if batt else "nobatt"
        ok(f"1_energy_balance_every_step_{tag}", np.nanmax(np.abs(bal[:, sc])) < 1e-9,
           f"(max {np.nanmax(np.abs(bal[:, sc])):.1e} kW)")
        on = r["rec_on"]
        per_unit_min = np.where(on, (R["D"] / np.where(R["KU"] > 0, R["KU"], 1))[:, :, None], np.inf)
        ok(f"2_min_load_and_one_unit_online_{tag}",
           (np.nanmin(np.where(np.isnan(R["D"]), np.inf, R["D"] - R["MU"])) >= -1e-9)
           and on[:, sc, :].any(axis=2).all() and np.isfinite(per_unit_min).any())
        pv_ok = (R["pv_load"][:, sc] <= pav[sc][None, :] + 1e-9).all() and (R["curt"][:, sc] >= -1e-12).all() \
            and (R["pv_load"][:, sc] + R["chg"][:, sc] + R["curt"][:, sc]
                 - pav[sc][None, :] <= 1e-9).all()
        ok(f"8_curtailment_nonneg_pv_used_le_available_{tag}", pv_ok)
        # 6. no unit switched off inside its minimum up time
        viol = 0
        for k in range(on.shape[0]):
            for u in range(on.shape[2]):
                x = on[k, sc, u].astype(int)
                runs = np.diff(np.r_[0, x, 0])
                st, en = np.flatnonzero(runs == 1), np.flatnonzero(runs == -1)
                ln = en - st
                viol += int(((ln < MIN_UP) & (en < len(x))).sum())
        ok(f"6_min_up_time_respected_{tag}", viol == 0, f"({viol} short runs)")
        # 9. fuel identity
        a_f, b_f = FUEL["cat_c32"]
        lit = (a_f * R["KU"] + b_f * R["D"]) * DT
        ok(f"9_fuel_identity_{tag}", abs(np.nansum(lit, axis=1) - r["total"]["fuel_L"]).max() < 1e-6)
    # 3. handbook rule, no battery, fleet above peak -> no UE
    r0 = simulate(L, pav, handbook, dd, h)
    ok("3_handbook_no_UE", r0["total"]["ue_kWh"][0] == 0.0 and L.max() < sum(UNITS_BASE),
       f"(peak {L.max():.0f} kW vs fleet {sum(UNITS_BASE):.0f} kW)")
    # 4. oracle: no UE, fuel at most the handbook rule's
    r1 = simulate(L, pav, np.vstack([handbook, oracle]), dd, h)
    ok("4_oracle_no_UE_and_fuel_le_handbook", r1["total"]["ue_kWh"][1] == 0.0
       and r1["total"]["fuel_L"][1] <= r1["total"]["fuel_L"][0] + 1e-9,
       f"(fuel {r1['total']['fuel_L'][1]:.0f} vs {r1['total']['fuel_L'][0]:.0f} L)")
    # 5. chosen set = brute force (no carry-over), both fleets
    good = True
    for ratings in (UNITS_BASE, UNITS_BASE + UNITS_EXTRA):
        fl = Fleet(ratings)
        U = len(ratings)
        subsets = [np.array(b, bool) for b in itertools.product([0, 1], repeat=U) if any(b)]
        for kreq in np.r_[np.linspace(0, sum(ratings), 173), [1000, 2000, 2500, 3500, 3499.99, 3500.01]]:
            kq = max(kreq, fl.smallest)
            on0 = rng.random(U) < 0.5
            new = fl.choose(np.array([kq]), on0[None, :], np.zeros((1, U), bool))[0]
            feas = [(np.dot(b, ratings), b.sum()) for b in subsets if np.dot(b, ratings) >= kq - 1e-9]
            best = min(feas) if feas else (sum(ratings), U)
            good &= (abs(np.dot(new, ratings) - best[0]) < 1e-9 and new.sum() == best[1])
    ok("5_online_set_equals_brute_force", good)
    # 7. zero-power battery = no battery
    rb0 = simulate(L, pav, C_all, dd, h, battery=True, batt_p=0.0)
    rnb = simulate(L, pav, C_all, dd, h, battery=False)
    ok("7_zero_power_battery_equals_no_battery",
       all(np.array_equal(rb0["daily"][k], rnb["daily"][k]) for k in rnb["daily"]))
    g2, L2, pav2, cs2 = _toy(3, n_days=5)
    keep = np.random.default_rng(5).random(len(g2)) > 0.04
    C2 = _counted(pav2, cs2, np.random.default_rng(9), K=3)
    idx_k = g2[keep]
    same = True
    for batt, rule in ((False, CREDIT_RULE),) + tuple((True, r_) for r_ in CREDIT_RULES):
        gg, (Lg, Pg) = to_grid(idx_k, L2[keep], pav2[keep])
        Cg = np.vstack([to_grid(idx_k, C2[k, keep])[1][0] for k in range(3)])
        Lg = np.where(np.isnan(Lg), 0.0, Lg)
        rg = simulate(Lg, Pg, Cg, day_ids(gg), 6, battery=batt, credit=rule)
        for k in range(3):
            ref = simulate_reference(L2[keep], pav2[keep], C2[k, keep], idx_k, 6, battery=batt, credit=rule)
            days = sorted(ref)
            for j, dname in enumerate(days):
                same &= abs(ref[dname]["fuel_L"] - rg["daily"]["fuel_L"][k, j]) < 1e-7
                same &= abs(ref[dname]["ue_kWh"] - rg["daily"]["ue_kWh"][k, j]) < 1e-7
                same &= abs(ref[dname]["curt_kWh"] - rg["daily"]["curt_kWh"][k, j]) < 1e-7
                same &= ref[dname]["starts"] == rg["daily"]["starts"][k, j]
    ok("10_calendar_equals_plain_loop_reference_gappy", same)
    r = simulate(L, pav, C_all, dd, h, record=True)
    R, on_, up_ = r["rec"], r["rec_on"], r["rec_up"]
    night = pav == 0  # NaN (missing daytime) is not night
    same_state = (on_ == on_[:1]).all(axis=(0, 2)) & (up_ == up_[:1]).all(axis=(0, 2))
    agree = np.ones(len(grid), bool)
    for k in ("D", "KU", "curt", "ue"):
        agree &= np.isclose(R[k], R[k][:1]).all(axis=0)
    viol, merged, nights = 0, 0, 0
    s_ = 0
    while s_ < len(grid):
        if not night[s_]:
            s_ += 1
            continue
        e_ = s_
        while e_ < len(grid) and night[e_]:
            e_ += 1
        nights += 1
        hit = np.flatnonzero(same_state[s_:e_])
        if len(hit):
            merged += 1
            f0 = s_ + hit[0]
            viol += int((~agree[f0:e_]).sum())
        s_ = e_
    ok("11_night_outcomes_identical_once_states_merge", viol == 0 and merged == nights,
       f"({merged}/{nights} nights merge, {viol} disagreeing steps after merging)")
    # 12. [D-088] battery reserve credit is deliverable: handbook rule + battery, fleet above
    #     peak load, zero UE at every lead (5, 15, 30, 60 min), on an evening-peak toy with
    #     gaps where the battery is drained through the lead window. Power check: rule A0
    #     (stage S6 as run, D-087 Finding A) produces UE on the same toy at 60 min.
    g3, L3, pav3, cs3 = _toy(11, n_days=8, pv_scale=900.0)
    hod3 = (g3.hour + g3.minute / 60).values
    L3 = L3 + 1800.0 * np.exp(-0.5 * ((hod3 - 20.0) / 1.5) ** 2)  # evening peak
    dd3 = day_ids(g3)
    hb = np.zeros((1, len(g3)))
    ue4 = {hh: simulate(L3, pav3, hb, dd3, hh, battery=True, credit="A4")["total"]["ue_kWh"][0]
           for hh in (1, 3, 6, 12)}
    ue0 = simulate(L3, pav3, hb, dd3, 12, battery=True, credit="A0")["total"]["ue_kWh"][0]
    ok("12_handbook_battery_no_UE_every_lead_A4", L3.max() < sum(UNITS_BASE)
       and all(v == 0.0 for v in ue4.values()) and ue0 > 0,
       f"(A4 UE {[float(round(v, 3)) for v in ue4.values()]} kWh at 5/15/30/60 min; "
       f"A0 {float(ue0):.1f} kWh at 60 min, peak {L3.max():.0f} kW)")
    # 13. [D-088] A4 never credits more than A2 (the protected band), and A2 never more
    #     than A0; the credit is zero once the band is exhausted
    Eb = BATT_E
    Ei = np.linspace(0, Eb, 41)
    cr = np.abs(np.random.default_rng(2).normal(300, 200, (6, 41)))
    c0, c2, c4 = (_credit(r_, Ei, cr, E_LO * Eb, E_FLOOR * Eb, BATT_P, 6) for r_ in ("A0", "A2", "A4"))
    ok("13_credit_ordering_A4_le_A2_le_A0", (c4 <= c2 + 1e-12).all() and (c2 <= c0 + 1e-12).all()
       and (c4[Ei <= E_LO * Eb] == 0).all())
    # extra: load construction
    prof = np.tile(np.r_[np.full(24, 0.6), np.full(48, 1.2), np.full(24, 0.9)], (365, 1))
    u = unit_profile(prof)
    idx = pd.date_range("2024-01-01", "2024-12-31 23:55", freq="5min")
    la = load_on(idx, u, "A")
    ok("x_shift_maps_US_jan1_to_AU_jul2",
       (au_day_noleap(pd.DatetimeIndex(["2023-07-02"]))[0] - SHIFT_DAYS) % 365 == 0
       and (au_day_noleap(pd.DatetimeIndex(["2024-07-02"]))[0] - SHIFT_DAYS) % 365 == 0)
    lb = load_on(idx, u, "B")
    ok("x_shapeB_same_energy_half_variation", abs(lb.mean() - la.mean()) < 1e-9
       and abs(np.std(lb) - 0.5 * np.std(la)) < 1e-9)
    m15 = la.reshape(-1, 3)
    ok("x_15min_held_over_three_5min_steps", (m15 == m15[:, :1]).all())
    # ComStock parser: EST interval-end to Phoenix interval-start
    ts = pd.date_range("2018-01-01 00:15", periods=35040, freq="15min")
    hr_est_start = ((ts - pd.Timedelta(minutes=15)).hour.values)
    kwh = np.where(hr_est_start == 7, 10.0, 1.0)  # 07:00-08:00 EST = 05:00 MST
    txt = pd.DataFrame({"timestamp": ts.strftime("%Y-%m-%d %H:%M:%S"),
                        "out.electricity.total.energy_consumption": kwh}).to_csv(index=False)
    p = parse_comstock(txt)
    ok("x_comstock_est_end_to_phoenix_start", np.allclose(p[:, 20:24], 40.0) and np.allclose(p[:, 24:], 4.0)
       and np.allclose(p[:, :20], 4.0))
    # selection helpers
    tot = dict(ue_share=np.array([1e-3, 1e-5, 0.0, 0.0]), fuel_L=np.array([10.0, 11.0, 12.0, 13.0]),
               ue_kWh=np.array([5.0, 0.05, 0, 0]))
    pAR, met = select_AR(tot, np.array([0.8, 0.85, 0.9, 0.95]), "cqr")
    ok("x_select_AR_least_fuel_meeting_target", pAR == 0.85 and met)
    pAR2, met2 = select_AR(dict(tot, ue_share=np.full(4, 1e-3)), np.array([0.8, 0.85, 0.9, 0.95]), "cqr")
    ok("x_select_AR_conservative_when_none_meets", pAR2 == 0.95 and not met2)
    ok("x_fuel_at_target_interpolates", abs(fuel_at_target(tot, 1e-5) - 11.0) < 1e-9)
    return res


def _selftest_v21(verbose=True) -> dict:
    """v2.1 simulator (minimum feasible commitment, elapsed clocks), with and without the battery."""
    res = {}

    def ok(name, cond, msg=""):
        res[name] = bool(cond)
        if verbose:
            print(f"  [{'PASS' if cond else 'FAIL'}] {name} {msg}")

    with primary_v21():
        return _selftest_v21_body(ok, res)


def _selftest_v21_body(ok, res):
    rng = np.random.default_rng(7)
    grid, L, pav, cs = _toy(1)
    dd = day_ids(grid)
    h = 3
    C = _counted(pav, cs, rng)
    handbook = np.zeros((1, len(grid)))
    oracle = np.nan_to_num(pav)[None, :]
    C_all = np.vstack([C, handbook, oracle])
    sc = ~np.isnan(pav)
    for batt in (False, True):
        tag = "batt" if batt else "nobatt"
        for rule in ((CREDIT_RULE,) if not batt else CREDIT_RULES):
            r = simulate(L, pav, C_all, dd, h, battery=batt, record=True, credit=rule)
            R, on = r["rec"], r["rec_on"]
            t2 = f"{tag}_{rule}" if batt else tag
            bal = R["D"] + R["pv_load"] + R["dis"] + R["ue"] - R["dump"] - L[None, :]
            ok(f"v21_1_energy_balance_every_step_{t2}", np.nanmax(np.abs(bal[:, sc])) < 1e-9)
            ok(f"v21_2_min_load_and_one_unit_online_{t2}",
               np.nanmin(np.where(np.isnan(R["D"]), np.inf, R["D"] - R["MU"])) >= -1e-9 and on[:, sc, :].any(axis=2).all())
            ok(f"v21_8_curtailment_nonneg_pv_used_le_available_{t2}",
               (R["pv_load"][:, sc] <= pav[sc][None, :] + 1e-9).all() and (R["curt"][:, sc] >= -1e-12).all()
               and (R["pv_load"][:, sc] + R["chg"][:, sc] + R["curt"][:, sc] - pav[sc][None, :] <= 1e-9).all())
            # 6 in calendar time: gap steps carry the recorded unit state, so a run is measured in elapsed steps
            viol = 0
            for k in range(on.shape[0]):
                for u in range(on.shape[2]):
                    x = on[k, :, u].astype(int)
                    runs = np.diff(np.r_[0, x, 0])
                    st, en = np.flatnonzero(runs == 1), np.flatnonzero(runs == -1)
                    viol += int((((en - st) < MIN_UP) & (en < len(x))).sum())
            ok(f"v21_6_min_up_time_respected_in_elapsed_time_{t2}", viol == 0, f"({viol} short runs)")
            a_f, b_f = FUEL["cat_c32"]
            lit = (a_f * R["KU"] + b_f * R["D"]) * DT
            ok(f"v21_9_fuel_identity_{t2}", abs(np.nansum(lit, axis=1) - r["total"]["fuel_L"]).max() < 1e-6)
            if batt:
                E = R["E"][:, sc]
                ok(f"v21_battery_energy_within_bounds_{t2}", (E >= E_LO * BATT_E - 1e-6).all() and (E <= E_HI * BATT_E + 1e-6).all())
                ok(f"v21_battery_no_simultaneous_charge_discharge_{t2}",
                   not ((R["chg"][:, sc] > 1e-9) & (R["dis"][:, sc] > 1e-9)).any())
    # 3, 4: handbook and oracle
    r0 = simulate(L, pav, handbook, dd, h)
    ok("v21_3_handbook_no_UE", r0["total"]["ue_kWh"][0] == 0.0)
    r1 = simulate(L, pav, np.vstack([handbook, oracle]), dd, h)
    ok("v21_4_oracle_no_UE_and_fuel_le_handbook", r1["total"]["ue_kWh"][1] == 0.0
       and r1["total"]["fuel_L"][1] <= r1["total"]["fuel_L"][0] + 1e-9)
    # 7: zero-power battery = no battery
    rb0 = simulate(L, pav, C_all, dd, h, battery=True, batt_p=0.0)
    rnb = simulate(L, pav, C_all, dd, h, battery=False)
    ok("v21_7_zero_power_battery_equals_no_battery", all(np.array_equal(rb0["daily"][k], rnb["daily"][k]) for k in rnb["daily"]))
    # 11 (invariant part): once every policy is in the same unit state at a night step, outcomes coincide until
    #    daylight. Merging within each night is a property of the registered tie-break and is reported, not required.
    r = simulate(L, pav, C_all, dd, h, record=True)
    R, on_, up_ = r["rec"], r["rec_on"], r["rec_up"]
    night = pav == 0
    same_state = (on_ == on_[:1]).all(axis=(0, 2)) & (up_ == up_[:1]).all(axis=(0, 2))
    agree = np.ones(len(grid), bool)
    for k in ("D", "KU", "curt", "ue"):
        agree &= np.isclose(R[k], R[k][:1]).all(axis=0)
    viol, merged, nights, s_ = 0, 0, 0, 0
    while s_ < len(grid):
        if not night[s_]:
            s_ += 1
            continue
        e_ = s_
        while e_ < len(grid) and night[e_]:
            e_ += 1
        nights += 1
        hit = np.flatnonzero(same_state[s_:e_])
        if len(hit):
            merged += 1
            viol += int((~agree[s_ + hit[0]:e_]).sum())
        s_ = e_
    ok("v21_11_night_outcomes_identical_once_states_merge", viol == 0 and merged >= 1,
       f"({merged}/{nights} nights merge, {viol} disagreeing steps after merging)")
    # 12: battery reserve credit deliverable under the v2.1 commitment (handbook rule + battery, A4, every lead)
    g3, L3, pav3, cs3 = _toy(11, n_days=8, pv_scale=900.0)
    hod3 = (g3.hour + g3.minute / 60).values
    L3 = L3 + 1800.0 * np.exp(-0.5 * ((hod3 - 20.0) / 1.5) ** 2)
    dd3 = day_ids(g3)
    hb = np.zeros((1, len(g3)))
    ue4 = {hh: simulate(L3, pav3, hb, dd3, hh, battery=True, credit="A4")["total"]["ue_kWh"][0] for hh in (1, 3, 6, 12)}
    ok("v21_12_handbook_battery_no_UE_every_lead_A4", all(v == 0.0 for v in ue4.values()),
       f"(A4 UE {[float(round(v, 3)) for v in ue4.values()]} kWh at 5/15/30/60 min)")
    # greedy commitment with held state, passed explicitly, reproduces the release 2.0.0 run exactly:
    with registered_v200():
        ref = simulate(L, pav, C_all, dd, h, battery=True)
    exp = simulate(L, pav, C_all, dd, h, battery=True, commit="greedy", gap="hold")
    ok("v21_explicit_greedy_hold_equals_registered", all(np.array_equal(ref["daily"][k], exp["daily"][k]) for k in ref["daily"]))
    return res


def selftest(verbose=True) -> dict:
    """Registered self-tests under the v2.0.0 simulator, then the v2.1 self-tests under the primary defaults."""
    with registered_v200():
        res = _selftest_registered(verbose)
    res.update(_selftest_v21(verbose))
    return res


# ================================================================ stage S1L
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="S1L")
    ap.add_argument("--out", default=None)
    ap.add_argument("--offline", default=None, help="local copy of the ComStock CSV (tests)")
    ap.add_argument("--raw-dir", default=None)
    ap.add_argument("--map", default=None)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        r = selftest()
        sys.exit(0 if all(r.values()) else 1)
    sys.path.insert(0, str(CODE / "utils"))
    sys.path.insert(0, str(HERE))
    import config as C  # noqa: E402
    import ce1_pvmap as PM  # noqa: E402
    t0 = time.time()
    out = Path(a.out) if a.out else C.RESULTS / "ce1" / a.stage
    runs = (out / "_runs") if a.out else C.RESULTS / "ce1" / "_runs"
    out.mkdir(parents=True, exist_ok=True); runs.mkdir(parents=True, exist_ok=True)
    checks = {}

    def check(name, cond, msg=""):
        checks[name] = dict(ok=bool(cond), msg=msg)
        log(f"{'PASS' if cond else 'FAIL'}  {name}  {msg}")

    st = selftest(verbose=False)
    check("selftests", all(st.values()), f"({sum(st.values())}/{len(st)}"
          + ("" if all(st.values()) else "; failed " + ", ".join(k for k, v in st.items() if not v)) + ")")
    got = retrieve_profile(out, Path(a.offline) if a.offline else None)
    prof, src = got["profile"], got["source"]
    log(f"profile from {src['source']} ({src['bytes']} bytes, md5 {src['md5']})")
    check("profile_complete_positive", prof.shape == (365, 96) and np.isfinite(prof).all()
          and (prof > 0).all(), f"(shape {prof.shape}, min {prof.min():.1f} kW, max {prof.max():.1f} kW)")
    daily_mean = prof.mean(axis=0)
    hmin = int(np.argmin(daily_mean)) / 4.0
    hmax = int(np.argmax(daily_mean)) / 4.0
    check("profile_local_time_plausible", 0.0 <= hmin < 7.0,
          f"(mean-day minimum at {hmin:.2f} h, maximum at {hmax:.2f} h, Phoenix local standard time)")
    u = unit_profile(prof)
    np.savez(out / "load_profile.npz", unit96=u, raw_kw96=prof, shift_days=SHIFT_DAYS,
             flat_weight=FLAT_WEIGHT)
    # plant data 2023-2024 on the grid
    m = PM.load_map(a.map)
    d = PM.load_years((2023, 2024), raw_dir=a.raw_dir)
    scales, plaus = {}, {}
    res_year = {}
    for y in (2023, 2024):
        dy = d[d.index.year == y]
        g, (ghi, tot) = to_grid(dy.index, dy["ghi"].values, dy["TOTAL"].values,
                                start=pd.Timestamp(f"{y}-01-01"), end=pd.Timestamp(f"{y}-12-31 23:55"))
        pav = available_pv_on_grid(g, ghi, m)
        res_year[y] = dict(grid=g, pav=pav, total=tot, day=day_ids(g))
    g23 = res_year[2023]
    ulA = load_on(g23["grid"], u, "A")
    scales["A"] = dict(primary=scale_energy_share(ulA, g23["pav"]), peak6MW=scale_peak(ulA))
    ulB = load_on(g23["grid"], u, "B")
    dev = abs(ulB.mean() - ulA.mean()) / ulA.mean()
    check("shape_B_has_shape_A_energy", dev < 0.005,
          f"(B = 0.5 A + 0.5 x mean at every scale of A [D-084]; 2023 mean differs by {dev:.2e})")
    # measured 2023 spill of the site total against the map (the plant-grounded target)
    ok23 = ~np.isnan(g23["pav"]) & ~np.isnan(g23["total"])
    spill23 = float(1 - np.clip(g23["total"][ok23], 0, None).sum() / g23["pav"][ok23].sum())

    def curt_share(mean_kw, shp, y=2023):
        ry = res_year[y]
        L = mean_kw * load_on(ry["grid"], u, shp)
        r = simulate(L, ry["pav"], np.zeros((1, len(L))), ry["day"], 3,
                     ratings=fleet_for_peak(L.max()))
        return float(r["total"]["curt_share"][0]), r

    for shp in ("A",):
        lo, hi = 0.02 * scales[shp]["primary"], 5.0 * scales[shp]["primary"]
        clo, chi = curt_share(lo, shp)[0], curt_share(hi, shp)[0]
        bracketed = (clo - spill23) * (chi - spill23) < 0
        c_lo0, c_hi0 = clo, chi
        if bracketed:
            for _ in range(30):
                mid = 0.5 * (lo + hi)
                cm = curt_share(mid, shp)[0]
                if (cm - spill23) * (clo - spill23) > 0:
                    lo, clo = mid, cm
                else:
                    hi = mid
            scales[shp]["spill_matched"] = 0.5 * (lo + hi)
        else:
            scales[shp]["spill_matched"] = float("nan")
        check(f"spill_matched_scale_found_{shp}", bracketed,
              f"(2023 measured spill {spill23:.4f}; handbook curtailment {c_lo0:.3f} at 0.02x and "
              f"{c_hi0:.3f} at 5x the primary scale; matched mean {scales[shp]['spill_matched']:.0f} kW)")
    scales["B"] = dict(scales["A"])  # Shape B inherits A's scales
    # plausibility of the primary scale (never a tuning target on the test year)
    for shp in ("A", "B"):
        for y in (2023, 2024):
            cs_, r = curt_share(scales[shp]["primary"], shp, y)
            L = scales[shp]["primary"] * load_on(res_year[y]["grid"], u, shp)
            plaus[f"{shp}_{y}"] = dict(handbook_curtailment_share=round(cs_, 4),
                                       handbook_ue_kWh=float(r["total"]["ue_kWh"][0]),
                                       handbook_fuel_L=round(float(r["total"]["fuel_L"][0]), 0),
                                       peak_kW=round(float(L.max()), 1), mean_kW=round(float(L.mean()), 1),
                                       load_factor=round(float(L.mean() / L.max()), 4),
                                       energy_MWh=round(float(L.sum() * DT / 1000), 1),
                                       missing_daytime_steps=int(r["missing_steps"]),
                                       extra_units=bool(L.max() > PEAK_EXTRA_KW))
    check("handbook_rule_serves_all_load_primary",
          all(v["handbook_ue_kWh"] == 0.0 for v in plaus.values()),
          "(fleet above peak under the primary scale, both shapes, 2023 and 2024)")
    table = {}
    for shp in ("A", "B"):
        table[shp] = {}
        for k, mk in scales[shp].items():
            L = mk * load_on(g23["grid"], u, shp)
            table[shp][k] = dict(mean_kW=round(mk, 1), peak_kW=round(float(L.max()), 1),
                                 load_factor=round(float(L.mean() / L.max()), 4),
                                 energy_MWh_2023=round(float(L.sum() * DT / 1000), 1),
                                 extra_units_UNVERIFIED=bool(L.max() > PEAK_EXTRA_KW))
    check("pv_is_about_30pct_of_peak_primary_A",
          0.10 <= m["cap"] * m["rated_total"] / table["A"]["primary"]["peak_kW"] <= 0.60,
          f"(map cap {m['cap'] * m['rated_total']:.0f} kW / peak {table['A']['primary']['peak_kW']:.0f} kW; "
          f"design statement about 30%)")
    out_json = dict(source=src, conversion=dict(timestamps="EST interval end -> Phoenix MST interval start",
                                                shift_days=SHIFT_DAYS, feb29="uses 28 February"),
                    profile_minimum_hour=hmin, profile_maximum_hour=hmax,
                    measured_spill_2023=round(spill23, 4), scales=table,
                    plausibility_handbook=plaus, arena_spill=PM.ARENA_SPILL,
                    pav_energy_MWh={str(y): round(float(np.nansum(res_year[y]["pav"]) * DT / 1000), 1)
                                    for y in (2023, 2024)},
                    daytime_missing_steps={str(y): int(np.isnan(res_year[y]["pav"]).sum()) for y in (2023, 2024)},
                    parameters=dict(units=UNITS_BASE, min_load=MIN_LOAD, fuel=FUEL, min_up_steps=MIN_UP,
                                    battery=(BATT_P, BATT_E), ue_target=UE_TARGET, fuel_price=FUEL_PRICE,
                                    voll=VOLL))
    json.dump(out_json, open(out / "load_scales.json", "w"), indent=1)
    ok = all(c["ok"] for c in checks.values())
    detail = dict(stage=a.stage, smoke=False, ok=ok, seconds=round(time.time() - t0, 1), checks=checks,
                  outputs={p.name: md5(p) for p in sorted(out.glob("*")) if p.is_file()},
                  code={p.name: md5(p) for p in (Path(__file__), HERE / "ce1_pvmap.py", HERE / "ce1_sensor.py",
                                                  HERE / "ce1_causal.py")},
                  versions=dict(python=platform.python_version(), numpy=np.__version__,
                                pandas=pd.__version__), host=platform.node())
    json.dump(detail, open(runs / f"{a.stage}.detail.json", "w"), indent=1)
    print(json.dumps(dict(scales=table, plausibility=plaus, spill23=spill23), indent=1))
    log(f"{'ALL CHECKS PASSED' if ok else 'SOME CHECKS FAILED'} in {time.time() - t0:.0f}s")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
