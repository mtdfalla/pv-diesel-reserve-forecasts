"""Stage S7D: decision-value contrasts (rules DR1 to DR6, D-091) from the stage S6, S6B and S6P
outputs, with paired day-block bootstrap intervals (B = 10,000, seed 42). Nothing is
re-simulated.

Usage (from the repository root unless the path starts at ce1/, then from 03_code):
    python3 03_code/ce1/ce1_decision.py [--B 10000]
    python3 03_code/ce1/ce1_decision.py --selftest
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

B_DEF = 10000
SEED = 42
FOLDS = [2019, 2020, 2021, 2022, 2023, 2024]
HORIZONS = [1, 3, 6, 12]
H_PRIMARY = 3
UE_TARGET = 0.002 / 100  # equals ce1_load.UE_TARGET (asserted in main)
CP = ["icp", "mondrian", "cqr", "mondrian_cqr"]
POLICIES = CP + ["raw_q", "tod_q", "nicp_var", "frac_pv"]
CONTRASTS = [("best_cp", "raw_q", "DR2"), ("cqr", "raw_q", "DR2-part"), ("mondrian_cqr", "raw_q", "DR2-part"),
             ("mondrian_cqr", "cqr", "DR3"), ("mondrian", "icp", "DR3"), ("mondrian", "nicp_var", "DR3")]
SHAPE_PAIRS = [("main", "battery"), ("shapeB", "batteryB")]
UNDEF_MAX = 0.025
FOLD_MIN = 4
THETA_LIMIT, THETA_TOP = 1.0, 0.999


def md5(p: Path) -> str:
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def log(m):
    print(f"[S7D] {m}", flush=True)


# ---------------------------------------------------------------- frontier value
def frontier_F(ue, fuel, target=UE_TARGET):
    """F for every column of ue, fuel (n_levels, m). Returns (F, undefined mask).
    Envelope: points sorted by UE, least fuel at UE <= x. Linear between the two points
    bracketing the target (np.interp on the envelope), flat beyond the largest UE.
    """
    ue = np.asarray(ue, float); fuel = np.asarray(fuel, float)
    if ue.ndim == 1:
        ue, fuel = ue[:, None], fuel[:, None]
    o = np.argsort(ue, axis=0, kind="stable")
    u = np.take_along_axis(ue, o, 0)
    e = np.minimum.accumulate(np.take_along_axis(fuel, o, 0), axis=0)
    n, m = u.shape
    j = (u <= target).sum(axis=0)  # points meeting the target
    F = np.full(m, np.nan)
    und = j == 0
    allm = j == n
    F[allm] = e[-1, allm]
    mid = ~und & ~allm
    if mid.any():
        c = np.flatnonzero(mid); jj = j[mid]
        u0, u1 = u[jj - 1, c], u[jj, c]
        e0, e1 = e[jj - 1, c], e[jj, c]
        w = np.where(u1 > u0, (target - u0) / np.where(u1 > u0, u1 - u0, 1.0), 0.0)
        F[c] = e0 + w * (e1 - e0)
    return F, und


def bracket_flags(ue, fuel, params, target=UE_TARGET):
    """flags on one frontier: (bracket is {0.999, 1}, F sits on the limit level)."""
    ue, fuel, params = map(lambda x: np.asarray(x, float), (ue, fuel, params))
    o = np.argsort(ue, kind="stable")
    u, f, p = ue[o], fuel[o], params[o]
    meet = u <= target
    if not meet.any():
        return False, False
    lo = np.flatnonzero(meet)[np.argmin(f[meet])]  # active point achieving the envelope below
    env_lo = f[lo]
    up = None
    above = np.flatnonzero(~meet)
    if len(above):
        k = above[0]
        env_k = min(env_lo, f[k])
        if f[k] < env_lo and env_k == f[k]:
            up = k
    pair = {round(p[lo], 4)} | ({round(p[up], 4)} if up is not None else set())
    return pair == {THETA_TOP, THETA_LIMIT}, (up is None and round(p[lo], 4) == THETA_LIMIT)


# ---------------------------------------------------------------- bootstrap
def draw_counts(n_days, B, seed=SEED):
    """(B, n_days) counts of each day in each replicate (the day-block draws)."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n_days, size=(B, n_days))
    C = np.zeros((B, n_days), dtype=np.float64)
    np.add.at(C, (np.repeat(np.arange(B), n_days), idx.ravel()), 1.0)
    return C


def ci(x, alpha=0.05):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    if not len(x):
        return np.nan, np.nan
    return float(np.percentile(x, 100 * alpha / 2)), float(np.percentile(x, 100 * (1 - alpha / 2)))


class Job:
    """One fold, horizon and scenario: daily arrays, point and replicate totals."""

    def __init__(self, z, sel, scen, C, T):
        """Point estimates from the float64 test totals T (exactly what S6 selected on);
        replicates from the float32 daily arrays (their sums match T to about 1e-7).
        """
        self.pol = np.asarray(z["rows_policy"]).astype(str)
        self.par = np.asarray(z["rows_param"], float)
        self.d = {k: np.asarray(z[f"{scen}__{k}"], np.float64) for k in ("fuel_L", "ue_kWh", "load_kWh")}
        self.sel = sel["selections"][scen]
        self.pd_ = {k: v.sum(axis=1) for k, v in self.d.items()}  # daily-sum point (check only)
        self.pt = {k: T[k].to_numpy(float) for k in ("fuel_L", "ue_kWh", "load_kWh", "ue_share")}
        self.bt = {k: v @ C.T for k, v in self.d.items()}  # (K, B)
        self.bt["ue_share"] = self.bt["ue_kWh"] / self.bt["load_kWh"]

    def rows(self, pol):
        return np.flatnonzero(self.pol == pol)

    def row_of(self, pol, param):
        ix = self.rows(pol)
        if np.isnan(param):
            return int(ix[0])
        k = ix[np.isclose(self.par[ix], param, atol=1e-9)]
        assert len(k) == 1, (pol, param)
        return int(k[0])

    def F(self, pol):
        ix = self.rows(pol)
        Fp, _ = frontier_F(self.pt["ue_share"][ix], self.pt["fuel_L"][ix])
        Fb, und = frontier_F(self.bt["ue_share"][ix], self.bt["fuel_L"][ix])
        fl = bracket_flags(self.pt["ue_share"][ix], self.pt["fuel_L"][ix], self.par[ix])
        return float(Fp[0]), Fb, und, fl


# ---------------------------------------------------------------- rules
def dr4_pass(levels):
    """levels: {(shape_pair_index, h): (lvl_nobatt, met_n, lvl_batt, met_b)}. D-091 DR4."""
    strict = False
    for s in (0, 1):
        ok = 0
        for h in HORIZONS:
            ln, mn, lb, mb = levels[(s, h)]
            if mn and mb and lb >= ln - 1e-12:
                ok += 1
                strict |= lb > ln + 1e-12
        if ok < 3:
            return False
    return strict


def kendall_tau(a, b):
    """Kendall tau-a between two score vectors on the same items (no ties expected)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    n = len(a); s = 0
    for i in range(n):
        for j in range(i + 1, n):
            s += np.sign(a[i] - a[j]) * np.sign(b[i] - b[j])
    return float(s / (n * (n - 1) / 2))


# ---------------------------------------------------------------- self-tests
def selftest(verbose=True):
    import ce1_load as LM
    res = {}

    def ok(name, cond, msg=""):
        res[name] = bool(cond)
        if verbose:
            print(f"  {'ok  ' if cond else 'FAIL'} {name} {msg}")

    rng = np.random.default_rng(1)
    agree, n_fin = True, 0
    for _ in range(2000):
        n = rng.integers(3, 30)
        ue = np.abs(rng.normal(0, 3e-5, n)) * (rng.random(n) < 0.8)
        fu = rng.uniform(1e6, 2e6, n)
        ref = LM.fuel_at_target({"ue_share": ue, "fuel_L": fu})
        F, und = frontier_F(ue, fu)
        if np.isfinite(ref):
            n_fin += 1
            agree &= abs(F[0] - ref) <= 1e-9 * abs(ref)
    ok("F_equals_ce1_load_where_finite", agree and n_fin > 500, f"({n_fin} finite cases)")
    F, und = frontier_F(np.array([0, 1e-6, 5e-6]), np.array([3.0, 2.0, 2.5]))
    ok("F_flat_extension_all_meet", F[0] == 2.0 and not und[0])
    F, und = frontier_F(np.array([3e-5, 4e-5]), np.array([1.0, 0.5]))
    ok("F_undefined_when_none_meets", und[0] and np.isnan(F[0]))
    d = rng.uniform(0, 10, (7, 20)); C = draw_counts(20, 50, seed=3)
    idx = np.random.default_rng(3).integers(0, 20, size=(50, 20))
    ok("counts_matmul_equals_explicit_sum", np.allclose(d @ C.T, d[:, idx].sum(axis=2)))
    ok("counts_rows_sum_to_n_days", np.all(C.sum(axis=1) == 20))
    p = np.array([0.99, 0.995, 0.999, 1.0]); u = np.array([5e-5, 3e-5, 1e-5, 0.0]); f = np.array([1.0, 1.1, 1.2, 1.5])
    ok("opp020_flag_bracket_0999_1", bracket_flags(np.array([5e-5, 4e-5, 3e-5, 0.0]), f, p) == (True, False))
    ok("opp020_no_flag_interior", bracket_flags(u, f, p) == (False, False))
    lv = {(s, h): (0.99, True, 0.995, True) for s in (0, 1) for h in HORIZONS}
    ok("dr4_pass_all_higher", dr4_pass(lv))
    lv2 = dict(lv); lv2.update({(0, 1): (0.99, True, 0.995, False), (0, 3): (0.99, True, 0.995, False)})
    ok("dr4_fallback_counts_not_satisfied", not dr4_pass(lv2))
    lv3 = {(s, h): (0.99, True, 0.99, True) for s in (0, 1) for h in HORIZONS}
    ok("dr4_needs_one_strict", not dr4_pass(lv3))
    ok("kendall_known", kendall_tau([1, 2, 3, 4], [1, 2, 3, 4]) == 1.0 and kendall_tau([1, 2, 3], [3, 2, 1]) == -1.0)
    return res


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--B", type=int, default=B_DEF)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--root", default=str(ROOT))
    ap.add_argument("--allow-unverified", action="store_true")
    # [D-092] the extended record: --s6 S6E --s6b S6E --s6p S6P2 --out S7E (defaults: as run for S7D)
    ap.add_argument("--s6", default="S6")
    ap.add_argument("--s6b", default="S6B")
    ap.add_argument("--s6p", default="S6P")
    ap.add_argument("--out", default="S7D")
    a = ap.parse_args()
    if a.selftest:
        r = selftest()
        sys.exit(0 if all(r.values()) else 1)
    import ce1_load as LM
    assert LM.UE_TARGET == UE_TARGET
    t0 = time.time()
    R = Path(a.root) / "04_results" / "ce1"
    out, runs = R / a.out, R / "_runs"
    out.mkdir(parents=True, exist_ok=True)
    checks, inputs = {}, {}

    def check(name, cond, msg=""):
        checks[name] = dict(ok=bool(cond), msg=msg)
        log(f"{'PASS' if cond else 'FAIL'}  {name}  {msg}")

    st = selftest(verbose=False)
    check("selftests", all(st.values()), f"({sum(st.values())}/{len(st)})")
    for s in dict.fromkeys((a.s6, a.s6b, a.s6p)):
        stp = runs / f"{s}.json"
        v = stp.exists() and json.load(open(stp)).get("ok", False)
        check(f"input_stage_{s}_verified", v or a.allow_unverified)

    ident, Frows, Crows, D1, D6, lv4, D5 = [], [], [], [], [], {}, []
    level_sets, hb_ue_main, f_mismatch, undef_max = set(), 0.0, [], 0.0
    for yr in FOLDS:
        for h in HORIZONS:
            srcs = [(a.s6, ("main", "shapeB")), (a.s6b, ("battery", "batteryB"))]
            jobs = {}
            for st_, scens in srcs:
                stem = R / st_ / f"f{yr}_h{h}"
                zf, sf, tf = (Path(f"{stem}_{x}") for x in ("daily.npz", "select.json", "totals.parquet"))
                for p in (zf, sf, tf):
                    inputs[f"{st_}/{p.name}"] = md5(p)
                z, sel, tot = np.load(zf), json.load(open(sf)), pd.read_parquet(tf)
                n_days = z[f"{scens[0]}__fuel_L"].shape[1]
                C = draw_counts(n_days, a.B)
                for sc in scens:
                    T = tot[(tot.scenario == sc) & (tot.set == "test")]
                    J = Job(z, sel, sc, C, T)
                    assert list(T.policy.astype(str)) == list(J.pol)
                    assert np.allclose(T.param.to_numpy(float), J.par, equal_nan=True)
                    worst = 0.0
                    for k in ("fuel_L", "ue_kWh", "load_kWh"):
                        ref = T[k].to_numpy(float)
                        worst = max(worst, float(np.max(np.abs(J.pd_[k] - ref) / np.maximum(np.abs(ref), 1.0))))
                    ident.append(dict(fold=yr, h=h, scenario=sc, max_rel_diff=worst))
                    level_sets.add(tuple(np.round(np.nan_to_num(J.par, nan=-1), 4)))
                    if sc == "main":
                        hb_ue_main = max(hb_ue_main, float(J.pt["ue_kWh"][J.rows("handbook")[0]]))
                    jobs[sc] = J
            for sc, J in jobs.items():
                Fv = {}
                for pol in POLICIES:
                    Fp, Fb, und, fl = J.F(pol)
                    rec = J.sel["policies"][pol]["fuel_at_target_test"]
                    if rec is not None and np.isfinite(rec) and abs(Fp - rec) > 1e-12 * abs(rec):
                        f_mismatch.append((yr, h, sc, pol, Fp, rec))
                    lo, hi = ci(Fb)
                    Frows.append(dict(fold=yr, h=h, scenario=sc, policy=pol, F=Fp, ci_lo=lo, ci_hi=hi,
                                      undefined_share=float(und.mean()), recorded=rec,
                                      flag_0999_1=fl[0], flag_at_limit=fl[1],
                                      F_H2=J.sel["policies"][pol]["fuel_at_target_H2"]))
                    Fv[pol] = (Fp, Fb, und)
                h2 = {p: J.sel["policies"][p]["fuel_at_target_H2"] for p in ("cqr", "mondrian_cqr")}
                h2v = {p: (np.inf if v is None or not np.isfinite(v) else v) for p, v in h2.items()}
                best = "mondrian_cqr" if h2v["mondrian_cqr"] < h2v["cqr"] else "cqr"
                Fv["best_cp"] = Fv[best]
                for aa, bb, rule in CONTRASTS:
                    d = Fv[aa][1] - Fv[bb][1]
                    und = (Fv[aa][2] | Fv[bb][2]).mean()
                    undef_max = max(undef_max, float(und)) if rule.startswith("DR") and h == H_PRIMARY and sc == "main" else undef_max
                    lo, hi = ci(d)
                    Crows.append(dict(fold=yr, h=h, scenario=sc, rule=rule, a=aa if aa != "best_cp" else f"best_cp={best}",
                                      b=bb, d=Fv[aa][0] - Fv[bb][0], ci_lo=lo, ci_hi=hi, undefined_share=float(und),
                                      rel=(Fv[aa][0] - Fv[bb][0]) / Fv[bb][0]))
                # DR1 and DR6 rows (A-R and A-C)
                hbx = J.row_of("handbook", np.nan)
                for pol in POLICIES:
                    for proto in ("A_R", "A_C"):
                        s = J.sel["policies"][pol][proto]
                        k = J.row_of(pol, s["param"])
                        dfu = J.bt["fuel_L"][k] - J.bt["fuel_L"][hbx]
                        due = J.bt["ue_kWh"][k] - J.bt["ue_kWh"][hbx]
                        D1.append(dict(fold=yr, h=h, scenario=sc, policy=pol, protocol=proto, param=s["param"],
                                       met_H2=s.get("target_met_on_H2"), fuel_L=J.pt["fuel_L"][k],
                                       ue_kWh=J.pt["ue_kWh"][k], ue_share=J.pt["ue_share"][k],
                                       fuel_handbook_L=J.pt["fuel_L"][hbx],
                                       d_fuel_vs_handbook=J.pt["fuel_L"][k] - J.pt["fuel_L"][hbx],
                                       d_fuel_ci=ci(dfu), d_ue_kWh_ci=ci(due),
                                       passes=bool(J.pt["fuel_L"][k] < J.pt["fuel_L"][hbx]
                                                   and J.pt["ue_share"][k] <= UE_TARGET + 1e-15)))
                    if pol in CP:
                        for base in ("frac_pv", "tod_q"):
                            for proto in ("A_R", "A_C"):
                                ka = J.row_of(pol, J.sel["policies"][pol][proto]["param"])
                                kb = J.row_of(base, J.sel["policies"][base][proto]["param"])
                                row = dict(fold=yr, h=h, scenario=sc, policy=pol, baseline=base, protocol=proto)
                                for q in ("fuel_L", "ue_kWh", "ue_share"):
                                    row[f"d_{q}"] = J.pt[q][ka] - J.pt[q][kb]
                                    row[f"d_{q}_ci"] = ci(J.bt[q][ka] - J.bt[q][kb])
                                D6.append(row)
                            dF = Fv[pol][1] - Fv[base][1]
                            D6.append(dict(fold=yr, h=h, scenario=sc, policy=pol, baseline=base, protocol="F",
                                           d_fuel_L=Fv[pol][0] - Fv[base][0], d_fuel_L_ci=ci(dF)))
                # DR4 levels
                for si, (sn, sb) in enumerate(SHAPE_PAIRS):
                    if sc == sn:
                        for pol in POLICIES:
                            s = J.sel["policies"][pol]["A_R"]
                            lv4.setdefault((yr, pol), {}).setdefault((si, h), [None] * 4)[0:2] = [s["param"], s["target_met_on_H2"]]
                    if sc == sb:
                        for pol in POLICIES:
                            s = J.sel["policies"][pol]["A_R"]
                            lv4.setdefault((yr, pol), {}).setdefault((si, h), [None] * 4)[2:4] = [s["param"], s["target_met_on_H2"]]
            # DR5
            pf = R / a.s6p / f"f{yr}_h{h}_proxy_rows.parquet"
            inputs[f"{a.s6p}/{pf.name}"] = md5(pf)
            P = pd.read_parquet(pf)
            P = P[(P["mode"] == "reset") & (P.ratio == 10.0) & (P.batt == "default") & (P.protocol == "A") & (P.lam == 0.0)]
            vc = P.set_index("policy")["vc_mean"].reindex(POLICIES).to_numpy(float)
            Fm = np.array([r["F"] for r in Frows if r["fold"] == yr and r["h"] == h and r["scenario"] == "main"
                           and r["policy"] in POLICIES])
            D5.append(dict(fold=yr, h=h, kendall_tau=kendall_tau(vc, -Fm),
                           top_proxy=POLICIES[int(np.nanargmax(vc))], top_load=POLICIES[int(np.nanargmin(Fm))],
                           proxy_order=[POLICIES[i] for i in np.argsort(-vc)],
                           load_order=[POLICIES[i] for i in np.argsort(Fm)]))
            log(f"fold {yr} h={h} done ({time.time() - t0:.0f}s)")

    I = pd.DataFrame(ident); FR = pd.DataFrame(Frows); CR = pd.DataFrame(Crows)
    DD1 = pd.DataFrame(D1); DD6 = pd.DataFrame(D6); DD5 = pd.DataFrame(D5)
    DD4 = []
    for (yr, pol), lv in lv4.items():
        full = {k: tuple(v) for k, v in lv.items()}
        DD4.append(dict(fold=yr, policy=pol, passes=dr4_pass(full),
                        **{f"s{s}_h{h}": full[(s, h)] for s in (0, 1) for h in HORIZONS}))
    DD4 = pd.DataFrame(DD4)

    # checks (D-091)
    check("daily_sums_equal_test_totals", I.max_rel_diff.max() <= 1e-4, f"(max relative {I.max_rel_diff.max():.1e}, {len(I)} files x scenarios)")
    check("F_equals_S6_recorded_where_finite", not f_mismatch, f"({len(f_mismatch)} mismatches)")
    check("handbook_zero_UE_main", hb_ue_main == 0.0, f"(max {hb_ue_main})")
    check("level_sets_identical_across_folds", len(level_sets) == 1, f"({len(level_sets)} distinct sets)")
    check("every_job_done", len(I) == len(FOLDS) * len(HORIZONS) * 4, f"({len(I)})")

    # verdicts
    V = {}
    p1 = DD1[(DD1.h == H_PRIMARY) & (DD1.protocol == "A_R") & DD1.scenario.isin(["main", "shapeB"])]
    both = p1.groupby(["policy", "fold"]).passes.all().groupby("policy").sum()
    V["DR1"] = {pol: dict(folds_passing_both_shapes=int(both[pol]), holds=bool(both[pol] >= FOLD_MIN)) for pol in POLICIES}
    pc = CR[(CR.h == H_PRIMARY) & (CR.scenario == "main")]
    for rule_key, (aa, bb) in {"DR2": ("best_cp", "raw_q"), "DR3_mcqr_vs_cqr": ("mondrian_cqr", "cqr"),
                               "DR3_mondrian_vs_icp": ("mondrian", "icp"),
                               "DR3_mondrian_vs_nicp_var": ("mondrian", "nicp_var")}.items():
        sub = pc[(pc.a.str.startswith(aa)) & (pc.b == bb)]
        gain = ((sub.ci_hi < 0) & (sub.undefined_share <= UNDEF_MAX)).sum()
        cost = ((sub.ci_lo > 0) & (sub.undefined_share <= UNDEF_MAX)).sum()
        V[rule_key] = dict(folds_gain=int(gain), folds_cost=int(cost), credited=bool(gain >= FOLD_MIN),
                           costs_fuel=bool(cost >= FOLD_MIN), best_cp_by_fold=list(sub.a) if aa == "best_cp" else None)
    fl = FR[(FR.h == H_PRIMARY) & (FR.scenario == "main") & FR.policy.isin(["cqr", "mondrian_cqr", "raw_q", "mondrian", "icp", "nicp_var"])]
    V["limit_bracket_trigger"] = dict(fired=bool(fl.flag_0999_1.any()), cases=fl[fl.flag_0999_1][["fold", "policy"]].values.tolist(),
                               at_limit_cases=fl[fl.flag_at_limit][["fold", "policy"]].values.tolist())
    cp4 = DD4[DD4.policy.isin(CP)].groupby("policy").passes.sum()
    V["DR4"] = dict(folds_passing={p: int(cp4[p]) for p in CP},
                    outcome=("KEPT" if (cp4 >= FOLD_MIN).all() else "REVISED" if (cp4 >= FOLD_MIN).any() else "DROPPED"),
                    baselines={p: int(DD4[DD4.policy == p].passes.sum()) for p in POLICIES if p not in CP})
    V["DR5"] = dict(differs_any=bool((DD5.kendall_tau < 1).any()), top_agree=int((DD5.top_proxy == DD5.top_load).sum()),
                    n=len(DD5), note="load model governs the main text either way")
    V["undefined_share_max_primary"] = undef_max
    check("undefined_share_within_rule", undef_max <= UNDEF_MAX, f"(max {undef_max:.4f})")

    for df, name in ((I, "checks_identity.csv"), (FR, "frontier_F.csv"), (CR, "contrasts_F.csv"), (DD1, "dr1.csv"),
                     (DD4, "dr4.csv"), (DD5, "dr5.csv"), (DD6, "dr6.csv")):
        df.to_csv(out / name, index=False)
    json.dump(V, open(out / "verdicts.json", "w"), indent=1, default=str)
    ok = all(c["ok"] for c in checks.values())
    detail = dict(stage=a.out, inputs_from=[a.s6, a.s6b, a.s6p], smoke=False, ok=ok, B=a.B, seed=SEED, rule="D-091", seconds=round(time.time() - t0, 1),
                  checks=checks, outputs={p.name: md5(p) for p in sorted(out.glob("*")) if p.is_file()},
                  inputs=inputs, code={p: md5(HERE / p) for p in ("ce1_decision.py", "ce1_load.py")},
                  versions=dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__),
                  host=platform.node())
    json.dump(detail, open(runs / f"{a.out}.detail.json", "w"), indent=1)
    now = dt.datetime.now().isoformat(timespec="seconds")
    json.dump(dict(stage=a.out, ok=ok, error="" if ok else "checks failed", finished=now,
                   minutes=round((time.time() - t0) / 60, 1), host=platform.node(), smoke=False,
                   where="Linux"), open(runs / f"{a.out}.json", "w"), indent=2)
    log(f"{'ALL CHECKS PASSED' if ok else 'SOME CHECKS FAILED'} in {time.time() - t0:.0f}s")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
