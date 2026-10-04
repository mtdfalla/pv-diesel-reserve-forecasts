"""Reserve dispatch (v1 code): one-sided lower-bound policies at level theta, the reserve level
selected on data that exclude the test year, and daily cost vectors for the day-block
bootstrap. Used by ce1_dispatch.py.

Usage (from the repository root unless the path starts at ce1/, then from 03_code):
    python3 r1/r1_dispatch.py <site: yulara|asp> <h_steps: 1|3|6|12> [task]
"""
import sys, os, json, time, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0, "utils")
import numpy as np, pandas as pd
import datasets as D

# ---------------------------------------------------------------- parameters
DT       = 5 / 60.0  # hours per step
C_O_DEF  = 0.30  # $/kWh over-procured (committed diesel wasted)
RATIO_DEF = 10.0  # c_u / c_o
BATT     = {"small": (250.0, 250.0), "default": (600.0, 400.0), "large": (1500.0, 600.0)}
ETA      = 0.949  # one-way efficiency (round trip ~0.90)
SOC_MIN, SOC_MAX = 0.10, 1.0
THETA_GRID = np.round(np.arange(0.75, 0.9501, 0.025), 4)
THETA_R0_GRID = np.array([0.5, 0.6, 0.7, 0.8, 0.9, 0.95])
LAMBDAS  = [0.0, 0.25, 0.5, 0.75, 1.0]
CVAR_ALPHA = 0.95
POLICIES = ["deterministic", "icp", "mondrian", "cqr", "mondrian_cqr", "oracle"]
CP_POLICIES = ["icp", "mondrian", "cqr", "mondrian_cqr"]
SEED = 42

_m = np.load("/tmp/ghi_pv_map.npz")
CENT, VALS, CAP = _m["centers"], _m["vals"], float(_m["cap"])
def pv_of(ghi):
    """Deterministic capacity-clipped binned-median GHI->PV curve f."""
    return np.clip(np.interp(ghi, CENT, VALS, left=0, right=CAP), 0, CAP)


# ---------------------------------------------------------------- prediction cache
def load_preds(site, h):
    """Return (calib 2023, test 2024) prediction sets from the S0 cache."""
    basep = "/tmp/base.parquet" if site == "yulara" else "/tmp/base_asp.parquet"
    base = pd.read_parquet(basep)
    d = D.make_xy(base, h).sort_index()
    ca = d[d.year == 2023]; te = d[d.year == 2024]
    pre = f"/tmp/r1cache/{site}_h{h}"
    meta = json.load(open(f"{pre}_meta.json"))
    taus = sorted(meta["taus"])
    extra = f"/tmp/r1cache_extra/{site}_h{h}"
    for t in (0.175, 0.225):
        if os.path.exists(f"{extra}_q{t}_c.npy"):
            taus.append(t)
    taus = sorted(set(taus))
    def grab(name, which):
        f1, f2 = f"{pre}_{name}_{which}.npy", f"{extra}_{name}_{which}.npy"
        return np.load(f1) if os.path.exists(f1) else np.load(f2)
    out = {}
    for tag, dd, w in (("c", ca, "c"), ("t", te, "t")):
        out[tag] = dict(idx=dd.index, y=dd["y_ghi"].values.astype(float),
                        p=grab("point", w).astype(float),
                        g=dd["base_regime"].values,
                        q={t: grab(f"q{t}", w).astype(float) for t in taus})
    out["taus"] = taus
    out["test_rmse"] = meta["test_rmse"]
    return out


def subset(ps, mask):
    return dict(idx=ps["idx"][mask], y=ps["y"][mask], p=ps["p"][mask], g=ps["g"][mask],
                q={t: v[mask] for t, v in ps["q"].items()})


def concat(a, b):
    return dict(idx=a["idx"].append(b["idx"]), y=np.r_[a["y"], b["y"]], p=np.r_[a["p"], b["p"]],
                g=np.r_[a["g"], b["g"]], q={t: np.r_[a["q"][t], b["q"][t]] for t in a["q"]})


# ---------------------------------------------------------------- policies
def counted_ghi(policy, theta, fit, ev, taus):
    """GHI the operator counts on: one-sided lower bound at level theta.

    `fit` supplies the conformal scores (the calibration set); `ev` is evaluated.
    """
    if policy == "deterministic":
        return ev["p"].copy()
    if policy == "oracle":
        return ev["y"].copy()
    if policy == "icp":
        e = fit["y"] - fit["p"]
        return np.clip(ev["p"] + np.quantile(e, 1 - theta), 0, None)
    if policy == "mondrian":
        e = fit["y"] - fit["p"]
        out = np.empty(len(ev["p"]))
        for g in np.unique(ev["g"]):
            eg = e[fit["g"] == g]
            off = np.quantile(eg if len(eg) > 30 else e, 1 - theta)
            out[ev["g"] == g] = np.clip(ev["p"][ev["g"] == g] + off, 0, None)
        return out
    if policy in ("cqr", "mondrian_cqr"):
        tau = min(taus, key=lambda z: abs(z - (1 - theta)))
        s_all = fit["q"][tau] - fit["y"]
        if policy == "cqr":
            return np.clip(ev["q"][tau] - np.quantile(s_all, theta), 0, None)
        out = np.empty(len(ev["p"]))
        for g in np.unique(ev["g"]):
            sg = s_all[fit["g"] == g]
            adj = np.quantile(sg, theta) if len(sg) > 30 else np.quantile(s_all, theta)
            out[ev["g"] == g] = np.clip(ev["q"][tau][ev["g"] == g] - adj, 0, None)
        return out
    raise ValueError(policy)


# ---------------------------------------------------------------- simulation
def day_matrix(idx):
    """Pad the evaluation period into an (n_days x max_steps) index matrix."""
    day = idx.normalize().values
    udays, inv = np.unique(day, return_inverse=True)
    counts = np.bincount(inv, minlength=len(udays))
    W = int(counts.max())
    order = np.argsort(inv, kind="stable")  # groups row indices by day
    pos = np.concatenate([np.arange(c) for c in counts])  # within-day position
    M = np.full((len(udays), W), -1, dtype=np.int64)
    M[inv[order], pos] = order
    return udays, M


def simulate(counted, actual, M, E_max, P_max, c_u, c_o):
    """Vectorised SoC dispatch. Returns (daily cost vector, soc_min_frac, soc_max_frac)."""
    pv_c = pv_of(counted); pv_a = pv_of(actual)
    valid = M >= 0
    Mi = np.where(valid, M, 0)
    C = np.where(valid, pv_c[Mi], 0.0)
    A = np.where(valid, pv_a[Mi], 0.0)
    n, W = M.shape
    soc = np.full(n, SOC_MIN * E_max)
    cost = np.zeros(n)
    lo = soc.copy(); hi = soc.copy()
    for k in range(W):
        diff = A[:, k] - C[:, k]
        deficit = np.maximum(-diff, 0.0)
        surplus = np.maximum(diff, 0.0)
        dis = np.clip(np.minimum(np.minimum(deficit, P_max), (soc - SOC_MIN * E_max) * ETA / DT), 0, None)
        chg = np.clip(np.minimum(np.minimum(surplus, P_max), (SOC_MAX * E_max - soc) / (ETA * DT)), 0, None)
        soc = soc - dis * DT / ETA + chg * ETA * DT
        cost += c_u * (deficit - dis) * DT + c_o * (surplus - chg) * DT
        lo = np.minimum(lo, soc); hi = np.maximum(hi, soc)
    return cost, float(lo.min() / E_max), float(hi.max() / E_max)


def cvar(x, alpha=CVAR_ALPHA):
    k = max(1, int(np.ceil((1 - alpha) * len(x))))
    return float(np.mean(np.sort(x)[::-1][:k]))


def cvar_rows(X, alpha=CVAR_ALPHA):
    """CVaR of every row of X (used by the bootstrap)."""
    k = max(1, int(np.ceil((1 - alpha) * X.shape[1])))
    part = np.partition(X, X.shape[1] - k, axis=1)[:, X.shape[1] - k:]
    return part.mean(axis=1)


# ---------------------------------------------------------------- selection
def sweep(fit, ev, M, taus, thetas, E, P, c_u, c_o, policies=CP_POLICIES):
    """Daily-cost vector for every (policy, theta) on `ev`, scores fit on `fit`."""
    out = {}
    for pol in policies:
        for th in thetas:
            cg = counted_ghi(pol, th, fit, ev, taus)
            dc, slo, shi = simulate(cg, ev["y"], M, E, P, c_u, c_o)
            out[(pol, round(float(th), 4))] = (dc, slo, shi)
    return out


def anchors(fit, ev, M, taus, E, P, c_u, c_o):
    """deterministic and oracle daily-cost vectors (theta-free)."""
    a = {}
    for pol in ("deterministic", "oracle"):
        cg = counted_ghi(pol, 0.5, fit, ev, taus)
        a[pol] = simulate(cg, ev["y"], M, E, P, c_u, c_o)
    return a


def select_theta(sw, pol, thetas, lam=0.0):
    """argmin over theta of (1-lam)*E[C] + lam*CVaR_0.95[C] on the SELECTION set."""
    best, best_th = np.inf, None
    for th in thetas:
        dc = sw[(pol, round(float(th), 4))][0]
        obj = (1 - lam) * float(dc.mean()) + lam * cvar(dc)
        if obj < best:
            best, best_th = obj, round(float(th), 4)
    return best_th, best


# ---------------------------------------------------------------- bootstrap
def boot_ci(dc_method, dc_det, B=10000, seed=SEED, alpha=CVAR_ALPHA):
    """Paired day-block bootstrap -> 95% CI on mean, CVaR, VC(mean), VC(CVaR).

    Oracle cost is identically zero (counted == actual), so
    VC(mean) = 1 - mean_m/mean_det and VC(CVaR) = 1 - CVaR_m/CVaR_det.
    """
    rng = np.random.default_rng(seed)
    n = len(dc_method)
    idx = rng.integers(0, n, size=(B, n))
    Xm = dc_method[idx]; Xd = dc_det[idx]
    mm = Xm.mean(axis=1); md = Xd.mean(axis=1)
    cm = cvar_rows(Xm, alpha); cd = cvar_rows(Xd, alpha)
    q = lambda v: [round(float(np.percentile(v, 2.5)), 4), round(float(np.percentile(v, 97.5)), 4)]
    return dict(mean_ci=q(mm), cvar_ci=q(cm),
                vc_mean_ci=q(1 - mm / np.where(md > 0, md, np.nan)),
                vc_cvar_ci=q(1 - cm / np.where(cd > 0, cd, np.nan)))


# ---------------------------------------------------------------- protocols
def build_sets(P):
    """Split calibration year 2023 into halves; 2024 is the test set."""
    c = P["c"]; t = P["t"]
    h1 = c["idx"].month <= 6
    return dict(H1=subset(c, h1.values if hasattr(h1, "values") else h1),
                H2=subset(c, ~(h1.values if hasattr(h1, "values") else h1)),
                FULL=c, TEST=t)


def run_main(site, h, c_o=C_O_DEF, ratio=RATIO_DEF, batt="default", P=None):
    c_u = c_o * ratio
    E, Pw = BATT[batt]
    P = P or load_preds(site, h)
    taus = P["taus"]; S = build_sets(P)
    Mt = day_matrix(S["TEST"]["idx"])[1]
    M1 = day_matrix(S["H1"]["idx"])[1]
    M2 = day_matrix(S["H2"]["idx"])[1]

    # ---- selection stage (never touches 2024) ----
    sel_A = sweep(S["H1"], S["H2"], M2, taus, THETA_GRID, E, Pw, c_u, c_o)  # scores H1 -> select on H2
    sel_S = sweep(S["H2"], S["H1"], M1, taus, THETA_GRID, E, Pw, c_u, c_o)  # swapped halves

    # ---- evaluation stage: scores refit on FULL 2023, evaluated once on 2024 ----
    ev = sweep(S["FULL"], S["TEST"], Mt, taus, np.r_[THETA_GRID, 0.909], E, Pw, c_u, c_o)
    ev_r0 = sweep(S["FULL"], S["TEST"], Mt, taus, THETA_R0_GRID, E, Pw, c_u, c_o)
    anc = anchors(S["FULL"], S["TEST"], Mt, taus, E, Pw, c_u, c_o)
    det_dc = anc["deterministic"][0]; ora_dc = anc["oracle"][0]
    det_mean, det_cvar = float(det_dc.mean()), cvar(det_dc)

    def row(pol, th, protocol, lam=None, sel_obj=None):
        dc, slo, shi = ev[(pol, round(float(th), 4))] if (pol, round(float(th), 4)) in ev \
            else ev_r0[(pol, round(float(th), 4))]
        m, cv = float(dc.mean()), cvar(dc)
        return dict(site=site, horizon_min=h * 5, protocol=protocol, policy=pol,
                    theta=round(float(th), 4), lam=lam,
                    mean_daily=round(m, 2), cvar95_daily=round(cv, 2),
                    value_captured_mean=round(1 - m / det_mean, 4) if det_mean > 0 else None,
                    value_captured_cvar=round(1 - cv / det_cvar, 4) if det_cvar > 0 else None,
                    soc_min=round(slo, 3), soc_max=round(shi, 3),
                    selection_objective=None if sel_obj is None else round(float(sel_obj), 2))

    rows = []
    rows.append(dict(site=site, horizon_min=h * 5, protocol="anchor", policy="deterministic",
                     theta=None, lam=None, mean_daily=round(det_mean, 2),
                     cvar95_daily=round(det_cvar, 2), value_captured_mean=0.0,
                     value_captured_cvar=0.0, soc_min=round(anc["deterministic"][1], 3),
                     soc_max=round(anc["deterministic"][2], 3), selection_objective=None))
    rows.append(dict(site=site, horizon_min=h * 5, protocol="anchor", policy="oracle",
                     theta=None, lam=None, mean_daily=round(float(ora_dc.mean()), 2),
                     cvar95_daily=round(cvar(ora_dc), 2), value_captured_mean=1.0,
                     value_captured_cvar=1.0, soc_min=round(anc["oracle"][1], 3),
                     soc_max=round(anc["oracle"][2], 3), selection_objective=None))

    theta_log = []
    for pol in CP_POLICIES:
        th_r0 = min(THETA_R0_GRID, key=lambda z: ev_r0[(pol, round(float(z), 4))][0].mean())
        rows.append(row(pol, th_r0, "R0_test_argmin_coarse"))
        th_r0f = min(THETA_GRID, key=lambda z: ev[(pol, round(float(z), 4))][0].mean())
        rows.append(row(pol, th_r0f, "R0_test_argmin_fine"))
        # Protocol A: selected on 2023-H2 (scores on 2023-H1), evaluated once on 2024
        for lam in LAMBDAS:
            thA, objA = select_theta(sel_A, pol, THETA_GRID, lam)
            tag = "A" if lam == 0.0 else f"A_meancvar_lam{lam}"
            rows.append(row(pol, thA, tag, lam=lam, sel_obj=objA))
            if lam == 0.0:
                thS, objS = select_theta(sel_S, pol, THETA_GRID, lam)
                rows.append(row(pol, thS, "A_swapped", lam=lam, sel_obj=objS))
                theta_log.append(dict(site=site, horizon_min=h * 5, policy=pol,
                                      theta_R0_test_argmin_coarse=float(th_r0),
                                      theta_R0_test_argmin_fine=float(th_r0f),
                                      theta_protocolA=float(thA), theta_protocolA_swapped=float(thS),
                                      theta_protocolB=0.909))
        # Protocol B: zero tuning, theta = c_u/(c_u+c_o)
        rows.append(row(pol, 0.909, "B_critical_fractile"))

    # ---- bootstrap CIs on the headline protocols ----
    cis = []
    for pol in CP_POLICIES:
        for tag, th in [("R0_test_argmin_coarse", min(THETA_R0_GRID, key=lambda z: ev_r0[(pol, round(float(z), 4))][0].mean())),
                        ("A", select_theta(sel_A, pol, THETA_GRID, 0.0)[0]),
                        ("A_meancvar_lam0.5", select_theta(sel_A, pol, THETA_GRID, 0.5)[0]),
                        ("B_critical_fractile", 0.909)]:
            key = (pol, round(float(th), 4))
            dc = ev[key][0] if key in ev else ev_r0[key][0]
            ci = boot_ci(dc, det_dc)
            cis.append(dict(site=site, horizon_min=h * 5, protocol=tag, policy=pol,
                            theta=round(float(th), 4), **ci))

    # ---- full frontier on the fine grid, for the efficient-set figure ----
    front = []
    for (pol, th), (dc, slo, shi) in ev.items():
        front.append(dict(site=site, horizon_min=h * 5, policy=pol, theta=th,
                          set="test2024", mean_daily=round(float(dc.mean()), 2),
                          cvar95_daily=round(cvar(dc), 2)))
    for (pol, th), (dc, _, _) in sel_A.items():
        front.append(dict(site=site, horizon_min=h * 5, policy=pol, theta=th,
                          set="selection2023H2", mean_daily=round(float(dc.mean()), 2),
                          cvar95_daily=round(cvar(dc), 2)))

    daily = {f"{pol}|{th}": ev[(pol, th)][0] for (pol, th) in ev}
    daily["deterministic|na"] = det_dc; daily["oracle|na"] = ora_dc
    return dict(rows=rows, theta_log=theta_log, cis=cis, frontier=front, daily=daily,
                det_mean=det_mean, det_cvar=det_cvar,
                n_test_days=int(Mt.shape[0]), n_test_steps=int(len(S["TEST"]["y"])),
                params=dict(c_o=c_o, c_u=c_u, ratio=ratio, batt=batt, E_max=E, P_max=Pw,
                            ETA=ETA, SOC_MIN=SOC_MIN, CAP=CAP, theta_grid=THETA_GRID.tolist()))


def run_battery(site, h, P=None):
    P = P or load_preds(site, h); taus = P["taus"]; S = build_sets(P)
    Mt = day_matrix(S["TEST"]["idx"])[1]; M2 = day_matrix(S["H2"]["idx"])[1]
    c_o, c_u = C_O_DEF, C_O_DEF * RATIO_DEF
    out = []
    for name, (E, Pw) in BATT.items():
        sel = sweep(S["H1"], S["H2"], M2, taus, THETA_GRID, E, Pw, c_u, c_o)
        anc = anchors(S["FULL"], S["TEST"], Mt, taus, E, Pw, c_u, c_o)
        dm = float(anc["deterministic"][0].mean()); dcv = cvar(anc["deterministic"][0])
        for pol in ["deterministic", "oracle"] + CP_POLICIES:
            if pol in ("deterministic", "oracle"):
                dc, slo, shi = anc[pol]; th = None
            else:
                th = select_theta(sel, pol, THETA_GRID, 0.0)[0]
                cg = counted_ghi(pol, th, S["FULL"], S["TEST"], taus)
                dc, slo, shi = simulate(cg, S["TEST"]["y"], Mt, E, Pw, c_u, c_o)
            m, cv = float(dc.mean()), cvar(dc)
            out.append(dict(site=site, horizon_min=h * 5, batt=name, E_max=E, P_max=Pw,
                            policy=pol, theta=th, protocol="A",
                            mean_daily=round(m, 2), cvar95_daily=round(cv, 2),
                            value_captured_mean=round(1 - m / dm, 4) if dm > 0 else None,
                            value_captured_cvar=round(1 - cv / dcv, 4) if dcv > 0 else None,
                            soc_min=round(slo, 3), soc_max=round(shi, 3)))
    return out


def run_costratio(site, h, P=None):
    P = P or load_preds(site, h); taus = P["taus"]; S = build_sets(P)
    Mt = day_matrix(S["TEST"]["idx"])[1]; M2 = day_matrix(S["H2"]["idx"])[1]
    E, Pw = BATT["default"]; c_o = C_O_DEF
    out = []
    for ratio in [3.0, 5.0, 10.0, 19.0]:
        c_u = c_o * ratio
        thB = round(ratio / (1 + ratio), 4)
        sel = sweep(S["H1"], S["H2"], M2, taus, THETA_GRID, E, Pw, c_u, c_o)
        anc = anchors(S["FULL"], S["TEST"], Mt, taus, E, Pw, c_u, c_o)
        dm = float(anc["deterministic"][0].mean()); dcv = cvar(anc["deterministic"][0])
        for pol in ["deterministic", "oracle"] + CP_POLICIES:
            for proto in (["-"] if pol in ("deterministic", "oracle") else ["A", "B"]):
                if pol in ("deterministic", "oracle"):
                    dc, slo, shi = anc[pol]; th = None
                else:
                    th = select_theta(sel, pol, THETA_GRID, 0.0)[0] if proto == "A" else thB
                    cg = counted_ghi(pol, th, S["FULL"], S["TEST"], taus)
                    dc, slo, shi = simulate(cg, S["TEST"]["y"], Mt, E, Pw, c_u, c_o)
                m, cv = float(dc.mean()), cvar(dc)
                out.append(dict(site=site, horizon_min=h * 5, ratio=ratio, c_u=c_u, c_o=c_o,
                                policy=pol, protocol=proto, theta=th,
                                mean_daily=round(m, 2), cvar95_daily=round(cv, 2),
                                value_captured_mean=round(1 - m / dm, 4) if dm > 0 else None,
                                value_captured_cvar=round(1 - cv / dcv, 4) if dcv > 0 else None))
    return out


# ---------------------------------------------------------------- CLI
if __name__ == "__main__":
    site = sys.argv[1]; h = int(sys.argv[2])
    task = sys.argv[3] if len(sys.argv) > 3 else "all"
    t0 = time.time()
    os.makedirs("/tmp/r1j5out", exist_ok=True)
    P = load_preds(site, h)
    res = {}
    if task in ("main", "all"):
        m = run_main(site, h, P=P)
        np.savez_compressed(f"/tmp/r1j5out/{site}_h{h}_daily.npz", **m.pop("daily"))
        res["main"] = m
        print(f"[{site} h={h*5}] main done @{time.time()-t0:.0f}s", flush=True)
    if task in ("battery", "all"):
        res["battery"] = run_battery(site, h, P=P)
        print(f"[{site} h={h*5}] battery done @{time.time()-t0:.0f}s", flush=True)
    if task in ("costratio", "all"):
        res["costratio"] = run_costratio(site, h, P=P)
        print(f"[{site} h={h*5}] costratio done @{time.time()-t0:.0f}s", flush=True)
    json.dump(res, open(f"/tmp/r1j5out/{site}_h{h}_{task}.json", "w"), default=str)
    print(f"[{site} h={h*5}] ALL_DONE_R1J5 in {time.time()-t0:.0f}s", flush=True)
