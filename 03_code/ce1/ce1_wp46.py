"""Stage S46: regime-threshold robustness of the coverage results. Nine regime definitions, one
constant varied at a time, every static method at 90% (rule D-095).

Usage (from the repository root unless the path starts at ce1/, then from 03_code):
    python3 03_code/ce1/ce1_wp46.py --root . [--only 2024:1] [--B 10000]
    python3 03_code/ce1/ce1_wp46.py --selftest
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
import ce1_causal as K  # noqa: E402
import ce1_conformal as CF  # noqa: E402
import ce1_folds as FC  # noqa: E402
from ce1_stats import DayBoot, day_codes  # noqa: E402

BASE = dict(KT_CLEAR=0.85, KT_OVERCAST=0.50, VAR_STABLE=0.10, VAR_WIN=12)
DEFS = [("base", {})] + [(f"{k}={v}", {k: v}) for k, vals in
                         (("KT_CLEAR", (0.80, 0.90)), ("KT_OVERCAST", (0.40, 0.60)),
                          ("VAR_STABLE", (0.075, 0.125)), ("VAR_WIN", (6, 24))) for v in vals]
FOLDS = [2019, 2020, 2021, 2022, 2023, 2024]
HORIZONS = [1, 3]
COV = 0.90
REGIMES = ["clear", "transitional", "cloudy"]
METHODS = ["icp", "icp_norm", "mondrian", "cqr", "mondrian_cqr", "raw_q", "tod", "nicp_var"]
MIN_ROWS = 30
B_DEF = 10000


def md5(p: Path) -> str:
    h = hashlib.md5()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def log(m):
    print(f"[S46] {m}", flush=True)


class patched:
    """Temporarily set ce1_causal's regime constants (restored on exit)."""

    def __init__(self, **kw):
        self.kw, self.old = kw, {}

    def __enter__(self):
        for k, v in self.kw.items():
            self.old[k] = getattr(K, k)
            setattr(K, k, v)

    def __exit__(self, *a):
        for k, v in self.old.items():
            setattr(K, k, v)


def labels(geo, cal, h, idx, overrides):
    """Issue-time and target-time regime labels at the issue times idx under a definition."""
    with patched(**overrides):
        base = K.build_base(geo, cal, "locf6")
    reg = base["regime"]
    return reg.reindex(idx).values.astype(str), reg.shift(-h).reindex(idx).values.astype(str)


def status(lo, hi, nominal=COV):
    if not np.isfinite(lo):
        return "n/a"
    return "below" if hi < nominal else ("above" if lo > nominal else "includes")


def evaluate(cache, g_it, g_tt, B, fold, h, name):
    """Coverage rows of every method and scope for one definition (labels already set)."""
    c = cache[cache["split"] == "calib"].copy()
    t = cache[cache["split"] == "test"].copy()
    nc = (cache["split"] == "calib").values
    c["g"], t["g"] = g_it[nc], g_it[~nc]
    iv, _ = CF.static_intervals(c, t, COV)
    y = t["y"].values
    code, nd = day_codes(t.index.values)
    boot = DayBoot(code, nd, B)
    git, gtt = g_it[~nc], g_tt[~nc]
    n = len(y)
    scopes = [("all", np.ones(n, bool))] + [(r, git == r) for r in REGIMES] + [(f"tt_{r}", gtt == r) for r in REGIMES]
    rows = []
    for m in METHODS:
        lo_, hi_ = iv[m]
        hit = ((y >= lo_) & (y <= hi_)).astype(float)
        for sc, mask in scopes:
            k = int(mask.sum())
            if k < MIN_ROWS:
                rows.append(dict(definition=name, fold=fold, horizon_min=5 * h, method=m, scope=sc, n=k,
                                 share=k / n, picp=np.nan, ci_lo=np.nan, ci_hi=np.nan, status="n/a"))
                continue
            p, a, b, _ = boot.mean_ci(hit, mask)
            rows.append(dict(definition=name, fold=fold, horizon_min=5 * h, method=m, scope=sc, n=k, share=k / n,
                             picp=p, ci_lo=a, ci_hi=b, status=status(a, b)))
    return rows


def reading(cov: pd.DataFrame) -> dict:
    """D-095 reading rule on the issue-time transitional scope."""
    tr = cov[cov.scope == "transitional"]
    key = ["fold", "horizon_min", "method"]
    base = tr[tr.definition == "base"].set_index(key)["status"]
    out, changes = {}, []
    for d in tr.definition.unique():
        if d == "base":
            continue
        cur = tr[tr.definition == d].set_index(key)["status"]
        for k, s0 in base.items():
            s1 = cur.get(k, "n/a")
            if s1 != s0:
                changes.append(dict(definition=d, fold=k[0], horizon_min=k[1], method=k[2], base=s0, now=s1))
    ch = pd.DataFrame(changes, columns=["definition", "fold", "horizon_min", "method", "base", "now"])
    icp_base_below = base[[k for k in base.index if k[2] == "icp" and base[k] == "below"]]
    cl01 = all(tr[(tr.definition == d) & (tr.fold == k[0]) & (tr.horizon_min == k[1]) & (tr.method == "icp")]
               .status.iloc[0] == "below" for d in tr.definition.unique() for k in icp_base_below.index)
    mc_base_ok = base[[k for k in base.index if k[2] == "mondrian_cqr" and base[k] in ("includes", "above")]]
    cl03 = all(tr[(tr.definition == d) & (tr.fold == k[0]) & (tr.horizon_min == k[1]) & (tr.method == "mondrian_cqr")]
               .status.iloc[0] in ("includes", "above") for d in tr.definition.unique() for k in mc_base_ok.index)
    out.update(CL01_02_robust=bool(cl01), CL03_robust=bool(cl03), n_status_changes=int(len(ch)),
               icp_below_cells_base=int(len(icp_base_below)), mcqr_not_below_cells_base=int(len(mc_base_ok)),
               not_evaluable=int((cov.status == "n/a").sum()))
    return out, ch


def selftest(verbose=True) -> dict:
    res = {}

    def ok(name, cond, msg=""):
        res[name] = bool(cond)
        if verbose:
            print(f"  [{'PASS' if cond else 'FAIL'}] {name} {msg}")

    ok("nine_definitions", len(DEFS) == 9 and DEFS[0] == ("base", {}))
    ok("base_constants_match_module", all(getattr(K, k) == v for k, v in BASE.items()))
    with patched(KT_CLEAR=0.9):
        inside = K.KT_CLEAR == 0.9
    ok("patch_restores", inside and K.KT_CLEAR == 0.85)
    ok("status_rule", status(0.8, 0.89) == "below" and status(0.88, 0.92) == "includes"
       and status(0.91, 0.95) == "above" and status(np.nan, np.nan) == "n/a")
    geo = K._synthetic_geo(n_days=6)
    cal = 1.0
    idx = K.build_base(geo, cal).index[::7]
    a, _ = labels(geo, cal, 3, idx, {})
    b, _ = labels(geo, cal, 3, idx, {"KT_CLEAR": 0.99})
    ok("labels_move_with_threshold", (a == "clear").sum() >= (b == "clear").sum())
    rows = []
    for d in ("base", "X"):
        for m, s in (("icp", "below"), ("mondrian_cqr", "includes")):
            rows.append(dict(definition=d, fold=2024, horizon_min=5, method=m, scope="transitional",
                             status=s if not (d == "X" and m == "icp") else "includes"))
    r, ch = reading(pd.DataFrame(rows))
    ok("reading_rule", (not r["CL01_02_robust"]) and r["CL03_robust"] and len(ch) == 1)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--root", default=".")
    ap.add_argument("--only", default=None)
    ap.add_argument("--B", type=int, default=B_DEF)
    ap.add_argument("--out", default="S46")
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
        log(f"{'PASS' if cond else 'FAIL'}  {name}  {msg}")

    st = selftest(verbose=False)
    check("selftests", all(st.values()), f"({sum(st.values())}/{len(st)})")
    gp = R / "S1" / "yulara_geo_5min.parquet"
    geo = pd.read_parquet(gp)
    inputs = {str(gp.relative_to(R)): md5(gp)}
    jobs = [(y, h) for y in FOLDS for h in HORIZONS]
    if a.only:
        jobs = [(int(p.split(":")[0]), int(p.split(":")[1])) for p in a.only.split(",")]
    rows = []
    for yr, h in jobs:
        t1 = time.time()
        cp = R / "S2" / f"yulara_f{yr}_h{h}.parquet"
        inputs[str(cp.relative_to(R))] = md5(cp)
        cache = FC.load_cache(cp)
        f = K.fold(yr)
        cal, _ = K.fit_scalar(geo, f["train_max"])
        for name, ov in DEFS:
            g_it, g_tt = labels(geo, cal, h, cache.index, ov)
            if name == "base":
                check(f"base_labels_equal_cache_f{yr}_h{h}",
                      np.array_equal(g_it, cache["g"].values.astype(str))
                      and np.array_equal(g_tt, cache["g_t"].values.astype(str)))
            rows += evaluate(cache, g_it, g_tt, a.B, yr, h, name)
        log(f"fold {yr} h={h}: {time.time() - t1:.0f} s")
    cov = pd.DataFrame(rows)
    cov.to_csv(out / "coverage.csv", index=False)
    rd, ch = reading(cov)
    ch.to_csv(out / "status_changes.csv", index=False)
    json.dump(rd, open(out / "reading.json", "w"), indent=1)
    ok = all(c["ok"] for c in checks.values())
    now = time.strftime("%Y-%m-%dT%H:%M:%S")
    outs = {p.name: md5(p) for p in sorted(out.glob("*")) if p.is_file()}
    json.dump(dict(stage=a.out, ok=ok, error="" if ok else "checks failed", finished=now,
                   seconds=round(time.time() - t0, 1), inputs=inputs, outputs=outs,
                   code={"ce1_wp46.py": md5(Path(__file__))},
                   env=dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__)), open(runs / f"{a.out}.json", "w"), indent=2)
    json.dump(dict(stage=a.out, checks=checks, outputs=outs, inputs=inputs, code={"ce1_wp46.py": md5(Path(__file__))}),
              open(runs / f"{a.out}.detail.json", "w"), indent=1)
    log(f"{len(checks)} checks, {sum(not c['ok'] for c in checks.values())} failed; {time.time() - t0:.0f} s")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
