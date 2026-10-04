#!/usr/bin/env python3
"""Coverage check C3 of the held-back fold 2025 and the DKASC one-step-lag reading (rules D-129,
D-132).

Usage (from the repository root unless the path starts at ce1/, then from 03_code):
    python3 03_code/ce1/ce1_wp67_c3lag.py
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "utils"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import ce1_stats as ST  # noqa: E402

NOM = 0.9


def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def cls(lo, hi, nom=NOM):
    return "below" if hi < nom else ("above" if lo > nom else "includes")


def trans_cov(z, B=ST.B_DEF):
    """{(method, variant): (PICP full precision, lo, hi, n)} at 0.9, transitional issue-time regime."""
    _, rc = ST.interval_stats(z, "x", 0, 5, B)
    out = {}
    ti = ST.REGIMES.index("transitional")
    mask = z["reg_it"] == ti
    for r in rc:
        if r["scope"] == "transitional" and abs(r["nominal"] - NOM) < 1e-12:
            v = z[f"cov__{r['method']}|{r['variant']}|0.9"].astype(float)[mask]
            out[(r["method"], r["variant"])] = (float(v.mean()), r["PICP_ci_lo"], r["PICP_ci_hi"], int(mask.sum()), r["PICP"])
    return out


def main(root="."):
    R = Path(root) / "04_results" / "ce1"
    out = R / "WP67" / "c3lag"
    out.mkdir(parents=True, exist_ok=True)
    checks, inputs = {}, {}

    def check(name, cond, msg=""):
        checks[name] = dict(ok=bool(cond), msg=str(msg))
        print(f"{'PASS' if cond else 'FAIL'}  {name}  {msg}", flush=True)

    # ---------------------------------------------------------------- lag: reproduction first
    s7 = pd.read_csv(R / "S7" / "coverage_ci.csv")
    inputs["S7/coverage_ci.csv"] = md5(R / "S7" / "coverage_ci.csv")
    base, lag = {}, {}
    for h in (1, 3):
        pb, pl = R / "S5" / f"asp_f2024_h{h}_obs.npz", R / "S5E" / "lag" / f"asp_f2024_h{h}_obs.npz"
        inputs[str(pb.relative_to(R))], inputs[str(pl.relative_to(R))] = md5(pb), md5(pl)
        base[h] = trans_cov(np.load(pb))
        ref = s7[(s7.site == "asp") & (s7.fold == 2024) & (s7.horizon_min == 5 * h) & (s7.scope == "transitional")
                 & np.isclose(s7.nominal, NOM)]
        bad = []
        for _, r in ref.iterrows():
            b = base[h].get((r.method, r.variant))
            if b is None or abs(b[4] - r.PICP) > 1e-12 or abs(b[1] - r.PICP_ci_lo) > 1e-12 or abs(b[2] - r.PICP_ci_hi) > 1e-12:
                bad.append((r.method, r.variant))
        check(f"asp_f2024_h{h}_recomputed_equals_S7", len(ref) > 0 and not bad and len(ref) == len(base[h]),
              f"({len(ref)} S7 rows, {len(base[h])} recomputed, mismatches {bad})")
    if not all(c["ok"] for c in checks.values()):
        json.dump(dict(ok=False, checks=checks), open(out / "verdict.json", "w"), indent=1)
        return False
    rows = []
    for h in (1, 3):
        lag[h] = trans_cov(np.load(R / "S5E" / "lag" / f"asp_f2024_h{h}_obs.npz"))
        common = sorted(set(base[h]) & set(lag[h]))
        check(f"lag_h{h}_same_method_set", set(base[h]) == set(lag[h]), f"({len(common)} methods)")
        for k in common:
            b, l_ = base[h][k], lag[h][k]
            rows.append(dict(horizon_min=5 * h, method=k[0], variant=k[1], n_base=b[3], picp_base=b[0], lo_base=b[1],
                             hi_base=b[2], class_base=cls(b[1], b[2]), n_lag=l_[3], picp_lag=l_[0], lo_lag=l_[1],
                             hi_lag=l_[2], class_lag=cls(l_[1], l_[2])))
    L = pd.DataFrame(rows)
    L["class_changes"] = L.class_base != L.class_lag
    L.to_csv(out / "lag_classes.csv", index=False)
    lag_fires = bool(L.class_changes.any())
    # ---------------------------------------------------------------- C3
    c3 = []
    for site in ("yulara", "asp"):
        p = R / "S5E" / "f2025" / f"{site}_f2025_h1_obs.npz"
        inputs[str(p.relative_to(R))] = md5(p)
        tc = trans_cov(np.load(p))
        for meth, part, test in (("icp", "a", lambda x: x < 0.85), ("mondrian_cqr", "b", lambda x: 0.875 <= x <= 0.925)):
            v = tc[(meth, "static")]
            c3.append(dict(site=site, part=part, method=meth, n=v[3], picp=v[0], picp_4dp=v[4], ci_lo=v[1], ci_hi=v[2],
                           class_1day=cls(v[1], v[2]), holds=bool(test(v[0]))))
        iv = pd.read_csv(R / "S5E" / "f2025" / "intervals.csv")
        for meth in ("icp", "mondrian_cqr"):
            r = iv[(iv.site == site) & (iv.horizon_min == 5) & (iv.method == meth) & (iv.variant == "static")
                   & np.isclose(iv.nominal, NOM) & (iv.scope == "transitional")]
            v = tc[(meth, "static")]
            check(f"c3_{site}_{meth}_equals_S5E_intervals", len(r) == 1 and abs(float(r.PICP.iloc[0]) - v[0]) < 5e-5
                  and int(r.n.iloc[0]) == v[3], (None if len(r) != 1 else float(r.PICP.iloc[0]), v[0]))
    C3 = pd.DataFrame(c3)
    C3.to_csv(out / "c3.csv", index=False)
    ok = all(c["ok"] for c in checks.values())
    verdict = dict(ok=ok,
                   lag=dict(rule="D-129 step 3 item 3", methods=int(len(L) / 2), fires=lag_fires,
                            changes=L[L.class_changes][["horizon_min", "method", "variant", "class_base", "class_lag"]].to_dict("records")),
                   c3={f"{r.site}_{r.part}": dict(method=r.method, picp=r.picp, holds=r.holds) for r in C3.itertuples()},
                   checks=checks, inputs=inputs, code={"ce1_wp67_c3lag.py": md5(Path(__file__)), "ce1_stats.py": md5(HERE / "ce1_stats.py")})
    json.dump(verdict, open(out / "verdict.json", "w"), indent=1, default=float)
    print(L.to_string(index=False))
    print(C3.to_string(index=False))
    print(json.dumps({k: verdict[k] for k in ("lag", "c3")}, indent=1, default=float))
    return ok


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
