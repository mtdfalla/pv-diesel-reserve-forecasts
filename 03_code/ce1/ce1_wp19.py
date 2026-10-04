#!/usr/bin/env python3
"""Evidence tables for the D-071 reading (end-of-day battery energy) and for the A-R fuel saving
against the handbook rule (DR1). Writes 04_results/ce1/WP19/.

Usage (from the repository root unless the path starts at ce1/, then from 03_code):
    python3 03_code/ce1/ce1_wp19.py --root .
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

H = {1: 5, 3: 15, 6: 30, 12: 60}
QPOL = ("raw_q", "cqr", "mondrian_cqr")
VARIANTS = ("carry", "settle", "strict")


def md5(p: Path) -> str:
    return hashlib.md5(p.read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    a = ap.parse_args()
    R = Path(a.root) / "04_results" / "ce1"
    for s in ("S6P2", "S7E"):
        st = json.loads((R / "_runs" / f"{s}.json").read_text(encoding="utf-8-sig"))
        assert st.get("ok") and not st.get("smoke"), f"{s} has no completed run"
    out = R / "WP19"
    out.mkdir(exist_ok=True)
    inputs = {}

    rk_rows, th_rows = [], []
    for h, hm in H.items():
        f = R / "S6P2" / f"f2024_h{h}_proxy_rows.parquet"
        inputs[f.name] = md5(f)
        p = pd.read_parquet(f)
        A = p[(p.protocol == "A") & (p.ratio.astype(str) == "10.0") & (p.lam == 0.0)]
        D = A[A.batt == "default"]
        base = list(D[D["mode"] == "reset"].sort_values("vc_mean", ascending=False).policy)
        for m in VARIANTS:
            r = list(D[D["mode"] == m].sort_values("vc_mean", ascending=False).policy)
            rk_rows.append(dict(h_min=hm, variant=m, same_as_reset=r == base,
                                reset=" > ".join(base), variant_order=" > ".join(r)))
        if h == 1:
            T = A[A.param_kind == "theta"].pivot_table(index=["mode", "policy"], columns="batt", values="theta")
            for (m, pol), t in T.iterrows():
                th_rows.append(dict(mode=m, policy=pol, small=t["small"], default=t["default"], large=t["large"],
                                    nondecreasing=bool(t["small"] <= t["default"] <= t["large"])))
    pd.DataFrame(rk_rows).to_csv(out / "d071_ranking.csv", index=False)
    pd.DataFrame(th_rows).to_csv(out / "d071_theta.csv", index=False)

    f = R / "S7E" / "dr1.csv"
    inputs[f.name] = md5(f)
    d = pd.read_csv(f)
    d = d[(d.scenario == "main") & (d.protocol == "A_R") & d.policy.isin(QPOL)].copy()
    d["saving_pct"] = -d.d_fuel_vs_handbook / d.fuel_handbook_L * 100
    d[["fold", "h", "policy", "fuel_L", "fuel_handbook_L", "saving_pct", "passes"]].to_csv(
        out / "dr1_saving.csv", index=False)

    json.dump(dict(inputs=inputs, outputs={p.name: md5(p) for p in sorted(out.glob("*.csv"))}),
              open(out / "manifest.json", "w"), indent=1)
    print("WP19 evidence written:", sorted(x.name for x in out.glob("*")))


if __name__ == "__main__":
    main()
