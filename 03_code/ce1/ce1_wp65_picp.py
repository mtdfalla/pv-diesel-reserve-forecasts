"""Fold-2024 coverage (PICP) at full precision, recomputed from the row-level coverage flags of
stage S5. Writes 04_results/ce1/WP65/picp_unrounded.csv.

Usage (from the repository root unless the path starts at ce1/, then from 03_code):
    python3 03_code/ce1/ce1_wp65_picp.py [--root .]
"""
from __future__ import annotations

import argparse
import hashlib
import json
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

import numpy as np
import pandas as pd

HSTEP = {1: 5, 3: 15, 6: 30, 12: 60}
SCOPE = {"clear": ("reg_it", 0), "transitional": ("reg_it", 1), "cloudy": ("reg_it", 2),
         "tt_clear": ("reg_tt", 0), "tt_transitional": ("reg_tt", 1), "tt_cloudy": ("reg_tt", 2)}


def r4(x: float) -> float:
    return float(Decimal(repr(float(x))).quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--root", default=str(Path(__file__).resolve().parents[2]))
    root = Path(ap.parse_args().root); R = root / "04_results" / "ce1"
    stored = pd.read_csv(R / "S7/coverage_ci.csv")
    stored = stored[stored.fold == 2024]
    rows, src = [], {}
    for site in ("yulara", "asp"):
        for hs, hm in HSTEP.items():
            p = R / "S5" / f"{site}_f2024_h{hs}_obs.npz"
            src[str(p.relative_to(root))] = hashlib.md5(p.read_bytes()).hexdigest()
            d = np.load(p)
            for k in d.keys():
                if not k.startswith("cov__"):
                    continue
                meth, var, nom = k[5:].split("|")
                cov = d[k].astype(bool)
                for scope in ("all",) + tuple(SCOPE):
                    if scope == "all":
                        m = np.ones(cov.size, bool)
                    else:
                        key, code = SCOPE[scope]; m = d[key] == code
                    n = int(m.sum()); k_ = int(cov[m].sum())
                    rows.append({"site": site, "fold": 2024, "horizon_min": hm, "method": meth, "variant": var,
                                 "nominal": float(nom), "scope": scope, "n": n, "covered": k_, "picp": k_ / n if n else np.nan})
    out = pd.DataFrame(rows)
    key = ["site", "fold", "horizon_min", "method", "variant", "nominal", "scope"]
    j = stored.merge(out, on=key, how="left", suffixes=("", "_new"))
    fails = []
    if j.picp.isna().any():
        fails.append(f"{int(j.picp.isna().sum())} stored cells without a recomputed value")
    if not (j.n == j.n_new).all():
        fails.append(f"n differs in {int((j.n != j.n_new).sum())} cells")
    bad = j[[r4(a) != round(b, 4) for a, b in zip(j.picp, j.PICP)]]
    if len(bad):
        fails.append(f"PICP does not round to the stored value in {len(bad)} cells")
    tr = json.loads((R / "WP63/traces.json").read_text())["picp_same_class_cells"]["cells"]
    for c in tr:
        v = out[(out.site == "yulara") & (out.horizon_min == c["horizon_min"]) & (out.method == c["method"]) & (out.variant == "static")
                & (out.nominal == 0.9) & (out.scope == c["scope"])].picp
        if len(v) != 1 or abs(float(v.iloc[0]) - c["picp"]) > 1e-12:
            fails.append(f"trace cell {c} not reproduced")
    s4 = j[(j.nominal == 0.9) & j.scope.isin(["all", "clear", "transitional", "cloudy"])]
    flips = s4[[f"{a:.3f}" != f"{b:.3f}" for a, b in zip(s4.picp, s4.PICP)]]
    if fails:
        raise SystemExit("FAIL: " + "; ".join(fails))
    (R / "WP65").mkdir(exist_ok=True)
    out.to_csv(R / "WP65/picp_unrounded.csv", index=False)
    chk = {"cells": int(len(j)), "n_equal": True, "rounds_to_stored": True, "trace_cells_reproduced": len(tr),
           "s4_cells_90pct_regime_scopes": int(len(s4)), "s4_cells_changed_at_3_places": int(len(flips)),
           "changed": flips[key + ["PICP", "picp"]].to_dict("records"), "sources_md5": src, "ok": True}
    (R / "WP65/picp_checks.json").write_text(json.dumps(chk, indent=1, default=float))
    print(f"ok: {len(j)} stored cells reproduced; S4 90% cells {len(s4)}, {len(flips)} change at three places")
    for r_ in chk["changed"]:
        print("   ", r_)


if __name__ == "__main__":
    main()
