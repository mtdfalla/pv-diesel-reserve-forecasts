#!/usr/bin/env python3
"""Storage as reserve (rule D-099): diesel capacity and fuel released by the battery, and the
two-by-two decomposition of battery and forecast against the handbook rule, from stage S6E.

Usage (from the repository root unless the path starts at ce1/, then from 03_code):
    python3 03_code/ce1/ce1_wp52.py --root .
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

H = {1: 5, 3: 15, 6: 30, 12: 60}
FOLDS = range(2019, 2025)
QPOL = ("raw_q", "cqr", "mondrian_cqr")
SHAPES = {"A": ("main", "battery"), "B": ("shapeB", "batteryB")}
TARGET = 0.002 / 100


def md5(p: Path) -> str:
    return hashlib.md5(p.read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    a = ap.parse_args()
    R = Path(a.root) / "04_results" / "ce1"
    st = json.loads((R / "_runs" / "S6E.json").read_text(encoding="utf-8-sig"))
    assert st.get("ok") and not st.get("smoke"), "S6E has no completed run"
    out = R / "WP52"
    out.mkdir(exist_ok=True)
    checks, cells, missing, mism = {}, [], 0, 0
    inputs = {}
    for y in FOLDS:
        for h, hm in H.items():
            tf, sf = R / "S6E" / f"f{y}_h{h}_totals.parquet", R / "S6E" / f"f{y}_h{h}_select.json"
            inputs[tf.name], inputs[sf.name] = md5(tf), md5(sf)
            T = pd.read_parquet(tf)
            T = T[T.set == "test"]
            S = json.load(open(sf))["selections"]

            def outcome(scen, pol):
                nonlocal missing, mism
                if pol == "handbook":
                    r = T[(T.scenario == scen) & (T.policy == "handbook")]
                    assert len(r) == 1
                    r = r.iloc[0]
                    return dict(param=np.nan, fuel=r.fuel_L, online=r.online_kW_h, ue=r.ue_share)
                ar = S[scen]["policies"].get(pol, {}).get("A_R")
                if ar is None or ar.get("param") is None:
                    missing += 1
                    return None
                r = T[(T.scenario == scen) & (T.policy == pol) & np.isclose(T.param.astype(float), float(ar["param"]))]
                if len(r) != 1 or not np.isclose(r.iloc[0].fuel_L, ar["test"]["fuel_L"], rtol=1e-12):
                    mism += 1
                r = r.iloc[0]
                return dict(param=float(ar["param"]), fuel=r.fuel_L, online=r.online_kW_h, ue=r.ue_share)

            for shp, (nb, wb) in SHAPES.items():
                hb_nb, hb_wb = outcome(nb, "handbook"), outcome(wb, "handbook")
                F0 = hb_nb["fuel"]
                for pol in ("handbook",) + QPOL:
                    o_nb, o_wb = outcome(nb, pol), outcome(wb, pol)
                    if o_nb is None or o_wb is None:
                        continue
                    batt_alone = (F0 - hb_wb["fuel"]) / F0 * 100
                    fc_alone = (F0 - o_nb["fuel"]) / F0 * 100
                    combined = (F0 - o_wb["fuel"]) / F0 * 100
                    cells.append(dict(
                        fold=y, h_min=hm, shape=shp, policy=pol, theta_nb=o_nb["param"], theta_wb=o_wb["param"],
                        online_released_pct=(o_nb["online"] - o_wb["online"]) / o_nb["online"] * 100,
                        fuel_released_pct=(o_nb["fuel"] - o_wb["fuel"]) / F0 * 100,
                        battery_alone_pct=batt_alone, forecast_alone_pct=fc_alone, combined_pct=combined,
                        interaction_pct=combined - batt_alone - fc_alone,
                        ue_share_nb=o_nb["ue"], ue_share_wb=o_wb["ue"],
                        target_met_nb=bool(o_nb["ue"] <= TARGET), target_met_wb=bool(o_wb["ue"] <= TARGET)))
    C = pd.DataFrame(cells)
    C.to_csv(out / "cells.csv", index=False)
    ident = (C.combined_pct - (C.battery_alone_pct + C.forecast_alone_pct + C.interaction_pct)).abs().max()
    _st = json.load(open(R / "_runs" / "S6E.json"))
    checks["S6E_completed"] = dict(ok=bool(_st.get("ok")) and not _st.get("smoke"))
    checks["every_cell_has_A_R"] = dict(ok=missing == 0, msg=f"{missing} missing")
    checks["A_R_outcome_equals_totals_row"] = dict(ok=mism == 0, msg=f"{mism} mismatches")
    checks["decomposition_identity [definition identity: interaction is defined as the remainder]"] = dict(ok=bool(ident < 1e-9), msg=f"max {ident:.1e}")
    checks["cell_count"] = dict(ok=len(C) == 6 * 4 * 2 * 4, msg=f"{len(C)} of 192")

    rows = []
    q = ["online_released_pct", "fuel_released_pct", "battery_alone_pct", "forecast_alone_pct",
         "combined_pct", "interaction_pct"]
    for (hm, shp), g in C.groupby(["h_min", "shape"]):
        for pol, gg in list(g.groupby("policy")) + [("quantile_pooled", g[g.policy.isin(QPOL)])]:
            r = dict(h_min=hm, shape=shp, policy=pol, n=len(gg),
                     target_met_wb=int(gg.target_met_wb.sum()), target_met_nb=int(gg.target_met_nb.sum()))
            for k in q:
                if pol == "quantile_pooled":
                    med = gg.groupby("fold")[k].median().median()  # median over folds of the per-fold policy median
                else:
                    med = gg[k].median()
                r[f"{k}_median"], r[f"{k}_min"], r[f"{k}_max"] = med, gg[k].min(), gg[k].max()
            rows.append(r)
    pd.DataFrame(rows).to_csv(out / "summary.csv", index=False)
    ok = all(c["ok"] for c in checks.values())
    json.dump(dict(ok=ok, checks=checks, inputs=inputs,
                   outputs={p.name: md5(p) for p in sorted(out.glob("*.csv"))}), open(out / "checks.json", "w"), indent=1)
    for k, v in checks.items():
        print(("PASS " if v["ok"] else "FAIL ") + k, v.get("msg", ""))
    print("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED")


if __name__ == "__main__":
    main()
