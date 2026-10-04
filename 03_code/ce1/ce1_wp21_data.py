#!/usr/bin/env python3
"""Data-description numbers (grid slots, measured slots, first-hour share, regime shares) from the
S1 and S2 outputs. Writes 04_results/ce1/WP21/data_numbers.json.

Usage (from the repository root unless the path starts at ce1/, then from 03_code):
    python3 03_code/ce1/ce1_wp21_data.py --root .
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

SITES = {"yulara": (-25.2406, 130.9889), "asp": (-23.7621, 133.8745)}  # ce1_causal.SITES
YFOLDS = [2019, 2020, 2021, 2022, 2023, 2024]


def md5(p: Path) -> str:
    return hashlib.md5(p.read_bytes()).hexdigest()


def haversine(a, b, r=6371.0):
    la1, lo1, la2, lo2 = map(math.radians, (*a, *b))
    x = math.sin((la2 - la1) / 2) ** 2 + math.cos(la1) * math.cos(la2) * math.sin((lo2 - lo1) / 2) ** 2
    return 2 * r * math.asin(math.sqrt(x))


def grid_numbers(geo: pd.DataFrame) -> dict:
    day = geo["is_day"].astype(bool)
    valid = geo["ghi"].notna()
    g = geo.loc[day, "ghi"]
    rank = day[day].groupby(day[day].index.date).cumcount()  # 0-based daytime slot of the day
    first = (rank < 12).reindex(g.index).values
    e_all = float(g.fillna(0).sum())
    e_first = float(g[first].fillna(0).sum())
    return dict(slots=int(len(geo)), valid=int(valid.sum()), day_slots=int(day.sum()),
                day_valid=int((day & valid).sum()), first=str(geo.index.min()), last=str(geo.index.max()),
                first_hour_share=e_first / e_all)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    a = ap.parse_args()
    R = Path(a.root) / "04_results" / "ce1"
    out, inputs = {}, []

    for site in ("yulara", "asp"):
        p = R / "S1" / f"{site}_geo_5min.parquet"
        inputs.append(p)
        out[f"{site}_grid"] = grid_numbers(pd.read_parquet(p))

    def shares(files):
        g, vnan = [], 0
        for p in files:
            inputs.append(p)
            c = pd.read_parquet(p, columns=["split", "g", "v"])
            vnan += int(c["v"].isna().sum())
            g.append(c.loc[c["split"] == "test", "g"])
        s = pd.concat(g).value_counts(normalize=True)
        return {k: float(s.get(k, 0.0)) for k in ("clear", "transitional", "cloudy")}, int(sum(len(x) for x in g)), vnan

    ys, yn, yv = shares([R / "S2" / f"yulara_f{Y}_h3.parquet" for Y in YFOLDS])
    as_, an, av = shares([R / "S2" / "asp_f2024_h3.parquet"])
    out["regime_share_test_15min"] = dict(yulara=ys, yulara_n=yn, asp=as_, asp_n=an)
    out["v_undefined_rows"] = dict(yulara_h3_all_folds=yv, asp_h3=av)

    rc = list(csv.DictReader(open(R / "S1" / "retained_counts.csv", newline="")))
    inputs.append(R / "S1" / "retained_counts.csv")
    car = {}
    for r in rc:
        if r["site"] == "yulara" and r["fold_test"] == "2024" and r["frame"] == "ce1_locf6":
            car[r["h_min"]] = float(r["test_carried"]) / float(r["test"])
    out["carried_share_yulara_f2024_test"] = car

    sc = json.load(open(R / "S1" / "scalars.json"))
    inputs.append(R / "S1" / "scalars.json")
    v = [sc["yulara"][str(Y)]["scalar"] for Y in YFOLDS]
    out["scalar_range_yulara"] = [min(v), max(v)]

    js = json.load(open(R / "S2" / "yulara_f2024_h3.json"))
    inputs.append(R / "S2" / "yulara_f2024_h3.json")
    out["n_taus"] = len([k for k in js["iterations"] if k.startswith("q")])
    out["es_block_yulara_f2024_h3"] = dict(n_train=js["n_train"], n_val=js["n_val"], val_first=js["val_first"],
                                           val_share=js["n_val"] / js["n_train"])
    out["distance_km"] = haversine(SITES["yulara"], SITES["asp"])

    # invariants
    assert out["v_undefined_rows"]["yulara_h3_all_folds"] == 0 and out["v_undefined_rows"]["asp_h3"] == 0
    assert abs(sum(ys.values()) - 1) < 1e-12 and abs(sum(as_.values()) - 1) < 1e-12
    assert out["yulara_grid"]["slots"] == 920451 and out["asp_grid"]["slots"] == 526176  # S1 checks
    out["inputs_md5"] = {str(p.relative_to(R)): md5(p) for p in inputs}
    (R / "WP21").mkdir(exist_ok=True)
    (R / "WP21" / "data_numbers.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != "inputs_md5"}, indent=1))


if __name__ == "__main__":
    main()
