#!/usr/bin/env python3
"""Record counts per site and period, and the fleet's N-1 firm capacity against the representative
load. Writes 04_results/ce1/WP72/.

Usage (from the repository root unless the path starts at ce1/, then from 03_code):
    python3 03_code/ce1/ce1_wp72_trace.py
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path.cwd()
R = ROOT / "04_results" / "ce1"
OUT = R / "WP72"
sys.path.insert(0, str(ROOT / "03_code" / "ce1"))
import ce1_load as LM  # noqa: E402

DAY_CS = 20.0


def record_counts():
    rows = []
    for site in ("yulara", "asp"):
        d = pd.read_parquet(R / "S1E" / f"{site}_geo_5min.parquet")
        d = d[d.index.year <= 2025]  # the 288 slots of 2026-01-01 are in no fold
        yrs = sorted(set(d.index.year))
        periods = [(str(y), d.index.year == y) for y in yrs]
        periods += [("to2024", d.index.year <= 2024), ("to2025", d.index.year <= 2025)]
        for lab, m in periods:
            x = d[m]
            day = x[x["ghi_cs_raw"] > DAY_CS]
            rows.append(dict(site=site, period=lab, slots=len(x), measured=int(x["ghi"].notna().sum()),
                             first=str(x.index.min()), last=str(x.index.max()),
                             daytime_steps=len(day), daytime_missing=int(day["ghi"].isna().sum()),
                             daytime_missing_share=float(day["ghi"].isna().mean())))
    return rows


def firm_capacity():
    fleet = np.asarray(LM.UNITS_BASE, float)
    firm = float(fleet.sum() - fleet.max())
    u = np.load(R / "S1L" / "load_profile.npz")["unit96"]
    sc = json.loads((R / "S1L" / "load_scales.json").read_text(encoding="utf-8"))["scales"]
    out = []
    for shape in ("A", "B"):
        unit = u if shape == "A" else LM.FLAT_WEIGHT * 1.0 + (1 - LM.FLAT_WEIGHT) * u
        for scale, v in sc[shape].items():
            load = unit * float(v["mean_kW"])
            above = load > firm
            out.append(dict(shape=shape, scale=scale, installed_kW=float(fleet.sum()), largest_kW=float(fleet.max()),
                            firm_kW=firm, peak_kW=float(load.max()), recorded_peak_kW=float(v["peak_kW"]),
                            slots15_above=int(above.sum()), hours_above=float(above.sum()) / 4.0,
                            days_above=int(above.any(axis=1).sum())))
    return out


def main():
    rc = record_counts()
    fc = firm_capacity()
    bad = []
    dn = json.loads((R / "WP21" / "data_numbers.json").read_text(encoding="utf-8"))
    for site, key in (("yulara", "yulara_grid"), ("asp", "asp_grid")):
        r = next(x for x in rc if x["site"] == site and x["period"] == "to2024")
        g = dn[key]
        if (r["slots"], r["measured"], r["first"], r["last"]) != (g["slots"], g["valid"], g["first"], g["last"]):
            bad.append(f"check 1 {site}: {r} vs {g}")
    with open(R / "WP59C" / "missing.csv", newline="", encoding="utf-8") as f:
        ref = {x["fold"]: x for x in csv.DictReader(f) if x["site"] == "yulara" and x["label"] == "all"}
    for y in range(2019, 2025):
        r = next(x for x in rc if x["site"] == "yulara" and x["period"] == str(y))
        g = ref[str(y)]
        if (r["daytime_steps"], r["daytime_missing"]) != (int(g["daytime_steps"]), int(g["missing"])):
            bad.append(f"check 2 {y}: {r['daytime_steps']},{r['daytime_missing']} vs {g['daytime_steps']},{g['missing']}")
    a = next(x for x in fc if x["shape"] == "A" and x["scale"] == "primary")
    if abs(a["peak_kW"] - a["recorded_peak_kW"]) >= 0.1:
        bad.append(f"check 3 {a['peak_kW']} vs {a['recorded_peak_kW']}")
    if bad:
        print("CHECK FAIL, nothing written:\n  " + "\n  ".join(bad))
        sys.exit(1)
    OUT.mkdir(parents=True, exist_ok=True)
    for name, rows in (("record_counts.csv", rc), ("firm_capacity.csv", fc)):
        with open(OUT / name, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    print("checks 1-3 PASS; wrote", ", ".join(str(OUT / n) for n in ("record_counts.csv", "firm_capacity.csv")))


if __name__ == "__main__":
    main()
