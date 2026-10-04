"""Step changes of the mapped PV signal against the measured pooled output of the three
uncurtailed arrays, 2023 and 2024.

Usage (from the repository root unless the path starts at ce1/, then from 03_code):
    python3 03_code/ce1/ce1_wp65_ramps.py [--selftest]
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
ROOT = HERE.parents[1]
OUT = ROOT / "04_results" / "ce1" / "WP65"
SEL = ("LD", "SS", "SD")
ELEV_MIN = 5.0


def changes(x: pd.Series, lag: int, ok: pd.Series) -> np.ndarray:
    """x_t - x_{t-lag} on a regular 5-min grid, kept where ok holds at t and t-lag and x exists."""
    idx = x.index
    assert idx.is_monotonic_increasing and (pd.Series(idx).diff().iloc[1:] == pd.Timedelta("5min")).all(), "grid must be regular 5-min"
    d = x - x.shift(lag)
    good = ok & ok.shift(lag, fill_value=False) & x.notna() & x.shift(lag).notna()
    return d[good].to_numpy()


def stats(d: np.ndarray) -> dict:
    fall = -d
    return dict(n=int(d.size), sd=float(np.std(d, ddof=1)), fall_p99=float(np.quantile(fall, 0.99)),
                fall_p999=float(np.quantile(fall, 0.999)), share_fall_gt10=float(np.mean(fall > 0.10)),
                share_fall_gt20=float(np.mean(fall > 0.20)))


def selftest():
    i = pd.date_range("2023-01-01", periods=6, freq="5min")
    x = pd.Series([0.0, 0.1, np.nan, 0.4, 0.2, 0.5], index=i)
    ok = pd.Series(True, index=i)
    assert np.allclose(changes(x, 1, ok), [0.1, -0.2, 0.3])
    assert np.allclose(changes(x, 3, ok), [0.4, 0.1])  # (0.0 -> 0.4), (0.1 -> 0.2); the NaN pair dropped
    ok2 = ok.copy(); ok2.iloc[4] = False
    assert np.allclose(changes(x, 1, ok2), [0.1])
    s = stats(np.array([-0.3, -0.05, 0.0, 0.1]))
    assert s["share_fall_gt10"] == 0.25 and s["share_fall_gt20"] == 0.25 and s["n"] == 4
    print("selftest ok")


def main():
    import ce1_pvmap as PM
    import pvlib
    d = PM.load_years()  # raw power channels + horizontal GHI record
    m = PM.load_map(ROOT / "04_results" / "ce1" / "S1M" / "available_pv_map.npz")
    full = pd.date_range(d.index.min().floor("D"), d.index.max().ceil("D"), freq="5min", inclusive="left")
    d = d.reindex(full)
    rated = sum(PM.RATED[a] for a in SEL)
    assert abs(rated - 661.1) < 1e-6, rated
    sig = pd.DataFrame(index=full)
    ghi = d["ghi"].to_numpy()
    mapped = np.full(len(full), np.nan)
    k = np.isfinite(ghi)
    mapped[k] = PM.pv_available_from_ghi(ghi[k], full[k], m) / PM.RATED_TOTAL
    sig["mapped"] = mapped
    sig["measured"] = d[list(SEL)].sum(axis=1, min_count=len(SEL)) / rated
    for a in SEL:
        sig[a] = d[a] / PM.RATED[a]
    sp = pvlib.solarposition.get_solarposition(full.tz_localize(PM.SITE["tz"]), PM.SITE["lat"], PM.SITE["lon"],
                                               altitude=PM.SITE["alt"])
    up = pd.Series(sp["apparent_elevation"].to_numpy() > ELEV_MIN, index=full)
    # a pair is scored only where BOTH main signals exist, so the two distributions share their steps
    both = up & sig["mapped"].notna() & sig["measured"].notna()
    rows = []
    for year in (2023, 2024, None):
        yr = pd.Series(True, index=full) if year is None else pd.Series(full.year == year, index=full)
        for lag, h in ((1, 5), (3, 15)):
            ok = both & yr
            for name in ["mapped", "measured"] + list(SEL):
                okn = ok & sig[name].notna()
                rows.append(dict(year="pooled" if year is None else year, horizon_min=h, signal=name,
                                 **stats(changes(sig[name], lag, okn))))
    r = pd.DataFrame(rows)
    verdict = {}
    for h in (5, 15):
        cmp = []
        for y in (2023, 2024):
            a = r[(r.year == y) & (r.horizon_min == h)].set_index("signal")
            cmp.append((a.loc["mapped", "fall_p999"] > a.loc["measured", "fall_p999"],
                        a.loc["mapped", "share_fall_gt10"] > a.loc["measured", "share_fall_gt10"]))
        flat = [v for c in cmp for v in c]
        verdict[h] = "overstates" if all(flat) else "understates" if not any(flat) else "mixed"
    # checks
    n5 = r[(r.horizon_min == 5) & (r.signal.isin(["mapped", "measured"]))].groupby("year").n.nunique()
    assert (n5 == 1).all(), "mapped and measured must share the scored steps"
    OUT.mkdir(parents=True, exist_ok=True)
    r.to_csv(OUT / "ramps.csv", index=False, float_format="%.6g")
    src = {p: hashlib.md5((ROOT / p).read_bytes()).hexdigest() for p in
           ("02_data/Original Dataset/Yulara_2023.csv", "02_data/Original Dataset/Yulara_2024.csv",
            "04_results/ce1/S1M/available_pv_map.npz")}
    js = dict(rule="docs/REGISTERED_RULES.md (written before any number)", arrays=list(SEL),
              rated_kW=rated, elevation_min_deg=ELEV_MIN, verdict=verdict, sources=src,
              script_md5=hashlib.md5(Path(__file__).read_bytes()).hexdigest())
    (OUT / "ramps.json").write_text(json.dumps(js, indent=1))
    pd.set_option("display.width", 200)
    print(r.to_string(float_format=lambda v: "%.4f" % v))
    print("verdict", verdict)


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        selftest()
    else:
        selftest()
        main()
