"""v1 cleaning of the raw Yulara files to a 5-min grid with quality flags. Its outputs are used only
by the cross-checks of stage S1 (ce1_preprocess.py); the v2 record is built from the raw files.

Gap filling: by default a gap is filled only when its whole run is at most 6 steps. With
--legacy-gapfill the first 6 steps of a gap of any length are filled, as in the v1 files the S1
cross-checks were run against. Every Yulara_<year>.csv in the raw folder is read.
"""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "utils"))
import config as C
import numpy as np, pandas as pd

LEGACY_GAPFILL = "--legacy-gapfill" in sys.argv

def log(m): print(f"[clean] {m}", flush=True)

# ---- 1-3. Load, parse, concat, dedupe ----
keep_raw = [C.TS] + list(C.COLS.values())
ren = {v: k for k, v in C.COLS.items()}
frames = []
for f in sorted(C.DATA_RAW.glob("Yulara_*.csv")):
    df = pd.read_csv(f, usecols=lambda c: c in keep_raw)
    df = df.rename(columns=ren)
    df["ts"] = pd.to_datetime(df[C.TS], errors="coerce")
    bad = df["ts"].isna().sum()
    if bad: log(f"{f.name}: dropped {bad} unparseable timestamps")
    df = df.dropna(subset=["ts"]).drop(columns=[C.TS])
    frames.append(df)
    log(f"loaded {f.name}: {len(df)} rows")

if not frames:
    raise SystemExit(
        "[clean] no input files found.\n"
        "        Looked for:  %s\n"
        "        The raw data are third-party and are not redistributed with this code.\n"
        "        Download them and place the per-year CSVs DIRECTLY in that directory,\n"
        "        named Yulara_<year>.csv (no nested subdirectory). See data/README.md."
        % (C.DATA_RAW / "Yulara_*.csv"))

raw = pd.concat(frames, ignore_index=True)
dup = raw.duplicated(subset="ts").sum()
raw = raw.sort_values("ts").drop_duplicates(subset="ts", keep="first").set_index("ts")
log(f"combined {len(raw)} unique rows (removed {dup} duplicate timestamps from file overlaps)")

# ---- 4. Regular 5-min grid ----
full_idx = pd.date_range(raw.index.min(), raw.index.max(), freq=C.FREQ)
grid = raw.reindex(full_idx)
grid.index.name = "ts"
n_inserted = len(grid) - len(raw)
log(f"reindexed to regular grid: {len(grid)} rows ({n_inserted} missing slots inserted)")

cols = list(C.COLS.keys())

# ---- 5. Sentinels + physical bounds -> NaN ----
sent_counts, bound_counts = {}, {}
for col in cols:
    s = grid[col]
    sent = (s.abs() >= C.SENTINEL_ABS)
    sent_counts[col] = int(sent.sum())
    s = s.mask(sent)
    lo, hi = C.BOUNDS[col]
    oob = pd.Series(False, index=s.index)
    if lo is not None: oob |= (s < lo)
    if hi is not None: oob |= (s > hi)
    bound_counts[col] = int(oob.sum())
    grid[col] = s.mask(oob)

# ---- 6. Missingness before interpolation ----
miss_before = grid[cols].isna().sum()

flags = pd.DataFrame(index=grid.index)
withheld_counts = {}
for col in cols:
    s = grid[col]
    isnan = s.isna()
    # run length of each NaN run, broadcast back onto every cell in that run
    runid = (isnan != isnan.shift()).cumsum()
    runlen = isnan.groupby(runid).transform("sum").where(isnan, 0)
    short_gap = isnan & (runlen <= C.SHORT_GAP_STEPS)
    interp = s.interpolate(method="time", limit=C.SHORT_GAP_STEPS,
                           limit_area="inside")
    if LEGACY_GAPFILL:
        filled = interp
    else:
        filled = s.where(~short_gap, interp)
    flags[col + "_imputed"] = isnan & filled.notna()
    # cells the old behaviour would have fabricated and this guard withholds
    would_withhold = int((isnan & interp.notna() & ~short_gap).sum())
    withheld_counts[col] = 0 if LEGACY_GAPFILL else would_withhold
    grid[col] = filled
miss_after = grid[cols].isna().sum()
imputed_counts = {c: int(flags[c + "_imputed"].sum()) for c in cols}
if LEGACY_GAPFILL:
    log("LEGACY GAP-FILL MODE: run-length guard DISABLED; gaps filled exactly as the "
        "shipped article artifacts were produced (first "
        f"{C.SHORT_GAP_STEPS} steps of any gap)")
log("run-length guard (SF-23/D-039a): withheld "
    + ", ".join(f"{c}={withheld_counts[c]}" for c in cols)
    + f" cells that lay inside NaN runs longer than {C.SHORT_GAP_STEPS} steps")

# ---- 7. Clip night negatives for irradiance/power ----
for col in ["ghi", "ghi_alt", "pv_total"]:
    neg = (grid[col] < 0)
    grid.loc[neg, col] = 0.0
log("clipped negative GHI/PV (night offset) to 0")

# ---- 8. Save ----
clean_path = C.DATA_CLEAN / "yulara_clean_5min.parquet"
flags_path = C.DATA_CLEAN / "yulara_quality_flags.parquet"
grid.to_parquet(clean_path)
flags.to_parquet(flags_path)
log(f"saved cleaned data -> {clean_path.name}")

# ---- Quality report (overall + per year) ----
n = len(grid)
overall = {
    "rows_total_grid": n,
    "span_start": str(grid.index.min()),
    "span_end": str(grid.index.max()),
    "duplicate_ts_removed": int(dup),
    "missing_slots_inserted": int(n_inserted),
    "gapfill_mode": "legacy" if LEGACY_GAPFILL else "guarded",
}
per_var = []
for col in cols:
    per_var.append(dict(
        variable=col,
        sentinels=sent_counts[col],
        out_of_bounds=bound_counts[col],
        missing_before_interp=int(miss_before[col]),
        pct_missing_before=round(miss_before[col] / n * 100, 3),
        interpolated=imputed_counts[col],
        withheld_by_runlength_guard=withheld_counts[col],
        missing_after_interp=int(miss_after[col]),
        pct_missing_after=round(miss_after[col] / n * 100, 3),
    ))
per_var_df = pd.DataFrame(per_var)
per_var_df.to_csv(C.TAB / "p1_cleaning_summary.csv", index=False)

# Per-year GHI completeness (post-clean, daytime proxy not yet available)
gy = grid["ghi"].notna().groupby(grid.index.year).mean().mul(100).round(2)
year_tbl = gy.rename("ghi_complete_pct_after").to_frame()
year_tbl.to_csv(C.TAB / "p1_ghi_completeness_by_year.csv")

with open(C.MET / "p1_cleaning_summary.json", "w") as fh:
    json.dump({"overall": overall, "per_variable": per_var,
               "ghi_complete_pct_by_year": gy.to_dict()}, fh, indent=2)

log("=== SUMMARY ===")
print(per_var_df.to_string(index=False))
print("\nGHI completeness after clean, by year (%):")
print(year_tbl.to_string())
print("\noverall:", json.dumps(overall, indent=2))
log("done.")
