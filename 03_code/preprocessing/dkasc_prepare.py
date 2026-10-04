"""v1 cleaning and regime labels for DKASC Alice Springs. Used only by the cross-checks of stage S1.
"""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "utils"))
import config as C
import numpy as np, pandas as pd
import pvlib

RAW = C.DATA_DKASC / "raw"
CLEAN = C.DATA_DKASC / "cleaned"; CLEAN.mkdir(parents=True, exist_ok=True)
REG = C.DATA_DKASC / "regime_labels"; REG.mkdir(parents=True, exist_ok=True)

# Alice Springs DKASC (Desert Knowledge Precinct); ACST (UTC+9:30), same tz as Yulara
SITE = dict(name="DKASC_AliceSprings", latitude=-23.7621, longitude=133.8745, altitude=545, tz="Australia/Darwin")
W = "101_DKA_WeatherStation_"
COLS = {"ghi": W+"Global_Horizontal_Radiation", "ghi_diffuse": W+"Diffuse_Horizontal_Radiation",
        "temp_air": W+"Weather_Temperature_Celsius", "humidity": W+"Weather_Relative_Humidity",
        "wind_dir": W+"Wind_Direction", "rain_day": W+"Weather_Daily_Rainfall"}
def log(m): print(f"[dkasc] {m}", flush=True)

# ---- load + concat weather columns from all years ----
frames = []
for y in [2020, 2021, 2022, 2023, 2024]:
    f = RAW / f"Alice_Springs_{y}.csv"
    df = pd.read_csv(f, usecols=["timestamp"] + list(COLS.values()))
    frames.append(df); log(f"loaded {y}: {len(df)} rows")
raw = pd.concat(frames, ignore_index=True)
raw = raw.rename(columns={v: k for k, v in COLS.items()})
raw["timestamp"] = pd.to_datetime(raw["timestamp"])
raw = raw.drop_duplicates("timestamp").set_index("timestamp").sort_index()

# regular 5-min grid
full = pd.date_range(raw.index.min().floor("D"), raw.index.max().ceil("D"), freq="5min", inclusive="left")
df = raw.reindex(full)
log(f"reindexed to regular 5-min grid: {len(df)} rows ({df.index.min()} .. {df.index.max()})")

# clean: sentinels + plausibility bounds + night clip
SENT = 9e4
for c in df.columns:
    df.loc[df[c].abs() >= SENT, c] = np.nan
BOUNDS = {"ghi": (-50, 1600), "ghi_diffuse": (-50, 1600), "temp_air": (-15, 60),
          "humidity": (0, 100), "wind_dir": (0, 360), "rain_day": (0, 500)}
for c, (lo, hi) in BOUNDS.items():
    if c in df: df.loc[(df[c] < lo) | (df[c] > hi), c] = np.nan
# short-gap interpolation on GHI (<=30 min)
df["ghi_imputed"] = df["ghi"].isna()
df["ghi"] = df["ghi"].interpolate(limit=6, limit_area="inside")
df["ghi_imputed"] = df["ghi_imputed"] & df["ghi"].notna()
df.loc[df["ghi"] < 0, "ghi"] = 0.0
df.to_parquet(CLEAN / "asp_clean_5min.parquet")
log(f"saved cleaned: GHI nonnull={df['ghi'].notna().mean():.3f}, imputed={int(df['ghi_imputed'].sum())}")

# ---- clear-sky + regimes (replicate p3_clearsky_regimes) ----
KT_CLEAR, KT_OVERCAST, VAR_WIN, VAR_STABLE, CS_DAY_MIN = 0.85, 0.50, 12, 0.10, 20.0
loc = pvlib.location.Location(SITE["latitude"], SITE["longitude"], tz=SITE["tz"],
                              altitude=SITE["altitude"], name=SITE["name"])
idx = df.index.tz_localize(SITE["tz"], ambiguous="NaT", nonexistent="NaT")
ok = idx.notna(); df = df[ok]; idx = idx[ok]
meas = df["ghi"].values
solpos = loc.get_solarposition(idx); zenith = solpos["zenith"].values
cosz = np.cos(np.radians(zenith))
ghi_cs_raw = loc.get_clearsky(idx, model="ineichen")["ghi"].values
i0h = np.clip(pvlib.irradiance.get_extra_radiation(idx).values * cosz, 0, None)
day = ghi_cs_raw > CS_DAY_MIN

dd = pd.DataFrame({"meas": meas, "cs": ghi_cs_raw, "day": day}, index=df.index)
def daycorr(x):
    x = x[x.day]
    if len(x) < 20 or x["meas"].isna().all() or x["meas"].std() == 0: return np.nan
    return x["meas"].corr(x["cs"])
median_daycorr = float(dd.groupby(dd.index.date).apply(daycorr).median())
log(f"per-day corr median={median_daycorr:.3f}")

kt_raw = pd.Series(np.where(day, meas/np.where(ghi_cs_raw > 0, ghi_cs_raw, np.nan), np.nan), index=df.index)
kt_raw_var = kt_raw.rolling(VAR_WIN, min_periods=VAR_WIN//2).std()  # trailing (causal)
is_train = (df.index.year <= 2022)
band = day & kt_raw.between(0.85, 1.15).values & (kt_raw_var.values < 0.05) & is_train
cal = float(np.nanmedian(kt_raw.values[band]))
log(f"clear-day calibration factor = {cal:.4f} (n_band={int(band.sum())}, train-only)")

ghi_cs = ghi_cs_raw * cal
kt = np.clip(np.where(day, meas/np.where(ghi_cs > 0, ghi_cs, np.nan), np.nan), 0, 1.5)
kt_clearness = np.clip(np.where(i0h > CS_DAY_MIN, meas/np.where(i0h > 0, i0h, np.nan), np.nan), 0, 1.2)

out = pd.DataFrame(index=df.index)
out["ghi"] = meas; out["ghi_cs"] = ghi_cs; out["i0h"] = i0h; out["zenith"] = zenith
out["is_day"] = day; out["kt"] = kt; out["kt_clearness"] = kt_clearness
out["kt_var"] = pd.Series(kt, index=df.index).rolling(VAR_WIN, min_periods=VAR_WIN//2).std()  # trailing (causal)
out["ghi_imputed"] = df["ghi_imputed"].values
for c in ["temp_air", "humidity", "wind_dir", "rain_day", "ghi_diffuse"]:
    out[c] = df[c].values

ktv = out["kt_var"].values
reg = np.full(len(out), "night", dtype=object)
d = out["is_day"].values & ~np.isnan(kt)
trans = d & (ktv >= VAR_STABLE)
clear = d & ~trans & (kt >= KT_CLEAR)
cloudy = d & ~trans & (kt < KT_OVERCAST)
mid = d & ~trans & ~clear & ~cloudy
reg[trans] = "transitional"; reg[clear] = "clear"; reg[cloudy] = "cloudy"; reg[mid] = "transitional"
out["regime"] = reg
out.to_parquet(REG / "asp_regimes_5min.parquet")

reg_day = out.loc[out["is_day"], "regime"].value_counts()
reg_day_pct = (reg_day/reg_day.sum()*100).round(2)
align = dict(site=SITE["name"], clearsky_model="ineichen_clearband_calibrated_causal_trainonly", calibration_factor=round(cal, 4),
             median_per_day_corr=round(median_daycorr, 4), median_kt_clearband=round(float(np.nanmedian(kt[band])), 4),
             daytime_steps=int(out["is_day"].sum()), regime_daytime_pct=reg_day_pct.to_dict(),
             years="2020-2024", n_rows=int(len(out)))
json.dump(align, open(C.MET / "j3_asp_clearsky_alignment.json", "w"), indent=2)
log("=== ASP REGIME DISTRIBUTION (daytime) ===")
print(pd.DataFrame({"count": reg_day, "pct": reg_day_pct}).to_string())
print(json.dumps(align, indent=2))
log("done.")
