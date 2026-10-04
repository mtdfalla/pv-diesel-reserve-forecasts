"""v1 clear-sky index and weather-regime labels for Yulara. Used only by the cross-checks of stage S1.
"""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "utils"))
import config as C
import numpy as np, pandas as pd
import pvlib

def log(m): print(f"[regimes] {m}", flush=True)

# Regime thresholds (data-driven, documented)
KT_CLEAR    = 0.85
KT_OVERCAST = 0.50
VAR_WIN     = 12  # 1 hour
VAR_STABLE  = 0.10
CS_DAY_MIN  = 20.0

df = pd.read_parquet(C.DATA_CLEAN / "yulara_clean_5min.parquet")
log(f"loaded cleaned data: {len(df)} rows")

loc = pvlib.location.Location(C.SITE["latitude"], C.SITE["longitude"],
                              tz=C.SITE["tz"], altitude=C.SITE["altitude"], name=C.SITE["name"])
idx = df.index.tz_localize(C.SITE["tz"], ambiguous="NaT", nonexistent="NaT")
ok = idx.notna(); df = df[ok]; idx = idx[ok]
meas = df["ghi"].values

solpos = loc.get_solarposition(idx)
zenith = solpos["zenith"].values
cosz = np.cos(np.radians(zenith))
ghi_cs_raw = loc.get_clearsky(idx, model="ineichen")["ghi"].values
i0h = np.clip(pvlib.irradiance.get_extra_radiation(idx).values * cosz, 0, None)
day = ghi_cs_raw > CS_DAY_MIN

# alignment via per-day correlation
dd = pd.DataFrame({"meas": meas, "cs": ghi_cs_raw, "day": day}, index=df.index)
def daycorr(x):
    x = x[x.day]
    if len(x) < 20 or x["meas"].isna().all() or x["meas"].std() == 0: return np.nan
    return x["meas"].corr(x["cs"])
percorr = dd.groupby(dd.index.date).apply(daycorr)
median_daycorr = float(percorr.median())
log(f"per-day corr median={median_daycorr:.3f}")
assert median_daycorr > 0.90, "alignment failed"

TRAIN_MAX_YEAR = 2022
kt_raw = pd.Series(np.where(day, meas/np.where(ghi_cs_raw>0,ghi_cs_raw,np.nan), np.nan), index=df.index)
kt_raw_var = kt_raw.rolling(VAR_WIN, min_periods=VAR_WIN//2).std()  # trailing (causal)
is_train = (df.index.year <= TRAIN_MAX_YEAR)
band = day & kt_raw.between(0.85,1.15).values & (kt_raw_var.values < 0.05) & is_train
cal = float(np.nanmedian(kt_raw.values[band]))
log(f"clear-day calibration factor = {cal:.4f} (n_band={int(band.sum())}, train-only)")

ghi_cs = ghi_cs_raw * cal
kt = np.clip(np.where(day, meas/np.where(ghi_cs>0,ghi_cs,np.nan), np.nan), 0, 1.5)
kt_clearness = np.clip(np.where(i0h>CS_DAY_MIN, meas/np.where(i0h>0,i0h,np.nan), np.nan), 0, 1.2)

out = pd.DataFrame(index=df.index)
out["ghi_cs"]=ghi_cs; out["ghi_cs_raw"]=ghi_cs_raw; out["i0h"]=i0h
out["zenith"]=zenith; out["is_day"]=day; out["kt"]=kt; out["kt_clearness"]=kt_clearness
out["kt_var"]=pd.Series(kt,index=df.index).rolling(VAR_WIN,min_periods=VAR_WIN//2).std()  # trailing (causal)
log(f"median kt on clear-band = {np.nanmedian(kt[band]):.3f} (target ~1.0)")

# per-timestep regime
ktv=out["kt_var"].values
reg=np.full(len(out),"night",dtype=object)
d=out["is_day"].values & ~np.isnan(kt)
trans=d & (ktv>=VAR_STABLE)
clear=d & ~trans & (kt>=KT_CLEAR)
cloudy=d & ~trans & (kt<KT_OVERCAST)
mid=d & ~trans & ~clear & ~cloudy
reg[trans]="transitional"; reg[clear]="clear"; reg[cloudy]="cloudy"; reg[mid]="transitional"
out["regime"]=reg

# daily class
dsub=out.loc[out["is_day"]]
dly=dsub.groupby(dsub.index.date)["kt"].agg(["mean","std"])
def dc(r):
    if r["mean"]>=KT_CLEAR and r["std"]<0.12: return "clear"
    if r["mean"]<KT_OVERCAST: return "overcast"
    return "mixed"
dly["day_class"]=dly.apply(dc,axis=1)
m=dly["day_class"].to_dict()
out["day_class"]=[m.get(t.date()) for t in out.index]

out.to_parquet(C.DATA_REGIME/"yulara_regimes_5min.parquet")
log("saved regimes parquet")

reg_day=out.loc[out["is_day"],"regime"].value_counts()
reg_day_pct=(reg_day/reg_day.sum()*100).round(2)
pd.DataFrame({"count_daytime":reg_day,"pct_of_daytime":reg_day_pct}).to_csv(C.TAB/"p1_regime_distribution.csv")
dayclass=dly["day_class"].value_counts()
align=dict(clearsky_model="ineichen_clearband_calibrated_causal_trainonly", calibration_factor=round(cal,4),
           median_per_day_corr=round(median_daycorr,4),
           median_kt_clearband=round(float(np.nanmedian(kt[band])),4),
           thresholds=dict(KT_CLEAR=KT_CLEAR,KT_OVERCAST=KT_OVERCAST,VAR_WIN_steps=VAR_WIN,
                           VAR_STABLE=VAR_STABLE,CS_DAY_MIN=CS_DAY_MIN),
           daytime_steps=int(out["is_day"].sum()),
           regime_daytime_pct=reg_day_pct.to_dict(),
           day_class_counts={k:int(v) for k,v in dayclass.items()}, n_days=int(len(dly)))
with open(C.MET/"p1_clearsky_alignment.json","w") as fh: json.dump(align,fh,indent=2)
log("=== REGIME DISTRIBUTION (daytime) ===")
print(pd.DataFrame({"count_daytime":reg_day,"pct_of_daytime":reg_day_pct}).to_string())
print("\nDaily class:", {k:int(v) for k,v in dayclass.items()})
print(json.dumps(align,indent=2))
log("done.")
