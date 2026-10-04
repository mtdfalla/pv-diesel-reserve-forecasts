"""Orientation of the two Yulara irradiance sensors: fits the plane (tilt, azimuth) of each
channel on clear-sky segments with a Hay-Davies transposition. Inputs: per-year extracts of
the timestamp, irradiance and power columns of the raw Yulara files (folder SRC) and the stage S1
geometry record (GEO). The analysis steps are separate blocks, each guarded by `if False:`; set a
block's guard to True to run it, after the load and preparation blocks.
"""
SRC = "02_data/interim/sensor_columns"
GEO = "04_results/ce1/S1/yulara_geo_5min.parquet"

# ---------------------------------------------------------------- load (load.py)
import pandas as pd, numpy as np, glob, re

MAP={"8":"CA","5":"DG","11":"LD","7":"SD1","9":"SD2","6":"SD3","10":"SS","3050":"TOTAL","3051":"SD_TOTAL"}
RATED={"DG":1058.4,"LD":327.6,"SS":226.8,"CA":105.9,"SD1":np.nan,"SD2":np.nan,"SD3":np.nan,"SD":106.7}
def load():
    fr=[]
    for y in range(2016,2025):
        d=pd.read_csv(f"{SRC}/yulara_{y}_pvcols.csv.gz")
        ren={}
        for c in d.columns:
            if c=="timestamp": ren[c]="ts"; continue
            if c.endswith("Pyranometer_1"): ren[c]="ghi"; continue
            if c.endswith("Global_Horizontal_Radiation"): ren[c]="ghi_alt"; continue
            if c.endswith("Temperature_Celsius"): ren[c]="tamb"; continue
            ren[c]=MAP[c.split("_")[0]]
        d=d.rename(columns=ren); d["ts"]=pd.to_datetime(d["ts"],errors="coerce"); d["file_year"]=y
        fr.append(d)
    d=pd.concat(fr,ignore_index=True).dropna(subset=["ts"])
    d=d.sort_values("ts",kind="stable").drop_duplicates("ts",keep="first").set_index("ts")
    num=d.columns.drop("file_year")
    d[num]=d[num].where(d[num].abs()<9e4)
    d["SD"]=d[["SD1","SD2","SD3"]].sum(axis=1,min_count=3)
    return d
if False:
    d=load(); d.to_pickle("yul.pkl"); print(d.shape); print(d.describe().T.round(1))
    print(d.index.to_series().diff().value_counts().head())

# ---------------------------------------------------------------- orientation fit (tilt.py)
import pandas as pd, numpy as np, pvlib, warnings; warnings.filterwarnings("ignore")
g=pd.read_parquet(GEO)
d=load(); d=d.where(d.abs()<8e4)
g=g.join(d[['ghi_alt']],how='left')
lat,lon,alt=-25.2406,130.9889,492
t=g.index.tz_localize('Australia/Darwin')
sp=pvlib.solarposition.get_solarposition(t,lat,lon,alt)
loc=pvlib.location.Location(lat,lon,'Australia/Darwin',alt)
cs=loc.get_clearsky(t,model='ineichen',solar_position=sp)
cs.index=g.index; sp.index=g.index
kt=g.ghi/cs.ghi.where(cs.ghi>300)
var=kt.rolling(12,min_periods=6).std()
clear=(kt.between(0.8,1.45))&(var<0.04)&(sp.apparent_elevation>15)
def poa(beta,az):
    r=pvlib.irradiance.get_total_irradiance(beta,az,sp.apparent_zenith,sp.azimuth,cs.dni,cs.ghi,cs.dhi,albedo=0.25)
    return r['poa_global']
periods={'2017-07..2022':('2017-07-01','2022-12-31'),'2023':('2023-01-01','2023-12-31'),'2024':('2024-01-01','2024-12-31'),'2016-04..06':('2016-04-01','2016-06-30')}
cands=[(b,a) for b in range(0,41,2) for a in (300,330,0,30,60)]
P={c:poa(*c) for c in cands}
for ch in ('ghi','ghi_alt'):
  for name,(a0,a1) in periods.items():
    m=clear[a0:a1]; m=m[m].index
    y=g.loc[m,ch]; ok=y.notna()&(y>50); m=m[ok.values]
    if len(m)<500: continue
    mon=m.to_period('M')
    best=None
    for c in cands:
        ratio=(g.loc[m,ch]/P[c].loc[m]).groupby(mon).median()
        lr=np.log(ratio); sc=np.exp(lr.mean()); err=float(np.sqrt(((lr-lr.mean())**2).mean()))
        if best is None or err<best[0]: best=(err,c,sc)
    flat=(g.loc[m,ch]/cs.ghi.loc[m]).groupby(mon).median()
    print(f"{ch:8s} {name:14s} n={len(m):6d} best tilt={best[1][0]:2d} az={best[1][1]:3d} scale={best[2]:.3f} seasonal rms={best[0]:.3f} | horizontal-model rms={np.log(flat).std(ddof=0):.3f}")
