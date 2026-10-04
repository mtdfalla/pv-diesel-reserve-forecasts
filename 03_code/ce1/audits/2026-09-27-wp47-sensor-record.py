"""Analysis of the Yulara irradiance sensor record (blocks a1 to a16): channel exchange
calendar, fitted sensor planes, logging scale of the tilted sensor and the correction of the
2024 horizontal-sensor step. The adopted treatment is implemented in ce1_sensor.py. Each block is a
separate analysis step guarded by `if False:`; set a block's guard to True to run it. The load and
preparation blocks build the raw.pkl and geo.pkl inputs of the others. The blocks were run as
separate files named in their headings (load.py, prep.py, a1.py to a16.py); to rerun a block, save
it under that name.
"""

# ======================================================================== load.py
if False:  # load.py
    import pandas as pd, numpy as np
    SRC="02_data/interim/sensor_columns"
    GEO="04_results/ce1/S1/yulara_geo_5min.parquet"
    MAP={"8":"CA","5":"DG","11":"LD","7":"SD1","9":"SD2","6":"SD3","10":"SS","3050":"TOTAL","3051":"SD_TOTAL"}
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
                ren[c]=MAP.get(c.split("_")[0],c)
            d=d.rename(columns=ren); d["ts"]=pd.to_datetime(d["ts"],errors="coerce"); d["file_year"]=y
            fr.append(d)
        d=pd.concat(fr,ignore_index=True).dropna(subset=["ts"])
        d=d.sort_values("ts",kind="stable").drop_duplicates("ts",keep="first").set_index("ts")
        num=d.columns.drop("file_year")
        d[num]=d[num].where(d[num].abs()<8e4)
        return d
    if __name__=="__main__":
        d=load(); print(d.shape, list(d.columns)); d.to_pickle("raw.pkl")
        g=pd.read_parquet(GEO); print(g.shape, g.columns.tolist(), g.index[:2])

# ======================================================================== prep.py
if False:  # prep.py
    import pandas as pd, numpy as np, pvlib, warnings; warnings.filterwarnings("ignore")
    from load import GEO
    d=pd.read_pickle("raw.pkl"); g=pd.read_parquet(GEO)
    g=g.join(d[["ghi_alt","tamb"]],how="left")
    lat,lon,alt=-25.2406,130.9889,492
    t=g.index.tz_localize("Etc/GMT-9") if False else g.index.tz_localize(pd.Timedelta(hours=9.5).__str__() and "Australia/Darwin")
    sp=pvlib.solarposition.get_solarposition(t,lat,lon,alt)
    loc=pvlib.location.Location(lat,lon,"Australia/Darwin",alt)
    cs=loc.get_clearsky(t,model="ineichen",solar_position=sp)
    for c in ["apparent_zenith","azimuth","apparent_elevation"]: g[c]=sp[c].values
    for c in ["ghi","dni","dhi"]: g["cs_"+c]=cs[c].values
    g["dni_extra"]=pvlib.irradiance.get_extra_radiation(t).values
    g.to_pickle("geo.pkl"); print(g.describe().T.round(2))

# ======================================================================== a1.py
if False:  # a1.py
    import pandas as pd, numpy as np
    g=pd.read_pickle("geo.pkl")
    g["ghi_alt"]=g.ghi_alt.where(g.ghi_alt.between(-50,1600))
    day=(g.apparent_elevation>15)
    r=(g.ghi/g.ghi_alt).where(day&(g.ghi>100)&(g.ghi_alt>20))
    print("monthly median ratio ghi/ghi_alt (elev>15)")
    m=r.groupby([g.index.year,g.index.month]).median().unstack().round(3); print(m.to_string())
    kt=g.ghi/g.cs_ghi
    ka=g.ghi_alt/g.cs_ghi
    noon=(g.apparent_elevation>g.groupby(g.index.date).apparent_elevation.transform("max")-5)
    x=pd.DataFrame({"kt":kt[noon],"ka":ka[noon]})
    dd=x.groupby(x.index.date).agg(["median","count"])
    win=dd.loc[pd.Timestamp("2022-12-10").date():pd.Timestamp("2023-02-10").date()]
    print(win.round(3).to_string())
    # missing data around change
    s=g.ghi["2022-12-25":"2023-01-15"]
    print("ghi valid per day:", s.notna().groupby(s.index.date).sum().to_dict())
    s=g.ghi_alt["2022-12-25":"2023-01-15"]
    print("alt valid per day:", s.notna().groupby(s.index.date).sum().to_dict())

# ======================================================================== a2.py
if False:  # a2.py
    import pandas as pd, numpy as np
    g=pd.read_pickle("geo.pkl"); d=pd.read_pickle("raw.pkl")
    g["ghi_alt"]=g.ghi_alt.where(g.ghi_alt.between(-50,1600))
    g["fy"]=d.file_year.reindex(g.index)
    kt=g.ghi/g.cs_ghi; ka=g.ghi_alt/g.cs_ghi
    mx=g.groupby(g.index.date).apparent_elevation.transform("max")
    noon=g.apparent_elevation>mx-5
    x=pd.DataFrame({"kt":kt[noon],"ka":ka[noon],"r":(g.ghi/g.ghi_alt)[noon]})
    dd=x.groupby(x.index.date).median()
    dd.index=pd.to_datetime(dd.index)
    # find switch dates: daily ratio > 3 vs < 3
    sw=(dd.r>3).astype(int); ch=sw.diff().fillna(0); ch=ch[ch!=0]
    print("switch days (ratio crosses 3):"); print(ch.to_string())
    for a,b in [("2018-04-20","2018-05-10"),("2018-12-22","2019-01-08"),("2021-04-22","2021-05-06")]:
        print(dd[a:b].round(3).to_string())
    print(g.fy.groupby([g.index.year,g.index.month]).agg(lambda s: sorted(s.dropna().unique().astype(int).tolist())).unstack().to_string())

# ======================================================================== a3.py
if False:  # a3.py
    import pandas as pd, numpy as np, pvlib, warnings; warnings.filterwarnings("ignore")
    g=pd.read_pickle("geo.pkl")
    g["ghi_alt"]=g.ghi_alt.where(g.ghi_alt.between(-50,1600))
    idx=g.index; doy=idx.dayofyear
    # segment: 'JA' = Jan 2 .. Apr 25 ; 'MD' = Apr 26 .. Jan 1 (pre-2023)
    ja=((idx.month==1)&(idx.day>=2))|(idx.month.isin([2,3]))|((idx.month==4)&(idx.day<=25))
    g["seg"]=np.where(idx.year>=2023, idx.year.astype(str), np.where(ja,"pre_JA","pre_MD"))
    g.loc[(idx.year==2023)&(idx.month==1)&(idx.day==1),"seg"]="pre_MD"
    g.to_pickle("geo2.pkl")
    sp_z,sp_a=g.apparent_zenith,g.azimuth
    def clear_mask(ch):
        k=g[ch]/g.cs_ghi.where(g.cs_ghi>300)
        v=k.rolling(12,min_periods=6).std()
        return k.between(0.6,1.5)&(v<0.03)&(g.apparent_elevation>15)&(g[ch]>50)
    cands=[(b,a) for b in range(0,41,2) for a in range(270,451,15)]
    res=[]
    for ch in ("ghi","ghi_alt"):
        cm=clear_mask(ch)
        for seg in ["pre_JA","pre_MD","2023","2024"]:
            m=cm&(g.seg==seg)
            if m.sum()<500: continue
            s=g[m]
            key=[s.index.month, (s.index.hour*60+s.index.minute)//60]
            y=np.log(s[ch])
            best=[]
            for b,a in cands:
                p=pvlib.irradiance.get_total_irradiance(b,a%360,s.apparent_zenith,s.azimuth,s.cs_dni,s.cs_ghi,s.cs_dhi,dni_extra=s.dni_extra,model="haydavies",albedo=0.25)["poa_global"]
                lr=(y-np.log(p)).groupby(key).median()
                best.append((float(lr.std(ddof=0)),b,a%360,float(np.exp(lr.mean()))))
            best.sort()
            hz=[r for r in best if r[1]==0][0]
            res.append((ch,seg,int(m.sum()),*best[0],hz[0]))
            print(ch,seg,m.sum(),"best rms %.4f tilt %d az %d scale %.3f | horizontal rms %.4f"%(best[0][0],best[0][1],best[0][2],best[0][3],hz[0]),flush=True)

# ======================================================================== a4.py
if False:  # a4.py
    import pandas as pd, numpy as np, pvlib, warnings; warnings.filterwarnings("ignore")
    g=pd.read_pickle("geo2.pkl")
    P=lambda s,b,a: pvlib.irradiance.get_total_irradiance(b,a,s.apparent_zenith,s.azimuth,s.cs_dni,s.cs_ghi,s.cs_dhi,dni_extra=s.dni_extra,model="haydavies",albedo=0.25)["poa_global"]
    g["poaT"]=P(g,16,345)
    # swapped series
    MD=g.seg=="pre_MD"
    g["H"]=np.where(MD,g.ghi,g.ghi_alt); g["T"]=np.where(MD,g.ghi_alt,g.ghi)
    # T scale in MD
    k=g.H/g.cs_ghi.where(g.cs_ghi>300); v=k.rolling(12,min_periods=6).std()
    clear=k.between(0.7,1.4)&(v<0.03)&(g.apparent_elevation>15)
    m=clear&MD&(g["T"]>2)
    print("MD: T channel / poaT median:",(g["T"]/g.poaT)[m].median(), " n",m.sum())
    print((g["T"]/g.poaT)[m].groupby([g.index[m].year,g.index[m].month]).median().unstack().round(4).to_string())
    # monthly clear-sky scale of H and T (T normalised by poaT, pre_MD T rescaled later)
    hs=(g.H/g.cs_ghi)[clear]; 
    print("\nH / cs_ghi monthly median (clear steps)"); print(hs.groupby([hs.index.year,hs.index.month]).median().unstack().round(3).to_string())
    ts=(g["T"]/g.poaT)[clear&~MD]
    print("\nT / poaT monthly median (clear by H), labels-correct periods"); print(ts.groupby([ts.index.year,ts.index.month]).median().unstack().round(3).to_string())
    g.to_pickle("geo3.pkl")

# ======================================================================== a5.py
if False:  # a5.py
    import pandas as pd, numpy as np
    g=pd.read_pickle("geo3.pkl")
    day=g.apparent_elevation>10
    r=(g.ghi/g.ghi_alt).where(day&(g.ghi>30)&(g.ghi_alt>0.5))
    dr=r.groupby(g.index.date).median(); dr.index=pd.to_datetime(dr.index)
    st=pd.Series(np.where(dr>4,"SWAP","LABEL"),index=dr.index).where(dr.notna())
    # runs
    s=st.dropna(); run=(s!=s.shift()).cumsum()
    runs=s.groupby(run).agg(["first","size"]); runs["start"]=s.index.to_series().groupby(run).first().values; runs["end"]=s.index.to_series().groupby(run).last().values
    runs=runs[["first","start","end","size"]]
    pd.set_option("display.width",200)
    print(runs.to_string())
    # ambiguous days: ratio between 2 and 5
    amb=dr[(dr>1.6)&(dr<6)]; print("ambiguous days:",len(amb)); print(amb.round(2).to_string()[:1500])
    nod=st[st.isna()]; print("days w/o ratio:",len(nod))
    st.to_pickle("daystate.pkl")

# ======================================================================== a6.py
if False:  # a6.py
    import pandas as pd, numpy as np
    g=pd.read_pickle("geo3.pkl")
    # intraday boundary check
    for a,b in [("2018-04-25 20:00","2018-04-26 10:00"),("2019-01-01 20:00","2019-01-02 10:00")]:
        s=g.loc[a:b,["ghi","ghi_alt","apparent_elevation"]]; s=s[s.apparent_elevation>-2]
        print(s.round(1).to_string()[:900])
    # 2024 H step: daily noon H/cs and T/poaT ; ratio
    mx=g.groupby(g.index.date).apparent_elevation.transform("max"); noon=g.apparent_elevation>mx-8
    x=pd.DataFrame({"H":(g.H/g.cs_ghi)[noon],"T":(g["T"]/g.poaT)[noon]})
    x=x[g.index.year[noon]==2024]
    dd=x.groupby(x.index.date).median(); dd["q"]=dd.H/dd["T"]
    print(dd.loc[pd.Timestamp("2024-03-20").date():pd.Timestamp("2024-04-20").date()].round(3).to_string())
    dd.index=pd.to_datetime(dd.index)
    print(dd.q.resample("MS").describe()[["count","25%","50%","75%"]].round(3).to_string())

# ======================================================================== a7.py
if False:  # a7.py
    import pandas as pd, numpy as np
    g=pd.read_pickle("geo3.pkl")
    mx=g.groupby(g.index.date).apparent_elevation.transform("max"); noon=g.apparent_elevation>mx-8
    x=pd.DataFrame({"H":(g.H/g.cs_ghi)[noon],"T":(g["T"]/g.poaT)[noon]})
    dd=x.groupby(x.index.date).median(); dd.index=pd.to_datetime(dd.index); dd["q"]=dd.H/dd["T"]
    clr=(dd["T"].between(0.95,1.12))
    q=dd.q[clr]
    out=q.groupby([q.index.year,q.index.month]).agg(["count","median",lambda s:s.quantile(.1),lambda s:s.quantile(.9)]).round(3)
    out.columns=["n","med","p10","p90"]; print(out.loc[[2022,2023,2024]].to_string())

# ======================================================================== a8.py
if False:  # a8.py
    import pandas as pd, numpy as np
    g=pd.read_pickle("geo3.pkl"); st=pd.read_pickle("daystate.pkl")
    D=g.index.normalize(); doy=g.index.dayofyear
    rule=(D>=pd.Timestamp("2016-07-04"))&(D<=pd.Timestamp("2023-01-01"))&((doy>=116)|(doy==1))
    # rule vs classified days
    sd=pd.Series(rule,index=g.index).groupby(D).first()
    cmp=pd.DataFrame({"rule":np.where(sd,"SWAP","LABEL"),"obs":st.reindex(sd.index)}).dropna()
    print("days classified:",len(cmp)," mismatches:",(cmp.rule!=cmp.obs).sum())
    print(cmp[cmp.rule!=cmp.obs].to_string())
    g["swap"]=rule
    g["H"]=np.where(rule,g.ghi,g.ghi_alt)
    g["T"]=np.where(rule,g.ghi_alt/0.084,g.ghi)  # provisional T scale in swap
    g.to_pickle("geo4.pkl")
    k=g.H/g.cs_ghi.where(g.cs_ghi>300); v=k.rolling(12,min_periods=6).std()
    clear=k.between(0.7,1.4)&(v<0.03)&(g.apparent_elevation>15)
    for nm,ser in [("H (reconstructed)",g.H),("Pyranometer_1 (R1 target)",g.ghi)]:
        kk=(ser/g.cs_ghi)[clear]
        m=kk.groupby([kk.index.year,kk.index.month]).median()
        rms=np.log(m).groupby(level=0).std(ddof=0)
        print(nm, "seasonal rms per year:", rms.round(3).to_dict())
    kk=(g.H/g.cs_ghi)[clear&(g.index<"2024-04-03")]
    m=kk.groupby([kk.index.year,kk.index.month]).median(); print("H to 2024-04-02:", np.log(m).groupby(level=0).std(ddof=0).round(3).to_dict())

# ======================================================================== a9.py
if False:  # a9.py
    import pandas as pd, numpy as np, pvlib, warnings, time; warnings.filterwarnings("ignore")
    g=pd.read_pickle("geo4.pkl")
    lab=~g.swap
    # physics: GTI-DIRINT inverse transposition on plane (16,345), 2023, labels state
    s=g[(g.index.year==2023)&lab&(g.apparent_elevation>5)&g["T"].notna()&g.H.notna()].copy()
    t0=time.time()
    aoi=pvlib.irradiance.aoi(16,345,s.apparent_zenith,s.azimuth)
    tz=s.index.tz_localize("Australia/Darwin")
    outs=[]
    for mth,ss in s.groupby(s.index.month):
        tzz=ss.index.tz_localize("Australia/Darwin")
        r=pvlib.irradiance.gti_dirint(pd.Series(ss["T"].values/1.035*1.047,index=tzz),pd.Series(aoi.loc[ss.index].values,index=tzz),
            pd.Series(ss.apparent_zenith.values,index=tzz),pd.Series(ss.azimuth.values,index=tzz),tzz,16,345,albedo=0.25,model="perez",max_iterations=30)
        outs.append(pd.Series(r["ghi"].values,index=ss.index))
    s["B1"]=pd.concat(outs); print("gti_dirint %.0fs"%(time.time()-t0))
    # empirical: H/T ratio climatology by (week-of-year, hour) from labels-state days <= 2022, applied to T
    tr=g[lab&(g.index.year<=2022)&(g.apparent_elevation>5)&(g.H>20)&(g["T"]>20)]
    key=lambda x:[x.index.isocalendar().week.values.astype(int)//2, x.index.hour.values]
    rat=np.log(tr.H/tr["T"]).groupby(key(tr)).median()
    # only Jan-Apr available pre-2023 in labels state -> coverage limited; fall back to plane-model ratio
    cs_ratio=(g.cs_ghi/g.poaT)
    kk=pd.MultiIndex.from_arrays(key(s))
    emp=pd.Series(rat.reindex(kk).values,index=s.index)
    s["B2"]=np.where(emp.notna(), s["T"]*np.exp(emp), s["T"]*cs_ratio.loc[s.index]*1.047/1.035)
    s["B3"]=s["T"]*cs_ratio.loc[s.index]*1.047/1.035  # clear-sky plane ratio (pure beam-geometry scaling)
    s["kt"]=s.H/s.cs_ghi
    bins=pd.cut(s.kt,[0,0.3,0.6,0.85,1.5],labels=["<0.3","0.3-0.6","0.6-0.85",">0.85"])
    res=[]
    for m in ["B1","B2","B3"]:
        e=s[m]-s.H; ok=e.notna()
        r=dict(method=m, n=int(ok.sum()), rmse=np.sqrt((e[ok]**2).mean()), mbe=e[ok].mean(),
               nrmse_pct=100*np.sqrt((e[ok]**2).mean())/s.H[ok].mean())
        for b in bins.cat.categories:
            mm=ok&(bins==b); r["rmse_"+b]=np.sqrt((e[mm]**2).mean())
        # ramp preservation: 5-min changes correlation
        dH=s.H.diff(); dM=s[m].diff(); c=pd.concat([dH,dM],axis=1).dropna(); r["ramp_corr"]=c.corr().iloc[0,1]
        res.append(r)
    print(pd.DataFrame(res).round(3).to_string())
    print("mean H daytime 2023:",s.H.mean().round(1))
    s[["H","T","B1","B2","B3"]].to_pickle("b2023.pkl")

# ======================================================================== a10.py
if False:  # a10.py
    import pandas as pd, numpy as np
    g=pd.read_pickle("geo4.pkl")
    mx=g.groupby(g.index.date).apparent_elevation.transform("max"); noon=g.apparent_elevation>mx-8
    def daily_q(H):
        x=pd.DataFrame({"h":(H/g.cs_ghi)[noon],"t":(g["T"]/g.poaT)[noon]})
        dd=x.groupby(x.index.date).median(); dd.index=pd.to_datetime(dd.index)
        dd=dd[dd.t.between(0.95,1.12)&dd.h.between(0.8,1.5)]
        return dd.h/dd.t
    def factor(q, win=5, band=0.05):
        # factor for day D = median of q over the last `win` valid days strictly before D
        days=pd.date_range(q.index.min(),q.index.max()+pd.Timedelta(days=1),freq="D")
        qq=q.reindex(days)
        f=[]; hist=[]
        for dday,v in qq.items():
            m=np.median(hist[-win:]) if hist else 1.0
            f.append(m if abs(m-1)>band else 1.0)
            if not np.isnan(v): hist.append(v)
        return pd.Series(f,index=days)
    q=daily_q(g.H)
    print("daily q by state and year (median, p05, p95):")
    st=pd.Series(g.swap.values,index=g.index).groupby(g.index.normalize()).first().reindex(q.index)
    print(q.groupby([q.index.year,st.values]).describe()[["count","5%","50%","95%"]].round(3).to_string() if False else
          q.groupby([q.index.year,st.values]).agg(["count","median",lambda s:s.quantile(.05),lambda s:s.quantile(.95)]).round(3).to_string())
    f=factor(q)
    print("\ndays with factor != 1 by year:", (f!=1).groupby(f.index.year).sum().to_dict())
    fa=f[f!=1]; print("first corrected days:", fa.index[:5].date.tolist(), " last uncorrected post-step:", f["2024-04-01":"2024-04-10"].round(3).to_dict())
    print("2024 factor monthly median:", f["2024-04":].resample("MS").median().round(3).to_dict())
    # simulation: artificial step 1.18 on 2023-04-03 on H, then causal correction; error vs true H
    Hs=g.H.copy(); m=(g.index>="2023-04-03")&(g.index<"2024-01-01"); Hs[m]*=1.18
    qs=daily_q(Hs); fs=factor(qs)
    fs_step=fs.reindex(g.index.normalize()).values
    Hc=Hs/fs_step
    e=(Hc-g.H)[m&(g.apparent_elevation>5)]; truth=g.H[m&(g.apparent_elevation>5)]
    print("\nSIM 2023 step x1.18 -> causal correction: days not yet corrected:",int(((fs["2023-04-03":"2023-12-31"]==1)).sum()),
          " scale error after convergence: median %.4f, p5 %.4f, p95 %.4f"%tuple(np.nanpercentile((Hc/g.H)[m&(g.apparent_elevation>20)&(g.index>="2023-04-10")],[50,5,95])),
          " rmse W/m2 %.1f (nrmse %.2f%%)"%(np.sqrt(np.nanmean(e**2)),100*np.sqrt(np.nanmean(e**2))/np.nanmean(truth)))
    f.to_pickle("factor.pkl")

# ======================================================================== a11.py
if False:  # a11.py
    import pandas as pd, numpy as np
    g=pd.read_pickle("geo4.pkl")
    lab_day=pd.Series(~g.swap.values,index=g.index).groupby(g.index.normalize()).first()
    mx=g.groupby(g.index.date).apparent_elevation.transform("max"); noon=g.apparent_elevation>mx-8
    WIN,BAND=7,0.05
    def daily_q(H):
        x=pd.DataFrame({"h":(H/g.cs_ghi)[noon],"t":(g["T"]/g.poaT)[noon]})
        both=(x.h.notna()&x.t.notna()).groupby(x.index.date).mean()
        dd=x.groupby(x.index.date).median(); dd.index=pd.to_datetime(dd.index); both.index=pd.to_datetime(both.index)
        dd=dd[dd.t.between(0.95,1.12)&dd.h.between(0.8,1.5)&(both.reindex(dd.index)>=0.75)]
        q=dd.h/dd.t; return q[lab_day.reindex(q.index).fillna(False).values]
    def factor(q):
        days=lab_day.index; f=[]; hist=[]; qq=q.reindex(days)
        for dday,v in qq.items():
            m=float(np.median(hist[-WIN:])) if len(hist)>=3 else 1.0
            f.append(m if (abs(m-1)>BAND and lab_day[dday]) else 1.0)
            if not np.isnan(v): hist.append(v)
        return pd.Series(f,index=days)
    q=daily_q(g.H); f=factor(q)
    print("labels-state days with factor!=1 by year:",(f!=1).groupby(f.index.year).sum().to_dict())
    print("2024-04-01..08:",f["2024-04-01":"2024-04-08"].round(3).tolist())
    # retrospective transition flag: days where causal factor differs from centred 15-day median of q by >3%
    qc=q.reindex(f.index).rolling(15,center=True,min_periods=3).median()
    trans=(f/qc.where(qc.notna(),1)-1).abs()>0.03
    trans=trans&lab_day
    print("transition days (flagged, targets excluded):",trans[trans].index.date.tolist())
    def sim(step_day="2023-04-03",k=1.18):
        Hs=g.H.copy(); m=(g.index>=step_day)&(g.index<"2024-01-01"); Hs[m]*=k
        qs=daily_q(Hs); fs=factor(qs)
        qcs=qs.reindex(fs.index).rolling(15,center=True,min_periods=3).median()
        trs=((fs/qcs.where(qcs.notna(),1)-1).abs()>0.03)&lab_day
        Hc=Hs/fs.reindex(g.index.normalize()).values
        keep=m&(g.apparent_elevation>5)&~trs.reindex(g.index.normalize()).values
        e=(Hc-g.H)[keep]; t=g.H[keep]
        return int(trs[step_day:"2023-12-31"].sum()), np.sqrt(np.nanmean(e**2)), 100*np.sqrt(np.nanmean(e**2))/np.nanmean(t), np.nanmean(e)
    for sd,k in [("2023-04-03",1.18),("2023-07-15",1.18),("2023-04-03",1.08),("2023-10-01",0.90)]:
        n,r,nr,mbe=sim(sd,k); print(f"SIM step {k} on {sd}: transition days {n}, rmse {r:.1f} W/m2, nrmse {nr:.2f}%, mbe {mbe:.2f}")
    f.to_pickle("factor.pkl"); trans.to_pickle("trans.pkl")
    g["Hc"]=g.H/f.reindex(g.index.normalize()).values
    g["trans"]=trans.reindex(g.index.normalize()).values
    g.to_pickle("geo5.pkl")
    kk=(g.Hc/g.cs_ghi); v=kk.rolling(12,min_periods=6).std()
    clear=kk.between(0.7,1.4)&(v<0.03)&(g.apparent_elevation>15)&~g.trans
    m=kk[clear].groupby([g.index[clear].year,g.index[clear].month]).median()
    print("criterion 2, corrected H seasonal rms per year:",np.log(m).groupby(level=0).std(ddof=0).round(3).to_dict())
    print("2024 monthly H/cs corrected:",m.loc[2024].round(3).to_dict())

# ======================================================================== a12.py
if False:  # a12.py
    import pandas as pd, numpy as np
    exec(open("a11.py").read().split("q=daily_q(g.H)")[0])
    def transition(q,f):
        days=f.index; qq=q.reindex(days); flag=[]; on=False
        for d in days:
            v=qq[d]
            if not np.isnan(v):
                dev=abs(v/f[d]-1)
                if dev>BAND: on=True
                elif dev<=0.03: on=False
            flag.append(on)
        return pd.Series(flag,index=days)
    q=daily_q(g.H); f=factor(q); tr=transition(q,f)
    print("transition days:",tr[tr].index.date.tolist())
    print("q/f around:",(q/f.reindex(q.index))["2024-03-30":"2024-04-12"].round(3).to_dict())
    res=[]
    for sd,k in [("2023-04-03",1.18),("2023-07-15",1.18),("2023-04-03",1.08),("2023-10-01",0.90)]:
        Hs=g.H.copy(); m=(g.index>=sd)&(g.index<"2024-01-01"); Hs[m]*=k
        qs=daily_q(Hs); fs=factor(qs); ts=transition(qs,fs)
        Hc=Hs/fs.reindex(g.index.normalize()).values
        keep=m&(g.apparent_elevation>5)&~ts.reindex(g.index.normalize()).values
        e=(Hc-g.H)[keep]; t=g.H[keep]
        res.append(dict(step=k,day=sd,transition_days=int(ts[sd:"2023-12-31"].sum()),rmse=np.sqrt(np.nanmean(e**2)),nrmse_pct=100*np.sqrt(np.nanmean(e**2))/np.nanmean(t),mbe=np.nanmean(e),
                        scale_p5=np.nanpercentile((Hc/g.H)[keep&(g.apparent_elevation>20)],5),scale_p95=np.nanpercentile((Hc/g.H)[keep&(g.apparent_elevation>20)],95)))
    print(pd.DataFrame(res).round(3).to_string())
    g["Hc"]=g.H/f.reindex(g.index.normalize()).values
    g["trans"]=tr.reindex(g.index.normalize()).values.astype(bool)
    g.to_pickle("geo5.pkl"); f.to_pickle("factor.pkl"); tr.to_pickle("trans.pkl"); q.to_pickle("q.pkl")

# ======================================================================== a13.py
if False:  # a13.py
    import pandas as pd, numpy as np
    g=pd.read_pickle("geo5.pkl"); d=pd.read_pickle("raw.pkl")
    RATED={"LD":327.6,"SS":226.8,"SD":106.7}
    d["SD"]=d[["SD1","SD2","SD3"]].sum(axis=1,min_count=3)
    x=g[["Hc","ghi","trans"]].join(d[["LD","SS","SD"]],how="left")
    x=x[(x.index.year>=2023)]
    for a,r in RATED.items(): x[a]=x[a].where(x[a].between(-5,1.25*r))/r
    def fit(xx,col):
        xx=xx[(xx[col]>0)].dropna(subset=[col,"LD","SS","SD"])
        sp=xx[["LD","SS","SD"]].mean(axis=1)
        b=np.floor(xx[col]/25).astype(int)
        med=sp.groupby(b).agg(["median","count"]); med=med[med["count"]>=20]["median"]
        med=med.reindex(range(0,b.max()+1)).interpolate().fillna(0).cummax()
        return med
    def pred(m,v): i=np.clip(np.floor(v/25).astype(int),0,len(m)-1); return m.values[i]
    def r2(y,yh): return 1-np.nansum((y-yh)**2)/np.nansum((y-np.nanmean(y))**2)
    for basis,col in [("T (Pyranometer_1, current map)","ghi"),("H corrected (proposed)","Hc")]:
        tr=x[(x.index.year==2023)&~x.trans]; te=x[(x.index.year==2024)&~x.trans]
        m=fit(tr,col); out=[]
        for a in RATED:
            t=te[(te[col]>0)].dropna(subset=[col,a]); out.append(round(r2(t[a].values,pred(m,t[col].values)),3))
        m2=fit(te,col); out2=[]
        for a in RATED:
            t=tr[(tr[col]>0)].dropna(subset=[col,a]); out2.append(round(r2(t[a].values,pred(m2,t[col].values)),3))
        print(f"{basis:34s} fit2023->2024 R2 LD/SS/SD {out}   fit2024->2023 {out2}")
    g5=g
    x["Htr"]=(g5.Hc*(g5.poaT/g5.cs_ghi).where(g5.cs_ghi>20).clip(0,3)).reindex(x.index)
    for basis,col in [("H corrected x clear-sky plane ratio","Htr")]:
        tr=x[(x.index.year==2023)&~x.trans]; te=x[(x.index.year==2024)&~x.trans]
        m=fit(tr,col); out=[]
        for a in RATED:
            t=te[(te[col]>0)].dropna(subset=[col,a]); out.append(round(r2(t[a].values,pred(m,t[col].values)),3))
        m2=fit(te,col); out2=[]
        for a in RATED:
            t=tr[(tr[col]>0)].dropna(subset=[col,a]); out2.append(round(r2(t[a].values,pred(m2,t[col].values)),3))
        print(f"{basis:34s} fit2023->2024 R2 LD/SS/SD {out}   fit2024->2023 {out2}")

# ======================================================================== a14.py
if False:  # a14.py
    import pandas as pd, numpy as np, pvlib, warnings; warnings.filterwarnings("ignore")
    a=pd.read_parquet("04_results/ce1/S1/asp_geo_5min.parquet")
    lat,lon,alt=-23.7621,133.8745,545
    t=a.index.tz_localize("Australia/Darwin")
    sp=pvlib.solarposition.get_solarposition(t,lat,lon,alt)
    cs=pvlib.location.Location(lat,lon,"Australia/Darwin",alt).get_clearsky(t,model="ineichen",solar_position=sp)
    a["el"]=sp.apparent_elevation.values; a["cs"]=cs.ghi.values
    k=a.ghi/a.cs.where(a.cs>300); v=k.rolling(12,min_periods=6).std()
    clear=k.between(0.7,1.4)&(v<0.03)&(a.el>15)
    kk=k[clear]; idx=kk.index
    m=kk.groupby([idx.year,idx.month]).median().unstack().round(3); print("DKASC GHI monthly clear-sky index"); print(m.to_string())
    print("seasonal rms per year:",np.log(kk.groupby([idx.year,idx.month]).median()).groupby(level=0).std(ddof=0).round(3).to_dict())
    ja=((idx.month==1)&(idx.day>=2))|idx.month.isin([2,3])|((idx.month==4)&(idx.day<=25))
    # compare Apr 1-25 vs Apr 26-30, Jan 1 vs Jan 2-10 per year (splice signature)
    for y in sorted(set(idx.year)):
        s=kk[idx.year==y]; i=s.index
        a1=s[(i.month==4)&(i.day<=25)].median(); a2=s[((i.month==4)&(i.day>=26))|((i.month==5)&(i.day<=10))].median()
        print(y,"Apr1-25 %.3f vs Apr26-May10 %.3f"%(a1,a2))
    # daily noon kt step detection: largest day-to-day jumps in 15-day rolling median
    dn=kk.groupby(idx.normalize()).median(); rm=dn.rolling(15,min_periods=8).median()
    jump=(rm/rm.shift(15)-1).abs().sort_values(ascending=False).head(8); print("largest 15-day shifts of clear kt:",jump.round(3).to_dict())

# ======================================================================== a15.py
if False:  # a15.py
    import pandas as pd, numpy as np, pvlib, warnings; warnings.filterwarnings("ignore")
    a=pd.read_parquet("04_results/ce1/S1/asp_geo_5min.parquet")
    lat,lon,alt=-23.7621,133.8745,545
    t=a.index.tz_localize("Australia/Darwin")
    sp=pvlib.solarposition.get_solarposition(t,lat,lon,alt)
    cs=pvlib.location.Location(lat,lon,"Australia/Darwin",alt).get_clearsky(t,model="ineichen",solar_position=sp)
    for c in ["apparent_zenith","azimuth","apparent_elevation"]: a[c]=sp[c].values
    for c in ["ghi","dni","dhi"]: a["cs_"+c]=cs[c].values
    a["dni_extra"]=pvlib.irradiance.get_extra_radiation(t).values
    k=a.ghi/a.cs_ghi.where(a.cs_ghi>300); v=k.rolling(12,min_periods=6).std()
    cm=k.between(0.6,1.5)&(v<0.03)&(a.apparent_elevation>15)&(a.ghi>50)
    cands=[(b,az) for b in range(0,21,1) for az in range(270,451,15)]
    g=pd.read_pickle("geo5.pkl")
    def fitseg(df,ch,m,label):
        s=df[m]; key=[s.index.month,(s.index.hour*60+s.index.minute)//60]; y=np.log(s[ch]); best=[]
        for b,az in cands:
            p=pvlib.irradiance.get_total_irradiance(b,az%360,s.apparent_zenith,s.azimuth,s.cs_dni,s.cs_ghi,s.cs_dhi,dni_extra=s.dni_extra,model="haydavies",albedo=0.25)["poa_global"]
            lr=(y-np.log(p)).groupby(key).median(); best.append((float(lr.std(ddof=0)),b,az%360,float(np.exp(lr.mean()))))
        best.sort(); hz=[r for r in best if r[1]==0][0]
        print(f"{label:28s} n={m.sum():6d} best rms {best[0][0]:.4f} tilt {best[0][1]} az {best[0][2]} scale {best[0][3]:.3f} | horizontal rms {hz[0]:.4f} scale {hz[3]:.3f}",flush=True)
    for y in [2020,2021,2022,2023,2024]:
        fitseg(a,"ghi",cm&(a.index.year==y),f"DKASC GHI {y}")
    kH=g.Hc/g.cs_ghi.where(g.cs_ghi>300); vH=kH.rolling(12,min_periods=6).std()
    cmH=kH.between(0.6,1.5)&(vH<0.03)&(g.apparent_elevation>15)&(g.Hc>50)&~g.trans
    gg=g.rename(columns={"Hc":"HC"})
    for y in [2018,2021,2023,2024]:
        fitseg(gg,"HC",cmH&(g.index.year==y),f"Yulara H corrected {y}")

# ======================================================================== a16.py
if False:  # a16.py
    import pandas as pd, numpy as np, pvlib, warnings; warnings.filterwarnings("ignore")
    g=pd.read_pickle("geo5.pkl"); d=pd.read_pickle("raw.pkl"); rec=pd.read_pickle("record.pkl")
    RATED={"LD":327.6,"SS":226.8,"SD":106.7}
    d["SD"]=d[["SD1","SD2","SD3"]].sum(axis=1,min_count=3)
    x=g[["apparent_zenith","azimuth","dni_extra","cs_ghi"]].join(d[["LD","SS","SD"]],how="left")
    x["H"]=rec.ghi.reindex(x.index)
    x=x[x.index.year>=2023]
    for a,r in RATED.items(): x[a]=x[a].where(x[a].between(-5,1.25*r))/r
    tz=x.index.tz_localize("Australia/Darwin")
    er=pvlib.irradiance.erbs(x.H.clip(lower=0).fillna(0).values,x.apparent_zenith.values,x.dni_extra.values)
    x["dni"]=np.asarray(er["dni"]); x["dhi"]=np.asarray(er["dhi"])
    def poa(b,az): return pvlib.irradiance.get_total_irradiance(b,az,x.apparent_zenith,x.azimuth,x.dni,x.H.clip(lower=0),x.dhi,dni_extra=x.dni_extra,model="haydavies",albedo=0.25)["poa_global"].where(x.H.notna())
    def fit(xx,col):
        xx=xx[(xx[col]>0)].dropna(subset=[col,"LD","SS","SD"]); sp=xx[["LD","SS","SD"]].mean(axis=1)
        b=np.floor(xx[col]/25).astype(int); med=sp.groupby(b).agg(["median","count"]); med=med[med["count"]>=20]["median"]
        return med.reindex(range(0,b.max()+1)).interpolate().fillna(0).cummax()
    def pred(m,v): i=np.clip(np.floor(v/25).astype(int),0,len(m)-1); return m.values[i]
    def r2(y,yh): return 1-np.nansum((y-yh)**2)/np.nansum((y-np.nanmean(y))**2)
    def evalcol(col,tr_year,te_year):
        tr=x[x.index.year==tr_year]; te=x[x.index.year==te_year]; m=fit(tr,col); out={}
        for a in RATED:
            t=te[te[col]>0].dropna(subset=[col,a]); out[a]=round(r2(t[a].values,pred(m,t[col].values)),3)
        return out
    # plane search on 2023 in-sample R2 (pooled), then held-out
    best=None
    for b in range(0,36,5):
        for az in (330,345,0,15,30):
            x["P"]=poa(b,az); tr=x[x.index.year==2023]; m=fit(tr,"P")
            t=tr[tr.P>0].dropna(subset=["P","LD","SS","SD"]); sp=t[["LD","SS","SD"]].mean(axis=1)
            s=r2(sp.values,pred(m,t.P.values))
            if best is None or s>best[0]: best=(s,b,az)
    print("best array plane on 2023 (in-sample pooled R2 %.3f): tilt %d az %d"%best)
    x["P"]=poa(best[1],best[2])
    print("H basis        2023->2024",evalcol("H",2023,2024)," 2024->2023",evalcol("H",2024,2023))
    print("H->POA (fitted) 2023->2024",evalcol("P",2023,2024)," 2024->2023",evalcol("P",2024,2023))
    x["T"]=d.ghi.reindex(x.index)
    print("T basis (old)  2023->2024",evalcol("T",2023,2024)," 2024->2023",evalcol("T",2024,2023))
