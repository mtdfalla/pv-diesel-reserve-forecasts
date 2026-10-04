"""Compact GRU and GRU-TCN forecasters of the clear-sky index, used by the point benchmark
(ce1_point.py, stage S4).

Usage (from the repository root unless the path starts at ce1/, then from 03_code):
    python models/deep_gru_tcn.py --epochs 15 --subsample 150000 --horizons 1 3 6 12
"""
import sys, json, time, argparse, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"utils"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"evaluation"))
import numpy as np, pandas as pd
import config as CFG, metrics as M
import torch, torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
torch.set_num_threads(max(1, (__import__("os").cpu_count() or 2)))

W = 12  # 60-min input window

def build_windows(base, h):
    """Vectorised contiguous kt windows -> target kt(t+h); leakage-free, daytime."""
    kt=base["kt"].values.astype("float32"); ghi=base["ghi"].values.astype("float32")
    cs=base["ghi_cs"].values.astype("float32"); isday=base["is_day"].values.astype(bool)
    imp=base["ghi_imputed"].fillna(True).values.astype(bool); reg=base["regime"].values
    idx=base.index.values; n=len(kt)
    sw=np.lib.stride_tricks.sliding_window_view(kt, W)  # [n-W+1, W]
    J=np.arange(0, n-h-W+1)  # window-start positions
    t=J+W-1; tt=t+h  # end / target indices
    win=sw[J]
    ok=(~np.isnan(win).any(1)) & isday[t] & isday[tt] & ~imp[tt] \
       & ~np.isnan(kt[tt]) & ~np.isnan(ghi[tt])
    J,t,tt=J[ok],t[ok],tt[ok]
    X=sw[J][...,None].astype("float32")
    return (X, kt[tt].astype("float32"), cs[tt], ghi[tt], reg[t], base.index.year.values[t], idx[t])

class GRUNet(nn.Module):
    def __init__(s,hid=32): super().__init__(); s.gru=nn.GRU(1,hid,batch_first=True); s.fc=nn.Linear(hid,1)
    def forward(s,x): o,_=s.gru(x); return s.fc(o[:,-1,:]).squeeze(-1)

class GRUTCN(nn.Module):
    def __init__(s,hid=32):
        super().__init__(); s.gru=nn.GRU(1,hid,batch_first=True)
        s.tcn=nn.Sequential(nn.Conv1d(hid,hid,3,padding=1,dilation=1),nn.ReLU(),
                            nn.Conv1d(hid,hid,3,padding=2,dilation=2),nn.ReLU())
        s.fc=nn.Linear(hid,1)
    def forward(s,x):
        o,_=s.gru(x); z=s.tcn(o.transpose(1,2)); return s.fc(z[:,:,-1]).squeeze(-1)

def train_eval(model,Xtr,ytr,Xte,epochs,bs=512,lr=1e-3,seed=42):
    torch.manual_seed(seed)
    dl=DataLoader(TensorDataset(torch.from_numpy(Xtr),torch.from_numpy(ytr)),batch_size=bs,shuffle=True)
    opt=torch.optim.Adam(model.parameters(),lr=lr); lf=nn.MSELoss()
    for _ in range(epochs):
        model.train()
        for xb,yb in dl: opt.zero_grad(); lf(model(xb),yb).backward(); opt.step()
    model.eval()
    with torch.no_grad(): return model(torch.from_numpy(Xte)).numpy()

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--epochs",type=int,default=15)
    ap.add_argument("--subsample",type=int,default=150000)
    ap.add_argument("--horizons",type=int,nargs="+",default=[1,3,6,12])
    a=ap.parse_args()
    base=pd.read_parquet(CFG.DATA_CLEAN/"yulara_clean_5min.parquet")
    reg =pd.read_parquet(CFG.DATA_REGIME/"yulara_regimes_5min.parquet")
    fl  =pd.read_parquet(CFG.DATA_CLEAN/"yulara_quality_flags.parquet")
    base=base.join(reg[["ghi_cs","kt","is_day","regime"]]); base["ghi_imputed"]=fl["ghi_imputed"]
    rec=lambda kt,cs: np.clip(np.clip(kt,0,1.5)*cs,0,None)
    rows=[]
    for h in a.horizons:
        X,ykt,ycs,ygh,rg,yy,ts=build_windows(base,h)
        tr=yy<=2022; te=yy==2024
        Xtr,ytr=X[tr],ykt[tr]
        if a.subsample and tr.sum()>a.subsample:
            sel=np.random.RandomState(42).choice(np.where(tr)[0],a.subsample,replace=False)
            Xtr,ytr=X[sel],ykt[sel]
        Xte=X[te]; y=ygh[te]; cs=ycs[te]; tste=pd.to_datetime(ts[te])
        # GBM preds for fair DM (align on timestamps)
        gp=CFG.DATA_INTERIM/f"p2_test_pred_h{h}.parquet"
        gbm=pd.read_parquet(gp)["gbm"] if gp.exists() else None
        for name,Net in [("gru",GRUNet),("gru_tcn",GRUTCN)]:
            t0=time.time(); pred=rec(train_eval(Net(),Xtr,ytr,Xte,a.epochs),cs)
            m=M.all_metrics(y,pred); m.update(model=name,horizon_min=h*5,n=int(len(y)),train_s=round(time.time()-t0,1))
            if gbm is not None:
                s=pd.Series(pred,index=tste); common=s.index.intersection(gbm.index)
                if len(common)>1000:
                    yc=pd.Series(y,index=tste).loc[common].values
                    dm,pv=M.diebold_mariano(yc,s.loc[common].values,gbm.loc[common].values,h=h)
                    m["DM_vs_gbm"]=round(dm,2); m["p_vs_gbm"]=float(f"{pv:.2e}"); m["n_common"]=int(len(common))
            rows.append(m)
            print(f"h={h*5} {name}: RMSE={m['RMSE']:.2f} MAE={m['MAE']:.2f} R2={m['R2']:.3f} ({m['train_s']}s)",flush=True)
    out=pd.DataFrame(rows); (CFG.TAB).mkdir(parents=True,exist_ok=True)
    out.to_csv(CFG.TAB/"p2_deep_metrics.csv",index=False)
    json.dump(rows,open(CFG.MET/"p2_deep_metrics.json","w"),indent=2,default=str)
    print("\nSAVED 04_results/tables/p2_deep_metrics.csv\n"+out.to_string(index=False))

if __name__=="__main__": main()
