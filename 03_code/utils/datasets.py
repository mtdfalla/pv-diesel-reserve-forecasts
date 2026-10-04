"""Dataset construction in clear-sky-index space: predict kt(t+h), then GHI = kt_hat x
GHI_clearsky(t+h).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C
import numpy as np, pandas as pd

LAGS = [0, 1, 2, 3, 4, 5, 6, 9, 12]  # steps back (0 = current), 5min each
ROLL = 6  # rolling window (30 min)

def build_base():
    clean = pd.read_parquet(C.DATA_CLEAN / "yulara_clean_5min.parquet")
    reg   = pd.read_parquet(C.DATA_REGIME / "yulara_regimes_5min.parquet")
    flags = pd.read_parquet(C.DATA_CLEAN / "yulara_quality_flags.parquet")
    df = pd.DataFrame(index=clean.index)
    df["ghi"]    = clean["ghi"]
    df["ghi_cs"] = reg["ghi_cs"]
    df["kt"]     = reg["kt"]
    df["zenith"] = reg["zenith"]
    df["is_day"] = reg["is_day"]
    df["regime"] = reg["regime"]
    df["ghi_imputed"] = flags["ghi_imputed"]
    df["year"]   = df.index.year
    # cyclical time features
    hod = df.index.hour + df.index.minute/60
    df["hod_sin"] = np.sin(2*np.pi*hod/24); df["hod_cos"] = np.cos(2*np.pi*hod/24)
    doy = df.index.dayofyear
    df["doy_sin"] = np.sin(2*np.pi*doy/365); df["doy_cos"] = np.cos(2*np.pi*doy/365)
    df["cosz"] = np.cos(np.radians(df["zenith"].clip(0,90)))
    # lag features on kt
    for L in LAGS:
        df[f"kt_l{L}"] = df["kt"].shift(L)
    df["kt_rmean"] = df["kt"].rolling(ROLL, min_periods=ROLL).mean()
    df["kt_rstd"]  = df["kt"].rolling(ROLL, min_periods=ROLL).std()
    return df

FEATURES = [f"kt_l{L}" for L in LAGS] + ["kt_rmean","kt_rstd","cosz",
            "hod_sin","hod_cos","doy_sin","doy_cos"]

def make_xy(df, h):
    """Return supervised frame for horizon h (steps). Strict, leakage-free."""
    d = df.copy()
    d["y_kt"]      = d["kt"].shift(-h)
    d["y_ghi"]     = d["ghi"].shift(-h)
    d["y_ghi_cs"]  = d["ghi_cs"].shift(-h)
    d["y_is_day"]  = d["is_day"].shift(-h)
    d["y_imputed"] = d["ghi_imputed"].shift(-h)
    # base info kept for baselines / stratification
    d["base_ghi"]  = d["ghi"]
    d["base_kt"]   = d["kt"]
    d["base_regime"] = d["regime"]
    need = FEATURES + ["y_kt","y_ghi","y_ghi_cs","base_ghi","base_kt"]
    valid = (d["is_day"] & d["y_is_day"].fillna(False) & (~d["y_imputed"].fillna(True))
             & d[need].notna().all(axis=1))
    d = d[valid]
    return d

def split_years(d, train=range(2016,2023), calib=(2023,), test=(2024,)):
    tr = d[d["year"].isin(list(train))]
    ca = d[d["year"].isin(list(calib))]
    te = d[d["year"].isin(list(test))]
    return tr, ca, te
