"""Reading of stage S48 under D-090: coverage and decision results of fold 2024 on the primary
record against the two alternative records.

Usage (from the repository root unless the path starts at ce1/, then from 03_code):
    python3 03_code/ce1/ce1_wp48_read.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import ce1_decision as CD  # noqa: E402

R = ROOT / "04_results" / "ce1"
REC = {"E": (R / "S5", R / "S6", R / "S6B"),
       "SF": (R / "S48/SF/S5", R / "S48/SF/S6", R / "S48/SF/S6B"),
       "B": (R / "S48/B/S5", R / "S48/B/S6", R / "S48/B/S6B")}
STATIC = ["icp", "icp_norm", "mondrian", "cqr", "mondrian_cqr", "raw_q", "tod", "nicp_var"]
ACI = ["aci|anticipative", "aci_regime|anticipative", "aci|delayed", "aci_regime|delayed"]
PAIRS = [("cqr", "raw_q"), ("mondrian_cqr", "cqr"), ("mondrian", "icp"), ("mondrian", "nicp_var")]
TRANS = 1  # reg_it code of the transitional regime


def coverage():
    C = pd.read_csv(R / "S7/coverage_ci.csv")
    C = C[(C.site == "yulara") & (C.fold == 2024) & (C.nominal == 0.9) & (C.scope == "transitional")]
    rows = []
    for h in CD.HORIZONS:
        Z = {v: np.load(p[0] / f"yulara_f2024_h{h}_obs.npz") for v, p in REC.items()}
        common = np.intersect1d(np.intersect1d(Z["E"]["ts_ns"], Z["SF"]["ts_ns"]), Z["B"]["ts_ns"])
        ix = {v: np.searchsorted(Z[v]["ts_ns"], common) for v in Z}
        regE = Z["E"]["reg_it"][ix["E"]]
        for k in [f"{m}|static|0.9" for m in STATIC] + [f"{a}|0.9" for a in ACI]:
            m, var = k.split("|")[:2]
            c = C[(C.horizon_min == 5 * h) & (C.method == m) & (C.variant == var)]
            hw = float((c.PICP_ci_hi - c.PICP_ci_lo).iloc[0] / 2)
            r = dict(h_min=5 * h, method=m, variant=var, n_common=len(common), E_halfwidth=hw)
            for v in Z:
                cov = Z[v][f"cov__{k}"][ix[v]]
                r[v] = float(cov[Z[v]["reg_it"][ix[v]] == TRANS].mean())
                r[f"{v}_Elabels"] = float(cov[regE == TRANS].mean())
                r[f"{v}_below"] = bool(0.9 - r[v] > hw)
            r["status_unchanged_SF"] = r["SF_below"] == r["E_below"]
            r["status_unchanged_B"] = r["B_below"] == r["E_below"]
            rows.append(r)
    return pd.DataFrame(rows)


def decision():
    rows, con, d4 = [], [], []
    for v, (_, s6, s6b) in REC.items():
        for h in CD.HORIZONS:
            J = {}
            for d, scs in ((s6, ("main", "shapeB")), (s6b, ("battery", "batteryB"))):
                z = np.load(d / f"f2024_h{h}_daily.npz")
                sel = json.load(open(d / f"f2024_h{h}_select.json"))
                tot = pd.read_parquet(d / f"f2024_h{h}_totals.parquet")
                C = CD.draw_counts(z[f"{scs[0]}__fuel_L"].shape[1], 2000)
                for sc in scs:
                    J[sc] = CD.Job(z, sel, sc, C, tot[(tot.scenario == sc) & (tot.set == "test")])
            for sc, sb in (("main", "battery"), ("shapeB", "batteryB")):
                j = J[sc]
                hb = j.pt["fuel_L"][j.row_of("handbook", np.nan)]
                orc = j.sel["anchors"]["oracle"]["fuel_L"]
                for p in CD.POLICIES:
                    s = j.sel["policies"][p]["A_R"]
                    k = j.row_of(p, s["param"])
                    F = j.F(p)[0]
                    rows.append(dict(record=v, h_min=5 * h, scenario=sc, policy=p, param_AR=s["param"],
                                     d_fuel_pct_vs_handbook=100 * (j.pt["fuel_L"][k] - hb) / hb,
                                     ue_share=j.pt["ue_share"][k],
                                     dr1_fold_pass=bool(j.pt["fuel_L"][k] < hb and j.pt["ue_share"][k] <= CD.UE_TARGET),
                                     F=F, value_captured=(hb - F) / (hb - orc)))
                    lb = J[sb].sel["policies"][p]["A_R"]["param"]
                    d4.append(dict(record=v, h_min=5 * h, shape=sc, policy=p, level_no_batt=s["param"],
                                   level_batt=lb, direction=int(np.sign(round(lb - s["param"], 9)))))
            m = J["main"]
            for a, b in PAIRS:
                fa, ba = m.F(a)[:2]
                fb, bb = m.F(b)[:2]
                lo, hi = CD.ci(ba - bb)
                con.append(dict(record=v, h_min=5 * h, a=a, b=b, d_F=fa - fb, ci_lo=lo, ci_hi=hi))
    return pd.DataFrame(rows), pd.DataFrame(con), pd.DataFrame(d4)


def point_aci():
    rows = []
    for h in CD.HORIZONS:
        r = dict(h_min=5 * h)
        for v, p in REC.items():
            z = np.load(p[0] / f"yulara_f2024_h{h}_obs.npz")
            r[f"gbm_rmse_{v}"] = float(np.sqrt(np.mean((z["y"] - z["p"]) ** 2)))
            r[f"aci_delayed_width_increase_{v}"] = float(z["wid__aci|delayed|0.9"].mean()
                                                         / z["wid__aci|anticipative|0.9"].mean() - 1)
        rows.append(r)
    return pd.DataFrame(rows)


def main():
    out = R / "S48" / "reading"
    out.mkdir(parents=True, exist_ok=True)
    cv = coverage(); cv.to_csv(out / "coverage.csv", index=False)
    dec, con, d4 = decision()
    dec.to_csv(out / "decision.csv", index=False); con.to_csv(out / "contrasts.csv", index=False)
    d4.to_csv(out / "dr4.csv", index=False)
    pa = point_aci(); pa.to_csv(out / "point_aci.csv", index=False)
    print("coverage status changes: SF", int((~cv.status_unchanged_SF).sum()), "B", int((~cv.status_unchanged_B).sum()),
          "of", len(cv))
    print("DR4 direction counts:", d4.groupby("record").direction.value_counts().to_dict())
    print(pa.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
