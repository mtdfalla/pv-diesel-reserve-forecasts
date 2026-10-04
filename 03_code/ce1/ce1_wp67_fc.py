#!/usr/bin/env python3
"""Lower-tail scores at the selected levels and a block-bootstrap exceedance test (rule D-129,
items 8 and 9).

Usage (from the repository root unless the path starts at ce1/, then from 03_code):
    python3 03_code/ce1/ce1_wp67_fc.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "utils"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import ce1_dispatch as D  # noqa: E402
import ce1_folds as FC  # noqa: E402
import ce1_s6 as S6  # noqa: E402
import ce1_s63 as S63  # noqa: E402
import ce1_wp63 as W63  # noqa: E402

FOLDS = [2019, 2020, 2021, 2022, 2023, 2024]
H = 3
THP = S63.THP
REGS = S63.REGS
B = 10_000


def main():
    R = Path(".") / "04_results" / "ce1"
    out = R / "WP67"
    D.extend_theta_low()
    cases = pd.read_csv(R / "WP63" / "cases.csv", float_precision="round_trip")
    rows8, rows9 = [], []
    draws = {}
    for yr in FOLDS:
        c = FC.load_cache_ext(R / "S2" / f"yulara_f{yr}_h3.parquet", R / "S2X" / f"yulara_f{yr}_h3.parquet")
        fit, ev = S6.as_dict(c[c.split == "calib"]), S6.as_dict(c[c.split == "test"])
        y = np.asarray(ev["y"], float)
        g_it = np.asarray(ev["g"]).astype(str)
        days = pd.DatetimeIndex(ev["idx"]).normalize()
        du, dix = np.unique(days.values, return_inverse=True)
        nd = len(du)
        if nd not in draws:
            draws[nd] = {L: W63.block_counts(nd, B, L) for L in (1, 7)}
        # ---- item 8
        for pol in THP:
            for sname in ("main", "shapeB"):
                for sel in ("A_R", "b_xfit_0.001"):
                    th = float(cases[(cases.fold == yr) & (cases.scenario == sname) & (cases.policy == pol)
                                     & (cases.selection == sel)].param.iloc[0])
                    lim = th >= 1.0 - 1e-9
                    b_ = np.zeros(len(y)) if lim else D.counted_ghi(pol, th, fit, ev)
                    e = y < b_
                    for reg in REGS + ["all"]:
                        mm = np.ones(len(y), bool) if reg == "all" else g_it == reg
                        rows8.append(dict(fold=yr, scenario=sname, policy=pol, selection=sel, theta=th, at_limit=lim,
                                          regime=reg, n=int(mm.sum()), exceed=float(e[mm].mean()), expected=1 - th,
                                          ratio=float(e[mm].mean() / (1 - th)) if not lim else np.nan,
                                          pinball=S63.pinball(y[mm], b_[mm], 1 - th) if not lim else np.nan))
        # ---- item 9
        for pol in THP:
            for th in S63.TAIL_THETAS:
                b_ = D.counted_ghi(pol, th, fit, ev)
                e = (y < b_).astype(float)
                for reg in REGS + ["all"]:
                    mm = np.ones(len(y), bool) if reg == "all" else g_it == reg
                    xd = np.bincount(dix[mm], weights=e[mm], minlength=nd)
                    nd_ = np.bincount(dix[mm], minlength=nd).astype(float)
                    n, x = int(nd_.sum()), int(xd.sum())
                    lr, pv = S63.kupiec(x, n, 1 - th)
                    r = dict(fold=yr, policy=pol, theta=th, regime=reg, n=n, exceed_n=x, exceed=x / n if n else np.nan,
                             expected=1 - th, kupiec_reject=bool(lr > S63.KUPIEC_CRIT))
                    for L in (1, 7):
                        Cn = draws[nd][L]
                        num, den = Cn @ xd, Cn @ nd_
                        rate = np.divide(num, den, out=np.full(B, np.nan), where=den > 0)
                        lo, hi = np.nanpercentile(rate, [2.5, 97.5])
                        r[f"ci{L}_lo"], r[f"ci{L}_hi"] = float(lo), float(hi)
                        r[f"block{L}_reject"] = bool((1 - th) < lo or (1 - th) > hi)
                    rows9.append(r)
        print(f"fold {yr} done", flush=True)
    T8 = pd.DataFrame(rows8)
    T9 = pd.DataFrame(rows9)
    T8.to_csv(out / "tail_selected.csv", index=False)
    T9.to_csv(out / "tail_block.csv", index=False)
    # reading: the D-122 rule with each test
    rd = {}
    for test in ("kupiec_reject", "block1_reject", "block7_reject"):
        per = {}
        for pol in THP:
            det = {}
            ok_all = True
            for th in [t for t in S63.TAIL_THETAS if t <= 0.99]:
                g = T9[(T9.policy == pol) & (T9.theta == th) & T9.regime.isin(REGS)]
                pf = g.groupby("fold").apply(lambda d: int((~d[test]).sum()))
                fo = int((pf >= 2).sum())
                det[str(th)] = fo
                ok_all &= fo >= 4
            per[pol] = dict(reliable=bool(ok_all), folds_2_of_3_not_rejected=det)
        rd[test] = per
    # the Kupiec recomputation must reproduce the S63 reading
    s63 = json.loads((R / "S63" / "reading.json").read_text(encoding="utf-8"))["4_tail"]
    import ast
    s63 = ast.literal_eval(s63) if isinstance(s63, str) else s63
    for pol in THP:
        assert rd["kupiec_reject"][pol]["folds_2_of_3_not_rejected"] == s63["per_policy"][pol]["folds_with_2_of_3_regimes_not_rejected"], pol
        assert rd["kupiec_reject"][pol]["reliable"] == s63["per_policy"][pol]["reliable_lower_tail"]
    verdict_k = {p: rd["kupiec_reject"][p]["reliable"] for p in THP}
    verdict_b = {p: rd["block1_reject"][p]["reliable"] for p in THP}
    sel = T8[(T8.regime != "all") & ~T8.at_limit]
    summ8 = sel.groupby(["selection", "policy", "regime"]).agg(exceed_median=("exceed", "median"), ratio_median=("ratio", "median"),
                                                               pinball_median=("pinball", "median"), cases=("n", "size")).reset_index()
    summ8.to_csv(out / "tail_selected_summary.csv", index=False)
    rec = dict(item="D-129 step 2 items 8 and 9",
               scope="pop=Yulara six folds, h=15 min, theta policies (ICP, Mondrian, CQR, M-CQR, raw q, ToD, VN-ICP), "
                     "issue-time regime; item 9 blocks=1 day primary, 7 days check, B 10,000, seed 42",
               kupiec_reproduces_S63=True, reliable_kupiec=verdict_k, reliable_block1=verdict_b,
               reliable_block7={p: rd["block7_reject"][p]["reliable"] for p in THP},
               verdicts_differ=[p for p in THP if verdict_k[p] != verdict_b[p]],
               detail=rd,
               cells_reject=dict(kupiec=int(T9[T9.regime.isin(REGS)].kupiec_reject.sum()),
                                 block1=int(T9[T9.regime.isin(REGS)].block1_reject.sum()),
                                 block7=int(T9[T9.regime.isin(REGS)].block7_reject.sum()),
                                 n=int(T9.regime.isin(REGS).sum())),
               item8_cases_at_limit=int(T8[(T8.regime == "all")].at_limit.sum()),
               item8_cases=int((T8.regime == "all").sum()))
    (out / "tail_block.json").write_text(json.dumps(rec, indent=1, default=float), encoding="utf-8")
    print(json.dumps({k: v for k, v in rec.items() if k != "detail"}, indent=1, default=float))


if __name__ == "__main__":
    main()
