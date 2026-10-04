#!/usr/bin/env python3
"""Stage S6E: the extended load-model record (theta set from 0.500, D-092). Merges the S6, S6B
and S6X totals per fold and horizon and recomputes the selections with the unchanged
ce1_s6.select_policy. The output has the S6 file format.

Usage (from the repository root unless the path starts at ce1/, then from 03_code):
    python3 03_code/ce1/ce1_extend.py [--root .] [--only 2024:3]
    python3 03_code/ce1/ce1_extend.py --selftest
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "utils"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import ce1_dispatch as D  # noqa: E402
import ce1_s6 as S  # noqa: E402

FOLDS, HORIZONS = S.FOLDS, S.HORIZONS
BATT = ("battery", "batteryB")
SEL_KEYS = ("A_R", "A_C", "reference_R", "reference_C", "A_R_target_0.003pct")


def md5(p: Path) -> str:
    return hashlib.md5(p.read_bytes()).hexdigest()


def log(m):
    print(f"[S6E] {m}", flush=True)


def key_of(pol, par):
    return (str(pol), -1.0 if pd.isna(par) else round(float(par), 4))


def order_rows(tab: pd.DataFrame, rows) -> pd.DataFrame:
    """tab (one scenario and set) in the order of rows; asserts every row present once."""
    k = [key_of(p, q) for p, q in zip(tab["policy"], tab["param"])]
    pos = {kk: i for i, kk in enumerate(k)}
    assert len(pos) == len(k), "duplicate rows"
    want = [key_of(p, q) for p, q in rows]
    missing = [w for w in want if w not in pos]
    assert not missing and len(want) == len(k), f"rows missing {missing[:3]} or extra ({len(k)} vs {len(want)})"
    return tab.iloc[[pos[w] for w in want]].reset_index(drop=True)


def totals_dict(t: pd.DataFrame):
    return {k: t[k].to_numpy(float) for k in S.TOT_KEYS}


def sel_equal(a: dict, b: dict) -> list:
    """Differences between two select_policy outputs (params and F)."""
    d = []
    for k in SEL_KEYS:
        if not np.isclose(a[k]["param"], b[k]["param"], atol=1e-9):
            d.append((k, b[k]["param"], a[k]["param"]))
    for k in ("fuel_at_target_test", "fuel_at_target_H2", "fuel_at_target_test_0.003pct"):
        x, y = a[k], b[k]
        if not ((x is None or not np.isfinite(x)) and (y is None or not np.isfinite(y))) and \
                not np.isclose(x, y, rtol=1e-12, atol=0):
            d.append((k, y, x))
    return d


def merge_fold_h(R: Path, yr: int, h: int, rows_ext, rows_old, dry=False):
    """Extended totals, selections and daily arrays of one fold and horizon."""
    stem = f"f{yr}_h{h}"
    t6 = pd.read_parquet(R / "S6" / f"{stem}_totals.parquet")
    t6b = pd.read_parquet(R / "S6B" / f"{stem}_totals.parquet")
    tx = pd.read_parquet(R / "S6X" / f"{stem}_totals.parquet")
    s6 = json.load(open(R / "S6" / f"{stem}_select.json"))
    s6b = json.load(open(R / "S6B" / f"{stem}_select.json"))
    old = pd.concat([t6[~t6.scenario.isin(BATT)], t6b], ignore_index=True)
    scen = list(dict.fromkeys(old.scenario))
    if dry:
        scen = [x for x in scen if x in set(tx.scenario)]
    assert set(tx.scenario) == set(scen), f"S6X scenarios {sorted(set(tx.scenario))} vs {scen}"
    tabs, sel, changes, edge, ident = [], {}, [], [], []
    for sc in scen:
        rec = (s6b if sc in BATT else s6)["selections"][sc]
        parts = {}
        for part in ("H2", "test"):
            o = order_rows(old[(old.scenario == sc) & (old.set == part)], rows_old)
            x = tx[(tx.scenario == sc) & (tx.set == part)]
            e = order_rows(pd.concat([o, x], ignore_index=True), rows_ext)
            parts[part] = (o, e)
            tabs.append(e)
        tH_o, tT_o = totals_dict(parts["H2"][0]), totals_dict(parts["test"][0])
        tH_e, tT_e = totals_dict(parts["H2"][1]), totals_dict(parts["test"][1])
        pols = {}
        for pol in D.ALL_POLICIES:
            so = S.select_policy(tH_o, tT_o, rows_old, pol)  # identity with the record
            dd = sel_equal(so, rec["policies"][pol])
            ident.append(dict(fold=yr, h=h, scenario=sc, policy=pol, n_diff=len(dd), diffs=str(dd)))
            se = S.select_policy(tH_e, tT_e, rows_ext, pol)
            pols[pol] = se
            for k, was, now in sel_equal(se, rec["policies"][pol]):
                changes.append(dict(fold=yr, h=h, scenario=sc, policy=pol, item=k, record=was, extended=now))
            if pol != "frac_pv":
                for k in SEL_KEYS:
                    if abs(se[k]["param"] - 0.5) < 1e-9:
                        edge.append(dict(fold=yr, h=h, scenario=sc, policy=pol, item=k))
        sel[sc] = dict(scenario=rec["scenario"], anchors=rec["anchors"], missing_steps=rec["missing_steps"],
                       policies=pols)
    # daily arrays in the extended order
    z6, z6b, zx = (np.load(R / d / f"{stem}_daily.npz") for d in ("S6", "S6B", "S6X"))
    daily = dict(rows_policy=np.array([r[0] for r in rows_ext]), rows_param=np.array([r[1] for r in rows_ext]))
    for sc in S.DAILY_SCEN:
        src = z6b if sc in BATT else z6
        ko = [key_of(p, q) for p, q in zip(src["rows_policy"], src["rows_param"])]
        kx = [key_of(p, q) for p, q in zip(zx["rows_policy"], zx["rows_param"])]
        pos = {k: ("o", i) for i, k in enumerate(ko)} | {k: ("x", i) for i, k in enumerate(kx)}
        for k in S.DAILY_KEYS:
            A, X = src[f"{sc}__{k}"], zx[f"{sc}__{k}"]
            daily[f"{sc}__{k}"] = np.vstack([(A if pos[key_of(p, q)][0] == "o" else X)[pos[key_of(p, q)][1]]
                                             for p, q in rows_ext])
    return pd.concat(tabs, ignore_index=True), sel, daily, changes, edge, ident


def selftest(verbose=True) -> dict:
    res = {}

    def rep(n, ok, m=""):
        res[n] = bool(ok)
        if verbose:
            print(f"{'PASS' if ok else 'FAIL'}  {n}  {m}")

    rows = [("handbook", np.nan), ("cqr", 0.8), ("cqr", 0.6), ("cqr", 1.0)]
    t = pd.DataFrame(dict(policy=["cqr", "handbook", "cqr", "cqr"], param=[1.0, np.nan, 0.6, 0.8], v=[4, 1, 3, 2]))
    o = order_rows(t, rows)
    rep("order_rows", list(o.v) == [1, 2, 3, 4])
    try:
        order_rows(t.iloc[:3], rows)
        rep("order_rows_rejects_missing", False)
    except AssertionError:
        rep("order_rows_rejects_missing", True)
    # selection on a set with an added cheaper level that meets the target moves A-R down
    rows = [("frac_pv", f) for f in (0.2, 0.4, 0.6)]
    mk = lambda fuel, ue: {k: np.array(v, float) for k, v in dict(  # noqa: E731
        fuel_L=fuel, ue_share=ue, ue_kWh=np.array(ue) * 1e6, load_kWh=[1e6] * len(ue)).items()} | \
        {k: np.zeros(len(ue)) for k in S.TOT_KEYS if k not in ("fuel_L", "ue_share", "ue_kWh", "load_kWh")}
    tot = mk([10.0, 11.0, 12.0], [5e-5, 1e-5, 0.0])
    s = S.select_policy(tot, tot, rows, "frac_pv")
    rep("select_policy_available", s["A_R"]["param"] == 0.4, f"(A-R {s['A_R']['param']})")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(HERE.parent.parent))
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--only", default=None)
    ap.add_argument("--allow-unverified", action="store_true")
    ap.add_argument("--dry", action="store_true", help="test run: scenarios S6X lacks are skipped")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(0 if all(selftest().values()) else 1)
    t0 = time.time()
    R = Path(a.root) / "04_results" / "ce1"
    out, runs = R / "S6E", R / "_runs"
    out.mkdir(parents=True, exist_ok=True)
    checks = {}

    def check(n, ok, m=""):
        checks[n] = dict(ok=bool(ok), msg=m)
        log(f"{'PASS' if ok else 'FAIL'}  {n}  {m}")

    st = selftest(verbose=False)
    check("selftests", all(st.values()), f"({sum(st.values())}/{len(st)})")
    for s_ in ("S6", "S6B", "S6X"):
        p = runs / f"{s_}.json"
        v = p.exists() and json.load(open(p)).get("ok", False)
        check(f"input_stage_{s_}_verified", v or a.allow_unverified)
    rows_old = S.row_table()
    D.extend_theta_low()
    rows_ext = S.row_table()
    check("row_tables", len(rows_old) == 474 and len(rows_ext) == 474 + 350
          and set(rows_old) <= set(rows_ext), f"({len(rows_old)} -> {len(rows_ext)} rows)")
    jobs = [(y, h) for y in FOLDS for h in HORIZONS]
    if a.only:
        jobs = [(int(p.split(":")[0]), int(p.split(":")[1])) for p in a.only.split(",")]
    CH, ED, ID, DS, inputs = [], [], [], [], {}
    exp_cells = 0  # expected identity cells from each job's own source scenarios
    for yr, h in jobs:
        stem = f"f{yr}_h{h}"
        for d in ("S6", "S6B", "S6X"):
            for x in ("totals.parquet", "daily.npz") + (("select.json",) if d != "S6X" else ()):
                p = R / d / f"{stem}_{x}"
                inputs[f"{d}/{p.name}"] = md5(p)
        tab, sel, daily, ch, ed, idn = merge_fold_h(R, yr, h, rows_ext, rows_old, dry=a.dry)
        src_scen = set(pd.read_parquet(R / "S6" / f"{stem}_totals.parquet", columns=["scenario"]).scenario) | set(
            pd.read_parquet(R / "S6B" / f"{stem}_totals.parquet", columns=["scenario"]).scenario)
        exp_cells += len(D.ALL_POLICIES) * len(src_scen)  # 'arrays' exists in folds 2023-2024 only
        for sc in S.DAILY_SCEN:
            T = tab[(tab.scenario == sc) & (tab.set == "test")]
            for k in ("fuel_L", "ue_kWh", "load_kWh"):
                ref = T[k].to_numpy(float)
                DS.append(float(np.max(np.abs(daily[f"{sc}__{k}"].astype(float).sum(axis=1) - ref)
                                       / np.maximum(np.abs(ref), 1.0))))
        tab.to_parquet(out / f"{stem}_totals.parquet", compression="zstd")
        np.savez_compressed(out / f"{stem}_daily.npz", **daily)
        json.dump(dict(summary=dict(fold=yr, h=h, rows=len(rows_ext), theta_low=0.5, source="S6+S6B+S6X [D-092]"),
                       selections=sel, rows=len(rows_ext)), open(out / f"{stem}_select.json", "w"), indent=1,
                  default=float)
        CH += ch; ED += ed; ID += idn
        log(f"fold {yr} h={h}: {len(ch)} changed items, {len(ed)} at the lower edge ({time.time() - t0:.0f}s)")
    I = pd.DataFrame(ID)
    pd.DataFrame(CH, columns=["fold", "h", "scenario", "policy", "item", "record", "extended"]).to_csv(
        out / "changes.csv", index=False)
    pd.DataFrame(ED, columns=["fold", "h", "scenario", "policy", "item"]).to_csv(out / "lower_edge.csv", index=False)
    I.to_csv(out / "identity.csv", index=False)
    check("old_rows_reproduce_every_recorded_selection", bool(len(I)) and (I.n_diff == 0).all(),
          f"({int((I.n_diff > 0).sum())} of {len(I)} policy-scenario cells differ)")
    check("daily_sums_equal_totals", max(DS) <= 1e-4, f"(max relative {max(DS):.1e})")
    check("every_job_done", len(I) == exp_cells and len(I.groupby(["fold", "h"])) == len(jobs),
          f"({len(jobs)} jobs, {len(I)} of {exp_cells} expected policy-scenario cells)")
    ok = all(c["ok"] for c in checks.values())
    summ = dict(changed_items=len(CH), lower_edge_items=len(ED))
    detail = dict(stage="S6E", smoke=bool(a.dry or a.only), ok=ok, seconds=round(time.time() - t0, 1), checks=checks, summary=summ,
                  outputs={p.name: md5(p) for p in sorted(out.glob("*")) if p.is_file()}, inputs=inputs,
                  code={p.name: md5(p) for p in (Path(__file__), HERE / "ce1_s6.py", HERE / "ce1_dispatch.py",
                                                  HERE / "ce1_load.py")},
                  versions=dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__),
                  host=platform.node())
    json.dump(detail, open(runs / "S6E.detail.json", "w"), indent=1, default=float)
    now = dt.datetime.now().isoformat(timespec="seconds")
    json.dump(dict(stage="S6E", ok=ok, error="" if ok else "checks failed", started=now, finished=now,
                   minutes=round((time.time() - t0) / 60, 1), host=platform.node(), smoke=False),
              open(runs / "S6E.json", "w"), indent=2)
    log(f"{'ALL CHECKS PASSED' if ok else 'SOME CHECKS FAILED'}: {summ}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
