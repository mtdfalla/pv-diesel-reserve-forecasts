"""Unserved-energy events with their size (peak shortfall, energy, duration, start hour, online
capacity) for Protocols A-R and A-D, re-simulated from the S51 level paths.

Usage (from the repository root unless the path starts at ce1/, then from 03_code):
    python3 03_code/ce1/ce1_wp61e.py --root .
"""
from __future__ import annotations

import argparse, json, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "utils"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import ce1_adaptive as A  # noqa: E402
import ce1_dispatch as D  # noqa: E402
import ce1_folds as FC  # noqa: E402
import ce1_load as LM  # noqa: E402
import ce1_s6 as S6  # noqa: E402
from ce1_wp55 import events_of, md5  # noqa: E402

H = 3
SCEN = ("main", "shapeB")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(HERE.parents[1]))
    a = ap.parse_args()
    R = Path(a.root) / "04_results" / "ce1"
    out = R / "WP61"
    out.mkdir(parents=True, exist_ok=True)
    checks, inputs = {}, {}

    def check(name, cond, msg=""):
        checks[name] = dict(ok=bool(cond), msg=str(msg))
        print(("PASS  " if cond else "FAIL  ") + name + "  " + str(msg), flush=True)

    geo, m, unit96, scales, _ = S6.load_inputs(R / "S1", R / "S1M", R / "S1L")
    D.extend_theta_low()
    rows = S6.row_table()
    ref = pd.read_csv(R / "WP55" / "ad_totals.csv")
    inputs["WP55/ad_totals.csv"] = md5(R / "WP55" / "ad_totals.csv")
    evrows = []
    for yr in A.FOLDS:
        t1 = time.time()
        fz = R / "S51" / f"f{yr}_h{H}_paths.npz"
        inputs[str(fz.relative_to(R))] = md5(fz)
        pz = np.load(fz)
        specs = [s.split("|") for s in pz["specs"].astype(str)]
        specs = [(p, float(g), float(k)) for p, g, k in specs]
        cache = FC.load_cache_ext(R / "S2" / f"yulara_f{yr}_h{H}.parquet", R / "S2X" / f"yulara_f{yr}_h{H}.parquet")
        P = S6.prepare(cache, H, yr, geo, m, rows)["test"]
        grid, pav, day = P["grid"], P["pav"], P["day"]
        pol_C = {}
        for pol in A.POLICIES:
            ix = [i for i, r in enumerate(rows) if r[0] == pol]
            lv = np.array([rows[i][1] for i in ix])
            o = np.argsort(lv)
            pol_C[pol] = (lv[o], P["C"][np.array(ix)[o]])
        for sname in SCEN:
            sc = S6.SCENARIOS[sname]
            L = S6.load_for(grid, unit96, scales, sc)
            ratings = np.asarray(LM.fleet_for_peak(L.max()), float)
            sim_kw = dict(ratings=list(ratings), min_load=sc["min_load"], fuel=sc["fuel"], battery=sc["battery"])
            lev = pz[f"{sname}__level"].astype(np.float64)
            sel = [k for k, (p, g, kp) in enumerate(specs) if (g == A.GAMMA and kp == A.KAPPA) or g == 0.0]
            idx = {}
            for k in sel:
                levels = pol_C[specs[k][0]][0]
                j = np.abs(levels[None, :].astype(np.float32).astype(np.float64) - lev[k][:, None]).argmin(axis=1)
                assert np.array_equal(levels[j].astype(np.float32), pz[f"{sname}__level"][k])
                idx[k] = j
            C = np.vstack([A.path_row(pol_C[specs[k][0]][1], idx[k], day) for k in sel])
            rr = LM.simulate(L, pav, C, day, H, record=True, **sim_kw)
            check(f"reproduces_S51_f{yr}_{sname}",
                  np.array_equal(rr["daily"]["fuel_L"].astype(np.float32), pz[f"{sname}__fuel_L"][sel])
                  and np.array_equal(rr["daily"]["ue_kWh"].astype(np.float32), pz[f"{sname}__ue_kWh"][sel]))
            ue = rr["rec"]["ue"]  # kW per step
            ev, _, _ = events_of(ue, day)
            hours = grid.hour + grid.minute / 60.0
            for j, k in enumerate(sel):
                pol, g, kp = specs[k]
                proto = "A-D" if g == A.GAMMA else "A-R"
                r_ = ref[(ref.fold == yr) & (ref.scenario == sname) & (ref.policy == pol) & (ref.gamma == g) & (ref.kappa == kp)]
                check(f"events_equal_WP55_f{yr}_{sname}_{pol}_{proto}", len(r_) == 1 and int(r_.events.iloc[0]) == int(ev[j]),
                      (int(ev[j]), None if len(r_) != 1 else int(r_.events.iloc[0])))
                pos = ue[j] > 1e-9
                st = np.flatnonzero(pos & ~np.r_[False, pos[:-1] & (day[1:] == day[:-1])])
                tot_e = 0.0
                for s0 in st:
                    s1 = s0
                    while s1 + 1 < len(pos) and pos[s1 + 1] and day[s1 + 1] == day[s1]:
                        s1 += 1
                    seg = ue[j, s0:s1 + 1]
                    pk = s0 + int(np.argmax(seg))
                    on = rr["rec_on"][j, pk]
                    e = float(seg.sum() * LM.DT)
                    tot_e += e
                    evrows.append(dict(fold=yr, scenario=sname, policy=pol, protocol=proto, start=str(grid[s0]),
                                       start_hour=float(hours[s0]), steps=int(s1 - s0 + 1), energy_kWh=e,
                                       peak_kW=float(seg.max()), online_kW=float(rr["rec"]["KU"][j, pk]),
                                       largest_online_kW=float(ratings[on].max()) if on.any() else 0.0,
                                       load_kW=float(L[pk]), pav_kW=float(pav[pk])))
                check(f"event_energy_sums_to_UE_f{yr}_{sname}_{pol}_{proto}",
                      abs(tot_e - float(rr["total"]["ue_kWh"][j])) <= 1e-6 * max(1.0, tot_e), (tot_e, float(rr["total"]["ue_kWh"][j])))
        print(f"fold {yr}: {time.time() - t1:.0f} s", flush=True)
    E = pd.DataFrame(evrows)
    E["peak_share_online"] = E.peak_kW / E.online_kW
    E["peak_share_largest"] = E.peak_kW / E.largest_online_kW
    E.to_csv(out / "events.csv", index=False)
    grp = E.groupby(["scenario", "protocol", "policy"])
    S = grp.agg(events=("peak_kW", "size"), peak_kW_median=("peak_kW", "median"), peak_kW_p90=("peak_kW", lambda x: x.quantile(0.9)),
                peak_kW_max=("peak_kW", "max"), steps_median=("steps", "median"), steps_max=("steps", "max"),
                share_largest_median=("peak_share_largest", "median"), share_largest_max=("peak_share_largest", "max"),
                above_largest=("peak_share_largest", lambda x: int((x > 1).sum())),
                energy_median=("energy_kWh", "median")).reset_index()
    S.to_csv(out / "events_summary.csv", index=False)
    ok = all(c["ok"] for c in checks.values())
    (out / "events_checks.json").write_text(json.dumps(dict(ok=ok, checks=checks, inputs_md5=inputs,
                                                            outputs_md5={f: md5(out / f) for f in ("events.csv", "events_summary.csv")},
                                                            code={"ce1_wp61e.py": md5(Path(__file__))}), indent=1), encoding="utf-8")
    print(f"{sum(c['ok'] for c in checks.values())} of {len(checks)} checks pass; " + ("WP61E PASS" if ok else "WP61E FAIL"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
