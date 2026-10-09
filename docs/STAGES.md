# Stages and scripts

Stages of the Windows runner (`03_code/ce1/ce1_run.ps1`), in run order:

| Stage | Script | Content |
|---|---|---|
| S0 | `ce1_envcheck.py` | Environment check (packages, data inputs, free disk) |
| S1 | `ce1_preprocess.py` | Causal preprocessing of both sites, fold-specific clear-sky scalars |
| S1M | `ce1_pvmap.py` | Available-PV plant model for Yulara |
| S1L | `ce1_load.py` | Representative load: retrieval and scales |
| S2 | `ce1_folds.py` | Per-fold GBM point and quantile caches, both sites |
| S4 | `ce1_point.py` | Point-forecast benchmark (classical models every fold, GRU and GRU-TCN on fold 2024) |
| S5 | `ce1_conformal.py` | Interval layer: static conformal methods, baselines, ACI in event time |
| S6 | `ce1_s6.py` | Load-based dispatch, every fold, horizon and scenario |
| S6P | `ce1_s6.py` | Load-free proxy dispatch (fold-2024 variants) |
| S6B | `ce1_s6.py` | Battery scenarios |
| S48 | `ce1_wp48.py` | Fold-2024 target sensitivities (records SF and B) |
| S48D | `ce1_wp48.py` | Dispatch steps of S48 only (S6, S6B) on its stored forecast records, written into S48 |
| S2X | `ce1_folds.py` | Additional GBM quantiles (tau 0.255 to 0.500) for theta 0.500 to 0.745, Yulara |
| S6X | `ce1_s6.py` | Dispatch at the added low theta levels |
| S6P2 | `ce1_s6.py` | Proxy dispatch on the extended theta set |
| S56 | `ce1_wp56.py` | Dispatch-realism sensitivities (contingency reserve, no-load fuel, minimum loading) |
| S57 | `ce1_wp57.py` | First half of each calibration year scored with scores fitted on its second half |
| S1E | `ce1_wp67_pc.py` | Preprocessing extended to 2025 |
| S2E | `ce1_wp67_pc.py` | Caches for fold 2025, the DKASC one-step lag and the missing-step refits |
| S5E | `ce1_wp67_pc.py` | Interval layer on the S2E caches |

Scripts in `03_code/ce1/` (the two sensor analyses are in `03_code/ce1/audits/`). Script and result-folder names carry the stage label (`s63`, `S63`) or the analysis label (`wp56`, `WP56`) of the part of the study they belong to. Rules are identified by the decision record (D-number) that fixed them, see `REGISTERED_RULES.md`.

| Script | Content |
|---|---|
| `2026-09-27-sensor-orientation.py` | Orientation of the two Yulara irradiance sensors: fits the plane (tilt, azimuth) of each channel on clear-sky segments with a Hay-Davies transposition. |
| `2026-09-27-wp47-sensor-record.py` | Analysis of the Yulara irradiance sensor record (blocks a1 to a16): channel exchange calendar, fitted sensor planes, logging scale of the tilted sensor and the correction of the 2024 horizontal-sensor step. |
| `ce1_aci.py` | Adaptive conformal inference with delayed feedback in event time. |
| `ce1_adaptive.py` | Stage S51: adaptive reserve level (Protocol A-D). |
| `ce1_causal.py` | Strictly causal preprocessing with a fold-specific clear-sky scalar. |
| `ce1_conformal.py` | Stage S5: the interval layer for every site, fold and horizon, from the stage S2 caches. |
| `ce1_decision.py` | Stage S7D: decision-value contrasts (rules DR1 to DR6, D-091) from the stage S6, S6B and S6P outputs, with paired day-block bootstrap intervals (B = 10,000, seed 42). |
| `ce1_dispatch.py` | Decision layer: reserve policies as one-sided lower bounds on GHI at level theta, mapped to counted PV, on the theta set 0.750 to 0.990 (step 0.005) plus 0.991 to 0.999 (step 0.001), united with the critical fractile of every cost ratio. |
| `ce1_envcheck.py` | Stage S0: environment check for the Windows runner. |
| `ce1_extend.py` | Stage S6E: the extended load-model record (theta set from 0.500, D-092). |
| `ce1_fig1.py` | System and method schematic (the first figure of the article), drawn in vector by code. |
| `ce1_figures.py` | Draws every figure of the article and the graphical abstract from the stage outputs under 04_results/ce1/, asserts that the plotted values equal the reported values, and writes a manifest (source and output md5, sizes, alt text). |
| `ce1_folds.py` | Stage S2: per-fold point and quantile caches for both sites. |
| `ce1_literal.py` | Value checks against reference values from earlier runs. |
| `ce1_load.py` | Load-based dispatch model of the Yulara mini-grid. |
| `ce1_paths.py` | Resolves the result paths read by ce1_figures.py. |
| `ce1_point.py` | Stage S4: point-forecast benchmark on the causal folds. |
| `ce1_preprocess.py` | Stage S1: causal preprocessing for both sites. |
| `ce1_pvmap.py` | Stage S1M: available-PV plant model for Yulara. |
| `ce1_s6.py` | Stage S6 and its variants (S6P, S6B, S6X, S6P2): multi-year load-based dispatch of every reserve policy, in every fold, horizon and registered scenario, with the selection protocols A-R and A-C. |
| `ce1_s63.py` | Stage S63: forecasting diagnostics registered in D-122 (order statistics, lower-tail tests, complete-day coverage and related items), Yulara, six folds, from the cached forecasts. |
| `ce1_s64.py` | Stage S64: dispatch items registered in D-122 and read by D-123 (spill-matched load scale, commitment flexibility and related sensitivities), 15 min, six folds, both load shapes. |
| `ce1_s67.py` | Stage S67: PV doubled and tripled (with and without the largest-unit reserve), the sorted rerun and the 2022 load scale (rules D-129, D-131). |
| `ce1_s69.py` | Stage S69: the plain, two-term N-1 and additive N-1 constructions, and the persistence fraction, for every selection rule (rule D-136). |
| `ce1_s69_read.py` | Readings and descriptive items of stage S69 (D-136), from 04_results/ce1/S69. |
| `ce1_sensor.py` | The Yulara horizontal irradiance record (D-083). |
| `ce1_stats.py` | Stage S7: statistical evidence from the S4 and S5 outputs (rules D-089). |
| `ce1_wp19.py` | Evidence tables for the D-071 reading (end-of-day battery energy) and for the A-R fuel saving against the handbook rule (DR1). |
| `ce1_wp21_data.py` | Data-description numbers (grid slots, measured slots, first-hour share, regime shares) from the S1 and S2 outputs. |
| `ce1_wp46.py` | Stage S46: regime-threshold robustness of the coverage results. |
| `ce1_wp48.py` | Stage S48: fold-2024 target sensitivities (rule D-090). |
| `ce1_wp48_read.py` | Reading of stage S48 under D-090: coverage and decision results of fold 2024 on the primary record against the two alternative records. |
| `ce1_wp52.py` | Storage as reserve (rule D-099): diesel capacity and fuel released by the battery, and the two-by-two decomposition of battery and forecast against the handbook rule, from stage S6E. |
| `ce1_wp55.py` | Start-up fuel, unserved-energy steps and events, and the target sweep, from existing outputs. |
| `ce1_wp56.py` | Stage S56: dispatch-realism sensitivities (rule D-112): contingency reserve for the largest online unit, doubled no-load fuel, and minimum loading of 30% and 50%, every policy and level, 15 min, six folds. |
| `ce1_wp56_read.py` | Reading of stage S56 under D-112 and D-113: oracle saving, A-R and A-D savings, target met, and the share of steps where the PV term binds. |
| `ce1_wp57.py` | Matched static baselines for Protocol A-D (rule D-112): half allowance, cross-fitted halves of the calibration year, and all earlier test years. |
| `ce1_wp58.py` | One-sided exceedance of the lower bounds the decision uses, by regime, and the complete-history persistence ensemble as probabilistic reference (rule D-112). |
| `ce1_wp59c.py` | Seasonal coverage of the static methods and the regime mix behind marginal ICP's transitional coverage in test year 2020, from the S2 caches and S5 records. |
| `ce1_wp61aw.py` | Stage AW: daily reserve controller with the latent level clipped from above at h (rule D-117). |
| `ce1_wp61b.py` | Stage S61B: battery reserve credit limited to the available PV measured at issue time, R_b' = min(R_b, P_av(t_issue)), 0 at night, for the battery scenarios of stage S6. |
| `ce1_wp61e.py` | Unserved-energy events with their size (peak shortfall, energy, duration, start hour, online capacity) for Protocols A-R and A-D, re-simulated from the S51 level paths. |
| `ce1_wp61read.py` | Selection-rule tables, load-scale comparison and issue-time battery credit, read from existing outputs. |
| `ce1_wp61ts.py` | Timestamp-convention test of the DKASC 5-min irradiance values (rule D-118): the offset that best aligns measured GHI with the clear-sky curve on clear days. |
| `ce1_wp61ts_diag.py` | Post-hoc diagnostic of the timestamp test (not the registered rule): looser clear-day filter and the sensor plane fitted jointly with the offset. |
| `ce1_wp63.py` | Selection-rule table, sign tests, unserved-energy events and single-value traces, read from existing stage outputs. |
| `ce1_wp65_picp.py` | Fold-2024 coverage (PICP) at full precision, recomputed from the row-level coverage flags of stage S5. |
| `ce1_wp65_ramps.py` | Step changes of the mapped PV signal against the measured pooled output of the three uncurtailed arrays, 2023 and 2024. |
| `ce1_wp67.py` | Analyses from existing outputs under the rules of D-129, one subcommand per item. |
| `ce1_wp67_c3lag.py` | Coverage check C3 of the held-back fold 2025 and the DKASC one-step-lag reading (rules D-129, D-132). |
| `ce1_wp67_f2025.py` | The held-back test year 2025: dispatch of every selection rule at 15 min, both load shapes, with nothing fitted or tuned on 2025 (rule D-129). |
| `ce1_wp67_f2025_read.py` | Readings of the held-back fold 2025 (rule D-129). |
| `ce1_wp67_fc.py` | Lower-tail scores at the selected levels and a block-bootstrap exceedance test (rule D-129, items 8 and 9). |
| `ce1_wp67_item13.py` | Rule D-129 item 13: binding share of the PV cap, the largest PV falls on the dispatch signal, and related plant-model checks. |
| `ce1_wp67_item14.py` | Rule D-129 item 14: bound on unserved energy when daytime steps without a measured GHI are filled as total PV loss. |
| `ce1_wp67_pc.py` | Runner stages S1E, S2E and S5E: the record extended to 2025, the item-14 refits, the DKASC one-step lag and the caches and intervals of fold 2025 (rules D-129, D-131). |
| `ce1_wp67_precheck2025.py` | Sensor-record check of the 2025 Yulara data (channel state and fitted planes) before any 2025 forecast (rules D-129, D-131). |
| `ce1_wp67_sim.py` | Re-simulation of every selection rule at 15 min with the step record, from the stored inputs and the unchanged dispatch, for the D-129 step 2 items. |
| `ce1_wp72_trace.py` | Record counts per site and period, and the fleet's N-1 firm capacity against the representative load. |
| `ce1_wp74_trace.py` | Paired lead of the quantile policies over the persistence fraction, unserved-energy event sizes, the 2025 record and fold tables, and the DKASC sensor tilt 2020 to 2025, from existing outputs. |
| `ce1_wp76_trace.py` | Complete days and forecast-covered savings in fold 2025, and event-energy values, from existing files. |
| `ce1_wp87.py` | Dispatch simulator comparison. |

## Run order

Commands run from the repository root. Runner stages can also be run without PowerShell, from `03_code`:
`python ce1/<script> --stage <stage>` (script as in the stage table). Each step reads only outputs of the
steps before it; scripts that check a stage's run record need that stage's completed (non-smoke) run.

| Step | Command |
|---|---|
| 1 | With only the 2016 to 2024 Yulara files in the raw folder: `python3 03_code/preprocessing/p2_clean.py --legacy-gapfill`, then `p3_clearsky_regimes.py` and `dkasc_prepare.py` (v1 cleaned and regime files, used by the S1 cross-checks). Add the 2025 file before step 19 |
| 2 | runner stages `S0,S1,S1M,S1L,S2,S6,S6P,S6B,S4,S5` |
| 3 | from `03_code`: `python ce1/ce1_stats.py --stage S7` |
| 4 | runner stage `S48` (stage `S48D` reruns only its dispatch steps on the stored S48 records), then `python3 03_code/ce1/ce1_wp48_read.py` |
| 5 | `python3 03_code/ce1/ce1_decision.py --root .` (S7D) and `python3 03_code/ce1/ce1_wp46.py --root .` (S46) |
| 6 | runner stages `S2X,S6X,S6P2` |
| 7 | `python3 03_code/ce1/ce1_extend.py --root .` (S6E), then `python3 03_code/ce1/ce1_decision.py --root . --s6 S6E --s6b S6E --s6p S6P2 --out S7E` |
| 8 | `ce1_wp19.py --root .`, `ce1_wp21_data.py --root .`, `ce1_wp52.py --root .` |
| 9 | `python3 03_code/ce1/ce1_adaptive.py --root . --ext --out S51 --only <year>:1,<year>:3,...` for the 5 and 15 min jobs of 2019 to 2024, and the same with `--out S51S` for the 30 and 60 min jobs (`:6`, `:12`) |
| 10 | `python3 03_code/ce1/ce1_wp55.py --paths`, then `--analyse` |
| 11 | runner stages `S56,S57`, then `ce1_wp56_read.py` and `ce1_wp57.py --analyse --root .` |
| 12 | `ce1_wp58.py --root .`, `ce1_wp59c.py --root .` |
| 13 | `ce1_wp61aw.py --root .` (AW), `ce1_wp61b.py --root .` (S61B), `ce1_wp61e.py --root .`, `ce1_wp61read.py --root .`, `ce1_wp61ts.py --root .`, `ce1_wp61ts_diag.py --root .` |
| 14 | `ce1_wp63.py --root .` |
| 15 | `ce1_s63.py --root .` (S63); `ce1_s64.py --root . --fold <year>` for 2019 to 2024, then `ce1_s64.py --root . --analyse` (S64) |
| 16 | `ce1_wp65_picp.py --root .`, `ce1_wp65_ramps.py` |
| 17 | `ce1_s67.py --root . --fold <year>` for 2019 to 2024, then `ce1_s67.py --root . --analyse` (S67) |
| 18 | `ce1_wp67_sim.py --root . --all`; then `ce1_wp67.py <item>` for `c1`, `item1`, `item2`, `item3`, `item6`, `item10`, `item11`, `items_sim`, `item12`; `ce1_wp67_fc.py`; `ce1_wp67_item13.py all` |
| 19 | `ce1_wp67_precheck2025.py`, runner stages `S1E,S2E,S5E`, then `ce1_wp67_item14.py`, `ce1_wp67_c3lag.py`, `ce1_wp67_f2025.py --root . --fold 2024` (validation), `ce1_wp67_f2025.py --root . --fold 2025`, `ce1_wp67_f2025_read.py` |
| 20 | `ce1_s69.py --root . --fold <year>` for 2019 to 2024 in order, `ce1_s69.py --root . --assemble`, `ce1_s69.py --root . --s69b --fold <year>` for 2019 to 2024 in order, `ce1_s69.py --root . --s69b --assemble`, then `ce1_s69_read.py` and `ce1_s69_read.py --b` |
| 21 | With the `04_results/ce1` folder of release 2.0.0 placed at `04_results/ce1_v200` (read by `v20`): `ce1_wp87.py --variant <v> --fold <year>` for 2019 to 2024 in order, then `ce1_wp87.py --variant <v> --assemble`, for `v20`, `c02`, `c03`, `c23`, `f7a` and `f7b`; then `ce1_wp87.py --read` (dispatch simulator comparison) |
| 22 | `ce1_wp72_trace.py`, `ce1_wp74_trace.py`, `ce1_wp76_trace.py` |
| 23 | `ce1_figures.py` |

Scripts in steps 8 and later are in `03_code/ce1/` and are run as `python3 03_code/ce1/<script> ...`.
