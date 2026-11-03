# pv-diesel-reserve-forecasts

Code and result tables for the article:

> M. Abdelmagid, T. H. M. El-Fouly, M. Maalouf. *A reliability margin enables machine-learning forecasts to replace the full-PV spinning-reserve rule in a remote PV-diesel mini-grid model: a six-year chronological evaluation*.

Version 2.1.0. MIT licence. Archived on Zenodo: https://doi.org/10.5281/zenodo.23262936.

The pipeline rebuilds the irradiance and plant records from the public source files, fits point and quantile
forecasters year by year with causal inputs, calibrates prediction intervals (split conformal, CQR, Mondrian
variants, adaptive conformal inference with delayed feedback) and evaluates reserve rules in a dispatch model of
the Yulara PV-diesel plant (Northern Territory, Australia) with a representative load, at the 5-min step over six
test years.

## Data

The source data are public and are not redistributed here.

* DKASC five-minute monitoring data, <https://dkasolarcentre.com.au/download>: Yulara weather station 2016 to
  2025 as `02_data/Original Dataset/Yulara_<year>.csv`, Alice Springs 2020 to 2025 as
  `02_data/DKASC/raw/Alice_Springs_<year>.csv`.
* NREL ComStock (TMY3 release 1, Maricopa County, large hotel), downloaded by stage S1L from the URL in
  `03_code/ce1/ce1_load.py` and checked against the md5 recorded in `04_results/ce1/S1L/load_scales.json`.

## Environment

Python 3.11 or later with the versions in `requirements.txt`. With matplotlib 3.10.9, `ce1_figures.py`
reproduces the article's figures exactly (written to `04_results/ce1/figs/`) when the TeX Gyre Heros font is
installed.

## Running

The full run order is in `docs/STAGES.md`. It starts with the v1 preprocessing (step 1), then the model-fitting
stages, which run through the resumable Windows runner, one stage or a comma-separated chain per call:

    powershell -ExecutionPolicy Bypass -File "<repository folder>\03_code\ce1\ce1_run.ps1" -Stage S0,S1,S1M,S1L

Each stage writes a run record to `04_results/ce1/_runs/` (status, times, input and output md5, package
versions). A test run (`-Smoke`) writes its record as `<stage>_smoke.json` and its outputs to
`<stage>_smoke/`. Analysis scripts that fit no model run directly with Python. `docs/STAGES.md` gives the
stages and scripts, and `docs/REGISTERED_RULES.md` lists the decision rules with the
date each was fixed. Modules with a `--selftest` option check themselves. Code md5 values inside result
files refer to the files that produced those results. The released files compute the same results. They
differ in comments, documentation, message and key labels, and in accepting a stage's run record when its
`ok` flag is set.

## Contents

* `03_code/ce1/` pipeline, analysis and figure scripts.
* `03_code/conformal`, `r1`, `utils`, `models`, `evaluation` and `preprocessing`: shared modules and the
  v1 preprocessing they use.
* `04_results/ce1/` result tables (model caches, binary intermediates, figures and run records are rebuilt).
* `docs/` stage map and registered rules.
* `RELEASE_RECORD.json` md5 of every file in this release.

Repository: https://github.com/mtdfalla/pv-diesel-reserve-forecasts
