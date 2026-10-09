#!/usr/bin/env python3
"""Resolves the result paths read by ce1_figures.py. Without a phase file, as released, every record
resolves to 04_results/ce1.

Usage (from the repository root unless the path starts at ce1/, then from 03_code):
    python3 03_code/ce1/ce1_paths.py --selftest
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PHASE_FILE = HERE / "ce1_results_phase.json"
V200 = "ce1_v200"

ARCHIVED = (
    "AW", "S6", "S6B", "S6X", "S6E", "S7D", "S7E", "S51", "S51S", "S56", "S57", "S61B", "S63", "S64", "S67",
    "S69", "S69b", "WP18X", "WP19", "WP52", "WP55", "WP56", "WP57", "WP58", "WP61", "WP63", "WP67", "WP68",
    "WP74", "WP76",
    "S48/SF/S6", "S48/SF/S6B", "S48/B/S6", "S48/B/S6B", "S48/reading",
)
COPY_BACK = (
    "WP61/timestamp.json", "WP61/timestamp_days.csv", "WP61/timestamp_diag.json",  # ce1_wp61ts, _diag
    "WP67/c3lag", "WP67/c3lag.log", "WP67/precheck2025",  # ce1_wp67_c3lag, _precheck2025
    "WP67/item13/cap_binding.csv", "WP67/item13/cap_measured.csv", "WP67/item13/dkasc_tilt.csv",  # item 13 parts A, C
    "WP67/item13/handbook_pages.json", "WP67/item13/yulara_sensor_rule_days.csv", "WP67/item13/item13_checks.json",
    "WP74/dkasc_tilt_2020_2025.csv", "WP74/s1_folds_2025.csv", "WP74/s1_missing_2025.csv", "WP74/s1_record_2025.csv",
)
STAMP_STAGES = ("AW", "S51", "S51S", "S56", "S57", "S6", "S6B", "S6X", "S61B", "S63", "S64", "S67", "S69", "S69b",
                "S6E", "S7D", "S7E", "WP55", "WP57", "WP58", "WP63")
STAMP_SUFFIXES = (".json", ".detail.json", ".log")


ARCHIVED_ALL = ARCHIVED + tuple(f"_runs/{s}{x}" for s in STAMP_STAGES for x in STAMP_SUFFIXES)


def phase() -> str:
    try:
        return json.loads(PHASE_FILE.read_text(encoding="utf-8")).get("text", "v2.1")
    except FileNotFoundError:
        return "v2.1"


def archived(rel: str) -> bool:
    rel = rel.replace("\\", "/").strip("/")
    return any(rel == a or rel.startswith(a + "/") for a in ARCHIVED_ALL)


class Results(os.PathLike):
    """04_results/ce1: R / "WP63" / "cases.csv" resolves to 04_results/ce1 (to 04_results/ce1_v200 only when a
    phase file sets v2.0.0 and that folder exists). Other attributes are the 04_results/ce1 path's.
    """

    def __init__(self, root: Path, text_phase: str | None = None):
        self.ce1 = Path(root) / "04_results" / "ce1"
        self.v200 = Path(root) / "04_results" / V200
        self.text_phase = text_phase or phase()
        self.use_v200 = self.text_phase == "v2.0.0" and self.v200.exists()

    def __truediv__(self, name):
        rel = str(name).replace("\\", "/").strip("/")
        if self.use_v200:
            if archived(rel):
                return self.v200 / rel
            if any(a.startswith(rel + "/") for a in ARCHIVED_ALL):  # S48, _runs: some members archived
                return _Partial(self, rel)
        return self.ce1 / rel

    def resolve_rel(self, rel: str) -> Path:
        """The path of a 04_results/ce1-relative path under the current phase (for string-joined callers)."""
        rel = rel.replace("\\", "/").strip("/")
        if self.use_v200 and archived(rel):
            return self.v200 / rel
        return self.ce1 / rel

    def __fspath__(self):
        return str(self.ce1)

    def __str__(self):
        return str(self.ce1)

    def __repr__(self):
        return f"Results({self.ce1}, text={self.text_phase}, v200={self.use_v200})"

    def __getattr__(self, k):
        return getattr(self.ce1, k)


class _Partial:
    """A folder of which some subfolders are archived: keeps resolving under the same rule."""

    def __init__(self, res: Results, rel: str):
        self.res, self.rel = res, rel

    def __truediv__(self, name):
        return self.res / f"{self.rel}/{str(name).strip('/')}"

    def __fspath__(self):
        return str(self.res.ce1 / self.rel)

    def __str__(self):
        return self.__fspath__()

    def __getattr__(self, k):
        return getattr(self.res.ce1 / self.rel, k)


def results(root: Path) -> Results:
    return Results(root)


def selftest(verbose=True) -> dict:
    import tempfile
    res = {}

    def ok(name, cond, msg=""):
        res[name] = bool(cond)
        if verbose:
            print(f"  [{'PASS' if cond else 'FAIL'}] {name} {msg}")

    with tempfile.TemporaryDirectory() as t:
        root = Path(t)
        (root / "04_results" / "ce1" / "S1").mkdir(parents=True)
        (root / "04_results" / "ce1" / "WP63").mkdir(parents=True)
        (root / "04_results" / "ce1" / "S48" / "SF" / "S2").mkdir(parents=True)
        (root / "04_results" / V200 / "WP63").mkdir(parents=True)
        (root / "04_results" / V200 / "S48" / "SF" / "S6").mkdir(parents=True)
        R = Results(root, "v2.0.0")
        ok("archived_folder_to_v200", R / "WP63" / "cases.csv" == root / "04_results" / V200 / "WP63" / "cases.csv")
        ok("kept_folder_to_ce1", R / "S1" / "x.parquet" == root / "04_results" / "ce1" / "S1" / "x.parquet")
        ok("nested_archived_to_v200", R / "S48" / "SF" / "S6" / "f.parquet" == root / "04_results" / V200 / "S48" / "SF" / "S6" / "f.parquet")
        ok("nested_kept_to_ce1", R / "S48" / "SF" / "S2" / "f.parquet" == root / "04_results" / "ce1" / "S48" / "SF" / "S2" / "f.parquet")
        ok("slash_form_same", R / "WP63/cases.csv" == R / "WP63" / "cases.csv")
        ok("resolve_rel", R.resolve_rel("WP63/table4_rows.csv") == root / "04_results" / V200 / "WP63" / "table4_rows.csv")
        R2 = Results(root, "v2.1")
        ok("phase_v21_reads_ce1", R2 / "WP63" / "cases.csv" == root / "04_results" / "ce1" / "WP63" / "cases.csv")
        ok("fspath_is_ce1", os.fspath(R) == str(root / "04_results" / "ce1") and R.parent == root / "04_results")
        ok("copy_back_inside_archived", all(archived(c) for c in COPY_BACK))
        ok("stamp_to_v200", R / "_runs" / "S61B.json" == root / "04_results" / V200 / "_runs" / "S61B.json"
           and R / "_runs" / "S1.json" == root / "04_results" / "ce1" / "_runs" / "S1.json")
        ok("stamps_are_archived_stages", all(s in ARCHIVED for s in STAMP_STAGES if s not in ("WP55", "WP57", "WP58", "WP63"))
           and all(s in ARCHIVED for s in ("WP55", "WP57", "WP58", "WP63")))
    return res


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        r = selftest()
        sys.exit(0 if all(r.values()) else 1)
    print(repr(results(Path.cwd())))
