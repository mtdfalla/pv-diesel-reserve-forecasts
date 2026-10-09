#!/usr/bin/env python3
"""Value checks against reference values from earlier runs. Under the release 2.0.0
simulator (the registered_v200 context of ce1_load) a mismatch fails the calling check. Under
the current simulator the reference and new values are written to <stage folder>/literal_diffs.json
and the check passes. Structural checks (row, case and cell counts) do not use this module.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import ce1_load as LM  # noqa: E402

FILE = "literal_diffs.json"


def registered() -> bool:
    """True when the registered v2.0.0 simulator is in force."""
    return LM.sim_defaults()["version"] == "v2.0.0"


def _plain(v):
    """JSON-safe copy of a value (numpy scalars, tuples, nested lists and dicts)."""
    if isinstance(v, dict):
        return {str(k): _plain(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_plain(x) for x in v]
    if hasattr(v, "item") and not isinstance(v, (str, bytes)):
        try:
            return v.item()
        except Exception:
            pass
    if isinstance(v, (str, int, float, bool)) or v is None:
        return v
    return str(v)


class Literal:
    def __init__(self, stage_dir, script: str):
        self.path = Path(stage_dir) / FILE
        self.script = script
        self.fresh = True  # the first record of a run replaces this script's old entries

    def __call__(self, name: str, new, old, equal=None) -> bool:
        eq = bool(new == old) if equal is None else bool(equal)
        if registered():
            return eq
        self.record(name, new, old, eq)
        return True

    def record(self, name, new, old, eq):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            data = json.loads(self.path.read_text(encoding="utf-8")) if self.path.exists() else {}
        except Exception:
            data = {}
        if self.fresh:
            data[self.script] = {}
            self.fresh = False
        data.setdefault(self.script, {})[name] = dict(old=_plain(old), new=_plain(new), equal=eq,
                                                       simulator=LM.sim_defaults()["version"])
        self.path.write_text(json.dumps(data, indent=1), encoding="utf-8")


def selftest() -> dict:
    import tempfile
    r = {}
    with tempfile.TemporaryDirectory() as d:
        L = Literal(d, "t.py")
        r["v21_records_and_passes"] = L("a", 1, 2) is True and L("b", 3, 3) is True
        j = json.loads((Path(d) / FILE).read_text())
        r["v21_file_content"] = (j["t.py"]["a"] == dict(old=2, new=1, equal=False, simulator="v2.1")
                                 and j["t.py"]["b"]["equal"] is True)
        with LM.registered_v200():
            r["v200_fails_on_change"] = L("c", 1, 2) is False and L("d", 2, 2) is True
        j = json.loads((Path(d) / FILE).read_text())
        r["v200_records_nothing"] = "c" not in j["t.py"]
        L2 = Literal(d, "t.py")
        L2("e", (1.5, 2), (1.5, 3))
        j = json.loads((Path(d) / FILE).read_text())
        r["rerun_replaces_script_entries"] = set(j["t.py"]) == {"e"} and j["t.py"]["e"]["old"] == [1.5, 3]
        L3 = Literal(d, "u.py")
        L3("f", 1, 1, equal=abs(1 - 1.01) < 0.05)
        j = json.loads((Path(d) / FILE).read_text())
        r["scripts_kept_apart"] = set(j) == {"t.py", "u.py"}
        import numpy as np
        L3("g", np.float64(0.5), np.int64(1))
        j = json.loads((Path(d) / FILE).read_text())
        r["numpy_values_serialised"] = j["u.py"]["g"]["new"] == 0.5 and j["u.py"]["g"]["old"] == 1
    r["default_is_v21"] = not registered()
    return r


if __name__ == "__main__":
    res = selftest()
    for k, v in res.items():
        print("ok  " if v else "FAIL", k)
    sys.exit(0 if all(res.values()) else 1)
