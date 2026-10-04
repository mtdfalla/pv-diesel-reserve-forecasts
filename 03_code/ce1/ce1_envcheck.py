"""Stage S0: environment check for the Windows runner. Checks the Python packages, the data
inputs and free disk space, and writes 04_results/ce1/_runs/S0.detail.json. A missing item is
listed with the command that installs it.
"""
import importlib
import json
import platform
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
NEED = {"numpy": "numpy", "pandas": "pandas", "scipy": "scipy", "sklearn": "scikit-learn",
        "pyarrow": "pyarrow", "matplotlib": "matplotlib", "pvlib": "pvlib", "torch": None}
DATA = ["02_data/Original Dataset", "02_data/DKASC"]

missing, versions = [], {}
for mod, pipname in NEED.items():
    try:
        m = importlib.import_module(mod)
        versions[mod] = getattr(m, "__version__", "ok")
    except Exception:
        missing.append(mod)
        versions[mod] = None

data = {d: (ROOT / d).exists() for d in DATA}
free_gb = round(shutil.disk_usage(ROOT).free / 1e9, 1)
cpu = __import__("os").cpu_count()

detail = dict(stage="S0", disk_fail_gb=1.0, disk_warn_gb=3.0, python=sys.version.split()[0], platform=platform.platform(),
              cpu_count=cpu, free_disk_gb=free_gb, versions=versions, data_present=data,
              missing=missing)
out = ROOT / "04_results" / "ce1" / "_runs" / "S0.detail.json"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(detail, indent=2), encoding="utf-8")

print(json.dumps(detail, indent=2))
problems = []
if missing:
    pip = [NEED[m] for m in missing if NEED[m]]
    if pip:
        problems.append("Install with:  python -m pip install " + " ".join(pip))
    if "torch" in missing:
        problems.append("Install torch with:  python -m pip install torch --index-url "
                        "https://download.pytorch.org/whl/cpu")
if not all(data.values()):
    problems.append("Missing data folders: " + ", ".join(k for k, v in data.items() if not v))
DISK_FAIL_GB, DISK_WARN_GB = 1.0, 3.0
if free_gb < DISK_FAIL_GB:
    problems.append("Less than %.1f GB free on the project drive (%s GB)" % (DISK_FAIL_GB, free_gb))
elif free_gb < DISK_WARN_GB:
    print("WARNING: only %s GB free on the project drive; enough for CE1 (about 1 GB needed), "
          "but little margin." % free_gb)
if problems:
    print("\nENVIRONMENT NOT READY:\n  " + "\n  ".join(problems))
    sys.exit(1)
print("\nENVIRONMENT READY (%d CPU cores, %s GB free)" % (cpu, free_gb))
