"""Retained direct-file entry for run_long_integration_global."""
import importlib
import runpy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "compat"))

if __name__ == "__main__":
    runpy.run_module('ocean_solver.runtime.entry', run_name="__main__")
else:
    sys.modules[__name__] = importlib.import_module('ocean_solver.runtime.entry')
