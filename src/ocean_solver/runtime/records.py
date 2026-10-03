"""Compatibility alias; implementation owner: ocean_solver.io.records."""
import importlib
import runpy
import sys

if __name__ == "__main__":
    runpy.run_module('ocean_solver.io.records', run_name="__main__")
else:
    sys.modules[__name__] = importlib.import_module('ocean_solver.io.records')
