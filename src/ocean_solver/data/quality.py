"""Compatibility alias; implementation owner: ocean_solver.io.data_quality."""
import importlib
import runpy
import sys

if __name__ == "__main__":
    runpy.run_module('ocean_solver.io.data_quality', run_name="__main__")
else:
    sys.modules[__name__] = importlib.import_module('ocean_solver.io.data_quality')
