"""Legacy import/script bridge to ocean_solver.data.sources."""
import importlib
import runpy
import sys

if __name__ == "__main__":
    runpy.run_module('ocean_solver.io.input_sources', run_name="__main__")
else:
    sys.modules[__name__] = importlib.import_module('ocean_solver.io.input_sources')
