"""Legacy import/script bridge to zhenmode_research.candidates.fv.paired."""
import importlib
import runpy
import sys

if __name__ == "__main__":
    runpy.run_module('zhenmode_research.candidates.fv.paired', run_name="__main__")
else:
    sys.modules[__name__] = importlib.import_module('zhenmode_research.candidates.fv.paired')
