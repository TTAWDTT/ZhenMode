import sys; sys.path.insert(0, "src")
from grid import make_global_grid, GlobalGridConfig
from config import DEFAULT_CONFIG
import numpy as np
g = make_global_grid(GlobalGridConfig(lat_max=60.0, ny=120),
                     DEFAULT_CONFIG.bathymetry_file, smooth_passes=30, min_depth=100.0)
print(type(g))
for a in sorted(dir(g)):
    if a.startswith("_") or callable(getattr(g, a)):
        continue
    v = getattr(g, a)
    sh = getattr(v, "shape", None)
    print(f"  {a}: {sh if sh is not None else type(v).__name__} {'' if sh is not None else v}")
