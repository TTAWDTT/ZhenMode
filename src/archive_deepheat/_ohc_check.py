"""Validate the OHC unit scale: total OHC of the init and the 10-yr ckpt
must land near the ~33,000 ZJ quoted in the spin-up series.  If total OHC is
right, the +108.95 ZJ/yr one-step tendency is real and the gap vs the measured
+5.9 ZJ/yr trend is a state/config difference, not a unit bug.
"""
import sys, glob, os
import numpy as np

ROOT = r"C:\Users\zhen.luo\ocean_solver"
init = np.load(os.path.join(ROOT, "init_fields_g360x120.npz"))
lat, lon, z = init["lat"], init["lon"], init["z"]
wet = init["wet_mask"]
print("init keys:", list(init.keys()))
print("T_init shape", init["T_init"].shape, "wet_mask shape", wet.shape)
print("z =", np.asarray(z, float))

# rebuild the grid exactly as the solver does
sys.path.insert(0, os.path.join(ROOT, "src"))
from grid import make_global_grid, GlobalGridConfig
from config import DEFAULT_CONFIG
g = make_global_grid(GlobalGridConfig(lat_max=60.0, ny=120),
                     DEFAULT_CONFIG.bathymetry_file, smooth_passes=30, min_depth=100.0)
AREA = np.asarray(g.dx_2d, float) * float(g.dy)
DZN = np.array([5, 7.5, 12.5, 17.5, 22.5, 25, 37.5, 50, 75, 150, 350, 750, 1500, 2000], float)
wm3 = np.asarray(g.wet_mask_3d, float)
# consistency: does grid wet3 match the init file's wet_mask?
print("\ngrid wet3 sum", wm3.sum(), " init wet_mask sum", wet.sum(),
      " g.wet_mask sum", np.asarray(g.wet_mask).sum())
print("AREA sum (1e12 m2) =", AREA.sum() / 1e12)

vol = wm3 * AREA[:, :, None] * DZN[None, None, :]
print("ocean volume (1e18 m3) =", vol.sum() / 1e18, " (reference 1.2035)")

def ohc(T):
    """ZJ, referenced to T_ref=15 C."""
    T = np.asarray(T, float)
    return float(((T - 15.0) * vol).sum() * 1025.0 * 3992.0 / 1e21)

print("\nOHC of WOA init (T-Tref=15C)  = %+.1f ZJ" % ohc(init["T_init"]))
for f in sorted(glob.glob(os.path.join(ROOT, "results", "ckpt_*.npz"))):
    d = np.load(f)
    if "T" not in d.files:
        continue
    try:
        print("OHC %-34s = %+.1f ZJ   (cur_step=%s)"
              % (os.path.basename(f), ohc(d["T"]), d["cur_step"] if "cur_step" in d.files else "?"))
    except Exception as e:
        print(os.path.basename(f), "ERR", e)
