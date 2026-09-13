"""Vertical T/S profile of the WOA init and the 10-yr checkpoint.

Mean ocean T should be ~3.5 C.  If the init shows ~15 C the wet mask /
ghost sentinel handling in my budget is wrong, and every ZJ number I have
computed so far is suspect.
"""
import sys, os
import numpy as np
ROOT = r"C:\Users\zhen.luo\ocean_solver"
sys.path.insert(0, os.path.join(ROOT, "src"))
from grid import make_global_grid, GlobalGridConfig
from config import DEFAULT_CONFIG

g = make_global_grid(GlobalGridConfig(lat_max=60.0, ny=120),
                     DEFAULT_CONFIG.bathymetry_file, smooth_passes=30, min_depth=100.0)
AREA = np.asarray(g.dx_2d, float) * float(g.dy)
DZN = np.array([5, 7.5, 12.5, 17.5, 22.5, 25, 37.5, 50, 75, 150, 350, 750, 1500, 2000], float)
wm3 = np.asarray(g.wet_mask_3d, float)
vol = wm3 * AREA[:, :, None] * DZN[None, None, :]

init = np.load(os.path.join(ROOT, "init_fields_g360x120.npz"))
d10 = np.load(os.path.join(ROOT, "results", "ckpt_tenyr_ms_gm.npz"))

def prof(T, tag):
    T = np.asarray(T, float)
    print(f"\n--- {tag} ---")
    print("  raw field: min %.3f  max %.3f  mean %.3f" % (T.min(), T.max(), T.mean()))
    print("  wet-only : min %.3f  max %.3f  mean %.3f"
          % (T[wm3 > 0.5].min(), T[wm3 > 0.5].max(), T[wm3 > 0.5].mean()))
    print("  k   z_top   vol-mean T   raw-mean T   n_wet")
    ZTOP = [0, 5, 15, 30, 50, 75, 100, 150, 200, 300, 500, 1000, 2000, 4000]
    for k in range(14):
        m = wm3[:, :, k] > 0.5
        v = vol[:, :, k].sum()
        tm = (T[:, :, k] * vol[:, :, k]).sum() / v if v > 0 else np.nan
        print("  %2d  %-7.0f  %8.4f     %8.4f    %6d" % (k, ZTOP[k], tm, T[:, :, k].mean(), m.sum()))
    tot = (T * vol).sum() / vol.sum()
    print("  VOLUME-MEAN T (wet) = %.4f C   [real ocean ~3.5 C]" % tot)

prof(init["T_init"], "WOA init T")
prof(d10["T"], "10-yr ckpt T")
prof(init["S_init"], "WOA init S (not vol-averaged)")

print("\n=== ghost sentinel check ===")
print("init T at cells where wet3==0: unique-ish min %.3f max %.3f"
      % (np.asarray(init["T_init"])[wm3 == 0].min(), np.asarray(init["T_init"])[wm3 == 0].max()))
print("init T at wet cells: min %.3f max %.3f"
      % (np.asarray(init["T_init"])[wm3 > 0.5].min(), np.asarray(init["T_init"])[wm3 > 0.5].max()))
