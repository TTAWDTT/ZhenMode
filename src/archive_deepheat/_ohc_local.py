"""Ground-truth OHC series for the LOCAL tenyr run, computed directly.

The remote _remote_ohc.py numbers are internally inconsistent (a +11356 ZJ
rise over 100 yr cannot coexist with a "full-record trend of +0.34 ZJ/yr").
This recomputes everything from the local snapshots with explicit units and
prints BOTH the per-step rates and the integrated rise so they can be checked
against each other.

  OHC(Tref) = sum( (T - Tref) * vol ) * rho * cp        [J]
  rate from a linear fit of OHC(t) in ZJ against t in years.
"""
import glob
import os
import numpy as np

RHO, CP = 1025.0, 3992.0
SEC_PER_YR = 3.1536e7
ZJ = 1e21

import sys
sys.path.insert(0, "src")
from grid import make_global_grid, GlobalGridConfig
from config import DEFAULT_CONFIG

g = make_global_grid(GlobalGridConfig(lat_max=60.0, ny=120),
                     DEFAULT_CONFIG.bathymetry_file, smooth_passes=30, min_depth=100.0)
AREA = np.asarray(g.dx_2d, float) * float(g.dy)
WET = np.asarray(g.wet_mask_3d, float)
DZ = np.array([5, 7.5, 12.5, 17.5, 22.5, 25, 37.5, 50, 75, 150, 350, 750, 1500, 2000])
VOL = WET * AREA[:, :, None] * DZ[None, None, :]

files = sorted(glob.glob("results/global_tenyr_ms_gm_3d/snap_*.npy"))
print("snapshots:", len(files))
print("ocean volume %.6e m3" % VOL.sum())

OHC = np.zeros(len(files))
per_lvl = np.zeros((len(files), 14))
for i, f in enumerate(files):
    s = np.load(f)
    T = np.asarray(s[0] if s.ndim == 4 else s, float)
    per_lvl[i] = (T * VOL).sum(axis=(0, 1)) / VOL.sum(axis=(0, 1))
    OHC[i] = ((T - 15.0) * VOL).sum() * RHO * CP / ZJ

# assume snap spacing from the runner; infer from OHC monotone progression is unsafe,
# so report as a function of snapshot index AND of years if dt_snap is known.
print("\nassume 30-day snapshot spacing (10 yr / 123 snaps):")
dt_yr = 10.0 / (len(files) - 1)
t = np.arange(len(files)) * dt_yr

print("\n  i    yr     OHC(ZJ,Tref=15)   dOHC/dt(ZJ/yr)   T0     T5     T10    T13")
for i in range(0, len(files), max(1, len(files) // 15)):
    r = (OHC[i] - OHC[i - 1]) / dt_yr if i else np.nan
    print("  %3d %6.2f   %12.2f     %10.2f     %6.2f %6.2f %6.2f %6.2f"
          % (i, t[i], OHC[i], r, per_lvl[i, 0], per_lvl[i, 5],
             per_lvl[i, 10], per_lvl[i, 13]))

print("\nINTEGRATED: OHC[0]=%.2f  OHC[-1]=%.2f  rise=%.2f ZJ over %.2f yr"
      % (OHC[0], OHC[-1], OHC[-1] - OHC[0], t[-1]))
print("equivalent mean rate = %.2f ZJ/yr" % ((OHC[-1] - OHC[0]) / t[-1]))
print("\nSLOPE OF OHC vs t (ZJ/yr), by window:")
for name, sl in [("full", slice(None)),
                 ("last20%", slice(int(0.8 * len(files)), None)),
                 ("last10%", slice(int(0.9 * len(files)), None)),
                 ("last5snap", slice(len(files) - 5, None))]:
    x, y = t[sl], OHC[sl]
    a = np.polyfit(x, y, 1)[0]
    print("   %-10s n=%2d  slope=%+8.3f ZJ/yr   OHC range %.2f .. %.2f"
          % (name, len(x), a, y.min(), y.max()))

print("\nper-level vol-mean T, first vs last:")
print("  k   T_first   T_last    dT")
for k in range(14):
    print("  %2d  %8.4f %8.4f  %+8.4f" % (k, per_lvl[0, k], per_lvl[-1, k],
                                          per_lvl[-1, k] - per_lvl[0, k]))
