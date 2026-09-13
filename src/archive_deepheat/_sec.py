"""CORRECTED secular per-level heat budget of the ten-year run."""
import numpy as np, sys
import jax.numpy as jnp
sys.path.insert(0, "src")
from grid import make_global_grid, GlobalGridConfig
from config import DEFAULT_CONFIG, PhysicsConfig
import jax_solver_global as JS

RHO_0, C_P = 1025.0, 3992.0
g = make_global_grid(GlobalGridConfig(lat_max=60.0, ny=120),
                     DEFAULT_CONFIG.bathymetry_file, smooth_passes=30, min_depth=100.0)
AREA = np.asarray(g.dx_2d, float) * float(g.dy)
_, _, _, p, _ = JS.make_solver_global(g, PhysicsConfig(), 3600.0, return_params=True)
wet3 = np.asarray(p.wet_mask_z, float)
DZN = np.asarray(p.dz_node).ravel()
vol = wet3 * AREA[:, :, None] * DZN[None, None, :]
ZJ = RHO_0 * C_P / 1e21

d = np.load("results/global_tenyr_ms_gm.npz")
days = np.asarray(d["days"], float)
T0 = np.load("results/global_tenyr_ms_gm_3d/snap_00000.npy")[0].astype(float)
T1 = np.load("results/global_tenyr_ms_gm_3d/snap_00122.npy")[0].astype(float)
dT = T1 - T0

print("=== 10-YEAR secular per-level heat change (ZJ), 3D mask ===")
per = np.array([(dT[:, :, k] * vol[:, :, k]).sum() * ZJ for k in range(14)])
for k in range(14):
    print("  k%-2d dz=%7.1f m  cellvol %8.3e   %+9.3f ZJ  %+8.3f ZJ/yr  dT_bar %+7.4f K"
          % (k, DZN[k], vol[:, :, k].sum(), per[k], per[k] / 10.0,
             (dT[:, :, k] * vol[:, :, k]).sum() / vol[:, :, k].sum()))
print("  TOTAL %+.3f ZJ  (%+.3f ZJ/yr)" % (per.sum(), per.sum() / 10.0))
print("  upper k0-7  %+8.3f ZJ/yr    mid k8-9 %+8.3f ZJ/yr    deep k10-13 %+8.3f ZJ/yr"
      % (per[:8].sum() / 10.0, per[8:10].sum() / 10.0, per[10:].sum() / 10.0))

print("\n=== decadal trajectory (volume-mean T, OHC) ===")
print("  day    volmeanT     OHC(ZJ)    d vs t0")
ohc0 = None
for idx in range(0, 123, 6):
    Ti = np.load("results/global_tenyr_ms_gm_3d/snap_%05d.npy" % idx)[0].astype(float)
    o = (Ti * vol).sum() * ZJ
    if ohc0 is None:
        ohc0 = o
    print("  %5.0f   %8.4f   %10.2f   %+9.2f"
          % (days[idx], (Ti * vol).sum() / vol.sum(), o, o - ohc0))

print("\n=== warming rate by decade-half ===")
T_h1 = np.load("results/global_tenyr_ms_gm_3d/snap_00060.npy")[0].astype(float)
T_h2 = np.load("results/global_tenyr_ms_gm_3d/snap_00122.npy")[0].astype(float)
for nm, Ta, Tb, yr in [("yr0->yr5", T0, T_h1, 5.0), ("yr5->yr10", T_h1, T_h2, 5.0)]:
    dd = Tb - Ta
    pr = np.array([(dd[:, :, k] * vol[:, :, k]).sum() * ZJ / yr for k in range(14)])
    print("  %-9s total %+8.3f ZJ/yr  upper(k0-7) %+8.3f  deep(k10-13) %+8.3f"
          % (nm, pr.sum(), pr[:8].sum(), pr[10:].sum()))
