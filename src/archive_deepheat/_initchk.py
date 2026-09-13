import numpy as np, sys
import jax.numpy as jnp
sys.path.insert(0, "src")
from dataclasses import replace
from grid import make_global_grid, GlobalGridConfig
from config import DEFAULT_CONFIG, PhysicsConfig
from forcing import air_temp_profile, BULK_LAMBDA_DEFAULT
import jax_solver_global as JS

g = make_global_grid(GlobalGridConfig(lat_max=60.0, ny=120),
                     DEFAULT_CONFIG.bathymetry_file, smooth_passes=30, min_depth=100.0)
AREA = np.asarray(g.dx_2d, float) * float(g.dy)
_, _, _, p, _ = JS.make_solver_global(g, PhysicsConfig(), 3600.0, return_params=True)
wet3 = np.asarray(p.wet_mask_z, float)
DZN = np.asarray(p.dz_node).ravel()
vol = wet3 * AREA[:, :, None] * DZN[None, None, :]
print("wet3 sum", wet3.sum(), " vol = %.9e" % vol.sum())
print("reference                 1.203473617931776e+18")
RHO_0, C_P = 1025.0, 3992.0
ZJ = RHO_0 * C_P / 1e21
vm = lambda T: (np.asarray(T, float) * vol).sum() / vol.sum()

nf = np.load("init_fields_g360x120.npz")
Ti = np.asarray(nf["T_init"], float)
Ta = np.asarray(np.load("results/global_tenyr_ms_gm.npz")["T_init"], float)
Ts0 = np.load("results/global_tenyr_ms_gm_3d/snap_00000.npy")[0].astype(float)
Ts1 = np.load("results/global_tenyr_ms_gm_3d/snap_00122.npy")[0].astype(float)
Tck = np.asarray(np.load("results/ckpt_tenyr_ms_gm.npz")["T"], float)

for nm, T in [("npz init", Ti), ("archive T_init", Ta), ("snap0 day0", Ts0),
              ("snap122 day3650", Ts1), ("ckpt", Tck)]:
    print("%-18s volmeanT %.4f  OHC %9.1f ZJ  min %8.4f max %8.4f"
          % (nm, vm(T), (np.asarray(T, float) * vol).sum() * ZJ, np.asarray(T, float).min(), np.asarray(T, float).max()))

print("\nmax|archive_T_init-npz| %.4f  max|snap0-npz| %.4f" %
      (np.abs(Ta - Ti).max(), np.abs(Ts0 - Ti).max()))
print("wet3[:, :, 13].sum() =", wet3[:, :, 13].sum(), " (deepest level ocean cells)")
print("wet3[:, :, 0].sum()  =", wet3[:, :, 0].sum())
