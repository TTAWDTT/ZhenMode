"""Does the SURFACE FLUX account for the +531 ZJ the ocean gained in 10 yr?

If yes, the deep warming is a legitimate forced response (surface heat mixed
and advected downward), not a numerical source.
"""
import numpy as np, sys, json
import jax.numpy as jnp
sys.path.insert(0, "src")
from grid import make_global_grid, GlobalGridConfig
from config import DEFAULT_CONFIG, PhysicsConfig
from forcing import air_temp_profile, BULK_LAMBDA_DEFAULT
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

nf = np.load("init_fields_g360x120.npz")
T_atm = np.asarray(air_temp_profile(g, nf["T_init"].astype(np.float64)[:, :, 0]), float)
print("T_atm ocean-mean %.4f   global-mean %.4f   min %.3f max %.3f"
      % ((T_atm * wet3[:, :, 0]).sum() / wet3[:, :, 0].sum(), T_atm.mean(),
         T_atm.min(), T_atm.max()))
print("bulk lambda = %.1f W/m^2/K   (Q = lam*(T_atm - SST))" % BULK_LAMBDA_DEFAULT)

d = np.load("results/global_tenyr_ms_gm.npz")
days = np.asarray(d["days"], float)

print("\n day      meanSST   Q_ocean_mean   dQdt_implied    OHC")
print("                       (W/m^2)       (ZJ/yr)")
ohc_prev, day_prev = None, None
tot_q = 0.0
for idx in range(0, 123, 12):
    T = np.load("results/global_tenyr_ms_gm_3d/snap_%05d.npy" % idx)[0].astype(float)
    sst = T[:, :, 0]
    w = wet3[:, :, 0]
    q = (BULK_LAMBDA_DEFAULT * (T_atm - sst) * w * AREA).sum()
    o = (T * vol).sum() * ZJ
    print("  %5.0f    %7.3f     %+8.3f" % (days[idx], (sst * w * AREA).sum() / (w * AREA).sum(), q / (w * AREA).sum()) + "        (integral below)")
    tot_q += q

# integrate Q properly over the whole record
print("\n=== cumulative surface heat flux ===")
qs, ohcs, ds = [], [], []
for idx in range(123):
    T = np.load("results/global_tenyr_ms_gm_3d/snap_%05d.npy" % idx)[0].astype(float)
    w = wet3[:, :, 0]
    q = (BULK_LAMBDA_DEFAULT * (T_atm - T[:, :, 0]) * w * AREA).sum()
    qs.append(q)
    ohcs.append((T * vol).sum() * ZJ)
    ds.append(days[idx])
qs = np.array(qs); ohcs = np.array(ohcs); ds = np.array(ds)
cum = np.concatenate([[0.0], np.cumsum(0.5 * (qs[1:] + qs[:-1]) * np.diff(ds) * 86400.0)]) / 1e21

print(" day     cumulative Q (ZJ)     actual dOHC (ZJ)    residual")
ohc0 = ohcs[0]
for idx in range(0, 123, 12):
    print("  %5.0f    %+12.1f      %+12.1f     %+10.1f"
          % (ds[idx], cum[idx], ohcs[idx] - ohc0, (ohcs[idx] - ohc0) - cum[idx]))
print("\nTOTAL Q over 10 yr   = %+.1f ZJ" % cum[-1])
print("TOTAL dOHC over 10yr = %+.1f ZJ" % (ohcs[-1] - ohc0))
print("RESIDUAL (unexplained) = %+.1f ZJ  (%.2f%% of |Q|)"
      % ((ohcs[-1] - ohc0) - cum[-1],
         100.0 * abs((ohcs[-1] - ohc0) - cum[-1]) / max(abs(cum[-1]), 1e-9)))
print("\nocean-area-weighted mean Q = %+.3f W/m^2  (real Earth: +0.9 W/m^2)"
      % (qs.mean() / (wet3[:, :, 0] * AREA).sum()))
