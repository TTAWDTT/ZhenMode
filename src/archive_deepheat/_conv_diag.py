"""Diagnose the whole-column convective mask at the checkpoint.

conv_mask_3d = any(unstable_iface, axis=-1, keepdims=True) gates the WHOLE
column. If a large fraction of columns are flagged, kappa_conv=0.05 (5000x
kappa_v) is being applied to the entire deep column every step.
"""
import sys
from dataclasses import replace
import numpy as np
import jax, jax.numpy as jnp
jax.config.update("jax_platform_name", "cpu")
sys.path.insert(0, "src")
from grid import make_global_grid, GlobalGridConfig
from config import DEFAULT_CONFIG, PhysicsConfig
from jax_solver_global import (make_solver_global, JaxStateG, _density_anomaly,
                               _fill_ghost_bottom, _d_dz)
from forcing import air_temp_profile, BULK_LAMBDA_DEFAULT

g = make_global_grid(GlobalGridConfig(lat_max=60.0, ny=120),
                     DEFAULT_CONFIG.bathymetry_file, smooth_passes=30, min_depth=100.0)
AREA = np.asarray(g.dx_2d, float) * float(g.dy)
d = np.load("results/ckpt_tenyr_ms_gm.npz")
T0 = jnp.asarray(np.asarray(d["T"], np.float64)); S0 = jnp.asarray(np.asarray(d["S"], np.float64))
u0 = jnp.asarray(np.asarray(d["u"], np.float64)); v0 = jnp.asarray(np.asarray(d["v"], np.float64))
e0 = jnp.asarray(np.asarray(d["eta"], np.float64))
init = np.load("init_fields_g360x120.npz")
T_atm = air_temp_profile(g, init["T_init"].astype(np.float64)[:, :, 0])
phys = replace(PhysicsConfig(), nu_h=5e6, nu_bi=2e14, kappa_bi=2e14,
               kappa_gm=1000.0, kappa_redi=1000.0, kappa_v=1e-5,
               kappa_conv=0.05, gm_slope_max=0.005)
step, init_fn, diag, p, terms_fn = make_solver_global(
    g, phys, 3600.0, T_atm=jnp.asarray(T_atm), lambda_bulk=BULK_LAMBDA_DEFAULT,
    mode_split=True, dt_bt=300.0, return_params=True)
st = JaxStateG(u=u0, v=v0, T=T0, S=S0, eta=e0)

rho = np.asarray(_density_anomaly(st.T, st.S, p)) * np.asarray(p.wet_mask_z)
wet_z = np.asarray(p.wet_mask_z)
wet_iface = (wet_z[:, :, :-1] > 0.5) & (wet_z[:, :, 1:] > 0.5)
unstable = (rho[:, :, :-1] > rho[:, :, 1:]) & wet_iface
col_mask = unstable.any(axis=-1)
surf_wet = wet_z[:, :, 0] > 0.5

print("wet surface cells        : %d" % int(surf_wet.sum()))
print("columns with >=1 unstable iface: %d  (%.1f%% of wet surface)"
      % (int(col_mask.sum()), 100.0*col_mask.sum()/surf_wet.sum()))
print("total unstable interfaces: %d of %d wet interfaces (%.2f%%)"
      % (int(unstable.sum()), int(wet_iface.sum()), 100.0*unstable.sum()/wet_iface.sum()))

print("\nunstable interfaces per level k (k -> k+1):")
for k in range(13):
    n = int(unstable[:, :, k].sum())
    if n:
        zc = -0.5*(np.array([0,5,15,30,50,75,100,150,200,300,500,1000,2000,4000],float)[k]
                   + np.array([0,5,15,30,50,75,100,150,200,300,500,1000,2000,4000],float)[k+1])
        print("   iface %2d (zc %7.0f m): %6d  (%.1f%% of wet ifaces at that level)"
              % (k, zc, n, 100.0*n/max(1, int(wet_iface[:, :, k].sum()))))

lat = init["lat"]; lon = init["lon"]
if col_mask.any():
    jj, ii = np.where(col_mask)
    print("\nlat range of flagged columns: %.1f .. %.1f" % (lat[jj].min(), lat[jj].max()))
    h, _ = np.histogram(lat[jj], bins=12, range=(-60, 60))
    print("  lat histogram:", h)

print("\nrho' profile (ocean mean, wet cells only):")
for k in range(14):
    m = wet_z[:, :, k] > 0.5
    print("   k=%2d  rho' mean %+.6f  T mean %+.3f  S mean %+.3f"
          % (k, rho[:, :, k][m].mean(), np.asarray(st.T)[:, :, k][m].mean(),
             np.asarray(st.S)[:, :, k][m].mean()))
