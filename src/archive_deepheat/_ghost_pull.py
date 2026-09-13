"""Test the ghost-pull hypothesis for the _d2_dz2 column leak.

For columns whose bottom wet level is k<13, the bottom wet node's INTERIOR
stencil (which spans k-1,k,k+1 and is used for nodes 1..12) reaches into the
ghost layers below the seafloor. Those hold T_ref=15C. The ghost tendency is
masked away, so the heat is destroyed -> a spurious column sink.

Measure: (a) how many columns are partial, (b) the leak with and without
_fill_ghost_bottom applied first.
"""
import sys
from dataclasses import replace
import numpy as np
import jax, jax.numpy as jnp
jax.config.update("jax_platform_name", "cpu")
sys.path.insert(0, "src")
from grid import make_global_grid, GlobalGridConfig
from config import DEFAULT_CONFIG, PhysicsConfig
from jax_solver_global import (make_solver_global, JaxStateG, _d2_dz2,
                               _fill_ghost_bottom, _conv_flux_tendency)
from forcing import air_temp_profile, BULK_LAMBDA_DEFAULT

RHO_0, C_P = 1025.0, 3992.0
g = make_global_grid(GlobalGridConfig(lat_max=60.0, ny=120),
                     DEFAULT_CONFIG.bathymetry_file, smooth_passes=30, min_depth=100.0)
AREA = np.asarray(g.dx_2d, float) * float(g.dy)
d = np.load("results/ckpt_tenyr_ms_gm.npz")
T0 = jnp.asarray(np.asarray(d["T"], np.float64)); S0 = jnp.asarray(np.asarray(d["S"], np.float64))
u0 = jnp.asarray(np.asarray(d["u"], np.float64)); v0 = jnp.asarray(np.asarray(d["v"], np.float64))
e0 = jnp.asarray(np.asarray(d["eta"], np.float64))
init = np.load("init_fields_g360x120.npz")
T_atm = air_term = air_temp_profile(g, init["T_init"].astype(np.float64)[:, :, 0])
phys = replace(PhysicsConfig(), nu_h=5e6, nu_bi=2e14, kappa_bi=2e14,
               kappa_gm=1000.0, kappa_redi=1000.0, kappa_v=1e-5,
               kappa_conv=0.05, gm_slope_max=0.005)
step, init_fn, diag, p, terms_fn = make_solver_global(
    g, phys, 3600.0, T_atm=jnp.asarray(T_atm), lambda_bulk=BULK_LAMBDA_DEFAULT,
    mode_split=True, dt_bt=300.0, return_params=True)
st = JaxStateG(u=u0, v=v0, T=T0, S=S0, eta=e0)

wet = np.asarray(p.wet_mask_z) > 0.5
DZN = np.asarray(p.dz_node).ravel()
vol = wet.astype(float) * AREA[:, :, None] * DZN[None, None, :]
ZJ = RHO_0 * C_P / 1e21 * 3600.0 * 8760.0

nlev = wet.sum(axis=-1)
surf = wet[:, :, 0]
print("columns with bottom wet level:")
for k in range(14):
    n = int(((nlev - 1) == k)[surf].sum())
    if n:
        print("   kbot=%2d : %6d columns (%.1f%%)" % (k, n, 100.0*n/surf.sum()))

lap_raw = np.asarray(_d2_dz2(st.T, p))
T_fill = _fill_ghost_bottom(st.T, p)
lap_fill = np.asarray(_d2_dz2(T_fill, p))

print("\ncolumn-integrated kappa_v*_d2_dz2 (ZJ/yr), kappa_v=1e-5:")
print("   raw          : %+10.4f" % (1e-5 * (lap_raw * vol).sum() * ZJ))
print("   ghost-filled : %+10.4f" % (1e-5 * (lap_fill * vol).sum() * ZJ))

print("\nper-level (ZJ/yr):")
print("   k   raw        filled")
for k in range(14):
    a = 1e-5 * (lap_raw[:, :, k] * vol[:, :, k]).sum() * ZJ
    b = 1e-5 * (lap_fill[:, :, k] * vol[:, :, k]).sum() * ZJ
    print("  %2d  %+10.4f  %+10.4f" % (k, a, b))

# how big is the ghost contamination at the bottom wet node?
print("\nbottom-wet-node ghost pull (raw):")
for kbot in [9, 10, 11, 12]:
    m = surf & ((nlev - 1) == kbot)
    if m.sum() == 0:
        continue
    v = 1e-5 * lap_raw[:, :, kbot][m] * vol[:, :, kbot][m]
    print("   kbot=%2d n=%5d  mean rate %+.4e K/s  column contrib %+.4f ZJ/yr"
          % (kbot, int(m.sum()), lap_raw[:, :, kbot][m].mean(), v.sum() * ZJ))

print("\nT value in ghost layers of partial columns: distinct values =",
      np.unique(np.asarray(st.T)[~wet])[:5], " count", int((~wet).sum()))
