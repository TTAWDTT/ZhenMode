"""Is the node-form vertical diffusion (_d2_dz2) heat-conservative?

_kappa_v * _d2_dz2 is the NODE form. The conv operator was migrated to an
interface-FLUX form precisely because the node form is not conservative on a
non-uniform grid. diff_v_T still uses the node form. Measure the column
integral of kappa_v*_d2_dz2 directly.
"""
import sys
from dataclasses import replace
import numpy as np
import jax, jax.numpy as jnp
jax.config.update("jax_platform_name", "cpu")
sys.path.insert(0, "src")
from grid import make_global_grid, GlobalGridConfig
from config import DEFAULT_CONFIG, PhysicsConfig
from jax_solver_global import make_solver_global, JaxStateG, _d2_dz2
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
T_atm = air_temp_profile(g, init["T_init"].astype(np.float64)[:, :, 0])
phys = replace(PhysicsConfig(), nu_h=5e6, nu_bi=2e14, kappa_bi=2e14,
               kappa_gm=1000.0, kappa_redi=1000.0, kappa_v=1e-5,
               kappa_conv=0.05, gm_slope_max=0.005)
step, init_fn, diag, p, terms_fn = make_solver_global(
    g, phys, 3600.0, T_atm=jnp.asarray(T_atm), lambda_bulk=BULK_LAMBDA_DEFAULT,
    mode_split=True, dt_bt=300.0, return_params=True)
st = JaxStateG(u=u0, v=v0, T=T0, S=S0, eta=e0)

wet3 = np.asarray(p.wet_mask_z, float)
DZN = np.asarray(p.dz_node).ravel()
vol = wet3 * AREA[:, :, None] * DZN[None, None, :]
ZJ = RHO_0 * C_P / 1e21

lap = np.asarray(_d2_dz2(st.T, p))
print("_d2_dz2(T) raw: min %+.4e max %+.4e" % (lap.min(), lap.max()))
col = (lap * vol).sum(axis=(0, 1)) * ZJ * 3600.0 * 8760.0
print("\ncolumn contribution of _d2_dz2 per level (ZJ/yr), sum over levels:")
print("  TOTAL = %+.4f ZJ/yr   (must be 0 for a conservative operator)" % col.sum())
for k in range(14):
    print("   k=%2d  %+10.5f" % (k, col[k]))

print("\namply kappa_v * that:")
print("  TOTAL = %+.4f ZJ/yr" % (1e-5 * col.sum()))

# Self-adjointness check: <1, dz*L(T)> should vanish
print("\nper-COLUMN sums (should be zero if conservative):")
colsum = (lap * DZN[None, None, :]).sum(axis=-1)
m = wet3[:, :, 0] > 0.5
print("  over wet columns: min %+.4e max %+.4e mean %+.4e"
      % (colsum[m].min(), colsum[m].max(), colsum[m].mean()))

# Also test on a smooth analytic profile to separate grid effect from data
print("\n-- analytic test: T = a + b*z on wet columns --")
Z = np.array([0, 5, 15, 30, 50, 75, 100, 150, 200, 300, 500, 1000, 2000, 4000], float)
Tlin = jnp.asarray(np.broadcast_to((-1e-3 * Z)[None, None, :], (360, 120, 14)).copy())
lap2 = np.asarray(_d2_dz2(Tlin, p))
cs = (lap2 * DZN[None, None, :]).sum(axis=-1)
print("  linear profile: max |column sum| = %.4e (should be ~0)" % np.abs(cs[m]).max())

Tquad = jnp.asarray(np.broadcast_to((1e-6 * Z**2)[None, None, :], (360, 120, 14)).copy())
lap3 = np.asarray(_d2_dz2(Tquad, p))
cs3 = (lap3 * DZN[None, None, :]).sum(axis=-1)
print("  quadratic profile: max |column sum| = %.4e" % np.abs(cs3[m]).max())
print("  quadratic expected 2e-6 everywhere; interior values:")
print("   ", np.round(lap3[0, 60, :], 8))
