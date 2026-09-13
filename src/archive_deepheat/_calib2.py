"""Resolve the surface-vs-interior budget with the solver's OWN bulk-flux expr.

Also check: is gm_T identical to redi_T? (both _isopycnal_slope with same taper)
and does adv+diff_v+gm+redi+diff_h+conv+bulk reproduce the real step?
"""
import sys
from dataclasses import replace
import numpy as np
import jax, jax.numpy as jnp
jax.config.update("jax_platform_name", "cpu")
sys.path.insert(0, "src")
from grid import make_global_grid, GlobalGridConfig
from config import DEFAULT_CONFIG, PhysicsConfig
from jax_solver_global import make_solver_global, JaxStateG
from forcing import air_temp_profile, BULK_LAMBDA_DEFAULT

RHO_0, C_P = 1025.0, 3992.0
SPY = 8760.0
g = make_global_grid(GlobalGridConfig(lat_max=60.0, ny=120),
                     DEFAULT_CONFIG.bathymetry_file, smooth_passes=30, min_depth=100.0)
AREA = np.asarray(g.dx_2d, float) * float(g.dy)
d = np.load("results/ckpt_tenyr_ms_gm.npz")
u0 = jnp.asarray(np.asarray(d["u"], np.float64)); v0 = jnp.asarray(np.asarray(d["v"], np.float64))
T0 = jnp.asarray(np.asarray(d["T"], np.float64)); S0 = jnp.asarray(np.asarray(d["S"], np.float64))
e0 = jnp.asarray(np.asarray(d["eta"], np.float64))
init = np.load("init_fields_g360x120.npz")
T_atm = air_temp_profile(g, init["T_init"].astype(np.float64)[:, :, 0])
phys = replace(PhysicsConfig(), nu_h=5e6, nu_bi=2e14, kappa_bi=2e14,
               kappa_gm=1000.0, kappa_redi=1000.0, kappa_v=1e-5,
               kappa_conv=0.05, gm_slope_max=0.005)
step, init_fn, diag, p, terms_fn = make_solver_global(
    g, phys, 3600.0, T_atm=jnp.asarray(T_atm), lambda_bulk=BULK_LAMBDA_DEFAULT,
    mode_split=True, dt_bt=300.0, return_params=True)

DZN = np.asarray(p.dz_node).ravel()
wet3 = np.asarray(p.wet_mask_z, float)
vol = wet3 * AREA[:, :, None] * DZN[None, None, :]
ZJ = RHO_0 * C_P / 1e21
def zj_per_yr(field3d):
    return float((field3d * vol).sum()) * ZJ * 3600.0 * SPY

print("dz_surface =", float(p.dz_surface), " lambda_bulk =", float(p.lambda_bulk))
print("surface_mask shape", np.asarray(p.surface_mask).shape,
      " sum", float(np.asarray(p.surface_mask).sum()))

# solver's exact bulk expression
heat_factor = 1.0 / (RHO_0 * C_P * float(p.dz_surface))
bulk_T = (float(p.lambda_bulk) * (T_atm[:, :, None] - np.asarray(T0)[:, :, 0:1])
          * heat_factor * np.asarray(p.surface_mask))
print("\nbulk_T surface rate: min %+.3e max %+.3e" % (bulk_T.min(), bulk_T.max()))
print("  ocean-mean |T_atm - SST| over surface wet = %.4f C"
      % np.abs((T_atm - np.asarray(T0)[:, :, 0]))[wet3[:, :, 0] > 0.5].mean())
print("  bulk contribution = %+.3f ZJ/yr" % zj_per_yr(bulk_T))

qh_T = (np.asarray(p.Q_heat_2d)[:, :, None] * heat_factor * np.asarray(p.surface_mask))
print("  Q_heat contribution = %+.3f ZJ/yr  (ocean mean %.6f W/m2)"
      % (zj_per_yr(qh_T), float((np.asarray(p.Q_heat_2d) * AREA * (wet3[:, :, 0] > 0.5)).sum()
                                / (AREA * (wet3[:, :, 0] > 0.5)).sum())))

st = JaxStateG(u=u0, v=v0, T=T0, S=S0, eta=e0)
tn = np.asarray(terms_fn(st))
names = ["adv_T", "diff_h_T", "diff_v_T", "conv_T", "gm_T", "redi_T"]
print("\ninterior terms (ZJ/yr):")
tot = 0.0
for i, n in enumerate(names):
    z = zj_per_yr(tn[i]); tot += z
    print("   %-9s %+10.4f" % (n, z))
print("   %-9s %+10.4f" % ("interior SUM", tot))
print("   %-9s %+10.4f" % ("bulk", zj_per_yr(bulk_T)))
print("   %-9s %+10.4f" % ("Q_heat", zj_per_yr(qh_T)))
print("   %-9s %+10.4f" % ("GRAND", tot + zj_per_yr(bulk_T) + zj_per_yr(qh_T)))

def ohc(T):
    return float((jnp.where(wet3 > 0.5, T, 0.0) * vol).sum()) * ZJ
o0 = ohc(T0); st1 = step(st); o1 = ohc(st1.T)
print("\nREAL one-step dOHC = %+.4f ZJ/yr" % ((o1 - o0) * SPY))
print("gm_T == redi_T bitwise? ", bool(np.array_equal(tn[4], tn[5])))
