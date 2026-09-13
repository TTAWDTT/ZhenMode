"""Pin the spurious source: is it Fz[0] itself, or the Fz_top closure form?

Identity:  sum_k vol_k * adv_k = -sum_cells area*Fz[0]*T[0]
Let me test candidate closures by swapping Fz_top and re-integrating.
"""
import sys
import numpy as np
import jax, jax.numpy as jnp
jax.config.update("jax_platform_name", "cpu")
sys.path.insert(0, "src")
from dataclasses import replace
from grid import make_global_grid, GlobalGridConfig
from config import DEFAULT_CONFIG, PhysicsConfig
from forcing import air_temp_profile, BULK_LAMBDA_DEFAULT
import jax_solver_global as JS

RHO_0, C_P = 1025.0, 3992.0
ZJYR = RHO_0 * C_P / 1e21 * 3.1536e7
g = make_global_grid(GlobalGridConfig(lat_max=60.0, ny=120),
                     DEFAULT_CONFIG.bathymetry_file, smooth_passes=30, min_depth=100.0)
AREA = np.asarray(g.dx_2d, float) * float(g.dy)
d = np.load("results/ckpt_tenyr_ms_gm.npz")
T0 = np.asarray(d["T"], np.float64)
u0 = np.asarray(d["u"], np.float64); v0 = np.asarray(d["v"], np.float64)
init = np.load("init_fields_g360x120.npz")
T_atm_np = np.asarray(air_temp_profile(g, init["T_init"].astype(np.float64)[:, :, 0]), float)
BASE = dict(nu_h=5e6, nu_bi=2e14, kappa_bi=2e14, kappa_gm=1000.0,
            kappa_redi=1e3, kappa_v=1e-5, kappa_conv=0.05, gm_slope_max=0.005)
_, _, _, p, _ = JS.make_solver_global(
    g, replace(PhysicsConfig(), **BASE), 3600.0, T_atm=jnp.asarray(T_atm_np),
    lambda_bulk=BULK_LAMBDA_DEFAULT, mode_split=True, dt_bt=300.0, return_params=True)
wet3 = np.asarray(p.wet_mask_z, float)
DZN = np.asarray(p.dz_node).ravel()
vol = wet3 * AREA[:, :, None] * DZN[None, None, :]
surfm = wet3[:, :, 0] > 0.5

Fz = JS._vertical_transport_iface(jnp.asarray(u0), jnp.asarray(v0), p)
f0 = np.asarray(Fz[:, :, 0], float)
A = AREA * surfm
Ttop = T0[:, :, 0]
print("=== is the identity sum_k vol*adv_k = -sum area*Fz[0]*T[0] exact? ===")
advT = np.asarray(JS._advection_scalar(jnp.asarray(T0), jnp.asarray(u0), jnp.asarray(v0), Fz, p), float)
lhs = (advT * vol).sum() * ZJYR
rhs = -(f0 * A * Ttop).sum() * ZJYR
print("  LHS sum vol*adv   = %+12.4f ZJ/yr" % lhs)
print("  RHS -sum a*Fz0*T0 = %+12.4f ZJ/yr" % rhs)
print("  identity residual = %+12.4e" % (lhs - rhs))

print("\n=== the culprit is Fz[0] != 0: it is the rigid-lid leak ===")
print("  sum over wet cols of Fz[0]*area = %+.4e m^3/s" % (f0 * A).sum())
print("  sum over wet cols of |Fz[0]|*area = %+.4e m^3/s" % (np.abs(f0) * A).sum())
print("  -> there IS a substantial non-cancelling net volume source")

print("\n=== is Fz[0] consistent with div(ubt)*H? ===")
ubt, vbt = JS._barotropic_velocity(jnp.asarray(u0), jnp.asarray(v0), p)
divbt = JS._divergence_conservative(ubt * p.wet_mask, vbt * p.wet_mask, p)
db = np.asarray(divbt, float)
print("  corr(Fz[0], -divbt*H)?  rms Fz0 %.3e   rms divbt*H %.3e"
      % (np.sqrt((f0[surfm] ** 2).mean()), np.sqrt(((db * np.asarray(p.H_sw if hasattr(p,'H_sw') else 1.0, float))[surfm] ** 2).mean())))
for attr in ("H", "H_sw", "depth", "col_depth"):
    if hasattr(p, attr):
        print("   p.%s present, shape %s" % (attr, np.asarray(getattr(p, attr)).shape))

print("\n=== candidate closures: their global integral ===")
Fz_top_cur = f0 * Ttop
Fz_corr = f0 * (Ttop - (Ttop * A).sum() / A.sum())
print("  current  Fz0*T0        -> %+12.4f ZJ/yr" % (-(Fz_top_cur * A).sum() * ZJYR))
print("  mass-cor Fz0*(T0-Tbar)-> %+12.4f ZJ/yr" % (-(Fz_corr * A).sum() * ZJYR))
print("  zero     Fz0*0         -> %+12.4f ZJ/yr" % 0.0)
print("\n  If the mass-corrected closure is ~0 while current is +66.6,")
print("  the defect is flushed entirely from the surface layer.")
