"""What dominates dT in the "everything off" control?

Test 1's kappa_v=0 control still moved T by 0.285 K in one step. With
kappa_h=kappa_v=kappa_conv=kappa_bi=kappa_gm=kappa_redi=0, lambda_bulk=0 and
u=v=0, only advection (trivially 0), heat_T, and the surface_mask BCs are left.
Attribute the control step to each remaining term.
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

g = make_global_grid(GlobalGridConfig(lat_max=60.0, ny=120),
                     DEFAULT_CONFIG.bathymetry_file, smooth_passes=30, min_depth=100.0)
init = np.load("init_fields_g360x120.npz")
T_atm = air_temp_profile(g, init["T_init"].astype(np.float64)[:, :, 0])
dt = 3600.0

CFG = dict(nu_h=0.0, nu_v=0.0, kappa_h=0.0, kappa_v=0.0, kappa_conv=0.0,
           kappa_bi=0.0, kappa_gm=0.0, kappa_redi=0.0, gm_slope_max=0.005)
d = np.load("results/ckpt_tenyr_ms_gm.npz")

def mk(**kw):
    c = dict(CFG); c.update(kw)
    step, _, _, p, tf = JS.make_solver_global(
        g, replace(PhysicsConfig(), **c), dt, T_atm=jnp.asarray(T_atm),
        lambda_bulk=BULK_LAMBDA_DEFAULT, mode_split=False,
        return_params=True)
    return step, p, tf

T0 = jnp.asarray(d["T"]); S0 = jnp.asarray(d["S"])
Z = jnp.zeros_like(T0)
st = JS.JaxStateG(Z, Z, T0, S0, jnp.asarray(d["eta"]))

step, p, tf = mk()
new = step(st); jax.block_until_ready(new)
dT = np.asarray(new.T - T0, float)
m = np.asarray(p.wet_mask_z, float) > 0.5
print("ALL-OFF control: max|dT| %.6e  mean|dT|wet %.6e" % (np.abs(dT).max(),
                                                           np.abs(dT[m]).mean()))
print("\nQ_heat_2d: ", None if p.Q_heat_2d is None
      else "max %.6e  sum %.6e" % (float(jnp.max(p.Q_heat_2d)),
                                   float(jnp.sum(p.Q_heat_2d))))
print("surface_mask sum %.1f  dz_surface %.4f" % (float(p.surface_mask.sum()),
                                                  float(p.dz_surface)))
print("lambda_bulk %.3f" % float(p.lambda_bulk))
Tatm3 = np.asarray(p.T_atm_3d, float)
print("T_atm_3d shape %s  min %.2f max %.2f" % (Tatm3.shape, Tatm3.min(), Tatm3.max()))
hf = 1.0 / (1025.0 * 3992.0 * float(p.dz_surface))
print("heat_factor 1/(rho cp dz_surf) = %.6e" % hf)

print("\nper-level max|dT|:")
for k in range(14):
    print("  k=%2d  max %.6e  n_wet %d" % (k, np.abs(dT[:, :, k]).max(),
                                           int(m[:, :, k].sum())))

T_atm_v = Tatm3.ravel()
print("\ncompare to the bulk term magnitude: lambda*(Tatm-T)*heat_factor*dt")
print("  T_atm range %.2f..%.2f  -> max |bulk*dt| ~ %.6e"
      % (Tatm3.min(), Tatm3.max(),
         float(p.lambda_bulk) * max(abs(Tatm3.min() - 20), abs(Tatm3.max())) * hf * dt))
