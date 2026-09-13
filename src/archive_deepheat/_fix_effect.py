"""Does fixing the doubled kappa_v/nu_v change the deep-warming dipole?

200 steps from ckpt_tenyr_ms_gm, per-level dT, PATCHED vs BASELINE.  The patch
halves the effective vertical diffusivity (kappa_v 2x -> 1x, nu_v 2x -> 1x),
so the expected signature is a SLOWER deep warming.

Baseline (from _arm_test.py, same protocol):
  per-level dT/200 steps: -0.0943 -0.0351 ... +0.0013 +0.0009  dOHC +3.82 ZJ
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
AREA = np.asarray(g.dx_2d, float) * float(g.dy)
init = np.load("init_fields_g360x120.npz")
T_atm = air_temp_profile(g, init["T_init"].astype(np.float64)[:, :, 0])
PROD = dict(nu_h=5e6, nu_bi=2e14, kappa_bi=2e14, kappa_gm=1000.0,
            kappa_redi=1000.0, kappa_v=1e-5, kappa_conv=0.05, gm_slope_max=0.005)
CP = 1025.0 * 3992.0

d = np.load("results/ckpt_tenyr_ms_gm.npz")
ST0 = JS.JaxStateG(jnp.asarray(d["u"]), jnp.asarray(d["v"]), jnp.asarray(d["T"]),
                   jnp.asarray(d["S"]), jnp.asarray(d["eta"]))
T0 = np.asarray(d["T"], float)

_, _, _, p, _ = JS.make_solver_global(
    g, replace(PhysicsConfig(), **PROD), 3600.0, T_atm=jnp.asarray(T_atm),
    lambda_bulk=BULK_LAMBDA_DEFAULT, mode_split=True, dt_bt=300.0,
    return_params=True)
VOL = np.asarray(p.wet_mask_z, float) * AREA[:, :, None] * np.asarray(p.dz_node).ravel()[None, None, :]

def levels(T):
    return np.array([(T[:, :, k] * VOL[:, :, k]).sum() / VOL[:, :, k].sum()
                     for k in range(14)])

NSTEP = 200
step, _, _, _, _ = JS.make_solver_global(
    g, replace(PhysicsConfig(), **PROD), 3600.0, T_atm=jnp.asarray(T_atm),
    lambda_bulk=BULK_LAMBDA_DEFAULT, mode_split=True, dt_bt=300.0,
    return_params=True)
s = ST0
for _ in range(NSTEP):
    s = step(s)
jax.block_until_ready(s)

L = levels(np.asarray(s.T, float)) - levels(T0)
o = ((np.asarray(s.T) - T0) * VOL).sum() * CP / 1e21
print("PATCHED (kappa_v/nu_v applied once) %d steps from tenyr ckpt:" % NSTEP)
print("  " + " ".join("%+.4f" % x for x in L))
print("  dOHC %+8.2f ZJ   (ZJ/yr %+.1f)" % (o, o * 8760.0 / NSTEP))
print("\n  baseline was: -0.0943 -0.0351 -0.0192 -0.0137 -0.0118 -0.0112 "
      "-0.0058 -0.0028 +0.0032 +0.0029 +0.0019 +0.0015 +0.0013 +0.0009  dOHC +3.82")
print("\n  upper k0-7 mean dT: patched %+.4f" % L[:8].mean())
print("  deep  k11-13 mean dT: patched %+.4f" % L[11:].mean())
