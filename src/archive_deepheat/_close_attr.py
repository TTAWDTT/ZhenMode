"""Close the deep-warming attribution + test the surface-forcing hypothesis.

Established so far (ckpt_tenyr_ms_gm, 200 steps):
  * With the surface flux zeroed the column total moves only 1.8 ZJ out of
    21600 ZJ (0.008%), converging -> the solver is essentially conservative.
  * The deep warms and the upper cools at nearly identical rate in EVERY
    ablation arm (gm/redi/kappa_v/slope/topface) -> the dipole is not any one
    closure; it is the whole diffusive operator set relaxing the WOA profile
    toward the model's own volume-mean temperature.

This script:
  A. 200-step ablation of the three operators NOT yet tested:
     conv=0, kappa_h=0, biharmonic off, and "all mixing off" together.
  B. The surface-forcing accounting: ocean-area-weighted mean T_atm, and the
     implied equilibrium the restoring pulls toward (lambda=40 W/m^2/K is so
     strong that SST ~= T_atm almost exactly).
"""
import sys
from dataclasses import replace
import numpy as np
import jax, jax.numpy as jnp
jax.config.update("jax_platform_name", "cpu")
sys.path.insert(0, "src")
from grid import make_global_grid, GlobalGridConfig
from config import DEFAULT_CONFIG, PhysicsConfig
from forcing import air_temp_profile, BULK_LAMBDA_DEFAULT
import jax_solver_global as JS

RHO_0, C_P = 1025.0, 3992.0
SPY_S = 365.25 * 24.0                     # steps/yr at dt=3600 s  = 8766
g = make_global_grid(GlobalGridConfig(lat_max=60.0, ny=120),
                     DEFAULT_CONFIG.bathymetry_file, smooth_passes=30, min_depth=100.0)
AREA = np.asarray(g.dx_2d, float) * float(g.dy)
d = np.load("results/ckpt_tenyr_ms_gm.npz")
ST0 = JS.JaxStateG(u=jnp.asarray(np.asarray(d["u"], np.float64)),
                   v=jnp.asarray(np.asarray(d["v"], np.float64)),
                   T=jnp.asarray(np.asarray(d["T"], np.float64)),
                   S=jnp.asarray(np.asarray(d["S"], np.float64)),
                   eta=jnp.asarray(np.asarray(d["eta"], np.float64)))
T0_np = np.asarray(d["T"], float)
init = np.load("init_fields_g360x120.npz")
T_atm_np = air_temp_profile(g, init["T_init"].astype(np.float64)[:, :, 0])
BASE = dict(nu_h=5e6, nu_bi=2e14, kappa_bi=2e14, kappa_gm=1000.0,
            kappa_redi=1000.0, kappa_v=1e-5, kappa_conv=0.05, gm_slope_max=0.005)

_, _, _, p_ref, _ = JS.make_solver_global(
    g, replace(PhysicsConfig(), **BASE), 3600.0, T_atm=jnp.asarray(T_atm_np),
    lambda_bulk=BULK_LAMBDA_DEFAULT, mode_split=True, dt_bt=300.0, return_params=True)
wet3 = np.asarray(p_ref.wet_mask_z, float)
DZN = np.asarray(p_ref.dz_node).ravel()
vol = wet3 * AREA[:, :, None] * DZN[None, None, :]
ZJ = RHO_0 * C_P / 1e21
deep_m = (np.arange(14)[None, None, :] >= 10)
up_m = (np.arange(14)[None, None, :] <= 7)
vol_d, vol_u = (vol * deep_m).sum(), (vol * up_m).sum()

print("=== A. remaining 200-step ablation arms ===\n")
print("%-14s %10s %10s %10s | %11s %11s" % ("arm", "deep dT", "upper dT",
                                            "tot dOHC", "k13 dT", "k0 dT"))
def arm(tag, over=None, nstep=200):
    phys = replace(PhysicsConfig(), **{**BASE, **(over or {})})
    step, _, _, p, _ = JS.make_solver_global(
        g, phys, 3600.0, T_atm=jnp.asarray(T_atm_np), lambda_bulk=BULK_LAMBDA_DEFAULT,
        mode_split=True, dt_bt=300.0, return_params=True)
    s = ST0
    for _ in range(nstep):
        s = step(s)
    jax.block_until_ready(s)
    dT = np.asarray(s.T, float) - T0_np
    val = lambda m: (dT * vol * m).sum() / (vol * m).sum()
    tot = (dT * vol).sum() * ZJ
    print("%-14s %+10.5f %+10.5f %+10.3f | %+11.5f %+11.5f"
          % (tag, val(deep_m), val(up_m), tot, dT[:, :, 13].mean(), dT[:, :, 0].mean()),
          flush=True)
    return dT

arm("baseline")
arm("conv=0", dict(kappa_conv=0.0))
arm("kappa_h=0", dict(kappa_h=0.0))
arm("biharm=0", dict(nu_bi=0.0, kappa_bi=0.0))
arm("ALLMIX=0", dict(kappa_conv=0.0, kappa_h=0.0, kappa_v=0.0,
                     kappa_gm=0.0, kappa_redi=0.0, nu_bi=0.0, kappa_bi=0.0))

print("\n=== B. surface-forcing accounting ===")
w2 = np.asarray(p_ref.wet_mask, float)
A2 = AREA * w2
Tatm2 = np.asarray(T_atm_np, float)
print("  ocean-area-weighted mean T_atm = %+7.3f C" % ((Tatm2 * A2).sum() / A2.sum()))
print("  global-mean  (unweighted) T_atm = %+7.3f C" % Tatm2.mean())
print("  lambda_bulk                     = %7.2f W/m^2/K" % BULK_LAMBDA_DEFAULT)
print("  real global-mean SST            = ~17.5 C")
sst = np.asarray(p_ref.T_clim_3d, float)
Ttop = T0_np[:, :, 0]
print("  ckpt SST ocean-mean             = %+7.3f C" % ((Ttop * A2).sum() / A2.sum()))
print("  ckpt volume-mean T              = %+7.3f C"
      % ((T0_np * vol).sum() / vol.sum()))
print("  WOA init volume-mean T          = %+7.3f C"
      % ((init["T_init"].astype(float) * vol).sum() / vol.sum()))
print("\ndone")
