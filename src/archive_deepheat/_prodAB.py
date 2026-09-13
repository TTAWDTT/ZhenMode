"""Production-physics A/B: does the shear projection fix the dipole?

Previous run diverged at step 5 -- MY config error: I set kappa_h=5e6, but
the runner never assigns kappa_h, so production keeps the PhysicsConfig
default 100.0 m^2/s. kappa_h is NOT subcycled in the L step, so 5e6 at
dt/2=1800 gives 5e6*1800/(1.1e5)^2 = 0.74 >> 0.25 (the explicit-diffusion
CFL) -> both arms blew up. Reproduce the runner's physics EXACTLY as
run_long_integration_global.py:352-358 does.
"""
import sys
import numpy as np
import jax, jax.numpy as jnp
jax.config.update("jax_platform_name", "cpu")
sys.path.insert(0, "src")
from dataclasses import replace
from grid import make_global_grid, GlobalGridConfig
from config import DEFAULT_CONFIG, PhysicsConfig
from forcing import air_temp_profile
import jax_solver_global as JS

RHO_0, C_P = 1025.0, 3992.0
DT = 3600.0
ZJY = RHO_0 * C_P / 1e21 * 3.1536e7
g = make_global_grid(GlobalGridConfig(lat_max=60.0, ny=120),
                     DEFAULT_CONFIG.bathymetry_file, smooth_passes=30, min_depth=100.0)
d = np.load("results/ckpt_tenyr_ms_gm.npz")
u0 = np.asarray(d["u"], np.float64); v0 = np.asarray(d["v"], np.float64)
T0 = np.asarray(d["T"], np.float64); S0 = np.asarray(d["S"], np.float64)
e0 = np.asarray(d["eta"], np.float64)
init = np.load("init_fields_g360x120.npz")
T_atm_np = np.asarray(air_temp_profile(g, init["T_init"].astype(np.float64)[:, :, 0]), float)

# EXACTLY the runner's construction (line 352-358); kappa_h left at the
# PhysicsConfig default 100.0 because the runner has no --kappa-h flag.
PROD = dict(nu_h=5e6, nu_bi=2e14, kappa_bi=2e14, kappa_gm=1000.0,
            kappa_redi=1000.0, kappa_v=1e-5, kappa_conv=0.05,
            gm_slope_max=0.005)
phys = replace(PhysicsConfig(), **PROD)
print("physics: kappa_h=%.1f nu_h=%.3g nu_bi=%.3g kappa_bi=%.3g kappa_gm=%.3g"
      % (phys.kappa_h, phys.nu_h, phys.nu_bi, phys.kappa_bi, phys.kappa_gm))
print("         kappa_redi=%.3g kappa_v=%.3g kappa_conv=%.3g gm_slope_max=%.4g"
      % (phys.kappa_redi, phys.kappa_v, phys.kappa_conv, phys.gm_slope_max))
# CFL sanity
dx = float(np.min(g.dx_2d))
print("  kappa_h*dt_half/dx_min^2 = %.4f  (must be < 0.25)"
      % (phys.kappa_h * (DT / 2) / dx ** 2))

_, _, _, p, _ = JS.make_solver_global(
    g, phys, DT, T_atm=jnp.asarray(T_atm_np), lambda_bulk=40.0,
    mode_split=True, dt_bt=300.0, polar_cap_rows=2, polar_cap_taper=3,
    T_init=jnp.asarray(init["T_init"], np.float64),
    S_init=jnp.asarray(init["S_init"], np.float64),
    return_params=True)
print("  n_subcyc=%d nu_nsub=%s adv_nsub=%s conv_nsub=%s"
      % (p.n_subcyc, p.nu_nsub, p.adv_nsub, getattr(p, "conv_nsub", "?")))

wet3 = np.asarray(p.wet_mask_z, float)
DZN = np.asarray(p.dz_node).ravel()
AREA = np.asarray(g.dx_2d, float) * float(g.dy)
vol = wet3 * AREA[:, :, None] * DZN[None, None, :]
DZ = jnp.asarray(DZN).reshape(1, 1, -1)
M = jnp.asarray(wet3)
ORIG_VTI = JS._vertical_transport_iface
ORIG_ADV = JS._advection_scalar


def shear(a):
    Hcol = jnp.sum(M * DZ, axis=-1, keepdims=True)
    c = jnp.sum(a * M * DZ, axis=-1, keepdims=True) / jnp.maximum(Hcol, 1e-9)
    return (a - c) * M


def install(variant):
    if variant == "base":
        JS._vertical_transport_iface = ORIG_VTI
        JS._advection_scalar = ORIG_ADV
        return

    def vti(u, v, pp):
        return ORIG_VTI(shear(u), shear(v), pp)

    def adv(T, u, v, Fz, pp):
        return ORIG_ADV(T, shear(u), shear(v), Fz, pp)

    JS._vertical_transport_iface = vti
    JS._advection_scalar = adv


NSTEP = 720          # 30 days
print()
print("=" * 94)
print("Production physics, %d steps (%.1f days): per-level T trend" % (NSTEP, NSTEP * DT / 86400))
print("=" * 94)
res = {}
for vname in ("base", "shear"):
    install(vname)
    s = JS.JaxStateG(u=jnp.asarray(u0), v=jnp.asarray(v0), T=jnp.asarray(T0),
                     S=jnp.asarray(S0), eta=jnp.asarray(e0))
    T_start = np.asarray(s.T, float).copy()
    n = NSTEP
    for k in range(NSTEP):
        s = JS._step_impl(s, p)
        if not np.isfinite(np.asarray(s.T)).all():
            print("  %s DIVERGED at step %d" % (vname, k + 1))
            n = k + 1
            break
    T_end = np.asarray(s.T, float)
    yrs = n * DT / 3.1536e7
    dT = (T_end - T_start) / yrs
    res[vname] = dT
    tot = float((dT * vol).sum() * RHO_0 * C_P / 1e21)
    print("  --- %s ---  n=%d  total %+10.3f ZJ/yr" % (vname, n, tot))
    print("     k   depth(m)   dT/yr(K)   dH/yr(ZJ)")
    for k in range(14):
        vk = vol[:, :, k]
        dh = float((dT[:, :, k] * vk).sum() * RHO_0 * C_P / 1e21)
        print("    %2d   %8.1f   %+10.5f  %+10.3f"
              % (k, DZN[k], float(dT[:, :, k][wet3[:, :, k] > 0].mean()), dh))

install("base")
if len(res) == 2:
    print()
    print("=" * 94)
    print("Difference (shear - base) per level   [K/yr]")
    print("=" * 94)
    dd = res["shear"] - res["base"]
    for k in range(14):
        print("    k=%2d depth %8.1f m   %+10.5f" % (k, DZN[k],
              float(dd[:, :, k][wet3[:, :, k] > 0].mean())))
    print("  volume-weighted total change: %+10.3f ZJ/yr"
          % float((dd * vol).sum() * RHO_0 * C_P / 1e21))
