"""Decompose the production dipole: which term carries the per-level signal?

The _prodAB base arm showed the classic dipole (upper cools ~-0.9 K/yr at
k=2, deep warms +0.09 K/yr at k=9, net +176.8 ZJ/yr).  Before chasing more
advection fixes, find WHICH term (adv_T, diff_v, gm, redi, bulk, conv, ...)
dominates each level.  terms_fn gives the per-term tendency decomposition.
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
g = make_global_grid(GlobalGridConfig(lat_max=60.0, ny=120),
                     DEFAULT_CONFIG.bathymetry_file, smooth_passes=30, min_depth=100.0)
d = np.load("results/ckpt_tenyr_ms_gm.npz")
u0 = np.asarray(d["u"], np.float64); v0 = np.asarray(d["v"], np.float64)
T0 = np.asarray(d["T"], np.float64); S0 = np.asarray(d["S"], np.float64)
e0 = np.asarray(d["eta"], np.float64)
init = np.load("init_fields_g360x120.npz")
T_atm_np = np.asarray(air_temp_profile(g, init["T_init"].astype(np.float64)[:, :, 0]), float)

PROD = dict(nu_h=5e6, nu_bi=2e14, kappa_bi=2e14, kappa_gm=1000.0,
            kappa_redi=1000.0, kappa_v=1e-5, kappa_conv=0.05,
            gm_slope_max=0.005)
phys = replace(PhysicsConfig(), **PROD)
_, _, _, p, terms_fn = JS.make_solver_global(
    g, phys, DT, T_atm=jnp.asarray(T_atm_np), lambda_bulk=40.0,
    mode_split=True, dt_bt=300.0, polar_cap_rows=2, polar_cap_taper=3,
    T_init=jnp.asarray(init["T_init"], np.float64),
    S_init=jnp.asarray(init["S_init"], np.float64),
    return_params=True)

wet3 = np.asarray(p.wet_mask_z, float)
DZN = np.asarray(p.dz_node).ravel()
AREA = np.asarray(g.dx_2d, float) * float(g.dy)
vol = wet3 * AREA[:, :, None] * DZN[None, None, :]

s0 = JS.JaxStateG(u=jnp.asarray(u0), v=jnp.asarray(v0), T=jnp.asarray(T0),
                  S=jnp.asarray(S0), eta=jnp.asarray(e0))
TERM_NAMES = ["adv", "diff_h", "diff_v", "conv", "gm", "redi"]
term = np.asarray(terms_fn(s0), float)          # (6, nx, ny, nz), tendencies K/s
print("terms_fn shape:", term.shape)
YRS = DT / 3.1536e7                             # years per step
RHO_CP = RHO_0 * C_P / 1e21


def per_level(a):
    """volume-mean of a *per-step delta-T* field -> K/yr."""
    a = np.asarray(a, float)
    out = []
    for k in range(14):
        wk = vol[:, :, k]
        n = wk.sum()
        out.append(0.0 if n <= 0 else float((a[:, :, k] * wk).sum() / n) / YRS)
    return out


def total(a):
    """volume integral of a *per-step delta-T* field -> ZJ/yr."""
    return float((np.asarray(a, float) * vol).sum() * RHO_CP / YRS)


print()
print("=" * 100)
print("Per-term, per-level dT/dt  [K/yr]  and volume-integrated dH/dt [ZJ/yr]")
print("  (terms_fn returns tendencies K/s -> multiply by DT for the per-step delta)")
print("=" * 100)
print("  %-12s" % "term" + "".join("%9s" % ("k%d" % k) for k in range(14)) + "%11s" % "TOT ZJ/yr")
summed = np.zeros((360, 120, 14))
for i, k in enumerate(TERM_NAMES):
    contrib = term[i] * DT                      # K per step
    summed = summed + contrib
    print("  %-12s" % k + "".join("%+9.4f" % x for x in per_level(contrib))
          + "%+11.2f" % total(contrib))
print()
print("  %-12s" % "SUM(listed)" + "".join("%+9.4f" % x for x in per_level(summed))
      + "%+11.2f" % total(summed))

s1 = JS._step_impl(s0, p)
dT = np.asarray(s1.T, float) - np.asarray(s0.T, float)   # K per step
print("  %-12s" % "ACTUAL step" + "".join("%+9.4f" % x for x in per_level(dT))
      + "%+11.2f" % total(dT))
print("  %-12s" % "MISSING" + "".join("%+9.4f" % x for x in per_level(dT - summed))
      + "%+11.2f" % total(dT - summed))
print()
print("  MISSING = surface bulk flux (k=0) + biharmonic kappa_bi + L-step kappa_h")
print("            + polar cap + mask/hold + barotropic projection.")


