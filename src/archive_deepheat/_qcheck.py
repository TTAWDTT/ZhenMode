"""EXACT heat-conservation test WITH the bulk flux on.

Accumulate the surface heat actually applied each step (computed from the
state, no sampling/aliasing) and compare to the heat-content change.

If dH == SUM(Q dt), the solver conserves heat and the apparent 10-yr loss in
the (annual-sampled) archive is seasonal aliasing. If dH << SUM(Q dt), heat
is being destroyed in the surface layer / advection.
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
DT = 3600.0
g = make_global_grid(GlobalGridConfig(lat_max=60.0, ny=120),
                     DEFAULT_CONFIG.bathymetry_file, smooth_passes=30, min_depth=100.0)
AREA = np.asarray(g.dx_2d, float) * float(g.dy)
d = np.load("results/ckpt_tenyr_ms_gm.npz")
ST0 = JS.JaxStateG(**{k: jnp.asarray(np.asarray(d[k], np.float64))
                      for k in ("u", "v", "T", "S", "eta")})
init = np.load("init_fields_g360x120.npz")
T_atm_np = np.asarray(air_temp_profile(g, init["T_init"].astype(np.float64)[:, :, 0]), float)
BASE = dict(nu_h=5e6, nu_bi=2e14, kappa_bi=2e14, kappa_gm=1000.0,
            kappa_redi=1000.0, kappa_v=1e-5, kappa_conv=0.05, gm_slope_max=0.005)

_, _, _, p, _ = JS.make_solver_global(
    g, replace(PhysicsConfig(), **BASE), DT, T_atm=jnp.asarray(T_atm_np),
    lambda_bulk=BULK_LAMBDA_DEFAULT, mode_split=True, dt_bt=300.0, return_params=True)
wet3 = np.asarray(p.wet_mask_z, float)
surf = np.asarray(p.surface_mask, float)
DZN = np.asarray(p.dz_node).ravel()
vol = wet3 * AREA[:, :, None] * DZN[None, None, :]
ZJ = RHO_0 * C_P / 1e21


def run(nstep=200, lam=BULK_LAMBDA_DEFAULT, Tatm=None, cap=2, tstart=0):
    phys = replace(PhysicsConfig(), **BASE)
    ta = jnp.asarray(T_atm_np if Tatm is None else Tatm)
    step, _, _, p, _ = JS.make_solver_global(
        g, phys, DT, T_atm=ta, lambda_bulk=lam, mode_split=True, dt_bt=300.0,
        polar_cap_rows=cap, return_params=True)
    s = ST0
    qsum = 0.0
    H = lambda st: float((np.asarray(st.T, float) * vol).sum() * ZJ)
    H0 = H(s)
    A = AREA * wet3[:, :, 0]
    for _ in range(nstep):
        q = np.asarray(ta, float) - np.asarray(s.T, float)[:, :, 0]
        qsum += (lam * q * A).sum() * DT / 1e21
        s = step(s)
    jax.block_until_ready(s)
    return qsum, H(s) - H0


print("=== 200-step EXACT heat budget, bulk flux ON ===")
print("%-26s %12s %12s %12s %10s" % ("arm", "SUM(Q dt)", "dH", "residual", "retained"))
for tag, kw in [("baseline (lam=40)", {}),
                ("cap0", dict(cap=0)),
                ("lam=0", dict(lam=0.0)),
                ("Tatm=SST (Q=0)", dict(Tatm=np.asarray(ST0.T, float)[:, :, 0]))]:
    q, dh = run(**kw)
    print("%-26s %+12.3f %+12.3f %+12.3f %9.1f%%"
          % (tag, q, dh, dh - q, 100.0 * dh / q if abs(q) > 1e-9 else float("nan")), flush=True)

print("\n=== does the loss scale with step count? (leak rate check) ===")
for n in (50, 100, 200):
    q, dh = run(nstep=n)
    print("  nstep=%3d  SUM(Q dt) %+8.3f  dH %+8.3f  residual %+8.3f  (%.1f%% retained)"
          % (n, q, dh, dh - q, 100.0 * dh / q), flush=True)
