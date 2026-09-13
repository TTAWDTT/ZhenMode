"""DECISIVE numeric test for the vertical-diffusion double-application.

Read of the code says:
  * _compute_tracer_tendency (N step, RK2 over dt) contains
        diff_v_T = kappa_v * _d2_dz2(T)
  * _linear_half_step (called TWICE, dt/2 each) contains
        T = T + kappa_v * _d2_dz2(T) * dt_half
  * _compute_tracer_residual subtracts only kappa_h*laplacian_h, NOT kappa_v*d2dz2

=> total vertical diffusion per step = kappa_v*dt (from N) + 2*(kappa_v*dt/2)
   (from L) = 2 * kappa_v * dt.

Test: isolate vertical diffusion. Zero out every other term (u=v=0 so adv=0,
kappa_h=0, kappa_conv=0, kappa_gm=kappa_redi=0, kappa_bi=0, lambda_bulk=0,
Q_heat=0, no wind). Then one step must give
    T_new - T = COEF * _d2_dz2(T) * dt
and COEF is 1.0 if correct, 2.0 if double-applied.

Same test for momentum nu_v (u,v): with everything else off, one step gives
    u_new - u = COEF * nu_v * _d2_dz2(u) * dt.
"""
import sys
import numpy as np
import jax, jax.numpy as jnp
jax.config.update("jax_platform_name", "cpu")
sys.path.insert(0, "src")
from dataclasses import replace
from grid import make_global_grid, GlobalGridConfig
from config import DEFAULT_CONFIG, PhysicsConfig
import jax_solver_global as JS

g = make_global_grid(GlobalGridConfig(lat_max=60.0, ny=120),
                     DEFAULT_CONFIG.bathymetry_file, smooth_passes=30, min_depth=100.0)
DT = 3600.0
KV = 1.0e-5
NUV = 1.0e-3          # chosen big enough to dominate round-off

def build(**kw):
    cfg = dict(nu_h=0.0, nu_v=0.0, kappa_h=0.0, kappa_v=0.0, kappa_conv=0.0,
               kappa_bi=0.0, kappa_gm=0.0, kappa_redi=0.0)
    cfg.update(kw)
    step, _, _, p, _ = JS.make_solver_global(
        g, replace(PhysicsConfig(), **cfg), DT, lambda_bulk=0.0,
        mode_split=False, return_params=True)
    return step, p

d = np.load("results/ckpt_tenyr_ms_gm.npz")
T0 = jnp.asarray(d["T"])
# build a nonzero u,v to test momentum vertical diffusion
rng = np.random.default_rng(0)
U0 = jnp.asarray(rng.normal(0.0, 0.05, d["u"].shape) * np.asarray(d["T"]) * 0.0 + 0.05)
wet = None

def state(u, v, T, S, eta):
    return JS.JaxStateG(u, v, T, S, eta)

print("=" * 90)
print("TEST 1: tracer vertical diffusion")
print("=" * 90)
for label, kv, want in [("kappa_v=0 (control)", 0.0, 0.0),
                        ("kappa_v=%.0e" % KV, KV, 1.0)]:
    step, p = build(kappa_v=kv)
    st = state(jnp.zeros_like(d["u"]), jnp.zeros_like(d["v"]), T0,
               jnp.asarray(d["S"]), jnp.asarray(d["eta"]))
    new = step(st)
    jax.block_until_ready(new)
    dT = np.asarray(new.T - T0, float)
    lap = np.asarray(JS._d2_dz2(T0, p), float)
    m = np.asarray(p.wet_mask_z, float) > 0.5
    # exclude the top node: surface_mask/boundary handling differs there
    md = m.copy(); md[:, :, 0] = False
    num = (dT[md] * lap[md]).sum()
    den = (lap[md] ** 2).sum()
    coef = num / den / (kv * DT) if kv else float("nan")
    print("  %-22s  max|dT| = %.6e   implied COEF (want %.1f) = %s"
          % (label, np.abs(dT[md]).max(), want,
             ("%.4f" % coef) if kv else "n/a"))

print()
print("=" * 90)
print("TEST 2: momentum vertical diffusion (nu_v)")
print("=" * 90)
for label, nuv, want in [("nu_v=0 (control)", 0.0, 0.0),
                         ("nu_v=%.0e" % NUV, NUV, 1.0)]:
    step, p = build(nu_v=nuv)
    st = state(U0, jnp.zeros_like(U0), T0, jnp.asarray(d["S"]),
               jnp.asarray(d["eta"]))
    new = step(st)
    jax.block_until_ready(new)
    du = np.asarray(new.u - U0, float)
    lap = np.asarray(JS._d2_dz2(U0, p), float)
    # exclude boundaries where the mask/cliff handling dominates
    md = np.asarray(p.wet_mask_z, float) > 0.5
    md[:, :, 0] = False
    md[:, :, -1] = False
    md[:, 0, :] = False
    md[:, -1, :] = False
    md[:, :, :2] = False
    num = (du[md] * lap[md]).sum()
    den = (lap[md] ** 2).sum()
    coef = num / den / (nuv * DT) if nuv else float("nan")
    print("  %-22s  max|du| = %.6e   implied COEF (want %.1f) = %s"
          % (label, np.abs(du[md]).max(), want,
             ("%.4f" % coef) if nuv else "n/a"))
