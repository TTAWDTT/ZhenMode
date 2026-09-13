"""Rigorous version of _nuv_test.py.

Two fixes over the first attempt:
  1. isolate the kappa_v/nu_v-dependent part by DIFFERENCING against the
     same-config run with the coefficient set to 0 (the control is not
     identically zero, so a raw projection is contaminated);
  2. use a vertically VARYING u for the momentum test -- a constant field has
     _d2_dz2 == 0 and made the projection degenerate (den = 0 -> nan).

Expectation from the code:
   N step  : diff_v = kappa_v * _d2_dz2(T)              (inside _compute_tracer_tendency)
   L step  : T += kappa_v * _d2_dz2(T) * (dt/2)         (twice)
   residual: subtracts kappa_h*laplacian ONLY, not kappa_v*_d2_dz2
   => COEF = 2.0.  Correct would be 1.0.
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
KV = 1.0e-4      # large enough that the diffusive signal dominates round-off
NUV = 1.0e-3     # nu_v*dt/dz^2 = 0.144 at dz=5m -- inside the explicit limit

d = np.load("results/ckpt_tenyr_ms_gm.npz")
T0 = jnp.asarray(d["T"])
S0 = jnp.asarray(d["S"])
ETA = jnp.asarray(d["eta"])
Z = jnp.zeros_like(T0)

# vertically varying u so _d2_dz2(u) != 0
zz = np.array([0, -5, -15, -30, -50, -75, -100, -150, -200, -300, -500,
               -1000, -2000, -4000], float)
U0 = jnp.asarray(0.05 + 0.002 * np.cos(zz / 60.0)[None, None, :]
                 * np.ones_like(np.asarray(d["T"])))

BASE = dict(nu_h=0.0, nu_v=0.0, kappa_h=0.0, kappa_v=0.0, kappa_conv=0.0,
            kappa_bi=0.0, kappa_gm=0.0, kappa_redi=0.0, gm_slope_max=0.005)

def step_of(**kw):
    c = dict(BASE); c.update(kw)
    step, _, _, p, _ = JS.make_solver_global(
        g, replace(PhysicsConfig(), **c), DT, lambda_bulk=0.0,
        mode_split=False, return_params=True)
    return step, p

def one(step, st):
    n = step(st); jax.block_until_ready(n); return n

print("=" * 92)
print("TEST 1  tracer: is kappa_v applied once or twice per step?")
print("=" * 92)
s0, p = step_of()
s1, _ = step_of(kappa_v=KV)
st = JS.JaxStateG(Z, Z, T0, S0, ETA)
a0 = one(s0, st); a1 = one(s1, st)
dd = np.asarray(a1.T - a0.T, float)          # pure kappa_v contribution
lap = np.asarray(JS._d2_dz2(T0, p), float)
m = np.asarray(p.wet_mask_z, float) > 0.5
md = m.copy(); md[:, :, 0] = False           # k=0 has the surface BC
num = (dd[md] * lap[md]).sum(); den = (lap[md] ** 2).sum()
print("  <ddT, lap>/<lap,lap> = %.6e   = COEF * kappa_v * dt" % (num / den))
print("  implied COEF = %.4f   (correct 1.0, double-applied 2.0)"
      % (num / den / (KV * DT)))
r = dd[md] - (num / den) * lap[md]
print("  residual after removing the kappa_v*lap component: max %.3e (ddT max %.3e)"
      % (np.abs(r).max(), np.abs(dd[md]).max()))

print()
print("=" * 92)
print("TEST 2  momentum: is nu_v applied once or twice per step?")
print("=" * 92)
s0, p = step_of()
s1, _ = step_of(nu_v=NUV)
st = JS.JaxStateG(U0, jnp.zeros_like(U0), T0, S0, ETA)
a0 = one(s0, st); a1 = one(s1, st)
dd = np.asarray(a1.u - a0.u, float)
lap = np.asarray(JS._d2_dz2(U0, p), float)
m = np.asarray(p.wet_mask_z, float) > 0.5
md = m.copy(); md[:, :, 0] = False; md[:, :, -1] = False
md[:, :2, :] = False; md[:, -2:, :] = False
num = (dd[md] * lap[md]).sum(); den = (lap[md] ** 2).sum()
print("  |lap| max %.4e   |ddU| max %.4e" % (np.abs(lap[md]).max(), np.abs(dd[md]).max()))
print("  implied COEF = %.4f   (correct 1.0, double-applied 2.0)"
      % (num / den / (NUV * DT)))
