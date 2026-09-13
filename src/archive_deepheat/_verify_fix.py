"""Verify the kappa_v / nu_v double-application fix.

Same isolation as _nuv_test2.py, but run against the PATCHED module by
importing it fresh.  Expect COEF == 1.0 (was 1.88 tracer / 2.12 momentum).
Also: differencing the kappa_v=0 and kappa_v=KV runs must now give a clean
kappa_v*_d2_dz2*dt field with near-zero residual.
"""
import sys
import numpy as np
import jax, jax.numpy as jnp
jax.config.update("jax_platform_name", "cpu")
sys.path.insert(0, "src")
from dataclasses import replace
from grid import make_global_grid, GlobalGridConfig
from config import DEFAULT_CONFIG, PhysicsConfig
import importlib
import jax_solver_global as JS
importlib.reload(JS)

g = make_global_grid(GlobalGridConfig(lat_max=60.0, ny=120),
                     DEFAULT_CONFIG.bathymetry_file, smooth_passes=30, min_depth=100.0)
DT = 3600.0
KV = 1.0e-4
NUV = 1.0e-3

d = np.load("results/ckpt_tenyr_ms_gm.npz")
T0 = jnp.asarray(d["T"]); S0 = jnp.asarray(d["S"]); ETA = jnp.asarray(d["eta"])
Z = jnp.zeros_like(T0)
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
print("POST-FIX  tracer kappa_v")
print("=" * 92)
s0, p = step_of(); s1, _ = step_of(kappa_v=KV)
st = JS.JaxStateG(Z, Z, T0, S0, ETA)
dd = np.asarray(one(s1, st).T - one(s0, st).T, float)
lap = np.asarray(JS._d2_dz2(T0, p), float)
m = np.asarray(p.wet_mask_z, float) > 0.5
md = m.copy(); md[:, :, 0] = False
num = (dd[md] * lap[md]).sum(); den = (lap[md] ** 2).sum()
print("  implied COEF = %.4f   (want 1.0; pre-fix 1.88)" % (num / den / (KV * DT)))
print("  residual max %.3e  (ddT max %.3e)  ratio %.3f"
      % (np.abs(dd[md] - (num / den) * lap[md]).max(), np.abs(dd[md]).max(),
         np.abs(dd[md] - (num / den) * lap[md]).max() / np.abs(dd[md]).max()))

print()
print("=" * 92)
print("POST-FIX  momentum nu_v")
print("=" * 92)
s0, p = step_of(); s1, _ = step_of(nu_v=NUV)
st = JS.JaxStateG(U0, jnp.zeros_like(U0), T0, S0, ETA)
dd = np.asarray(one(s1, st).u - one(s0, st).u, float)
lap = np.asarray(JS._d2_dz2(U0, p), float)
md = m.copy(); md[:, :, 0] = False; md[:, :, -1] = False
md[:, :2, :] = False; md[:, -2:, :] = False
num = (dd[md] * lap[md]).sum(); den = (lap[md] ** 2).sum()
print("  implied COEF = %.4f   (want 1.0; pre-fix 2.12)" % (num / den / (NUV * DT)))

print()
print("=" * 92)
print("sanity: production config still steps without NaN")
print("=" * 92)
PROD = dict(nu_h=5e6, nu_bi=2e14, kappa_bi=2e14, kappa_gm=1000.0,
            kappa_redi=1000.0, kappa_v=1e-5, kappa_conv=0.05, gm_slope_max=0.005)
from forcing import air_temp_profile, BULK_LAMBDA_DEFAULT
init = np.load("init_fields_g360x120.npz")
T_atm = air_temp_profile(g, init["T_init"].astype(np.float64)[:, :, 0])
step, _, _, p, _ = JS.make_solver_global(
    g, replace(PhysicsConfig(), **PROD), DT, T_atm=jnp.asarray(T_atm),
    lambda_bulk=BULK_LAMBDA_DEFAULT, mode_split=True, dt_bt=300.0,
    return_params=True)
st = JS.JaxStateG(jnp.asarray(d["u"]), jnp.asarray(d["v"]), T0, S0, ETA)
for i in range(20):
    st = step(st)
jax.block_until_ready(st)
print("  20 steps OK. max|T| %.4f  max|u| %.4f  any NaN: %s"
      % (float(jnp.nanmax(jnp.abs(st.T))), float(jnp.nanmax(jnp.abs(st.u))),
         bool(jnp.any(jnp.isnan(st.T)))))
