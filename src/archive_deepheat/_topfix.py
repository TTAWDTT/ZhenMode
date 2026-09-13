"""Test the top-face closure fix for the tracer advection.

Finding: the tracer advection operator creates +66.58 ZJ/yr, and the WHOLE of
it is the top-face term  sum_columns area * Fz[0] * T[0], where Fz[0] is the
column-integrated horizontal divergence of the 3D velocity.  Every interior
face telescopes exactly (adv(T=1) == 0 pointwise to 1e-21).

Fz[0] is NOT the free-surface signal: rms 1.28e-5 m^2/s vs -deta/dt rms
3.5e-7, correlation +0.018.  It is mode-split projection residual.

Variants tested:
  V0  baseline
  V1  top-face advective flux = 0   (no heat crosses the free surface)
"""
import sys, time
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
d = np.load("results/ckpt_tenyr_ms_gm.npz")
init = np.load("init_fields_g360x120.npz")
T_atm_np = air_temp_profile(g, init["T_init"].astype(np.float64)[:, :, 0])
phys = replace(PhysicsConfig(), nu_h=5e6, nu_bi=2e14, kappa_bi=2e14,
               kappa_gm=1000.0, kappa_redi=1000.0, kappa_v=1e-5,
               kappa_conv=0.05, gm_slope_max=0.005)

src = open("src/jax_solver_global.py", encoding="utf-8").read()
OLD = "    Fz_top = Fz_in[:, :, :1] * T[..., :1]"
assert src.count(OLD) == 1
NEW = "    Fz_top = jnp.zeros_like(T[..., :1])"
patched = src.replace(OLD, NEW)
open("_js_topzero.py", "w", encoding="utf-8").write(patched)
print("patched _js_topzero.py written")

import _js_topzero as JSZ

wet3 = np.asarray(p := None) if False else None
st = None

def mk(mod):
    step, _, _, p, terms = mod.make_solver_global(
        g, phys, 3600.0, T_atm=jnp.asarray(T_atm_np), lambda_bulk=BULK_LAMBDA_DEFAULT,
        mode_split=True, dt_bt=300.0, return_params=True)
    return step, p, terms

step0, p, terms0 = mk(JS)
step1, _, terms1 = mk(JSZ)
wet3 = np.asarray(p.wet_mask_z, float)
DZN = np.asarray(p.dz_node).ravel()
vol = wet3 * AREA[:, :, None] * DZN[None, None, :]
ZJYR = 1025.0 * 3992.0 / 1e21 * 3.1536e7
T0 = np.asarray(d["T"], np.float64); S0 = np.asarray(d["S"], np.float64)
u0 = np.asarray(d["u"], np.float64); v0 = np.asarray(d["v"], np.float64)
e0 = np.asarray(d["eta"], np.float64)
st = JS.JaxStateG(u=jnp.asarray(u0), v=jnp.asarray(v0), T=jnp.asarray(T0),
                  S=jnp.asarray(S0), eta=jnp.asarray(e0))

def cs(f):
    f = np.asarray(f)
    return np.array([(f[:, :, k] * vol[:, :, k]).sum() * ZJYR for k in range(14)])

print("\n" + "=" * 96)
print("A. advection conservation")
print("=" * 96)
Fz0 = JS._vertical_transport_iface(st.u, st.v, p)
a0 = cs(JS._advection_scalar(st.T, st.u, st.v, Fz0, p))
a1 = cs(JSZ._advection_scalar(st.T, st.u, st.v,
        JSZ._vertical_transport_iface(st.u, st.v, p), p))
print("baseline adv_T : " + " ".join("%+7.1f" % x for x in a0) + " | %+8.3f" % a0.sum())
print("topzero  adv_T : " + " ".join("%+7.1f" % x for x in a1) + " | %+8.3f" % a1.sum())

print("\n" + "=" * 96)
print("B. one-step OHC tendency")
print("=" * 96)
def prof(step, tag):
    t = time.time()
    s1 = step(st); jax.block_until_ready(s1)
    r = cs(np.asarray(s1.T) - T0)
    print("%-12s (%2.0fs) " % (tag, time.time() - t) + " ".join("%+7.1f" % x for x in r)
          + " | %+8.2f" % r.sum())
    return r
r0 = prof(step0, "baseline")
r1 = prof(step1, "topzero")
print("delta        " + " ".join("%+7.1f" % x for x in (r1 - r0)) + " | %+8.2f" % (r1.sum() - r0.sum()))

print("\n" + "=" * 96)
print("C. stability: 40-step integration")
print("=" * 96)
for step, tag in [(step0, "baseline"), (step1, "topzero")]:
    s = st
    t = time.time()
    ok = True
    for i in range(40):
        s = step(s)
    jax.block_until_ready(s)
    Tn = np.asarray(s.T)
    print("%-10s 40 steps in %5.1fs  min %.4f max %.4f  nan %d  dOHC %+.4f ZJ"
          % (tag, time.time() - t, Tn.min(), Tn.max(), int(np.isnan(Tn).sum()),
             ((Tn - T0) * vol).sum() * 1025 * 3992 / 1e21))
