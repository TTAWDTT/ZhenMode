"""Where does the -0.0231 ZJ/step actually go?

_decomp.out: applying the model's own advection operator, subcycled exactly
as _compute_tracer_tendency does it, CHANGES H by +0.00763743 ZJ/step.
The full _step_impl on the same state changes H by -0.02306811 ZJ/step.
So the leak is NOT the advection operator's integral -- it is introduced by
the STEPPER around it. Instrument each stage.

Stages of _step_impl (kappa_*=0, lambda_bulk=0 so the L steps are identity
on T):
  1. _linear_half_step x2          -> T unchanged (mask/hold is identity)
  2. _explicit_full_step:
       dT1 = residual(T)           integral = I1
       T_pred = T + dT1*dt
       dT2 = residual(T_pred)      integral = I2
       T_new = T + 0.5*(dT1+dT2)*dt
  3. _polar_cap_3d(T)              -> zonal average near the poles
  4. mask/hold                     -> identity on wet
Measure I1, I2, and the polar-cap delta.
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
ZJ = RHO_0 * C_P / 1e21
g = make_global_grid(GlobalGridConfig(lat_max=60.0, ny=120),
                     DEFAULT_CONFIG.bathymetry_file, smooth_passes=30, min_depth=100.0)
AREA = np.asarray(g.dx_2d, float) * float(g.dy)
d = np.load("results/ckpt_tenyr_ms_gm.npz")
u0 = np.asarray(d["u"], np.float64); v0 = np.asarray(d["v"], np.float64)
T0 = np.asarray(d["T"], np.float64); S0 = np.asarray(d["S"], np.float64)
e0 = np.asarray(d["eta"], np.float64)
init = np.load("init_fields_g360x120.npz")
T_atm_np = np.asarray(air_temp_profile(g, init["T_init"].astype(np.float64)[:, :, 0]), float)

NOTRACER = dict(kappa_conv=0.0, kappa_h=0.0, kappa_v=0.0, kappa_gm=0.0,
                kappa_redi=0.0, kappa_bi=0.0, nu_h=5e6, nu_bi=2e14)

def build(cap=2):
    _, _, _, pp, _ = JS.make_solver_global(
        g, replace(PhysicsConfig(), **NOTRACER), DT, T_atm=jnp.asarray(T_atm_np),
        lambda_bulk=0.0, mode_split=True, dt_bt=300.0, polar_cap_rows=cap,
        polar_cap_taper=(3 if cap > 0 else 0), return_params=True)
    return pp

p = build(2)
wet3 = np.asarray(p.wet_mask_z, float)
DZN = np.asarray(p.dz_node).ravel()
vol = wet3 * AREA[:, :, None] * DZN[None, None, :]

def H(Tf):
    return float((np.asarray(Tf, float) * vol).sum() * ZJ)

def I(f):
    return float((np.asarray(f, float) * vol).sum() * ZJ * DT)

st = JS.JaxStateG(u=jnp.asarray(u0), v=jnp.asarray(v0), T=jnp.asarray(T0),
                  S=jnp.asarray(S0), eta=jnp.asarray(e0))

print("=" * 84)
print("Stage-by-stage heat budget, one step   [ZJ]")
print("=" * 84)
h_start = H(st.T)

# --- stage 1: linear half steps (T identity here) ---
sL = JS._linear_half_step(st, p, DT / 2)
print("  L half-step #1  dH %+14.8f" % (H(sL.T) - h_start))
sN = JS._explicit_full_step(sL, p, DT)
dT1, _ = JS._compute_tracer_residual(sL, p)
T_pred = sL.T + dT1 * DT
print("  N step          dH %+14.8f   (I1 %+14.8f)" % (H(sN.T) - H(sL.T), I(dT1)))
dT2, _ = JS._compute_tracer_residual(
    JS.JaxStateG(sL.u, sL.v, T_pred, sL.S, sL.eta), p)
print("                            I2 %+14.8f" % I(dT2))
print("                            I(0.5*(I1+I2))*dt = %+14.8f"
      % (I(0.5 * (dT1 + dT2))))
sL2 = JS._linear_half_step(sN, p, DT / 2)
print("  L half-step #2  dH %+14.8f" % (H(sL2.T) - H(sN.T)))
cap = JS._polar_cap_3d(sL2.T, p)
print("  polar cap       dH %+14.8f   <<<<<<<<<<<<" % (H(cap) - H(sL2.T)))
wm = p.wet_mask_z
fin = cap * wm + sL2.T * (1.0 - wm)
print("  mask/hold       dH %+14.8f" % (H(fin) - H(cap)))
print("  ----------------")
print("  TOTAL           dH %+14.8f" % (H(fin) - h_start))

print()
print("=" * 84)
print("Is the polar cap the sink?  cap=0 vs cap=2 over 40 steps")
print("=" * 84)

def run(cap, nstep=40):
    pp = build(cap)
    s = JS.JaxStateG(u=jnp.asarray(u0), v=jnp.asarray(v0), T=jnp.asarray(T0),
                     S=jnp.asarray(S0), eta=jnp.asarray(e0))
    h0 = H(s.T)
    for k in range(nstep):
        s = JS._step_impl(s, pp)
        if not np.isfinite(np.asarray(s.T)).all():
            return h0, float("nan"), k + 1
    return h0, H(np.asarray(s.T)), nstep

for cap in (0, 2):
    h0, h1, n = run(cap)
    print("  polar_cap_rows=%d  n=%3d  dH %+14.8f ZJ  (%+10.3f ZJ/yr)"
          % (cap, n, h1 - h0, (h1 - h0) / (n * DT / 3.1536e7)), flush=True)

print()
print("=" * 84)
print("Per-row diagnostic: which latitudes does the cap touch?")
print("=" * 84)
lat = np.asarray(g.lat, float)
ncap = 2; ntap = 3
print("  ny=%d  lat range %.1f .. %.1f" % (len(lat), lat[0], lat[-1]))
print("  rows 0..%d (full cap)  lats %s" % (ncap - 1, np.round(lat[:ncap], 2)))
print("  rows %d..%d (taper)     lats %s" % (ncap, ncap + ntap - 1,
                                             np.round(lat[ncap:ncap + ntap], 2)))
print("  same at the south end: lats %s" % np.round(lat[-(ncap + ntap):], 2))
# ocean fraction in the cap band
wm2 = np.asarray(p.wet_mask, float)
band = np.zeros(len(lat), bool)
band[:ncap + ntap] = True
band[-(ncap + ntap):] = True
print("  ocean fraction in the cap band: %.3f  (vs %.3f global)"
      % (wm2[:, band].mean(), wm2.mean()))
