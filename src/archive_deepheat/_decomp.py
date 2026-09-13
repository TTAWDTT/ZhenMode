"""Decompose the ACTUAL tracer tendency integral - find the dominant leak.

_trace.out: with every closure off and no forcing, the model loses
-0.0231 ZJ/step, while the raw advection operator integral is +0.0076 ZJ/step.
So the dominant sink is NOT _advection_scalar. Candidates in the M-step path:

  _compute_tracer_tendency
      adv (subcycled, adv_nsub=6)
      + kappa_h*laplacian, kappa_v*d2dz2, conv, gm, redi   (all 0 here)
      + bulk flux                                        (0 here)
      then _dealias_h_fd(tend)            <-- 2/3 FFT in lon + binomial in lat
  _compute_tracer_residual = tendency - kappa_h*lap - kappa_v*d2dz2
  _explicit_full_step: RK2 T_new = T + 0.5*(dT1+dT2)*dt
  _linear_half_step: T += kappa_h*lap*dt/2 twice   (0 here), mask-and-hold

Measure each integral separately so the culprit is unambiguous.
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
_, _, _, p, _ = JS.make_solver_global(
    g, replace(PhysicsConfig(), **NOTRACER), DT, T_atm=jnp.asarray(T_atm_np),
    lambda_bulk=0.0, mode_split=True, dt_bt=300.0, return_params=True)

wet3 = np.asarray(p.wet_mask_z, float)
DZN = np.asarray(p.dz_node).ravel()
vol = wet3 * AREA[:, :, None] * DZN[None, None, :]
AREAJ = AREA[:, :, None] * DZN[None, None, :]      # no wet gate, for filter sums

def I(f):
    """ZJ per step of a tendency field (K/s)."""
    return float((np.asarray(f, float) * vol).sum() * ZJ * DT)

st = JS.JaxStateG(u=jnp.asarray(u0), v=jnp.asarray(v0), T=jnp.asarray(T0),
                  S=jnp.asarray(S0), eta=jnp.asarray(e0))

print("=" * 90)
print("Integrals in ZJ per (dt=3600 s) step. Exact conservation requires 0.")
print("=" * 90)

Fz = JS._vertical_transport_iface(st.u, st.v, p)
adv_raw = JS._advection_scalar(st.T, st.u, st.v, Fz, p)
print("  adv raw (no dealias)                %+14.8f" % I(adv_raw))

adv_dl = JS._dealias_h_fd(adv_raw, p)
print("  dealias(adv raw)                    %+14.8f" % I(adv_dl))
print("  --> dealias alone leaks             %+14.8f" % (I(adv_dl) - I(adv_raw)))

# the subcycled advection as _compute_tracer_tendency actually applies it
n_a = int(p.adv_nsub)
dts = DT / n_a
tT = st.T
acc = []
for _ in range(n_a):
    aT = JS._advection_scalar(tT, st.u, st.v, Fz, p)
    acc.append(aT)
    tT = tT + aT * dts
adv_sub = jnp.mean(jnp.stack(acc), axis=0)
print("  adv subcycled mean rate n=%d        %+14.8f" % (n_a, I(adv_sub)))
print("  dealias(adv subcycled)              %+14.8f" % I(JS._dealias_h_fd(adv_sub, p)))

full = JS._compute_tracer_tendency(st, p)
print("  FULL _compute_tracer_tendency       %+14.8f" % I(full[0]))
res = JS._compute_tracer_residual(st, p)
print("  _compute_tracer_residual            %+14.8f" % I(res[0]))

print()
print("=" * 90)
print("Does the dealias filter conserve under the AREA-weighted inner product?")
print("=" * 90)
ones = jnp.ones_like(st.T)
print("  I(dealias(ones))                    %+14.8e  (must be 0)" % I(JS._dealias_h_fd(ones, p) - ones))
f = np.asarray(st.T, float)
filt = np.asarray(JS._dealias_h_fd(st.T, p), float)
print("  I(filt(T)) - I(T)                   %+14.8f ZJ/step" % (I(filt) - I(st.T)))

print()
print("=" * 90)
print("One actual step, decomposed")
print("=" * 90)
H = lambda Tf: float((np.asarray(Tf, float) * vol).sum() * ZJ)
h0 = H(st.T)
s1 = JS._step_impl(st, p)
h1 = H(s1.T)
print("  H0 %.6f  H1 %.6f  dH %+14.8f" % (h0, h1, h1 - h0))

# now the pure advection effect over the same dt, computed by hand
T_adv = np.asarray(st.T, float).copy()
Tn = np.asarray(st.T, float)
for _ in range(n_a):
    aT = np.asarray(JS._advection_scalar(jnp.asarray(Tn), st.u, st.v, Fz, p), float)
    Tn = Tn + aT * dts
print("  hand subcycled advection only: dH %+14.8f" % (H(Tn) - h0))
