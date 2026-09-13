"""Reconcile: operator integral says +66.58 ZJ/yr, time integration said -204.7.

If dT/dt == adv exactly, then over n steps
    dH  ==  sum_steps  [ integral adv(T_n) dV ] * dt
Any gap is introduced by the STEPPER (masking / polar cap / land hold /
Strang residual structure), not by the advection operator.

Also reports the per-step rate series, so a sign flip or a transient spike
is immediately visible.
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
ZJYR = RHO_0 * C_P / 1e21 * 3.1536e7
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
ZJ = RHO_0 * C_P / 1e21

def H(T):
    return float((np.asarray(T, float) * vol).sum() * ZJ)

print("physics: kappa_* all 0, lambda_bulk=0 -> T changes ONLY by advection")
print("mode_split=True dt=3600 dt_bt=300, adv_nsub=%s" % (p.adv_nsub if hasattr(p, 'adv_nsub') else '?'))
print()
print("%5s %14s %14s %14s %13s" % ("step", "H (ZJ)", "dH/step (ZJ)", "int adv*dt", "ratio"))
s = JS.JaxStateG(u=jnp.asarray(u0), v=jnp.asarray(v0), T=jnp.asarray(T0),
                 S=jnp.asarray(S0), eta=jnp.asarray(e0))
Hprev = H(T0)
cum_dh = 0.0
cum_op = 0.0
NSTEP = 60
for k in range(NSTEP):
    # operator integral at the CURRENT state (before the step)
    Fz = JS._vertical_transport_iface(s.u, s.v, p)
    adv = JS._advection_scalar(s.T, s.u, s.v, Fz, p)
    op = float((np.asarray(adv, float) * vol).sum() * ZJ * DT)
    s = JS._step_impl(s, p)
    Hn = H(np.asarray(s.T))
    dh = Hn - Hprev
    cum_dh += dh; cum_op += op
    if k < 12 or k % 10 == 9:
        print("%5d %14.6f %+14.8f %+14.8f %13.4f"
              % (k + 1, Hn, dh, op, dh / op if abs(op) > 1e-15 else float("nan")), flush=True)
    Hprev = Hn
    if not np.isfinite(Hn):
        print("NaN at step %d" % (k + 1)); break
print()
print("cumulative dH      %+12.6f ZJ  (%.3f ZJ/yr)" % (cum_dh, cum_dh / (NSTEP * DT / 3.1536e7)))
print("cumulative int adv %+12.6f ZJ  (%.3f ZJ/yr)" % (cum_op, cum_op / (NSTEP * DT / 3.1536e7)))
print("gap (stepper)      %+12.6f ZJ  = %.1f%% of |dH|"
      % (cum_dh - cum_op, 100 * (cum_dh - cum_op) / abs(cum_dh) if abs(cum_dh) > 1e-12 else float('nan')))
