"""The -268 ZJ/yr is RK2 stage-2 advection by a PGF-accelerated velocity.

At u=v=0: residual(state)=0 exactly -> dT1=0, T_pred=T.
But momentum residual != 0 (baroclinic PGF from rho'), so
   u_pred = 0 + du1*dt   != 0
and stage 2 evaluates dT2 = advection(T, u_pred, v_pred, Fz_pred).
T_new = T + 0.5*(dT1+dT2)*dt = T + 0.5*dT2*dt.

So the ENTIRE step's heat change is the advection operator applied to the
real T field with a velocity that has a large, unbalanced (undamped when
nu_h=0) divergent component.

Hypothesis: the discrete advection is NOT heat-conservative because
Fz[...,0] = column-integrated horizontal divergence != 0 (the docstring's
"rigid-lid leak"). The vertical flux at the surface node is Fz[0]*T[0], so
the column gains heat Fz[0]*T[0] -- a spurious source proportional to the
velocity, with NO counterpart in any other column. That is the "advection
heat integral near-singular in velocity".

TEST: enforce sum_k div_h[k]*dz_node[k] == 0 exactly by removing the
column mean of div_h before the cumsum, then re-measure. This makes
Fz[0] == 0 by construction.
"""
import sys
import numpy as np
import jax, jax.numpy as jnp
jax.config.update("jax_platform_name", "cpu")
sys.path.insert(0, "src")
from dataclasses import replace as _dc
from grid import make_global_grid, GlobalGridConfig
from config import DEFAULT_CONFIG, PhysicsConfig
import jax_solver_global as JS

RHO_0, C_P = 1025.0, 3992.0
RHO_CP = RHO_0 * C_P / 1e21
DT = 3600.0
SEC = 3.1536e7

g = make_global_grid(GlobalGridConfig(lat_max=60.0, ny=120),
                     DEFAULT_CONFIG.bathymetry_file, smooth_passes=30, min_depth=100.0)
ALL_OFF = dict(nu_h=0.0, nu_bi=0.0, kappa_bi=0.0, kappa_gm=0.0, kappa_redi=0.0,
               kappa_v=0.0, kappa_conv=0.0, gm_slope_max=0.005, kappa_h=0.0,
               T_ref=20.0, S_ref=35.0)
_, _, _, p, _ = JS.make_solver_global(
    g, _dc(PhysicsConfig(), **ALL_OFF), DT, T_atm=jnp.zeros((g.nx, g.ny)),
    lambda_bulk=0.0, mode_split=True, dt_bt=300.0,
    polar_cap_rows=2, polar_cap_taper=3, return_params=True)

wet3 = np.asarray(p.wet_mask_z, float)
DZN = np.asarray(p.dz_node).ravel()
AREA = np.asarray(g.dx_2d, float) * float(g.dy)
vol = wet3 * AREA[:, :, None] * DZN[None, None, :]
nx, ny, nz = int(p.nx), int(p.ny), int(p.nz)

d = np.load("results/ckpt_tenyr_ms_gm.npz")
T0 = np.asarray(d["T"], np.float64)
S0 = np.asarray(d["S"], np.float64)
zero = jnp.zeros((nx, ny, nz))
s0 = JS.JaxStateG(u=zero, v=zero, T=jnp.asarray(T0), S=jnp.asarray(S0),
                  eta=jnp.zeros((nx, ny)))


def dE(f):
    return float((np.asarray(f, float) * vol).sum()) * RHO_CP


print("=" * 100)
print("RK2 stage decomposition at u=v=0")
print("=" * 100)
dT1, dS1 = JS._compute_tracer_residual(s0, p)
print("  stage1 dT1: max=%.3e  dE/step=%+ .4f" % (np.abs(np.asarray(dT1)).max(), dE(np.asarray(dT1)) * DT))
du1, dv1 = JS._compute_momentum_residual(s0, p)
up = np.asarray(du1) * DT
vp = np.asarray(dv1) * DT
print("  stage1 momentum residual * dt : max|u_pred| = %.4e m/s   max|v_pred| = %.4e m/s"
      % (np.abs(up).max(), np.abs(vp).max()))
print("    u_pred rms over wet = %.4e  (0.05 m/s is a strong current)" % np.sqrt((up[wet3 > 0] ** 2).mean()))

sp = JS.JaxStateG(u=jnp.asarray(du1) * DT, v=jnp.asarray(dv1) * DT,
                  T=s0.T + dT1 * DT, S=s0.S + dS1 * DT, eta=s0.eta)
dT2, dS2 = JS._compute_tracer_residual(sp, p)
dT2 = np.asarray(dT2)
print("  stage2 dT2: max=%.3e K/s  dE/step=%+ .4f   <-- THE LEAK" % (np.abs(dT2).max(), dE(dT2) * DT))
print("    dE/step as ZJ/yr: %+ .3f" % (dE(dT2) * DT / (DT / SEC)))

# Fz[0] (surface) at the predicted velocity
Fzp = np.asarray(JS._vertical_transport_iface(sp.u, sp.v, p), float)
print()
print("  Fz[...,0] (column-integrated horizontal divergence) at predicted velocity:")
print("    max|Fz0| = %.6e   rms = %.6e" % (np.abs(Fzp[:, :, 0]).max(),
                                            np.sqrt((Fzp[:, :, 0] ** 2).mean())))
Tsurf = T0[:, :, 0]
flx = Fzp[:, :, 0] * Tsurf
print("    surface heat flux Fz0*T0: dE/step = %+ .5f  -> %+ .3f ZJ/yr"
      % (dE(flx[:, :, None]), dE(flx[:, :, None]) / (DT / SEC)))
print("    (compare total stage2 leak %+ .3f ZJ/yr)" % (dE(dT2) * DT / (DT / SEC)))

# ---- the fix: remove the column mean of div_h so Fz[0] == 0 exactly ----
print()
print("=" * 100)
print("FIX TEST: enforce sum_k div_h[k]*dz_node[k] == 0 (column-mean removal)")
print("=" * 100)
ORIG_VTI = JS._vertical_transport_iface


def vti_fixed(u, v, pp):
    div_h = JS._divergence_h(u, v, pp)
    integrand = div_h * pp.dz_node
    # column mean of the integrand -> subtract so the sum is exactly 0
    w = pp.wet_mask_z
    wsum = jnp.maximum(jnp.sum(w * pp.dz_node, axis=-1, keepdims=True), 1e-30)
    mean = jnp.sum(integrand, axis=-1, keepdims=True) / wsum
    integrand = (integrand - mean * w)
    Fz_int = jnp.cumsum(integrand[..., ::-1], axis=-1)[..., ::-1]
    return jnp.concatenate([Fz_int, jnp.zeros_like(Fz_int[..., :1])], axis=-1)


JS._vertical_transport_iface = vti_fixed
Fzf = np.asarray(JS._vertical_transport_iface(sp.u, sp.v, p), float)
print("  after fix: max|Fz0| = %.6e  (was %.6e)" % (np.abs(Fzf[:, :, 0]).max(), np.abs(Fzp[:, :, 0]).max()))
dT2f, _ = JS._compute_tracer_residual(sp, p)
dT2f = np.asarray(dT2f)
print("  stage2 dT2 after fix: max=%.3e  dE/step=%+ .4f  -> %+ .3f ZJ/yr"
      % (np.abs(dT2f).max(), dE(dT2f) * DT, dE(dT2f) * DT / (DT / SEC)))
print("  leakage removed: %+ .3f ZJ/yr" % ((dE(dT2) - dE(dT2f)) * DT / (DT / SEC)))

# full step with the fix
st_f = JS._step_impl(s0, p)
dTf_full = np.asarray(st_f.T, float) - T0
print()
print("  FULL STEP with fix: dE/step=%+ .6f  -> %+ .3f ZJ/yr   max|dT|=%.3e"
      % (dE(dTf_full), dE(dTf_full) / (DT / SEC), np.abs(dTf_full).max()))
JS._vertical_transport_iface = ORIG_VTI
st_o = JS._step_impl(s0, p)
dTo_full = np.asarray(st_o.T, float) - T0
print("  FULL STEP original : dE/step=%+ .6f  -> %+ .3f ZJ/yr   max|dT|=%.3e"
      % (dE(dTo_full), dE(dTo_full) / (DT / SEC), np.abs(dTo_full).max()))
