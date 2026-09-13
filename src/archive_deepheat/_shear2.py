"""Refine the shear fix: normalize by the ACTUAL wet column depth H_col.

The shear variant in _sheartest.out cuts the heat leak 65x (+66.58 ->
-1.02 ZJ/yr) while keeping adv(1) at machine zero. The residual should be
the column-depth normalization.

Derivation. _advection_scalar's vertical branch reaches
  Fz[0] = sum_k div_h(u_k)*dz[k]
and div_h is LINEAR in the velocity, per level, with the SAME operator L at
every level, so
  sum_k div_h(u_k)*dz[k] = div_h( sum_k u_k*dz[k] ) = L( L_span * ubt_raw )
with L_span = sum_k dz[k] = 5002.5.

Now advect with u'_k = wm_k*(u_k - ubt_2d). The mask wm_k varies with k in
partial columns, and
  sum_k div_h(wm_k u_k) dz[k] = L( sum_k wm_k u_k dz[k] ) = L( H_col * ubt_noded )
  sum_k div_h(wm_k ubt_2d) dz[k] = L( ubt_2d * H_col )
so Fz'[0] = 0 EXACTLY iff
  ubt_2d = sum_k wm_k*u_k*dz[k] / sum_k wm_k*dz[k] = ubt_noded
i.e. normalized by H_col (the WET column depth), not by L_span. _sheartest
used L_span -> the -1.02 residual.

Variants:
  base    raw u, v
  shearL  u - (col sum*1/L_span)          [previous, -1.02]
  shearH  u - (col sum / H_col)           [predicted 0]
  shearB  u - _barotropic_velocity        [the wrong depth average, control]
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
g = make_global_grid(GlobalGridConfig(lat_max=60.0, ny=120),
                     DEFAULT_CONFIG.bathymetry_file, smooth_passes=30, min_depth=100.0)
AREA = np.asarray(g.dx_2d, float) * float(g.dy)
d = np.load("results/ckpt_tenyr_ms_gm.npz")
u0 = jnp.asarray(np.asarray(d["u"], np.float64))
v0 = jnp.asarray(np.asarray(d["v"], np.float64))
T0 = jnp.asarray(np.asarray(d["T"], np.float64))
S0 = jnp.asarray(np.asarray(d["S"], np.float64))
e0 = jnp.asarray(np.asarray(d["eta"], np.float64))
init = np.load("init_fields_g360x120.npz")
T_atm_np = np.asarray(air_temp_profile(g, init["T_init"].astype(np.float64)[:, :, 0]), float)
BASE = dict(nu_h=5e6, nu_bi=2e14, kappa_bi=2e14, kappa_gm=1000.0,
            kappa_redi=1000.0, kappa_v=1e-5, kappa_conv=0.05, gm_slope_max=0.005)
_, _, _, p, _ = JS.make_solver_global(
    g, replace(PhysicsConfig(), **BASE), 3600.0, T_atm=jnp.asarray(T_atm_np),
    lambda_bulk=BULK_LAMBDA_DEFAULT, mode_split=True, dt_bt=300.0, return_params=True)

wet3 = np.asarray(p.wet_mask_z, float)
DZN = np.asarray(p.dz_node).ravel()
vol = wet3 * AREA[:, :, None] * DZN[None, None, :]
st = JS.JaxStateG(u=u0, v=v0, T=T0, S=S0, eta=e0)
surfm = wet3[:, :, 0] > 0.5

def heat_rate(adv):
    return float((np.asarray(adv, float) * vol).sum()) * RHO_0 * C_P * 3.1536e7 / 1e21

def shearL(u, v, pp):
    wm = pp.wet_mask_z
    L = jnp.sum(pp.dz_node)
    ub = jnp.sum(u * wm * pp.dz_node, axis=-1) / L
    vb = jnp.sum(v * wm * pp.dz_node, axis=-1) / L
    return (u - ub[:, :, None]) * wm, (v - vb[:, :, None]) * wm

def shearH(u, v, pp):
    wm = pp.wet_mask_z
    H = jnp.sum(wm * pp.dz_node, axis=-1)
    H = jnp.where(H > 0, H, 1.0)
    ub = jnp.sum(u * wm * pp.dz_node, axis=-1) / H
    vb = jnp.sum(v * wm * pp.dz_node, axis=-1) / H
    return (u - ub[:, :, None]) * wm, (v - vb[:, :, None]) * wm

def shearB(u, v, pp):
    ub, vb = JS._barotropic_velocity(u, v, pp)
    wm = pp.wet_mask_z
    return (u - ub[:, :, None]) * wm, (v - vb[:, :, None]) * wm

def evaluate(tag, uu, vv):
    Fz = JS._vertical_transport_iface(uu, vv, p)
    adv = JS._advection_scalar(st.T, uu, vv, Fz, p)
    a1 = np.asarray(JS._advection_scalar(jnp.ones_like(st.T), uu, vv, Fz, p), float)
    f0 = np.asarray(Fz[:, :, 0], float)
    r = np.array([(np.asarray(adv, float)[:, :, k] * vol[:, :, k]).sum()
                  for k in range(14)]) * RHO_0 * C_P * 3.1536e7 / 1e21
    print("  %-7s heat %+9.5f | adv(1) %.3e | Fz0 rms %.3e | sumA*Fz0 %+.2e"
          % (tag, heat_rate(adv), np.abs(a1[wet3 > 0.5]).max(),
             np.sqrt((f0[surfm] ** 2).mean()), (f0 * AREA * surfm).sum()))
    print("          per level: " + " ".join("%+6.1f" % x for x in r))

print("=" * 98)
print("Heat leak (ZJ/yr, must be 0) and mass consistency (adv(1), must be ~1e-21)")
print("=" * 98)
evaluate("base", st.u, st.v)
usL, vsL = shearL(st.u, st.v, p); evaluate("shearL", usL, vsL)
usH, vsH = shearH(st.u, st.v, p); evaluate("shearH", usH, vsH)
usB, vsB = shearB(st.u, st.v, p); evaluate("shearB", usB, vsB)

# also: does shearH differ from shearL much?
m = wet3 > 0.5
dif = np.asarray(st.u - usH, float) - np.asarray(st.u - usL, float)
print("\nshearH vs shearL removed-velocity difference: rms %.4e m/s (%.2f%% of u rms)"
      % (np.sqrt((dif[m] ** 2).mean()),
         100 * np.sqrt((dif[m] ** 2).mean()) / np.sqrt((np.asarray(st.u, float)[m] ** 2).mean())))

# ── time gate ──
print("\n" + "=" * 98)
print("TIME INTEGRATION (advection-only, closed, no Q): dH must be EXACTLY 0")
print("=" * 98)
NOTRACER = dict(kappa_conv=0.0, kappa_h=0.0, kappa_v=0.0, kappa_gm=0.0,
                kappa_redi=0.0, kappa_bi=0.0, nu_h=5e6, nu_bi=2e14)
_, _, _, pN, _ = JS.make_solver_global(
    g, replace(PhysicsConfig(), **NOTRACER), 3600.0, T_atm=jnp.asarray(T_atm_np),
    lambda_bulk=0.0, mode_split=True, dt_bt=300.0, return_params=True)
_orig_vti = JS._vertical_transport_iface
volN = wet3 * AREA[:, :, None] * np.asarray(pN.dz_node).ravel()[None, None, :]

def heatT(T):
    return float((np.asarray(T, float) * volN).sum()) * RHO_0 * C_P / 1e21

def run(tag, fn, nstep=60):
    JS._vertical_transport_iface = fn
    s = JS.JaxStateG(u=u0, v=v0, T=T0, S=S0, eta=e0)
    h0 = heatT(T0)
    for k in range(nstep):
        s = JS._step_impl(s, pN)
        if not np.isfinite(np.asarray(s.T)).all():
            print("  %-7s NaN at step %d" % (tag, k + 1)); return
    h1 = heatT(np.asarray(s.T))
    print("  %-7s dH %+12.6f ZJ over %d steps  (%+8.3f ZJ/yr)"
          % (tag, h1 - h0, nstep, (h1 - h0) / (nstep * 3600.0 / 3.1536e7)), flush=True)

run("base", _orig_vti)
def mk(fn):
    def vti(u, v, pp):
        uu, vv = fn(u, v, pp)
        return _orig_vti(uu, vv, pp)
    return vti
run("shearH", mk(shearH))
JS._vertical_transport_iface = _orig_vti
