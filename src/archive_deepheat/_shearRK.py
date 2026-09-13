"""Does the shear projection cure the RK2 ill-conditioning AND conserve heat
over a real time integration?

_stage2.out: I(dT1)=+0.00763743 but I(dT2@pred)=-0.05343117. The advection
operator's heat integral is a near-singular function of velocity (a 4%
velocity change flips its sign) because it is the residual of the huge
cancellation that Fz[0] embodies:  I(adv(T)) = -sum AREA*Fz[0]*T[0].

If the leak is the cause, removing it (shear projection, _sheartest.out:
-1.02 ZJ/yr vs +66.58) must ALSO make dT2@pred agree with dT2@old, and must
make the multi-step H budget flat.

Projection used (standard rigid-lid): advect with u' = u - ubt where
ubt = sum_k u_k*dz_node_k / sum_k dz_node_k, so the column integral of
div_h(u') vanishes identically -> Fz'[0] = 0 exactly, and mass stays exact.

Patch the two module globals _compute_tracer_tendency looks up:
  _vertical_transport_iface(u,v,p)  -> project first, then original
  _advection_scalar(T,u,v,Fz,p)     -> project u,v
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
    lambda_bulk=0.0, mode_split=True, dt_bt=300.0, polar_cap_rows=2,
    polar_cap_taper=3, return_params=True)

wet3 = np.asarray(p.wet_mask_z, float)
DZN = np.asarray(p.dz_node).ravel()
AREA = np.asarray(g.dx_2d, float) * float(g.dy)
vol = wet3 * AREA[:, :, None] * DZN[None, None, :]
WM = jnp.asarray(wet3)
DZ = jnp.asarray(DZN).reshape(1, 1, -1)
LSPAN = jnp.sum(DZ)


def I(f):
    return float((np.asarray(f, float) * vol).sum() * ZJ * DT)


def H(Tf):
    return float((np.asarray(Tf, float) * vol).sum() * ZJ)


def shear(a, pp, m):
    """Remove the dz_node-weighted depth mean, then re-mask."""
    num = jnp.sum(a * m * DZ, axis=-1, keepdims=True)
    return (a - num / LSPAN) * m


# ---- originals ----
ORIG_VTI = JS._vertical_transport_iface
ORIG_ADV = JS._advection_scalar


def vti_shear(u, v, pp):
    return ORIG_VTI(shear(u, pp, pp.wet_mask_z), shear(v, pp, pp.wet_mask_z), pp)


def adv_shear(T, u, v, Fz, pp):
    return ORIG_ADV(T, shear(u, pp, pp.wet_mask_z), shear(v, pp, pp.wet_mask_z), Fz, pp)


def run(nstep=100, use_shear=True):
    if use_shear:
        JS._vertical_transport_iface = vti_shear
        JS._advection_scalar = adv_shear
    else:
        JS._vertical_transport_iface = ORIG_VTI
        JS._advection_scalar = ORIG_ADV
    s = JS.JaxStateG(u=jnp.asarray(u0), v=jnp.asarray(v0), T=jnp.asarray(T0),
                     S=jnp.asarray(S0), eta=jnp.asarray(e0))
    h0 = H(s.T)
    hs = [h0]
    for k in range(nstep):
        s = JS._step_impl(s, pp_ := p)
        hs.append(H(np.asarray(s.T)))
        if not np.isfinite(hs[-1]):
            return hs, k + 1
    return hs, nstep


print("=" * 84)
print("RK2 stage agreement: base vs shear")
print("=" * 84)
s = JS.JaxStateG(u=jnp.asarray(u0), v=jnp.asarray(v0), T=jnp.asarray(T0),
                 S=jnp.asarray(S0), eta=jnp.asarray(e0))

for name, on in (("base", False), ("shear", True)):
    if on:
        JS._vertical_transport_iface = vti_shear
        JS._advection_scalar = adv_shear
    else:
        JS._vertical_transport_iface = ORIG_VTI
        JS._advection_scalar = ORIG_ADV
    dT1, _ = JS._compute_tracer_residual(s, p)
    Tp = s.T + dT1 * DT
    st = JS.JaxStateG(s.u, s.v, Tp, s.T, s.eta)
    du1, dv1 = JS._compute_momentum_residual(st, p)
    bm = p.bottom_mask * p.wet_mask_z
    up = s.u + (du1 + p.r_bot * s.u * bm) * DT
    vp = s.v + (dv1 + p.r_bot * s.v * bm) * DT
    dT2p, _ = JS._compute_tracer_residual(
        JS.JaxStateG(up, vp, Tp, s.T, s.eta), p)
    dT2o, _ = JS._compute_tracer_residual(st, p)
    i1, i2p, i2o = I(dT1), I(dT2p), I(dT2o)
    print("  %-6s I(dT1) %+13.8f  I(dT2@old) %+13.8f  I(dT2@pred) %+13.8f"
          % (name, i1, i2o, i2p))
    print("         stage disagreement |I2pred-I2old| %13.8f   (%.1fx)"
          % (abs(i2p - i2o), abs(i2p - i2o) / (abs(i1) + 1e-30)))

JS._vertical_transport_iface = ORIG_VTI
JS._advection_scalar = ORIG_ADV

print()
print("=" * 84)
print("100-step H budget, advection-only (every closure off, no forcing)")
print("=" * 84)
for name, on in (("base", False), ("shear", True)):
    hs, n = run(100, on)
    dH = hs[-1] - hs[0]
    print("  %-6s n=%3d  dH %+13.6f ZJ  rate %+11.3f ZJ/yr  drift/step %+12.3e"
          % (name, n, dH, dH / (n * DT / 3.1536e7), dH / n))
    ei = min(range(len(hs)), key=lambda k: abs(hs[k] - hs[0]))
    print("         min |H-H0| at step %d (%.3e ZJ); final/initial %.6f"
          % (ei, hs[ei] - hs[0], hs[-1] / hs[0]))

JS._vertical_transport_iface = ORIG_VTI
JS._advection_scalar = ORIG_ADV
