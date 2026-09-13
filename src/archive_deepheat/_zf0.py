"""RK2 stage disagreement is the top face only.  Test three repairs.

_shearRK.out: I(dT1) base +0.00763743 -> shear -0.00009870 (78x smaller), but
I(dT2@pred) is +0.29 base / +0.10 shear -- both huge.  The second-stage blowup
is NOT the mean leak; it is the Fz[0]*T[0] surface term responding to T_pred.
And shear's breakup makes its stage disagreement WORSE.

Key structural fact: _vertical_transport_iface is called with state.u/state.v
ONLY (jax_solver_global.py:887,1363,1409) -- never with a predicted velocity.
So Fz[0] is K-independent.  The only velocity-dependent part of the RK2 stage
disagreement is the horizontal advective flux divergence, which telescopes:
   I[-div(u'T')] = sum_cells AREA * (Fx_west - Fx_east) * T
only through the TOP face of the horizontal flux... which does NOT exist.
So hypothesis: patch Fz[0] itself (variant 'zf0'), leaving u,v untouched --
Fz'[0] := 0 exactly.

  1. base : unmodified
  2. shear: u' = u - depth-mean (mass exact, Fz[0] ~1e-5 residual)
  3. zf0  : Fz'[0] := 0 exactly (heat exact by construction)

Measure vs the ACTUAL T_pred of _explicit_full_step, not the T_pred from dT1
(the earlier run made that mistake and inflated base to +0.29).
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
DZ = jnp.asarray(DZN).reshape(1, 1, -1)
LSPAN = jnp.sum(DZ)


def I(f):
    return float((np.asarray(f, float) * vol).sum() * ZJ * DT)


def H(Tf):
    return float((np.asarray(Tf, float) * vol).sum() * ZJ)


ORIG_VTI = JS._vertical_transport_iface
ORIG_ADV = JS._advection_scalar


def _shear(a, m):
    return (a - jnp.sum(a * m * DZ, axis=-1, keepdims=True) / LSPAN) * m


def make(variant):
    if variant == "base":
        return ORIG_VTI, ORIG_ADV

    def vti(u, v, pp):
        if variant == "shear":
            u, v = _shear(u, pp.wet_mask_z), _shear(v, pp.wet_mask_z)
        Fz = ORIG_VTI(u, v, pp)
        if variant == "zf0":
            Fz = Fz.at[:, :, 0].set(0.0)
        return Fz

    def adv(T, u, v, Fz, pp):
        if variant == "shear":
            u, v = _shear(u, pp.wet_mask_z), _shear(v, pp.wet_mask_z)
        return ORIG_ADV(T, u, v, Fz, pp)

    return vti, adv


def install(variant):
    a, b = make(variant)
    JS._vertical_transport_iface = a
    JS._advection_scalar = b


VARIANTS = ("base", "shear", "zf0")

print("=" * 90)
print("RK2 stage agreement with the REAL T_pred, and Fz[0] statistics")
print("=" * 90)
s = JS.JaxStateG(u=jnp.asarray(u0), v=jnp.asarray(v0), T=jnp.asarray(T0),
                 S=jnp.asarray(S0), eta=jnp.asarray(e0))
for vname in VARIANTS:
    install(vname)
    Fz = JS._vertical_transport_iface(s.u, s.v, p)
    f0 = np.asarray(Fz[:, :, 0], float)
    A2 = AREA * wet3[:, :, 0]
    dT1, _ = JS._compute_tracer_residual(s, p)
    sN = JS._explicit_full_step(s, p, DT)
    # real second stage: reproduce dT2 at the real predicted state
    Tp = s.T + dT1 * DT
    st = JS.JaxStateG(s.u, s.v, Tp, s.S, s.eta)
    du1, dv1 = JS._compute_momentum_residual(st, p)
    bm = p.bottom_mask * p.wet_mask_z
    up = s.u + (du1 + p.r_bot * s.u * bm) * DT
    vp = s.v + (dv1 + p.r_bot * s.v * bm) * DT
    dT2p, _ = JS._compute_tracer_residual(JS.JaxStateG(up, vp, Tp, s.S, s.eta), p)
    dT2o, _ = JS._compute_tracer_residual(st, p)
    print("  %-6s Fz[0] rms %.4e  sum(area*Fz0) %+.3e  I(adv(1)) %+.3e"
          % (vname, np.sqrt((f0 ** 2).mean()), float((A2 * f0).sum()),
             I(JS._advection_scalar(jnp.ones_like(s.T), s.u, s.v, Fz, p))))
    print("         I(dT1) %+13.8f   I(dT2@old) %+13.8f   I(dT2@realpred) %+13.8f"
          % (I(dT1), I(dT2o), I(dT2p)))
    print("         stage disagreement %13.8f     real dH/step %+13.8f"
          % (abs(I(dT2p) - I(dT2o)), H(sN.T) - H(s.T)))

install("base")
print()
print("=" * 90)
print("200-step H budget, advection-only (all closures off, no forcing)")
print("=" * 90)
for vname in VARIANTS:
    install(vname)
    s = JS.JaxStateG(u=jnp.asarray(u0), v=jnp.asarray(v0), T=jnp.asarray(T0),
                     S=jnp.asarray(S0), eta=jnp.asarray(e0))
    h0 = H(s.T)
    hs = [h0]
    n = 0
    for k in range(200):
        s = JS._step_impl(s, p)
        hs.append(H(np.asarray(s.T)))
        n = k + 1
        if not np.isfinite(hs[-1]):
            break
    dH = hs[-1] - hs[0]
    mono = all(hs[i + 1] <= hs[i] + 1e-9 for i in range(len(hs) - 1))
    print("  %-6s n=%3d  dH %+13.6f ZJ  rate %+11.3f ZJ/yr  min %+12.6f  max %+12.6f"
          % (vname, n, dH, dH / (n * DT / 3.1536e7),
             min(hs) - h0, max(hs) - h0))
    print("         monotone-decreasing: %s   |dH|/step: min %.3e max %.3e"
          % (mono, min(abs(hs[i + 1] - hs[i]) for i in range(len(hs) - 1)),
             max(abs(hs[i + 1] - hs[i]) for i in range(len(hs) - 1))))

install("base")
