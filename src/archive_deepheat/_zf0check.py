"""Safety check before shipping the zf0 fix.

zf0 (Fz[...,0] := 0) gives exact heat AND mass conservation over 200 steps.
But it changes the pointwise closure at the SURFACE CELL: in base, adv(1)[0]
== 0 (the uniform-tracer stays uniform property); with Fz_top=0,
adv(1)[0] = -Fz[0]/dz[0] != 0. The failure mode this project already lived
through (the old node-w scheme) was a POINTWISE non-closure acting as a
T-proportional pump: resid*T with resid up to 1.2e-5/s grew max|T| 29.6 ->
52.6 in 30 steps at the North Brazil Current node.

So measure, in the ACTUAL time integration:
  - max|T|, max|S|, max|u| trajectories (boundedness)
  - max|adv(1)| pointwise and its location
  - whether any single node grows without bound
for base vs zf0, and confirm zf0's non-closure is confined to the surface
layer (k=0) and is the physical rigid-lid free-surface term.
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
ORIG_VTI = JS._vertical_transport_iface


def H(Tf):
    return float((np.asarray(Tf, float) * vol).sum() * ZJ)


def install(variant):
    if variant == "base":
        JS._vertical_transport_iface = ORIG_VTI
        return

    def vti(u, v, pp):
        return ORIG_VTI(u, v, pp).at[:, :, 0].set(0.0)

    JS._vertical_transport_iface = vti


print("=" * 88)
print("Pointwise adv(1) structure   [per second]")
print("=" * 88)
s0 = JS.JaxStateG(u=jnp.asarray(u0), v=jnp.asarray(v0), T=jnp.asarray(T0),
                  S=jnp.asarray(S0), eta=jnp.asarray(e0))
for vname in ("base", "zf0"):
    install(vname)
    Fz = JS._vertical_transport_iface(s0.u, s0.v, p)
    a1 = np.asarray(JS._advection_scalar(jnp.ones_like(s0.T), s0.u, s0.v, Fz, p), float)
    a1w = np.abs(a1 * wet3)
    k0 = np.abs(a1[:, :, 0] * wet3[:, :, 0])
    kd = np.abs(a1[:, :, 1:] * wet3[:, :, 1:])
    print("  %-5s max|adv(1)| all %.3e | surface k=0 %.3e | k>=1 %.3e"
          % (vname, a1w.max(), k0.max(), kd.max()))
    print("        integral over volume %.4e   1/step drift if T=15C: %.3e K"
          % (float((a1 * vol).sum()),
             float((a1 * vol).sum()) * DT * 15.0 / vol.sum()))
    ij = np.unravel_index(np.argmax(k0), k0.shape)
    print("        worst surface node (i,j)=(%d,%d) lat %.2f adv(1) %+.3e /s"
          % (ij[0], ij[1], float(g.lat[ij[1]]), a1[ij[0], ij[1], 0]))

install("base")

print()
print("=" * 88)
print("60-step boundedness, advection-only (all closures off, no forcing)")
print("=" * 88)
for vname in ("base", "zf0"):
    install(vname)
    s = JS.JaxStateG(u=jnp.asarray(u0), v=jnp.asarray(v0), T=jnp.asarray(T0),
                     S=jnp.asarray(S0), eta=jnp.asarray(e0))
    h0 = H(s.T)
    rows = []
    for k in range(60):
        s = JS._step_impl(s, p)
        Tn = np.asarray(s.T, float)
        Sn = np.asarray(s.S, float)
        un = np.asarray(s.u, float)
        en = np.asarray(s.eta, float)
        if not np.isfinite(Tn).all():
            print("  %s DIVERGED at step %d" % (vname, k))
            break
        if (k + 1) % 15 == 0:
            rows.append((k + 1, H(s.T) - h0, np.abs(Tn * wet3).max(),
                         np.abs(Sn * wet3).max(), np.abs(un * wet3).max(),
                         np.abs(en).max()))
    print("  --- %s ---" % vname)
    print("     step       dH(ZJ)     max|T|    max|S|    max|u|   max|eta|")
    for r in rows:
        print("     %4d  %+12.6f  %8.3f  %8.3f  %8.4f  %8.4f" % r)

install("base")
