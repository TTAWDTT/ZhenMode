"""Clean conservation discriminator: uniform field + zero velocity.

An operator that is geometrically sound must return EXACTLY zero tendency on
a uniform tracer field, independent of which volume norm you integrate with.
This removes the dz_node-vs-dz_iface ambiguity in my earlier "total".

Cases:
  1. T,S uniform + u=v=0 + no forcing + no bulk  -> dT must be 0 everywhere
  2. same, but kappa_v only (all else off)       -> isolates _d2_dz2
  3. same, but kappa_h only                      -> isolates _laplacian_h
  4. same, but kappa_bi only                     -> isolates _biharmonic_h
  5. T linear in z, u=v=0, kappa_v only          -> boundary-form test:
     a true 2nd derivative of a linear profile is 0; the mirror-ghost
     boundary form 2*(C1-C0)/h0^2 is NOT (it sees 2a/h0).
  6. real checkpoint T, u=v=0, bulk=0, all traps off -> field-dependent
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
PROD = dict(nu_h=0.0, nu_bi=0.0, kappa_bi=0.0, kappa_gm=0.0, kappa_redi=0.0,
            kappa_v=0.0, kappa_conv=0.0, gm_slope_max=0.005, kappa_h=0.0,
            T_ref=20.0, S_ref=35.0)

_, _, _, p, _ = JS.make_solver_global(
    g, _dc(PhysicsConfig(), **PROD), DT, T_atm=jnp.zeros((g.nx, g.ny)),
    lambda_bulk=0.0, mode_split=True, dt_bt=300.0,
    polar_cap_rows=2, polar_cap_taper=3, return_params=True)

wet3 = np.asarray(p.wet_mask_z, float)
DZN = np.asarray(p.dz_node).ravel()
AREA = np.asarray(g.dx_2d, float) * float(g.dy)
vol = wet3 * AREA[:, :, None] * DZN[None, None, :]
nx, ny, nz = int(p.nx), int(p.ny), int(p.nz)
print("nx=%d ny=%d nz=%d  wet cols=%d  vol=%.5e"
      % (nx, ny, nz, int((wet3[:, :, 0] > 0).sum()), vol.sum()))


def mk(Tf, U=0.0):
    if np.isscalar(Tf):
        T = jnp.full((nx, ny, nz), float(Tf))
    else:
        T = jnp.broadcast_to(jnp.asarray(Tf, float).reshape(1, 1, -1), (nx, ny, nz))
    return JS.JaxStateG(u=jnp.full((nx, ny, nz), U), v=jnp.zeros((nx, ny, nz)),
                        T=T, S=jnp.full((nx, ny, nz), 35.0), eta=jnp.zeros((nx, ny)))


def step(s, over):
    pp = p._replace(**{k: v for k, v in over.items()} & p._fields) if False else p._replace(
        **{k: v for k, v in over.items()})
    return JS._step_impl(s, pp)


def check(tag, s, over):
    s2 = step(s, over)
    dT = np.asarray(s2.T, float) - np.asarray(s.T, float)
    dT = dT * wet3
    dE = float((dT * vol).sum()) * RHO_CP
    m = wet3 > 0
    print("  %-34s max|dT|=%.4e K  rms=%.4e  dE=%+ .5e ZJ/step (%+ .3f ZJ/yr)"
          % (tag, np.abs(dT).max(), np.sqrt((dT[m] ** 2).mean()), dE, dE / (DT / SEC)))
    return dT


ALL_OFF = dict(kappa_h=0.0, kappa_v=0.0, kappa_bi=0.0, kappa_conv=0.0,
               kappa_gm=0.0, kappa_redi=0.0, nu_h=0.0, nu_bi=0.0)

print()
print("=" * 100)
print("CASE 1-4: UNIFORM field T=20, u=v=0, no forcing, no bulk")
print("=" * 100)
s_uni = mk(20.0)
check("all operators OFF (null test)", s_uni, ALL_OFF)
check("kappa_v=1e-5 only", s_uni, dict(ALL_OFF, kappa_v=1e-5))
check("kappa_h=100 only", s_uni, dict(ALL_OFF, kappa_h=100.0))
check("kappa_bi=2e14 only", s_uni, dict(ALL_OFF, kappa_bi=2e14))
check("kappa_conv=0.05 only", s_uni, dict(ALL_OFF, kappa_conv=0.05))
check("kappa_gm=1000 only", s_uni, dict(ALL_OFF, kappa_gm=1000.0))
check("kappa_redi=1000 only", s_uni, dict(ALL_OFF, kappa_redi=1000.0))

print()
print("=" * 100)
print("CASE 5: LINEAR-IN-Z profile, u=v=0  (true d2/dz2 == 0)")
print("=" * 100)
# node depths from the grid definition used by make_fd_params
z = np.array([0, -5, -15, -30, -50, -75, -100, -150, -200, -300, -500,
              -1000, -2000, -4000], float)
# co-locate nodes at the midpoint of the surrounding interfaces for a clean
# linear-in-z test; use the actual node depths z_node = -(cumsum back)
z_node = np.zeros(nz)
z_node[0] = -DZN[0] / 2.0
for k in range(1, nz):
    z_node[k] = z_node[k - 1] - (DZN[k - 1] + DZN[k]) / 2.0
print("  node depths (m):", np.array2string(z_node, precision=1))
A = 0.02                                   # K/m
Tlin = 20.0 + A * z_node                   # warmer at depth
print("  T profile: %.3f (k=0) .. %.3f (k=13)" % (Tlin[0], Tlin[-1]))
s_lin = mk(Tlin)
check("linear T, kappa_v=1e-5 only", s_lin, dict(ALL_OFF, kappa_v=1e-5))
check("linear T, all OFF", s_lin, ALL_OFF)

# raw operator check: what should _d2_dz2 give for a linear profile?
d2lin = np.asarray(JS._d2_dz2(jnp.asarray(Tlin).reshape(1, 1, nz), p)).ravel()
print()
print("  raw _d2_dz2 of the linear profile (should be ~0 everywhere):")
for k in range(nz):
    print("    k=%2d z=%8.1f   d2Tdz2 = %+ .6e" % (k, z_node[k], d2lin[k]))
print("  max |d2Tdz2| interior (k=1..12) = %.3e" % np.abs(d2lin[1:-1]).max())
print("  boundary k=0 = %+ .6e   k=13 = %+ .6e" % (d2lin[0], d2lin[-1]))
print("  analytic spurious boundary curvature 2a/h0 = %+ .6e (h0=%.1f)"
      % (2 * A / DZN[0], DZN[0]))

print()
print("=" * 100)
print("CASE 6: REAL checkpoint T, u=v=0, bulk=0")
print("=" * 100)
d = np.load("results/ckpt_tenyr_ms_gm.npz")
Tck = np.asarray(d["T"], np.float64)
s_ck = JS.JaxStateG(u=jnp.zeros((nx, ny, nz)), v=jnp.zeros((nx, ny, nz)),
                    T=jnp.asarray(Tck), S=jnp.asarray(d["S"]), eta=jnp.zeros((nx, ny)))
check("real T, ALL OFF", s_ck, ALL_OFF)
check("real T, kappa_v=1e-5 only", s_ck, dict(ALL_OFF, kappa_v=1e-5))
check("real T, kappa_h=100 only", s_ck, dict(ALL_OFF, kappa_h=100.0))
check("real T, kappa_bi=2e14 only", s_ck, dict(ALL_OFF, kappa_bi=2e14))
check("real T, kappa_conv=0.05 only", s_ck, dict(ALL_OFF, kappa_conv=0.05))
check("real T, gm=1000 only", s_ck, dict(ALL_OFF, kappa_gm=1000.0))
check("real T, redi=1000 only", s_ck, dict(ALL_OFF, kappa_redi=1000.0))
