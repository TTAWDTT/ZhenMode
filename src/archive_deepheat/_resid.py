"""With all closures OFF and u=v=0, which residual term is nonzero?

_compute_tracer_tendency at u=v=0, kappa_*=0, lambda_bulk=0, Q_heat=0:
  adv       -> should be exactly 0 (u=v=0, but Fz comes from _vertical_transport_iface
               which may NOT be zero at u=v=0! check)
  diff_*    -> 0 (kappas 0)
  conv      -> 0 (kappa_conv=0)
  heat/bulk -> 0 (Q_heat=0, lambda_bulk=0)
  gm/redi   -> 0 (kappas 0)
So the ONLY candidate is adv_T via Fz = _vertical_transport_iface(u,v,p) at u=v=0.
If _vertical_transport_iface returns nonzero Fz at zero velocity, the vertical
advection term -Fz*dT/dz moves heat with a zero velocity field -- which is
exactly a non-conservative discretization.
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
print("RESIDUAL AT u=v=0, ALL CLOSURES OFF  (must be identically 0)")
print("=" * 100)
dTdt = JS._compute_tracer_tendency(s0, p)[0]
dTdt = np.asarray(dTdt, float)
print("  full dTdt:  max|.|=%.4e K/s   dE/dt = %+ .5e ZJ/s (integrated: %+ .3f ZJ/step)"
      % (np.abs(dTdt).max(), dE(dTdt), dE(dTdt) * DT))
resT = np.asarray(JS._compute_tracer_residual(s0, p)[0], float)
print("  residual:   max|.|=%.4e K/s   dE = %+ .3f ZJ/step" % (np.abs(resT).max(), dE(resT) * DT))

print()
print("  terms_fn at u=v=0 (each, K/s and ZJ/step):")
trm = np.asarray(JS._tracer_terms(s0, p), float)
for i, nm in enumerate(["adv", "diff_h", "diff_v", "conv", "gm", "redi"]):
    print("    %-8s max|.|=%.4e   dE=%+ .4f ZJ/step" % (nm, np.abs(trm[i]).max(), dE(trm[i]) * DT))

print()
print("=" * 100)
print("Fz = _vertical_transport_iface(u,v,p) at u=v=0")
print("=" * 100)
Fz = np.asarray(JS._vertical_transport_iface(s0.u, s0.v, p), float)
print("  Fz shape=%s   max|Fz| = %.6e" % (Fz.shape, np.abs(Fz).max()))
print("  nonzero Fz fractions: >1e-30: %.6f   >1e-12: %.6f   >1e-6: %.6f"
      % ((np.abs(Fz) > 1e-30).mean(), (np.abs(Fz) > 1e-12).mean(), (np.abs(Fz) > 1e-6).mean()))
if np.abs(Fz).max() > 0:
    idx = np.unravel_index(np.argmax(np.abs(Fz)), Fz.shape)
    print("  argmax at %s (i,j,k_iface) = %.6e" % (idx, Fz[idx]))
    print("  Fz stats on wet: min %+ .4e max %+ .4e mean %+ .4e"
          % (Fz.min(), Fz.max(), Fz.mean()))

print()
print("  Is the nonzero Fz a MASK/ghost artifact? sample the k=0 iface by row:")
for j in [0, 1, 2, 3, 60, 117, 118, 119]:
    print("    j=%3d  Fz[:,j,0]: min %+ .4e max %+ .4e   wet-cells %d"
          % (j, Fz[:, j, 0].min(), Fz[:, j, 0].max(), int((wet3[:, j, 0] > 0).sum())))

print()
print("=" * 100)
print("adv_T = _advection_scalar(T, 0, 0, Fz, p)  decomposition")
print("=" * 100)
advT = np.asarray(JS._advection_scalar(s0.T, s0.u, s0.v, jnp.asarray(Fz), p), float)
print("  adv_T: max|.|=%.4e K/s  dE=%+ .4f ZJ/step" % (np.abs(advT).max(), dE(advT) * DT))
print("  dE of adv over the column, per level:")
for k in range(nz):
    print("    k=%2d  %+ .6e ZJ/step" % (k, dE(advT[:, :, k:k + 1]) * DT))

# check _vertical_transport_iface source for a hidden constant/forcing term
import inspect
src = inspect.getsource(JS._vertical_transport_iface)
print()
print("=" * 100)
print("SOURCE of _vertical_transport_iface")
print("=" * 100)
print(src)
