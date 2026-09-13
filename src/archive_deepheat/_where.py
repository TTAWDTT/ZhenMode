"""Locate the structural -268 ZJ/yr: WHERE and from WHICH stage.

All closures off, u=v=0, no bulk, no forcing. The only T-touching code left in
_step_impl is:
  L half-step  (kappas=0 -> identity, but mask-and-hold + polar-cap run in the
                mode_split path? check)
  N step       (RK2: residual should be identically 0 at u=v=0)
  bt subcycle  (touches u,v,eta -- not T)
  polar cap    (zonal mean blend on T!)
  mask/hold    (holds pre-step T outside wet_mask_z)

Report dE per stage, and the spatial support of the final dT (which rows and
which levels) so the culprit is obvious.
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
s0 = JS.JaxStateG(u=jnp.zeros((nx, ny, nz)), v=jnp.zeros((nx, ny, nz)),
                  T=jnp.asarray(np.asarray(d["T"], np.float64)),
                  S=jnp.asarray(np.asarray(d["S"], np.float64)), eta=jnp.zeros((nx, ny)))
T0 = np.asarray(s0.T, float)


def dE(dTf):
    return float((dTf * wet3 * AREA[:, :, None] * DZN[None, None, :]).sum()) * RHO_CP


def show(tag, Tnow):
    dT = np.asarray(Tnow, float) - T0
    e = dE(dT)
    print("  %-26s dE=%+ .6e ZJ/step  (%+10.3f ZJ/yr)  max|dT|=%.3e"
          % (tag, e, e / (DT / SEC), np.abs(dT).max()))
    return dT


print("=" * 100)
print("STAGE DECOMPOSITION, all closures OFF, u=v=0")
print("=" * 100)
dt_half = DT / 2.0
land_T = s0.T
st = JS._linear_half_step(s0, p, dt_half)
dT_L1 = show("L half-step #1", st.T)
st = JS._explicit_full_step(st, p, p.dt)
dT_sub = np.asarray(st.T, float) - T0
print("    -> after N step: dE=%+ .6e ZJ/step (increment %+ .6e)"
      % (dE(dT_sub), dE(np.asarray(st.T, float) - (T0 + dT_L1))))
st = JS._linear_half_step(st, p, dt_half)
dT_L2 = show("L half-step #2 (L/N/L done)", st.T)
T_before_cap = np.asarray(st.T, float).copy()
# barotropic subcycle (no T touch)
F_rho_x, F_rho_y = JS._compute_bt_rho_pgf(st, p)
ubt0, vbt0 = JS._barotropic_velocity(st.u, st.v, p)
eta, ubt, vbt = st.eta, ubt0, vbt0
for _ in range(int(p.n_subcyc)):
    eta, ubt, vbt = JS._free_surface_step_fd(eta, ubt, vbt, p, F_rho_x, F_rho_y, dt_half=p.dt_bt)
st_bt = JS.JaxStateG(st.u + (ubt - ubt0)[:, :, None], st.v + (vbt - vbt0)[:, :, None],
                     st.T, st.S, eta)
show("bt subcycle (T untouched)", st_bt.T)

T_cap = JS._polar_cap_3d(st_bt.T, p)
dT_cap = show("polar cap", T_cap)
wmask = p.wet_mask_z
T_mask = T_cap * wmask + land_T * (1.0 - wmask)
dT_mask = show("mask/hold (FINAL)", T_mask)

print()
print("  --- where is the polar-cap change? ---")
cap_only = np.asarray(T_cap, float) - np.asarray(st_bt.T, float)
m = np.abs(cap_only) > 1e-12
print("    nonzero cells: %d   max|d| = %.4e K" % (m.sum(), np.abs(cap_only).max()))
if m.sum():
    rows = np.where(m.any(axis=(0, 2)))[0]
    print("    rows with change: %s" % (rows[:40],))
    print("    rows min=%d max=%d  (ny=%d, polar_cap_rows=%d taper=%d)"
          % (rows.min(), rows.max(), ny, int(p.polar_cap_rows), int(p.polar_cap_taper)))
    lv = np.where(m.any(axis=(0, 1)))[0]
    print("    levels with change: %s" % (lv,))
    print("    dE of cap alone = %+ .6e ZJ/step (%+10.3f ZJ/yr)"
          % (dE(cap_only), dE(cap_only) / (DT / SEC)))

print()
print("  --- where is the FINAL change? ---")
fin = np.asarray(T_mask, float) - T0
mm = np.abs(fin) > 1e-12
print("    nonzero cells: %d  max|d| = %.4e K" % (mm.sum(), np.abs(fin).max()))
r = np.where(mm.any(axis=(0, 2)))[0]
print("    rows min=%d max=%d  count=%d" % (r.min(), r.max(), len(r)))
print("    dE by row band: cap rows [0:5]=%+ .5e  [115:]=%+ .5e  middle=%+ .5e"
      % (dE(fin[:, :5]), dE(fin[:, -5:]), dE(fin[:, 5:-5])))

print()
print("  --- is it the wet/land mask boundary at the pole? ---")
# How many wet cells per row in the cap band, and does the row span all 360 lon?
for j in list(range(0, 8)) + list(range(ny - 8, ny)):
    wj = (wet3[:, j, 0] > 0).sum()
    print("    j=%3d  wet cells at k=0: %3d / %d   lat-ish row" % (j, wj, nx))
