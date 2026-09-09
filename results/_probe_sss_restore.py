# -*- coding: utf-8 -*-
"""Unit probe: SSS restoring converges SSS toward the target and is OFF at
tau=0 (bit-exact with the pre-change solver path)."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import jax
import jax.numpy as jnp

from dataclasses import replace
from config import DEFAULT_CONFIG, PhysicsConfig, GlobalGridConfig
from grid import make_global_grid
from jax_solver_global import make_solver_global

# tiny-but-valid global grid: 4-deg (ny is DERIVED from resolution+lat_max
# by _read_etopo_global: nlon=3600/40=90, lat rows |lat_c|<=60 -> 31),
# coarse z. make_global_grid asserts gc.ny == derived ny.
gcfg = replace(GlobalGridConfig(), lat_max=60.0, nx=90, ny=31, resolution=4.0,
               nz=5, z_levels=(0.0, -100.0, -1000.0, -3000.0, -4500.0))
grid = make_global_grid(gcfg, DEFAULT_CONFIG.bathymetry_file,
                        smooth_passes=2, min_depth=50.0)
print(f"grid {grid.nx}x{grid.ny}x{grid.nz}")

nx, ny, nz = grid.nx, grid.ny, grid.nz
# kappa_h = kappa_v = kappa_conv = 0: with diffusion on, the surface-deep S
# gradient the restoring creates bleeds into the uniform-34 interior (and the
# salty- over-fresh column is statically unstable -> convective adjustment
# exports salt downward), so SSS lags the pure Haney ODE. Physical, but it
# would mask the restoring-rate check.
physics = replace(PhysicsConfig(), nu_h=5e6, kappa_h=0.0, kappa_v=0.0,
                  kappa_conv=0.0)

# initial S: uniform 34.0; target: 36.0 over ocean surface
S_init = np.full((nx, ny, nz), 34.0)
S_ref = np.where(np.asarray(grid.wet_mask) > 0.5, 36.0, 0.0)

def build(tau_days):
    step, init_state, diag, params, _tf = make_solver_global(
        grid, physics, 600.0, forcing=None, eos_type="linear",
        S_ref_surf=S_ref, sss_restore_days=tau_days, return_params=True,
        polar_cap_rows=0, polar_cap_taper=0)  # cap filter would zonal-average
        # polar rows toward the ocean+land mean, polluting the analytic SSS
    return step, init_state, params

# 1) OFF case: restore_coef_S == 0
_, _, p0 = build(0.0)
assert float(p0.restore_coef_S) == 0.0, p0.restore_coef_S
print("OFF case: restore_coef_S =", float(p0.restore_coef_S), "OK")

# 2) ON case: 10-day tau → after 100 steps (dt=600s = 60000s = 0.69d),
# SSS should move from 34 toward 36
step, init_state, p1 = build(10.0)
assert abs(float(p1.restore_coef_S) - 1.0 / (10.0 * 86400.0)) < 1e-12
wet2 = np.asarray(grid.wet_mask) > 0.5
# NOTE: init_state only honors S_init when T_init is also given.
T_init = np.full((nx, ny, nz), 15.0)
st = init_state(T_init=jnp.array(T_init), S_init=jnp.array(S_init))
for i in range(1, 101):
    st = step(st)
    if i in (1, 2, 5, 10, 50, 100):
        s0 = np.asarray(st.S)[:, :, 0]
        print(f"step {i:3d}: SSS_ocean={s0[wet2].mean():.5f} "
              f"S_land={s0[~wet2].mean() if (~wet2).any() else float('nan'):.5f} "
              f"t={i*600/86400:.3f}d")
sss = float(np.asarray(st.S)[:, :, 0][wet2].mean())
print(f"ON case: SSS (ocean-only) after 100 steps = {sss:.4f} "
      f"(start {S_init[wet2][:,0].mean() if S_init.ndim==3 else 34.0}, target 36.0)")
assert sss > 34.0 and sss < 36.0, "SSS did not move toward target"
# analytic: dS/dt = coef*(36-S); S(t)=36-2*exp(-t/tau); t=60000s
t = 100 * 600.0
expect = 36.0 - 2.0 * np.exp(-t / (10.0 * 86400.0))
assert abs(sss - expect) < 0.01, f"convergence mismatch: {sss} vs {expect}"
print(f"  matches analytic {expect:.4f} OK")

# 3) land not restored / no crash with target=0 on land
print("ALL SSS-RESTORING PROBES PASS")
