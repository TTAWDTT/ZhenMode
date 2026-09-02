"""Diagnostic for the global FD solver baroclinic blow-up (G2 blocker).

GOAL: isolate WHY the real-global-grid run diverges (T 30 -> 1737 over 40 steps)
and find a dissipation setting that arrests it.

HYPOTHESIS (verified against source first):
  - jax_solver_global.py defines _biharmonic_h but NEVER CALLS it. nu_bi/kappa_bi
    are passed into params but unused. The linear half-step and residual use
    ONLY the Laplacian. So "biharmonic ON" in prior memory notes was wrong —
    the run diverged under PURE Laplacian, nu_h=100 m^2/s.
  - At 1 deg (dx~111km), the Laplacian grid-scale damping timescale
    dx^2/nu_h ~ 14 days, but the baroclinic PGF from steep WOA gradients +
    land boundaries grows on hour-day timescales. 100 m^2/s is 1-2 orders
    too low for 1 deg.

This script:
  1. Builds the real global grid (ETOPO + WOA).
  2. Runs N steps under each candidate dissipation setting.
  3. Reports max|T|, max|u|, KE growth per step -> identifies which setting
     arrests the blow-up.

Usage:  cd src && export PYTHONPATH=/c/Users/zhen.luo/Python/Python314/site-packages \
        && /c/Python314/python.exe diag_global_stab.py
"""
import os
os.environ.setdefault('JAX_PLATFORMS', 'cpu')
import sys
import numpy as np
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
from dataclasses import replace

from config import DEFAULT_CONFIG, PhysicsConfig, GlobalGridConfig
from grid import make_global_grid
from woa_data import get_initial_fields
from jax_solver_global import make_solver_global

# ── bathymetry path ──
BATHY = r"C:\Users\zhen.luo\Desktop\ETOPO_2022_v1_r3600x1800_surface.nc"


def build_grid_and_init():
    gcfg = GlobalGridConfig()
    grid = make_global_grid(gcfg, BATHY)
    T_init, S_init = get_initial_fields(grid)
    return grid, T_init, S_init


def run_config(grid, T_init, S_init, phys, dt, n_steps, tag, T_atm=None,
               lambda_bulk=0.0, locate=False):
    """Run n_steps under a PhysicsConfig; print per-step diagnostics."""
    forcing = None  # no wind for the stability probe (isolate thermodynamics)
    step, init_state, _ = make_solver_global(
        grid, phys, dt, forcing=forcing, eos_type='linear',
        T_atm=T_atm, lambda_bulk=lambda_bulk)
    state = init_state(T_init, S_init)

    print(f"\n=== {tag} ===")
    print(f"  nu_h={phys.nu_h}  nu_bi={phys.nu_bi}  kappa_h={phys.kappa_h}  "
          f"kappa_bi={phys.kappa_bi}  dt={dt}")
    max_T0 = float(jnp.max(jnp.abs(state.T)))
    print(f"  step  0: max_T={max_T0:.3f}")

    diverged = False
    for k in range(1, n_steps + 1):
        state = step(state)
        max_T = float(jnp.max(jnp.abs(state.T)))
        max_u = float(jnp.max(jnp.abs(state.u)))
        nan = int(jnp.sum(jnp.isnan(state.T)))
        if k % 5 == 0 or k <= 5 or not np.isfinite(max_T):
            loc = ""
            if locate and np.isfinite(max_u) and max_u > 0.5:
                au = np.abs(np.array(state.u))
                iy, iz = np.unravel_index(np.argmax(au), au.shape)[1:]
                # lat of that row
                lat = grid.lat[iy]
                loc = f"  @lat={lat:.1f} iz={iz}"
            print(f"  step {k:3d}: max|u|={max_u:.3e} max_T={max_T:.3f} "
                  f"nan={nan}{loc}")
        if not np.isfinite(max_T) or max_T > 1e4:
            print(f"  DIVERGING at step {k}")
            diverged = True
            break
    return diverged, max_T


def main():
    print("Building real global grid (ETOPO + WOA)...")
    grid, T_init, S_init = build_grid_and_init()
    print(f"  grid {grid.nx}x{grid.ny}x{grid.nz}, ocean {float(grid.wet_mask.mean()):.1%}")
    print(f"  T_init range [{T_init.min():.2f}, {T_init.max():.2f}]")

    N = 8
    DT = 60.0

    # Baseline WITH localization — where does the blow-up nucleate?
    run_config(grid, T_init, S_init, DEFAULT_CONFIG.physics, DT, N,
               "baseline (locate nucleation)", locate=True)

    print("\n=== Done ===")


if __name__ == "__main__":
    main()
