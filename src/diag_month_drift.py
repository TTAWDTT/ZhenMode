"""
Diagnostic: locate the month-scale temperature drift hot spot (branch a).

A 30-day forced run (verify_month_30d.log) showed max|T| drifting monotonically
24.4 -> 105.3 C while top-mean T COOLED (19.5 -> 16.9 C) and max|u| stayed
bounded (~1-2 m/s).  This pattern = LOCALIZED, sustained heating, not a
domain-wide flux imbalance (net Q over ocean == 0, verified).

This script runs a short forced integration and tracks, at each diagnostic
step:
  - the (i, j, k) location of the single hottest cell
  - max|w| (vertical velocity) -- hypothesis: spurious large w * dT/dz in the
    thin 5m top layer pumps heat into specific cells
  - max T at each vertical level (is the hot spot at the surface or deep?)
  - mean horizontal position (is heat converging to a boundary / coast / point?)
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
import numpy as np

from config import DEFAULT_CONFIG
from grid import make_grid
from jax_solver import make_solver
from woa_data import get_initial_fields
from forcing import wind_stress_gyre, heat_flux_meridional

DT = 300.0


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=float, default=8.0)
    ap.add_argument("--noheat", action="store_true",
                    help="disable heat flux (wind only) to isolate driver")
    ap.add_argument("--restore-days", type=float, default=0.0,
                    help="surface T restoring timescale in days "
                         "(Haney relaxation to initial SST); 0 = disabled")
    args = ap.parse_args()

    grid = make_grid(DEFAULT_CONFIG.grid, DEFAULT_CONFIG.bathymetry_file)
    physics = DEFAULT_CONFIG.physics
    dt = DT
    days = args.days
    n_steps = int(round(days * 86400.0 / DT))

    print("=" * 60)
    print(f"DIAG DRIFT - {'WIND ONLY' if args.noheat else 'WIND+HEAT'} ({days:.0f} days)")
    print("=" * 60)
    T_init, S_init = get_initial_fields(grid)
    print(f"T_init range=[{T_init.min():.2f}, {T_init.max():.2f}]C")

    tau_x, tau_y = wind_stress_gyre(grid, tau0=0.1)
    Q_heat = heat_flux_meridional(grid, Q0=50.0) if not args.noheat else \
        np.zeros_like(tau_x)
    forcing = (tau_x, tau_y, Q_heat)

    # Surface temperature restoring target = initial upper-level T.
    # When --restore-days > 0, this anchors the surface temperature to the
    # initial field, preventing spurious wind-driven thermal runaway.
    T_sst = T_init[:, :, 0] if args.restore_days > 0 else None

    if T_sst is not None:
        step_fn, init_state, diag_fn = make_solver(
            grid, physics, dt, forcing=forcing,
            T_sst=T_sst, tau_restore_days=args.restore_days)
    else:
        step_fn, init_state, diag_fn = make_solver(grid, physics, dt, forcing=forcing)
    state = init_state(T_init=T_init, S_init=S_init)

    t0 = time.perf_counter()
    state = step_fn(state)
    jax.block_until_ready(state.u)
    print(f"JIT+first step: {time.perf_counter()-t0:.2f}s")

    check_interval = max(50, n_steps // 40)
    for i in range(2, n_steps + 1):
        state = step_fn(state)
        if i % check_interval == 0 or i == n_steps:
            jax.block_until_ready(state.u)
            T = np.asarray(state.T)
            u = np.asarray(state.u)
            # hot cell location
            im, jm, km = np.unravel_index(np.argmax(T), T.shape)
            maxT = T[im, jm, km]
            # vertical structure at hot column
            T_col = T[im, jm, :]
            # max w via diag
            rho, pres, w = diag_fn(state)
            jax.block_until_ready(w)
            wmax = float(np.abs(np.asarray(w)).max())
            # mean top T
            topmean = float(T[:, :, 0].mean())
            print(f"  day {i*dt/86400.0:6.2f} step {i:6d}: "
                  f"maxT={maxT:7.2f} @(i={im},j={jm},k={km}) "
                  f"Tcol_top={T_col[0]:.2f} Tcol_bot={T_col[-1]:.2f} "
                  f"max|w|={wmax:.2e} topmeanT={topmean:6.2f}")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
