"""
Smoke test: run the solver with the real NCEP/NCAR R1 10m wind forcing.

Verifies `wind_reanalysis.real_wind_forcing` integrates cleanly into
`make_solver` (branch b: real wind replaces the idealized Stommel gyre
wind). Runs a short multi-day forced integration with real wind + the
meridional heat flux, and reports stability / temperature bounds.

Run:  python src/verify_real_wind.py [--days N]
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
from forcing import heat_flux_meridional
from wind_reanalysis import real_wind_forcing

DT = 300.0


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=float, default=4.0)
    ap.add_argument("--noheat", action="store_true")
    args = ap.parse_args()

    grid = make_grid(DEFAULT_CONFIG.grid, DEFAULT_CONFIG.bathymetry_file)
    physics = DEFAULT_CONFIG.physics
    dt = DT
    days = args.days
    n_steps = int(round(days * 86400.0 / DT))

    print("=" * 60)
    print(f"REAL WIND FORCED RUN - NCEP/NCAR 10m ({days:.1f} days)")
    print("=" * 60)

    T_init, S_init = get_initial_fields(grid)
    print(f"T_init range=[{T_init.min():.2f}, {T_init.max():.2f}]C")

    # Real wind stress (branch b) + optional meridional heat flux
    tau_x, tau_y = real_wind_forcing(grid=grid)
    print(f"real tau_x range=[{tau_x.min():.4f},{tau_x.max():.4f}] N/m^2")
    print(f"real tau_y range=[{tau_y.min():.4f},{tau_y.max():.4f}] N/m^2")
    Q_heat = heat_flux_meridional(grid, Q0=50.0) if not args.noheat \
        else np.zeros_like(tau_x)
    forcing = (tau_x, tau_y, Q_heat)

    step_fn, init_state, diag_fn = make_solver(grid, physics, dt, forcing=forcing)
    state = init_state(T_init=T_init, S_init=S_init)

    t0 = time.perf_counter()
    state = step_fn(state)
    jax.block_until_ready(state.u)
    print(f"JIT+first step: {time.perf_counter()-t0:.2f}s")

    worst_T = 0.0
    check_interval = max(100, n_steps // 50)
    for i in range(2, n_steps + 1):
        state = step_fn(state)
        if i % check_interval == 0 or i == n_steps:
            jax.block_until_ready(state.u)
            max_u = float(jnp.max(jnp.abs(state.u)))
            maxT = float(jnp.max(jnp.abs(state.T)))
            topmean = float(jnp.mean(state.T[..., 0]))
            nan = int(jnp.isnan(state.u).sum() + jnp.isnan(state.T).sum())
            worst_T = max(worst_T, maxT)
            print(f"  day {i*dt/86400.0:6.2f} step {i:6d}: "
                  f"max|u|={max_u:7.3f}  max|T|={maxT:7.3f}  "
                  f"top-mean T={topmean:7.3f}  NaN={nan}")

    stable = (not bool(jnp.isnan(state.u).any() or jnp.isnan(state.T).any())
              and not bool(jnp.isinf(state.u).any() or jnp.isinf(state.T).any())
              and float(jnp.max(jnp.abs(state.u))) < 10.0)
    print("=" * 60)
    print(f"  max|T| over run = {worst_T:.3f} C")
    print(f"  STABLE = {stable}")
    print("=" * 60)
    return 0 if stable else 1


if __name__ == "__main__":
    sys.exit(main())
