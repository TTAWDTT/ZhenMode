"""
Stability verification with real WOA2023 initial fields.

Loads temperature/salinity from WOA climatology, initializes the solver
with realistic stratification, and runs with wind + heat forcing to
verify stability under real ocean conditions.

Tests:
  1. WOA initial fields, no forcing (free adjustment of density field)
  2. WOA initial fields + Stommel gyre wind + meridional heat flux
  3. WOA initial fields + wind only (no heat flux)
"""
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
import numpy as np

from config import DEFAULT_CONFIG, RHO_0
from grid import make_grid
from jax_solver import make_solver, JaxState
from woa_data import get_initial_fields
from forcing import wind_stress_gyre, heat_flux_meridional


def run_test(name, grid, physics, dt, n_steps, forcing=None,
             T_init=None, S_init=None):
    """Run a stability test and report diagnostics."""
    print(f"\n{'='*60}")
    print(f"Test: {name}")
    print(f"  dt={dt}s, steps={n_steps}, "
          f"forcing={'yes' if forcing else 'no'}, "
          f"init={'WOA' if T_init is not None else 'uniform'}")
    print(f"{'='*60}")

    step_fn, init_state, diag_fn = make_solver(
        grid, physics, dt, forcing=forcing
    )

    state = init_state(T_init=T_init, S_init=S_init)

    # Report initial state
    print(f"  Init: T=[{float(state.T.min()):.2f}, "
          f"{float(state.T.max()):.2f}]C")
    print(f"        S=[{float(state.S.min()):.2f}, "
          f"{float(state.S.max()):.2f}] PSU")
    print(f"        NaN check: T={int(jnp.isnan(state.T).sum())}, "
          f"S={int(jnp.isnan(state.S).sum())}")

    # JIT compile
    t0 = time.perf_counter()
    state = step_fn(state)
    jax.block_until_ready(state.u)
    t1 = time.perf_counter()
    print(f"  JIT compile + first step: {t1-t0:.2f}s")

    # Check first step
    max_u = float(jnp.max(jnp.abs(state.u)))
    max_T = float(jnp.max(jnp.abs(state.T)))
    max_eta = float(jnp.max(jnp.abs(state.eta)))
    print(f"  Step  1: max|u|={max_u:.4e}  max|T|={max_T:.4e}  "
          f"max|eta|={max_eta:.4e}")

    # Run remaining steps
    check_interval = max(1, n_steps // 10)
    for i in range(2, n_steps + 1):
        state = step_fn(state)
        if i % check_interval == 0 or i == n_steps:
            jax.block_until_ready(state.u)
            max_u = float(jnp.max(jnp.abs(state.u)))
            max_T = float(jnp.max(jnp.abs(state.T)))
            max_eta = float(jnp.max(jnp.abs(state.eta)))
            nan_u = int(jnp.isnan(state.u).sum())
            print(f"  Step {i:3d}: max|u|={max_u:.4e}  "
                  f"max|T|={max_T:.4e}  max|eta|={max_eta:.4e}  "
                  f"NaN(u)={nan_u}")

    # Final diagnostics
    rho, pressure, w = diag_fn(state)
    print(f"  Final: rho=[{float(rho.min()):.2f}, "
          f"{float(rho.max()):.2f}] kg/m^3")
    print(f"         w=[{float(w.min()):.4e}, "
          f"{float(w.max()):.4e}] m/s")

    has_nan = bool(jnp.isnan(state.u).any() or jnp.isnan(state.T).any())
    has_inf = bool(jnp.isinf(state.u).any() or jnp.isinf(state.T).any())
    stable = not has_nan and not has_inf and max_u < 1e3

    if stable:
        print(f"  RESULT: STABLE")
    else:
        print(f"  RESULT: UNSTABLE (nan={has_nan}, inf={has_inf}, "
              f"max_u={max_u:.2e})")

    return stable


def main():
    grid = make_grid(DEFAULT_CONFIG.grid, DEFAULT_CONFIG.bathymetry_file)
    physics = DEFAULT_CONFIG.physics
    dt = 300.0
    n_steps = 100

    print("=" * 60)
    print("WOA Real Initial Fields - Stability Verification")
    print("=" * 60)
    print(f"Grid: {grid.nx}x{grid.ny}x{grid.nz}")
    print(f"  lon: [{grid.lon[0]:.1f}, {grid.lon[-1]:.1f}]E")
    print(f"  lat: [{grid.lat[0]:.1f}, {grid.lat[-1]:.1f}]N")
    print(f"  z:   {grid.z}")
    print(f"Physics: nu_h={physics.nu_h}, kappa_h={physics.kappa_h}")
    print(f"  T_ref={physics.T_ref}, S_ref={physics.S_ref}")
    print()

    # Load WOA initial fields
    print("Loading WOA2023 climatology...")
    T_init, S_init = get_initial_fields(grid)
    print(f"  T_init: shape={T_init.shape}, "
          f"range=[{T_init.min():.2f}, {T_init.max():.2f}]C")
    print(f"  S_init: shape={S_init.shape}, "
          f"range=[{S_init.min():.2f}, {S_init.max():.2f}] PSU")
    print(f"  NaN: T={int(np.isnan(T_init).sum())}, "
          f"S={int(np.isnan(S_init).sum())}")

    # Forcing fields
    tau_x, tau_y = wind_stress_gyre(grid, tau0=0.1)
    Q_heat = heat_flux_meridional(grid, Q0=50.0)

    all_pass = True

    # Test 1: WOA initial fields, no forcing
    all_pass &= run_test(
        "WOA init, no forcing (free adjustment)",
        grid, physics, dt, n_steps,
        forcing=None, T_init=T_init, S_init=S_init
    )

    # Test 2: WOA init + wind + heat
    all_pass &= run_test(
        "WOA init + Stommel wind + meridional heat",
        grid, physics, dt, n_steps,
        forcing=(tau_x, tau_y, Q_heat),
        T_init=T_init, S_init=S_init
    )

    # Test 3: WOA init + wind only
    forcing_wind_only = (tau_x, tau_y, np.zeros_like(tau_x))
    all_pass &= run_test(
        "WOA init + wind only (no heat flux)",
        grid, physics, dt, n_steps,
        forcing=forcing_wind_only,
        T_init=T_init, S_init=S_init
    )

    print("\n" + "=" * 60)
    if all_pass:
        print("ALL TESTS PASSED - Solver is stable with real WOA data")
    else:
        print("SOME TESTS FAILED - See output above")
    print("=" * 60)


if __name__ == "__main__":
    main()
