"""Verify semi-implicit baroclinic PGF fix: stable at dt=300s for 200+ steps.

Tests both no-forcing (perturbation only) and full forcing (wind + heat).
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp

from config import DEFAULT_CONFIG
from grid import make_grid
from forcing import wind_stress_gyre, heat_flux_meridional
from jax_solver import make_solver, JaxState


def run_test(dt, n_steps, forcing=None, label=""):
    """Run n_steps and report max|u|, max|T-Tref|, max|eta| every 50 steps."""
    grid = make_grid(DEFAULT_CONFIG.grid, DEFAULT_CONFIG.bathymetry_file)
    physics = DEFAULT_CONFIG.physics

    step_fn, init_state, _ = make_solver(grid, physics, dt, forcing=forcing)

    state = init_state()

    # Add small perturbation to trigger instability (this was the blowup seed)
    key = jax.random.PRNGKey(42)
    key1, key2, key3 = jax.random.split(key, 3)
    state = JaxState(
        u=state.u + jax.random.normal(key1, state.u.shape) * 0.001,
        v=state.v + jax.random.normal(key2, state.v.shape) * 0.001,
        T=state.T + jax.random.normal(key3, state.T.shape) * 0.001,
        S=state.S,
        eta=state.eta,
    )

    print(f"\n=== {label} ===")
    print(f"dt={dt}s, n_steps={n_steps}, forcing={'full' if forcing is not None else 'none'}")
    print(f"  Step  0: max|u|={jnp.max(jnp.abs(state.u)):.6e}  "
          f"max|T-Tref|={jnp.max(jnp.abs(state.T - physics.T_ref)):.6e}  "
          f"max|eta|={jnp.max(jnp.abs(state.eta)):.6e}")

    for i in range(1, n_steps + 1):
        state = step_fn(state)

        if i % 50 == 0 or i == n_steps:
            max_u = float(jnp.max(jnp.abs(state.u)))
            max_T = float(jnp.max(jnp.abs(state.T - physics.T_ref)))
            max_eta = float(jnp.max(jnp.abs(state.eta)))

            # Check for NaN
            has_nan = bool(jnp.any(jnp.isnan(state.u)))

            status = "NaN!" if has_nan else "OK"
            print(f"  Step {i:3d}: max|u|={max_u:.6e}  "
                  f"max|T-Tref|={max_T:.6e}  "
                  f"max|eta|={max_eta:.6e}  {status}")

            if has_nan:
                print(f"  -> BLOWUP at step {i}")
                return False

            # Check for blowup (values > 1e6)
            if max_u > 1e6 or max_T > 1e6 or max_eta > 1e6:
                print(f"  -> BLOWUP at step {i} (values > 1e6)")
                return False

    max_u_final = float(jnp.max(jnp.abs(state.u)))
    print(f"  -> STABLE: max|u|={max_u_final:.6e} after {n_steps} steps")
    return True


def main():
    print("=== Semi-implicit baroclinic PGF stability verification ===")

    # Test 1: No forcing, perturbation only (was the original blowup case)
    ok1 = run_test(dt=300.0, n_steps=200, forcing=None,
                   label="Test 1: No forcing, perturbation only")

    # Test 2: Full forcing (wind + heat)
    grid = make_grid(DEFAULT_CONFIG.grid, DEFAULT_CONFIG.bathymetry_file)
    tau_x, tau_y = wind_stress_gyre(grid, tau0=0.1)
    Q_heat = heat_flux_meridional(grid, Q0=50.0)
    forcing = (tau_x, tau_y, Q_heat)
    ok2 = run_test(dt=300.0, n_steps=200, forcing=forcing,
                   label="Test 2: Full forcing (wind + heat)")

    # Test 3: Longer run at dt=600 (default config) with full forcing
    ok3 = run_test(dt=600.0, n_steps=200, forcing=forcing,
                   label="Test 3: dt=600s, full forcing")

    print("\n" + "=" * 60)
    print(f"Results: Test1={'PASS' if ok1 else 'FAIL'}  "
          f"Test2={'PASS' if ok2 else 'FAIL'}  "
          f"Test3={'PASS' if ok3 else 'FAIL'}")
    if ok1 and ok2 and ok3:
        print("ALL TESTS PASSED - instability fixed!")
    else:
        print("SOME TESTS FAILED - instability persists")
    print("=" * 60)


if __name__ == "__main__":
    main()
