"""
Longer multi-day stability verification on real WOA2023 fields, WITH forcing.

The short 100-step verification (verify_real_run.py, ~8.3 h) proved the
scale-selective biharmonic fix (`33fb58c`) stabilizes the forcing-coupled
dt=300 config that previously blew up at step ~30-40 (report 5.3).  This
script extends that check to a multi-day integration to confirm the fix
holds over longer time horizons and that sustained forcing does NOT cause
the slow temperature drift (T: 23.88 -> ~29.5C) seen before the fix.

Config: WOA init + Stommel gyre wind + meridional heat flux, dt=300s,
        integrated for 4 days (345,600 s = 1152 steps).

Pass criteria:
  - no NaN / Inf in u, v, T
  - max|u| stays bounded (< 10 m/s, far below the DEGRADED >10 m/s threshold)
  - no monotonic surface/domain temperature drift: the max domain temperature
    over the final quarter must not exceed (init max + 2.0 C) nor show a
    sustained upward trend.

Run:  python src/verify_real_run_long.py > logs/verify_real_run_long.log 2>&1
"""
import sys
import os
import time

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
N_STEPS = 1152            # 4 days at dt=300s
DAYS = N_STEPS * DT / 86400.0
DRIFT_TOL_C = 2.0         # allowed max temperature growth over the run (C)
MAX_U_BOUND = 10.0        # m/s, below the DEGRADED (>10) threshold


def main():
    grid = make_grid(DEFAULT_CONFIG.grid, DEFAULT_CONFIG.bathymetry_file)
    physics = DEFAULT_CONFIG.physics
    dt = DT

    print("=" * 60)
    print("LONG FORCED RUN - Real WOA + wind + heat (multi-day)")
    print("=" * 60)
    print(f"Grid: {grid.nx}x{grid.ny}x{grid.nz}")
    print(f"Physics: nu_h={physics.nu_h}, kappa_h={physics.kappa_h}, "
          f"nu_v={physics.nu_v}, kappa_v={physics.kappa_v}")
    print(f"dt={dt}s, steps={N_STEPS} => {DAYS:.1f} days simulated")
    print()

    # WOA initial fields
    print("Loading WOA2023 climatology...")
    T_init, S_init = get_initial_fields(grid)
    T_init_max = float(np.max(T_init))
    print(f"  T_init range=[{T_init.min():.2f}, {T_init_max:.2f}]C")

    # Forcing: Stommel gyre wind + meridional heat flux
    tau_x, tau_y = wind_stress_gyre(grid, tau0=0.1)
    Q_heat = heat_flux_meridional(grid, Q0=50.0)
    forcing = (tau_x, tau_y, Q_heat)

    step_fn, init_state, diag_fn = make_solver(
        grid, physics, dt, forcing=forcing
    )
    state = init_state(T_init=T_init, S_init=S_init)

    # JIT compile
    t0 = time.perf_counter()
    state = step_fn(state)
    jax.block_until_ready(state.u)
    print(f"JIT compile + first step: {time.perf_counter()-t0:.2f}s")

    # Track max|T| history to detect sustained drift
    maxT_history = [float(jnp.max(state.T))]

    check_interval = 100
    n_checks = 0
    max_u_peak = 0.0

    for i in range(2, N_STEPS + 1):
        state = step_fn(state)
        if i % check_interval == 0 or i == N_STEPS:
            jax.block_until_ready(state.u)
            max_u = float(jnp.max(jnp.abs(state.u)))
            max_T = float(jnp.max(jnp.abs(state.T)))
            mean_T_top = float(jnp.mean(state.T[..., 0]))
            nan = int(jnp.isnan(state.u).sum() + jnp.isnan(state.T).sum())
            max_u_peak = max(max_u_peak, max_u)
            maxT_history.append(max_T)
            n_checks += 1
            print(f"  day {i*dt/86400.0:5.2f}  step {i:5d}: "
                  f"max|u|={max_u:7.3f}  max|T|={max_T:7.3f}  "
                  f"top-mean T={mean_T_top:7.3f}  NaN={nan}")

    # Final diagnostics
    rho, pressure, w = diag_fn(state)
    print(f"  Final rho range=[{float(rho.min()):.2f}, "
          f"{float(rho.max()):.2f}] kg/m^3")
    print(f"        w range=[{float(w.min()):.4e}, {float(w.max()):.4e}] m/s")

    # ---- Pass criteria ----
    has_nan = bool(jnp.isnan(state.u).any() or jnp.isnan(state.v).any()
                   or jnp.isnan(state.T).any())
    has_inf = bool(jnp.isinf(state.u).any() or jnp.isinf(state.v).any()
                   or jnp.isinf(state.T).any())
    max_u_final = float(jnp.max(jnp.abs(state.u)))
    max_T_final = float(jnp.max(jnp.abs(state.T)))

    # Drift: compare first-half vs final max temperature.  If the final
    # quarter's max exceeds (init max + tolerance), the run is drifting up.
    quarters = len(maxT_history)
    first_half_maxT = max(maxT_history[:quarters // 2])
    final_quarter_maxT = max(maxT_history[3 * quarters // 4:])
    drift_up = final_quarter_maxT - T_init_max > DRIFT_TOL_C
    monotonic_drift = final_quarter_maxT - first_half_maxT > DRIFT_TOL_C

    print()
    print("=" * 60)
    print(f"  max|u| peak   = {max_u_peak:.3f} m/s (bound {MAX_U_BOUND})")
    print(f"  max|u| final  = {max_u_final:.3f} m/s")
    print(f"  max|T| final  = {max_T_final:.3f} C (init max {T_init_max:.2f})")
    print(f"  maxT first-half  = {first_half_maxT:.3f} C")
    print(f"  maxT final-quarter= {final_quarter_maxT:.3f} C")
    print(f"  drift above init ({DRIFT_TOL_C}C tol) = {drift_up}")
    print(f"  drift second-half ({DRIFT_TOL_C}C tol) = {monotonic_drift}")
    print("=" * 60)

    stable = (not has_nan and not has_inf
              and max_u_final < MAX_U_BOUND
              and not drift_up and not monotonic_drift)

    if stable:
        print(f"PASS: Stable over {DAYS:.1f} days forced run, no T drift")
    else:
        print(f"FAIL: nan={has_nan} inf={has_inf} "
              f"max_u={max_u_final:.2f} drift_up={drift_up} "
              f"monotonic_drift={monotonic_drift}")
    print("=" * 60)
    return 0 if stable else 1


if __name__ == "__main__":
    sys.exit(main())
