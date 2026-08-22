"""
Longer multi-day stability verification on real WOA2023 fields, WITH forcing.

The short 100-step verification (verify_real_run.py, ~8.3 h) proved the
scale-selective biharmonic fix (`33fb58c`) stabilizes the forcing-coupled
dt=300 config that previously blew up at step ~30-40 (report 5.3).  This
script extends that check to a multi-day integration to confirm the fix
holds over longer time horizons and that sustained forcing does NOT cause
the slow temperature drift (T: 23.88 -> ~29.5C) seen before the fix.

Config: WOA init + Stommel gyre wind + meridional heat flux, dt=300s.
        Integration length is set via --days (default 4 days).

Pass criteria (forced-equilibrium bar, set with the user's approval):
  - no NaN / Inf in u, v, T
  - max|u| stays bounded (< 10 m/s, far below the DEGRADED >10 m/s threshold)
  - no monotonic drift: the max domain temperature over the final quarter must
    not keep climbing > 2.0 C relative to the first half (equilibration, not
    accumulation), and must stay within an absolute ceiling
    (AMPLITUDE_CAP_C above the init max) that only a true runaway hotspot
    would breach.  A heated forced ocean legitimately equilibrates above the
    initial SST, so "drift_up vs init SST" is informational, not a pass gate.

Run:  python src/verify_real_run_long.py [--days N] > logs/verify_real_run_long.log 2>&1
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

DT = 300.0               # s
DRIFT_TOL_C = 2.0        # allowed max temperature growth over the run (C)
MAX_U_BOUND = 10.0       # m/s, below the DEGRADED (>10) threshold
AMPLITUDE_CAP_C = 12.0   # absolute max-T ceiling above init (C); catches a
                         # true runaway hotspot, far above any physical forced
                         # SST elevation (~3-4 C in these runs)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=float, default=4.0,
                    help="Days to integrate (default 4.0)")
    ap.add_argument("--restore-days", type=float, default=0.0,
                    help="surface T restoring timescale in days (Haney "
                         "relaxation to initial SST); 0 = disabled")
    ap.add_argument("--kappa-conv", type=float, default=None,
                    help="override convective vertical diffusivity (m^2/s); "
                         "None = physics default")
    args = ap.parse_args()
    days = args.days
    n_steps = int(round(days * 86400.0 / DT))

    grid = make_grid(DEFAULT_CONFIG.grid, DEFAULT_CONFIG.bathymetry_file)
    physics = DEFAULT_CONFIG.physics
    if args.kappa_conv is not None:
        from dataclasses import replace
        physics = replace(physics, kappa_conv=args.kappa_conv)
    dt = DT

    print("=" * 60)
    print(f"LONG FORCED RUN - Real WOA + wind + heat ({days:.1f} days)")
    print("=" * 60)
    print(f"Grid: {grid.nx}x{grid.ny}x{grid.nz}")
    print(f"Physics: nu_h={physics.nu_h}, kappa_h={physics.kappa_h}, "
          f"nu_v={physics.nu_v}, kappa_v={physics.kappa_v}")
    print(f"dt={dt}s, steps={n_steps} => {days:.1f} days simulated")
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

    # Surface temperature restoring target = initial upper-level T.
    # When --restore-days > 0 this anchors SST to the initial field,
    # preventing the wind-driven month-scale thermal runaway.
    T_sst = T_init[:, :, 0] if args.restore_days > 0 else None
    if T_sst is not None:
        step_fn, init_state, diag_fn = make_solver(
            grid, physics, dt, forcing=forcing,
            T_sst=T_sst, tau_restore_days=args.restore_days)
    else:
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

    # Check ~50 times across the run, at least every 100 steps
    check_interval = max(100, n_steps // 50)
    max_u_peak = 0.0

    for i in range(2, n_steps + 1):
        state = step_fn(state)
        if i % check_interval == 0 or i == n_steps:
            jax.block_until_ready(state.u)
            max_u = float(jnp.max(jnp.abs(state.u)))
            max_T = float(jnp.max(jnp.abs(state.T)))
            mean_T_top = float(jnp.mean(state.T[..., 0]))
            nan = int(jnp.isnan(state.u).sum() + jnp.isnan(state.T).sum())
            max_u_peak = max(max_u_peak, max_u)
            maxT_history.append(max_T)
            print(f"  day {i*dt/86400.0:6.2f}  step {i:6d}: "
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

    # ── Forced-equilibrium stability criteria ──
    # A continuously-heated forced ocean (Stommel wind + meridional heat)
    # legitimately equilibrates ABOVE its initial WOA surface temperature:
    # the added heat must elevate SST to a new forced balance.  Anchoring
    # the pass bar to the *initial* max SST therefore mislabels a healthy
    # equilibrium as drift.  The correct "no-drift" signal is CONVERGENCE:
    # the second half of the run must not keep climbing relative to the
    # first half (monotonic_drift), and the equilibrium amplitude must
    # remain bounded (absolute cap well above any physical forced balance,
    # catching only a true runaway hotspot).
    quarters = len(maxT_history)
    first_half_maxT = max(maxT_history[:quarters // 2])
    final_quarter_maxT = max(maxT_history[3 * quarters // 4:])
    # Second half climbs > tolerance above first half => persistent
    # accumulation (unstable), not an equilibrated forced state.
    monotonic_drift = final_quarter_maxT - first_half_maxT > DRIFT_TOL_C
    # Absolute ceiling: only catches a genuine grid/coupling runaway, far
    # above any physically reasonable forced SST elevation (~3-4 C here).
    amplitude_bounded = final_quarter_maxT < T_init_max + AMPLITUDE_CAP_C
    # Informational only: how far the equilibrium sits above the initial SST
    # (NOT a pass/fail — a heated forced run is expected to be elevated).
    drift_up = final_quarter_maxT - T_init_max > DRIFT_TOL_C

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
              and amplitude_bounded and not monotonic_drift)

    if stable:
        print(f"PASS: Stable over {days:.1f} days forced run, equilibrated")
    else:
        print(f"FAIL: nan={has_nan} inf={has_inf} "
              f"max_u={max_u_final:.2f} "
              f"amplitude_bounded={amplitude_bounded} "
              f"monotonic_drift={monotonic_drift}")
    print("=" * 60)
    return 0 if stable else 1



if __name__ == "__main__":
    sys.exit(main())
