"""
Tier-2 test T2-1: Inertial oscillation frequency (f-plane, uniform flow).

Verifies the solver's Coriolis term reproduces an exact inertial
oscillation: a horizontally-uniform velocity on an f-plane, with a flat
free surface and uniform density, should rotate at exactly the local
inertial frequency f0 = 2*Omega*sin(lat_center).

Why this is a clean analytic anchor:
  - Seed u=+U, v=0 everywhere (all depths), eta=0, uniform T/S, no forcing.
  - Laplacian & biharmonic viscosity of a *uniform* field are exactly zero,
    so the horizontal turbulent closures cannot contaminate the measurement.
  - Flat surface (eta=0) -> no barotropic pressure gradient force.
  - Uniform density -> no baroclinic pressure gradient force.
  - Uniform velocity -> divergence zero -> no free-surface gravity wave
    excitation and no vertical motion.
  - The ONLY active term is the Coriolis rotation, which the solver
    implements as an *exact* analytic rotation (f0*dt).

To keep it pure we isolate Coriolis per the Tier-1 precedent (T1d):
bottom friction (a non-gradient Rayleigh drag) is disabled via
dataclasses.replace, since a depth-uniform seed would otherwise feel it
on the bottom layer and slowly decay in amplitude. With dissipation
isolated out, the measured rotation frequency should match f0 to within
RK2 time-stepping truncation. Threshold: 0.1% relative error on |f|.

Northern-hemisphere inertial motion is CLOCKWISE: starting from (U,0),
the exact rotation gives (u,v) = (U cos, -U sin), so atan2(v,u) goes
NEGATIVE over time at rate -f0. We therefore compare |rotation rate|
against f0.

Run:  python src/bench_t2_inertial.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataclasses import replace

import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
import numpy as np

from config import DEFAULT_CONFIG
from grid import make_grid
from jax_solver import make_solver, JaxState

# ── Test parameters ─────────────────────────────────────────────────
U0 = 0.3          # m/s  uniform zonal velocity seed
DT = 200.0        # s    time step (well below CFL)
N_INERTIAL = 8    # propagate for 8 inertial periods

THRESH_REL = 1e-3  # 0.1% relative error on |frequency|
THRESH_AMP = 1e-3  # 0.1% rotational amplitude drift (RK2 truncation scale)


def main():
    grid = make_grid(DEFAULT_CONFIG.grid, DEFAULT_CONFIG.bathymetry_file)
    # Isolate Coriolis: disable non-gradient bottom friction (see docstring).
    physics = replace(DEFAULT_CONFIG.physics, bottom_friction='none', r_bot=0.0)

    f0 = grid.f0
    T_inertial = 2.0 * np.pi / f0
    print("=" * 60)
    print("T2-1 INERTIAL OSCILLATION (f-plane, uniform flow)")
    print("=" * 60)
    print(f"grid {grid.nx}x{grid.ny}x{grid.nz}  f0={f0:.6e} s^-1  "
          f"T_inertial={T_inertial:.1f} s = {T_inertial/3600:.2f} h")

    # Create solver (default forcing = zero / rest state)
    step, init_state, _ = make_solver(grid, physics, DT)
    _ = init_state()  # warm jit with zero state

    # Seed: uniform u=+U0, v=0 at all depths, flat eta, uniform T/S.
    base = init_state()
    state = JaxState(
        u=jnp.full_like(base.u, U0),
        v=jnp.zeros_like(base.v),
        T=base.T,
        S=base.S,
        eta=jnp.zeros_like(base.eta),
    )

    # Record u,v at a central point over time.
    i = grid.nx // 2
    j = grid.ny // 2
    k = 0  # top level
    n_steps = int(round(N_INERTIAL * T_inertial / DT))

    times = np.zeros(n_steps + 1)
    us = np.zeros(n_steps + 1)
    vs = np.zeros(n_steps + 1)
    times[0] = 0.0
    us[0] = float(state.u[i, j, k])
    vs[0] = float(state.v[i, j, k])

    t = 0.0
    for n in range(1, n_steps + 1):
        state = step(state)
        t += DT
        times[n] = t
        us[n] = float(state.u[i, j, k])
        vs[n] = float(state.v[i, j, k])

    # Fit the rotation angle atan2(v, u); northern-hemisphere rotation is
    # clockwise so the phase decreases at rate -f0. Use |rate| vs f0.
    angle = np.unwrap(np.arctan2(vs, us))
    slope = np.polyfit(times, angle, 1)[0]
    f_meas = abs(slope)
    rel_err = abs(f_meas - f0) / f0

    # Independent check: amplitude of |(u,v)| stays constant (pure rotation).
    amp0 = np.hypot(us[0], vs[0])
    amp_final = np.hypot(us[-1], vs[-1])
    amp_drift = abs(amp_final - amp0) / amp0

    print(f"\nseed (u0,v0)=({U0}, 0) m/s at all depths, eta=0, no forcing, "
          f"bottom friction isolated")
    print(f"propagated {N_INERTIAL} inertial periods over "
          f"{n_steps} steps (dt={DT:.0f} s)")
    print(f"  f_theory = {f0:.10e} s^-1")
    print(f"  |f_meas| = {f_meas:.10e} s^-1   rel_err = {rel_err:.3e}")
    print(f"  rotation amp drift (should be 0, pure rotation): {amp_drift:.3e}")

    ok = (rel_err < THRESH_REL) and (amp_drift < THRESH_AMP)
    print(f"\n{'PASS' if ok else 'FAIL'}: "
          f"rel_err {rel_err:.3e} < {THRESH_REL:.0e} and "
          f"amp_drift {amp_drift:.3e} < {THRESH_AMP:.0e}")
    return ok


if __name__ == "__main__":
    ok = main()
    sys.exit(0 if ok else 1)
