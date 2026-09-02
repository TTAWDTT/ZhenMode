"""
Damping sweep diagnostic — isolate what stabilizes the long WOA run.

The 150-step no-forcing run at dt=300 blows up at step ~35 via a
growing baroclinic eddy instability (PGF -> advection -> PGF feedback).
This sweep boots nu_h and kappa_h to find the dissipation needed to make
the run stable, confirming diffusion magnitude is the lever and giving
the calibration for the biharmonic fix.

Run:  python src/diag_damping.py  (> logs/diag_damping.log 2>&1)
"""
import sys, os, time
import dataclasses

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
import numpy as np

from config import DEFAULT_CONFIG, RHO_0, G_EARTH
from grid import make_grid
from jax_solver import (
    JaxState, _compute_params,
    _linear_half_step, _explicit_full_step,
)
from woa_data import get_initial_fields


def _make_steps(params):
    @jax.jit
    def half_step(state, dt_h):
        return _linear_half_step(state, params, dt_h)
    @jax.jit
    def full_step(state, dt_f):
        return _explicit_full_step(state, params, dt_f)
    return half_step, full_step


def run_case(grid, T_init, S_init, physics, dt, n_steps):
    """Run n_steps Strang halves; return (blow_step, max|u|, max|T|)."""
    params = _compute_params(grid, physics, dt, forcing=None)
    half_step, full_step = _make_steps(params)
    nx, ny, nz = grid.nx, grid.ny, grid.nz
    state = JaxState(
        u=jnp.zeros((nx, ny, nz)),
        v=jnp.zeros((nx, ny, nz)),
        T=jnp.array(T_init), S=jnp.array(S_init),
        eta=jnp.zeros((nx, ny)),
    )
    mu_end = 0.0
    Tmax_end = 0.0
    for i in range(1, n_steps + 1):
        s_L1 = half_step(state, dt / 2.0)
        s_N = full_step(s_L1, dt)
        state = half_step(s_N, dt / 2.0)
        mu_end = float(jnp.max(jnp.abs(state.u)))
        Tmax_end = float(jnp.max(jnp.abs(state.T)))
        nan = bool(jnp.isnan(state.u).any() or jnp.isnan(state.T).any())
        if nan or mu_end > 1e6 or abs(Tmax_end) > 500:
            return i, mu_end, Tmax_end
    return None, mu_end, Tmax_end


def main():
    grid = make_grid(DEFAULT_CONFIG.grid, DEFAULT_CONFIG.bathymetry_file)
    base_physics = DEFAULT_CONFIG.physics
    T_init, S_init = get_initial_fields(grid)
    print(f"WOA T range: [{T_init.min():.2f}, {T_init.max():.2f}] C")
    print(f"WOA S range: [{S_init.min():.2f}, {S_init.max():.2f}] PSU")

    DT = 300.0
    N_STEPS = 150

    # Sweep table: (label, nu_h, kappa_h, nu_bi, kappa_bi)
    # First block: Laplacian-only sweep (nu_bi=0) — reference from the
    # original diagnosis. Second block: biharmonic sweep with Laplacian
    # held at baseline nu_h=kappa_h=1e2; biharmonic damps grid-scale
    # modes ∝ k^4 so it should stabilize without smearing basin flow.
    sweeps = [
        ("baseline nu_h=1e2", 1e2, 1e2, 0.0, 0.0),
        ("nu_h=1e4", 1e4, 1e2, 0.0, 0.0),
        ("nu_h=1e4,kappa=1e4", 1e4, 1e4, 0.0, 0.0),
        # Biharmonic sweep (Laplacian kept at baseline 1e2)
        ("bi 1e10", 1e2, 1e2, 1e10, 1e10),
        ("bi 1e11", 1e2, 1e2, 1e11, 1e11),
        ("bi 1e12", 1e2, 1e2, 1e12, 1e12),
        ("bi 1e12,kappa=3e12", 1e2, 1e2, 1e12, 3e12),
        ("bi 3e12", 1e2, 1e2, 3e12, 3e12),
        ("bi 1e13", 1e2, 1e2, 1e13, 1e13),
    ]

    print(f"\nDamping sweep: dt={DT}s, {N_STEPS} steps ({N_STEPS*DT/3600:.1f} h)")
    print(f"{'case':<26}{'blow_step':>10}{'max|u|':>12}{'max|T|':>12}")
    for label, nu_h, kappa_h, nu_bi, kappa_bi in sweeps:
        phys = dataclasses.replace(base_physics, nu_h=nu_h, kappa_h=kappa_h,
                                   nu_bi=nu_bi, kappa_bi=kappa_bi)
        t0 = time.time()
        blow, mu, Tmax = run_case(grid, T_init, S_init, phys, DT, N_STEPS)
        dt_ms = (time.time() - t0) * 1000
        bs = "stable" if blow is None else str(blow)
        print(f"{label:<26}{bs:>10}{mu:>12.4e}{Tmax:>12.3f}  ({dt_ms:.0f} ms)")



if __name__ == "__main__":
    main()
