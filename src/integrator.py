"""
IMEX Time Integrator -- Strang splitting for hydrostatic primitive equations.

Splits the RHS into linear (exactly integrated) and nonlinear (explicit Euler)
parts:

    du/dt = L(u) + N(u)

Linear part L (exact via matrix exponential / rotation):
  - Horizontal diffusion: nu_h * lap_h(u) for momentum, kappa_h * lap_h(T) for tracers
  - f-plane Coriolis rotation with f0

Nonlinear part N (explicit Euler):
  - Advection (flux form, dealiased)
  - Pressure gradient force
  - Vertical diffusion (finite difference on non-uniform z)
  - Beta-plane Coriolis correction: (f - f0)
  - Surface forcing (wind stress, heat flux)
  - Linear bottom friction

Strang splitting (symmetric, 2nd-order accurate):
  1. u* = exp(L * dt/2) u(t)          -- linear half-step
  2. u** = u* + N(u*) * dt             -- explicit full-step
  3. u(t+dt) = exp(L * dt/2) u**      -- linear half-step

The linear operators are unconditionally stable (exact matrix exponential
for diffusion, exact rotation for Coriolis). The explicit step is subject
to CFL constraints (advection, vertical diffusion), but these are mild at
v0.1 resolution (dx ~ 9 km, dt = 600 s).

Note: sea surface height (eta) is NOT evolved in v0.1 (rigid lid).
The surface pressure gradient is captured through the hydrostatic pressure
computation from eta=0.
"""
import numpy as np

from spectral_ops import linear_step_diffusion, laplacian_h
from momentum import compute_momentum_tendency, compute_vertical_velocity
from tracers import compute_tracer_tendency
from eos import compute_density
from pressure import compute_hydrostatic_pressure


# ── Internal helpers ─────────────────────────────────────────────────

def _coriolis_rotation(u, v, f0, dt):
    """
    Exact Coriolis rotation for the f-plane:
      du/dt = +f0 * v
      dv/dt = -f0 * u

    Solution is a rotation by angle f0*dt:
      u(t+dt) =  cos(f0*dt) * u + sin(f0*dt) * v
      v(t+dt) = -sin(f0*dt) * u + cos(f0*dt) * v

    This rotates a northward velocity (v>0) toward the east (u>0)
    in the Northern Hemisphere (f0 > 0), which is the correct
    physical direction for Coriolis deflection.

    Args:
        u, v: velocity arrays, any shape
        f0: Coriolis parameter at domain center [1/s]
        dt: time step [s]

    Returns: (u_new, v_new)
    """
    angle = f0 * dt
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)
    u_new =  cos_a * u + sin_a * v
    v_new = -sin_a * u + cos_a * v
    return u_new, v_new


def _linear_half_step(state, grid, physics, dt):
    """
    Apply linear operators (horizontal diffusion + f-plane Coriolis) for time dt.

    Uses exact matrix exponential for diffusion and exact rotation for Coriolis.
    Both are unconditionally stable -- no CFL constraint on the linear part.

    Modifies state in-place.
    """
    # Horizontal diffusion (exact spectral decay)
    state.u = linear_step_diffusion(state.u, physics.nu_h, grid.dx, grid.dy, dt)
    state.v = linear_step_diffusion(state.v, physics.nu_h, grid.dx, grid.dy, dt)
    state.T = linear_step_diffusion(state.T, physics.kappa_h, grid.dx, grid.dy, dt)
    state.S = linear_step_diffusion(state.S, physics.kappa_h, grid.dx, grid.dy, dt)

    # f-plane Coriolis rotation (exact)
    state.u, state.v = _coriolis_rotation(state.u, state.v, grid.f0, dt)


def _explicit_full_step(state, grid, physics, dt):
    """
    Apply nonlinear tendencies via explicit Euler for time dt.

    Computes the full tendency from compute_momentum_tendency and
    compute_tracer_tendency (which include ALL terms), then subtracts
    the linear parts (horizontal diffusion + f0 Coriolis) that are
    handled exactly by _linear_half_step.

    This avoids double-counting: the linear operators are applied
    exactly in the half-steps, and only the genuinely nonlinear/explicit
    terms are advanced by Euler.

    Modifies state in-place.
    """
    # Force fresh pressure computation from current T, S, eta
    state.p = None

    # Compute full tendencies (includes all terms)
    dudt, dvdt = compute_momentum_tendency(state, grid, physics)
    dTdt, dSdt = compute_tracer_tendency(state, grid, physics)

    # Subtract horizontal diffusion (handled by linear_step_diffusion)
    dudt -= physics.nu_h * laplacian_h(state.u, grid.dx, grid.dy)
    dvdt -= physics.nu_h * laplacian_h(state.v, grid.dx, grid.dy)
    dTdt -= physics.kappa_h * laplacian_h(state.T, grid.dx, grid.dy)
    dSdt -= physics.kappa_h * laplacian_h(state.S, grid.dx, grid.dy)

    # Subtract f0 Coriolis (handled by _coriolis_rotation)
    # Remaining Coriolis in explicit part: (f - f0) = beta-plane correction
    dudt -= grid.f0 * state.v     # remove +f0*v from du/dt
    dvdt += grid.f0 * state.u     # remove -f0*u from dv/dt

    # Explicit Euler update
    state.u += dudt * dt
    state.v += dvdt * dt
    state.T += dTdt * dt
    state.S += dSdt * dt

    # eta is not evolved in v0.1 (rigid lid)


def _update_diagnostics(state, grid, physics):
    """
    Recompute diagnostic fields (rho, p, w) from current prognostic state.

    Called after each full step to keep diagnostics consistent with
    the updated prognostic variables.
    """
    state.rho = compute_density(state.T, state.S, physics)
    state.p = compute_hydrostatic_pressure(state, grid, physics)
    state.w = compute_vertical_velocity(state, grid)


# ── Public API ───────────────────────────────────────────────────────

def step(state, grid, physics, time_cfg, dt=None):
    """
    Advance the state by one timestep using Strang splitting IMEX.

    Args:
        state: ModelState (modified in-place)
        grid: OceanGrid
        physics: PhysicsConfig
        time_cfg: TimeConfig (uses time_cfg.dt if dt is None)
        dt: override timestep [s] (optional, for testing)

    Returns:
        state (same object, modified in-place)
    """
    if dt is None:
        dt = time_cfg.dt

    # Strang splitting: L(dt/2) -> N(dt) -> L(dt/2)
    _linear_half_step(state, grid, physics, dt / 2.0)
    _explicit_full_step(state, grid, physics, dt)
    _linear_half_step(state, grid, physics, dt / 2.0)

    # Update diagnostics
    _update_diagnostics(state, grid, physics)

    return state


def integrate(state, grid, physics, time_cfg, callback=None):
    """
    Integrate the model forward in time.

    Args:
        state: ModelState (modified in-place)
        grid: OceanGrid
        physics: PhysicsConfig
        time_cfg: TimeConfig
        callback: optional function(t, state) called at output intervals

    Returns:
        state (same object, modified in-place)
    """
    dt = time_cfg.dt
    t_total = time_cfg.t_total
    dt_output = time_cfg.dt_output

    n_steps = int(round(t_total / dt))
    output_every = max(1, int(round(dt_output / dt)))

    # Initial diagnostics
    _update_diagnostics(state, grid, physics)

    if callback is not None:
        callback(0.0, state)

    for n in range(n_steps):
        step(state, grid, physics, time_cfg, dt)

        if callback is not None and (n + 1) % output_every == 0:
            callback((n + 1) * dt, state)

    return state
