"""
IMEX Time Integrator -- Strang splitting for hydrostatic primitive equations.

Splits the RHS into linear (exactly integrated) and nonlinear (explicit Euler)
parts:

    du/dt = L(u) + N(u)

Linear part L (exact via matrix exponential / rotation):
  - Horizontal diffusion: nu_h * lap_h(u) for momentum, kappa_h * lap_h(T) for tracers
  - f-plane Coriolis rotation with f0
  - Linear shallow water (free surface): (eta, ubt, vbt) per wavenumber

Nonlinear part N (explicit Euler):
  - Advection (flux form, dealiased)
  - Baroclinic pressure gradient force
  - Vertical diffusion (finite difference on non-uniform z)
  - Beta-plane Coriolis correction: (f - f0)
  - Surface forcing (wind stress, heat flux)
  - Linear bottom friction

Strang splitting (symmetric, 2nd-order accurate):
  1. u* = exp(L * dt/2) u(t)          -- linear half-step
  2. u** = u* + N(u*) * dt             -- explicit full-step
  3. u(t+dt) = exp(L * dt/2) u**      -- linear half-step

The linear operators are unconditionally stable (exact matrix exponential
for diffusion, exact rotation for Coriolis, exact SW wave for free surface).
The explicit step is subject to CFL constraints (advection, vertical
diffusion), but these are mild at v0.1 resolution (dx ~ 9 km, dt = 600 s).
"""
import numpy as np

from spectral_ops import (
    linear_step_diffusion, laplacian_h, d_dx, d_dy, wavenumber_grid,
)
from momentum import compute_momentum_tendency, compute_vertical_velocity
from tracers import compute_tracer_tendency
from eos import compute_density
from pressure import compute_hydrostatic_pressure
from config import G_EARTH
# ── Internal helpers ─────────────────────────────────────────────────

def _coriolis_rotation(u, v, f0, dt):
    """
    Exact Coriolis rotation for the f-plane:
      du/dt = +f0 * v
      dv/dt = -f0 * u

    Solution is a rotation by angle f0*dt:
      u(t+dt) =  cos(f0*dt) * u + sin(f0*dt) * v
      v(t+dt) = -sin(f0*dt) * u + cos(f0*dt) * v

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


def _barotropic_velocity(u, v, grid):
    """Depth-averaged (barotropic) horizontal velocity.

    ubt = sum(0.5*(u[k]+u[k+1]) * dz_norm, axis=-1)

    Uses H_sw = sum(dz) as the normalizing depth so that dz_norm
    sums to 1 and ubt is a true depth-average over the z-grid span.
    This keeps barotropic extraction, SW wave frequency, and
    projection back to 3D fields mutually consistent.

    Returns:
        ubt, vbt: (nx, ny) barotropic velocities
    """
    dz = np.array(grid.dz)
    H_sw = float(np.sum(dz))  # z-grid vertical span
    dz_norm = dz.reshape(1, 1, -1) / H_sw
    u_avg = 0.5 * (u[..., :-1] + u[..., 1:])
    v_avg = 0.5 * (v[..., :-1] + v[..., 1:])
    ubt = np.sum(u_avg * dz_norm, axis=-1)
    vbt = np.sum(v_avg * dz_norm, axis=-1)
    return ubt, vbt


def _free_surface_step(state, grid, dt):
    """Exact linear shallow water step in spectral space (no Coriolis).

    Solves the coupled (eta, ubt, vbt) system per wavenumber:
        d(eta)/dt  = -H * (ikx*ubt + iky*vbt)     [continuity]
        d(ubt)/dt  = -g * ikx * eta               [x-momentum]
        d(vbt)/dt  = -g * iky * eta               [y-momentum]

    via matrix exponential: expm(A*dt) = I + s*A + c2*A^2
    where s = sin(w*dt)/w,  c2 = (1-cos(w*dt))/w^2,  w = sqrt(g*H*k^2).

    The k=0 mode (omega=0) is identity - mean mass and momentum conserved.
    Barotropic velocity changes are projected uniformly back to 3D fields.

    Modifies state in-place.
    """
    dz = np.array(grid.dz)
    H_sw = float(np.sum(dz))  # effective SW depth = z-grid vertical span
    nx, ny = grid.nx, grid.ny

    # Wavenumbers (2D, full arrays)
    kx_1d = 2.0 * np.pi * np.fft.fftfreq(nx, d=grid.dx)
    ky_1d = 2.0 * np.pi * np.fft.fftfreq(ny, d=grid.dy)
    kx = kx_1d.reshape(nx, 1)
    ky = ky_1d.reshape(1, ny)
    k2 = kx ** 2 + ky ** 2

    # Shallow water frequency and Rodrigues coefficients
    omega = np.sqrt(G_EARTH * H_sw * k2)
    sw_sin = np.sin(omega * dt)
    sw_cos = np.cos(omega * dt)

    # Barotropic velocity
    ubt, vbt = _barotropic_velocity(state.u, state.v, grid)

    # Spectral space
    eta_hat = np.fft.fft2(state.eta)
    ubt_hat = np.fft.fft2(ubt)
    vbt_hat = np.fft.fft2(vbt)

    # Safe-division coefficients (k=0 mode -> 0, leaving mean unchanged)
    omega_safe = np.where(omega > 0.0, omega, 1.0)
    k2_safe = np.where(k2 > 0.0, k2, 1.0)
    sin_div_w = np.where(omega > 0.0, sw_sin / omega_safe, 0.0)
    omc_div_k2 = np.where(k2 > 0.0, (1.0 - sw_cos) / k2_safe, 0.0)

    ikx = 1j * kx
    iky = 1j * ky

    # eta:  eta_new = cos(w dt) * eta  -  H * sin(w dt)/w * (ikx*ubt + iky*vbt)
    eta_hat_new = (sw_cos * eta_hat
                   - H_sw * sin_div_w * (ikx * ubt_hat + iky * vbt_hat))

    # ubt:  ubt + s*(-g*ikx)*eta  -  (1-cos)/k^2 * (kx^2*ubt + kx*ky*vbt)
    ubt_hat_new = (ubt_hat
                   - G_EARTH * ikx * sin_div_w * eta_hat
                   - omc_div_k2 * (kx ** 2 * ubt_hat + kx * ky * vbt_hat))

    # vbt:  vbt + s*(-g*iky)*eta  -  (1-cos)/k^2 * (kx*ky*ubt + ky^2*vbt)
    vbt_hat_new = (vbt_hat
                   - G_EARTH * iky * sin_div_w * eta_hat
                   - omc_div_k2 * (kx * ky * ubt_hat + ky ** 2 * vbt_hat))

    # Back to physical space
    state.eta = np.real(np.fft.ifft2(eta_hat_new))
    ubt_new = np.real(np.fft.ifft2(ubt_hat_new))
    vbt_new = np.real(np.fft.ifft2(vbt_hat_new))

    # Project barotropic delta back to 3D velocity (uniform over depth)
    delta_ubt = (ubt_new - ubt)[:, :, np.newaxis]
    delta_vbt = (vbt_new - vbt)[:, :, np.newaxis]
    state.u = state.u + delta_ubt
    state.v = state.v + delta_vbt


def _linear_half_step(state, grid, physics, dt):
    """Linear half-step: diffusion + Coriolis + free surface (all exact)."""
    # Horizontal diffusion (exact spectral decay)
    state.u = linear_step_diffusion(state.u, physics.nu_h, grid.dx, grid.dy, dt)
    state.v = linear_step_diffusion(state.v, physics.nu_h, grid.dx, grid.dy, dt)
    state.T = linear_step_diffusion(state.T, physics.kappa_h, grid.dx, grid.dy, dt)
    state.S = linear_step_diffusion(state.S, physics.kappa_h, grid.dx, grid.dy, dt)

    # f-plane Coriolis rotation (exact)
    state.u, state.v = _coriolis_rotation(state.u, state.v, grid.f0, dt)

    # Free surface (exact linear shallow water)
    _free_surface_step(state, grid, dt)


def _explicit_full_step(state, grid, physics, dt):
    """Nonlinear tendencies via explicit Euler, minus linear parts."""
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
    dudt -= grid.f0 * state.v
    dvdt += grid.f0 * state.u

    # Subtract barotropic PGF: -g*grad(eta) is handled by _free_surface_step
    bt_pgf_x = -G_EARTH * d_dx(state.eta, grid.dx)
    bt_pgf_y = -G_EARTH * d_dy(state.eta, grid.dy)
    dudt -= bt_pgf_x[:, :, np.newaxis]
    dvdt -= bt_pgf_y[:, :, np.newaxis]

    # Explicit Euler update
    state.u += dudt * dt
    state.v += dvdt * dt
    state.T += dTdt * dt
    state.S += dSdt * dt


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
