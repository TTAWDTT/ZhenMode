"""
Momentum Equations — hydrostatic primitive equations.

Governing equations (Boussinesq, hydrostatic, f-plane/beta-plane):

    du/dt = -adv(u,v) + f*v - (1/rho_0)*dp/dx
            + nu_h * lap_h(u) + nu_v * d2u/dz2
            + tau_x / (rho_0 * dz_surface)   [surface]
            - r_bot * u_bottom                [bottom]

    dv/dt = -adv(u,v) - f*u - (1/rho_0)*dp/dy
            + nu_h * lap_h(v) + nu_v * d2v/dz2
            + tau_y / (rho_0 * dz_surface)   [surface]
            - r_bot * v_bottom                [bottom]

    dw/dz = -(du/dx + dv/dy)   [continuity, diagnostic for w]

Where:
  - adv = flux-form pseudospectral advection (dealiased)
  - f = full 2D Coriolis parameter grid.f (includes beta-plane)
  - p = hydrostatic pressure (from pressure.py)
  - Surface wind stress applied as body force in top layer
  - Linear bottom friction at deepest level

The RHS function returns tendencies du/dt, dv/dt for explicit time stepping.
"""
import numpy as np

from config import RHO_0, G_EARTH
from spectral_ops import (
    d_dx, d_dy, laplacian_h, d2_dz2, divergence_h,
    advection_flux_form,
)
from pressure import compute_pressure_gradient


def compute_momentum_tendency(state, grid, physics):
    """
    Compute du/dt and dv/dt for the hydrostatic primitive equations.

    Args:
        state: ModelState (u, v, T, S, eta; p computed if needed)
        grid: OceanGrid (f, z, dz, dx, dy, nz)
        physics: PhysicsConfig (nu_h, nu_v, tau_x, tau_y, r_bot)

    Returns: (dudt, dvdt) each (nx, ny, nz) [m/s^2]
    """
    nz = grid.nz
    z = grid.z   # (nz,) negative downward

    # ── 1. Nonlinear advection (flux form, dealiased) ──
    # advection_flux_form is 2D only, so apply per z-level
    adv_u = np.zeros_like(state.u)
    adv_v = np.zeros_like(state.v)
    for k in range(nz):
        au, av = advection_flux_form(
            state.u[:, :, k], state.v[:, :, k],
            grid.dx, grid.dy, dealias=True
        )
        adv_u[:, :, k] = au
        adv_v[:, :, k] = av

    # ── 2. Coriolis force (full 2D f) ──
    # f shape: (nx, ny) -> broadcast to (nx, ny, nz)
    f_3d = grid.f[:, :, np.newaxis]
    coriolis_u = f_3d * state.v   # +f*v in u-equation
    coriolis_v = -f_3d * state.u  # -f*u in v-equation

    # ── 3. Pressure gradient force ──
    pgf_x, pgf_y = compute_pressure_gradient(state, grid, physics)

    # ── 4. Horizontal diffusion (spectral Laplacian) ──
    diff_h_u = physics.nu_h * laplacian_h(state.u, grid.dx, grid.dy)
    diff_h_v = physics.nu_h * laplacian_h(state.v, grid.dx, grid.dy)

    # ── 5. Vertical diffusion (finite difference on non-uniform z) ──
    diff_v_u = physics.nu_v * d2_dz2(state.u, z)
    diff_v_v = physics.nu_v * d2_dz2(state.v, z)

    # ── 6. Surface wind stress (body force in top layer) ──
    # tau [N/m^2] / (rho_0 * dz_top) -> acceleration [m/s^2]
    # dz_top = thickness of surface layer = |z[0] - z[1]|
    dz_surface = abs(z[0] - z[1])
    wind_u = np.zeros_like(state.u)
    wind_v = np.zeros_like(state.v)
    wind_u[:, :, 0] = physics.tau_x / (RHO_0 * dz_surface)
    wind_v[:, :, 0] = physics.tau_y / (RHO_0 * dz_surface)

    # ── 7. Linear bottom friction (at deepest level) ──
    bot_u = np.zeros_like(state.u)
    bot_v = np.zeros_like(state.v)
    bot_u[:, :, -1] = -physics.r_bot * state.u[:, :, -1]
    bot_v[:, :, -1] = -physics.r_bot * state.v[:, :, -1]

    # ── Sum all tendencies ──
    dudt = adv_u + coriolis_u + pgf_x + diff_h_u + diff_v_u + wind_u + bot_u
    dvdt = adv_v + coriolis_v + pgf_y + diff_h_v + diff_v_v + wind_v + bot_v

    return dudt, dvdt


def compute_vertical_velocity(state, grid):
    """
    Diagnose vertical velocity from horizontal continuity.

    Hydrostatic continuity: dw/dz = -(du/dx + dv/dy)

    Integrate from bottom (w=0 at z=z_bottom) upward:
      w(z_k) = w(z_{k+1}) - integral_{z_k}^{z_{k+1}} (du/dx + dv/dy) dz

    Boundary condition: w = 0 at the bottom (rigid lid approximation).

    Args:
        state: ModelState (u, v)
        grid: OceanGrid

    Returns: w (nx, ny, nz) [m/s], with w[..., -1] = 0
    """
    nz = grid.nz

    # Horizontal divergence at each level
    div_h = divergence_h(state.u, state.v, grid.dx, grid.dy)  # (nx, ny, nz)

    # Integrate from bottom upward
    w = np.zeros_like(state.u)
    # w[..., nz-1] = 0 (bottom boundary condition)
    for k in range(nz - 2, -1, -1):
        # Layer-averaged divergence between z[k] and z[k+1]
        div_avg = 0.5 * (div_h[..., k] + div_h[..., k + 1])
        # w at level k = w at level k+1 - div * dz
        # (dz[k] = thickness between z[k] and z[k+1], positive)
        w[..., k] = w[..., k + 1] - div_avg * grid.dz[k]

    return w
