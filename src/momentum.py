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
from forcing import Forcing


def compute_momentum_tendency(state, grid, physics, forcing=None):
    """
    Compute du/dt and dv/dt for the hydrostatic primitive equations.

    Args:
        state: ModelState (u, v, T, S, eta; p computed if needed)
        grid: OceanGrid (f, z, dz, dx, dy, nz)
        physics: PhysicsConfig (nu_h, nu_v, tau_x, tau_y, r_bot)
        forcing: optional Forcing with 2D (nx, ny) tau_x, tau_y arrays.
            When provided, its fields override the physics scalars; any
            field left None falls back to the physics scalar.

    Returns: (dudt, dvdt) each (nx, ny, nz) [m/s^2]
    """
    nz = grid.nz
    z = grid.z   # (nz,) negative downward

    # ── 1. Nonlinear advection (flux form, dealiased) ──
    # advection_flux_form supports 3D arrays directly (batched over z-axis)
    adv_u, adv_v = advection_flux_form(
        state.u, state.v,
        grid.dx, grid.dy, dealias=True
    )

    # ── 2. Coriolis force (full 2D f) ──
    # f shape: (nx, ny) -> broadcast to (nx, ny, nz)
    f_3d = grid.f[:, :, np.newaxis]
    coriolis_u = f_3d * state.v   # +f*v in u-equation
    coriolis_v = -f_3d * state.u  # -f*u in v-equation

    # ── 3. Pressure gradient force ──
    pgf_x, pgf_y = compute_pressure_gradient(state, grid, physics)

    # ── 4. Horizontal diffusion ──
    # Background constant viscosity (always present)
    diff_h_u = physics.nu_h * laplacian_h(state.u, grid.dx, grid.dy)
    diff_h_v = physics.nu_h * laplacian_h(state.v, grid.dx, grid.dy)

    # Smagorinsky subgrid closure: nu_smg = (Cs * dx)^2 * |D|
    # where |D| is the deformation rate magnitude.
    # Only active when smag_cs > 0.
    if physics.smag_cs > 0:
        dudx = d_dx(state.u, grid.dx)
        dudy = d_dy(state.u, grid.dy)
        dvdx = d_dx(state.v, grid.dx)
        dvdy = d_dy(state.v, grid.dy)
        # Deformation rate: |D| = sqrt(2 * (dudx - dvdy)^2 + 2 * (dudy + dvdx)^2) / 2
        # Simplified: |D| = sqrt((dudx - dvdy)^2 + (dudy + dvdx)^2)
        strain = np.sqrt((dudx - dvdy) ** 2 + (dudy + dvdx) ** 2)
        dx = grid.dx
        nu_smg = (physics.smag_cs * dx) ** 2 * strain
        # Apply as Laplacian with spatially-varying viscosity:
        # d/dx(nu_smg * du/dx) + d/dy(nu_smg * du/dy)
        # For spectral solver, approximate as nu_smg * lap_h(u) (local scaling)
        diff_h_u = diff_h_u + nu_smg * laplacian_h(state.u, grid.dx, grid.dy)
        diff_h_v = diff_h_v + nu_smg * laplacian_h(state.v, grid.dx, grid.dy)

    # ── 5. Vertical diffusion (finite difference on non-uniform z) ──
    diff_v_u = physics.nu_v * d2_dz2(state.u, z)
    diff_v_v = physics.nu_v * d2_dz2(state.v, z)

    # ── 6. Surface wind stress (body force in top layer) ──
    # tau [N/m^2] / (rho_0 * dz_top) -> acceleration [m/s^2]
    # dz_top = thickness of surface layer = |z[0] - z[1]|
    # tau may be a 2D (nx, ny) field (from forcing) or a scalar (physics);
    # numpy broadcasting handles either when assigning into the top layer.
    dz_surface = abs(z[0] - z[1])
    tau_x = physics.tau_x if forcing is None or forcing.tau_x is None else forcing.tau_x
    tau_y = physics.tau_y if forcing is None or forcing.tau_y is None else forcing.tau_y
    wind_u = np.zeros_like(state.u)
    wind_v = np.zeros_like(state.v)
    wind_u[:, :, 0] = tau_x / (RHO_0 * dz_surface)
    wind_v[:, :, 0] = tau_y / (RHO_0 * dz_surface)

    # ── 7. Bottom friction ──
    bot_u = np.zeros_like(state.u)
    bot_v = np.zeros_like(state.v)
    if physics.bottom_friction == 'quadratic':
        u_bot = state.u[:, :, -1]
        v_bot = state.v[:, :, -1]
        speed = np.sqrt(u_bot ** 2 + v_bot ** 2)
        bot_u[:, :, -1] = -physics.cd * speed * u_bot
        bot_v[:, :, -1] = -physics.cd * speed * v_bot
    else:
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
    # Horizontal divergence at each level
    div_h = divergence_h(state.u, state.v, grid.dx, grid.dy)  # (nx, ny, nz)

    # Layer-averaged divergence between z[k] and z[k+1]
    div_avg = 0.5 * (div_h[..., :-1] + div_h[..., 1:])  # (nx, ny, nz-1)

    # Weighted by layer thickness, then reverse-cumsum from bottom
    # w[k] = w[k+1] - div_avg[k]*dz[k],  w[bottom]=0
    #  =>  w[k] = -sum_{j=k}^{nz-2} div_avg[j]*dz[j]
    dz = grid.dz.reshape([1] * (div_avg.ndim - 1) + [-1])
    integrand = div_avg * dz  # (nx, ny, nz-1)

    w = np.zeros_like(state.u)
    # Reverse, cumsum, reverse back — gives cumulative sum from each level to bottom
    w[..., :-1] = -np.cumsum(integrand[..., ::-1], axis=-1)[..., ::-1]

    return w
