"""
Tracer Equations — temperature and salinity transport.

Governing equations (Boussinesq, advective form):

    dT/dt = -adv_h(T, u, v) + kappa_h * lap_h(T) + kappa_v * d2T/dz2
            + Q_heat / (rho_0 * C_P * dz_surface)   [surface layer only]

    dS/dt = -adv_h(S, u, v) + kappa_h * lap_h(S) + kappa_v * d2S/dz2

Where:
  - adv_h = flux-form pseudoscalar advection (dealiased, conservative)
  - lap_h = spectral horizontal Laplacian
  - d2/dz2 = finite-difference vertical second derivative (non-uniform z)
  - Surface heat flux Q_heat enters as a warming rate in the top layer
  - No surface salinity flux in v0.1

The RHS function returns tendencies dT/dt, dS/dt for explicit time stepping.
"""
import numpy as np

from config import RHO_0, C_P
from spectral_ops import (
    laplacian_h, d2_dz2,
    advection_scalar,
)


def compute_tracer_tendency(state, grid, physics):
    """
    Compute dT/dt and dS/dt for the tracer transport equations.

    Args:
        state: ModelState (u, v, T, S)
        grid: OceanGrid (z, dz, dx, dy)
        physics: PhysicsConfig (kappa_h, kappa_v, Q_heat)

    Returns: (dTdt, dSdt) each (nx, ny, nz)
             dTdt [C/s], dSdt [psu/s]
    """
    z = grid.z   # (nz,) negative downward

    # ── 1. Horizontal advection (flux form, dealiased, 3D) ──
    adv_T = advection_scalar(state.T, state.u, state.v, grid.dx, grid.dy)
    adv_S = advection_scalar(state.S, state.u, state.v, grid.dx, grid.dy)

    # ── 2. Horizontal diffusion (spectral Laplacian) ──
    diff_h_T = physics.kappa_h * laplacian_h(state.T, grid.dx, grid.dy)
    diff_h_S = physics.kappa_h * laplacian_h(state.S, grid.dx, grid.dy)

    # ── 3. Vertical diffusion (finite difference on non-uniform z) ──
    diff_v_T = physics.kappa_v * d2_dz2(state.T, z)
    diff_v_S = physics.kappa_v * d2_dz2(state.S, z)

    # ── 4. Surface heat flux (body forcing in top layer) ──
    # Q_heat [W/m^2] / (rho_0 * C_P * dz_top) -> warming rate [C/s]
    # dz_top = thickness of surface layer = |z[0] - z[1]|
    dz_surface = abs(z[0] - z[1])
    heat_T = np.zeros_like(state.T)
    heat_T[:, :, 0] = physics.Q_heat / (RHO_0 * C_P * dz_surface)

    # ── Sum all tendencies ──
    dTdt = adv_T + diff_h_T + diff_v_T + heat_T
    dSdt = adv_S + diff_h_S + diff_v_S

    return dTdt, dSdt
