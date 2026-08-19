"""
Hydrostatic Pressure Module — computes physical pressure from density and SSH.

The hydrostatic primitive equations assume vertical momentum balance is purely
hydrostatic:

    dp/dz = -rho * g

Integrating from the free surface (z = eta) downward to depth z:

    p(z) = rho_0 * g * eta + integral_{z}^{0} rho * g * dz'

The first term is the pressure due to sea surface height (barotropic contribution).
The second term is the pressure due to density variations (baroclinic contribution),
computed via cumulative trapezoidal integration over the vertical grid.

Horizontal pressure gradients drive the momentum equation:

    -1/rho_0 * dp/dx = -g * d(eta)/dx - 1/rho_0 * d/dx [integral rho' g dz']
    -1/rho_0 * dp/dy = -g * d(eta)/dy - 1/rho_0 * d/dy [integral rho' g dz']

where rho' = rho - rho_0 is the density anomaly (Boussinesq).
"""
import numpy as np

from config import RHO_0, G_EARTH
from spectral_ops import d_dx, d_dy
from eos import density_anomaly


def compute_hydrostatic_pressure(state, grid, physics):
    """
    Compute full hydrostatic pressure on all z-levels.

    p(z_k) = rho_0 * g * eta + g * integral_{z_k}^{0} rho' * dz'

    The integral uses cumulative trapezoidal integration over the non-uniform
    z-grid. Since z is negative downward, integrating from z_k up to 0 means
    accumulating from the surface (k=0) downward.

    Args:
        state: ModelState (needs T, S, eta)
        grid: OceanGrid (needs z, dz)
        physics: PhysicsConfig

    Returns: pressure array (nx, ny, nz) [Pa]
    """
    # Density anomaly rho' = rho - rho_0
    rho_prime = density_anomaly(state.T, state.S, physics)  # (nx, ny, nz)

    # Baroclinic pressure: integrate rho' * g * dz from surface downward
    # p_bc(z_k) = g * sum_{j=0}^{k-1} 0.5*(rho'[j] + rho'[j+1]) * dz[j]
    # where dz[j] = |z[j+1] - z[j]| (positive thickness)
    nz = grid.nz
    p_bc = np.zeros_like(state.T)  # (nx, ny, nz)

    # Trapezoidal integration: layer by layer
    for k in range(nz - 1):
        # Average density anomaly across the layer between z[k] and z[k+1]
        rho_avg = 0.5 * (rho_prime[..., k] + rho_prime[..., k + 1])
        # Pressure contribution from this layer
        dp = G_EARTH * rho_avg * grid.dz[k]
        # Accumulate: p_bc at level k+1 includes all layers above it
        p_bc[..., k + 1] = p_bc[..., k] + dp

    # Barotropic pressure: rho_0 * g * eta (same at all depths)
    p_bt = RHO_0 * G_EARTH * state.eta  # (nx, ny)
    p_bt_3d = p_bt[:, :, np.newaxis]    # broadcast to (nx, ny, nz)

    # Total pressure
    return p_bt_3d + p_bc


def compute_pressure_gradient(state, grid, physics):
    """
    Compute horizontal pressure gradient force per unit mass.

    -1/rho_0 * dp/dx  and  -1/rho_0 * dp/dy

    This is the driving force in the momentum equation. Uses spectral
    derivatives for accuracy.

    Args:
        state: ModelState (will compute p if not present)
        grid: OceanGrid
        physics: PhysicsConfig

    Returns: (dpdx, dpdy) each (nx, ny, nz) [m/s^2]
    """
    if state.p is None:
        state.p = compute_hydrostatic_pressure(state, grid, physics)

    # Pressure gradient: spectral derivatives in x and y
    dpdx = d_dx(state.p, grid.dx)   # (nx, ny, nz)
    dpdy = d_dy(state.p, grid.dy)   # (nx, ny, nz)

    # Force per unit mass = -1/rho_0 * grad(p)
    return -dpdx / RHO_0, -dpdy / RHO_0
