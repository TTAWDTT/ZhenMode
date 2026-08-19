"""
Equation of State — linearized Boussinesq approximation.

  rho = rho_0 * [1 - alpha * (T - T_ref) + beta * (S - S_ref)]

This is the simplest nontrivial EOS. It captures the dominant density
effects of temperature (warm = lighter) and salinity (salty = denser)
without the full UNESCO polynomial. Sufficient for upper-ocean dynamics
at v0.1 resolution.

For the Boussinesq approximation, density only enters through the
buoyancy term in the momentum equation and the hydrostatic pressure
computation. The inertia term uses rho_0 everywhere.
"""
import numpy as np

from config import RHO_0, ALPHA_T, BETA_S


def compute_density(T, S, physics):
    """
    Linearized equation of state.

    Args:
        T: temperature field [C], shape (nx, ny, nz) or (nx, ny)
        S: salinity field [psu], same shape as T
        physics: PhysicsConfig (uses T_ref, S_ref)

    Returns: density [kg/m^3], same shape as T
    """
    return RHO_0 * (
        1.0 - ALPHA_T * (T - physics.T_ref) + BETA_S * (S - physics.S_ref)
    )


def compute_buoyancy(T, S, physics):
    """
    Buoyancy perturbation: b = -g * (rho - rho_0) / rho_0

    Equivalent to: b = g * [alpha*(T-T_ref) - beta*(S-S_ref)]

    Args:
        T: temperature [C]
        S: salinity [psu]
        physics: PhysicsConfig

    Returns: buoyancy [m/s^2], same shape as T
    """
    from config import G_EARTH
    return G_EARTH * (
        ALPHA_T * (T - physics.T_ref) - BETA_S * (S - physics.S_ref)
    )


def density_anomaly(T, S, physics):
    """rho' = rho - rho_0 [kg/m^3]. Convenience for pressure computation."""
    return RHO_0 * (
        -ALPHA_T * (T - physics.T_ref) + BETA_S * (S - physics.S_ref)
    )
