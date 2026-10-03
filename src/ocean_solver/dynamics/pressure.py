"""Mechanically preserved FD pressure implementation."""
from ocean_solver.config.definitions import G_EARTH, RHO_0
from ocean_solver.numerics.backend import jnp
from ocean_solver.numerics.horizontal import _gradient_conservative_3d
from ocean_solver.physics.eos import _density_anomaly


def _compute_hydrostatic_pressure(state, p):
    """Full hydrostatic pressure via cumulative trapezoidal integration.

    The density anomaly is masked by wet_mask_z (zeroed below the seafloor) BEFORE
    integration, so ghost layers contribute dp = 0 and the pressure stays constant
    beneath the bottom: no spurious horizontal gradient from columns of different
    ghost-water length. (D11)
    """
    rho_prime = _density_anomaly(state.T, state.S, p)
    rho_prime = rho_prime * p.wet_mask_z          # zero out ghost water
    rho_avg = 0.5 * (rho_prime[..., :-1] + rho_prime[..., 1:])
    dp = G_EARTH * rho_avg * p.dz_3d
    p_bc = jnp.zeros_like(state.T)
    p_bc = p_bc.at[..., 1:].set(jnp.cumsum(dp, axis=-1))
    p_bt = RHO_0 * G_EARTH * state.eta[:, :, None]
    return p_bt + p_bc

def _compute_pressure_gradient(state, p):
    """Horizontal pressure gradient force per unit mass (FD).

    Uses _gradient_conservative_3d (masked, cos(lat)-weighted adjoint), NOT the
    bare centered _d_dx/_d_dy, so the 3D PGF does not reach into land zeros at
    coastlines (D4/D11 in docs/decisions.md).
    """
    pressure = _compute_hydrostatic_pressure(state, p)
    pgf_x, pgf_y = _gradient_conservative_3d(pressure, p)
    return -pgf_x / RHO_0, -pgf_y / RHO_0

def _reference_depth_gradient(eta, p):
    """Negative adjoint of depth divergence in the A * H_ref velocity norm."""
    gradient_x, gradient_y = _gradient_conservative_3d(eta[..., None], p)
    return (jnp.sum(gradient_x * p.dz_norm, axis=-1),
            jnp.sum(gradient_y * p.dz_norm, axis=-1))

def _compute_bt_rho_pgf(state, p):
    """Barotropic (depth-averaged) PGF from density anomalies (FD).

    Depth average over the WET column of the same face-gated 3D baroclinic PGF the
    3D momentum feels (_gradient_conservative_3d), normalized by H_sw to match
    ubt = transport / H_sw in _barotropic_velocity:

        F_rho = (1/H_sw) * SUM_k pgf3d_layer_k * dz_k * wet_iface_k

    No ghost water and no below-bottom constant enters, so the spurious isobath
    forcing of the old trapezoidal-over-all-layers form is gone; what remains is
    the physical wet-column (JEBAR-type) coupling. (D26)
    """
    if p.column_geometry == 'nodal_dual_v1':
        acceleration_x, acceleration_y = _compute_pressure_gradient(state._replace(eta=jnp.zeros_like(state.eta)), p)
        return (jnp.sum(acceleration_x * p.dz_norm, axis=-1),
                jnp.sum(acceleration_y * p.dz_norm, axis=-1))
    rho_prime = _density_anomaly(state.T, state.S, p)
    rho_prime = rho_prime * p.wet_mask_z          # zero out ghost water
    rho_avg = 0.5 * (rho_prime[..., :-1] + rho_prime[..., 1:])
    dp = G_EARTH * rho_avg * p.dz_3d
    p_bc = jnp.zeros_like(state.T)
    p_bc = p_bc.at[..., 1:].set(jnp.cumsum(dp, axis=-1))
    # Layer-centered, face-gated 3D PGF, transport-weighted over the wet column.
    gx3, gy3 = _gradient_conservative_3d(p_bc, p)
    pgf_x_lay = 0.5 * (gx3[..., :-1] + gx3[..., 1:])
    pgf_y_lay = 0.5 * (gy3[..., :-1] + gy3[..., 1:])
    wet_iface = p.wet_mask_z[..., :-1] * p.wet_mask_z[..., 1:]
    fx = -jnp.sum(pgf_x_lay * wet_iface * p.dz_3d, axis=-1) / (RHO_0 * p.H_sw)
    fy = -jnp.sum(pgf_y_lay * wet_iface * p.dz_3d, axis=-1) / (RHO_0 * p.H_sw)
    return fx, fy
