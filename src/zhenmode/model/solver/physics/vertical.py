"""Vertical mixing coefficients, diffusion selection and convection gates."""

from zhenmode.model.solver.numerics.backend import jnp
from zhenmode.model.solver.numerics.vertical import (
    _d2_dz2,
    _d2_dz2_flux,
    _diffusive_interface_flux,
    _iface_flux_divergence,
)
from zhenmode.model.solver.physics.eos import _density_anomaly


def _conv_flux_tendency(tracer, conv_mask_3d, kappa, p, iface_gate=None):
    """Convective mixing in INTERFACE-flux form (exactly column-conservative).

    Reuses the GM/Redi interface discretization, so the column sum telescopes to
    F[bot]-F[top] = 0 for ANY mask and the operator is negative-semidefinite. The
    node form (kappa_conv*mask*_d2_dz2) does not telescope, so every convective
    episode created net heat, polar-concentrated. (D10)

    Args: tracer (nx, ny, nz); conv_mask_3d (nx, ny, 1) column gate; kappa [m^2/s]
    (p.kappa_conv); optional iface_gate (nx, ny, nz-1) PER-INTERFACE gate, which
    makes mixing need both gates (localized convection).
    """
    if kappa <= 0.0:
        return jnp.zeros_like(tracer)
    Fz_i = _diffusive_interface_flux(tracer, kappa, p)
    # Flux only across wet interfaces AND only in convecting columns.
    wet_iface_f = (p.wet_mask_z[..., :-1] > 0.5) & (p.wet_mask_z[..., 1:] > 0.5)
    gate = (jnp.asarray(conv_mask_3d)[..., :1] > 0.5)   # force (nx, ny, 1)
    allow = gate & wet_iface_f
    if iface_gate is not None:
        allow = allow & (jnp.asarray(iface_gate) > 0.5)
    Fz_i = jnp.where(allow, Fz_i, 0.0)
    # Padded flux array carries the zero-flux BC at surface/seafloor;
    # tend[k] = (F[k-1/2] - F[k+1/2]) / dz_node[k] (see _redi_skew_flux_tendency).
    return _iface_flux_divergence(Fz_i, p) * p.wet_mask_z


def _effective_kappa_v(p):
    """Return the effective vertical diffusivity including the coastal band."""
    return p.kappa_v + p.coastal_kappa_v_2d[:, :, None]


def _vertical_diffusion(tracer, kappa, p):
    """Vertical diffusion of one tracer.

    `conservative_kv` selects the exactly-column-conservative interface-flux
    form; the default keeps the legacy node form bit-exact (D9).
    """
    if p.conservative_kv:
        return _d2_dz2_flux(tracer, kappa, p)
    return kappa * _d2_dz2(tracer, p)


def _vertical_momentum_diffusion(velocity, p):
    """Use the reference-cell flux balance for the nodal geometry candidate."""
    if p.column_geometry == 'nodal_dual_v1':
        return _d2_dz2_flux(velocity, p.nu_v, p)
    return p.nu_v * _d2_dz2(velocity, p)


def _convective_mask(state, p):
    """Convection gates from the density profile: (column, per-interface).

    An interface is unstable when the layer above is denser than the layer
    below AND both layers are wet. The both-wet guard is essential: the
    bottommost wet layer is denser than the masked ghost below it, so without
    it every wet/ghost interface tests unstable and convects the whole column
    (D10).
    """
    rho_prime = _density_anomaly(state.T, state.S, p) * p.wet_mask_z
    wet_iface = (p.wet_mask_z[..., :-1] > 0.5) & (p.wet_mask_z[..., 1:] > 0.5)
    unstable_iface = (rho_prime[..., :-1] > rho_prime[..., 1:]) & wet_iface
    return jnp.any(unstable_iface, axis=-1, keepdims=True), unstable_iface
