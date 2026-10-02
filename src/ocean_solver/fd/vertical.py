"""Mechanically preserved FD vertical implementation."""
from .backend import jnp
from .eos import _density_anomaly


def _fill_ghost_bottom(u, p):
    """Seafloor no-flux fill: extend each column's bottom wet value downward.

    Ghost layers below the seafloor hold the T_ref/S_ref sentinel and never evolve,
    so any vertical stencil reaching them mixes that reservoir into the bottom wet
    layer. Replicating the bottom wet value is the standard MOM/NEMO no-flux
    treatment. Applied by every vertical operator. (D8)
    """
    wet = p.wet_mask_z
    idx = (wet > 0.5) * jnp.arange(u.shape[-1])
    kbot = jnp.max(idx, axis=-1)[..., None]        # (nx, ny, 1)
    kk = jnp.arange(u.shape[-1])[None, None, :]
    u_bot = jnp.take_along_axis(u, kbot, axis=-1)  # bottom wet value
    return jnp.where(kk >= kbot, u_bot, u)

def _d_dz(u, p):
    """Vertical first derivative, non-uniform grid."""
    du_interior = (u[..., 2:] - u[..., :-2]) / p.dz_denom_interior
    du_top = (u[..., 1:2] - u[..., 0:1]) / p.dz_bnd_top
    du_bot = (u[..., -1:] - u[..., -2:-1]) / p.dz_bnd_bot
    return jnp.concatenate([du_top, du_interior, du_bot], axis=-1)

def _d2_dz2(u, p):
    """Vertical second derivative, non-uniform grid.

    Top and bottom nodes use the zero-flux (ghost-point) form 2*(C1 - C0)/h0^2: the
    mirror ghost Cg = C1 makes the centered curvature diffusive at the boundary,
    where the plain centered-curvature form was anti-diffusive and drove the ITCZ
    salinity runaway. The field is ghost-filled first (_fill_ghost_bottom), so a
    seafloor that is not at the last grid level is a no-flux boundary too: without
    the fill the bottom wet node reads the T_ref/S_ref sentinel below the floor, a
    spurious seafloor flux up to 0.05 K/day. (D20)
    """
    u = _fill_ghost_bottom(u, p)
    d2u_interior = (
        u[..., 2:] * p.d2z_hm + u[..., :-2] * p.d2z_hp
        - u[..., 1:-1] * (p.d2z_hm + p.d2z_hp)
    ) / p.d2z_denom
    d2u_top = 2.0 * (u[..., 1:2] - u[..., 0:1]) / (p.d2z_h0_top ** 2)
    d2u_bot = 2.0 * (u[..., -2:-1] - u[..., -1:]) / (p.d2z_h0_bot ** 2)
    return jnp.concatenate([d2u_top, d2u_interior, d2u_bot], axis=-1)

def _iface_flux_divergence(Fz_i, p):
    """Divergence of an interface flux: tend[k] = (F[k-1/2] - F[k+1/2])/dz_node[k].

    Fz_i holds one flux per INTERFACE between nodes k and k+1 (length nz-1);
    the material top and bottom carry F = 0, which is the zero-flux BC. The
    column sum of tend*dz_node telescopes to F[bot]-F[top] = 0 exactly, for
    any flux and any mask (D9, D10, D18).
    """
    up = jnp.concatenate([jnp.zeros_like(Fz_i[..., :1]), Fz_i], axis=-1)
    dn = jnp.concatenate([Fz_i, jnp.zeros_like(Fz_i[..., :1])], axis=-1)
    return (up - dn) / p.dz_node

def _d2_dz2_flux(tracer, kappa, p):
    """Vertical diffusion in INTERFACE-flux form (exactly column-conservative).

    F[k+1/2] = -kappa*(C[k+1]-C[k])/dz_iface[k], zero at the material top and
    bottom; the column sum of tend*dz_node telescopes to F[bot]-F[top] = 0 for any
    kappa and any field. The node form does not telescope on a non-uniform grid, and
    vertical diffusion can only redistribute within a column. (D9)
    """
    # kappa may be scalar or a (nx, ny, nz) coastal-enhanced field; only the
    # scalar-zero shortcut can be branched on at trace time.
    if getattr(kappa, "ndim", 0) == 0 and kappa == 0.0:
        return jnp.zeros_like(tracer)
    tracer = _fill_ghost_bottom(tracer, p)
    Cm = tracer[..., :-1]
    Cp = tracer[..., 1:]
    Fz_i = -kappa * (Cp - Cm) / p.dz_iface                # (nx, ny, nz-1)
    wet_iface_f = (p.wet_mask_z[..., :-1] > 0.5) & (p.wet_mask_z[..., 1:] > 0.5)
    Fz_i = jnp.where(wet_iface_f, Fz_i, 0.0)
    return _iface_flux_divergence(Fz_i, p) * p.wet_mask_z

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
    # Seafloor no-flux fill (same T_ref=15 ghost guard as the Redi closure).
    tracer = _fill_ghost_bottom(tracer, p)
    Cm = tracer[..., :-1]                        # upper node value
    Cp = tracer[..., 1:]                         # lower node value
    Fz_i = -kappa * (Cp - Cm) / p.dz_iface       # (nx, ny, nz-1)
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
