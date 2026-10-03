"""Nonuniform vertical stencils and no-flux interface balances."""
from ocean_solver.numerics.backend import jnp


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
