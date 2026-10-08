"""Reference-cell contacts, including adjacent-node overlaps at bottom steps.

Fields remain values at fixed FD nodes. These balances use their declared
reference capacities; convergence of the point reconstruction is a separate
qualification from the exact geometric face closure.
"""

from zhenmode.model.solver.numerics.backend import jnp

OFFSETS = (0, 1, -1)


def shift_vertical(values, offset):
    """Return a[k+offset] without a vertical periodic seam."""
    if offset == 1:
        return jnp.concatenate((values[..., 1:], jnp.zeros_like(values[..., :1])), axis=-1)
    if offset == -1:
        return jnp.concatenate((jnp.zeros_like(values[..., :1]), values[..., :-1]), axis=-1)
    return values


def neighbours(values, axis):
    following = jnp.roll(values, -1, axis=axis)
    return jnp.stack([shift_vertical(following, s) for s in OFFSETS])


def incoming(fluxes, axis):
    """Scatter positive-face quantities onto each recipient node."""
    prior = jnp.roll(fluxes, 1, axis=axis + 1)
    if axis == 1:
        prior = prior.at[:, :, 0].set(0.0)
    return sum(shift_vertical(prior[k], -s) for k, s in enumerate(OFFSETS))


def recipients(fluxes):
    """Recipient-node alignment on the same face, without horizontal rolling."""
    return sum(shift_vertical(fluxes[k], -s) for k, s in enumerate(OFFSETS))


def contact_transports(u, v, p):
    u, v = (jnp.broadcast_to(a, p.dz_node.shape) for a in (u, v))
    cosine = 0.5 * (p.cos_lat + jnp.roll(p.cos_lat, -1))
    x = 0.5 * (u[None] + neighbours(u, 0)) * p.face_contacts[0]
    y = 0.5 * (v[None] + neighbours(v, 1)) * p.face_contacts[1] * cosine[None, None, :, None]
    return x, y


def contact_divergence(x, y, p):
    """Physical face flux differences per area, before division by cell width."""
    return (jnp.sum(x, axis=0) - incoming(x, 0)) * p.inv_dx + (
        jnp.sum(y, axis=0) - incoming(y, 1)
    ) * p.inv_dy / p.cos_lat[None, :, None]


def gradient_from_differences(delta_x, delta_y, p):
    """Scatter the half-face force to both momentum nodes in the pair."""
    x = delta_x * p.face_contacts[0]
    cosine = 0.5 * (p.cos_lat + jnp.roll(p.cos_lat, -1))
    y = delta_y * p.face_contacts[1] * cosine[None, None, :, None]
    return (
        0.5 * (jnp.sum(x, axis=0) + incoming(x, 0)) * p.inv_dx / p.dz_node,
        0.5
        * (jnp.sum(y, axis=0) + incoming(y, 1))
        * p.inv_dy
        / p.cos_lat[None, :, None]
        / p.dz_node,
    )


def contact_gradient(field, p):
    field = jnp.broadcast_to(field, p.dz_node.shape)
    return gradient_from_differences(
        neighbours(field, 0) - field[None], neighbours(field, 1) - field[None], p
    )


def contact_diffusion(field, diffusivity, p):
    """Symmetric, dissipative contact exchange of the nodal reference values."""
    cosine = 0.5 * (p.cos_lat + jnp.roll(p.cos_lat, -1))
    x = (
        0.5
        * (diffusivity[None] + neighbours(diffusivity, 0))
        * (neighbours(field, 0) - field[None])
        * p.inv_dx[None]
        * p.face_contacts[0]
    )
    y = (
        0.5
        * (diffusivity[None] + neighbours(diffusivity, 1))
        * (neighbours(field, 1) - field[None])
        * p.inv_dy
        * p.face_contacts[1]
        * cosine[None, None, :, None]
    )
    return contact_divergence(x, y, p) / p.dz_node


def contact_advection(field, faces, p):
    """Conservative donor exchange on every geometric contact, without dry nodes."""
    x, y = faces
    tracer_x = jnp.where(x >= 0.0, field[None], neighbours(field, 0))
    tracer_y = jnp.where(y >= 0.0, field[None], neighbours(field, 1))
    return contact_divergence(x * tracer_x, y * tracer_y, p)


def contact_material_derivative(field, faces, p):
    """Horizontal advective derivative on the actual velocity-contact fluxes.

    Equivalent to D(F*mean(field))-field*D(F), evaluated via differences to
    preserve constants without subtracting two large flux divergences. The
    existing nodal vertical derivative and dealias filter are separate stages;
    this does not claim full momentum/energy conservation or reconstruction
    accuracy for horizontally uniform vertical shear over stepped bottoms.
    """
    differences = [
        0.5 * flux * (neighbours(field, axis) - field[None]) for axis, flux in enumerate(faces)
    ]
    x, y = differences
    return (
        (jnp.sum(x, axis=0) + incoming(x, 0)) * p.inv_dx
        + (jnp.sum(y, axis=0) + incoming(y, 1)) * p.inv_dy / p.cos_lat[None, :, None]
    ) / p.dz_node
