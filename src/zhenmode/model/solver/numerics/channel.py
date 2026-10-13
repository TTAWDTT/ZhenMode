"""Paired fourth-order pressure and flux operators for flat Cartesian channels.

Scalar ghosts reflect evenly, normal velocity ghosts oddly. The face flux is
7/12 of the adjacent pair minus 1/12 of the next pair. Its difference is the
centred fourth-order derivative; the scalar gradient is its negative transpose.
No coastal gates or spherical metric approximation belong to this module.
"""

from zhenmode.model.solver.numerics.backend import jnp
from zhenmode.model.solver.numerics.column import point_mass_integral
from zhenmode.model.solver.numerics.vertical import fourth_uniform_tracer_gradient


def _reflected_latitude(field, *, normal):
    padding = [(0, 0)] * field.ndim
    padding[1] = (2, 2)
    padded = jnp.pad(field, padding, mode='symmetric')
    if normal:
        padded = padded.at[:, :2].multiply(-1.)
        padded = padded.at[:, -2:].multiply(-1.)
    return padded


def channel_face_velocities(u, v):
    """Positive face velocities, including an explicitly closed north face."""
    fx = (7 * (u + jnp.roll(u, -1, axis=0))
          - jnp.roll(u, 1, axis=0) - jnp.roll(u, -2, axis=0)) / 12
    padded = _reflected_latitude(v, normal=True)
    fy = (7 * (padded[:, 2:-2] + padded[:, 3:-1])
          - padded[:, 1:-3] - padded[:, 4:]) / 12
    return fx, fy.at[:, -1].set(0.)


def channel_divergence(u, v, params):
    """Flux difference; the south incoming face is also exactly zero."""
    fx, fy = channel_face_velocities(u, v)
    incoming = jnp.roll(fy, 1, axis=1).at[:, 0].set(0.)
    inverse_dx = params.inv_dx[..., :1] if u.ndim == 3 else params.inv_dx[..., 0]
    return ((fx - jnp.roll(fx, 1, axis=0)) * inverse_dx
            + (fy - incoming) * params.inv_dy)


def channel_gradient(field, params):
    """Negative adjoint of channel_divergence on uniform cell weights."""
    inverse_dx = params.inv_dx[..., :1] if field.ndim == 3 else params.inv_dx[..., 0]
    gx = (8 * (jnp.roll(field, -1, axis=0) - jnp.roll(field, 1, axis=0))
          - (jnp.roll(field, -2, axis=0) - jnp.roll(field, 2, axis=0))) * inverse_dx / 12
    padded = _reflected_latitude(field, normal=False)
    gy = (8 * (padded[:, 3:-1] - padded[:, 1:-3])
          - (padded[:, 4:] - padded[:, :-4])) * params.inv_dy / 12
    return gx, gy


def channel_tracer_fluxes(tracer, u, v, params, face_transport=None):
    """Reconstruct nodal products; keep supplied volume corrections conservative.

    The base flux differentiates u*T and v*T with the same wider stencil as
    continuity. Supplied extra transport carries a centred concentration, so
    its order is separate; constant tracer still follows that actual volume
    flux. Signed interpolation does not provide a monotonicity guarantee.
    """
    fx, fy = channel_face_velocities(u * tracer, v * tracer)
    if face_transport is not None:
        base_x, base_y = channel_face_velocities(u, v)
        concentration_x = .5 * (tracer + jnp.roll(tracer, -1, axis=0))
        concentration_y = .5 * (tracer + jnp.roll(tracer, -1, axis=1))
        fx = fx + (face_transport[0] / params.dz_node - base_x) * concentration_x
        fy = fy + (face_transport[1] / params.dz_node - base_y) * concentration_y
    return fx, fy.at[:, -1].set(0.)


def channel_paired_product_fluxes(tracer, u, v, params, face_transport):
    """Integrated scalar faces and matching moments, owned by actual mass.

    ``tracer`` already has the dispatch owner's common reference subtracted.
    Each signed stencil mass and the actual-mass residual is gated before
    both its scalar product and vertical moment. Exact zero actual mass is
    therefore a horizontal material barrier. This changes the product stencil
    at those barriers; fourth order there or for the complete rate is not
    claimed. Original vertical quadrature compatibility residuals remain.
    """
    if face_transport is None:
        raise ValueError('paired_product_fourth requires actual mass faces')
    shape = getattr(tracer, 'shape', ())
    if len(shape) != 3 or min(shape[:2]) < 5 or shape[-1] < 6:
        raise ValueError('paired_product_fourth requires a full (nx, ny, nz) tracer array')
    if len(face_transport) != 2:
        raise ValueError('paired_product_fourth requires both actual mass faces')
    fields = (('tracer', tracer, shape), ('u', u, shape), ('v', v, shape),
              ('Mx', face_transport[0], shape), ('My', face_transport[1], shape),
              ('dz_node', params.dz_node, (1, 1, shape[-1])),
              ('dz_iface', params.dz_iface, (1, 1, shape[-1]-1)))
    for name, field, expected_shape in fields:
        if (getattr(field, 'shape', None) != expected_shape
                or getattr(field, 'dtype', None) != jnp.dtype('float64')):
            raise ValueError(f'paired_product_fourth {name} requires shape {expected_shape} and float64')
    h = params.dz_node
    nominal = channel_face_velocities(u, v)
    padded_q = _reflected_latitude(tracer, normal=False)
    padded_v = _reflected_latitude(v, normal=True)
    scalar_nodes = (
        (jnp.roll(tracer, 1, axis=0), tracer, jnp.roll(tracer, -1, axis=0),
         jnp.roll(tracer, -2, axis=0)),
        (padded_q[:, 1:-3], padded_q[:, 2:-2], padded_q[:, 3:-1], padded_q[:, 4:]),
    )
    velocity_nodes = (
        (jnp.roll(u, 1, axis=0), u, jnp.roll(u, -1, axis=0), jnp.roll(u, -2, axis=0)),
        (padded_v[:, 1:-3], padded_v[:, 2:-2], padded_v[:, 3:-1], padded_v[:, 4:]),
    )

    def moment(mass, values):
        gradient = fourth_uniform_tracer_gradient(values, params)
        primitive = point_mass_integral(mass, params)
        return (values[..., 0]*jnp.sum(mass, axis=-1)
                + jnp.sum(h*gradient*primitive, axis=-1))

    faces, moments = [], []
    for axis, mass, base_velocity, q_nodes, u_nodes in zip(
            (0, 1), face_transport, nominal, scalar_nodes, velocity_nodes, strict=True):
        active = mass != 0.
        if axis == 1:
            active = active.at[:, -1].set(False)
        face_q = .5*(q_nodes[1]+q_nodes[2])
        residual = jnp.where(active, mass-h*base_velocity, 0.)
        flux = residual*face_q
        column = moment(residual, face_q)
        for coefficient, velocity, values in zip((-1., 7., 7., -1.), u_nodes, q_nodes,
                                                   strict=True):
            constituent = jnp.where(active, coefficient*h*velocity/12., 0.)
            flux = flux+constituent*values
            column = column+moment(constituent, values)
        if axis == 1:
            flux = flux.at[:, -1].set(0.)
            column = column.at[:, -1].set(0.)
        faces.append(flux)
        moments.append(column)
    # An externally supplied wall mass is incompatible with this closed channel.
    # Propagate an explicit invalid result under jit instead of discarding it.
    wall_valid = jnp.all(face_transport[1][:, -1] == 0.)
    return (tuple(jnp.where(wall_valid, face, jnp.nan) for face in faces),
            tuple(jnp.where(wall_valid, column, jnp.nan) for column in moments))
