"""Paired fourth-order pressure and flux operators for flat Cartesian channels.

Scalar ghosts reflect evenly, normal velocity ghosts oddly. The face flux is
7/12 of the adjacent pair minus 1/12 of the next pair. Its difference is the
centred fourth-order derivative; the scalar gradient is its negative transpose.
No coastal gates or spherical metric approximation belong to this module.
"""

from zhenmode.model.solver.numerics.backend import jnp


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
