"""Restricted physical-node column quadrature, separate from cell capacities."""

from zhenmode.model.solver.numerics.backend import jnp, np


def cubic_uniform_mean_weights(nodes):
    """Accumulate integrals of four-node cubics over all uniform intervals."""
    weights = np.zeros(nodes)
    weights[:4] += np.array([9., 19., -5., 1.]) / 24
    for interval in range(1, nodes-2):
        weights[interval-1:interval+3] += np.array([-1., 13., 13., -1.]) / 24
    weights[-4:] += np.array([1., -5., 19., 9.]) / 24
    return weights / (nodes-1)


def column_mean(field, params):
    """Use the same configured mean for external velocity and pressure owners."""
    weights = getattr(params, 'column_mean_weights', None)
    if weights is None:
        weights = params.dz_norm
    return jnp.sum(field * weights, axis=-1)


def cubic_reverse_primitive(nodes, spacing):
    """Interior cubic point inverse; original capacity mean at the surface."""
    intervals = np.zeros((nodes-1, nodes))
    intervals[0, :4] = spacing*np.array([9., 19., -5., 1.])/24
    for k in range(1, nodes-2):
        intervals[k, k-1:k+3] = spacing*np.array([-1., 13., 13., -1.])/24
    intervals[-1, -4:] = spacing*np.array([1., -5., 19., 9.])/24
    forward = np.vstack((np.zeros(nodes), np.cumsum(intervals, axis=0)))
    primitive = forward[-1][None, :] - forward
    primitive[0] = spacing
    primitive[0, [0, -1]] *= .5
    primitive[-1] = 0.
    return primitive


def cubic_point_integral(velocity, params):
    """Apply the same four-node primitive in linear work, without a dense dot."""
    step = params.dz_iface[..., :1]
    first = step*(9*velocity[..., :1]+19*velocity[..., 1:2]
                  -5*velocity[..., 2:3]+velocity[..., 3:4])/24
    middle = step*(-velocity[..., :-3]+13*velocity[..., 1:-2]
                   +13*velocity[..., 2:-1]-velocity[..., 3:])/24
    last = step*(velocity[..., -4:-3]-5*velocity[..., -3:-2]
                 +19*velocity[..., -2:-1]+9*velocity[..., -1:])/24
    intervals = jnp.concatenate((first, middle, last), axis=-1)
    interior = jnp.cumsum(intervals[..., ::-1], axis=-1)[..., ::-1]
    result = jnp.concatenate((interior, jnp.zeros_like(interior[..., :1])), axis=-1)
    return result.at[..., 0].set(jnp.sum(velocity*params.dz_node, axis=-1))


def point_mass_integral(mass, params):
    """Point inverse with the actual, pre-division surface mass operand."""
    result = cubic_point_integral(mass/params.dz_node, params)
    return result.at[..., 0].set(jnp.sum(mass, axis=-1))
