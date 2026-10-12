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
