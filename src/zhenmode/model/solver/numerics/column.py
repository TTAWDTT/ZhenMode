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
