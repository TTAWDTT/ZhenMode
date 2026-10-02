"""Sparse moving-coordinate graph controls before full material integration."""


import jax.numpy as jnp
import numpy as np

from research.experiments.material_rstar_coordinates.kernel import (
    make_reference_stencil,
)
from research.experiments.material_rstar_coordinates.sparse_diffusion import (
    make_diffusion_graph,
)
from tests.support.rstar.metric_controls import _fixture, _geometry


def _case(two_nodes=False):
    _, params, depths = _fixture()
    if two_nodes:
        wet = np.asarray(params.wet_mask_z).copy()
        wet[3, :, 2:] = 0.
        params.wet_mask_z = jnp.asarray(wet)
    stencil = make_reference_stencil(depths, params.wet_mask_z)
    graph = make_diffusion_graph(params, stencil)
    generator = np.random.default_rng(930101)
    eta = generator.uniform(-3., 2., params.wet_mask.shape) * np.asarray(params.wet_mask)
    geometry = _geometry(params, depths, eta)
    return params, depths, stencil, graph, geometry, eta

def _matrix(coefficients, graph):
    matrix = np.zeros((graph.wet.size, graph.wet.size))
    matrix[np.asarray(graph.pair_left), np.asarray(graph.pair_right)] = np.asarray(coefficients)
    matrix[np.asarray(graph.pair_right), np.asarray(graph.pair_left)] = np.asarray(coefficients)
    matrix[np.diag_indices_from(matrix)] = -matrix.sum(axis=1)
    return matrix
