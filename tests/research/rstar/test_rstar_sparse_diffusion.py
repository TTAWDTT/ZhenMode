"""Sparse moving-coordinate graph controls before full material integration."""

from tests.support.rstar.sparse_diffusion import _case, _matrix

import jax

import jax.numpy as jnp

import numpy as np

import pytest

from tests.support.rstar.metric_controls import _fixture, _geometry

from research.experiments.material_rstar_coordinates.dense_oracle import assemble_diffusion

from research.experiments.material_rstar_coordinates.kernel import (
    diffusion_content_rhs,
    make_reference_stencil,
)

from research.experiments.material_rstar_coordinates.sparse_diffusion import (
    accumulate_edges,
    bounded_euler,
    bounded_heun,
    edge_coefficients,
    graph_volume_rhs,
    make_diffusion_graph,
    neighbor_bounds,
)

@pytest.mark.parametrize("two_nodes", [False, True])
def test_sparse_coalesced_graph_matches_independent_dense_and_original_adjoint(two_nodes, record_property):
    params, depths, stencil, graph, geometry, eta = _case(two_nodes)
    generator = np.random.default_rng(930102)
    kappa_h = generator.uniform(500., 1500., params.wet_mask_z.shape)
    kappa_v = generator.uniform(.0005, .0015, params.wet_mask_z.shape)
    coefficients, valid = jax.jit(lambda current: edge_coefficients(graph, current, kappa_h, kappa_v))(geometry)
    assert bool(valid)
    dense = assemble_diffusion(eta, depths, np.asarray(params.dz_node), np.asarray(params.wet_mask_z),
                              np.asarray(params.dx_2d), float(params.dy), np.asarray(params.cos_lat), kappa_h, kappa_v)
    wet = np.flatnonzero(np.asarray(graph.wet))
    actual = _matrix(coefficients, graph)[np.ix_(wet, wet)]
    floor = 64. * np.finfo(float).eps * np.linalg.norm(dense.stiffness, ord=np.inf)
    record_property("independent_dense_matrix_difference", float(np.max(np.abs(actual - dense.stiffness))))
    record_property("independent_matrix_64eps_floor", floor)
    assert np.max(np.abs(actual - dense.stiffness)) <= floor
    field = jnp.asarray(generator.normal(size=graph.shape))
    rhs = graph_volume_rhs(field, coefficients, graph) / graph.area.reshape(graph.shape)
    reference = diffusion_content_rhs(field, geometry, stencil, params, kappa_h, kappa_v)
    rhs_floor = 64. * np.finfo(float).eps * np.max(np.abs(dense.stiffness) @ np.abs(np.asarray(field).ravel()[wet]) / dense.area)
    record_property("original_adjoint_rhs_difference", float(np.max(np.abs(np.asarray(rhs - reference)))))
    record_property("original_rhs_64eps_floor", float(rhs_floor))
    assert np.max(np.abs(np.asarray(rhs - reference))) <= rhs_floor
    for sentinel in (123., -1e6):
        changed = jnp.where(params.wet_mask_z > 0., field, sentinel)
        np.testing.assert_array_equal(graph_volume_rhs(changed, coefficients, graph), graph_volume_rhs(field, coefficients, graph))

@pytest.mark.parametrize("kind", ["pulse", "constant", "random"])
def test_sparse_local_limiter_preserves_bounds_content_and_fixed_mass_energy(kind, record_property):
    params, _, _, graph, geometry, _ = _case()
    coefficients, valid = edge_coefficients(graph, geometry, 1000., .001)
    mass = graph.area * geometry.thickness.ravel()
    rate = accumulate_edges(graph, jnp.maximum(coefficients, 0.), "unsigned")
    duration = .45 / float(jnp.max(jnp.where(graph.wet, rate / jnp.where(mass > 0., mass, 1.), 0.)))
    field = np.random.default_rng(930103).uniform(0., 1., graph.shape)
    if kind == "constant":
        field[:] = 35.
    elif kind == "pulse":
        field[:] = 0.
        matrix = _matrix(coefficients, graph)
        off_diagonal = matrix.copy()
        np.fill_diagonal(off_diagonal, 0.)
        row, column = np.unravel_index(np.argmin(off_diagonal), matrix.shape)
        assert off_diagonal[row, column] < 0.
        field.ravel()[column] = 1.
    original = jnp.asarray(field) * mass.reshape(graph.shape)
    wet = np.asarray(graph.wet)
    first = jax.jit(lambda content: bounded_euler(content, geometry, coefficients, valid, duration, graph))(original)
    advance = jax.jit(lambda content: bounded_heun(content, geometry, coefficients, valid, duration, graph))
    result = advance(original)
    assert bool(first.valid) and bool(result.valid)
    lower, upper = neighbor_bounds(graph, jnp.asarray(field).ravel(), coefficients)
    concentration = np.asarray(first.content).ravel() / np.where(np.asarray(mass) > 0., mass, 1.)
    floor = 64. * np.finfo(float).eps * (1. + np.max(np.abs(field)))
    assert np.all(concentration[wet] >= np.asarray(lower)[wet] - floor)
    assert np.all(concentration[wet] <= np.asarray(upper)[wet] + floor)
    concentration = np.asarray(result.content).ravel() / np.where(np.asarray(mass) > 0., mass, 1.)
    assert np.min(concentration[wet]) >= field.ravel()[wet].min() - floor
    assert np.max(concentration[wet]) <= field.ravel()[wet].max() + floor
    inventory_error = float(jnp.sum(result.content - original))
    inventory_floor = 64. * np.finfo(float).eps * float(jnp.sum(jnp.abs(original) + jnp.abs(result.content) + jnp.abs(result.exchange)))
    assert abs(inventory_error) <= inventory_floor
    average = np.sum(np.asarray(original)) / np.sum(mass)
    energy_before = .5 * np.sum(np.asarray(mass)[wet] * (field.ravel()[wet] - average) ** 2)
    energy_after = .5 * np.sum(np.asarray(mass)[wet] * (concentration[wet] - average) ** 2)
    record_property("fixed_mass_inventory_error", inventory_error)
    record_property("fixed_mass_inventory_64eps_floor", inventory_floor)
    record_property("variance_before", float(energy_before))
    record_property("variance_after", float(energy_after))
    record_property("minimum_limiter", float(result.minimum_limiter))
    assert energy_after <= energy_before + 64. * np.finfo(float).eps * max(1., energy_before)
    if kind == "constant":
        np.testing.assert_array_equal(result.content, original)
    for sentinel in (123., -1e6):
        dry_content = jnp.where(params.wet_mask_z > 0., original, sentinel)
        changed = advance(dry_content)
        np.testing.assert_array_equal(np.asarray(changed.content).ravel()[wet], np.asarray(result.content).ravel()[wet])
        np.testing.assert_array_equal(np.asarray(changed.content).ravel()[~wet], np.asarray(dry_content).ravel()[~wet])

def test_sparse_horizontal_diffusion_keeps_affine_physical_depth_at_rest():
    _, _, _, graph, geometry, _ = _case()
    coefficients, valid = edge_coefficients(graph, geometry, 1000., 0.)
    mass = graph.area * geometry.thickness.ravel()
    rate = accumulate_edges(graph, jnp.maximum(coefficients, 0.), "unsigned")
    duration = .45 / float(jnp.max(jnp.where(graph.wet, rate / jnp.where(mass > 0., mass, 1.), 0.)))
    field = 15. + .005 * geometry.node_depth
    content = field * mass.reshape(graph.shape)
    result = bounded_heun(content, geometry, coefficients, valid, duration, graph)
    assert bool(result.valid)
    difference = np.asarray(result.content - content).ravel() / np.where(np.asarray(mass) > 0., mass, 1.)
    floor = 64. * np.finfo(float).eps * (1. + float(jnp.max(jnp.abs(field))))
    assert np.max(np.abs(difference)) <= floor

@pytest.mark.parametrize("reason", ["negative_kappa", "nonfinite_kappa", "invalid_geometry", "duration", "cfl", "float32"])
def test_sparse_refusal_retains_original_content_without_clipping(reason):
    params, depths, _, graph, geometry, _ = _case()
    original = jnp.ones(graph.shape) * graph.area.reshape(graph.shape) * geometry.thickness
    kappa = -.001 if reason == "negative_kappa" else jnp.nan if reason == "nonfinite_kappa" else .001
    if reason == "invalid_geometry":
        depth = np.sum(np.asarray(params.dz_node) * np.asarray(params.wet_mask_z), axis=-1)
        geometry = _geometry(params, depths, -1.01 * depth)
        assert not bool(geometry.valid)
    coefficients, valid = edge_coefficients(graph, geometry, 1000., kappa)
    if reason == "float32":
        with pytest.raises(ValueError, match="float64"):
            bounded_heun(original.astype(jnp.float32), geometry, coefficients, valid, 1., graph)
        return
    duration = -1. if reason == "duration" else 1e12 if reason == "cfl" else 1.
    result = bounded_heun(original, geometry, coefficients, valid, duration, graph)
    assert not bool(result.valid)
    np.testing.assert_array_equal(result.content, original)

def test_sparse_topology_storage_scales_with_nodes_not_their_square(record_property):
    sizes = []
    for longitude_count in (8, 16, 32):
        _, params, depths = _fixture(stairs=False, nx=longitude_count, ny=6)
        graph = make_diffusion_graph(params, make_reference_stencil(depths, params.wet_mask_z))
        sizes.append(sum(leaf.nbytes for leaf in jax.tree.leaves(graph) if hasattr(leaf, "nbytes")))
        assert graph.pair_left.size < 32 * graph.wet.size
    assert sizes[1] / sizes[0] < 2.1
    assert sizes[2] / sizes[1] < 2.1
    record_property("sparse_topology_bytes_nx8_16_32", str(sizes))

def test_sparse_disconnected_basins_do_not_exchange_content():
    _, params, depths = _fixture()
    wet = np.asarray(params.wet_mask_z).copy()
    wet[[0, 4]] = 0.
    params.wet_mask_z = jnp.asarray(wet)
    params.wet_mask = jnp.asarray(wet[..., 0])
    graph = make_diffusion_graph(params, make_reference_stencil(depths, wet))
    geometry = _geometry(params, depths, jnp.zeros(params.wet_mask.shape))
    coefficients, valid = edge_coefficients(graph, geometry, 1000., .001)
    basin = np.broadcast_to((np.arange(wet.shape[0]) < 4)[:, None, None], wet.shape).ravel()
    assert np.all(basin[np.asarray(graph.pair_left)] == basin[np.asarray(graph.pair_right)])
    field = jnp.where(jnp.asarray(basin).reshape(wet.shape), 5., 35.)
    content = field * graph.area.reshape(wet.shape) * geometry.thickness
    result = jax.jit(lambda current: bounded_heun(current, geometry, coefficients, valid, 1., graph))(content)
    assert bool(result.valid)
    np.testing.assert_array_equal(result.content, content)

def test_sparse_heun_time_convergence_against_independent_matrix_exponential(record_property):
    params, depths, _, graph, geometry, surface = _case()
    coefficients, valid = edge_coefficients(graph, geometry, 1000., .001)
    wet = np.flatnonzero(np.asarray(graph.wet))
    mass = np.asarray(graph.area * geometry.thickness.ravel())[wet]
    dense = assemble_diffusion(surface, depths, np.asarray(params.dz_node), np.asarray(params.wet_mask_z),
                              np.asarray(params.dx_2d), float(params.dy), np.asarray(params.cos_lat), 1000., .001)
    matrix = dense.stiffness
    positive = matrix.copy()
    np.fill_diagonal(positive, 0.)
    rate = np.max(np.maximum(positive, 0.).sum(axis=1) / mass)
    window = .2 / rate
    longitude = jnp.arange(graph.shape[0])[:, None, None]
    field = 20. + .005 * geometry.node_depth + jnp.sin(2. * jnp.pi * longitude / graph.shape[0])
    original = field * graph.area.reshape(graph.shape) * geometry.thickness
    inverse_root = 1. / np.sqrt(mass)
    eigenvalues, eigenvectors = np.linalg.eigh(matrix * inverse_root[:, None] * inverse_root[None, :])
    initial = np.asarray(field).ravel()[wet]
    exact = inverse_root * (eigenvectors @ (np.exp(window * eigenvalues) * (eigenvectors.T @ (initial / inverse_root))))
    advance = jax.jit(lambda current, duration: bounded_heun(current, geometry, coefficients, valid, duration, graph))
    errors, limiters = [], []
    for count in (4, 8, 16):
        content = original
        smallest = 1.
        for _ in range(count):
            result = advance(content, window / count)
            assert bool(result.valid)
            content = result.content
            smallest = min(smallest, float(result.minimum_limiter))
        final = np.asarray(content).ravel()[wet] / mass
        errors.append(float(np.sqrt(np.sum(mass * (final - exact) ** 2) / mass.sum())))
        limiters.append(smallest)
    ratios = np.asarray(errors[:-1]) / errors[1:]
    record_property("time_errors", str(errors))
    record_property("time_ratios", str(ratios.tolist()))
    record_property("minimum_limiters", str(limiters))
    assert np.all((ratios > 3.5) & (ratios < 4.5))

def test_sparse_smooth_geometry_gradient_matches_fd_and_adjoint(record_property):
    params, depths, _, graph, _, eta = _case()
    field = jnp.asarray(np.random.default_rng(930104).normal(size=graph.shape))
    direction = jnp.asarray(np.random.default_rng(930105).normal(size=eta.shape)) * params.wet_mask
    cotangent = jnp.asarray(np.random.default_rng(930106).normal(size=graph.shape)) * params.wet_mask_z

    def tendency(surface):
        geometry = _geometry(params, depths, surface)
        coefficients, _ = edge_coefficients(graph, geometry, 1000., .001)
        mass = graph.area.reshape(graph.shape) * geometry.thickness
        return graph_volume_rhs(field, coefficients, graph) / jnp.where(mass > 0., mass, 1.)

    surface = jnp.asarray(eta)
    _, tangent = jax.jvp(jax.jit(tendency), (surface,), (direction,))
    duration = 1e-3
    difference = (tendency(surface + duration * direction) - tendency(surface - duration * direction)) / (2. * duration)
    error = float(jnp.max(jnp.abs(tangent - difference)) / jnp.max(jnp.abs(tangent)))
    record_property("smooth_coordinate_fd_relative_error", error)
    assert error < 2e-7
    _, pullback = jax.vjp(tendency, surface)
    transpose = pullback(cotangent)[0]
    forward_work = jnp.sum(cotangent * tangent)
    reverse_work = jnp.sum(transpose * direction)
    floor = 64. * np.finfo(float).eps * float(jnp.sum(jnp.abs(cotangent * tangent)) + jnp.sum(jnp.abs(transpose * direction)))
    record_property("adjoint_work_difference", float(forward_work - reverse_work))
    record_property("adjoint_64eps_floor", floor)
    assert abs(float(forward_work - reverse_work)) <= floor
