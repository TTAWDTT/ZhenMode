"""Physical sparse weak rates and independent consistent-mass flux correction."""

from tests.support.rstar.weak_bounded import _bounded_case, _bounded_diagnostic, _compiled_coefficients, _compiled_rate, _compiled_stage, _dense_operator, _independent_stage

import jax

import jax.numpy as jnp

import numpy as np

import pytest

from tests.support.rstar.metric_controls import _fixture

from tests.support.rstar.representation import _gauss_mass

from tests.support.rstar.weak_momentum import _consistent_case

from config import ALPHA_T, C_P, RHO_0

from research.experiments.material_rstar_coordinates.bed_completion import complete_bed_reference

from research.experiments.material_rstar_coordinates.kernel import rstar_geometry

from research.experiments.material_rstar_coordinates.nodal_mass import (
    apply_nodal_mass,
    solve_nodal_mass,
)

from research.experiments.material_rstar_coordinates.pressure_work import potential_conjugates

from research.experiments.material_rstar_coordinates.weak_bounded import bounded_weak_euler

from research.experiments.material_rstar_coordinates.weak_oracle import physical_weak_rates

from research.experiments.material_rstar_coordinates.weak_sparse import (
    make_weak_graph,
    weak_coefficients,
    weak_sparse_rate,
)

from research.experiments.material_rstar_coordinates.weak_transport import (
    WeakParameters,
    weak_content_rate,
)

@pytest.mark.parametrize("flat", [False, True])
@pytest.mark.parametrize("kind", ["constant", "pulse", "signed"])
def test_sparse_physical_rhs_matches_independent_five_point_gauss(flat, kind):
    params, weak_params, _, _, graph, geometry, _, surface, concentration, _, velocities, _ = _bounded_case(kind, flat)
    coefficients = _compiled_coefficients(velocities, surface, geometry, weak_params, graph)
    actual = _compiled_rate(concentration, coefficients, graph)
    expected, eta, scale, eta_scale = physical_weak_rates(concentration, velocities, surface, geometry, params)
    assert np.all(np.abs(np.asarray(actual) - expected) <= 64. * np.finfo(float).eps * (1. + scale))
    assert np.all(np.abs(np.asarray(coefficients.surface_rate) - eta) <= 64. * np.finfo(float).eps * (1e-20 + eta_scale))

def test_sparse_operator_matches_small_global_jacobian_and_exact_column_and_mass_rows():
    values = _bounded_case()
    _, weak_params, _, _, graph, geometry, _, surface, concentration, _, velocities, _ = values
    coefficients = _compiled_coefficients(velocities, surface, geometry, weak_params, graph)
    actual = _dense_operator(coefficients, graph)
    jacobian = jax.jit(jax.jacfwd(lambda field: weak_content_rate(field, velocities, surface, geometry, weak_params)[0]))(concentration)
    expected = np.asarray(jacobian).reshape(actual.shape)
    floor = 64. * np.finfo(float).eps * (1. + np.abs(expected).sum(axis=1, keepdims=True))
    assert np.all(np.abs(actual - expected) <= floor)
    assert np.all(np.abs(actual.sum(axis=0)) <= 64. * np.finfo(float).eps * np.abs(actual).sum(axis=0))
    target = np.asarray(weak_params.dx_2d * weak_params.dy)[..., None] * np.asarray(geometry.weights) * np.asarray(coefficients.surface_rate)[..., None]
    assert np.all(np.abs(actual.sum(axis=1).reshape(target.shape) - target) <= 64. * np.finfo(float).eps * (1. + np.abs(actual).sum(axis=1).reshape(target.shape)))

@pytest.mark.parametrize("nx", [31, 33])
def test_streamed_periodic_wrap_and_partial_last_block_match_independent_quadrature(nx):
    _, params, depths = _fixture(nx=nx)
    reference = complete_bed_reference(depths, params)
    weak_params = WeakParameters(reference.wet, reference.wet[..., 0], reference.widths,
                                 params.dx_2d, params.dy, params.cos_lat)
    generator = np.random.default_rng(930402)
    surface = jnp.asarray(generator.uniform(-3., 2., reference.wet.shape[:2])) * weak_params.wet_mask
    geometry = rstar_geometry(surface, reference.nodes, reference.widths, reference.wet)
    graph = make_weak_graph(weak_params)
    field = jnp.asarray(generator.normal(size=reference.wet.shape)) * reference.wet
    velocities = tuple(jnp.asarray(generator.normal(0., .03, reference.wet.shape)) * reference.wet for _ in range(2))
    coefficients = _compiled_coefficients(velocities, surface, geometry, weak_params, graph)
    actual = _compiled_rate(field, coefficients, graph)
    expected, eta, scale, eta_scale = physical_weak_rates(field, velocities, surface, geometry, weak_params)
    assert np.all(np.abs(np.asarray(actual) - expected) <= 64. * np.finfo(float).eps * (1. + scale))
    assert np.all(np.abs(np.asarray(coefficients.surface_rate) - eta) <= 64. * np.finfo(float).eps * (1e-20 + eta_scale))

@pytest.mark.parametrize("kind", ["constant", "pulse", "signed", "heat"])
def test_pair_limited_moving_content_matches_independent_matrix_solver_and_bounds(kind):
    values = _bounded_case(kind)
    params, weak_params, nodes, mass, graph, geometry, _, surface, _, content, velocities, source = values
    duration = 1000.
    coefficients = _compiled_coefficients(velocities, surface, geometry, weak_params, graph)
    result = _compiled_stage(content, surface, velocities, source, duration, weak_params, nodes, mass, graph)
    assert bool(result.valid)
    expected, low, high, before, after, decoded_old = _independent_stage(values, duration, coefficients)
    row_mass = after.sum(axis=1).reshape(content.shape)
    tracer_scale = 1. + float(np.max(np.abs(np.asarray(values[8]))))
    stock_floor = 64. * np.finfo(float).eps * (row_mass * tracer_scale + np.abs(expected) + np.abs(np.asarray(result.content)) + np.abs(np.asarray(content)))
    residual = before @ decoded_old.ravel() - np.asarray(content).ravel()
    backward_floor = 64. * np.finfo(float).eps * (np.abs(before) @ np.abs(decoded_old.ravel()) + np.abs(np.asarray(content).ravel()))
    assert np.all(np.abs(residual) <= backward_floor)
    assert np.all(np.abs(np.asarray(result.content) - expected) <= stock_floor)
    assert np.all(np.abs(np.asarray(result.low_row_content) - low) <= stock_floor)
    next_geometry = rstar_geometry(result.surface, nodes, params.dz_node, params.wet_mask_z)
    decoded = np.asarray(solve_nodal_mass(result.content, next_geometry, params.dx_2d * params.dy, mass))
    wet = np.asarray(mass.wet)
    bound_floor = 64. * np.finfo(float).eps * (1. + np.abs(decoded))
    assert np.all(decoded[wet] >= np.asarray(result.lower_bound)[wet] - bound_floor[wet])
    assert np.all(decoded[wet] <= np.asarray(result.upper_bound)[wet] + bound_floor[wet])
    residual = float(np.asarray(result.content).sum() - np.asarray(content).sum() - duration * np.asarray(source).sum())
    assert abs(residual) <= float(stock_floor.sum())
    if kind == "constant":
        assert np.max(np.abs(decoded[wet] - 35.)) <= 64. * np.finfo(float).eps * 36.
    if kind == "pulse":
        assert high[wet].min() < -1e-8
        assert decoded[wet].min() >= -64. * np.finfo(float).eps * 2.
        assert float(result.minimum_limiter) < 1.
    if kind == "heat":
        exact = duration * float(np.sum(100. * np.asarray(params.dx_2d) * params.dy * np.asarray(params.wet_mask)))
        measured = RHO_0 * C_P * float(np.asarray(result.source).sum())
        assert abs(measured - exact) <= 64. * np.finfo(float).eps * abs(exact)
    assert np.all(np.diag(after)[wet.ravel()] > 0.)

@pytest.mark.parametrize("sentinel", [123., -1e6, np.nan])
def test_dry_inputs_preserve_active_results_and_original_inactive_bytes(sentinel):
    params, weak_params, nodes, mass, graph, _, _, surface, _, content, velocities, source = _bounded_case()
    original = _compiled_stage(content, surface, velocities, source, 1000., weak_params, nodes, mass, graph)
    wet = np.asarray(mass.wet)
    changed = jnp.asarray(np.where(wet, content, sentinel))
    shifted_velocity = tuple(jnp.asarray(np.where(wet, value, sentinel)) for value in velocities)
    shifted_surface = jnp.where(params.wet_mask > 0., surface, sentinel)
    result = _compiled_stage(changed, shifted_surface, shifted_velocity, source, 1000., weak_params, nodes, mass, graph)
    assert bool(result.valid)
    assert np.asarray(result.content)[wet].tobytes() == np.asarray(original.content)[wet].tobytes()
    assert np.asarray(result.content)[~wet].tobytes() == np.asarray(changed)[~wet].tobytes()
    ocean = np.asarray(params.wet_mask) > 0.
    assert np.asarray(result.surface)[ocean].tobytes() == np.asarray(original.surface)[ocean].tobytes()
    assert np.asarray(result.surface)[~ocean].tobytes() == np.asarray(shifted_surface)[~ocean].tobytes()

@pytest.mark.parametrize("failure", ["cfl", "zero_duration", "nan_source", "dry_source", "active_nan", "graph_mask"])
def test_refused_stage_rolls_back_original_state_bytes(failure):
    _, weak_params, nodes, mass, graph, _, _, surface, _, content, velocities, source = _bounded_case()
    duration = 1e8 if failure == "cfl" else (0. if failure == "zero_duration" else 1000.)
    if failure == "nan_source":
        source = source.at[3, 2, 2].set(jnp.nan)
    elif failure == "dry_source":
        source = source.at[0, 2, 0].set(1.)
    elif failure == "active_nan":
        content = content.at[3, 2, 2].set(jnp.nan)
    elif failure == "graph_mask":
        graph = graph._replace(wet=graph.wet.at[0].set(False))
    result = _compiled_stage(content, surface, velocities, source, duration, weak_params, nodes, mass, graph)
    assert not bool(result.valid)
    assert np.asarray(result.content).tobytes() == np.asarray(content).tobytes()
    assert np.asarray(result.surface).tobytes() == np.asarray(surface).tobytes()

def test_zero_flow_does_not_change_consistent_content_within_original_roundoff_scale():
    _, weak_params, nodes, mass, graph, _, _, surface, _, content, velocities, source = _bounded_case()
    zeros = tuple(jnp.zeros_like(value) for value in velocities)
    result = _compiled_stage(content, surface, zeros, source, 1000., weak_params, nodes, mass, graph)
    assert bool(result.valid)
    assert np.all(np.abs(np.asarray(result.content - content)) <= 64. * np.finfo(float).eps * (1. + np.abs(np.asarray(content))))
    assert np.asarray(result.surface).tobytes() == np.asarray(surface).tobytes()

@pytest.mark.parametrize("failure", ["nonbinary", "disconnected", "all_dry", "bad_metric"])
def test_graph_constructor_refuses_incompatible_topology_or_metrics(failure):
    from types import SimpleNamespace
    params = _bounded_case()[0]
    fields = vars(params).copy()
    if failure == "bad_metric":
        fields["dy"] = np.nan
    else:
        mask = np.asarray(params.wet_mask_z).copy()
        if failure == "nonbinary":
            mask[3, 2, 2] = .5
        elif failure == "disconnected":
            mask[3, 2, 2] = 0.
        else:
            mask[:] = 0.
        fields["wet_mask_z"] = mask
    with pytest.raises(ValueError):
        make_weak_graph(SimpleNamespace(**fields))
