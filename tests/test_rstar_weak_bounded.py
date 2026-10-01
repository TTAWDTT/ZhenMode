"""Physical sparse weak rates and independent consistent-mass flux correction."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from test_rstar_metric_controls import _fixture
from test_rstar_representation import _gauss_mass
from test_rstar_weak_momentum import _consistent_case

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

_compiled_coefficients = jax.jit(weak_coefficients)
_compiled_rate = jax.jit(weak_sparse_rate)
_compiled_stage = jax.jit(bounded_weak_euler)


def _bounded_case(kind="signed", flat=False):
    case, mass, _ = _consistent_case(flat=flat)
    params, weak_params, geometry, basis, surface, _, _, velocities = case
    nodes = (np.asarray(geometry.node_depth) + np.asarray(surface)[..., None]) / np.asarray(geometry.scale)[..., None]
    nodes = np.where(np.asarray(mass.wet), nodes, 0.)
    nodes[..., 0] = 0.
    concentration = np.random.default_rng(930401).uniform(-3., 30., nodes.shape)
    if kind == "constant":
        concentration[:] = 35.
    elif kind == "pulse":
        concentration[:] = 0.
        concentration[3, 2, 2] = 1.
    elif kind == "heat":
        concentration = np.random.default_rng(930401).uniform(10., 20., nodes.shape)
    concentration = jnp.asarray(concentration) * mass.wet
    content = apply_nodal_mass(concentration, geometry, params.dx_2d * params.dy, mass)
    graph = make_weak_graph(params)
    source = np.zeros(nodes.shape)
    if kind == "heat":
        source[..., 0] = 100. * np.asarray(params.dx_2d) * float(params.dy) * np.asarray(params.wet_mask) / (RHO_0 * C_P)
    return params, weak_params, jnp.asarray(nodes), mass, graph, geometry, basis, surface, concentration, content, velocities, jnp.asarray(source)


def _dense_operator(coefficients, graph):
    operator = np.diag(np.asarray(coefficients.diagonal))
    left, right = np.asarray(graph.pair_left), np.asarray(graph.pair_right)
    operator[left, right] = np.asarray(coefficients.forward)
    operator[right, left] = np.asarray(coefficients.reverse)
    return operator


def _independent_stage(values, duration, coefficients):
    params, _, nodes, mass, graph, geometry, _, surface, _, content, _, source = values
    wet = np.asarray(mass.wet)
    area = np.asarray(params.dx_2d) * float(params.dy)
    endpoint = np.asarray(surface) + duration * np.asarray(coefficients.surface_rate)
    scale = 1. + endpoint / np.where(np.sum(np.asarray(params.dz_node), axis=-1) > 0., np.sum(np.asarray(params.dz_node), axis=-1), 1.)
    reference, _ = _gauss_mass(np.asarray(nodes), np.asarray(params.dz_node), wet)
    size, levels = wet.size, wet.shape[-1]
    before, after = np.zeros((size, size)), np.zeros((size, size))
    for column in np.ndindex(wet.shape[:2]):
        start = np.ravel_multi_index(column, wet.shape[:2]) * levels
        selected = slice(start, start + levels)
        before[selected, selected] = area[column] * np.asarray(geometry.scale)[column] * reference[column]
        after[selected, selected] = area[column] * scale[column] * reference[column]
    active = wet.ravel()
    original = np.where(active, np.asarray(content).ravel(), 0.)
    field = np.zeros(size)
    field[active] = np.linalg.solve(before[np.ix_(active, active)], original[active])
    operator = _dense_operator(coefficients, graph)
    viscosity = np.maximum(0., np.maximum(-operator, -operator.T))
    np.fill_diagonal(viscosity, 0.)
    np.fill_diagonal(viscosity, -viscosity.sum(axis=1))
    applied = duration * np.asarray(source).ravel()
    old_row, new_row = before.sum(axis=1), after.sum(axis=1)
    low = old_row * field + duration * ((operator + viscosity) @ field) + applied
    high_content = original + duration * (operator @ field) + applied
    high = np.zeros(size)
    high[active] = np.linalg.solve(after[np.ix_(active, active)], high_content[active])
    difference = field[:, None] - field[None, :]
    high_difference = high[:, None] - high[None, :]
    anti = after * high_difference - before * difference + duration * viscosity * difference
    np.fill_diagonal(anti, 0.)
    support = (np.abs(operator) + np.abs(operator.T) + before) > 0.
    lower, upper = field.copy(), field.copy()
    for node in np.flatnonzero(active):
        adjacent = np.flatnonzero(support[node])
        lower[node] = min(field[node], np.min(field[adjacent]))
        upper[node] = max(field[node], np.max(field[adjacent]))
    positive = np.maximum(anti, 0.).sum(axis=1)
    negative = -np.minimum(anti, 0.).sum(axis=1)
    plus = np.minimum(1., np.maximum(new_row * upper + applied - low, 0.) / np.where(positive > 0., positive, 1.))
    minus = np.minimum(1., np.maximum(low - new_row * lower - applied, 0.) / np.where(negative > 0., negative, 1.))
    limiter = np.where(anti >= 0., np.minimum(plus[:, None], minus[None, :]), np.minimum(minus[:, None], plus[None, :]))
    corrected_row = low + (limiter * anti).sum(axis=1)
    corrected = corrected_row / np.where(new_row > 0., new_row, 1.)
    expected = after @ corrected
    return expected.reshape(wet.shape), low.reshape(wet.shape), high.reshape(wet.shape), before, after, field.reshape(wet.shape)


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


def _bounded_diagnostic(kind, flat):
    values = _bounded_case(kind, flat)
    params, weak_params, nodes, mass, graph, geometry, basis, surface, _, content, velocities, source = values
    duration = 1000.
    coefficients = _compiled_coefficients(velocities, surface, geometry, weak_params, graph)
    result = _compiled_stage(content, surface, velocities, source, duration, weak_params, nodes, mass, graph)
    expected, low, high, _, after, _ = _independent_stage(values, duration, coefficients)
    next_geometry = rstar_geometry(result.surface, nodes, params.dz_node, params.wet_mask_z)
    decoded = np.asarray(solve_nodal_mass(result.content, next_geometry, params.dx_2d * params.dy, mass))
    wet = np.asarray(mass.wet)
    error = np.abs(np.asarray(result.content) - expected)
    row_mass = after.sum(axis=1).reshape(content.shape)
    tracer_scale = 1. + float(np.max(np.abs(np.asarray(values[8]))))
    floor = 64. * np.finfo(float).eps * (row_mass * tracer_scale + np.abs(expected) + np.abs(np.asarray(result.content)) + np.abs(np.asarray(content)))
    conjugate = potential_conjugates(result.high_content, result.surface, next_geometry, basis, weak_params)[0]
    transfer = float(jnp.sum(-RHO_0 * ALPHA_T * (result.content - result.high_content) * conjugate))
    inventory_floor = 64. * np.finfo(float).eps * float(np.sum(np.abs(np.asarray(result.content))) + np.sum(np.abs(np.asarray(content))) + np.sum(np.abs(np.asarray(result.source))))
    diagnostic = {"kind": kind, "flat": flat, "valid": bool(result.valid), "independent_stage_passed": bool(np.all(error <= floor)),
                  "outgoing_fraction": float(result.fraction), "minimum_limiter": float(result.minimum_limiter),
                  "inventory_residual": float(result.inventory_residual), "inventory_64eps_floor": inventory_floor,
                  "decoded_min": float(decoded[wet].min()), "decoded_max": float(decoded[wet].max()),
                  "unlimited_min": float(high[wet].min()), "unlimited_max": float(high[wet].max()),
                  "thermal_limited_minus_high_potential_joules": transfer, "source_content": float(np.asarray(result.source).sum()),
                  "applied_heat_joules": RHO_0 * C_P * float(np.asarray(result.source).sum()) if kind == "heat" else None,
                  "production_promotion": False}
    arrays = {"original_content": np.asarray(content), "accepted_content": np.asarray(result.content),
              "independent_content": expected, "content_64eps_floor": floor, "low_row_content": np.asarray(result.low_row_content),
              "independent_low_row_content": low, "unlimited_concentration": high, "accepted_concentration": decoded,
              "source": np.asarray(result.source), "original_surface": np.asarray(surface), "accepted_surface": np.asarray(result.surface),
              "decomposition_residual": np.asarray(result.decomposition_residual), "wet": wet,
              "lower_bound": np.asarray(result.lower_bound), "upper_bound": np.asarray(result.upper_bound)}
    return diagnostic, arrays
