"""Physical sparse weak rates and independent consistent-mass flux correction."""

import jax
import jax.numpy as jnp
import numpy as np

from config import ALPHA_T, C_P, RHO_0
from research.experiments.material_rstar_coordinates.kernel import rstar_geometry
from research.experiments.material_rstar_coordinates.nodal_mass import (
    apply_nodal_mass,
    solve_nodal_mass,
)
from research.experiments.material_rstar_coordinates.pressure_work import potential_conjugates
from research.experiments.material_rstar_coordinates.weak_bounded import bounded_weak_euler
from research.experiments.material_rstar_coordinates.weak_sparse import (
    make_weak_graph,
    weak_coefficients,
    weak_sparse_rate,
)
from tests.support.rstar.representation import _gauss_mass
from tests.support.rstar.weak_momentum import _consistent_case

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
