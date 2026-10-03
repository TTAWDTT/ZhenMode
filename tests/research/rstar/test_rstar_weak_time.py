"""Measured time order and joint moving-consistent-mass/source stage contracts."""
from functools import partial

import jax

import jax.numpy as jnp

import numpy as np

import pytest

from tests.support.rstar.representation import _gauss_mass

from tests.support.rstar.weak_bounded import _bounded_case

from ocean_solver.config.definitions import C_P, RHO_0

from research.experiments.material_rstar_coordinates.kernel import rstar_geometry

from research.experiments.material_rstar_coordinates.nodal_mass import (
    apply_nodal_mass,
    make_nodal_mass,
    solve_nodal_mass,
)

from research.experiments.material_rstar_coordinates.weak_bounded import bounded_weak_euler

from research.experiments.material_rstar_coordinates.weak_sparse import make_weak_graph

from research.experiments.material_rstar_coordinates.weak_time import bounded_weak_heun

from research.experiments.material_rstar_coordinates.weak_transport import WeakParameters

from tests.support.rstar.weak_time import (
    _compiled_heun,
    _compiled_euler,
    _flat_time_case,
    _flat_fields,
    _time_diagnostic,
    _state_fields,
    _moving_time_diagnostic,
    _heat_fields,
    _refusal_fields,
)

def test_heun_time_order_against_independent_semidiscrete_fourier_and_euler_negative_control():
    record = _time_diagnostic()
    assert min(record["orders"]["heun"]) >= 1.9, record
    assert max(record["orders"]["euler"]) < 1.5, record
    assert record["errors"]["heun"][-1] < record["errors"]["euler"][-1], record


def test_stage_time_and_updated_geometry_combination_against_independent_gauss_mass_decode():
    params, weak_params, nodes, mass, graph, geometry, _, surface, _, content, velocities, source = _bounded_case("heat")
    fields = partial(_state_fields, velocities, source)
    time, duration = 30., 1000.
    result = _compiled_heun(content, surface, time, duration, weak_params, nodes, mass, graph, fields)
    assert bool(result.valid) and bool(result.first_valid) and bool(result.second_valid)
    velocity0, source0 = fields(time, content, surface, geometry)
    first = _compiled_euler(content, surface, velocity0, source0, duration, weak_params, nodes, mass, graph)
    middle = rstar_geometry(first.surface, nodes, params.dz_node, params.wet_mask_z)
    velocity1, source1 = fields(time + duration, first.content, first.surface, middle)
    second = _compiled_euler(first.content, first.surface, velocity1, source1, duration, weak_params, nodes, mass, graph)
    wet = np.asarray(mass.wet)
    expected = .5 * (np.asarray(content) + np.asarray(second.content))
    floor = 64. * np.finfo(float).eps * (1. + np.abs(expected) + np.abs(np.asarray(content)))
    assert np.all(np.abs(np.asarray(result.content)[wet] - expected[wet]) <= floor[wet])
    np.testing.assert_allclose(result.source, .5 * (first.source + second.source), rtol=64. * np.finfo(float).eps, atol=0.)
    reference, _ = _gauss_mass(np.asarray(nodes), np.asarray(params.dz_node), wet)
    depth = np.sum(np.asarray(params.dz_node), axis=-1)
    area = np.asarray(params.dx_2d) * float(params.dy)
    decoded = np.zeros(content.shape)
    original = np.zeros(content.shape)
    decoded_second = np.zeros(content.shape)
    for column in np.ndindex(wet.shape[:2]):
        active = wet[column]
        if not np.any(active):
            continue
        base = area[column] * reference[column][np.ix_(active, active)]
        scale0 = 1. + float(surface[column]) / depth[column]
        scale2 = 1. + float(second.surface[column]) / depth[column]
        final_scale = 1. + float(result.surface[column]) / depth[column]
        original[column][active] = np.linalg.solve(scale0 * base, np.asarray(content)[column][active])
        decoded_second[column][active] = np.linalg.solve(scale2 * base, np.asarray(second.content)[column][active])
        decoded[column][active] = np.linalg.solve(final_scale * base, np.asarray(result.content)[column][active])
        convex = (scale0 * original[column][active] + scale2 * decoded_second[column][active]) / (scale0 + scale2)
        point_floor = 64. * np.finfo(float).eps * (1. + np.abs(convex) + np.abs(decoded[column][active]))
        assert np.all(np.abs(decoded[column][active] - convex) <= point_floor)
    bound_floor = 64. * np.finfo(float).eps * (1. + np.abs(decoded))
    assert np.all(decoded[wet] >= np.asarray(result.lower_bound)[wet] - bound_floor[wet])
    assert np.all(decoded[wet] <= np.asarray(result.upper_bound)[wet] + bound_floor[wet])
    exact_heat = duration * np.sum(100. * area * np.asarray(params.wet_mask)) * .5 * (2. + (2. * time + duration) / 2000.)
    assert abs(RHO_0 * C_P * float(np.sum(result.source)) - exact_heat) <= 64. * np.finfo(float).eps * exact_heat
    frozen_second = _compiled_euler(first.content, first.surface, velocity0, source0, duration, weak_params, nodes, mass, graph)
    assert float(jnp.max(jnp.abs(result.surface - .5 * (surface + frozen_second.surface)))) > 1e-8


def test_time_linear_heat_has_exact_independent_trapezoidal_physical_source_and_node_increment():
    params, nodes, mass, graph, surface, _, _ = _flat_time_case()
    concentration = jnp.full(params.wet_mask_z.shape, 15.)
    geometry = rstar_geometry(surface, nodes, params.dz_node, params.wet_mask_z)
    content = apply_nodal_mass(concentration, geometry, params.dx_2d * params.dy, mass)
    time, duration = 10., 1000.
    result = _compiled_heun(content, surface, time, duration, params, nodes, mass, graph, partial(_heat_fields, params))
    assert bool(result.valid)
    exact_source = duration * (100. + 5. * (time + .5 * duration))
    heat = RHO_0 * C_P * float(jnp.sum(result.source))
    expected_heat = exact_source * float(jnp.sum(params.dx_2d * params.dy))
    assert abs(heat - expected_heat) <= 64. * np.finfo(float).eps * expected_heat
    expected = np.asarray(concentration).copy()
    expected[..., 0] += exact_source / (RHO_0 * C_P * float(params.dz_node[0, 0, 0]))
    geometry = rstar_geometry(result.surface, nodes, params.dz_node, params.wet_mask_z)
    decoded = np.asarray(solve_nodal_mass(result.content, geometry, params.dx_2d * params.dy, mass))
    assert np.all(np.abs(decoded - expected) <= 64. * np.finfo(float).eps * (1. + np.abs(expected)))


@pytest.mark.parametrize("failure", ["first_cfl", "second_nan_source", "second_dry_source", "second_cfl", "nonfinite_time"])
def test_either_stage_refusal_returns_original_active_and_nan_inactive_bytes(failure):
    _, params, nodes, mass, graph, _, _, surface, _, content, velocities, source = _bounded_case()
    wet = np.asarray(mass.wet)
    content = jnp.where(mass.wet, content, jnp.nan)
    surface = jnp.where(params.wet_mask > 0., surface, jnp.nan)
    fields = partial(_refusal_fields, velocities, source, failure)
    duration = 1e8 if failure == "first_cfl" else 1000.
    time = np.nan if failure == "nonfinite_time" else 0.
    result = _compiled_heun(content, surface, time, duration, params, nodes, mass, graph, fields)
    assert not bool(result.valid)
    if failure.startswith("second_"):
        assert bool(result.first_valid) and not bool(result.second_valid)
    assert np.asarray(result.content).tobytes() == np.asarray(content).tobytes()
    assert np.asarray(result.surface).tobytes() == np.asarray(surface).tobytes()
    assert np.asarray(result.content)[~wet].tobytes() == np.asarray(content)[~wet].tobytes()


@pytest.mark.parametrize("sentinel", [123., -1e6, np.nan])
def test_valid_stages_preserve_inactive_bytes(sentinel):
    _, params, nodes, mass, graph, _, _, surface, _, content, velocities, source = _bounded_case()
    content = jnp.where(mass.wet, content, sentinel)
    surface = jnp.where(params.wet_mask > 0., surface, sentinel)
    fields = partial(_refusal_fields, velocities, source, "none")
    result = _compiled_heun(content, surface, 0., 1000., params, nodes, mass, graph, fields)
    assert bool(result.valid)
    wet, ocean = np.asarray(mass.wet), np.asarray(params.wet_mask) > 0.
    assert np.asarray(result.content)[~wet].tobytes() == np.asarray(content)[~wet].tobytes()
    assert np.asarray(result.surface)[~ocean].tobytes() == np.asarray(surface)[~ocean].tobytes()
