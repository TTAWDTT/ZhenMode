"""Independent prescribed-face moving-content and dimensional source controls."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from tests.support.rstar.sparse_diffusion import _case

from ocean_solver.config.definitions import C_P, RHO_0
from research.experiments.material_rstar_coordinates.moving_content import (
    moving_euler,
    moving_heun,
)


def _faces(params):
    wet = np.asarray(params.wet_mask_z)
    generator = np.random.default_rng(930107)
    east = generator.normal(0., .02, wet.shape) * wet * np.roll(wet, -1, axis=0)
    north = generator.normal(0., .02, wet.shape) * wet * np.roll(wet, -1, axis=1)
    north[:, -1] = 0.
    return jnp.asarray(east), jnp.asarray(north)


def _numpy_donor(field, faces, params, reference_width):
    """Independent face-by-face volume and donor-content exchange."""
    wet = np.asarray(params.wet_mask_z) > 0.
    area = np.asarray(params.dx_2d) * float(params.dy)
    volume = np.zeros(wet.shape)
    content = np.zeros(wet.shape)
    absolute_exchange = np.zeros(wet.shape)
    for index in np.ndindex(wet.shape):
        for axis in (0, 1):
            neighbor = list(index)
            neighbor[axis] += 1
            if axis == 0:
                neighbor[axis] %= wet.shape[axis]
            elif neighbor[axis] == wet.shape[axis]:
                continue
            neighbor = tuple(neighbor)
            if not (wet[index] and wet[neighbor]):
                continue
            measure = float(params.dy) if axis == 0 else float(np.asarray(params.dx_2d)[index[:2]] / np.asarray(params.cos_lat)[index[1]])
            flux = float(np.asarray(faces[axis])[index]) * measure
            donor = field[index] if flux >= 0. else field[neighbor]
            volume[index] -= flux
            volume[neighbor] += flux
            content[index] -= flux * donor
            content[neighbor] += flux * donor
            absolute_exchange[index] += abs(flux * donor)
            absolute_exchange[neighbor] += abs(flux * donor)
    surface_rate = volume.sum(axis=-1) / area
    widths = reference_width * wet
    fractions = widths / np.where(widths.sum(axis=-1, keepdims=True) > 0., widths.sum(axis=-1, keepdims=True), 1.)
    vertical = np.cumsum(volume / area[..., None] - fractions * surface_rate[..., None], axis=-1)
    for index in np.ndindex(wet.shape):
        neighbor = index[:2] + (index[2] + 1,)
        if neighbor[-1] == wet.shape[-1] or not (wet[index] and wet[neighbor]):
            continue
        flux = area[index[:2]] * vertical[index]
        donor = field[index] if flux >= 0. else field[neighbor]
        content[index] -= flux * donor
        content[neighbor] += flux * donor
        absolute_exchange[index] += abs(flux * donor)
        absolute_exchange[neighbor] += abs(flux * donor)
    mass_rate = area[..., None] * fractions * surface_rate[..., None]
    return content, mass_rate, surface_rate, absolute_exchange


@pytest.mark.parametrize("kind", ["constant", "pulse", "signed"])
def test_joint_moving_mass_diffusion_preserves_constant_bounds_inventory_and_dry_bytes(kind, record_property):
    params, depths, _, graph, geometry, surface = _case()
    faces = _faces(params)
    mass = np.asarray(graph.area).reshape(graph.shape) * np.asarray(geometry.thickness)
    wet = np.asarray(params.wet_mask_z) > 0.
    field = np.random.default_rng(930108).uniform(-3., 30., graph.shape)
    if kind == "constant":
        field[:] = 35.
    elif kind == "pulse":
        field[:] = 0.
        field[3, 2, 2] = 1.
    original = jnp.asarray(field * mass)
    duration = 30.
    advance = jax.jit(lambda content: moving_heun(content, jnp.asarray(surface), faces, jnp.zeros(graph.shape),
                                                 duration, params, depths, graph, 1000., .001))
    result = advance(original)
    record_property("joint_outgoing_fraction", float(result.fraction))
    assert bool(result.valid)
    _, mass_rate, surface_rate, _ = _numpy_donor(field, faces, params, np.asarray(params.dz_node))
    expected_mass = mass + duration * mass_rate
    expected_surface = surface + duration * surface_rate
    np.testing.assert_allclose(result.surface, expected_surface, rtol=0., atol=64. * np.finfo(float).eps * np.max(np.abs(surface)))
    concentration = np.asarray(result.content)[wet] / expected_mass[wet]
    floor = 64. * np.finfo(float).eps * (1. + np.max(np.abs(field)))
    assert concentration.min() >= field[wet].min() - floor
    assert concentration.max() <= field[wet].max() + floor
    if kind == "constant":
        assert np.max(np.abs(concentration - 35.)) <= floor
    inventory_error = float(np.sum(np.asarray(result.content)[wet] - np.asarray(original)[wet]))
    inventory_floor = 64. * np.finfo(float).eps * float(np.sum(np.abs(np.asarray(result.content)[wet]) + np.abs(np.asarray(original)[wet])))
    record_property("content_inventory_error", inventory_error)
    record_property("content_inventory_64eps_floor", inventory_floor)
    assert abs(inventory_error) <= inventory_floor
    for sentinel in (123., -1e6):
        changed = advance(jnp.where(params.wet_mask_z > 0., original, sentinel))
        np.testing.assert_array_equal(np.asarray(changed.content)[wet], np.asarray(result.content)[wet])
        np.testing.assert_array_equal(np.asarray(changed.content)[~wet], np.full(np.sum(~wet), sentinel))


def test_moving_euler_matches_independent_donor_and_applied_heat_ledger(record_property):
    params, depths, _, graph, geometry, surface = _case()
    faces = _faces(params)
    wet = np.asarray(params.wet_mask_z) > 0.
    area = np.asarray(graph.area).reshape(graph.shape)
    mass = area * np.asarray(geometry.thickness)
    field = np.random.default_rng(930109).uniform(10., 20., graph.shape)
    flux_heat = np.where(np.asarray(params.wet_mask) > 0., 100., 0.)
    source = np.zeros(graph.shape)
    source[..., 0] = flux_heat * area[..., 0] / (RHO_0 * C_P)
    duration = 30.
    original = jnp.asarray(field * mass)
    result = moving_euler(original, jnp.asarray(surface), faces, jnp.asarray(source), duration, params, depths, graph, 0., 0.)
    assert bool(result.valid)
    donor, mass_rate, _, absolute_exchange = _numpy_donor(field, faces, params, np.asarray(params.dz_node))
    exchange_floor = 64. * np.finfo(float).eps * duration * absolute_exchange
    record_property("independent_donor_max_difference", float(np.max(np.abs(np.asarray(result.exchange) - duration * donor))))
    record_property("independent_donor_64eps_max_ratio", float(np.max(np.abs(np.asarray(result.exchange) - duration * donor) / np.where(exchange_floor > 0., exchange_floor, 1.))))
    assert np.all(np.abs(np.asarray(result.exchange) - duration * donor) <= exchange_floor)
    mass_floor = 64. * np.finfo(float).eps * (np.abs(mass) + np.abs(mass + duration * mass_rate))
    assert np.all(np.abs(np.asarray(result.mass_residual)) <= mass_floor)
    np.testing.assert_array_equal(result.applied_source, duration * source)
    ledger_heat = RHO_0 * C_P * float(np.sum(np.asarray(result.exchange)[wet] + np.asarray(result.applied_source)[wet]))
    heat_change = RHO_0 * C_P * float(np.sum((np.asarray(result.content) - np.asarray(original))[wet]))
    applied_heat = duration * float(np.sum(flux_heat * area[..., 0]))
    heat_floor = 64. * np.finfo(float).eps * RHO_0 * C_P * float(np.sum(np.abs(np.asarray(original)[wet]) + np.abs(np.asarray(result.content)[wet])))
    record_property("heat_change_joules", heat_change)
    record_property("applied_heat_joules", applied_heat)
    record_property("heat_residual_joules", heat_change - applied_heat)
    record_property("heat_inventory_64eps_floor_joules", heat_floor)
    assert abs(heat_change - applied_heat) <= heat_floor
    ledger_floor = 64. * np.finfo(float).eps * RHO_0 * C_P * float(np.sum(duration * absolute_exchange + np.abs(duration * source)))
    record_property("heat_exchange_ledger_residual_joules", ledger_heat - applied_heat)
    record_property("heat_exchange_64eps_floor_joules", ledger_floor)
    assert abs(ledger_heat - applied_heat) <= ledger_floor
    local_residual = np.asarray(result.content) - np.asarray(original) - duration * donor - duration * source
    local_floor = 64. * np.finfo(float).eps * (np.abs(np.asarray(original)) + np.abs(np.asarray(result.content)) + np.abs(duration * donor) + np.abs(duration * source))
    assert np.all(np.abs(local_residual) <= local_floor)
    low_without_source = field * mass + duration * donor
    endpoint_mass = mass + duration * mass_rate
    safe = np.where(endpoint_mass > 0., endpoint_mass, 1.)
    np.testing.assert_allclose((np.asarray(result.content) / safe)[wet], ((low_without_source + duration * source) / safe)[wet], rtol=64. * np.finfo(float).eps, atol=0.)


def test_joint_heun_diffusion_source_ledger_does_not_use_initial_inventory(record_property):
    params, depths, _, graph, geometry, surface = _case()
    faces = _faces(params)
    area = graph.area.reshape(graph.shape)
    field = jnp.zeros(graph.shape).at[3, 2, 2].set(1.)
    original = field * area * geometry.thickness
    heat_flux = jnp.where(params.wet_mask > 0., 100. * jnp.cos(jnp.arange(graph.shape[0])[:, None]), 0.)
    source = jnp.zeros(graph.shape).at[..., 0].set(heat_flux * area[..., 0] / (RHO_0 * C_P))
    duration = 30.
    result = jax.jit(lambda current: moving_heun(current, jnp.asarray(surface), faces, source, duration, params, depths, graph, 1000., .001))(original)
    assert bool(result.valid)
    np.testing.assert_array_equal(result.applied_source, duration * source)
    exchange = np.asarray(result.exchange)
    exchange_residual = float(exchange.sum())
    exchange_floor = 64. * np.finfo(float).eps * float(np.abs(exchange).sum())
    assert abs(exchange_residual) <= exchange_floor
    expected_heat = duration * float(jnp.sum(heat_flux * area[..., 0]))
    actual_heat = RHO_0 * C_P * float(jnp.sum(result.applied_source))
    source_floor = 64. * np.finfo(float).eps * duration * float(jnp.sum(jnp.abs(heat_flux * area[..., 0])))
    record_property("diffusion_heun_exchange_residual", exchange_residual)
    record_property("diffusion_heun_exchange_64eps_floor", exchange_floor)
    record_property("applied_heat_source_residual_joules", actual_heat - expected_heat)
    record_property("heat_source_only_64eps_floor_joules", source_floor)
    assert abs(actual_heat - expected_heat) <= source_floor
    local_residual = np.asarray(result.content) - np.asarray(original) - exchange - np.asarray(result.applied_source)
    local_floor = 64. * np.finfo(float).eps * (np.abs(np.asarray(result.content)) + np.abs(np.asarray(original))
                                             + np.abs(exchange) + np.abs(np.asarray(result.applied_source)))
    assert np.all(np.abs(local_residual) <= local_floor)


@pytest.mark.parametrize("reason", ["closed_face", "negative_geometry", "cfl", "source_nan", "dry_source", "not_donor"])
def test_moving_content_refuses_without_mutating_original_surface_or_content(reason):
    params, depths, _, graph, geometry, surface = _case()
    original = jnp.asarray(graph.area).reshape(graph.shape) * geometry.thickness * 15.
    source = jnp.zeros(graph.shape)
    faces = _faces(params)
    duration = 30.
    if reason == "closed_face":
        faces = faces[0], faces[1].at[:, -1, 0].set(.01)
    elif reason == "negative_geometry":
        surface = -1.01 * np.sum(np.asarray(params.dz_node) * np.asarray(params.wet_mask_z), axis=-1)
    elif reason == "cfl":
        duration = 1e9
    elif reason == "source_nan":
        source = source.at[3, 2, 0].set(jnp.nan)
    elif reason == "dry_source":
        source = source.at[0, 2, 0].set(1.)
    elif reason == "not_donor":
        params.fct_adv = True
        with pytest.raises(ValueError, match="donor"):
            moving_heun(original, jnp.asarray(surface), faces, source, duration, params, depths, graph, 1000., .001)
        return
    result = moving_heun(original, jnp.asarray(surface), faces, source, duration, params, depths, graph, 1000., .001)
    assert not bool(result.valid)
    np.testing.assert_array_equal(result.content, original)
    np.testing.assert_array_equal(result.surface, surface)


def test_moving_donor_heun_time_convergence_against_independent_variable_mass_rk4(record_property):
    params, depths, _, graph, geometry, surface = _case()
    faces = _faces(params)
    wet = np.flatnonzero(np.asarray(graph.wet))
    mass = np.asarray(graph.area * geometry.thickness.ravel())[wet]
    matrix = np.empty((len(wet), len(wet)))
    for column, node in enumerate(wet):
        basis = np.zeros(graph.shape)
        basis.ravel()[node] = 1.
        derivative, _, _, _ = _numpy_donor(basis, faces, params, np.asarray(params.dz_node))
        matrix[:, column] = derivative.ravel()[wet]
    _, mass_rate, _, _ = _numpy_donor(np.ones(graph.shape), faces, params, np.asarray(params.dz_node))
    mass_rate = mass_rate.ravel()[wet]
    window = .2 / np.max(-np.diag(matrix) / mass)
    assert np.min(mass + window * mass_rate) > 0.
    longitude = np.arange(graph.shape[0])[:, None, None]
    field = 20. + .002 * np.asarray(geometry.node_depth) + np.sin(2. * np.pi * longitude / graph.shape[0])
    original = jnp.asarray(field) * graph.area.reshape(graph.shape) * geometry.thickness

    def reference(count):
        content = np.asarray(original).ravel()[wet].copy()
        duration = window / count
        for step in range(count):
            time = step * duration
            first = matrix @ (content / (mass + time * mass_rate))
            second = matrix @ ((content + .5 * duration * first) / (mass + (time + .5 * duration) * mass_rate))
            third = matrix @ ((content + .5 * duration * second) / (mass + (time + .5 * duration) * mass_rate))
            fourth = matrix @ ((content + duration * third) / (mass + (time + duration) * mass_rate))
            content += duration * (first + 2. * second + 2. * third + fourth) / 6.
        return content / (mass + window * mass_rate)

    coarse_reference, exact = reference(512), reference(1024)
    endpoint_mass = mass + window * mass_rate
    reference_error = float(np.sqrt(np.sum(endpoint_mass * (coarse_reference - exact) ** 2) / endpoint_mass.sum()))
    advance = jax.jit(lambda content, current_surface, duration: moving_heun(content, current_surface, faces, jnp.zeros(graph.shape),
                                                                             duration, params, depths, graph, 0., 0.))
    errors = []
    for count in (4, 8, 16):
        content, current_surface = original, jnp.asarray(surface)
        for _ in range(count):
            result = advance(content, current_surface, window / count)
            assert bool(result.valid)
            content, current_surface = result.content, result.surface
        final = np.asarray(content).ravel()[wet] / endpoint_mass
        errors.append(float(np.sqrt(np.sum(endpoint_mass * (final - exact) ** 2) / endpoint_mass.sum())))
    ratios = np.asarray(errors[:-1]) / errors[1:]
    record_property("prescribed_transport_window_seconds_not_actual_integration", window)
    record_property("moving_time_errors", str(errors))
    record_property("moving_time_ratios", str(ratios.tolist()))
    record_property("reference_refinement_error", reference_error)
    assert reference_error < 1e-4 * errors[-1]
    assert np.all((ratios > 3.5) & (ratios < 4.5))
