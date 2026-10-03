"""Independent inventory/work checks for the actual nonlinear FV step."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from zhenmode_research.candidates.fv.geometry import (
    ExtensiveState,
    VolumeFluxes,
    build_geometry,
    surface_volume,
)
from zhenmode_research.candidates.fv.momentum import LayerState, momentum_geometry
from zhenmode_research.candidates.fv.nonlinear import (
    DualVelocity,
    dual_advection,
    dual_rotation,
    nonlinear_momentum_surface_step,
)

jax.config.update("jax_enable_x64", True)


def fixture(partial=False, dtype=jnp.float64):
    depth = np.full((5, 4), 50.)
    if partial:
        depth[1, 1], depth[2, 2], depth[3, 1] = 22., 3., 0.
    geometry = build_geometry([0., 37., 131., 206., 298., 360.],
                              [-60., -27., -3., 15., 56.], [0., 7., 21., 50.],
                              depth, radius=2.1e6)
    eta = .3 * (depth > 0.)
    volume = surface_volume(geometry, jnp.asarray(eta))
    concentration = jnp.broadcast_to(jnp.array([12., 35.]), volume.shape + (2,))
    faces = momentum_geometry(geometry, volume)
    east = jnp.where(faces.east_area > 0., .1, 0.).astype(dtype)
    north = jnp.zeros(volume.shape, dtype)
    return geometry, LayerState(ExtensiveState(volume, volume[..., None] * concentration), east, north)


def host_map(geometry, field):
    latitude = np.asarray(geometry.latitude_edges)
    split = (np.sin(.5 * (latitude[:-1] + latitude[1:])) - np.sin(latitude[:-1])) / np.diff(np.sin(latitude))
    east = np.zeros_like(field)
    north = np.zeros((field.shape[0], field.shape[1] + 1, field.shape[2]))
    for lon_index, lat_index, layer_index in np.ndindex(field.shape):
        amount = field[lon_index, lat_index, layer_index]
        east[lon_index, lat_index, layer_index] += .5 * amount
        east[(lon_index - 1) % field.shape[0], lat_index, layer_index] += .5 * amount
        north[lon_index, lat_index, layer_index] += split[lat_index] * amount
        north[lon_index, lat_index + 1, layer_index] += (1. - split[lat_index]) * amount
    return east, north


def host_advection(velocity, fluxes, upwind):
    divergence = np.zeros_like(velocity)
    dissipation = 0.
    for position in np.ndindex(velocity.shape):
        for axis in range(3):
            following = list(position)
            following[axis] += 1
            if axis == 0:
                following[axis] %= velocity.shape[axis]
            elif following[axis] >= velocity.shape[axis]:
                continue
            following = tuple(following)
            flux_position = list(position)
            if axis:
                flux_position[axis] += 1
            amount = np.asarray(fluxes[axis])[tuple(flux_position)]
            face = .5 * (velocity[position] + velocity[following])
            if upwind:
                face += .5 * np.sign(amount) * (velocity[position] - velocity[following])
                dissipation += .5 * abs(amount) * (velocity[position] - velocity[following]) ** 2
            divergence[position] += amount * face
            divergence[following] -= amount * face
    return divergence, dissipation


def host_faces(geometry, volume):
    shape = volume.shape
    top = np.broadcast_to(geometry.interfaces[:-1], shape).copy()
    top[..., 0] = geometry.thickness[..., 0] - volume[..., 0] / geometry.area
    bottom = top + volume / geometry.area[..., None]
    areas = [np.zeros(shape), np.zeros(shape)]
    for position in np.ndindex(shape):
        for axis in range(2):
            adjacent = list(position)
            adjacent[axis] += 1
            if axis == 0:
                adjacent[axis] %= shape[axis]
            elif adjacent[axis] >= shape[axis]:
                continue
            adjacent = tuple(adjacent)
            base = (geometry.east_area, geometry.north_area)[axis][position]
            common = min(geometry.thickness[position], geometry.thickness[adjacent])
            if base > 0.:
                areas[axis][position] = base * max(0., min(bottom[position], bottom[adjacent]) - max(top[position], top[adjacent])) / common
    return areas


def host_rotation(volume, velocity, frequency):
    east, north = velocity
    force_east, force_north = np.zeros_like(east), np.zeros_like(north)
    for lon_index, lat_index, layer_index in np.ndindex(east.shape):
        west = ((lon_index - 1) % east.shape[0], lat_index, layer_index)
        position = (lon_index, lat_index, layer_index)
        south = (lon_index, lat_index, layer_index)
        upper = (lon_index, lat_index + 1, layer_index)
        mean_east = .5 * (east[west] + east[position])
        mean_north = .5 * (north[south] + north[upper])
        weight = volume[position] * frequency[position]
        force_east[west] += .5 * weight * mean_north
        force_east[position] += .5 * weight * mean_north
        force_north[south] -= .5 * weight * mean_east
        force_north[upper] -= .5 * weight * mean_east
    return force_east, force_north


def host_dual_fluxes(geometry, fluxes):
    east, north, vertical = (np.asarray(field) for field in fluxes)
    north_full = np.pad(north, ((0, 0), (1, 0), (0, 0)))
    east_dual = VolumeFluxes(*(.5 * (field + np.roll(field, -1, axis=0)) for field in (east, north_full, vertical)))
    latitude = geometry.latitude_edges
    split = (np.sin(.5 * (latitude[:-1] + latitude[1:])) - np.sin(latitude[:-1])) / np.diff(np.sin(latitude))
    north_east = np.zeros((east.shape[0], east.shape[1] + 1, east.shape[2]))
    north_north = np.zeros((east.shape[0], east.shape[1] + 2, east.shape[2]))
    north_vertical = np.zeros((east.shape[0], east.shape[1] + 1, east.shape[2] + 1))
    for lon_index, lat_index, layer_index in np.ndindex(east.shape):
        position = (lon_index, lat_index, layer_index)
        north_east[position] += .5 * east[position]
        north_east[lon_index, lat_index + 1, layer_index] += .5 * east[position]
        north_north[lon_index, lat_index + 1, layer_index] = (
            (1. - split[lat_index]) * north_full[position] + split[lat_index] * north_full[lon_index, lat_index + 1, layer_index]
            + (split[lat_index] - .5) * (east[position] - east[(lon_index - 1) % east.shape[0], lat_index, layer_index]))
    for position in np.ndindex(vertical.shape):
        lon_index, lat_index, layer_index = position
        north_vertical[position] += split[lat_index] * vertical[position]
        north_vertical[lon_index, lat_index + 1, layer_index] += (1. - split[lat_index]) * vertical[position]
    return east_dual, VolumeFluxes(north_east, north_north, north_vertical)


@pytest.mark.parametrize("upwind", [False, True])
def test_all_three_dual_advection_axes_match_independent_face_loop(upwind):
    rng = np.random.default_rng(803)
    velocity = rng.normal(size=(5, 4, 3))
    east = rng.normal(size=velocity.shape)
    north = rng.normal(size=(5, 5, 3))
    vertical = rng.normal(size=(5, 4, 4))
    north[:, 0] = north[:, -1] = 0.
    vertical[..., 0] = vertical[..., -1] = 0.
    fluxes = VolumeFluxes(*(jnp.asarray(field) for field in (east, north, vertical)))
    expected, dissipation = host_advection(velocity, fluxes, upwind)
    actual, loss = dual_advection(jnp.asarray(velocity), fluxes, upwind)
    np.testing.assert_allclose(actual, expected, atol=2e-15)
    np.testing.assert_allclose(loss, dissipation, atol=2e-14)
    net = east - np.roll(east, 1, axis=0) + north[:, 1:] - north[:, :-1] + vertical[..., 1:] - vertical[..., :-1]
    np.testing.assert_allclose(np.sum(velocity * actual), .5 * np.sum(velocity ** 2 * net) + dissipation, atol=2e-14)
    assert np.max(np.abs(expected - host_advection(velocity, fluxes._replace(vertical=jnp.zeros_like(fluxes.vertical)), upwind)[0])) > .1


def test_rotation_matches_independent_cell_scatter_and_is_skew():
    geometry, state = fixture(True)
    rng = np.random.default_rng(709)
    faces = momentum_geometry(geometry, state.inventory.volume)
    east = np.where(faces.east_area > 0., rng.normal(size=faces.height.shape), 0.)
    north = np.pad(np.where(faces.north_area > 0., rng.normal(size=faces.height.shape), 0.), ((0, 0), (1, 0), (0, 0)))
    frequency = rng.normal(size=faces.height.shape) * 1e-4
    force_east, force_north = host_rotation(np.asarray(state.inventory.volume), (east, north), frequency)
    actual = dual_rotation(state.inventory.volume, DualVelocity(jnp.asarray(east), jnp.asarray(north)), jnp.asarray(frequency))
    np.testing.assert_allclose(actual.east, force_east, rtol=2e-14, atol=1e-6)
    np.testing.assert_allclose(actual.north, force_north, rtol=2e-14, atol=1e-6)
    work = np.sum(east * actual.east) + np.sum(north * actual.north)
    scale = np.sum(abs(east * actual.east)) + np.sum(abs(north * actual.north))
    assert abs(work) <= 64. * np.finfo(float).eps * scale


@pytest.mark.parametrize("rate", [1e-6, -1e-6])
@pytest.mark.parametrize("incoming", [0., .24])
def test_uniform_isolated_source_changes_actual_momentum_and_kinetic_energy(rate, incoming):
    geometry, state = fixture()
    source = jnp.zeros_like(state.inventory.volume).at[..., 0].set(geometry.area * rate)
    source_velocity = DualVelocity(jnp.full_like(source, incoming), jnp.zeros_like(source))
    source_content = source[..., None] * jnp.array([12., 35.])
    result = nonlinear_momentum_surface_step(geometry, state, jnp.zeros_like(source), 600.,
                                            gravity=0., coriolis=0., curvature=False,
                                            volume_source=source, content_source=source_content,
                                            incoming_velocity=source_velocity)
    assert bool(result.valid), result.diagnostics
    old_mass, final_mass = host_map(geometry, np.asarray(state.inventory.volume))[0], host_map(geometry, np.asarray(result.state.inventory.volume))[0]
    expected_momentum = old_mass * .1
    if rate > 0.:
        expected_momentum += 600. * host_map(geometry, np.asarray(source))[0] * incoming
        expected = expected_momentum / final_mass
    else:
        expected = np.full_like(old_mass, .1)
    np.testing.assert_allclose(result.uncast_velocity.east, expected, rtol=1e-13, atol=2e-16)
    np.testing.assert_allclose(result.eta, .3 + 600. * rate, rtol=0., atol=3e-15)
    old_energy = .5 * np.sum(old_mass * .1 ** 2)
    final_energy = .5 * np.sum(final_mass * np.asarray(result.uncast_velocity.east) ** 2)
    expected_work = (float(result.diagnostics.source_incoming_energy) - float(result.diagnostics.source_mixing)
                     + float(result.diagnostics.source_outgoing_energy))
    assert abs(final_energy - old_energy - expected_work) <= 64. * np.finfo(float).eps * (old_energy + final_energy)
    if rate > 0.:
        assert float(result.diagnostics.source_mixing) >= 0.
        assert np.max(np.abs(np.asarray(result.uncast_velocity.east)[..., 0] - .1)) > 1e-6
        latitude = geometry.latitude_edges
        primitive = .5 * np.diff(latitude) + .25 * np.diff(np.sin(2. * latitude))
        radius = geometry.north_width[0, 0] / np.diff(latitude)[0]
        longitude_width = geometry.east_width[:, 0] / (radius * np.cos(.5 * (latitude[0] + latitude[1])))
        weight = radius ** 3 * longitude_width[:, None] * primitive[None, :]
        initial_height = np.asarray(state.inventory.volume) / geometry.area[..., None]
        final_height = np.asarray(result.state.inventory.volume) / geometry.area[..., None]
        angular0 = .1 * np.sum(weight[..., None] * initial_height)
        angular1 = np.sum(weight[..., None] * final_height * np.asarray(result.uncast_velocity.east))
        injected_angular = 600. * incoming * rate * np.sum(weight)
        assert abs(angular1 - angular0 - injected_angular) < 1e-12 * abs(injected_angular) + 64. * np.finfo(float).eps * (abs(angular0) + abs(angular1))


def independent_budget(geometry, state, result, dt, source, incoming, upwind, gravity):
    old_mass = host_map(geometry, np.asarray(state.inventory.volume))
    final_mass = host_map(geometry, np.asarray(result.state.inventory.volume))
    old_velocity = (np.asarray(state.east_velocity, float), np.pad(np.asarray(state.north_velocity, float), ((0, 0), (1, 0), (0, 0))))
    final_velocity = tuple(np.asarray(field) for field in result.uncast_velocity)
    step_velocity = tuple(np.asarray(field) for field in result.transport_velocity)
    positive, negative = host_map(geometry, np.maximum(source, 0.)), host_map(geometry, np.minimum(source, 0.))
    injected = (host_map(geometry, np.maximum(source, 0.) * incoming.east)[0], host_map(geometry, np.maximum(source, 0.) * incoming.north)[1])
    incoming_energy = (host_map(geometry, .5 * np.maximum(source, 0.) * incoming.east ** 2)[0], host_map(geometry, .5 * np.maximum(source, 0.) * incoming.north ** 2)[1])
    dual = host_dual_fluxes(geometry, result.fluxes)
    middle_volume = .5 * (np.asarray(state.inventory.volume) + np.asarray(result.state.inventory.volume))
    face_areas = host_faces(geometry, middle_volume)
    np.testing.assert_allclose(result.fluxes.east, face_areas[0] * step_velocity[0], rtol=2e-14, atol=1e-8)
    np.testing.assert_allclose(result.fluxes.north, face_areas[1] * step_velocity[1][:, 1:], rtol=2e-14, atol=1e-8)
    latitude = geometry.latitude_edges
    middle = .5 * (latitude[:-1] + latitude[1:])
    radius = geometry.north_width[0, 0] / np.diff(latitude)[0]
    mean_east = .5 * (step_velocity[0] + np.roll(step_velocity[0], 1, axis=0))
    frequency = 2. * 7.2921e-5 * np.sin(middle)[None, :, None] + mean_east * np.tan(middle)[None, :, None] / radius
    spin = host_rotation(middle_volume, step_velocity, frequency)
    mean_eta = np.asarray(result.mean_eta)
    pressure = (face_areas[0] * (mean_eta - np.roll(mean_eta, -1, axis=0))[..., None],
                np.pad(face_areas[1] * (mean_eta - np.pad(mean_eta[:, 1:], ((0, 0), (0, 1))))[..., None], ((0, 0), (1, 0), (0, 0))))
    opened = (geometry.east_area > 0., np.pad(geometry.north_area > 0., ((0, 0), (1, 0), (0, 0))))
    losses, mixing, outgoing, kinetic_change, injection = 0., 0., 0., 0., 0.
    for component in range(2):
        velocity = step_velocity[component]
        advection, loss = host_advection(velocity, dual[component], upwind)
        losses += dt * loss
        mixing += dt * np.sum(incoming_energy[component] - velocity * injected[component] + .5 * positive[component] * velocity ** 2)
        outgoing += .5 * dt * np.sum(negative[component] * velocity ** 2)
        injection += dt * np.sum(incoming_energy[component])
        kinetic_change += .5 * np.sum(final_mass[component] * final_velocity[component] ** 2 - old_mass[component] * old_velocity[component] ** 2)
        mass_change = final_mass[component] - old_mass[component]
        momentum_change = final_mass[component] * final_velocity[component] - old_mass[component] * old_velocity[component]
        tendency = (-advection + spin[component] + gravity * pressure[component] + np.asarray(result.held_force[component])
                    + injected[component] + negative[component] * velocity)
        row_residual = momentum_change - dt * tendency
        participating = np.where(opened[component], abs(momentum_change) + dt * abs(tendency), 0.)
        stored_scale = np.where(opened[component], abs(final_mass[component] * final_velocity[component]) + abs(old_mass[component] * old_velocity[component]), 0.)
        tolerance = 1e-11 * np.linalg.norm(participating) + 64. * np.finfo(float).eps * np.linalg.norm(stored_scale)
        assert np.linalg.norm(np.where(opened[component], row_residual, 0.)) <= tolerance
        np.testing.assert_allclose(result.wall_reaction[component], np.where(opened[component], 0., row_residual), rtol=1e-12, atol=.0001)
        identity = np.sum(velocity * momentum_change - .5 * velocity ** 2 * mass_change)
        direct = .5 * np.sum(final_mass[component] * final_velocity[component] ** 2 - old_mass[component] * old_velocity[component] ** 2)
        assert abs(identity - direct) < 1e-13 * np.sum(old_mass[component] * old_velocity[component] ** 2)
        if component == 0:
            assert np.max(np.abs(advection)) > 0.
    old_eta = np.asarray(state.inventory.volume)[..., 0] / geometry.area - geometry.thickness[..., 0]
    surface_change = .5 * gravity * np.sum(geometry.area * (np.asarray(result.eta) ** 2 - old_eta ** 2))
    surface_work = dt * gravity * np.sum(np.asarray(result.mean_eta) * source[..., 0])
    held_work = dt * sum(np.sum(np.asarray(force) * velocity) for force, velocity in zip(result.held_force, step_velocity))
    expected = injection - mixing + outgoing - losses + surface_work + held_work
    energy_scale = sum(np.sum(mass * velocity ** 2) for mass, velocity in zip(old_mass, old_velocity)) + gravity * np.sum(geometry.area * old_eta ** 2)
    assert abs(kinetic_change + surface_change - expected) < 1e-11 * abs(expected) + 64. * np.finfo(float).eps * energy_scale
    np.testing.assert_allclose(result.diagnostics.advection_dissipation, losses, rtol=1e-13, atol=1e-5)
    np.testing.assert_allclose(result.diagnostics.source_mixing, mixing, rtol=1e-13, atol=1e-5)


@pytest.mark.parametrize("upwind", [False, True])
@pytest.mark.parametrize("dtype", [jnp.float64, jnp.float32])
def test_actual_sheared_partial_moving_step_closes_independent_work_and_shared_inventory(upwind, dtype):
    geometry, state = fixture(True, dtype)
    rng = np.random.default_rng(844)
    faces = momentum_geometry(geometry, state.inventory.volume)
    east = jnp.asarray(np.where(faces.east_area > 0., rng.normal(size=faces.height.shape) * .08, 0.), dtype)
    north = jnp.asarray(np.where(faces.north_area > 0., rng.normal(size=faces.height.shape) * .08, 0.), dtype)
    state = state._replace(east_velocity=east, north_velocity=north)
    source = np.zeros(faces.height.shape)
    source[..., 0] = geometry.area * np.sin(np.arange(5)[:, None]) * 1e-6 * (geometry.thickness[..., 0] > 0.)
    incoming = DualVelocity(np.full_like(source, .04), np.full_like(source, -.03))
    result = nonlinear_momentum_surface_step(geometry, state, jnp.zeros_like(state.inventory.volume), 120.,
                                            volume_source=jnp.asarray(source), content_source=jnp.asarray(source[..., None] * [12., 35.]),
                                            incoming_velocity=DualVelocity(*(jnp.asarray(field) for field in incoming)), upwind=upwind)
    assert bool(result.valid), result.diagnostics
    independent_budget(geometry, state, result, 120., source, incoming, upwind, 9.81)
    east_flux, north_flux, vertical = (np.asarray(field) for field in result.fluxes)
    net = east_flux - np.roll(east_flux, 1, axis=0) + north_flux - np.pad(north_flux[:, :-1], ((0, 0), (1, 0), (0, 0))) + vertical[..., 1:] - vertical[..., :-1]
    expected_volume = np.asarray(state.inventory.volume) + 120. * (source - net)
    np.testing.assert_allclose(result.state.inventory.volume, expected_volume, rtol=3e-16, atol=.1)
    np.testing.assert_allclose(result.state.inventory.content / np.where(expected_volume > 0., expected_volume, 1.)[..., None],
                               np.where(expected_volume[..., None] > 0., [12., 35.], 0.), rtol=1e-14)
    assert np.max(abs(vertical[..., 1:-1])) > 0.
    assert np.max(abs(result.wall_reaction.north)) > 0.
    assert result.state.east_velocity.dtype == dtype


def test_positive_water_without_declared_incoming_velocity_is_rejected():
    geometry, state = fixture()
    source = jnp.zeros_like(state.inventory.volume).at[..., 0].set(geometry.area * 1e-6)
    result = nonlinear_momentum_surface_step(geometry, state, jnp.zeros_like(source), 10., volume_source=source,
                                            content_source=source[..., None] * jnp.array([12., 35.]))
    assert not bool(result.valid)


@pytest.mark.parametrize("defect", ["negative_volume", "interior_source", "closed_velocity", "bad_metric", "nan_source"])
def test_invalid_physical_inputs_fail_closed(defect):
    geometry, state = fixture(True)
    source = jnp.zeros_like(state.inventory.volume)
    if defect == "negative_volume":
        state = state._replace(inventory=state.inventory._replace(volume=state.inventory.volume.at[0, 0, 0].set(-1.)))
    elif defect == "interior_source":
        source = source.at[0, 0, 1].set(1.)
    elif defect == "closed_velocity":
        state = state._replace(north_velocity=state.north_velocity.at[0, -1, 0].set(.1))
    elif defect == "bad_metric":
        geometry = geometry._replace(area=geometry.area * 1.01)
    else:
        source = source.at[0, 0, 0].set(jnp.nan)
    result = nonlinear_momentum_surface_step(geometry, state, jnp.zeros_like(source), 10., volume_source=source,
                                            incoming_velocity=DualVelocity(jnp.zeros_like(source), jnp.zeros_like(source)))
    assert not bool(result.valid)


@pytest.mark.parametrize("defect", ["endpoint_q", "source_sign", "vertical_transport", "pressure_transpose"])
def test_independent_actual_step_audit_rejects_deliberate_defects(defect):
    geometry, state = fixture(True)
    rng = np.random.default_rng(915)
    state = state._replace(east_velocity=jnp.where(geometry.east_area > 0., jnp.asarray(rng.normal(size=state.east_velocity.shape) * .08), 0.),
                           north_velocity=jnp.where(geometry.north_area > 0., jnp.asarray(rng.normal(size=state.north_velocity.shape) * .08), 0.))
    source = np.zeros(state.inventory.volume.shape)
    source[..., 0] = geometry.area * 1e-6 * (geometry.thickness[..., 0] > 0.)
    incoming = DualVelocity(np.full_like(source, .04), np.full_like(source, -.03))
    result = nonlinear_momentum_surface_step(geometry, state, jnp.zeros_like(state.inventory.volume), 120., volume_source=jnp.asarray(source),
                                            content_source=jnp.asarray(source[..., None] * [12., 35.]),
                                            incoming_velocity=DualVelocity(*(jnp.asarray(field) for field in incoming)))
    assert bool(result.valid), result.diagnostics
    independent_budget(geometry, state, result, 120., source, incoming, False, 9.81)
    if defect == "endpoint_q":
        result = result._replace(fluxes=result.fluxes._replace(east=result.fluxes.east * 1.01))
    elif defect == "source_sign":
        source = -source
    elif defect == "vertical_transport":
        result = result._replace(fluxes=result.fluxes._replace(vertical=jnp.zeros_like(result.fluxes.vertical)))
    else:
        result = result._replace(mean_eta=-result.mean_eta)
    with pytest.raises(AssertionError):
        independent_budget(geometry, state, result, 120., source, incoming, False, 9.81)


def test_unconverged_outer_solve_is_rejected_without_energy_repair():
    geometry, state = fixture(True)
    result = nonlinear_momentum_surface_step(geometry, state, jnp.zeros_like(state.inventory.volume), 600., outer_iterations=1)
    assert not bool(result.valid)
    assert int(result.diagnostics.outer_iterations) == 1
    assert float(result.diagnostics.outer_change) > 2e-13


@pytest.mark.parametrize("field", ["source_dtype", "incoming_shape", "inventory_dtype"])
def test_invalid_api_contract_raises_before_solve(field):
    geometry, state = fixture()
    source = jnp.zeros_like(state.inventory.volume)
    incoming = DualVelocity(source, source)
    if field == "source_dtype":
        source = source.astype(jnp.float32)
    elif field == "incoming_shape":
        incoming = incoming._replace(north=source[:, :-1])
    else:
        state = state._replace(inventory=state.inventory._replace(content=state.inventory.content.astype(jnp.float32)))
    with pytest.raises(ValueError):
        nonlinear_momentum_surface_step(geometry, state, jnp.zeros_like(state.inventory.volume), 10.,
                                        volume_source=source, incoming_velocity=incoming)


@pytest.mark.parametrize("partial", [False, True])
def test_actual_nonlinear_time_refinement_is_second_order(partial):
    geometry, initial = fixture(partial)
    rng = np.random.default_rng(931)
    eta = (.3 + .04 * np.cos(np.arange(5)[:, None] + .4 * np.arange(4)[None, :])) * (geometry.thickness[..., 0] > 0.)
    volume = surface_volume(geometry, jnp.asarray(eta))
    initial = initial._replace(inventory=ExtensiveState(volume, volume[..., None] * jnp.array([12., 35.])),
                               east_velocity=jnp.where(geometry.east_area > 0., jnp.asarray(rng.normal(size=volume.shape) * .06), 0.),
                               north_velocity=jnp.where(geometry.north_area > 0., jnp.asarray(rng.normal(size=volume.shape) * .06), 0.))

    def trajectory(dt):
        state = initial
        for unused_step in range(round(1200. / dt)):
            result = nonlinear_momentum_surface_step(geometry, state, jnp.zeros_like(volume), dt)
            assert bool(result.valid), result.diagnostics
            state = result.state
        eta = np.asarray(state.inventory.volume)[..., 0] / geometry.area - geometry.thickness[..., 0]
        mass = host_map(geometry, np.asarray(state.inventory.volume))
        return np.concatenate((np.sqrt(mass[0]).ravel() * np.asarray(state.east_velocity).ravel(),
                               np.sqrt(mass[1][:, 1:]).ravel() * np.asarray(state.north_velocity).ravel(),
                               np.sqrt(9.81 * geometry.area).ravel() * eta.ravel()))

    reference = trajectory(18.75)
    errors = [np.linalg.norm(trajectory(dt) - reference) for dt in (300., 150., 75.)]
    ratios = np.asarray(errors[:-1]) / errors[1:]
    assert np.all(ratios > 3.3), (errors, ratios)


def test_curvature_north_acceleration_converges_to_physical_spherical_component():
    errors = []
    for ny in (12, 24, 48):
        geometry = build_geometry(np.linspace(0., 360., 9), np.linspace(-55., 55., ny + 1), [0., 50.], np.full((8, ny), 50.))
        volume = jnp.asarray(geometry.area[..., None] * geometry.thickness)
        mass = host_map(geometry, np.asarray(volume))
        east = jnp.full_like(volume, .3)
        north = jnp.full((8, ny + 1, 1), .12).at[:, 0].set(0.).at[:, -1].set(0.)
        latitude = geometry.latitude_edges
        radius = geometry.north_width[0, 0] / np.diff(latitude)[0]
        frequency = .3 * jnp.tan(jnp.asarray(.5 * (latitude[:-1] + latitude[1:])))[None, :, None] / radius
        actual = np.asarray(dual_rotation(volume, DualVelocity(east, north), frequency).north) / mass[1]
        expected = -.3 ** 2 * np.tan(latitude)[None, :, None] / radius
        common_faces = np.arange(3, 10) * (ny // 12)
        errors.append(np.max(abs(actual[:, common_faces] - expected[:, common_faces])))
    ratios = np.asarray(errors[:-1]) / errors[1:]
    assert np.all(ratios > 3.3), (errors, ratios)
