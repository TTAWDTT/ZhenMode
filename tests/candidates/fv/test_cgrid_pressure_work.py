"""Physical pressure force, shared-Q work, spatial consistency and derivatives."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ocean_solver.config.definitions import R_EARTH
from zhenmode_research.candidates.fv.barotropic import subcycle_barotropic
from zhenmode_research.candidates.fv.geometry import build_geometry, surface_volume
from zhenmode_research.candidates.fv.momentum import hydrostatic_pressure_force, momentum_geometry

jax.config.update("jax_enable_x64", True)


def _mass_oracle(geometry, volume):
    area = np.asarray(geometry.area)
    latitude = np.asarray(geometry.latitude_edges)
    middle = .5 * (latitude[:-1] + latitude[1:])
    delta_lon = area / (R_EARTH ** 2 * np.diff(np.sin(latitude))[None, :])
    east_length = np.broadcast_to(R_EARTH * np.diff(latitude)[None, :], area.shape)
    north_length = R_EARTH * delta_lon * np.cos(latitude[1:])[None, :]
    east_dual = .5 * (area + np.roll(area, -1, axis=0))
    north_limit = np.r_[middle[1:], middle[-1]]
    north_dual = R_EARTH ** 2 * delta_lon * (np.sin(north_limit) - np.sin(middle))[None, :]
    height = np.asarray(volume) / area[..., None]
    top = np.broadcast_to(geometry.interfaces[:-1], height.shape).copy()
    top[..., 0] = geometry.thickness[..., 0] - height[..., 0]
    bottom = top + height
    neighbor_top = np.concatenate((top[:, 1:], top[:, -1:]), axis=1)
    neighbor_bottom = np.concatenate((bottom[:, 1:], bottom[:, -1:]), axis=1)
    east_height = np.maximum(np.minimum(bottom, np.roll(bottom, -1, axis=0))
                             - np.maximum(top, np.roll(top, -1, axis=0)), 0.)
    north_height = np.maximum(np.minimum(bottom, neighbor_bottom) - np.maximum(top, neighbor_top), 0.)
    east_height = np.where(geometry.east_area > 0., east_height, 0.)
    north_height = np.where(geometry.north_area > 0., north_height, 0.)
    return (east_length[..., None] * east_height, north_length[..., None] * north_height,
            east_dual[..., None] * east_height, north_dual[..., None] * north_height)


def _fixture(irregular=False, partial=False):
    longitude = [0., 37., 131., 206., 298., 360.] if irregular else np.linspace(0., 360., 13)
    latitude = [-60., -27., -3., 15., 56.] if irregular else np.linspace(-60., 60., 7)
    shape = (len(longitude) - 1, len(latitude) - 1)
    depth = np.full(shape, 50.)
    if partial:
        depth[1:3, 1:3] = [[22., 3.], [0., 9.]]
    geometry = build_geometry(longitude, latitude, [0., 7., 21., 50.], depth)
    phase = 2. * np.pi * np.arange(shape[0])[:, None] / shape[0]
    eta = .2 * np.cos(phase) * np.cos(np.radians(.5 * (np.asarray(latitude[:-1]) + latitude[1:])))[None, :]
    volume = surface_volume(geometry, jnp.asarray(eta * (depth > 0.)))
    return geometry, volume


@pytest.mark.parametrize("irregular", [False, True])
@pytest.mark.parametrize("partial", [False, True])
def test_actual_fast_pressure_work_closes_with_independent_physical_mass(irregular, partial):
    geometry, volume = _fixture(irregular, partial)
    east_area, north_area, east_mass, north_mass = _mass_oracle(geometry, volume)
    active = geometry._replace(east_area=east_area, north_area=north_area)
    eta = np.asarray(volume[..., 0]) / geometry.area - geometry.thickness[..., 0]
    zero = jnp.zeros_like(jnp.asarray(eta))
    result = jax.jit(subcycle_barotropic, static_argnames="nsub")(active, jnp.asarray(eta), zero, zero, 1., 1)
    assert bool(result.valid)
    random = np.random.default_rng(963)
    for east, north in ((random.normal(size=volume.shape), random.normal(size=volume.shape)),
                        (np.ones(volume.shape), np.ones(volume.shape))):
        east_flux = np.sum(east_area * east, axis=-1)
        north_flux = np.sum(north_area * north, axis=-1)
        outflow = east_flux - np.roll(east_flux, 1, axis=0) + north_flux
        outflow[:, 1:] -= north_flux[:, :-1]
        potential = -9.81 * eta * outflow
        kinetic_east = east_mass * east * np.asarray(result.east_velocity)[..., None]
        kinetic_north = north_mass * north * np.asarray(result.north_velocity)[..., None]
        scale = np.sum(np.abs(potential)) + np.sum(np.abs(kinetic_east)) + np.sum(np.abs(kinetic_north))
        assert abs(np.sum(potential) + np.sum(kinetic_east) + np.sum(kinetic_north)) <= 1e-12 * scale


@pytest.mark.parametrize("irregular", [False, True])
def test_hydrostatic_reference_load_and_fast_gravity_use_same_contact_mass(irregular):
    geometry, volume = _fixture(irregular, partial=True)
    east_area, north_area, east_mass, north_mass = _mass_oracle(geometry, volume)
    eta = np.asarray(volume[..., 0]) / geometry.area - geometry.thickness[..., 0]
    active = geometry._replace(east_area=east_area, north_area=north_area)
    zero = jnp.zeros_like(jnp.asarray(eta))
    fast = subcycle_barotropic(active, jnp.asarray(eta), zero, zero, 1., 1)
    pressure = hydrostatic_pressure_force(geometry, volume, jnp.full(volume.shape, 2.), jnp.asarray([2., 0., 0.]))
    assert bool(fast.valid) and bool(pressure.valid)
    differences = (np.roll(eta, -1, axis=0) - eta,
                   np.concatenate((eta[:, 1:], eta[:, -1:]), axis=1) - eta)
    for actual, fast_force, face_area, mass, difference in zip(
            (pressure.east, pressure.north), (fast.east_velocity, fast.north_velocity),
            (east_area, north_area), (east_mass, north_mass), differences):
        expected = -9.81 * 2. / 1025. * face_area * difference[..., None] / np.where(mass > 0., mass, 1.)
        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-18)
        np.testing.assert_allclose(actual, np.where(face_area > 0., np.asarray(fast_force)[..., None] * 2. / 1025., 0.), rtol=1e-12, atol=1e-18)


def test_both_contact_force_components_match_integrated_partial_face_pressure():
    geometry, volume = _fixture(irregular=True, partial=True)
    east_area, north_area, east_mass, north_mass = _mass_oracle(geometry, volume)
    anomaly = np.random.default_rng(964).uniform(-.1, .1, geometry.area.shape)
    density = np.broadcast_to(anomaly[..., None], volume.shape)
    pressure = hydrostatic_pressure_force(geometry, volume, jnp.asarray(density))
    faces = momentum_geometry(geometry, volume)
    eta = np.asarray(volume[..., 0]) / geometry.area - geometry.thickness[..., 0]
    for actual, face_area, mass, start, end, axis in zip(
            (pressure.east, pressure.north), (east_area, north_area), (east_mass, north_mass),
            (faces.east_top, faces.north_top), (faces.east_bottom, faces.north_bottom), (0, 1)):
        neighbor_anomaly = np.roll(anomaly, -1, axis=0) if axis == 0 else np.concatenate((anomaly[:, 1:], anomaly[:, -1:]), axis=1)
        neighbor_eta = np.roll(eta, -1, axis=0) if axis == 0 else np.concatenate((eta[:, 1:], eta[:, -1:]), axis=1)
        common_midpoint = .5 * (np.asarray(start) + np.asarray(end))
        pressure_difference = 9.81 * ((neighbor_anomaly - anomaly)[..., None] * common_midpoint
                                     + (neighbor_anomaly * neighbor_eta - anomaly * eta)[..., None])
        expected = -face_area * pressure_difference / (1025. * np.where(mass > 0., mass, 1.))
        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-18)


@pytest.mark.parametrize("hydrostatic", [False, True])
def test_physical_pressure_mms_refines_at_second_order_on_regular_sphere(hydrostatic):
    errors = []
    for nx, ny in ((24, 12), (48, 24), (96, 48)):
        longitude = np.linspace(0., 2. * np.pi, nx + 1)
        latitude = np.linspace(-np.pi / 3., np.pi / 3., ny + 1)
        geometry = build_geometry(np.degrees(longitude), np.degrees(latitude), [0., 7., 21., 50.], np.full((nx, ny), 50.))
        middle_lon = .5 * (longitude[:-1] + longitude[1:])
        lon_average = (np.sin(longitude[1:]) - np.sin(longitude[:-1])) / np.diff(longitude)
        primitive = .5 * latitude + .25 * np.sin(2. * latitude)
        lat_average = np.diff(primitive) / np.diff(np.sin(latitude))
        scalar = lon_average[:, None] * lat_average[None, :]
        volume = surface_volume(geometry, jnp.zeros((nx, ny)) if hydrostatic else jnp.asarray(scalar))
        east_area, north_area, east_mass, north_mass = _mass_oracle(geometry, volume)
        expected = (9.81 / R_EARTH * np.broadcast_to(np.sin(longitude[1:])[:, None], (nx, ny)),
                    9.81 / R_EARTH * np.cos(middle_lon)[:, None] * np.sin(latitude[1:])[None, :])
        if hydrostatic:
            result = hydrostatic_pressure_force(geometry, volume, jnp.asarray(np.broadcast_to(scalar[..., None], volume.shape)))
            assert bool(result.valid)
            actual = (np.asarray(result.east), np.asarray(result.north))
            midpoint = np.array([3.5, 14., 35.5])
            expected = tuple(value[..., None] * midpoint / 1025. for value in expected)
            masses = (east_mass, north_mass)
        else:
            active = geometry._replace(east_area=east_area, north_area=north_area)
            zero = jnp.zeros((nx, ny))
            result = subcycle_barotropic(active, jnp.asarray(scalar), zero, zero, 1., 1)
            assert bool(result.valid)
            actual = (np.asarray(result.east_velocity), np.asarray(result.north_velocity))
            masses = tuple(np.sum(mass, axis=-1) for mass in (east_mass, north_mass))
        errors.append([np.sqrt(np.sum(mass * (observed - analytic) ** 2) / np.sum(mass * analytic ** 2))
                       for mass, observed, analytic in zip(masses, actual, expected)])
    ratios = np.asarray(errors[:-1]) / np.asarray(errors[1:])
    assert np.all(ratios >= 3.5), (errors, ratios)


@pytest.mark.parametrize("target", ["hydrostatic", "fast"])
def test_moving_contact_pressure_and_fast_gradient_have_consistent_local_adjoint(target):
    geometry, unused_volume = _fixture(irregular=True, partial=True)
    wet = geometry.thickness[..., 0] > 0.
    random = np.random.default_rng(965)
    eta = jnp.asarray(random.uniform(-.1, .1, wet.shape) * wet)
    direction = jnp.asarray(random.normal(size=wet.shape) * wet)
    output_shape = geometry.thickness.shape if target == "hydrostatic" else geometry.area.shape
    cotangent = jnp.asarray(random.normal(size=output_shape + (2,)))

    def force(surface):
        volume = surface_volume(geometry, surface)
        if target == "hydrostatic":
            result = hydrostatic_pressure_force(geometry, volume, jnp.full(volume.shape, 2.), jnp.asarray([2., 0., 0.]))
            return jnp.stack((result.east, result.north), axis=-1)
        faces = momentum_geometry(geometry, volume)
        active = geometry._replace(east_area=faces.east_area, north_area=faces.north_area)
        zero = jnp.zeros_like(surface)
        result = subcycle_barotropic(active, surface, zero, zero, 1., 1)
        return jnp.stack((result.east_velocity, result.north_velocity), axis=-1)

    unused, derivative = jax.jvp(force, (eta,), (direction,))
    epsilon = 1e-4
    difference = (force(eta + epsilon * direction) - force(eta - epsilon * direction)) / (2. * epsilon)
    assert np.all(np.isfinite(derivative))
    assert np.linalg.norm(derivative - difference) / np.linalg.norm(derivative) <= 1e-6
    unused, pullback = jax.vjp(force, eta)
    forward, reverse = float(jnp.vdot(cotangent, derivative)), float(jnp.vdot(pullback(cotangent)[0], direction))
    assert abs(forward - reverse) <= 1e-12 * max(abs(forward), abs(reverse))
