"""Independent pressure/rotation/active layer-coupling migration witnesses."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from cgrid_momentum import (
    LayerState,
    coriolis_tendency,
    hydrostatic_pressure_force,
    linear_momentum_surface_step,
    momentum_geometry,
    rotate_coriolis,
)
from config import OMEGA
from finite_volume import ExtensiveState, build_geometry, surface_volume

jax.config.update("jax_enable_x64", True)


def _fixture(stair=True, eta=None):
    depth = np.full((12, 6), 50.)
    if stair:
        depth[2:5, 1:4] = 27.
        depth[6:8, 2:4] = 2.
        depth[9:11, 1:3] = 0.
    geometry = build_geometry(np.linspace(0., 360., 13), np.linspace(-30., 30., 7),
                              [0., 5., 20., 50.], depth)
    surface = np.zeros(depth.shape) if eta is None else np.broadcast_to(eta, depth.shape) * (depth > 0.)
    volume = surface_volume(geometry, jnp.asarray(surface))
    return geometry, volume


def _reference_average(geometry, volume, coefficients):
    height = np.asarray(volume) / geometry.area[..., None]
    top = np.broadcast_to(geometry.interfaces[:-1], height.shape).copy()
    top[..., 0] = geometry.thickness[..., 0] - height[..., 0]
    center = top + .5 * height
    return coefficients[0] + coefficients[1] * center + coefficients[2] * (center ** 2 + height ** 2 / 12.)


def _rotation_oracle(geometry, volume, coriolis):
    area = np.asarray(geometry.area)
    height = np.asarray(volume) / area[..., None]
    top = np.broadcast_to(geometry.interfaces[:-1], height.shape).copy()
    top[..., 0] = geometry.thickness[..., 0] - height[..., 0]
    bottom = top + height
    latitude = np.asarray(geometry.latitude_edges)
    middle = .5 * (latitude[:-1] + latitude[1:])
    south_area = area * ((np.sin(middle) - np.sin(latitude[:-1])) / (np.sin(latitude[1:]) - np.sin(latitude[:-1])))[None, :]
    north_area = area - south_area
    shape = height.shape
    east_top, east_bottom = np.zeros(shape), np.zeros(shape)
    north_top, north_bottom = np.zeros(shape), np.zeros(shape)
    east_mass, north_mass = np.zeros(shape), np.zeros(shape)
    for position in np.ndindex(shape):
        longitude_index, latitude_index, layer_index = position
        adjacent_east = ((longitude_index + 1) % shape[0], latitude_index, layer_index)
        if geometry.thickness[position] > 0. and geometry.thickness[adjacent_east] > 0.:
            east_top[position] = max(top[position], top[adjacent_east])
            east_bottom[position] = min(bottom[position], bottom[adjacent_east])
            east_mass[position] = .5 * (area[position[:2]] + area[adjacent_east[:2]]) * max(east_bottom[position] - east_top[position], 0.)
        if latitude_index + 1 < shape[1]:
            adjacent_north = (longitude_index, latitude_index + 1, layer_index)
            if geometry.thickness[position] > 0. and geometry.thickness[adjacent_north] > 0.:
                north_top[position] = max(top[position], top[adjacent_north])
                north_bottom[position] = min(bottom[position], bottom[adjacent_north])
                north_mass[position] = (north_area[position[:2]] + south_area[adjacent_north[:2]]) * max(north_bottom[position] - north_top[position], 0.)
    size = int(np.prod(shape))
    matrix = np.zeros((2 * size, 2 * size))
    prescribed = None if coriolis is None else np.broadcast_to(coriolis, area.shape)
    for position in np.ndindex(shape):
        longitude_index, latitude_index, layer_index = position
        for east_index in ((longitude_index - 1) % shape[0], longitude_index):
            east_position = (east_index, latitude_index, layer_index)
            if east_mass[east_position] == 0.:
                continue
            for north_index, quadrant_area, boundary in (
                    (latitude_index - 1, .5 * south_area[position[:2]], latitude[latitude_index]),
                    (latitude_index, .5 * north_area[position[:2]], latitude[latitude_index + 1])):
                north_position = (longitude_index, north_index, layer_index)
                if north_index < 0 or north_mass[north_position] == 0.:
                    continue
                common = max(min(east_bottom[east_position], north_bottom[north_position])
                             - max(east_top[east_position], north_top[north_position]), 0.)
                coriolis_mean = OMEGA * (np.sin(middle[latitude_index]) + np.sin(boundary)) if prescribed is None else prescribed[position[:2]]
                coupling = quadrant_area * common * coriolis_mean
                east_flat = np.ravel_multi_index(east_position, shape)
                north_flat = np.ravel_multi_index(north_position, shape)
                matrix[east_flat, size + north_flat] += coupling / east_mass[east_position]
                matrix[size + north_flat, east_flat] -= coupling / north_mass[north_position]
    return east_mass, north_mass, matrix


@pytest.mark.parametrize("coefficients", [np.array([1., .01, 0.]), np.array([1., .01, .00002])])
def test_partial_cell_background_balanced_not_pressure_at_unequal_centers(coefficients):
    geometry, volume = _fixture()
    density = _reference_average(geometry, volume, coefficients)
    density[geometry.thickness == 0.] = np.nan
    result = jax.jit(hydrostatic_pressure_force)(geometry, volume, jnp.asarray(density), jnp.asarray(coefficients))
    assert bool(result.valid)
    assert np.max(np.abs(result.east)) <= 1e-12
    assert np.max(np.abs(result.north)) <= 1e-12
    centers = geometry.center_depth
    center_pressure = 9.81 * (coefficients[0] * centers + .5 * coefficients[1] * centers ** 2 + coefficients[2] * centers ** 3 / 3.)
    naive = -(np.roll(center_pressure, -1, axis=0) - center_pressure) / (1025. * geometry.east_distance[..., None])
    assert np.max(np.abs(naive[geometry.east_area > 0.])) > 1e-9


def test_nonzero_pressure_signal_agrees_with_common_face_integral():
    geometry, volume = _fixture()
    coefficients = np.array([1., .01, .00002])
    anomaly = .1 * np.sin(2. * np.pi * np.arange(12) / 12.)[:, None, None]
    density = _reference_average(geometry, volume, coefficients) + anomaly
    result = hydrostatic_pressure_force(geometry, volume, jnp.asarray(density), jnp.asarray(coefficients))
    assert bool(result.valid)
    common_height = np.minimum(geometry.thickness, np.roll(geometry.thickness, -1, axis=0))
    common_midpoint = geometry.interfaces[:-1] + .5 * common_height
    dual_area = .5 * (geometry.area + np.roll(geometry.area, -1, axis=0))
    mass = dual_area[..., None] * common_height
    expected = -9.81 * geometry.east_area * (np.roll(anomaly, -1, axis=0) - anomaly) * common_midpoint / (1025. * np.where(mass > 0., mass, 1.))
    expected = np.where(geometry.east_area > 0., expected, 0.)
    np.testing.assert_allclose(result.east, expected, rtol=1e-12, atol=1e-14)
    assert np.max(np.abs(result.east)) > 1e-9


def test_moving_top_geometry_and_reference_surface_load_are_not_discarded():
    eta = .2 * np.cos(2. * np.pi * np.arange(12)[:, None] / 12.)
    geometry, volume = _fixture(eta=eta)
    coefficients = np.array([2., .01, .00002])
    density = _reference_average(geometry, volume, coefficients)
    result = hydrostatic_pressure_force(geometry, volume, jnp.asarray(density), jnp.asarray(coefficients))
    assert bool(result.valid)
    clean_eta = np.broadcast_to(eta, geometry.area.shape) * (geometry.thickness[..., 0] > 0.)
    def primitive(depth):
        return coefficients[0] * depth + .5 * coefficients[1] * depth ** 2 + coefficients[2] * depth ** 3 / 3.

    load = -9.81 * primitive(-clean_eta)
    latitude = geometry.latitude_edges
    face_length = 6.371e6 * np.diff(latitude)[None, :]
    dual_area = .5 * (geometry.area + np.roll(geometry.area, -1, axis=0))
    expected = -face_length[..., None] * (np.roll(load, -1, axis=0) - load)[..., None] / (1025. * dual_area[..., None])
    expected = np.where(geometry.east_area > 0., expected, 0.)
    np.testing.assert_allclose(result.east, expected, rtol=1e-12, atol=1e-14)
    common_top = np.maximum(-clean_eta, -np.roll(clean_eta, -1, axis=0))
    common_bottom = np.minimum(geometry.thickness[..., 0], np.roll(geometry.thickness[..., 0], -1, axis=0))
    face_height = np.maximum(common_bottom - common_top, 0.)
    expected_area = np.where(geometry.east_area[..., 0] > 0., face_height * 6.371e6 * np.radians(10.), 0.)
    np.testing.assert_allclose(result.east_area[..., 0], expected_area, rtol=1e-13)
    assert np.max(np.abs(result.east)) > 1e-10


@pytest.mark.parametrize("invalid", ["wet_nan", "lower_volume", "depleted", "density_shape", "volume_dtype", "gravity"])
def test_pressure_invalid_inputs_reject_or_return_invalid(invalid):
    geometry, volume = _fixture()
    density = jnp.ones_like(volume)
    gravity = 9.81
    if invalid == "wet_nan":
        density = density.at[0, 0, 0].set(jnp.nan)
    elif invalid == "lower_volume":
        volume = volume.at[0, 0, 1].multiply(1.1)
    elif invalid == "depleted":
        volume = volume.at[0, 0, 0].set(-1.)
    elif invalid == "density_shape":
        density = density[..., :-1]
    elif invalid == "volume_dtype":
        volume = volume.astype(jnp.float32)
    else:
        gravity = -1.
    if invalid in ("density_shape", "volume_dtype"):
        with pytest.raises(ValueError):
            hydrostatic_pressure_force(geometry, volume, density, gravity=gravity)
    else:
        assert not bool(hydrostatic_pressure_force(geometry, volume, density, gravity=gravity).valid)


@pytest.mark.parametrize("dtype,tolerance", [(jnp.float64, 1e-12), (jnp.float32, 2e-6)])
def test_coriolis_weighted_work_and_midpoint_rotation_energy(dtype, tolerance):
    geometry, volume = _fixture()
    random = np.random.default_rng(942)
    east = jnp.asarray(np.where(geometry.east_area > 0., random.normal(size=volume.shape), np.nan), dtype)
    north = jnp.asarray(np.where(geometry.north_area > 0., random.normal(size=volume.shape), np.nan), dtype)
    force_east, force_north = coriolis_tendency(geometry, east, north, coriolis=.001)
    clean_east = np.nan_to_num(np.asarray(east, dtype=np.float64))
    clean_north = np.nan_to_num(np.asarray(north, dtype=np.float64))
    east_weight, north_weight, unused_matrix = _rotation_oracle(geometry, volume, .001)
    work = np.sum(east_weight * clean_east * force_east) + np.sum(north_weight * clean_north * force_north)
    scale = np.sum(np.abs(east_weight * clean_east * force_east)) + np.sum(np.abs(north_weight * clean_north * force_north))
    assert abs(work) <= 1e-12 * scale
    rotated = jax.jit(rotate_coriolis)(geometry, east, north, 10000., coriolis=.001)
    assert bool(rotated.valid)
    assert rotated.east_velocity.dtype == dtype
    assert float(rotated.energy_relative_change) <= tolerance
    assert float(rotated.solve_relative_residual) <= 1e-12


def test_rotation_matches_independent_dense_physical_velocity_system():
    geometry = build_geometry(np.linspace(0., 360., 5), [-20., -5., 10., 25.],
                              [0., 10., 50.], np.full((4, 3), 50.))
    shape = geometry.thickness.shape
    size = int(np.prod(shape))
    east_weight, north_weight, matrix = _rotation_oracle(geometry, geometry.area[..., None] * geometry.thickness, .15)
    random = np.random.default_rng(943)
    east = np.where(east_weight > 0., random.normal(size=shape), 0.)
    north = np.where(north_weight > 0., random.normal(size=shape), 0.)
    initial = np.r_[east.ravel(), north.ravel()]
    dt = 2.3
    identity = np.eye(2 * size)
    expected = np.linalg.solve(identity - .5 * dt * matrix, (identity + .5 * dt * matrix) @ initial)
    result = rotate_coriolis(geometry, jnp.asarray(east), jnp.asarray(north), dt, coriolis=.15)
    assert bool(result.valid)
    np.testing.assert_allclose(result.east_velocity, expected[:size].reshape(shape), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(result.north_velocity, expected[size:].reshape(shape), rtol=1e-12, atol=1e-12)


def test_rotation_cap_failure_is_not_accepted():
    geometry, volume = _fixture()
    random = np.random.default_rng(944)
    east = jnp.asarray(random.normal(size=volume.shape))
    north = jnp.asarray(random.normal(size=volume.shape))
    assert not bool(rotate_coriolis(geometry, east, north, 10000., coriolis=.001, maxiter=1).valid)


def test_uniform_interior_coriolis_has_correct_sign_and_consistency():
    geometry = build_geometry(np.linspace(0., 360., 9), np.linspace(-.001, .001, 9),
                              [0., 50.], np.full((8, 8), 50.))
    east = jnp.full(geometry.thickness.shape, 2.)
    north = jnp.where(geometry.north_area > 0., 3., 0.)
    acceleration_east, acceleration_north = coriolis_tendency(geometry, east, north, coriolis=.001)
    np.testing.assert_allclose(acceleration_east[:, 1:-2], .003, rtol=1e-8)
    np.testing.assert_allclose(acceleration_north[:, 1:-2], -.002, rtol=1e-8)


@pytest.mark.parametrize("orientation", ["east", "north"])
def test_partial_faces_uniform_velocity_common_depth_not_neighbor_mass(orientation):
    depth = np.full((8, 8), 50.)
    if orientation == "east":
        depth[3, :] = 21.
    else:
        depth[:, 3] = 21.
    geometry = build_geometry(np.linspace(0., 360., 9), np.linspace(-.001, .001, 9),
                              [0., 5., 20., 50.], depth)
    zero = jnp.zeros(geometry.thickness.shape)
    if orientation == "east":
        north = jnp.where(geometry.north_area > 0., 2., 0.)
        acceleration, unused = coriolis_tendency(geometry, zero, north, coriolis=.001)
        actual = acceleration[2:4, 2:6, 2]
        expected = .002
    else:
        east = jnp.where(geometry.east_area > 0., 2., 0.)
        unused, acceleration = coriolis_tendency(geometry, east, zero, coriolis=.001)
        actual = acceleration[2:6, 2:4, 2]
        expected = -.002
    np.testing.assert_allclose(actual, expected, rtol=1e-8, atol=0.)


@pytest.mark.parametrize("coriolis", [None, .001, "variable"])
@pytest.mark.parametrize("dtype", [jnp.float64, jnp.float32])
def test_physical_duals_moving_overlaps_match_independent_quadrant_oracle(coriolis, dtype):
    depth = np.array([[50., 9., 4., 50.], [23., 50., 0., 15.], [7., 29., 50., 2.],
                      [50., 50., 13., 50.], [16., 50., 50., 50.]])
    geometry = build_geometry([0., 37., 131., 206., 298., 360.], [-60., -27., -3., 15., 56.], [0., 7., 21., 50.], depth)
    random = np.random.default_rng(949)
    eta = jnp.asarray(random.uniform(-.1, .1, depth.shape) * (depth > 0.))
    volume = surface_volume(geometry, eta)
    parameter = random.uniform(-.001, .001, depth.shape) if isinstance(coriolis, str) else coriolis
    east_weight, north_weight, matrix = _rotation_oracle(geometry, volume, parameter)
    east = jnp.asarray(np.where(east_weight > 0., random.normal(size=volume.shape), np.nan), dtype)
    north = jnp.asarray(np.where(north_weight > 0., random.normal(size=volume.shape), np.nan), dtype)
    faces = jax.jit(momentum_geometry)(geometry, volume)
    assert bool(faces.valid)
    np.testing.assert_allclose(faces.east_volume, east_weight, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(faces.north_volume, north_weight, rtol=1e-12, atol=1e-12)
    actual_east, actual_north = coriolis_tendency(geometry, east, north, parameter, volume=volume)
    initial = np.r_[np.nan_to_num(np.asarray(east)).ravel(), np.nan_to_num(np.asarray(north)).ravel()]
    expected = matrix @ initial
    np.testing.assert_allclose(np.r_[np.asarray(actual_east).ravel(), np.asarray(actual_north).ravel()], expected, rtol=1e-12, atol=1e-12)
    rotated = rotate_coriolis(geometry, east, north, 600., parameter, volume=volume)
    assert bool(rotated.valid)
    before = np.sum(east_weight * np.nan_to_num(np.asarray(east, dtype=np.float64)) ** 2
                    + north_weight * np.nan_to_num(np.asarray(north, dtype=np.float64)) ** 2)
    after = np.sum(east_weight * np.asarray(rotated.east_velocity, dtype=np.float64) ** 2
                   + north_weight * np.asarray(rotated.north_velocity, dtype=np.float64) ** 2)
    assert abs(after - before) / before <= (2e-6 if dtype == jnp.float32 else 1e-12)


def test_common_overlap_tendency_bound_follows_geometry_not_force_clipping():
    geometry, volume = _fixture()
    random = np.random.default_rng(950)
    east = jnp.asarray(random.normal(size=volume.shape))
    north = jnp.asarray(random.normal(size=volume.shape))
    acceleration_east, acceleration_north = coriolis_tendency(geometry, east, north, .001)
    assert np.max(np.abs(acceleration_east)) <= .001 * np.max(np.abs(north)) * (1. + 1e-12)
    assert np.max(np.abs(acceleration_north)) <= .001 * np.max(np.abs(east)) * (1. + 1e-12)


@pytest.mark.parametrize("invalid", ["nan", "deep", "dry", "dtype", "shape"])
def test_rotation_physical_volume_invalid_inputs_reject_or_return_invalid(invalid):
    geometry, volume = _fixture()
    zero = jnp.zeros_like(volume)
    if invalid == "nan":
        volume = volume.at[0, 0, 0].set(jnp.nan)
    elif invalid == "deep":
        volume = volume.at[0, 0, 1].multiply(1.01)
    elif invalid == "dry":
        volume = volume.at[9, 1, 0].set(1.)
    elif invalid == "dtype":
        volume = volume.astype(jnp.float32)
    else:
        volume = volume[..., :-1]
    if invalid in ("dtype", "shape"):
        with pytest.raises(ValueError):
            rotate_coriolis(geometry, zero, zero, 60., volume=volume)
    else:
        assert not bool(momentum_geometry(geometry, volume).valid)
        assert not bool(rotate_coriolis(geometry, zero, zero, 60., volume=volume).valid)


def test_moving_volume_rotation_local_derivatives_including_dry_cells():
    geometry, unused_volume = _fixture()
    random = np.random.default_rng(951)
    wet = geometry.thickness[..., 0] > 0.
    eta = jnp.asarray(random.uniform(-.1, .1, wet.shape) * wet)
    direction = jnp.asarray(random.normal(size=wet.shape) * wet)
    east = jnp.asarray(.01 * random.normal(size=geometry.thickness.shape))
    north = jnp.asarray(.01 * random.normal(size=geometry.thickness.shape))
    cotangent = jnp.asarray(random.normal(size=geometry.thickness.shape + (2,)))

    def rotation(surface):
        volume = surface_volume(geometry, surface)
        result = rotate_coriolis(geometry, east, north, 60., .001, volume=volume)
        return jnp.stack((result.east_velocity, result.north_velocity), axis=-1)

    _, derivative = jax.jvp(rotation, (eta,), (direction,))
    epsilon = 1e-3
    difference = (rotation(eta + epsilon * direction) - rotation(eta - epsilon * direction)) / (2. * epsilon)
    assert np.all(np.isfinite(derivative))
    assert np.linalg.norm(derivative - difference) / np.linalg.norm(derivative) <= 1e-6
    _, pullback = jax.vjp(rotation, eta)
    forward = float(jnp.vdot(cotangent, derivative))
    reverse = float(jnp.vdot(pullback(cotangent)[0], direction))
    assert np.isfinite(reverse)
    assert abs(forward - reverse) <= 1e-12 * max(abs(forward), abs(reverse))


@pytest.mark.parametrize("dtype", [jnp.float64, jnp.float32])
def test_actual_layer_pressure_coupling_mean_and_shear_force_counted_once(dtype):
    geometry, volume = _fixture(stair=False)
    concentration = jnp.broadcast_to(jnp.asarray([12., 35.]), volume.shape + (2,))
    inventory = ExtensiveState(volume, volume[..., None] * concentration)
    zero = jnp.zeros(volume.shape, dtype)
    state = LayerState(inventory, zero, zero)
    anomaly = jnp.broadcast_to(jnp.sin(2. * jnp.pi * jnp.arange(12)[:, None, None] / 12.), volume.shape)
    result = linear_momentum_surface_step(geometry, state, anomaly, 60., 4, gravity=0., pressure_gravity=9.81, coriolis=0.)
    assert bool(result.valid)
    expected = 60. * result.pressure.east
    np.testing.assert_allclose(result.state.east_velocity, expected, rtol=2e-6 if dtype == jnp.float32 else 1e-12, atol=1e-15)
    assert np.max(np.abs(result.state.east_velocity)) > 1e-6
    assert np.max(np.ptp(result.state.east_velocity, axis=-1)) > 1e-6
    np.testing.assert_allclose(np.sum(result.state.inventory.content, axis=(0, 1, 2)),
                               np.sum(inventory.content, axis=(0, 1, 2)), rtol=1e-12)
    wet = volume > 0.
    actual = np.asarray(result.state.inventory.content)[wet] / np.asarray(result.state.inventory.volume)[wet, None]
    np.testing.assert_allclose(actual, np.broadcast_to([12., 35.], actual.shape), rtol=1e-12, atol=1e-12)
    assert float(result.surface_error) <= 1e-12


def test_pressure_and_rotation_local_derivatives():
    geometry, volume = _fixture(stair=False)
    random = np.random.default_rng(945)
    density = jnp.asarray(random.normal(size=volume.shape))
    direction = jnp.asarray(random.normal(size=volume.shape))
    cotangent = jnp.asarray(random.normal(size=volume.shape))

    def pressure(density_field):
        return hydrostatic_pressure_force(geometry, volume, density_field).east

    _, derivative = jax.jvp(pressure, (density,), (direction,))
    epsilon = 1e-4
    difference = (pressure(density + epsilon * direction) - pressure(density - epsilon * direction)) / (2. * epsilon)
    assert np.linalg.norm(derivative - difference) / np.linalg.norm(derivative) <= 1e-6
    _, pullback = jax.vjp(pressure, density)
    np.testing.assert_allclose(jnp.vdot(cotangent, derivative), jnp.vdot(pullback(cotangent)[0], direction), rtol=1e-12)
    east, north = density, direction

    def rotation(east_field):
        return rotate_coriolis(geometry, east_field, north, 60., coriolis=.001).east_velocity

    _, derivative = jax.jvp(rotation, (east,), (direction,))
    difference = (rotation(east + epsilon * direction) - rotation(east - epsilon * direction)) / (2. * epsilon)
    assert np.linalg.norm(derivative - difference) / np.linalg.norm(derivative) <= 1e-6
    _, pullback = jax.vjp(rotation, east)
    np.testing.assert_allclose(jnp.vdot(cotangent, derivative), jnp.vdot(pullback(cotangent)[0], direction), rtol=1e-12)


@pytest.mark.parametrize("scale", [1e-12, 1., 1e12])
def test_rotation_adjoint_uses_reverse_rhs_scale_not_primal_scale(scale):
    geometry, volume = _fixture(stair=False)
    random = np.random.default_rng(946)
    east = jnp.asarray(random.normal(size=volume.shape))
    north = jnp.asarray(random.normal(size=volume.shape))
    direction = jnp.asarray(random.normal(size=volume.shape))
    cotangent = scale * jnp.asarray(random.normal(size=volume.shape))

    def rotation(east_field):
        return rotate_coriolis(geometry, east_field, north, 60., coriolis=.001).east_velocity

    _, derivative = jax.jvp(rotation, (east,), (direction,))
    _, pullback = jax.vjp(rotation, east)
    forward = float(jnp.vdot(cotangent, derivative))
    reverse = float(jnp.vdot(pullback(cotangent)[0], direction))
    assert abs(forward - reverse) <= 1e-12 * max(abs(forward), abs(reverse))


@pytest.mark.parametrize("nsub", [0, -1, True, 1.5, None])
def test_coupled_substep_count_rejects_before_division(nsub):
    geometry, volume = _fixture()
    zero = jnp.zeros_like(volume)
    state = LayerState(ExtensiveState(volume, volume[..., None] * 35.), zero, zero)
    with pytest.raises(ValueError, match="positive static integer"):
        linear_momentum_surface_step(geometry, state, zero, 60., nsub)


def test_coupled_material_source_closure_and_constant_concentration():
    geometry, volume = _fixture()
    concentration = jnp.asarray([12., 35.])
    inventory = ExtensiveState(volume, volume[..., None] * concentration)
    zero = jnp.zeros_like(volume)
    state = LayerState(inventory, zero, zero)
    source = zero.at[..., 0].set(jnp.where(volume[..., 0] > 0., 1e-8 * geometry.area, 0.))
    content_source = source[..., None] * concentration
    result = linear_momentum_surface_step(geometry, state, zero, 60., 4, gravity=0., coriolis=0.,
                                          volume_source=source, content_source=content_source)
    assert bool(result.valid)
    np.testing.assert_allclose(result.state.inventory.volume, volume + 60. * source, rtol=1e-12)
    np.testing.assert_allclose(result.state.inventory.content, inventory.content + 60. * content_source, rtol=1e-12)
    wet = volume > 0.
    actual = np.asarray(result.state.inventory.content)[wet] / np.asarray(result.state.inventory.volume)[wet, None]
    np.testing.assert_allclose(actual, np.broadcast_to(concentration, actual.shape), rtol=1e-12)
    assert float(result.surface_error) <= 1e-12


@pytest.mark.parametrize("invalid", ["deep", "dry", "nan"])
def test_coupled_invalid_material_sources_do_not_pass(invalid):
    geometry, volume = _fixture()
    zero = jnp.zeros_like(volume)
    state = LayerState(ExtensiveState(volume, volume[..., None] * 35.), zero, zero)
    index = (0, 0, 1) if invalid == "deep" else (9, 1, 0) if invalid == "dry" else (0, 0, 0)
    source = zero.at[index].set(jnp.nan if invalid == "nan" else 1.)
    result = linear_momentum_surface_step(geometry, state, zero, 60., 4, gravity=0., coriolis=0., volume_source=source)
    assert not bool(result.valid)


def test_active_coupled_predictor_local_derivatives():
    geometry, volume = _fixture(stair=False)
    random = np.random.default_rng(947)
    east = jnp.asarray(.001 * random.normal(size=volume.shape))
    north = jnp.asarray(.001 * random.normal(size=volume.shape))
    state = LayerState(ExtensiveState(volume, volume[..., None] * jnp.asarray([12., 35.])), east, north)
    density = jnp.asarray(.1 * random.normal(size=volume.shape))
    direction = jnp.asarray(random.normal(size=volume.shape))
    cotangent = jnp.asarray(random.normal(size=volume.shape + (2,)))

    def predictor(density_field):
        result = linear_momentum_surface_step(geometry, state, density_field, 60., 4, coriolis=.001)
        return jnp.stack((result.state.east_velocity, result.state.north_velocity), axis=-1)

    accepted = linear_momentum_surface_step(geometry, state, density, 60., 4, coriolis=.001)
    assert bool(accepted.valid)
    _, derivative = jax.jvp(predictor, (density,), (direction,))
    epsilon = 1e-4
    difference = (predictor(density + epsilon * direction) - predictor(density - epsilon * direction)) / (2. * epsilon)
    assert np.linalg.norm(derivative - difference) / np.linalg.norm(derivative) <= 1e-6
    _, pullback = jax.vjp(predictor, density)
    forward = float(jnp.vdot(cotangent, derivative))
    reverse = float(jnp.vdot(pullback(cotangent)[0], direction))
    assert abs(forward - reverse) <= 1e-12 * max(abs(forward), abs(reverse))
