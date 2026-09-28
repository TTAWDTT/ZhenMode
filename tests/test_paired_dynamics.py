"""Independent physical-field matrices and coupled midpoint dynamics."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from cgrid_momentum import LayerState
from config import OMEGA
from finite_volume import ExtensiveState, build_geometry, surface_volume
from paired_dynamics import (
    VelocityPair,
    advance_paired_surface,
    column_outflow,
    kinetic_energy,
    mass_action,
    paired_momentum_surface_step,
    paired_operators,
    rotation_action,
    surface_pressure_force,
)

jax.config.update("jax_enable_x64", True)


def fixture(partial=True):
    depth = np.full((5, 4), 50.)
    if partial:
        depth[1, 1], depth[2, 2], depth[3, 1] = 22., 3., 0.
    geometry = build_geometry([0., 37., 131., 206., 298., 360.],
                              [-60., -27., -3., 15., 56.], [0., 7., 21., 50.],
                              depth, radius=2.1e6)
    eta = .2 * np.cos(np.arange(5)[:, None]) * np.ones_like(depth) * (depth > 0.)
    volume = surface_volume(geometry, jnp.asarray(eta))
    return geometry, volume, eta


def dense_oracle(geometry, volume, coriolis=None, order=48):
    area, height = np.asarray(geometry.area), np.asarray(volume) / geometry.area[..., None]
    shape, size = height.shape, height.size
    top = np.broadcast_to(geometry.interfaces[:-1], shape).copy()
    top[..., 0] = geometry.thickness[..., 0] - height[..., 0]
    bottom = top + height
    nodes, weights = np.polynomial.legendre.leggauss(order)
    fraction, weights = (nodes + 1.) * .5, weights * .5
    longitude_nodes, longitude_weights = np.polynomial.legendre.leggauss(7)
    longitude, longitude_weights = (longitude_nodes + 1.) * .5, longitude_weights * .5
    lat = np.asarray(geometry.latitude_edges)
    radius = geometry.north_width[0, 0] / (lat[1] - lat[0])
    mass, rotation = np.zeros((2 * size, 2 * size)), np.zeros((2 * size, 2 * size))
    divergence = np.zeros((shape[0] * shape[1], 2 * size))
    face_areas = np.zeros((2,) + shape)
    for position in np.ndindex(shape):
        lon_index, lat_index, layer_index = position
        for axis in (0, 1):
            adjacent = ((lon_index + 1) % shape[0], lat_index, layer_index) if axis == 0 else (lon_index, lat_index + 1, layer_index)
            if adjacent[1] >= shape[1] or geometry.thickness[position] == 0. or geometry.thickness[adjacent] == 0.:
                continue
            overlap = max(0., min(bottom[position], bottom[adjacent]) - max(top[position], top[adjacent]))
            width = radius * (lat[lat_index + 1] - lat[lat_index]) if axis == 0 else radius * np.cos(lat[lat_index + 1]) * geometry.east_width[lon_index, lat_index] / (radius * np.cos(.5 * (lat[lat_index] + lat[lat_index + 1])))
            face_areas[(axis,) + position] = width * overlap
            face = axis * size + np.ravel_multi_index(position, shape)
            divergence[np.ravel_multi_index(position[:2], shape[:2]), face] += width * overlap
            divergence[np.ravel_multi_index(adjacent[:2], shape[:2]), face] -= width * overlap
    for position in np.ndindex(shape):
        lon_index, lat_index, layer_index = position
        south, north = lat[lat_index:lat_index + 2]
        delta_phi = north - south
        delta_lambda = geometry.east_width[lon_index, lat_index] / (radius * np.cos(.5 * (south + north)))
        phi = np.arcsin(np.sin(south) + fraction * (np.sin(north) - np.sin(south)))
        beta = delta_phi / delta_lambda * (fraction - (phi - south) / delta_phi) / np.cos(phi)
        north_basis = np.stack((-beta, beta, (1. - fraction) * np.cos(south) / np.cos(phi), fraction * np.cos(north) / np.cos(phi)), axis=-1)
        east_basis = np.zeros((longitude.size, 4))
        east_basis[:, 0], east_basis[:, 1] = 1. - longitude, longitude
        frequency = 2. * OMEGA * np.sin(phi) if coriolis is None else np.broadcast_to(coriolis, shape[:2])[position[:2]]
        west_position = ((lon_index - 1) % shape[0], lat_index, layer_index)
        south_position = (lon_index, lat_index - 1, layer_index) if lat_index else position
        local_positions = (west_position, position, south_position, position)
        local_axes = (0, 0, 1, 1)
        intervals, indices = [], []
        for side, (face_position, axis) in enumerate(zip(local_positions, local_axes)):
            adjacent = ((face_position[0] + 1) % shape[0], face_position[1], layer_index) if axis == 0 else (lon_index, face_position[1] + 1, layer_index)
            opened = (side != 2 or lat_index > 0) and face_areas[(axis,) + face_position] > 0.
            intervals.append((max(top[face_position], top[adjacent]), min(bottom[face_position], bottom[adjacent])) if opened else (0., 0.))
            indices.append(axis * size + np.ravel_multi_index(face_position, shape))
        for left in range(4):
            for right in range(4):
                overlap = max(0., min(intervals[left][1], intervals[right][1]) - max(intervals[left][0], intervals[right][0]))
                kinetic = east_basis[:, left, None] * east_basis[:, right, None] + north_basis[None, :, left] * north_basis[None, :, right]
                spin = frequency * (east_basis[:, left, None] * north_basis[None, :, right] - north_basis[None, :, left] * east_basis[:, right, None])
                mass[indices[left], indices[right]] += area[position[:2]] * overlap * np.sum(longitude_weights[:, None] * weights[None, :] * kinetic)
                rotation[indices[left], indices[right]] += area[position[:2]] * overlap * np.sum(longitude_weights[:, None] * weights[None, :] * spin)
    return mass, rotation, divergence, face_areas


def flatten(velocity):
    return np.concatenate((np.asarray(velocity.east).ravel(), np.asarray(velocity.north).ravel()))


@pytest.mark.parametrize("partial", [False, True])
@pytest.mark.parametrize("coriolis", [None, 1.3e-4])
def test_physical_mass_rotation_and_divergence_match_independent_field_integrals(partial, coriolis):
    geometry, volume, eta = fixture(partial)
    operators = paired_operators(geometry, volume, coriolis=coriolis)
    mass, rotation, divergence, areas = dense_oracle(geometry, volume, coriolis)
    assert bool(operators.valid)
    coefficients = np.random.default_rng(109).normal(size=volume.shape)
    velocity = VelocityPair(jnp.where(areas[0] > 0., coefficients, 0.), jnp.where(areas[1] > 0., coefficients[::-1], 0.))
    state = flatten(velocity)
    np.testing.assert_allclose(flatten(mass_action(operators, velocity)), mass @ state, rtol=1e-12, atol=.05)
    np.testing.assert_allclose(flatten(rotation_action(operators, velocity)), rotation @ state, rtol=1e-12, atol=1e-6)
    np.testing.assert_allclose(np.asarray(column_outflow(operators, velocity)).ravel(), divergence @ state, rtol=1e-12, atol=1e-7)
    np.testing.assert_allclose(flatten(surface_pressure_force(operators, jnp.asarray(eta))), divergence.T @ eta.ravel(), rtol=1e-12, atol=1e-8)
    np.testing.assert_allclose(mass, mass.T, rtol=1e-14, atol=.01)
    np.testing.assert_allclose(rotation, -rotation.T, rtol=1e-14, atol=1e-8)
    active = np.diag(mass) > 0.
    assert np.linalg.eigvalsh(mass[np.ix_(active, active)])[0] > 0.
    assert np.max(np.abs(mass - np.diag(np.diag(mass)))) > .01 * np.max(np.diag(mass))
    higher = paired_operators(geometry, volume, coriolis=coriolis, quadrature_order=48)
    np.testing.assert_allclose(higher.mass, operators.mass, rtol=1e-12, atol=.01)
    np.testing.assert_allclose(higher.rotation, operators.rotation, rtol=1e-12, atol=1e-8)


@pytest.mark.parametrize("gravity", [0., 9.81])
def test_simultaneous_midpoint_matches_dense_oracle_and_actual_mean_q(gravity):
    geometry, volume, eta = fixture()
    mass, rotation, divergence, areas = dense_oracle(geometry, volume)
    active = np.diag(mass) > 0.
    coefficient = np.random.default_rng(231).normal(size=2 * volume.size) * .03
    coefficient[~active] = 0.
    velocity = VelocityPair(jnp.asarray(coefficient[:volume.size].reshape(volume.shape)), jnp.asarray(coefficient[volume.size:].reshape(volume.shape)))
    operators = paired_operators(geometry, volume)
    source = geometry.area * np.sin(np.arange(5)[:, None]) * 1e-6 * (geometry.thickness[..., 0] > 0.)
    force = VelocityPair(operators.east_diagonal * 3e-6, operators.north_diagonal * -2e-6)
    result = advance_paired_surface(operators, jnp.asarray(eta), velocity, 300., 3, gravity=gravity, force=force, volume_source=jnp.asarray(source))
    selected_mass, selected_rotation = mass[np.ix_(active, active)], rotation[np.ix_(active, active)]
    selected_divergence, surface_mass = divergence[:, active], np.diag(np.asarray(geometry.area).ravel())
    operator = np.block([[selected_rotation, gravity * selected_divergence.T], [-selected_divergence, np.zeros_like(surface_mass)]])
    metric = np.block([[selected_mass, np.zeros((active.sum(), surface_mass.shape[0]))], [np.zeros((surface_mass.shape[0], active.sum())), surface_mass]])
    state = np.concatenate((coefficient[active], eta.ravel()))
    held_source = np.concatenate((flatten(force)[active], np.asarray(source).ravel()))
    mean = np.zeros(active.sum())
    for unused_substep in range(3):
        final = np.linalg.solve(metric - 50. * operator, (metric + 50. * operator) @ state + 100. * held_source)
        mean += (state[:active.sum()] + final[:active.sum()]) / 6.
        state = final
    assert bool(result.valid)
    np.testing.assert_allclose(flatten(result.velocity)[active], state[:active.sum()], rtol=1e-11, atol=1e-13)
    np.testing.assert_allclose(result.eta.ravel(), state[active.sum():], rtol=1e-11, atol=1e-13)
    actual_mean = np.concatenate((np.asarray(result.mean_east).ravel(), np.asarray(result.mean_north).ravel()))[active]
    np.testing.assert_allclose(actual_mean, np.concatenate((areas[0].ravel(), areas[1].ravel()))[active] * mean, rtol=1e-11, atol=1e-7)
    assert float(result.energy_work_relative) <= 1e-11
    assert float(result.solve_relative_residual) <= 1e-12
    endpoint = np.concatenate((areas[0].ravel(), areas[1].ravel()))[active] * state[:active.sum()]
    assert np.max(np.abs(actual_mean - endpoint)) > 1.
    initial_energy = .5 * coefficient @ mass @ coefficient + .5 * gravity * np.sum(np.asarray(geometry.area) * eta ** 2)
    endpoint_velocity = flatten(result.velocity)
    final_energy = .5 * endpoint_velocity @ mass @ endpoint_velocity + .5 * gravity * np.sum(np.asarray(geometry.area) * np.asarray(result.eta) ** 2)
    assert abs(final_energy - initial_energy - float(result.external_work)) <= 1e-11 * (initial_energy + abs(float(result.external_work)))


@pytest.mark.parametrize("dtype", [jnp.float64, jnp.float32])
@pytest.mark.parametrize("source_sign", [-1., 0., 1.])
def test_actual_paired_step_advances_inventory_with_same_q_and_signed_source(dtype, source_sign):
    geometry, volume, eta = fixture()
    concentration = jnp.broadcast_to(jnp.array([15., 35.]), volume.shape + (2,))
    velocity = jnp.zeros(volume.shape, dtype)
    state = LayerState(ExtensiveState(volume, volume[..., None] * concentration), velocity, velocity)
    source = jnp.zeros_like(volume).at[..., 0].set(source_sign * geometry.area * 1e-6 * (geometry.thickness[..., 0] > 0.))
    result = paired_momentum_surface_step(geometry, state, jnp.zeros_like(volume), 60., 4, volume_source=source, content_source=source[..., None] * concentration)
    assert bool(result.valid)
    assert result.state.east_velocity.dtype == dtype
    assert result.fluxes.east.dtype == jnp.float64
    final_eta = result.state.inventory.volume[..., 0] / geometry.area - geometry.thickness[..., 0]
    np.testing.assert_allclose(final_eta, result.surface.eta, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(result.state.inventory.content, result.state.inventory.volume[..., None] * concentration, rtol=1e-12, atol=.5)
    expected_volume_change = 60. * np.sum(np.asarray(source))
    assert abs(float(jnp.sum(result.state.inventory.volume - volume)) - expected_volume_change) <= .02
    np.testing.assert_allclose(result.fluxes.east, result.surface.mean_east, rtol=0., atol=0.)
    assert float(result.surface.energy_work_relative) <= 1e-11
    if dtype == jnp.float64:
        assert float(result.surface.cast_work) == 0.


def test_unconverged_true_residual_rejects_without_state_repair():
    geometry, volume, eta = fixture()
    operators = paired_operators(geometry, volume)
    velocity = VelocityPair(jnp.ones_like(volume) * .3, jnp.ones_like(volume) * -.2)
    velocity = VelocityPair(jnp.where(operators.east_area > 0., velocity.east, 0.),
                            jnp.where(operators.north_area > 0., velocity.north, 0.))
    result = advance_paired_surface(operators, jnp.asarray(eta), velocity, 1e5, 1, restart=1, maxiter=1)
    assert not bool(result.valid)
    assert float(result.solve_relative_residual) > 1e-12


def test_diagonal_only_mass_is_not_same_physical_field_norm():
    geometry, volume, unused_eta = fixture()
    operators = paired_operators(geometry, volume)
    velocity = VelocityPair(jnp.where(operators.east_area > 0., .3, 0.),
                            jnp.where(operators.north_area > 0., -.2, 0.))
    diagonal_only = operators._replace(mass=operators.mass * jnp.eye(4))
    relative = abs(float(kinetic_energy(diagonal_only, velocity) / kinetic_energy(operators, velocity)) - 1.)
    assert relative > .01


def test_midpoint_time_refinement_matches_independent_exact_linear_solution():
    from scipy.linalg import expm

    geometry, volume, eta = fixture()
    mass, spin, divergence, areas = dense_oracle(geometry, volume)
    active = np.diag(mass) > 0.
    mass, spin, divergence = mass[np.ix_(active, active)], spin[np.ix_(active, active)], divergence[:, active]
    surface_mass = np.diag(np.asarray(geometry.area).ravel())
    operator = np.block([[spin, 9.81 * divergence.T], [-divergence, np.zeros_like(surface_mass)]])
    metric = np.block([[mass, np.zeros((active.sum(), surface_mass.shape[0]))], [np.zeros((surface_mass.shape[0], active.sum())), surface_mass]])
    initial = np.concatenate((np.zeros(active.sum()), eta.ravel()))
    exact = expm(7200. * np.linalg.solve(metric, operator)) @ initial
    operators = paired_operators(geometry, volume)
    velocity = VelocityPair(jnp.zeros_like(volume), jnp.zeros_like(volume))
    errors = []
    for substeps in (2, 4, 8):
        result = advance_paired_surface(operators, jnp.asarray(eta), velocity, 7200., substeps)
        assert bool(result.valid)
        actual = np.concatenate((flatten(result.velocity)[active], np.asarray(result.eta).ravel()))
        errors.append(np.linalg.norm(actual - exact))
    assert errors[0] / errors[1] >= 3.5
    assert errors[1] / errors[2] >= 3.5


def test_actual_fixed_geometry_solve_tangent_matches_finite_difference():
    geometry, volume, eta = fixture()
    operators = paired_operators(geometry, volume)
    velocity = VelocityPair(jnp.zeros_like(volume), jnp.zeros_like(volume))

    def solution(amplitude):
        result = advance_paired_surface(operators, jnp.asarray(eta) * amplitude, velocity, 300., 2)
        return jnp.concatenate((result.velocity.east.ravel(), result.velocity.north.ravel(), result.eta.ravel()))

    unused_value, tangent = jax.jvp(solution, (jnp.array(1.),), (jnp.array(1.),))
    difference = (solution(1.0001) - solution(.9999)) / .0002
    np.testing.assert_allclose(tangent, difference, rtol=1e-5, atol=1e-9)


def test_rest_has_no_spurious_solver_motion_or_work():
    geometry, unused_volume, unused_eta = fixture(False)
    volume = geometry.area[..., None] * geometry.thickness
    operators = paired_operators(geometry, volume)
    velocity = VelocityPair(jnp.zeros_like(volume), jnp.zeros_like(volume))
    result = advance_paired_surface(operators, jnp.zeros(volume.shape[:2]), velocity, 60., 4)
    assert bool(result.valid)
    assert np.all(flatten(result.velocity) == 0.)
    assert float(result.energy_work_residual) == 0.


@pytest.mark.parametrize("defect", ["area", "latitude", "radius", "pole", "nan_frequency"])
def test_invalid_physical_metrics_and_rotation_reject(defect):
    geometry, volume, unused_eta = fixture()
    coriolis = None
    if defect == "area":
        geometry = geometry._replace(area=geometry.area * 1.01)
    elif defect == "latitude":
        geometry = geometry._replace(latitude_edges=jnp.asarray(geometry.latitude_edges).at[1].set(geometry.latitude_edges[0]))
    elif defect == "radius":
        geometry = geometry._replace(north_width=jnp.asarray(geometry.north_width).at[1, 1].multiply(1.01))
    elif defect == "pole":
        geometry = geometry._replace(latitude_edges=jnp.asarray(geometry.latitude_edges).at[0].set(-.5 * jnp.pi))
    else:
        coriolis = jnp.nan
    assert not bool(paired_operators(geometry, volume, coriolis).valid)


@pytest.mark.parametrize("defect", ["negative_dt", "nan_dt", "negative_gravity", "dry_source", "closed_velocity", "nan_force"])
def test_invalid_surface_inputs_reject(defect):
    geometry, volume, eta = fixture()
    operators = paired_operators(geometry, volume)
    velocity = VelocityPair(jnp.zeros_like(volume), jnp.zeros_like(volume))
    force, source = velocity, jnp.zeros(volume.shape[:2])
    dt, gravity = 60., 9.81
    if defect == "negative_dt":
        dt = -60.
    elif defect == "nan_dt":
        dt = jnp.nan
    elif defect == "negative_gravity":
        gravity = -9.81
    elif defect == "dry_source":
        source = source.at[3, 1].set(1.)
    elif defect == "closed_velocity":
        velocity = velocity._replace(north=velocity.north.at[0, -1, 0].set(1.))
    else:
        force = force._replace(east=force.east.at[0, 0, 0].set(jnp.nan))
    result = advance_paired_surface(operators, jnp.asarray(eta), velocity, dt, 1,
                                    gravity=gravity, force=force, volume_source=source, restart=4, maxiter=1)
    assert not bool(result.valid)


@pytest.mark.parametrize("defect", ["content32", "source32", "content_source32", "lower_source"])
def test_actual_step_rejects_wrong_inventory_precision_and_lower_sources(defect):
    geometry, volume, unused_eta = fixture()
    velocity = jnp.zeros_like(volume)
    content = volume[..., None] * jnp.array([15., 35.])
    source, content_source = jnp.zeros_like(volume), jnp.zeros_like(content)
    if defect == "content32":
        content = content.astype(jnp.float32)
    elif defect == "source32":
        source = source.astype(jnp.float32)
    elif defect == "content_source32":
        content_source = content_source.astype(jnp.float32)
    else:
        source = source.at[0, 0, 1].set(1.)
    state = LayerState(ExtensiveState(volume, content), velocity, velocity)
    if defect == "lower_source":
        result = paired_momentum_surface_step(geometry, state, jnp.zeros_like(volume), 60., 4, volume_source=source)
        assert not bool(result.valid)
    else:
        with pytest.raises(ValueError, match="64"):
            paired_momentum_surface_step(geometry, state, jnp.zeros_like(volume), 60., 4,
                                         volume_source=source, content_source=content_source)


def test_actual_density_force_uses_new_coupled_mass_not_old_acceleration():
    geometry, volume, unused_eta = fixture()
    operators = paired_operators(geometry, volume, coriolis=0.)
    mass, unused_spin, unused_divergence, unused_areas = dense_oracle(geometry, volume, coriolis=0.)
    active = np.diag(mass) > 0.
    density = jnp.broadcast_to(.1 * jnp.sin(jnp.arange(5)[:, None, None]), volume.shape)
    content = volume[..., None] * jnp.array([15., 35.])
    velocity = jnp.zeros_like(volume)
    state = LayerState(ExtensiveState(volume, content), velocity, velocity)
    result = paired_momentum_surface_step(geometry, state, density, 60., 4,
                                          gravity=0., pressure_gravity=9.81, coriolis=0.)
    assert bool(result.valid)
    expected = 60. * np.linalg.solve(mass[np.ix_(active, active)], flatten(result.held_force)[active])
    np.testing.assert_allclose(flatten(VelocityPair(result.state.east_velocity, result.state.north_velocity))[active], expected,
                               rtol=1e-11, atol=1e-14)
    old_acceleration = 60. * flatten(VelocityPair(result.pressure.east, result.pressure.north))[active]
    assert np.linalg.norm(expected - old_acceleration) / np.linalg.norm(expected) > .01
    endpoint = result.surface.uncast_velocity
    mean = VelocityPair(result.surface.mean_east / jnp.where(operators.east_area > 0., operators.east_area, 1.),
                        result.surface.mean_north / jnp.where(operators.north_area > 0., operators.north_area, 1.))
    left = flatten(mass_action(operators, endpoint))
    right = 60. * flatten(result.held_force)
    np.testing.assert_allclose(left, right, rtol=1e-11, atol=.0001)
    work = 60. * (jnp.sum(mean.east * result.held_force.east) + jnp.sum(mean.north * result.held_force.north))
    np.testing.assert_allclose(work, result.surface.external_work, rtol=1e-12)
