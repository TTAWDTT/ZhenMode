"""Absolute fluid velocity versus material-coordinate shared transport."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from cgrid_momentum import LayerState, linear_momentum_surface_step, momentum_geometry
from finite_volume import ExtensiveState, build_geometry, closed_surface_fluxes, surface_volume
from physical_velocity import evaluate_physical_velocity, physical_velocity_from_fluxes
from wet_fluxes import evaluate_wet_flux, reconstruct_wet_fluxes

jax.config.update("jax_enable_x64", True)


def _fixture(partial=True):
    depth = np.full((5, 4), 50.)
    if partial:
        depth[1, 1], depth[2, 2], depth[3, 1] = 22., 3., 0.
    geometry = build_geometry([0., 37., 131., 206., 298., 360.], [-60., -27., -3., 15., 56.],
                              [0., 7., 21., 50.], depth, radius=2.1e6)
    eta = np.broadcast_to(.2 * np.cos(np.arange(5)[:, None]), depth.shape) * (depth > 0.)
    volume = surface_volume(geometry, jnp.asarray(eta))
    faces = momentum_geometry(geometry, volume)
    east = faces.east_area * jnp.sin(jnp.arange(5)[:, None, None] + .4) * .02
    north = faces.north_area * jnp.cos(jnp.arange(4)[None, :, None] + .3) * .01
    fluxes = closed_surface_fluxes(east, north)
    return geometry, volume, faces, fluxes, reconstruct_wet_fluxes(geometry, faces, fluxes)


@pytest.mark.parametrize("partial", [False, True])
def test_absolute_surface_w_must_not_be_material_coordinate_qz_over_area(partial):
    geometry, unused_volume, faces, unused_fluxes, transport = _fixture(partial)
    reconstruction = physical_velocity_from_fluxes(geometry, transport)
    point = evaluate_physical_velocity(reconstruction, .37, .61, faces.top)
    expected = transport.net_flux[..., 0] / geometry.area
    assert np.max(np.abs(expected)) > 1e-9
    np.testing.assert_allclose(point.downward[..., 0], expected, rtol=1e-12, atol=1e-16)
    naive = evaluate_wet_flux(transport, .37, .61, faces.top).vertical[..., 0] / geometry.area
    assert np.max(np.abs(naive - expected)) > 1e-9


@pytest.mark.parametrize("side", range(4))
def test_all_actual_contact_normals_shared_speeds_and_blocked_wedges(side):
    geometry, unused_volume, faces, fluxes, transport = _fixture()
    physical = physical_velocity_from_fluxes(geometry, transport)
    assert bool(physical.valid)
    depth = .5 * (transport.side_top[..., side] + transport.side_bottom[..., side])
    longitude = 0. if side == 0 else 1. if side == 1 else .37
    latitude = 0. if side == 2 else 1. if side == 3 else .61
    point = evaluate_physical_velocity(physical, longitude, latitude, depth)
    opened = transport.side_bottom[..., side] > transport.side_top[..., side]
    if side < 2:
        expected_flux = np.roll(fluxes.east, 1, axis=0) if side == 0 else fluxes.east
        expected_area = np.roll(faces.east_area, 1, axis=0) if side == 0 else faces.east_area
        actual = point.east
    else:
        expected_flux = np.concatenate((np.zeros_like(fluxes.north[:, :1]), fluxes.north[:, :-1]), axis=1) if side == 2 else fluxes.north
        expected_area = np.concatenate((np.zeros_like(faces.north_area[:, :1]), faces.north_area[:, :-1]), axis=1) if side == 2 else faces.north_area
        actual = point.north
    expected = np.asarray(expected_flux) / np.where(expected_area > 0., expected_area, 1.)
    np.testing.assert_allclose(np.asarray(actual)[opened], expected[opened], rtol=1e-12, atol=1e-16)
    assert np.all(np.asarray(actual)[~opened] == 0.)
    samples = faces.top[..., None] + faces.height[..., None] * jnp.array([.113, .537, .913])
    sampled = evaluate_physical_velocity(physical, longitude, latitude, samples)
    sampled_open = ((samples >= transport.side_top[..., side, None]) & (samples <= transport.side_bottom[..., side, None])
                    & opened[..., None])
    sampled_normal = sampled.east if side < 2 else sampled.north
    assert np.all(np.asarray(sampled_normal)[~np.asarray(sampled_open)] == 0.)


@pytest.mark.parametrize("source_amount", [0., 1e5, -1e5])
def test_signed_source_surface_frame_and_fixed_vertical_interfaces(source_amount):
    geometry, unused_volume, faces, fluxes, transport = _fixture()
    source = jnp.zeros(faces.height.shape, jnp.float64).at[1, 2, 0].set(source_amount)
    physical = physical_velocity_from_fluxes(geometry, transport, source)
    assert bool(physical.valid)
    top = evaluate_physical_velocity(physical, .37, .61, faces.top)
    bottom = evaluate_physical_velocity(physical, .37, .61, faces.top + faces.height)
    wet = np.asarray(faces.height) > 0.
    np.testing.assert_allclose((top.downward - top.grid_downward)[..., 0], source[..., 0] / geometry.area, rtol=1e-12, atol=1e-16)
    np.testing.assert_allclose(np.asarray(top.downward)[..., 1:][wet[..., 1:]],
                               np.asarray(fluxes.vertical)[..., 1:-1][wet[..., 1:]] / np.broadcast_to(geometry.area[..., None], faces.height[..., 1:].shape)[wet[..., 1:]],
                               rtol=1e-12, atol=1e-16)
    expected_bottom = fluxes.vertical[..., 1:] / geometry.area[..., None]
    np.testing.assert_allclose(np.asarray(bottom.downward)[wet], np.asarray(expected_bottom)[wet], rtol=1e-12, atol=1e-16)
    assert np.max(np.abs(bottom.grid_downward)) <= 1e-16
    seabed = wet & (np.concatenate((wet[..., 1:], np.zeros_like(wet[..., :1])), axis=-1) == 0.)
    assert np.max(np.abs(np.asarray(bottom.downward)[seabed])) <= 1e-16
    baseline = evaluate_physical_velocity(physical_velocity_from_fluxes(geometry, transport), .37, .61, faces.top + .413 * faces.height)
    sourced = evaluate_physical_velocity(physical, .37, .61, faces.top + .413 * faces.height)
    np.testing.assert_array_equal(sourced.downward, baseline.downward)
    np.testing.assert_allclose(sourced.relative_downward, sourced.downward - sourced.grid_downward, rtol=1e-12, atol=1e-16)


@pytest.mark.parametrize("source_amount", [1e5, -1e5])
def test_source_only_rest_is_boundary_inflow_not_bulk_compressibility(source_amount):
    geometry, unused_volume, faces, unused_fluxes, unused_transport = _fixture()
    fluxes = closed_surface_fluxes(jnp.zeros(faces.height.shape), jnp.zeros(faces.height.shape))
    transport = reconstruct_wet_fluxes(geometry, faces, fluxes)
    source = jnp.zeros(faces.height.shape).at[1, 2, 0].set(source_amount)
    physical = physical_velocity_from_fluxes(geometry, transport, source)
    point = evaluate_physical_velocity(physical, .37, .61, faces.top)
    assert bool(physical.valid)
    for field in (point.east, point.north, point.downward, point.divergence):
        assert np.all(np.asarray(field) == 0.)
    np.testing.assert_array_equal(point.grid_downward, -source / geometry.area[..., None])
    np.testing.assert_array_equal(point.relative_downward, source / geometry.area[..., None])


@pytest.mark.parametrize("partial", [False, True])
def test_independent_spherical_jvp_and_fd_divergence_reject_unlifted_frame(partial):
    geometry, unused_volume, faces, unused_fluxes, transport = _fixture(partial)
    physical = physical_velocity_from_fluxes(geometry, transport)
    depth = faces.top + .41327 * faces.height
    longitude, fraction = .3713, .6127

    def east(coordinate):
        return evaluate_physical_velocity(physical, coordinate, fraction, depth).east

    def meridional(coordinate):
        point = evaluate_physical_velocity(physical, longitude, coordinate, depth)
        latitude = jnp.arcsin(jnp.sin(transport.south_latitude) + coordinate * (jnp.sin(transport.north_latitude) - jnp.sin(transport.south_latitude)))
        return point.north * jnp.cos(latitude)

    def downward(coordinate):
        return evaluate_physical_velocity(physical, longitude, fraction, coordinate).downward

    latitude = jnp.arcsin(jnp.sin(transport.south_latitude) + fraction * (jnp.sin(transport.north_latitude) - jnp.sin(transport.south_latitude)))
    east_term = jax.jvp(east, (longitude,), (1.,))[1] / (physical.zonal_arc * jnp.cos(latitude))
    north_term = jax.jvp(meridional, (fraction,), (1.,))[1] / physical.meridional_area_width
    vertical_term = jax.jvp(downward, (depth,), (jnp.ones_like(depth),))[1]
    scale = np.abs(east_term) + np.abs(north_term) + np.abs(vertical_term)
    tolerance = (1e-12 + 64. * np.finfo(np.float64).eps) * np.asarray(scale)
    np.testing.assert_array_less(np.abs(east_term + north_term + vertical_term), tolerance + 1e-30)
    point = evaluate_physical_velocity(physical, longitude, fraction, depth)
    np.testing.assert_allclose(point.divergence, 0., rtol=0., atol=1e-18)
    delta = 1e-5
    east_fd = (east(longitude + delta) - east(longitude - delta)) / (2. * delta * physical.zonal_arc * jnp.cos(latitude))
    north_fd = (meridional(fraction + delta) - meridional(fraction - delta)) / (2. * delta * physical.meridional_area_width)
    depth_delta = jnp.maximum(faces.height, 1.) * delta
    down_fd = (downward(depth + depth_delta) - downward(depth - depth_delta)) / (2. * depth_delta)
    assert np.all(np.abs(east_fd + north_fd + down_fd) <= 1e-6 * scale + 1e-25)
    naive_divergence = evaluate_wet_flux(transport, longitude, fraction, depth).divergence_per_depth / geometry.area[..., None]
    assert np.max(np.abs(naive_divergence[..., 0])) > 1e-10


@pytest.mark.parametrize("dtype", [jnp.float64, jnp.float32])
@pytest.mark.parametrize("source_amount", [0., 1e5, -1e5])
def test_actual_step_grid_motion_matches_local_volume_increment(dtype, source_amount):
    geometry, volume, faces, unused_fluxes, unused_transport = _fixture()
    concentration = jnp.broadcast_to(jnp.array([12., 35.]), volume.shape + (2,))
    inventory = ExtensiveState(volume, volume[..., None] * concentration)
    state = LayerState(inventory, jnp.where(faces.east_area > 0., .02, 0.).astype(dtype),
                       jnp.where(faces.north_area > 0., -.01, 0.).astype(dtype))
    source = jnp.zeros_like(volume).at[1, 2, 0].set(source_amount)
    result = jax.jit(linear_momentum_surface_step, static_argnums=4)(geometry, state, jnp.zeros_like(volume), 60., 4,
                                                                   volume_source=source, content_source=source[..., None] * concentration)
    assert bool(result.valid) and bool(result.physical_velocity.valid)
    point = evaluate_physical_velocity(result.physical_velocity, .37, .61, result.physical_velocity.transport.top)
    expected = -60. * point.grid_downward * geometry.area[..., None]
    actual = result.state.inventory.volume - volume
    participating = 60. * (np.abs(result.flux_reconstruction.net_flux) + np.abs(source))
    stored = np.abs(volume) + np.abs(result.state.inventory.volume)
    assert np.all(np.abs(actual - expected) <= 1e-12 * participating + 64. * np.finfo(np.float64).eps * stored)
    assert point.east.dtype == point.downward.dtype == jnp.float64
    assert result.state.east_velocity.dtype == dtype


@pytest.mark.parametrize("failure", ["negative_area", "nan_metric", "metric_product", "radius", "lower_continuity", "lower_source", "land_source", "nan_source", "nan_transport", "negative_height", "pole"])
def test_invalid_physical_geometry_continuity_and_source_fail_closed(failure):
    geometry, unused_volume, faces, unused_fluxes, transport = _fixture()
    source = jnp.zeros_like(faces.height)
    if failure == "negative_area":
        geometry = geometry._replace(area=-geometry.area)
    elif failure == "nan_metric":
        geometry = geometry._replace(east_width=geometry.east_width * np.nan)
    elif failure == "metric_product":
        geometry = geometry._replace(east_width=geometry.east_width * 1.1)
    elif failure == "radius":
        geometry = geometry._replace(north_width=geometry.north_width * 2., east_width=geometry.east_width / 2.)
    elif failure == "lower_continuity":
        transport = transport._replace(net_flux=transport.net_flux.at[0, 0, 1].add(1.))
    elif failure == "lower_source":
        source = source.at[0, 0, 1].set(1.)
    elif failure == "land_source":
        source = source.at[3, 1, 0].set(1.)
    elif failure == "nan_source":
        source = source.at[0, 0, 0].set(jnp.nan)
    elif failure == "nan_transport":
        transport = transport._replace(top=transport.top.at[0, 0, 0].set(jnp.nan))
    elif failure == "negative_height":
        transport = transport._replace(height=transport.height.at[0, 0, 0].set(-1.))
    else:
        geometry = geometry._replace(latitude_edges=np.r_[-np.pi / 2., geometry.latitude_edges[1:]])
    physical = physical_velocity_from_fluxes(geometry, transport, source)
    assert not bool(physical.valid)
    assert not np.any(evaluate_physical_velocity(physical, .37, .61, faces.top).valid)


@pytest.mark.parametrize("failure", ["source_shape", "source_dtype", "geometry_shape", "geometry_dtype", "transport_shape", "transport_dtype", "x64"])
def test_physical_api_shape_precision_and_x64_errors(failure):
    geometry, unused_volume, faces, unused_fluxes, transport = _fixture()
    source = jnp.zeros_like(faces.height)
    if failure == "source_shape":
        source = source[..., 0]
    elif failure == "source_dtype":
        source = source.astype(jnp.float32)
    elif failure == "geometry_shape":
        geometry = geometry._replace(area=geometry.area[0])
    elif failure == "geometry_dtype":
        geometry = geometry._replace(area=geometry.area.astype(np.float32))
    elif failure == "transport_shape":
        transport = transport._replace(net_flux=transport.net_flux[..., 0])
    elif failure == "transport_dtype":
        transport = transport._replace(top=transport.top.astype(jnp.float32))
    if failure == "x64":
        with jax.enable_x64(False), pytest.raises(ValueError, match="X64"):
            physical_velocity_from_fluxes(geometry, transport, source)
    else:
        with pytest.raises(ValueError):
            physical_velocity_from_fluxes(geometry, transport, source)


@pytest.mark.parametrize("longitude,fraction,offset", [(-.1, .5, 0.), (.5, 1.1, 0.), (np.nan, .5, 0.), (.5, .5, -1.), (.5, .5, np.inf)])
def test_invalid_queries_are_not_repaired(longitude, fraction, offset):
    geometry, unused_volume, faces, unused_fluxes, transport = _fixture()
    physical = physical_velocity_from_fluxes(geometry, transport)
    point = evaluate_physical_velocity(physical, longitude, fraction, faces.top + offset)
    assert not np.any(point.valid)
