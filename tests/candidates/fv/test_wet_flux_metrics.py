"""Metric normal traces and physically integrated paired half-prism Q."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from cgrid_momentum import LayerState, linear_momentum_surface_step, momentum_geometry
from finite_volume import (
    ExtensiveState,
    VolumeFluxes,
    build_geometry,
    closed_surface_fluxes,
    surface_volume,
)
from wet_fluxes import evaluate_wet_flux, reconstruct_half_prism_transport, reconstruct_wet_fluxes

jax.config.update("jax_enable_x64", True)


def _fixture(partial):
    depth = np.full((5, 4), 50.)
    if partial:
        depth[1, 1], depth[2, 2], depth[3, 1] = 22., 3., 0.
    radius = 2.1e6
    geometry = build_geometry([0., 37., 131., 206., 298., 360.], [-60., -27., -3., 15., 56.], [0., 7., 21., 50.], depth, radius)
    eta = np.broadcast_to(.2 * np.cos(np.arange(5)[:, None]), depth.shape) * (depth > 0.)
    volume = surface_volume(geometry, jnp.asarray(eta))
    faces = momentum_geometry(geometry, volume)
    fluxes = closed_surface_fluxes(.02 * faces.east_area, jnp.zeros_like(faces.north_area))
    return geometry, volume, faces, fluxes, radius


@pytest.mark.parametrize("partial", [False, True])
def test_constant_physical_normal_speed_is_not_constant_area_coordinate_flux(partial):
    geometry, unused_volume, faces, fluxes, radius = _fixture(partial)
    result = reconstruct_wet_fluxes(geometry, faces, fluxes)
    latitude = geometry.latitude_edges
    delta_sin = np.diff(np.sin(latitude))[None, :, None]
    for sample in (.11, .57, .89):
        phi = np.arcsin(np.sin(latitude[:-1])[None, :, None] + sample * delta_sin)
        point = evaluate_wet_flux(result, 1., sample, .5 * (faces.east_top + faces.east_bottom))
        speed = np.asarray(point.east_per_depth) * np.cos(phi) / (radius * delta_sin)
        opened = np.asarray(faces.east_area) > 0.
        np.testing.assert_allclose(speed[opened], .02, rtol=1e-12, atol=64. * np.finfo(float).eps * .02)


def _signed_fluxes(faces):
    east = faces.east_area * jnp.sin(jnp.arange(5)[:, None, None] + .4) * .02
    north = faces.north_area * jnp.cos(jnp.arange(4)[None, :, None] + .3) * .01
    closed = closed_surface_fluxes(east, north)
    vertical = closed.vertical.at[..., 1:-1].add(jnp.where(faces.height[..., 1:] > 0., 1000., 0.))
    return VolumeFluxes(east, north, vertical)


def _host_map(field, latitude_edges, component):
    if component == "east":
        return .5 * (field + np.roll(field, -1, axis=0))
    middle = .5 * (latitude_edges[:-1] + latitude_edges[1:])
    south = (np.sin(middle) - np.sin(latitude_edges[:-1])) / np.diff(np.sin(latitude_edges))
    output = np.zeros((field.shape[0], field.shape[1] + 1, field.shape[2]))
    for latitude in range(field.shape[1]):
        output[:, latitude] += south[latitude] * field[:, latitude]
        output[:, latitude + 1] += (1. - south[latitude]) * field[:, latitude]
    return output


def _net(fluxes):
    east, north, vertical = map(np.asarray, fluxes)
    return east - np.roll(east, 1, axis=0) + north[:, 1:] - north[:, :-1] + vertical[..., 1:] - vertical[..., :-1]


@pytest.mark.parametrize("partial", [False, True])
def test_metric_contact_quadrature_and_paired_divergence(partial):
    geometry, unused_volume, faces, unused_fluxes, unused_radius = _fixture(partial)
    fluxes = _signed_fluxes(faces)
    result = reconstruct_wet_fluxes(geometry, faces, fluxes)
    nodes, weights = np.polynomial.legendre.leggauss(40)
    samples = (.5 + .5 * nodes)[None, None, None, :]
    depth = .5 * (faces.east_top + faces.east_bottom)[..., None]
    point = jax.jit(evaluate_wet_flux)(result, 1., samples, depth)
    integral = np.sum(np.asarray(point.east_per_depth) * (.5 * weights), axis=-1) * np.maximum(np.asarray(faces.east_bottom - faces.east_top), 0.)
    np.testing.assert_allclose(integral, fluxes.east, rtol=1e-12, atol=1e-8)
    depth = faces.top + .38123 * faces.height
    longitude, latitude = jnp.full_like(depth, .37), jnp.full_like(depth, .61)
    direction = jnp.ones_like(depth)
    east_derivative = jax.jvp(lambda sample: evaluate_wet_flux(result, sample, latitude, depth).east_per_depth, (longitude,), (direction,))[1]
    north_derivative = jax.jvp(lambda sample: evaluate_wet_flux(result, longitude, sample, depth).north_per_depth, (latitude,), (direction,))[1]
    vertical_derivative = jax.jvp(lambda sample: evaluate_wet_flux(result, longitude, latitude, sample).vertical, (depth,), (direction,))[1]
    expected = result.net_flux / jnp.where(faces.height > 0., faces.height, 1.)
    scale = np.sum(np.abs(result.side_density), axis=-1) + np.abs(expected)
    residual = np.asarray(east_derivative + north_derivative + vertical_derivative - expected)
    assert np.all(np.abs(residual) <= (1e-12 + 64. * np.finfo(float).eps) * scale + 1e-8)
    step = 1e-6
    finite_difference = (evaluate_wet_flux(result, longitude, latitude + step, depth).north_per_depth
                         - evaluate_wet_flux(result, longitude, latitude - step, depth).north_per_depth) / (2. * step)
    assert np.all(np.abs(np.asarray(finite_difference - north_derivative)) <= 1e-6 * scale + 1e-8)
    traces = jnp.where((depth[..., None] >= result.side_top) & (depth[..., None] <= result.side_bottom)
                       & (result.side_bottom > result.side_top), result.side_density, 0.)
    old_north_derivative = traces[..., 3] - traces[..., 2]
    wrong = np.asarray(east_derivative + old_north_derivative + vertical_derivative - expected)
    assert np.max(np.abs(wrong) / np.maximum(scale, 1.)) > .01


@pytest.mark.parametrize("partial", [False, True])
def test_physical_half_prism_mass_and_surface_quadrature_commute(partial):
    geometry, volume, faces, unused_fluxes, radius = _fixture(partial)
    fluxes = _signed_fluxes(faces)
    trace = reconstruct_wet_fluxes(geometry, faces, fluxes)
    dual = jax.jit(reconstruct_half_prism_transport)(geometry, volume, fluxes)
    assert bool(dual.valid) and float(dual.commutation_relative) < 1e-12
    longitude_edges = np.radians([0., 37., 131., 206., 298., 360.])
    latitude_edges = geometry.latitude_edges
    east_mass = np.zeros(volume.shape)
    north_mass = np.zeros((volume.shape[0], volume.shape[1] + 1, volume.shape[2]))
    for longitude, latitude, layer in np.ndindex(volume.shape):
        west, east = longitude_edges[longitude:longitude + 2]
        south, north = latitude_edges[latitude:latitude + 2]
        middle = .5 * (south + north)
        height = float(faces.height[longitude, latitude, layer])
        amount = radius ** 2 * (east - west) * (np.sin(north) - np.sin(south)) * height
        east_mass[longitude, latitude, layer] += .5 * amount
        east_mass[(longitude - 1) % volume.shape[0], latitude, layer] += .5 * amount
        north_mass[longitude, latitude, layer] += radius ** 2 * (east - west) * (np.sin(middle) - np.sin(south)) * height
        north_mass[longitude, latitude + 1, layer] += radius ** 2 * (east - west) * (np.sin(north) - np.sin(middle)) * height
    np.testing.assert_allclose(dual.east_volume, east_mass, rtol=1e-12, atol=1e-6)
    np.testing.assert_allclose(dual.north_volume, north_mass, rtol=1e-12, atol=1e-6)
    for field in (dual.east_volume, dual.north_volume):
        np.testing.assert_allclose(np.sum(field), np.sum(volume), rtol=1e-12)
    midpoint = .5 * (latitude_edges[:-1] + latitude_edges[1:])
    south_fraction = ((np.sin(midpoint) - np.sin(latitude_edges[:-1])) / np.diff(np.sin(latitude_edges)))[None, :, None]
    breaks = np.sort(np.concatenate((np.asarray(faces.top)[..., None], np.asarray(faces.top + faces.height)[..., None],
                                    np.asarray(trace.side_top), np.asarray(trace.side_bottom)), axis=-1), axis=-1)
    depth = .5 * (breaks[..., 1:] + breaks[..., :-1])
    width = np.diff(breaks, axis=-1)
    point = evaluate_wet_flux(trace, .37, south_fraction, depth)
    center_north = np.sum(np.asarray(point.north_per_depth) * width, axis=-1)
    np.testing.assert_allclose(dual.north_fluxes.north[:, 1:-1], center_north, rtol=1e-12, atol=1e-8)
    nodes, weights = np.polynomial.legendre.leggauss(40)
    contacts = np.maximum(np.asarray(faces.east_bottom - faces.east_top), 0.)
    depth = .5 * (faces.east_top + faces.east_bottom)[..., None]
    lower_samples = south_fraction[..., None] * (.5 + .5 * nodes)
    lower = evaluate_wet_flux(trace, 1., lower_samples, depth).east_per_depth
    integrated_lower = np.sum(np.asarray(lower) * weights * (.5 * south_fraction[..., None]), axis=-1) * contacts
    np.testing.assert_allclose(integrated_lower, .5 * np.asarray(fluxes.east), rtol=1e-12, atol=1e-8)
    expected_east = np.pad(integrated_lower, ((0, 0), (0, 1), (0, 0))) + np.pad(np.asarray(fluxes.east) - integrated_lower, ((0, 0), (1, 0), (0, 0)))
    np.testing.assert_allclose(dual.north_fluxes.east, expected_east, rtol=1e-12, atol=1e-8)
    north_full = np.concatenate((np.zeros_like(fluxes.north[:, :1]), np.asarray(fluxes.north)), axis=1)
    primary = VolumeFluxes(np.asarray(fluxes.east), north_full, np.asarray(fluxes.vertical))
    for component, paired in (("east", dual.east_fluxes), ("north", dual.north_fluxes)):
        expected = _host_map(_net(primary), latitude_edges, component)
        np.testing.assert_allclose(_net(paired), expected, rtol=1e-12, atol=1e-8)
    wrong_east = _host_map(np.asarray(fluxes.east), latitude_edges, "north")
    wrong = dual.north_fluxes._replace(east=wrong_east)
    assert np.max(np.abs(_net(wrong) - _host_map(_net(primary), latitude_edges, "north"))) > 1e4


@pytest.mark.parametrize("dtype,forced", [(jnp.float64, False), (jnp.float64, True), (jnp.float32, False), (jnp.float32, True)])
def test_actual_step_dual_mass_increment_uses_actual_q_and_local_source(dtype, forced):
    geometry, volume, unused_faces, unused_fluxes, unused_radius = _fixture(True)
    concentration = jnp.asarray([12., 35.])
    state = LayerState(ExtensiveState(volume, volume[..., None] * concentration),
                       jnp.where(geometry.east_area > 0., .02, 0.).astype(dtype),
                       jnp.where(geometry.north_area > 0., -.01, 0.).astype(dtype))
    source = jnp.zeros_like(volume)
    if forced:
        source = source.at[1, 2, 0].set(1e5)
    result = jax.jit(lambda state: linear_momentum_surface_step(geometry, state, jnp.zeros_like(volume), 60., 4, coriolis=.001,
                                                              volume_source=source, content_source=source[..., None] * concentration))(state)
    assert bool(result.valid) and bool(result.dual_transport.valid)
    for component in ("east", "north"):
        old = np.asarray(getattr(result.dual_transport, component + "_volume"))
        new = _host_map(np.asarray(result.state.inventory.volume), geometry.latitude_edges, component)
        expected = 60. * (_host_map(np.asarray(source), geometry.latitude_edges, component) - _net(getattr(result.dual_transport, component + "_fluxes")))
        tolerance = 1e-12 * np.abs(expected) + 64. * np.finfo(float).eps * (np.abs(old) + np.abs(new))
        assert np.all(np.abs(new - old - expected) <= tolerance)
        np.testing.assert_allclose(old, _host_map(np.asarray(volume), geometry.latitude_edges, component), rtol=1e-12)


@pytest.mark.parametrize("fault", ["pole", "nonfinite", "closed", "volume"])
def test_metric_or_dual_invalid_input_is_not_repaired(fault):
    geometry, volume, faces, fluxes, unused_radius = _fixture(True)
    if fault == "pole":
        latitude = np.asarray(geometry.latitude_edges).copy()
        latitude[0] = -.5 * np.pi
        geometry = geometry._replace(latitude_edges=latitude)
    elif fault == "nonfinite":
        latitude = np.asarray(geometry.latitude_edges).copy()
        latitude[1] = np.nan
        geometry = geometry._replace(latitude_edges=latitude)
    elif fault == "closed":
        fluxes = fluxes._replace(north=fluxes.north.at[0, -1, 0].set(1.))
    else:
        volume = volume.at[0, 0, 0].set(-1.)
    dual = jax.jit(reconstruct_half_prism_transport)(geometry, volume, fluxes)
    assert not bool(dual.valid)
    if fault != "volume":
        trace = jax.jit(reconstruct_wet_fluxes)(geometry, faces, fluxes)
        assert not bool(trace.valid)


def test_dual_shape_and_precision_are_explicit():
    geometry, volume, faces, fluxes, unused_radius = _fixture(True)
    with pytest.raises(ValueError, match="64"):
        reconstruct_half_prism_transport(geometry, volume.astype(jnp.float32), fluxes)
    with pytest.raises(ValueError, match="shape"):
        reconstruct_half_prism_transport(geometry, volume, fluxes._replace(vertical=fluxes.vertical[..., :-1]))
    with pytest.raises(ValueError, match="latitude"):
        reconstruct_wet_fluxes(geometry._replace(latitude_edges=np.asarray(geometry.latitude_edges, np.float32)), faces, fluxes)
    with pytest.raises(ValueError, match="64"):
        reconstruct_half_prism_transport(geometry._replace(east_area=np.asarray(geometry.east_area, np.float32)), volume, fluxes)
