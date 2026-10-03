"""Wet-wall flux traces and divergence on the actual shared-Q path."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from zhenmode_research.candidates.fv.fluxes import evaluate_wet_flux, reconstruct_wet_fluxes
from zhenmode_research.candidates.fv.geometry import (
    ExtensiveState,
    build_geometry,
    closed_surface_fluxes,
    surface_volume,
)
from zhenmode_research.candidates.fv.momentum import (
    LayerState,
    linear_momentum_surface_step,
    momentum_geometry,
)

jax.config.update("jax_enable_x64", True)


def _fixture(partial=True, moving=True):
    depth = np.full((5, 4), 50.)
    if partial:
        depth[1, 1], depth[2, 2], depth[3, 1] = 22., 3., 0.
    geometry = build_geometry([0., 37., 131., 206., 298., 360.], [-60., -27., -3., 15., 56.], [0., 7., 21., 50.], depth)
    eta = np.broadcast_to(.2 * np.cos(np.arange(5)[:, None]), depth.shape) * (depth > 0.) if moving else np.zeros_like(depth)
    volume = surface_volume(geometry, jnp.asarray(eta))
    faces = momentum_geometry(geometry, volume)
    east = faces.east_area * jnp.sin(jnp.arange(5)[:, None, None] + .4) * .02
    north = faces.north_area * jnp.cos(jnp.arange(4)[None, :, None] + .3) * .01
    fluxes = closed_surface_fluxes(east, north)
    return geometry, volume, faces, fluxes


@pytest.mark.parametrize("partial,moving", [(True, True), (True, False), (False, True), (False, False)])
def test_actual_wet_side_intervals_integrate_flux_and_forbid_wall_spreading(partial, moving):
    geometry, unused_volume, faces, fluxes = _fixture(partial, moving)
    reconstruction = jax.jit(reconstruct_wet_fluxes)(geometry, faces, fluxes)
    assert bool(reconstruction.valid)
    density, side_top, side_bottom = map(np.asarray, (reconstruction.side_density, reconstruction.side_top, reconstruction.side_bottom))
    east, north = np.asarray(fluxes.east), np.asarray(fluxes.north)
    south = np.concatenate((np.zeros_like(north[:, :1]), north[:, :-1]), axis=1)
    expected = np.stack((np.roll(east, 1, axis=0), east, south, north), axis=-1)
    np.testing.assert_allclose(density * (side_bottom - side_top), expected, rtol=1e-12, atol=1e-8)
    cell_top, cell_bottom = np.asarray(faces.top), np.asarray(faces.top + faces.height)
    wrong_wall_control = False
    for position in np.ndindex(cell_top.shape):
        if faces.height[position] <= 0.:
            continue
        for side in range(4):
            start, end = side_top[position + (side,)], side_bottom[position + (side,)]
            actual_flux = expected[position + (side,)]
            if end <= start:
                assert actual_flux == 0.
                continue
            interval_width = end - start
            assert start >= cell_top[position] - 1e-12 and end <= cell_bottom[position] + 1e-12
            for sample in (.211, .677):
                depth = start + sample * interval_width
                query = jnp.full(cell_top.shape, depth)
                longitude = 0. if side == 0 else 1. if side == 1 else .43
                latitude = 0. if side == 2 else 1. if side == 3 else .57
                point = evaluate_wet_flux(reconstruction, longitude, latitude, query)
                observed = (point.east_per_depth if side < 2 else point.north_per_depth)[position]
                if side < 2:
                    south, north = geometry.latitude_edges[position[1]:position[1] + 2]
                    phi = np.arcsin(np.sin(south) + latitude * (np.sin(north) - np.sin(south)))
                    observed = observed * (north - south) * np.cos(phi) / (np.sin(north) - np.sin(south))
                np.testing.assert_allclose(observed * interval_width, actual_flux, rtol=1e-12, atol=1e-8)
            wall_sample = None
            if start > cell_top[position] + 1e-10:
                wall_sample = .5 * (start + cell_top[position])
            elif end < cell_bottom[position] - 1e-10:
                wall_sample = .5 * (end + cell_bottom[position])
            if wall_sample is not None:
                point = evaluate_wet_flux(reconstruction, longitude, latitude, jnp.full(cell_top.shape, wall_sample))
                observed = (point.east_per_depth if side < 2 else point.north_per_depth)[position]
                assert observed == 0.
                if actual_flux != 0.:
                    assert actual_flux / faces.height[position] != 0.
                    wrong_wall_control = True
    if partial or moving:
        assert wrong_wall_control


def test_vertical_primitive_endpoints_and_independent_derivative_divergence():
    geometry, unused_volume, faces, fluxes = _fixture()
    reconstruction = reconstruct_wet_fluxes(geometry, faces, fluxes)
    top = evaluate_wet_flux(reconstruction, .37, .61, faces.top)
    bottom = evaluate_wet_flux(reconstruction, .37, .61, faces.top + faces.height)
    np.testing.assert_allclose(top.vertical, fluxes.vertical[..., :-1], rtol=1e-12, atol=1e-8)
    np.testing.assert_allclose(bottom.vertical, fluxes.vertical[..., 1:], rtol=1e-12, atol=1e-8)
    depth = faces.top + .38123 * faces.height
    longitude, latitude = jnp.full_like(depth, .37), jnp.full_like(depth, .61)
    direction = jnp.ones_like(depth)
    east_derivative = jax.jvp(lambda fraction: evaluate_wet_flux(reconstruction, fraction, latitude, depth).east_per_depth, (longitude,), (direction,))[1]
    north_derivative = jax.jvp(lambda fraction: evaluate_wet_flux(reconstruction, longitude, fraction, depth).north_per_depth, (latitude,), (direction,))[1]
    vertical_derivative = jax.jvp(lambda sample: evaluate_wet_flux(reconstruction, longitude, latitude, sample).vertical, (depth,), (direction,))[1]
    expected = reconstruction.net_flux / jnp.where(faces.height > 0., faces.height, 1.)
    scale = jnp.sum(jnp.abs(reconstruction.side_density), axis=-1) + jnp.abs(expected)
    residual = east_derivative + north_derivative + vertical_derivative - expected
    assert np.all(np.abs(residual) <= 1e-12 * np.asarray(scale) + 1e-8)
    finite_difference = (evaluate_wet_flux(reconstruction, longitude, latitude, depth + 1e-6).vertical
                         - evaluate_wet_flux(reconstruction, longitude, latitude, depth - 1e-6).vertical) / 2e-6
    wet = np.asarray(faces.height) > 0.
    assert np.all(np.abs(np.asarray(finite_difference - vertical_derivative)[wet]) <= 1e-6 * np.asarray(scale)[wet] + 1e-8)


def test_full_wet_uniform_height_retains_vertical_rt0_with_paired_latitude_metric():
    geometry, unused_volume, faces, fluxes = _fixture(False, False)
    reconstruction = reconstruct_wet_fluxes(geometry, faces, fluxes)
    depth = faces.top + .41 * faces.height
    point = evaluate_wet_flux(reconstruction, .37, .61, depth)
    side = reconstruction.side_density
    latitude = geometry.latitude_edges
    sine_width = np.diff(np.sin(latitude))[None, :, None]
    width = np.diff(latitude)[None, :, None]
    phi = np.arcsin(np.sin(latitude[:-1])[None, :, None] + .61 * sine_width)
    weight = sine_width / (width * np.cos(phi))
    primitive = (phi - latitude[:-1][None, :, None]) / width
    np.testing.assert_allclose(point.east_per_depth, weight * (.63 * side[..., 0] + .37 * side[..., 1]), rtol=1e-12, atol=1e-8)
    np.testing.assert_allclose(point.north_per_depth, .39 * side[..., 2] + .61 * side[..., 3] + (side[..., 1] - side[..., 0]) * (.61 - primitive), rtol=1e-12, atol=1e-8)
    np.testing.assert_allclose(point.vertical, .59 * fluxes.vertical[..., :-1] + .41 * fluxes.vertical[..., 1:], rtol=1e-12, atol=1e-8)


@pytest.mark.parametrize("fault", ["closed", "nonfinite", "material", "interval"])
def test_invalid_flux_or_geometry_is_not_repaired_into_acceptance(fault):
    geometry, unused_volume, faces, fluxes = _fixture()
    if fault == "closed":
        fluxes = fluxes._replace(north=fluxes.north.at[0, -1, 0].set(1.))
    elif fault == "nonfinite":
        fluxes = fluxes._replace(east=fluxes.east.at[0, 0, 0].set(jnp.nan))
    elif fault == "material":
        fluxes = fluxes._replace(vertical=fluxes.vertical.at[0, 0, 0].set(1.))
    else:
        faces = faces._replace(east_top=faces.east_top.at[0, 0, 0].set(-100.))
    result = jax.jit(reconstruct_wet_fluxes)(geometry, faces, fluxes)
    assert not bool(result.valid)


def test_shape_precision_and_query_validity_are_explicit():
    geometry, unused_volume, faces, fluxes = _fixture()
    with pytest.raises(ValueError, match="64"):
        reconstruct_wet_fluxes(geometry, faces, fluxes._replace(east=fluxes.east.astype(jnp.float32)))
    with pytest.raises(ValueError, match="shape"):
        reconstruct_wet_fluxes(geometry, faces, fluxes._replace(north=fluxes.north[:, :-1]))
    result = reconstruct_wet_fluxes(geometry, faces, fluxes)
    for longitude, latitude, depth in ((-.1, .5, faces.top), (.5, 1.1, faces.top), (.5, .5, faces.top - 1.), (.5, .5, faces.top + jnp.nan)):
        assert not np.any(evaluate_wet_flux(result, longitude, latitude, depth).valid)
    samples = jnp.stack((faces.top + .21 * faces.height, faces.top + .73 * faces.height), axis=-1)
    point = jax.jit(evaluate_wet_flux)(result, .37, .61, samples)
    assert point.vertical.shape == samples.shape
    np.testing.assert_array_equal(point.valid, jnp.broadcast_to((faces.height > 0.)[..., None], samples.shape))


@pytest.mark.parametrize("dtype", [jnp.float64, jnp.float32])
def test_actual_coupled_step_returns_same_shared_flux_and_accepted_reconstruction(dtype):
    geometry, volume, unused_faces, unused_fluxes = _fixture()
    content = volume[..., None] * jnp.asarray([12., 35.])
    state = LayerState(ExtensiveState(volume, content), jnp.where(geometry.east_area > 0., .02, 0.).astype(dtype),
                       jnp.where(geometry.north_area > 0., -.01, 0.).astype(dtype))
    result = jax.jit(lambda state: linear_momentum_surface_step(geometry, state, jnp.zeros_like(volume), 60., 4, coriolis=.001))(state)
    assert bool(result.valid) and bool(result.flux_reconstruction.valid)
    reconstruction = result.flux_reconstruction
    east, north = result.fluxes.east, result.fluxes.north
    np.testing.assert_allclose(reconstruction.side_density[..., 1] * (reconstruction.side_bottom[..., 1] - reconstruction.side_top[..., 1]), east, rtol=1e-12, atol=1e-8)
    np.testing.assert_allclose(reconstruction.side_density[..., 3] * (reconstruction.side_bottom[..., 3] - reconstruction.side_top[..., 3]), north, rtol=1e-12, atol=1e-8)
