"""Actual layer Q used by scalar inventory, not endpoint-velocity reconstruction."""
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
from zhenmode_research.candidates.fv.momentum import LayerState, linear_momentum_surface_step

jax.config.update("jax_enable_x64", True)


@pytest.mark.parametrize("dtype", [jnp.float64, jnp.float32])
@pytest.mark.parametrize("forced", [False, True])
def test_returned_actual_flux_closes_local_inventory_and_fast_mean_not_endpoint(dtype, forced):
    depth = np.full((5, 4), 50.)
    depth[1, 1], depth[2, 2], depth[3, 1] = 22., 3., 0.
    geometry = build_geometry([0., 37., 131., 206., 298., 360.], [-60., -27., -3., 15., 56.], [0., 7., 21., 50.], depth)
    phase = 2. * np.pi * np.arange(5)[:, None] / 5.
    eta = jnp.asarray(np.broadcast_to(.2 * np.cos(phase), depth.shape) * (depth > 0.))
    volume = surface_volume(geometry, eta)
    concentration = jnp.asarray([12., 35.])
    inventory = ExtensiveState(volume, volume[..., None] * concentration)
    state = LayerState(inventory, jnp.where(geometry.east_area > 0., .02, 0.).astype(dtype),
                       jnp.where(geometry.north_area > 0., -.01, 0.).astype(dtype))
    source = jnp.zeros_like(volume)
    if forced:
        source = source.at[1, 2, 0].set(1e5)
    density = jnp.broadcast_to(jnp.asarray(.1 * np.cos(phase))[..., None], volume.shape)
    result = jax.jit(lambda state: linear_momentum_surface_step(
        geometry, state, density, 60., 4, coriolis=.001, volume_source=source,
        content_source=source[..., None] * concentration))(state)
    assert bool(result.valid)
    fluxes = result.fluxes
    assert isinstance(fluxes, VolumeFluxes)
    assert all(field.dtype == jnp.float64 for field in fluxes)
    assert fluxes.east.shape == volume.shape and fluxes.north.shape == volume.shape
    assert fluxes.vertical.shape == volume.shape[:-1] + (volume.shape[-1] + 1,)
    east, north, vertical = map(np.asarray, fluxes)
    np.testing.assert_array_equal(vertical[..., 0], 0.)
    np.testing.assert_array_equal(vertical[..., -1], 0.)
    np.testing.assert_array_equal(north[:, -1], 0.)
    np.testing.assert_array_equal(east[np.asarray(result.pressure.east_area) == 0.], 0.)
    np.testing.assert_array_equal(north[np.asarray(result.pressure.north_area) == 0.], 0.)
    divergence = east - np.roll(east, 1, axis=0) + north
    divergence[:, 1:] -= north[:, :-1]
    divergence += vertical[..., 1:] - vertical[..., :-1]
    predicted_change = 60. * (np.asarray(source) - divergence)
    actual_volume = np.asarray(result.state.inventory.volume)
    observed_change = actual_volume - np.asarray(volume)
    scale = 60. * (np.abs(np.asarray(source)) + np.abs(divergence))
    floor = 64. * np.finfo(np.float64).eps * np.maximum(np.abs(actual_volume), np.abs(np.asarray(volume)))
    assert np.all(np.abs(observed_change - predicted_change) <= 1e-12 * scale + floor)
    for flux, fast_mean, endpoint, face_area in zip(
            (east, north), (result.barotropic.mean_east, result.barotropic.mean_north),
            (result.state.east_velocity, result.state.north_velocity),
            (result.pressure.east_area, result.pressure.north_area)):
        np.testing.assert_allclose(np.sum(flux, axis=-1), fast_mean, rtol=1e-12, atol=1e-6)
        rebuilt_endpoint = np.sum(np.asarray(face_area) * np.asarray(endpoint, dtype=np.float64), axis=-1)
        assert not np.allclose(rebuilt_endpoint, fast_mean, rtol=1e-4, atol=1e-6)
    wet = actual_volume > 0.
    actual_concentration = np.asarray(result.state.inventory.content)[wet] / actual_volume[wet, None]
    np.testing.assert_allclose(actual_concentration, np.broadcast_to([12., 35.], actual_concentration.shape), rtol=1e-12, atol=1e-12)
