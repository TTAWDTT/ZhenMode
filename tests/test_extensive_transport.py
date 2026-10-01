"""Independent physical-volume and shared-transport migration regressions."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from barotropic_transport import coupled_surface_step, subcycle_barotropic
from finite_volume import (
    ExtensiveState,
    VolumeFluxes,
    advance_contents,
    build_geometry,
    checked_transport_step,
    closed_surface_fluxes,
    horizontal_divergence,
    match_column_transport,
    surface_height,
    surface_volume,
)

jax.config.update("jax_enable_x64", True)


def _geometry(stair=False, nx=12, ny=6):
    depth = np.full((nx, ny), 50.)
    if stair:
        depth[3:6, 2:4] = 15.
        depth[7:9, 1:3] = 0.
    return build_geometry(np.linspace(0., 360., nx + 1),
                          np.linspace(-30., 30., ny + 1), [0., 5., 20., 50.], depth)


def _state(geometry, dtype=jnp.float64, constant=False):
    shape = geometry.thickness.shape
    eta = .2 * np.cos(2. * np.pi * np.arange(shape[0])[:, None] / shape[0])
    eta = np.broadcast_to(eta, shape[:2]) * (geometry.thickness[..., 0] > 0.)
    volume = surface_volume(geometry, jnp.asarray(eta, dtype=dtype))
    concentration = np.ones(shape + (2,)) * [12., 35.]
    if not constant:
        concentration[..., 0] += np.sin(np.arange(shape[0]))[:, None, None]
        concentration[..., 1] += np.arange(shape[2])[None, None, :] / 10.
    return ExtensiveState(volume, volume[..., None] * jnp.asarray(concentration, dtype)), eta


def test_physical_column_depth_and_exact_spherical_area():
    geometry = _geometry(stair=True)
    expected_depth = np.full((12, 6), 50.)
    expected_depth[3:6, 2:4] = 15.
    expected_depth[7:9, 1:3] = 0.
    np.testing.assert_allclose(np.sum(geometry.thickness, axis=-1), expected_depth,
                               rtol=0., atol=1e-13)
    np.testing.assert_array_equal(geometry.thickness[3, 2], [5., 10., 0.])
    np.testing.assert_array_equal(geometry.center_depth[3, 2], [2.5, 10., 0.])
    expected_area = 4. * np.pi * 6.371e6 ** 2 * np.sin(np.pi / 6.)
    assert np.sum(geometry.area) == pytest.approx(expected_area, rel=1e-13)
    assert np.sum(geometry.thickness[0, 0]) == 50.
    assert np.sum([5., 10., 22.5, 30.]) != 50.


def test_bottom_beyond_old_last_node_is_not_truncated():
    geometry = build_geometry([0., 180., 360.], [-10., 10.],
                              [0., 5., 4000., 8000.], np.array([[6100.], [7550.]]))
    np.testing.assert_allclose(np.sum(geometry.thickness, axis=-1), [[6100.], [7550.]])
    assert geometry.thickness[0, 0, -1] == 2100.


@pytest.mark.parametrize("lon,lat,levels,depth", [
    ([0., 170., 340.], [-10., 10.], [0., 50.], [[50.], [50.]]),
    ([0., 180., 360.], [10., -10.], [0., 50.], [[50.], [50.]]),
    ([0., 180., 360.], [-91., 10.], [0., 50.], [[50.], [50.]]),
    ([0., 180., 360.], [-10., 10.], [5., 50.], [[50.], [50.]]),
    ([0., 180., 360.], [-10., 10.], [0., 50., 30.], [[50.], [50.]]),
    ([0., 180., 360.], [-10., 10.], [0., 50.], [[51.], [50.]]),
    ([0., 180., 360.], [-10., 10.], [0., 50.], [[-1.], [50.]]),
    ([0., 180., 360.], [-10., 10.], [0., 50.], [[np.nan], [50.]]),
])
def test_invalid_geometry_is_rejected(lon, lat, levels, depth):
    with pytest.raises(ValueError):
        build_geometry(lon, lat, levels, depth)


def test_face_intersection_and_closed_nan_velocity():
    geometry = _geometry(stair=True)
    expected_east_height = np.minimum(geometry.thickness,
                                      np.roll(geometry.thickness, -1, axis=0))
    face_width = 6.371e6 * np.radians(10.)
    np.testing.assert_allclose(geometry.east_area, expected_east_height * face_width)
    velocity = jnp.where(geometry.east_area > 0., .1, jnp.nan)
    target = jnp.sum(jnp.asarray(geometry.east_area) * .2, axis=-1)
    result = match_column_transport(geometry.east_area, velocity, target)
    assert bool(result.valid)
    assert np.all(np.isfinite(result.flux))
    np.testing.assert_allclose(jnp.sum(result.flux, axis=-1), target, rtol=1e-13)
    assert np.all(np.asarray(result.flux)[geometry.east_area == 0.] == 0.)
    assert np.all(geometry.north_area[:, -1] == 0.)


def test_closed_target_or_open_nan_velocity_fails():
    geometry = _geometry(stair=True)
    zero = jnp.zeros_like(jnp.asarray(geometry.east_area))
    target = jnp.where(np.sum(geometry.east_area, axis=-1) == 0., 1., 0.)
    assert not bool(match_column_transport(geometry.east_area, zero, target).valid)
    velocity = zero.at[0, 0, 0].set(jnp.nan)
    assert not bool(match_column_transport(geometry.east_area, velocity,
                                         jnp.zeros(geometry.area.shape)).valid)


@pytest.mark.parametrize("dtype,tolerance", [(jnp.float64, 1e-12), (jnp.float32, 2e-6)])
@pytest.mark.parametrize("constant", [False, True])
def test_shared_flux_conserves_content_and_constant_or_bounded_tracer(dtype, tolerance, constant):
    geometry = _geometry(stair=True)
    state, _ = _state(geometry, dtype, constant)
    east = jnp.asarray(geometry.east_area, dtype) * jnp.asarray(.02, dtype)
    north = jnp.asarray(geometry.north_area, dtype) * jnp.asarray(-.01, dtype)
    fluxes = closed_surface_fluxes(east, north)
    result = jax.jit(advance_contents)(geometry, state, fluxes, jnp.asarray(60., dtype))
    assert bool(result.valid)
    wet = np.asarray(state.volume) > 0.
    initial = np.asarray(state.content)[wet] / np.asarray(state.volume)[wet, None]
    final = np.asarray(result.state.content)[wet] / np.asarray(result.state.volume)[wet, None]
    if constant:
        np.testing.assert_allclose(final, initial, rtol=tolerance, atol=tolerance)
    else:
        assert np.all(final.min(axis=0) >= initial.min(axis=0) - tolerance)
        assert np.all(final.max(axis=0) <= initial.max(axis=0) + tolerance)
    before = np.sum(np.asarray(state.content, dtype=np.float64), axis=(0, 1, 2))
    after = np.sum(np.asarray(result.state.content, dtype=np.float64), axis=(0, 1, 2))
    np.testing.assert_allclose(after, before, rtol=tolerance)
    np.testing.assert_allclose(result.state.volume[..., 1:], state.volume[..., 1:],
                               rtol=tolerance, atol=1e-6)
    assert np.max(np.abs(np.asarray(result.state.volume[..., 0] - state.volume[..., 0]))) > 0.


def test_explicit_freshwater_dilution_and_content_sources():
    geometry = _geometry()
    state, _ = _state(geometry, constant=True)
    zeros = jnp.zeros_like(state.volume)
    fluxes = closed_surface_fluxes(zeros, zeros)
    source = zeros.at[2, 3, 0].set(1e6)
    content_source = jnp.zeros_like(state.content).at[2, 3, 0, 0].set(8e6)
    result = checked_transport_step(geometry, state, fluxes, 60., source, content_source)
    np.testing.assert_allclose(result.state.volume, state.volume + 60. * source, rtol=1e-14)
    np.testing.assert_allclose(result.state.content, state.content + 60. * content_source,
                               rtol=1e-14)
    salt_before = state.content[2, 3, 0, 1] / state.volume[2, 3, 0]
    salt_after = result.state.content[2, 3, 0, 1] / result.state.volume[2, 3, 0]
    assert salt_after < salt_before


@pytest.mark.parametrize("invalid", ["cfl", "depletion", "dry_content", "dry_source", "top", "nan", "shape"])
def test_invalid_transport_is_fail_closed(invalid):
    geometry = _geometry(stair=True)
    state, _ = _state(geometry)
    zeros = jnp.zeros_like(state.volume)
    fluxes = closed_surface_fluxes(zeros, zeros)
    source = zeros
    if invalid == "cfl":
        fluxes = closed_surface_fluxes(zeros.at[0, 0, 0].set(state.volume[0, 0, 0] * 2.), zeros)
    elif invalid == "depletion":
        source = zeros.at[0, 0, 0].set(-state.volume[0, 0, 0] * 2.)
    elif invalid == "dry_content":
        state = state._replace(content=state.content.at[7, 1, 0, 0].set(1.))
    elif invalid == "dry_source":
        source = zeros.at[7, 1, 0].set(1.)
    elif invalid == "top":
        fluxes = fluxes._replace(vertical=fluxes.vertical.at[0, 0, 0].set(1.))
    elif invalid == "nan":
        fluxes = fluxes._replace(east=fluxes.east.at[0, 0, 0].set(jnp.nan))
    else:
        fluxes = VolumeFluxes(zeros[..., :-1], zeros, fluxes.vertical)
    with pytest.raises(ValueError):
        checked_transport_step(geometry, state, fluxes, 1., source)


def test_actual_barotropic_mean_not_endpoint_drives_surface():
    geometry = _geometry(nx=24, ny=8)
    state, eta = _state(geometry, constant=True)
    velocity_east = jnp.broadcast_to(.1 * jnp.sin(2. * jnp.pi * jnp.arange(24)[:, None] / 24.), eta.shape)
    velocity_north = jnp.zeros_like(velocity_east)
    result = subcycle_barotropic(geometry, jnp.asarray(eta), velocity_east, velocity_north,
                                150., 24)
    assert bool(result.valid)
    expected = eta - 3600. * horizontal_divergence(result.mean_east, result.mean_north) / geometry.area
    np.testing.assert_allclose(result.eta, expected, rtol=1e-12, atol=1e-13)
    endpoint = eta - 3600. * horizontal_divergence(
        jnp.sum(geometry.east_area, axis=-1) * result.east_velocity,
        jnp.sum(geometry.north_area, axis=-1) * result.north_velocity) / geometry.area
    scale = np.max(np.abs(np.asarray(result.eta) - eta))
    assert np.max(np.abs(np.asarray(result.eta - endpoint))) / scale > 1e-5
    coupled = coupled_surface_step(geometry, state, jnp.asarray(eta), velocity_east,
                                   velocity_north, jnp.zeros_like(state.volume),
                                   jnp.zeros_like(state.volume), 150., 24)
    assert bool(coupled.valid)
    assert float(coupled.surface_error) <= 1e-12
    wet = np.asarray(state.volume) > 0.
    concentration = np.asarray(coupled.transport.state.content)[wet] / np.asarray(coupled.transport.state.volume)[wet, None]
    np.testing.assert_allclose(concentration, np.asarray(state.content)[wet] / np.asarray(state.volume)[wet, None],
                               rtol=1e-12)


def test_unsafe_gravity_wave_or_initial_surface_is_rejected():
    geometry = _geometry()
    zeros = jnp.zeros(geometry.area.shape)
    result = subcycle_barotropic(geometry, zeros, zeros, zeros, 1e7, 1)
    assert not bool(result.valid)
    result = subcycle_barotropic(geometry, zeros - 6., zeros, zeros, 1., 1)
    assert not bool(result.valid)
    with pytest.raises(ValueError):
        subcycle_barotropic(geometry, zeros, zeros, zeros, 1., 0)


def test_shared_transport_local_jvp_and_vjp():
    geometry = _geometry()
    state, _ = _state(geometry)
    east = jnp.asarray(geometry.east_area) * .02
    north = jnp.asarray(geometry.north_area) * -.01
    fluxes = closed_surface_fluxes(east, north)
    direction = jnp.sin(jnp.arange(state.content.size)).reshape(state.content.shape) * state.volume[..., None]

    def update(content):
        return advance_contents(geometry, state._replace(content=content), fluxes, 60.).state.content

    _, tangent = jax.jvp(update, (state.content,), (direction,))
    delta = 1e-3
    finite_difference = (update(state.content + delta * direction) - update(state.content - delta * direction)) / (2. * delta)
    assert np.linalg.norm(np.asarray(tangent - finite_difference)) / np.linalg.norm(np.asarray(tangent)) < 1e-6
    cotangent = jnp.cos(jnp.arange(state.content.size)).reshape(state.content.shape)
    _, pullback = jax.vjp(update, state.content)
    left = jnp.vdot(tangent, cotangent)
    right = jnp.vdot(direction, pullback(cotangent)[0])
    assert float(left) == pytest.approx(float(right), rel=1e-12)


def test_oriented_flux_update_matches_independent_cell_loop():
    geometry = _geometry(stair=True, nx=12, ny=6)
    state, _ = _state(geometry)
    random = np.random.default_rng(731)
    east = random.uniform(-.01, .01, state.volume.shape) * geometry.east_area
    north = random.uniform(-.01, .01, state.volume.shape) * geometry.north_area
    vertical = np.zeros(state.volume.shape[:-1] + (state.volume.shape[-1] + 1,))
    vertical[..., 1:-1] = random.uniform(-100., 100., vertical[..., 1:-1].shape)
    wet = geometry.thickness > 0.
    vertical[..., 1:-1] *= wet[..., :-1] & wet[..., 1:]
    concentration = np.asarray(state.content) / np.where(np.asarray(state.volume)[..., None] > 0.,
                                                        np.asarray(state.volume)[..., None], 1.)
    expected_volume = np.array(state.volume)
    expected_content = np.array(state.content)
    dt = 60.
    for longitude in range(12):
        for latitude in range(6):
            for level in range(3):
                origin = (longitude, latitude, level)
                destinations = [((longitude + 1) % 12, latitude, level),
                                (longitude, latitude + 1, level),
                                (longitude, latitude, level + 1)]
                rates = [east[origin], north[origin], vertical[longitude, latitude, level + 1]]
                for destination, rate in zip(destinations, rates):
                    if rate == 0.:
                        continue
                    donor = origin if rate > 0. else destination
                    amount = dt * rate
                    expected_volume[origin] -= amount
                    expected_volume[destination] += amount
                    expected_content[origin] -= amount * concentration[donor]
                    expected_content[destination] += amount * concentration[donor]
    result = checked_transport_step(geometry, state, VolumeFluxes(east, north, vertical), dt)
    np.testing.assert_allclose(result.state.volume, expected_volume, rtol=1e-13, atol=1e-4)
    np.testing.assert_allclose(result.state.content, expected_content, rtol=1e-13, atol=.01)


def test_transport_flux_derivative_includes_moving_volume():
    geometry = _geometry()
    state, _ = _state(geometry)
    east = jnp.asarray(geometry.east_area) * .02
    north = jnp.asarray(geometry.north_area) * -.01
    direction = east

    def concentration_after(east_flux):
        result = advance_contents(geometry, state, closed_surface_fluxes(east_flux, north), 60.)
        return result.state.content / result.state.volume[..., None]

    _, tangent = jax.jvp(concentration_after, (east,), (direction,))
    delta = .05
    finite_difference = (concentration_after(east + delta * direction)
                         - concentration_after(east - delta * direction)) / (2. * delta)
    assert np.linalg.norm(np.asarray(tangent - finite_difference)) / np.linalg.norm(np.asarray(tangent)) < 1e-6


def test_coupled_explicit_top_sources_and_inconsistent_state():
    geometry = _geometry()
    state, eta = _state(geometry)
    zero = jnp.zeros(geometry.area.shape)
    layer_zero = jnp.zeros_like(state.volume)
    source = layer_zero.at[0, 0, 0].set(1e5)
    coupled = coupled_surface_step(geometry, state, jnp.asarray(eta), zero, zero,
                                   layer_zero, layer_zero, 10., 4, gravity=0.,
                                   volume_source=source)
    assert bool(coupled.valid)
    np.testing.assert_allclose(coupled.barotropic.eta, eta + 40. * source[..., 0] / geometry.area,
                               rtol=1e-12, atol=1e-14)
    wrong = state._replace(volume=state.volume.at[0, 0, 1].multiply(1.01))
    assert not bool(coupled_surface_step(geometry, wrong, jnp.asarray(eta), zero, zero,
                                        layer_zero, layer_zero, 10., 4).valid)
    assert not bool(coupled_surface_step(geometry, state, jnp.asarray(eta) + .01, zero, zero,
                                        layer_zero, layer_zero, 10., 4).valid)


def test_float32_surface_identity_on_data_free_earth_scale_grid():
    nx, ny = 180, 66
    depth = np.full((nx, ny), 50.)
    depth[30:60, 10:25] = 15.
    depth[80:90, 40:50] = 0.
    geometry = build_geometry(np.linspace(0., 360., nx + 1),
                              np.linspace(-66., 66., ny + 1), [0., 5., 20., 50.], depth)
    geometry = jax.tree_util.tree_map(lambda value: jnp.asarray(value, jnp.float32), geometry)
    phase = 2. * np.pi * np.arange(nx)[:, None] / nx
    requested = jnp.asarray(np.broadcast_to(.2 * np.cos(phase), (nx, ny)) * (depth > 0.), jnp.float32)
    volume = surface_volume(geometry, requested)
    physical_eta = (np.asarray(volume[..., 0], dtype=np.float64) / np.asarray(geometry.area, dtype=np.float64)
                    - np.asarray(geometry.thickness[..., 0], dtype=np.float64)).astype(np.float32)
    state = ExtensiveState(volume, volume[..., None] * jnp.asarray([12., 35.], jnp.float32))
    east = jnp.asarray(np.broadcast_to(.02 * np.sin(phase), (nx, ny)), jnp.float32)
    north = jnp.zeros_like(east)
    layer_zero = jnp.zeros_like(volume)
    result = jax.jit(lambda: coupled_surface_step(geometry, state, jnp.asarray(physical_eta), east,
                                                north, layer_zero, layer_zero, 15., 4))()
    assert bool(result.barotropic.valid)
    assert bool(result.transport.valid)
    assert float(result.surface_error) <= 2e-6
    assert bool(result.valid)


def test_surface_height_matches_independent_physical_ratio_and_keeps_dtype():
    geometry = _geometry()
    state, _ = _state(geometry, jnp.float32)
    actual = surface_height(geometry, state.volume)
    area = np.asarray(geometry.area, dtype=np.float32).astype(np.float64)
    height = np.asarray(geometry.thickness[..., 0], dtype=np.float32).astype(np.float64)
    expected = (np.asarray(state.volume[..., 0], dtype=np.float64) / area - height).astype(np.float32)
    assert actual.dtype == jnp.float32
    np.testing.assert_array_equal(actual, expected)
    with jax.enable_x64(False):
        with pytest.raises(ValueError, match="X64"):
            surface_height(geometry, state.volume)


def test_integer_surface_or_face_velocity_is_rejected():
    geometry = _geometry()
    with pytest.raises(ValueError, match="floating"):
        surface_volume(geometry, jnp.zeros(geometry.area.shape, dtype=jnp.int32))
    with pytest.raises(ValueError, match="floating"):
        match_column_transport(geometry.east_area, jnp.zeros(geometry.thickness.shape, dtype=jnp.int32),
                               jnp.zeros(geometry.area.shape))
