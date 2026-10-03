"""Independent gates for shared extensive FCT, not a full ocean qualification."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from zhenmode_research.candidates.fv.barotropic import coupled_surface_step
from zhenmode_research.candidates.fv.transport import advance_bounded_contents, euler_fct_step
from zhenmode_research.candidates.fv.geometry import (
    ExtensiveState,
    advance_contents,
    build_geometry,
    closed_surface_fluxes,
    surface_height,
    surface_volume,
)

jax.config.update("jax_enable_x64", True)


def _fixture(constant=False):
    depth = np.full((16, 6), 50.)
    depth[3:6, 2:4] = 15.
    depth[10:12, 1:3] = 0.
    geometry = build_geometry(np.linspace(0., 360., 17), np.linspace(-30., 30., 7),
                              [0., 5., 20., 50.], depth)
    phase = 2. * np.pi * np.arange(16)[:, None] / 16.
    eta = jnp.asarray(np.broadcast_to(.2 * np.cos(phase), depth.shape) * (depth > 0.))
    volume = surface_volume(geometry, eta)
    concentration = np.ones(volume.shape + (2,)) * [12., 35.]
    if not constant:
        concentration[..., 0] += np.sin(phase)[..., None] + np.arange(3)[None, None, :] / 10.
        concentration[..., 1] += np.arange(6)[None, :, None] / 100.
    state = ExtensiveState(volume, volume[..., None] * concentration)
    fluxes = closed_surface_fluxes(jnp.asarray(geometry.east_area) * .02,
                                  jnp.asarray(geometry.north_area) * -.01)
    return geometry, state, fluxes


@pytest.mark.parametrize("constant", [False, True])
@pytest.mark.parametrize("advance", [euler_fct_step, advance_bounded_contents])
def test_shared_three_direction_exchange_budget_bounds_and_partial_cells(constant, advance):
    geometry, state, fluxes = _fixture(constant)
    result = jax.jit(advance)(geometry, state, fluxes, 60.)
    assert bool(result.valid)
    wet = geometry.thickness > 0.
    before = np.asarray(state.content)[wet] / np.asarray(state.volume)[wet, None]
    after = np.asarray(result.state.content)[wet] / np.asarray(result.state.volume)[wet, None]
    assert np.all(after.min(axis=0) >= before.min(axis=0) - 1e-12)
    assert np.all(after.max(axis=0) <= before.max(axis=0) + 1e-12)
    np.testing.assert_allclose(np.sum(result.state.content, axis=(0, 1, 2)),
                               np.sum(state.content, axis=(0, 1, 2)), rtol=1e-12)
    np.testing.assert_allclose(np.sum(result.state.volume), np.sum(state.volume), rtol=1e-12)
    np.testing.assert_array_equal(np.asarray(result.state.content)[~wet], 0.)
    np.testing.assert_array_equal(np.asarray(result.state.volume)[~wet], 0.)
    if constant:
        np.testing.assert_allclose(after, before, rtol=1e-12, atol=1e-12)
    else:
        donor = advance_contents(geometry, state, fluxes, 60.)
        correction = np.asarray(result.state.content - donor.state.content)
        assert np.max(np.abs(correction)) > 1.
        np.testing.assert_allclose(np.sum(correction, axis=(0, 1, 2)), 0.,
                                   atol=1e-12 * float(np.max(state.content)))
    assert np.any(np.asarray(fluxes.vertical[..., 1:-1]) != 0.)


def test_explicit_sources_close_totals_and_dilute_without_salt_refill():
    geometry, state, fluxes = _fixture(True)
    zero = jnp.zeros_like(state.volume)
    fluxes = closed_surface_fluxes(zero, zero)
    source = zero.at[2, 3, 0].set(1e6)
    content_source = jnp.zeros_like(state.content).at[2, 3, 0, 0].set(8e6)
    result = advance_bounded_contents(geometry, state, fluxes, 60., source, content_source)
    assert bool(result.valid)
    np.testing.assert_allclose(result.state.volume, state.volume + 60. * source, rtol=1e-13)
    np.testing.assert_allclose(result.state.content, state.content + 60. * content_source, rtol=1e-13)
    assert result.state.content[2, 3, 0, 1] / result.state.volume[2, 3, 0] < 35.


def test_euler_matches_independent_cell_loop_all_direction_fct():
    geometry, state, fluxes = _fixture()
    volume, content = np.asarray(state.volume), np.asarray(state.content)
    wet = volume > 0.
    concentration = content / np.where(wet, volume, 1.)[..., None]
    height = volume / geometry.area[..., None]
    widths = (np.broadcast_to(geometry.east_width[..., None], volume.shape),
              np.broadcast_to(geometry.north_width[..., None], volume.shape), height)
    edges, lookup = [], {}
    face_arrays = (np.asarray(fluxes.east), np.asarray(fluxes.north), np.asarray(fluxes.vertical[..., 1:]))
    for origin in np.ndindex(volume.shape):
        for axis in range(3):
            following = list(origin)
            following[axis] += 1
            if axis == 0:
                following[axis] %= volume.shape[axis]
            elif following[axis] == volume.shape[axis]:
                continue
            following = tuple(following)
            if not (wet[origin] and wet[following]):
                continue
            if axis == 0:
                distance = geometry.east_distance[origin[:2]]
            elif axis == 1:
                distance = geometry.north_distance[origin[:2]]
            else:
                distance = .5 * (height[origin] + height[following])
            lookup[axis, origin] = (following, distance)
            edges.append((origin, following, axis, face_arrays[axis][origin]))
    slopes = np.zeros((3,) + content.shape)
    lower, upper = concentration.copy(), concentration.copy()
    for origin, following, axis, unused_flux in edges:
        lower[origin] = np.minimum(lower[origin], concentration[following])
        upper[origin] = np.maximum(upper[origin], concentration[following])
        lower[following] = np.minimum(lower[following], concentration[origin])
        upper[following] = np.maximum(upper[following], concentration[origin])
        previous = list(origin)
        previous[axis] -= 1
        if axis == 0:
            previous[axis] %= volume.shape[axis]
        previous = tuple(previous)
        if (axis, previous) not in lookup:
            continue
        distance_left = lookup[axis, previous][1]
        distance_right = lookup[axis, origin][1]
        gradient_left = (concentration[origin] - concentration[previous]) / distance_left
        gradient_right = (concentration[following] - concentration[origin]) / distance_right
        slopes[(axis,) + origin] = (distance_right * gradient_left + distance_left * gradient_right) / (distance_left + distance_right)
    low_volume, low_content = volume.copy(), content.copy()
    positive, negative = np.zeros_like(content), np.zeros_like(content)
    amounts = []
    for origin, following, axis, flux in edges:
        donor = origin if flux >= 0. else following
        low_amount = 60. * flux * concentration[donor]
        low_volume[origin] -= 60. * flux
        low_volume[following] += 60. * flux
        low_content[origin] -= low_amount
        low_content[following] += low_amount
        high = (concentration[origin] + .5 * widths[axis][origin] * slopes[(axis,) + origin]
                if flux >= 0. else concentration[following] - .5 * widths[axis][following] * slopes[(axis,) + following])
        amount = 60. * flux * (high - concentration[donor])
        positive[origin] += np.maximum(-amount, 0.)
        negative[origin] += np.maximum(amount, 0.)
        positive[following] += np.maximum(amount, 0.)
        negative[following] += np.maximum(-amount, 0.)
        amounts.append((origin, following, amount))
    ratio_positive = np.minimum(1., np.maximum(low_volume[..., None] * upper - low_content, 0.) / np.where(positive > 0., positive, 1.))
    ratio_negative = np.minimum(1., np.maximum(low_content - low_volume[..., None] * lower, 0.) / np.where(negative > 0., negative, 1.))
    expected = low_content.copy()
    for origin, following, amount in amounts:
        coefficient = np.where(amount >= 0., np.minimum(ratio_negative[origin], ratio_positive[following]),
                               np.minimum(ratio_positive[origin], ratio_negative[following]))
        expected[origin] -= coefficient * amount
        expected[following] += coefficient * amount
    result = euler_fct_step(geometry, state, fluxes, 60.)
    assert bool(result.valid)
    np.testing.assert_allclose(result.state.volume, low_volume, rtol=1e-13)
    np.testing.assert_allclose(result.state.content, expected, rtol=1e-13)


@pytest.mark.parametrize("invalid", ["dry_source", "nan", "top", "cfl", "depletion"])
def test_invalid_intermediate_steps_fail_closed(invalid):
    geometry, state, fluxes = _fixture()
    zero = jnp.zeros_like(state.volume)
    source = None
    if invalid == "dry_source":
        source = zero.at[10, 1, 0].set(1.)
    elif invalid == "nan":
        fluxes = fluxes._replace(east=fluxes.east.at[0, 0, 0].set(jnp.nan))
    elif invalid == "top":
        fluxes = fluxes._replace(vertical=fluxes.vertical.at[0, 0, 0].set(1.))
    elif invalid == "cfl":
        fluxes = closed_surface_fluxes(jnp.asarray(geometry.east_area) * 1e8, zero)
    else:
        source = -state.volume / 90.
        fluxes = closed_surface_fluxes(zero, zero)
        assert bool(advance_contents(geometry, state, fluxes, 60., source).valid)
    assert not bool(advance_bounded_contents(geometry, state, fluxes, 60., source).valid)


def test_nonuniform_cell_widths_are_not_center_distances():
    geometry = build_geometry([0., 30., 120., 240., 360.], [-30., -10., 30.],
                              [0., 50.], np.full((4, 2), 50.))
    expected_east = 6.371e6 * np.radians([30., 90., 120., 120.])[:, None] * np.cos(np.radians([-20., 10.]))[None, :]
    expected_north = np.broadcast_to(6.371e6 * np.radians([20., 40.])[None, :], (4, 2))
    np.testing.assert_allclose(geometry.east_width, expected_east, rtol=1e-13)
    np.testing.assert_allclose(geometry.north_width, expected_north, rtol=1e-13)
    assert not np.allclose(geometry.east_width, geometry.east_distance)
    assert not np.allclose(geometry.north_width, geometry.north_distance)


def _translate(nx, advance):
    geometry = build_geometry(np.linspace(0., 360., nx + 1), [-30., 0., 30.],
                              [0., 50.], np.full((nx, 2), 50.))
    volume = surface_volume(geometry, jnp.zeros((nx, 2)))
    phase = 2. * np.pi * (np.arange(nx) + .5) / nx
    concentration = np.broadcast_to((2. + np.cos(phase) * np.sinc(1. / nx))[:, None, None, None], (nx, 2, 1, 1))
    state = ExtensiveState(volume, volume[..., None] * concentration)
    fluxes = closed_surface_fluxes(volume * nx, jnp.zeros_like(volume))
    dt = .2 / nx
    steps = int(.25 / dt)

    @jax.jit
    def integrate(initial):
        def step(carry, unused):
            current, valid = carry
            result = advance(geometry, current, fluxes, dt)
            return (result.state, valid & result.valid), None
        return jax.lax.scan(step, (initial, jnp.asarray(True)), None, length=steps)[0]

    final, valid = integrate(state)
    assert bool(valid)
    actual = np.asarray(final.content / final.volume[..., None])[:, 0, 0, 0]
    exact = 2. + np.cos(phase - .5 * np.pi) * np.sinc(1. / nx)
    np.testing.assert_allclose(np.sum(final.content), np.sum(state.content), rtol=1e-12)
    assert actual.min() >= concentration.min() - 1e-12
    assert actual.max() <= concentration.max() + 1e-12
    return float(np.sqrt(np.mean((actual - exact) ** 2)))


def test_exact_cell_average_cosine_second_order_not_just_rk_name():
    errors = [_translate(nx, advance_bounded_contents) for nx in (32, 64, 128)]
    ratios = np.asarray(errors[:-1]) / errors[1:]
    donor_error = _translate(128, advance_contents)
    print(f"cosine L2 errors={errors}, ratios={ratios.tolist()}, donor128={donor_error}")
    assert np.all(ratios >= 3.2)
    assert errors[-1] < donor_error


def test_local_jvp_and_vjp_away_from_sign_and_limiter_ties():
    geometry, state, fluxes = _fixture()
    random = np.random.default_rng(732)
    state = state._replace(content=state.content * jnp.asarray(1. + .01 * random.normal(size=state.content.shape)))
    tangent = jnp.asarray(random.normal(size=state.content.shape)) * state.volume[..., None]
    cotangent = jnp.asarray(random.normal(size=state.content.shape))

    def update(content):
        return advance_bounded_contents(geometry, state._replace(content=content), fluxes, 60.).state.content

    _, derivative = jax.jvp(update, (state.content,), (tangent,))
    epsilon = 1e-5
    difference = (update(state.content + epsilon * tangent) - update(state.content - epsilon * tangent)) / (2. * epsilon)
    relative = np.linalg.norm(np.asarray(derivative - difference)) / np.linalg.norm(np.asarray(derivative))
    assert relative <= 1e-6
    _, pullback = jax.vjp(update, state.content)
    adjoint = pullback(cotangent)[0]
    np.testing.assert_allclose(jnp.vdot(cotangent, derivative), jnp.vdot(adjoint, tangent), rtol=1e-12)


def _couple(policy=None, inventory_dtype=jnp.float64, momentum_dtype=jnp.float32):
    geometry, state, _ = _fixture(True)
    eta = surface_height(geometry, state.volume).astype(momentum_dtype)
    zero = jnp.zeros(geometry.area.shape, momentum_dtype)
    layer_zero = jnp.zeros(state.volume.shape, momentum_dtype)
    state = ExtensiveState(state.volume.astype(inventory_dtype), state.content.astype(inventory_dtype))
    return coupled_surface_step(geometry, state, eta, zero, zero, layer_zero, layer_zero,
                                15., 4, inventory_precision=policy, transport_scheme="centered_fct")


def test_mixed_inventory_policy_is_explicit_and_preserves_dtypes():
    with pytest.raises(ValueError, match="dtype"):
        _couple()
    result = _couple("float64")
    assert bool(result.valid)
    assert result.transport.state.volume.dtype == jnp.float64
    assert result.transport.state.content.dtype == jnp.float64
    assert result.barotropic.eta.dtype == jnp.float32
    assert result.barotropic.east_velocity.dtype == jnp.float32


@pytest.mark.parametrize("policy,inventory", [("automatic", jnp.float64), ("float64", jnp.float32)])
def test_unregistered_inventory_policy_rejected(policy, inventory):
    with pytest.raises(ValueError):
        _couple(policy, inventory)
