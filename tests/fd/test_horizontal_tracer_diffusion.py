"""Conservation, dissipation and accuracy of the default scalar diffusion path."""


import jax
import jax.numpy as jnp
import numpy as np
import pytest

from config import R_EARTH
from jax_solver_global import (
    JaxStateG,
    _horizontal_biharmonic_tracer,
    _horizontal_tracer_diffusion,
    _linear_half_step,
)
from tests.support.fd.horizontal_diffusion import _parameters


@pytest.mark.parametrize("latitude_limit", [30., 65.])
@pytest.mark.parametrize("land", [False, True])
@pytest.mark.parametrize("seed", [2718, 3141, 1618])
@pytest.mark.parametrize("biharmonic", [False, True])
def test_default_diffusion_closes_weighted_budget(latitude_limit, land, seed, biharmonic):
    _, params, volume = _parameters(latitude_limit, land)
    tracer = jnp.asarray(np.random.default_rng(seed).uniform(5., 25., volume.shape))
    tendency = np.asarray(-_horizontal_biharmonic_tracer(tracer, params) if biharmonic
                          else _horizontal_tracer_diffusion(tracer, params))
    assert abs(np.sum(tendency * volume)) <= 5e-13 * np.sum(np.abs(tendency) * volume)

@pytest.mark.parametrize("biharmonic", [False, True])
def test_default_diffusion_does_not_read_closed_faces(biharmonic):
    _, params, _ = _parameters(65., land=True)
    wet = np.asarray(params.wet_mask_z) > 0.5
    tracer = np.random.default_rng(2718).uniform(5., 25., wet.shape)
    operator = _horizontal_biharmonic_tracer if biharmonic else _horizontal_tracer_diffusion
    first = operator(jnp.asarray(np.where(wet, tracer, -100.)), params)
    second = operator(jnp.asarray(np.where(wet, tracer, 100.)), params)
    np.testing.assert_array_equal(np.asarray(first)[wet], np.asarray(second)[wet])
    np.testing.assert_array_equal(np.asarray(first)[~wet], 0.)

@pytest.mark.parametrize("land", [False, True])
@pytest.mark.parametrize("biharmonic", [False, True])
def test_default_diffusion_preserves_constants_and_is_self_adjoint(land, biharmonic):
    _, params, volume = _parameters(65., land)
    random = np.random.default_rng(2718)
    first, second = [jnp.asarray(random.normal(size=volume.shape)) for _ in range(2)]
    def operator(field):
        return (-_horizontal_biharmonic_tracer(field, params) if biharmonic
                else _horizontal_tracer_diffusion(field, params))

    first_tendency = np.asarray(operator(first))
    second_tendency = np.asarray(operator(second))
    np.testing.assert_array_equal(operator(jnp.full(volume.shape, 20.)), 0.)
    assert np.sum(np.asarray(first) * first_tendency * volume) <= 0.
    forward = np.sum(np.asarray(first) * second_tendency * volume)
    backward = np.sum(np.asarray(second) * first_tendency * volume)
    assert forward == pytest.approx(backward, rel=5e-13)

def test_laplacian_safe_step_cannot_create_wet_extrema():
    _, params, _ = _parameters(65., land=True)
    wet = np.asarray(params.wet_mask_z) > 0.5
    tracer = jnp.asarray(np.random.default_rng(2718).uniform(5., 25., wet.shape))
    rate = params.kappa_h * float(jnp.max(params.inv_dx2) + params.inv_dy2)
    updated = np.asarray(tracer + (0.25 / rate) * _horizontal_tracer_diffusion(tracer, params))
    assert updated[wet].min() >= np.asarray(tracer)[wet].min()
    assert updated[wet].max() <= np.asarray(tracer)[wet].max()

@pytest.mark.parametrize("kappa_h,kappa_bi", [(100., 0.), (0., 1e15), (100., 1e15)])
def test_linear_diffusion_step_closes_heat_and_salt(kappa_h, kappa_bi):
    _, params, volume = _parameters(65., land=True)
    params = params._replace(kappa_h=kappa_h, kappa_bi=kappa_bi)
    random = np.random.default_rng(2718)
    temperature = jnp.asarray(random.uniform(5., 25., volume.shape))
    salinity = jnp.asarray(random.uniform(30., 37., volume.shape))
    zeros = jnp.zeros(volume.shape)
    state = JaxStateG(zeros, zeros, temperature, salinity,
                      jnp.zeros(volume.shape[:2]), jnp.zeros(volume.shape[:2]))
    updated = _linear_half_step(state, params, 30.)
    for name in ("T", "S"):
        before = np.asarray(getattr(state, name))
        after = np.asarray(getattr(updated, name))
        roundoff = 100. * np.finfo(before.dtype).eps * np.sum(np.abs(before) * volume)
        assert abs(np.sum((after - before) * volume)) <= roundoff
        mean = np.sum(before * volume) / np.sum(volume)
        assert np.sum((after - mean) ** 2 * volume) <= np.sum((before - mean) ** 2 * volume)

def test_background_diffusion_has_second_order_spherical_accuracy():
    errors = []
    for ny in (24, 48, 96):
        grid, params, volume = _parameters(50., ny=ny, nx=2 * ny)
        longitude = 2. * np.pi * np.arange(grid.nx) / grid.nx
        tracer = np.broadcast_to(
            np.sin(longitude)[:, None, None] * grid.cos_lat[None, :, None], volume.shape)
        exact = -2. * params.kappa_h * tracer / R_EARTH ** 2
        computed = np.asarray(_horizontal_tracer_diffusion(jnp.asarray(tracer), params))
        interior = (slice(None), slice(4, -4), slice(None))
        errors.append(np.sqrt(np.sum((computed[interior] - exact[interior]) ** 2 * volume[interior])
                              / np.sum(exact[interior] ** 2 * volume[interior])))
    assert errors[0] / errors[1] >= 3.5
    assert errors[1] / errors[2] >= 3.5

def test_diffusion_tangent_matches_finite_difference():
    _, params, _ = _parameters(65., land=True)
    random = np.random.default_rng(2718)
    tracer, direction = [jnp.asarray(random.normal(size=params.wet_mask_z.shape)) for _ in range(2)]
    def operator(field):
        return _horizontal_tracer_diffusion(field, params)

    _, tangent = jax.jvp(operator, (tracer,), (direction,))
    epsilon = 1e-4
    finite_difference = (operator(tracer + epsilon * direction) - operator(tracer - epsilon * direction)) / (2. * epsilon)
    np.testing.assert_allclose(tangent, finite_difference, rtol=1e-8, atol=1e-18)
