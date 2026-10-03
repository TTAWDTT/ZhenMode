"""Independent closed-face spherical scalar viscosity and spatial MMS gates."""
from dataclasses import replace
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ocean_solver.config.definitions import R_EARTH
from ocean_solver.geometry.fd import make_fd_params
from ocean_solver.numerics.horizontal import _biharmonic_h, _laplacian_h
from tests.support.grid import all_wet_grid
from tests.support.material.joint_momentum import _controlled_factory, _numpy_operators
from tests.support.material.reference_geometry import WIDTHS
from zhenmode_research.candidates.material.solver import _momentum_diffusion_norm_bound


def _face_quadratic(velocity, params):
    wet = np.asarray(params.wet_mask_z)
    area = np.asarray(params.dx_2d) * params.dy
    widths = np.asarray(params.dz_node)
    zonal = (np.roll(velocity, -1, axis=0) - velocity) ** 2
    zonal *= wet * np.roll(wet, -1, axis=0) * (area / np.asarray(params.dx_2d) ** 2)[..., None] * widths
    meridional = (velocity[:, 1:] - velocity[:, :-1]) ** 2
    cosine_face = .5 * (np.asarray(params.cos_lat)[1:] + np.asarray(params.cos_lat)[:-1])
    conductance = area[:, :-1] / np.asarray(params.cos_lat)[None, :-1] * cosine_face / params.dy ** 2
    meridional *= wet[:, 1:] * wet[:, :-1] * conductance[..., None] * widths
    return -np.sum(zonal) - np.sum(meridional)


@pytest.mark.parametrize("stairs", [False, True])
def test_closed_spherical_viscosity_matches_dense_faces_and_negative_quadratic(stairs):
    _, params, initial = _controlled_factory(stairs=stairs, metric=True)
    horizontal, _, operator = _numpy_operators(params)
    generator = np.random.default_rng(293002)
    velocity = generator.normal(size=initial.u.shape)
    velocity[np.asarray(params.wet_mask_z) == 0.] = 123.
    bound = float(np.max(np.sum(np.abs(horizontal), axis=1)))
    floor = 64. * np.finfo(float).eps * bound * max(1., np.max(np.abs(velocity)))
    actual = np.asarray(jax.jit(lambda field: _laplacian_h(field, params))(jnp.asarray(velocity)))
    expected = (horizontal @ velocity.ravel()).reshape(velocity.shape)
    assert np.max(np.abs(actual - expected)) <= floor
    weights = np.asarray(params.dx_2d)[..., None] * params.dy * WIDTHS
    weighted = horizontal * weights.ravel()[:, None]
    assert np.max(np.abs(weighted - weighted.T)) <= 64. * np.finfo(float).eps * np.max(np.abs(weighted))
    quadratic = float(np.sum(weights * velocity * actual))
    expected_quadratic = float(_face_quadratic(velocity, params))
    np.testing.assert_allclose(quadratic, expected_quadratic, rtol=1e-13, atol=0.)
    assert quadratic < 0.
    biharmonic = np.asarray(jax.jit(lambda field: _biharmonic_h(field, params))(jnp.asarray(velocity)))
    np.testing.assert_allclose(biharmonic, (horizontal @ horizontal @ velocity.ravel()).reshape(velocity.shape), rtol=1e-12, atol=floor * bound)
    np.testing.assert_allclose(np.sum(weights * velocity * biharmonic), np.sum(weights * actual ** 2), rtol=1e-13, atol=0.)
    assert np.max(np.sum(np.abs(operator), axis=1)) <= float(_momentum_diffusion_norm_bound(params)) * (1. + 1e-14)


def test_nodal_constant_coastal_velocity_never_reads_the_dry_sentinel():
    _, params, initial = _controlled_factory(stairs=True, metric=True)
    wet = np.asarray(params.wet_mask_z) > 0.
    baseline = None
    for sentinel in (0., 123., -1e6):
        velocity = jnp.asarray(np.where(wet, .1, sentinel))
        response = np.asarray(_laplacian_h(velocity, params))
        np.testing.assert_array_equal(response, np.zeros(initial.u.shape))
        np.testing.assert_array_equal(_biharmonic_h(velocity, params), np.zeros(initial.u.shape))
        if baseline is not None:
            np.testing.assert_array_equal(response, baseline)
        baseline = response
    old = params._replace(column_geometry="legacy")
    assert np.max(np.abs(np.asarray(_laplacian_h(jnp.asarray(np.where(wet, .1, 0.)), old))[wet])) > 1e-16


def test_spherical_scalar_mms_is_second_order_without_using_wall_error():
    errors = []
    for degrees in (2., 1., .5):
        longitude = np.arange(degrees / 2., 360., degrees)
        latitude = np.arange(-60. + degrees / 2., 60., degrees)
        grid = all_wet_grid(nx=len(longitude), ny=len(latitude), nz=2)
        cosine = np.cos(np.deg2rad(latitude))
        spacing = R_EARTH * np.deg2rad(degrees)
        grid = replace(grid, lon=longitude, lat=latitude, cos_lat=cosine, dy=spacing,
                       dx_2d=np.broadcast_to(spacing * cosine, (len(longitude), len(latitude))).copy())
        base = make_fd_params(grid, column_geometry="nodal_dual_v1")
        params = SimpleNamespace(**base._asdict(), column_geometry="nodal_dual_v1",
                                 coastal_kappa_h_2d=jnp.zeros((grid.nx, grid.ny)))
        field = np.cos(np.deg2rad(longitude))[:, None] * cosine[None, :]
        field = np.broadcast_to(field[..., None], params.wet_mask_z.shape)
        actual = np.asarray(jax.jit(lambda values: _laplacian_h(values, params))(jnp.asarray(field)))
        expected = -2. * field / R_EARTH ** 2
        interior_error = actual[:, 1:-1] - expected[:, 1:-1]
        area = np.asarray(params.dx_2d)[:, 1:-1] * params.dy
        errors.append(np.sqrt(np.sum(interior_error ** 2 * area[..., None]) / (area.sum() * field.shape[-1])))
    assert errors[0] / errors[1] >= 3.3 and errors[1] / errors[2] >= 3.3, errors
