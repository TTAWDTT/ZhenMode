"""Independent moving-coordinate controls before any material factory cutover."""

from tests.support.rstar.metric_controls import _fixture, _geometry, _pressure_coordinate_errors, _state

from dataclasses import replace

from types import SimpleNamespace

import jax

import jax.numpy as jnp

import numpy as np

import pytest

from tests.support.grid import all_wet_grid

from config import ALPHA_T, G_EARTH, R_EARTH, RHO_0

from jax_solver_global import (
    JaxStateG,
    _advection_scalar,
    _compute_hydrostatic_pressure,
    _compute_pressure_gradient,
    _d2_dz2_flux,
    _face_transport_divergence,
    _gradient_conservative_3d,
    _horizontal_diffusion_flux,
    make_fd_params,
)

from research.experiments.material_rstar_coordinates.dense_oracle import assemble_diffusion

from research.experiments.material_rstar_coordinates.kernel import (
    coordinate_pressure_gradient,
    diffusion_content_rhs,
    diffusion_weights,
    hydrostatic_pressure,
    make_reference_stencil,
    physical_diffusion_gradients,
    relative_vertical_transport,
    rstar_geometry,
)

def test_rstar_surface_bed_total_depth_and_invalid_geometry_are_retained():
    _, params, depths = _fixture()
    reference = np.asarray(params.dz_node) * np.asarray(params.wet_mask_z)
    depth = np.sum(reference, axis=-1)
    wet = depth > 0.
    eta = np.where(wet, -.6 * depth, 0.)
    geometry = jax.jit(lambda surface: _geometry(params, depths, surface))(jnp.asarray(eta))
    assert bool(geometry.valid)
    np.testing.assert_allclose(np.asarray(geometry.thickness).sum(axis=-1), depth + eta, rtol=1e-14)
    np.testing.assert_array_equal(geometry.interface_depth[..., 0], -eta)
    np.testing.assert_allclose(geometry.interface_depth[..., -1], depth, rtol=1e-14)
    assert np.all(np.asarray(geometry.thickness)[np.asarray(params.wet_mask_z) > 0.] > 0.)
    rejected = _geometry(params, depths, np.where(wet, -1.01 * depth, 0.))
    assert not bool(rejected.valid)
    assert np.min(rejected.thickness) < 0.
    with pytest.raises(ValueError, match="contiguous"):
        invalid_wet = np.asarray(params.wet_mask_z).copy()
        invalid_wet[3, 2, 1] = 0.
        make_reference_stencil(depths, invalid_wet)

@pytest.mark.parametrize("limited", [False, True])
def test_original_three_dimensional_transport_preserves_constant_rstar_content(limited):
    _, params, depths = _fixture()
    params.fct_adv = limited
    generator = np.random.default_rng(930001)
    wet = np.asarray(params.wet_mask_z)
    eta = jnp.full(params.wet_mask.shape, -2.6) * params.wet_mask
    geometry = _geometry(params, depths, eta)
    east = generator.normal(0., .0001, wet.shape) * wet * np.roll(wet, -1, axis=0)
    north = generator.normal(0., .0001, wet.shape) * wet * np.roll(wet, -1, axis=1)
    north[:, -1] = 0.
    faces = jnp.asarray(east), jnp.asarray(north)
    divergence = _face_transport_divergence(*faces, params)
    relative = relative_vertical_transport(divergence, geometry)
    eta_rate = -np.sum(np.asarray(divergence), axis=-1)
    concentration = jnp.full(wet.shape, 35.)
    zeros = jnp.zeros_like(concentration)
    content_rate = params.dz_node * _advection_scalar(concentration, zeros, zeros, relative, params, face_transport=faces)
    expected = 35. * np.asarray(geometry.weights) * eta_rate[..., None]
    scale = np.abs(expected) + 35. * np.abs(np.asarray(divergence)) + 35. * np.max(np.abs(np.asarray(relative)))
    floor = 64. * np.finfo(float).eps * (1e-20 + scale)
    assert np.all(np.abs(np.asarray(content_rate) - expected) <= floor)
    assert np.max(np.abs(np.asarray(relative)[..., -1])) <= 64. * np.finfo(float).eps * np.sum(np.abs(divergence))

def test_resting_affine_physical_density_needs_the_coordinate_pressure_correction():
    _, params, depths = _fixture()
    eta = jnp.full(params.wet_mask.shape, -2.6) * params.wet_mask
    geometry = _geometry(params, depths, eta)
    rho = 1.5 + .002 * geometry.node_depth
    pressure = hydrostatic_pressure(rho, eta, geometry, params)
    acceleration = coordinate_pressure_gradient(pressure, rho, geometry, params)
    bound = 64. * np.finfo(float).eps * float(jnp.max(jnp.abs(pressure))) / RHO_0
    bound *= max(float(jnp.max(params.inv_dx)), float(params.inv_dy))
    assert max(float(jnp.max(jnp.abs(value))) for value in acceleration) <= bound
    uncorrected = _gradient_conservative_3d(pressure, params)
    assert max(float(jnp.max(jnp.abs(value))) / RHO_0 for value in uncorrected) > 1000. * bound
    wet = np.asarray(params.wet_mask_z) > 0.
    for sentinel in (123., -1e6):
        dry_rho = jnp.asarray(np.where(wet, np.asarray(rho), sentinel))
        dry_pressure = hydrostatic_pressure(dry_rho, eta, geometry, params)
        np.testing.assert_array_equal(dry_pressure, pressure)
        for current, original in zip(coordinate_pressure_gradient(dry_pressure, dry_rho, geometry, params), acceleration, strict=True):
            np.testing.assert_array_equal(current, original)

def test_zero_surface_pressure_and_diffusion_agree_with_reference_operators():
    _, params, depths = _fixture()
    eta = jnp.zeros(params.wet_mask.shape)
    geometry = _geometry(params, depths, eta)
    generator = np.random.default_rng(930002)
    rho = jnp.asarray(generator.normal(size=params.wet_mask_z.shape))
    state = _state(rho, eta, params)
    decoded = RHO_0 * (-ALPHA_T * (state.T - params.T_ref))
    pressure = hydrostatic_pressure(decoded, eta, geometry, params)
    np.testing.assert_array_equal(pressure, _compute_hydrostatic_pressure(state, params))
    for actual, expected in zip(coordinate_pressure_gradient(pressure, decoded, geometry, params),
                                _compute_pressure_gradient(state, params), strict=True):
        np.testing.assert_array_equal(actual, expected)
    stencil = make_reference_stencil(depths, params.wet_mask_z)
    concentration = jnp.asarray(generator.normal(size=params.wet_mask_z.shape))
    actual = diffusion_content_rhs(concentration, geometry, stencil, params, 1000., .001)
    expected = params.dz_node * (_horizontal_diffusion_flux(concentration, jnp.full_like(concentration, 1000.), params)
                                 + _d2_dz2_flux(concentration, .001, params))
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-16)

def test_diffusion_work_is_the_negative_physical_face_square_and_never_reads_dry_values():
    _, params, depths = _fixture()
    generator = np.random.default_rng(930003)
    eta = jnp.asarray(generator.uniform(-3., 2., params.wet_mask.shape)) * params.wet_mask
    geometry = _geometry(params, depths, eta)
    stencil = make_reference_stencil(depths, params.wet_mask_z)
    field = jnp.asarray(generator.normal(size=params.wet_mask_z.shape))
    gradients = physical_diffusion_gradients(field, geometry, stencil, params)
    weights = diffusion_weights(geometry, params, 1000., .001)
    response = jax.jit(lambda values: diffusion_content_rhs(values, geometry, stencil, params, 1000., .001))(field)
    area = params.dx_2d * params.dy
    actual_work = float(jnp.sum(area[..., None] * field * response))
    expected_work = -sum(float(jnp.sum(weight * gradient ** 2)) for weight, gradient in zip(weights, gradients, strict=True))
    np.testing.assert_allclose(actual_work, expected_work, rtol=1e-13, atol=0.)
    assert actual_work < 0.
    floor = 64. * np.finfo(float).eps * float(jnp.sum(jnp.abs(area[..., None] * response)))
    assert abs(float(jnp.sum(area[..., None] * response))) <= floor
    for sentinel in (123., -1e6):
        replaced = jnp.where(params.wet_mask_z > 0., field, sentinel)
        np.testing.assert_array_equal(diffusion_content_rhs(replaced, geometry, stencil, params, 1000., .001),
                                      diffusion_content_rhs(field, geometry, stencil, params, 1000., .001))
    resting = 15. + .005 * geometry.node_depth
    horizontal_only = diffusion_content_rhs(resting, geometry, stencil, params, 1000., 0.)
    assert float(jnp.max(jnp.abs(horizontal_only))) < 1e-16

def test_quadratic_density_coordinate_pressure_mms_converges_at_second_order():
    errors = _pressure_coordinate_errors()
    assert all(3.5 < before / after < 4.5 for before, after in zip(errors[:-1], errors[1:], strict=True)), errors

def test_independent_dense_diffusion_and_old_divergence_energy_counterexample():
    _, params, depths = _fixture()
    generator = np.random.default_rng(930004)
    eta = generator.uniform(-3., 2., params.wet_mask.shape) * np.asarray(params.wet_mask)
    geometry = _geometry(params, depths, eta)
    stencil = make_reference_stencil(depths, params.wet_mask_z)
    dense = assemble_diffusion(eta, depths, np.asarray(params.dz_node),
                              np.asarray(params.wet_mask_z), np.asarray(params.dx_2d),
                              float(params.dy), np.asarray(params.cos_lat), 1000., .001)
    field = generator.normal(size=params.wet_mask_z.shape)
    actual = np.asarray(diffusion_content_rhs(jnp.asarray(field), geometry, stencil, params, 1000., .001))[dense.wet]
    expected = dense.stiffness @ field[dense.wet] / dense.area
    floor = 64. * np.finfo(float).eps * np.max(np.abs(dense.stiffness) @ np.abs(field[dense.wet]) / dense.area)
    assert np.max(np.abs(actual - expected)) <= floor
    matrix_floor = 64. * np.finfo(float).eps * np.linalg.norm(dense.stiffness, ord=np.inf)
    assert np.max(np.abs(dense.stiffness.sum(axis=0))) <= matrix_floor
    assert np.max(np.abs(dense.stiffness.sum(axis=1))) <= matrix_floor
    inverse_root_mass = 1. / np.sqrt(dense.mass)
    naive = dense.old_divergence_stiffness * inverse_root_mass[:, None] * inverse_root_mass[None, :]
    symmetric = .5 * (naive + naive.T)
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    eigen_floor = 64. * np.finfo(float).eps * np.linalg.norm(naive, ord=np.inf)
    assert eigenvalues[-1] > 1000. * eigen_floor
    witness = inverse_root_mass * eigenvectors[:, -1]
    naive_work = witness @ dense.old_divergence_stiffness @ witness
    corrected_work = witness @ dense.horizontal_stiffness @ witness
    assert naive_work > 1000. * eigen_floor
    assert corrected_work <= eigen_floor
    witness_field = np.zeros(field.shape)
    witness_field[dense.wet] = witness
    response = np.asarray(diffusion_content_rhs(jnp.asarray(witness_field), geometry, stencil, params, 1000., 0.))[dense.wet]
    np.testing.assert_allclose(np.sum(dense.area * witness * response), corrected_work, rtol=1e-12, atol=eigen_floor)

def test_negative_adjoint_energy_control_does_not_imply_a_maximum_principle():
    _, params, depths = _fixture()
    generator = np.random.default_rng(930004)
    eta = generator.uniform(-3., 2., params.wet_mask.shape) * np.asarray(params.wet_mask)
    dense = assemble_diffusion(eta, depths, np.asarray(params.dz_node),
                              np.asarray(params.wet_mask_z), np.asarray(params.dx_2d),
                              float(params.dy), np.asarray(params.cos_lat), 1000., .001)
    off_diagonal = dense.stiffness.copy()
    np.fill_diagonal(off_diagonal, 0.)
    row, column = np.unravel_index(np.argmin(off_diagonal), off_diagonal.shape)
    matrix_floor = 64. * np.finfo(float).eps * np.linalg.norm(dense.stiffness, ord=np.inf)
    assert off_diagonal[row, column] < -1000. * matrix_floor
    field = np.zeros(params.wet_mask_z.shape)
    field[dense.wet] = np.eye(1, len(dense.mass), column)[0]
    geometry = _geometry(params, depths, eta)
    stencil = make_reference_stencil(depths, params.wet_mask_z)
    rhs = np.asarray(diffusion_content_rhs(jnp.asarray(field), geometry, stencil, params, 1000., .001))[dense.wet]
    assert field[dense.wet][row] == 0.
    assert rhs[row] / dense.thickness[dense.wet][row] < -1000. * matrix_floor / dense.mass[row]

def test_smooth_coordinate_jvp_matches_finite_differences_and_vjp():
    _, params, depths = _fixture(nx=6, ny=4)
    generator = np.random.default_rng(930005)
    eta = jnp.asarray(generator.uniform(-3., 2., params.wet_mask.shape)) * params.wet_mask
    direction = jnp.asarray(generator.normal(size=eta.shape)) * params.wet_mask
    field = jnp.asarray(generator.normal(size=params.wet_mask_z.shape))
    density = jnp.asarray(generator.normal(size=params.wet_mask_z.shape))
    stencil = make_reference_stencil(depths, params.wet_mask_z)

    def response(surface):
        geometry = _geometry(params, depths, surface)
        pressure = hydrostatic_pressure(density, surface, geometry, params)
        force = coordinate_pressure_gradient(pressure, density, geometry, params)
        diffusion = diffusion_content_rhs(field, geometry, stencil, params, 1000., .001)
        return jnp.concatenate((geometry.thickness.ravel(), (1e6 * force[0]).ravel(),
                                (1e6 * force[1]).ravel(), (1e6 * diffusion).ravel()))

    response = jax.jit(response)
    original, tangent = jax.jvp(response, (eta,), (direction,))
    step = 1e-3
    finite_difference = (response(eta + step * direction) - response(eta - step * direction)) / (2. * step)
    relative = np.linalg.norm(np.asarray(tangent - finite_difference)) / np.linalg.norm(tangent)
    assert relative < 2e-7
    cotangent = jnp.asarray(generator.normal(size=original.shape))
    _, pullback = jax.vjp(response, eta)
    left = jnp.vdot(tangent, cotangent)
    right = jnp.vdot(direction, pullback(cotangent)[0])
    floor = 64. * np.finfo(float).eps * (jnp.sum(jnp.abs(tangent * cotangent))
                                       + jnp.sum(jnp.abs(direction * pullback(cotangent)[0])))
    assert abs(float(left - right)) <= float(floor)
