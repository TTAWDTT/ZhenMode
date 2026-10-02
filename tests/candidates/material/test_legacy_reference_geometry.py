"""Independent M1 gates for an opt-in repair of the original FD core."""

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from config import C_P, G_EARTH, RHO_0, PhysicsConfig
from diagnostics import compute_budget_diagnostics
from grid import nodal_control_thickness
from jax_solver_global import (
    _barotropic_velocity,
    _column_divergence,
    _compute_bt_rho_pgf,
    _compute_momentum_residual,
    _compute_momentum_tendency,
    _compute_pressure_gradient,
    _conv_flux_tendency,
    _d2_dz2_flux,
    _free_surface_step_fd,
    _horizontal_tracer_diffusion,
    _reference_depth_divergence,
    _reference_depth_gradient,
    _surface_heat_weights,
    _vertical_momentum_diffusion,
    make_fd_params,
    make_solver_global,
)
from tests.support.material.reference_geometry import WIDTHS, _fixture, _numpy_column_divergence


def test_dual_widths_close_the_declared_column_not_the_old_proxy():
    np.testing.assert_array_equal(nodal_control_thickness([0., -5., -20., -50.]), WIDTHS)
    np.testing.assert_array_equal(nodal_control_thickness([-50., -100., -200.]), [75., 75., 50.])
    grid, (_, _, _, params, _) = _fixture()
    np.testing.assert_array_equal(np.asarray(params.dz_node).ravel(), WIDTHS)
    np.testing.assert_array_equal(params.H_sw, 50.)
    assert params.dz_surface == 2.5
    legacy = make_fd_params(grid)
    assert np.sum(legacy.dz_node) == 67.5
    assert np.sum(legacy.dz_node) != np.sum(grid.dz)

@pytest.mark.parametrize("nodes", [[0.], [0., 0.], [1., -2.], [0., -np.inf], [0., -2., -1.]])
def test_invalid_node_coordinates_are_rejected(nodes):
    with pytest.raises(ValueError, match="z must contain"):
        nodal_control_thickness(nodes)

def test_staircase_depth_and_bottom_mask_are_explicit():
    grid, (_, _, _, params, _) = _fixture(stairs=True)
    expected_depth = np.sum(grid.wet_mask_3d * WIDTHS, axis=-1)
    np.testing.assert_array_equal(params.H_sw * params.wet_mask, expected_depth)
    np.testing.assert_allclose(np.sum(params.dz_norm, axis=-1), grid.wet_mask, rtol=0., atol=0.)
    assert expected_depth[4, 3] == 12.5
    assert grid.depth[4, 3] == 10.
    assert expected_depth[6, 3] == 2.5
    expected_bottom = np.zeros_like(grid.wet_mask_3d)
    for column_x in range(grid.nx):
        for column_y in range(grid.ny):
            indices = np.flatnonzero(grid.wet_mask_3d[column_x, column_y])
            if indices.size:
                expected_bottom[column_x, column_y, indices[-1]] = 1.
    np.testing.assert_array_equal(params.bottom_mask, expected_bottom)

def test_deep_bathymetry_does_not_silently_add_unrepresented_water():
    grid, _ = _fixture()
    deep = replace(grid, depth=np.full((8, 8), 5000.))
    params = make_fd_params(deep, column_geometry="nodal_dual_v1")
    np.testing.assert_array_equal(np.sum(params.dz_node * params.wet_mask_z, axis=-1), 50.)

def test_column_uniform_mask_is_expanded_before_identifying_the_bottom():
    grid, _ = _fixture()
    params = make_fd_params(replace(grid, wet_mask_3d=np.ones((8, 8, 1))), column_geometry="nodal_dual_v1")
    expected_bottom = np.zeros((8, 8, 4))
    expected_bottom[..., -1] = 1.
    np.testing.assert_array_equal(params.bottom_mask, expected_bottom)
    assert params.wet_mask_z.shape == expected_bottom.shape

@pytest.mark.parametrize("defect", ["gap", "fractional", "surface"])
def test_invalid_wet_column_contract_is_rejected(defect):
    grid, _ = _fixture()
    wet = grid.wet_mask_3d.copy()
    wet[3, 3, 1 if defect == "gap" else 0] = 0.5 if defect == "fractional" else 0.
    with pytest.raises(ValueError, match="binary, contiguous"):
        make_fd_params(replace(grid, wet_mask_3d=wet), column_geometry="nodal_dual_v1")

@pytest.mark.parametrize("stairs", [False, True])
def test_sheared_column_and_uniform_projection_use_actual_common_wet_faces(stairs):
    grid, (_, _, _, params, _) = _fixture(stairs=stairs)
    random = np.random.default_rng(20260929)
    velocity_x, velocity_y = random.normal(size=(2, 8, 8, 4))
    velocity_x[grid.wet_mask_3d == 0.] = 1e8
    velocity_y[grid.wet_mask_3d == 0.] = -1e8
    expected = _numpy_column_divergence(velocity_x, velocity_y, grid)
    actual = np.asarray(_column_divergence(jnp.asarray(velocity_x), jnp.asarray(velocity_y), params))
    np.testing.assert_allclose(actual, expected, rtol=1e-11, atol=1e-18)
    mean_x, mean_y = _barotropic_velocity(jnp.asarray(velocity_x), jnp.asarray(velocity_y), params)
    offset = actual - np.asarray(_reference_depth_divergence(mean_x, mean_y, params))
    correction_x, correction_y = random.normal(size=(2, 8, 8)) * grid.wet_mask
    projected = _numpy_column_divergence(velocity_x + correction_x[..., None] * grid.wet_mask_3d,
                                         velocity_y + correction_y[..., None] * grid.wet_mask_3d, grid)
    rebuilt = np.asarray(_reference_depth_divergence(mean_x + correction_x, mean_y + correction_y, params)) + offset
    np.testing.assert_allclose(rebuilt, projected, rtol=1e-11, atol=1e-18)
    area = grid.dx_2d * grid.dy
    assert abs(np.sum(actual * area)) <= 1e-11 * np.sum(np.abs(actual * area))

@pytest.mark.parametrize("stairs", [False, True])
def test_depth_gradient_is_the_independent_volume_weighted_negative_adjoint(stairs):
    grid, (_, _, _, params, _) = _fixture(stairs=stairs)
    random = np.random.default_rng(2718)
    eta, velocity_x, velocity_y = random.normal(size=(3, 8, 8)) * grid.wet_mask
    repeated_x = np.broadcast_to(velocity_x[..., None], (8, 8, 4))
    repeated_y = np.broadcast_to(velocity_y[..., None], (8, 8, 4))
    divergence = _numpy_column_divergence(repeated_x, repeated_y, grid)
    gradient_x, gradient_y = _reference_depth_gradient(jnp.asarray(eta), params)
    area = grid.dx_2d * grid.dy
    volume = area * np.sum(grid.wet_mask_3d * WIDTHS, axis=-1)
    divergence_work = np.sum(eta * divergence * area)
    gradient_work = np.sum((velocity_x * gradient_x + velocity_y * gradient_y) * volume)
    assert divergence_work == pytest.approx(-float(gradient_work), rel=1e-11)

def test_free_surface_uses_old_actual_sheared_column_transport():
    grid, (_, initialize, _, params, _) = _fixture(stairs=True)
    state = initialize()
    random = np.random.default_rng(3141)
    velocity_x, velocity_y = random.normal(size=(2, 8, 8, 4)) * grid.wet_mask_3d
    mean_x, mean_y = _barotropic_velocity(jnp.asarray(velocity_x), jnp.asarray(velocity_y), params)
    expected_divergence = _numpy_column_divergence(velocity_x, velocity_y, grid)
    offset = expected_divergence - np.asarray(_reference_depth_divergence(mean_x, mean_y, params))
    eta, _, _ = _free_surface_step_fd(state.eta, mean_x, mean_y, params, dt_half=5.,
                                     column_divergence_offset=jnp.asarray(offset))
    np.testing.assert_allclose(eta, -5. * expected_divergence, rtol=1e-11, atol=1e-17)
    wrong = np.asarray(_reference_depth_divergence(mean_x, mean_y, params))
    assert np.linalg.norm(wrong - expected_divergence) > 0.01 * np.linalg.norm(expected_divergence)

def test_density_forcing_is_the_same_nodal_average_as_three_dimensional_momentum():
    grid, (_, initialize, _, params, _) = _fixture(stairs=True)
    state = initialize()
    state = state._replace(T=state.T + jnp.asarray(np.random.default_rng(2718).normal(size=(8, 8, 4))))
    acceleration_x, acceleration_y = _compute_pressure_gradient(state, params)
    actual_x, actual_y = _compute_bt_rho_pgf(state, params)
    depth = np.sum(grid.wet_mask_3d * WIDTHS, axis=-1)
    inverse_depth = 1. / np.where(depth > 0., depth, 1.)
    for actual, acceleration in ((actual_x, acceleration_x), (actual_y, acceleration_y)):
        expected = np.sum(np.asarray(acceleration) * grid.wet_mask_3d * WIDTHS, axis=-1) * inverse_depth
        np.testing.assert_allclose(actual, expected, rtol=1e-11, atol=1e-20)

@pytest.mark.parametrize("split", [False, True])
def test_fast_slow_pressure_split_keeps_eta_shear_at_depth_steps(split):
    grid, (_, initialize, _, params, _) = _fixture(stairs=True, mode_split=split)
    random = np.random.default_rng(1618)
    state = initialize()._replace(eta=jnp.asarray(random.normal(size=(8, 8))) * grid.wet_mask)
    full_x, full_y = _compute_momentum_tendency(state, params)
    residual_x, residual_y = _compute_momentum_residual(state, params)
    rho_x, rho_y = _compute_bt_rho_pgf(state, params)
    eta_x, eta_y = _reference_depth_gradient(state.eta, params)
    for full, residual, rho, eta in ((full_x, residual_x, rho_x, eta_x), (full_y, residual_y, rho_y, eta_y)):
        reconstructed = residual + (rho - G_EARTH * eta)[..., None] * grid.wet_mask_3d
        np.testing.assert_allclose(reconstructed, full, rtol=1e-11, atol=1e-20)
    assert np.max(np.abs(np.asarray(residual_x))) > 1e-9

@pytest.mark.parametrize("operator", ["vertical_tracer", "vertical_momentum", "horizontal", "convection"])
def test_diffusion_and_convection_close_the_same_reference_inventory(operator):
    grid, (_, _, _, params, _) = _fixture(stairs=True)
    random = np.random.default_rng(2718)
    tracer = jnp.asarray(random.normal(size=(8, 8, 4)))
    if operator == "vertical_tracer":
        tendency = _d2_dz2_flux(tracer, 0.01, params)
    elif operator == "vertical_momentum":
        tendency = _vertical_momentum_diffusion(tracer, params._replace(nu_v=0.01))
    elif operator == "horizontal":
        tendency = _horizontal_tracer_diffusion(tracer, params._replace(kappa_h=100.))
    else:
        gate = jnp.asarray(random.integers(0, 2, size=(8, 8, 3)))
        tendency = _conv_flux_tendency(tracer, jnp.ones((8, 8, 1)), 0.01, params, iface_gate=gate)
    volume = grid.dx_2d[..., None] * grid.dy * grid.wet_mask_3d * WIDTHS
    weighted_tendency = np.asarray(tendency) * volume
    assert abs(np.sum(weighted_tendency)) <= 1e-11 * np.sum(np.abs(weighted_tendency))
    assert np.sum(np.asarray(tracer) * weighted_tendency) <= 0.

@pytest.mark.parametrize("mixed_depth", [None, 20., 100.])
def test_surface_heat_uses_actual_dual_widths_even_at_a_shallow_bottom(mixed_depth):
    grid, (_, _, _, params, _) = _fixture(stairs=True, mixed_layer_depth_m=mixed_depth)
    weights = np.asarray(_surface_heat_weights(params))
    np.testing.assert_allclose(np.sum(weights * grid.wet_mask_3d * WIDTHS, axis=-1),
                               grid.wet_mask, rtol=1e-11, atol=1e-15)

@pytest.mark.parametrize("split", [False, True])
def test_full_step_wind_and_prescribed_heat_are_counted_once(split):
    forcing = (np.full((8, 8), 0.05), np.zeros((8, 8)), np.full((8, 8), 100.))
    _, (step, initialize, _, _, _) = _fixture(mode_split=split, forcing=forcing)
    before = initialize()
    after = step(before)
    jax.block_until_ready(after)
    momentum = np.sum(np.asarray(after.u - before.u) * WIDTHS, axis=-1) * RHO_0
    heat = np.sum(np.asarray(after.T - before.T) * WIDTHS, axis=-1) * RHO_0 * C_P
    np.testing.assert_allclose(momentum, 0.05 * 10., rtol=1e-11, atol=1e-13)
    np.testing.assert_allclose(heat, 100. * 10., rtol=1e-11, atol=1e-6)

@pytest.mark.parametrize("wet_nodes", [2, 4])
def test_full_step_linear_bottom_drag_acts_at_the_wet_bottom_without_a_second_bt_drag(wet_nodes):
    physics = replace(PhysicsConfig(), nu_h=0., nu_v=0., nu_bi=0., kappa_h=0., kappa_v=0.,
                      kappa_bi=0., kappa_conv=0., kappa_gm=0., kappa_redi=0., r_bot=0.01)
    _, (step, initialize, _, _, _) = _fixture(physics=physics, wet_nodes=wet_nodes)
    before = initialize()._replace(u=jnp.full((8, 8, 4), 0.1))
    after = step(before)
    profile = np.full(4, 0.1)
    profile[wet_nodes - 1] *= np.exp(-0.1)
    expected = np.broadcast_to(profile, (8, 8, 4))
    np.testing.assert_allclose(after.u, expected, rtol=1e-11, atol=1e-14)

@pytest.mark.parametrize("initial_ice,heat", [(0., -100.), (1e-6, 100.), (1., -100.), (1., 100.)])
def test_full_step_ice_phase_heat_and_brine_use_the_same_reference_cells(initial_ice, heat):
    forcing = (np.zeros((8, 8)), np.zeros((8, 8)), np.full((8, 8), heat))
    _, (step, initialize, _, params, _) = _fixture(forcing=forcing, dynamic_ice=True, mixed_layer_depth_m=20.)
    before = initialize()._replace(T=jnp.full((8, 8, 4), params.ice_freeze_temp_c),
                                   ice=jnp.full((8, 8), initial_ice))
    after = step(before)
    water_heat = np.sum(np.asarray(after.T - before.T) * WIDTHS, axis=-1) * RHO_0 * C_P
    ice_change = np.asarray(after.ice - before.ice)
    expected_heat = heat * params.dt / (1. + initial_ice / params.ice_insulation_scale_m)
    np.testing.assert_allclose(water_heat - 917. * 3.34e5 * ice_change, expected_heat, rtol=1e-11, atol=1e-6)
    salt_change = np.sum(np.asarray(after.S - before.S) * WIDTHS, axis=-1) * RHO_0 / 1000.
    np.testing.assert_allclose(salt_change, 917. * 30. / 1000. * ice_change, rtol=1e-11, atol=1e-10)

def test_snapshot_inventory_cannot_silently_fall_back_to_old_endpoint_weights():
    grid, (_, initialize, _, _, _) = _fixture(stairs=True)
    state = initialize()
    diagnostic = compute_budget_diagnostics(state, grid, column_geometry="nodal_dual_v1")
    volume = grid.dx_2d[..., None] * grid.dy * WIDTHS * grid.wet_mask_3d
    assert diagnostic.total_volume_m3 == pytest.approx(np.sum(volume), rel=1e-11)
    assert diagnostic.heat_content_J == pytest.approx(np.sum(np.asarray(state.T) * volume) * RHO_0 * C_P, rel=1e-11)
    legacy = compute_budget_diagnostics(state, grid)
    assert legacy.total_volume_m3 != pytest.approx(diagnostic.total_volume_m3, rel=0.01)

@pytest.mark.parametrize("option", ["conservative_kv", "localize_conv"])
def test_candidate_rejects_incompatible_old_vertical_operators(option):
    with pytest.raises(ValueError, match="requires conservative_kv"):
        _fixture(**{option: False})

def test_candidate_vertical_diffusion_cfl_uses_control_width_not_node_distance():
    physics = replace(PhysicsConfig(), kappa_v=10.)
    with pytest.raises(ValueError, match="vertical tracer diffusion"):
        _fixture(physics=physics)

def test_default_geometry_is_still_legacy():
    grid, _ = _fixture()
    _, _, _, params, _ = make_solver_global(grid, PhysicsConfig(), 10., return_params=True, polar_cap_rows=0)
    assert params.column_geometry == "legacy"
    assert np.sum(params.dz_node) == 67.5
