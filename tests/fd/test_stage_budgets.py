"""Actual-stage bookkeeping must not change the solution or hide internal sources."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import zhenmode.model.numerics.horizontal as solver_horizontal
import zhenmode.model.state.types as solver_types
import zhenmode.model.timestepping.integration as solver_integration
from tests.support.fd.surface_energy import _setup
from zhenmode.model.audit.schema import SOURCE_NAMES, STAGE_NAMES
from zhenmode.model.audit.stages import make_budget_step
from zhenmode.model.config.definitions import C_P, RHO_0


def _inventory_change(before, after, params):
    area = np.asarray(params.dx_2d, dtype=float) * float(params.dy)
    volume = area[:, :, None] * np.asarray(params.dz_node, dtype=float) * np.asarray(params.wet_mask_z)
    water = RHO_0 * C_P * np.sum((np.asarray(after.T, dtype=float) - np.asarray(before.T, dtype=float)) * volume)
    latent = 917. * 3.34e5 * np.sum((np.asarray(after.ice, dtype=float) - np.asarray(before.ice, dtype=float))
                                 * area * np.asarray(params.wet_mask))
    salt = RHO_0 / 1000. * np.sum((np.asarray(after.S, dtype=float) - np.asarray(before.S, dtype=float)) * volume)
    displacement = np.sum((np.asarray(after.eta, dtype=float) - np.asarray(before.eta, dtype=float))
                          * area * np.asarray(params.wet_mask))
    return np.array([water - latent, salt, displacement])


@pytest.mark.parametrize("split", [False, True])
@pytest.mark.parametrize("scan", [False, True])
@pytest.mark.parametrize("ice", [False, True])
@pytest.mark.parametrize("dtype", [jnp.float64, jnp.float32])
def test_audited_step_is_the_actual_solver_step(split, scan, ice, dtype):
    _, params, state, _ = _setup(depth=20., heat=-100., ice=ice)
    params = params._replace(dt=60., mode_split=split, use_scan=scan,
                             n_subcyc=2, dt_bt=30., nu_nsub=1)
    params = params._replace(**{name: value.astype(dtype) for name, value in params._asdict().items()
                              if isinstance(value, jnp.ndarray) and jnp.issubdtype(value.dtype, jnp.floating)})
    state = solver_types.JaxStateG(*(field.astype(dtype) for field in state))
    normal = jax.jit(lambda current: solver_integration._step_impl(current, params))(state)
    audited, ledger = make_budget_step(params)(state)
    for name in state._fields:
        np.testing.assert_allclose(getattr(audited, name), getattr(normal, name),
                                   rtol=32. * np.finfo(np.dtype(dtype)).eps, atol=1e-12)
    expected = _inventory_change(state, audited, params)
    np.testing.assert_allclose(ledger["observed_change"], expected, rtol=1e-12, atol=1.)
    assert np.all(np.abs(ledger["decomposition_residual"]) <= 1e-12 * np.maximum(ledger["change_scale"], 1.))


def test_prescribed_heat_has_independent_integrated_input():
    _, params, state, _ = _setup(depth=20., heat=100.)
    updated, ledger = make_budget_step(params)(state)
    area = np.sum(np.asarray(params.dx_2d) * params.dy * np.asarray(params.wet_mask))
    expected = 100. * params.dt * area
    inputs = np.asarray(ledger["source_inputs"])
    assert inputs[SOURCE_NAMES.index("prescribed_heat"), 0] == pytest.approx(expected, rel=1e-12)
    np.testing.assert_allclose(_inventory_change(state, updated, params)[0], expected, rtol=1e-10)
    assert abs(float(ledger["budget_residual"][0])) <= 1e-10 * expected


def test_bulk_heat_input_uses_both_actual_rk_stages():
    _, params, state, _ = _setup(heat=0.)
    state = state._replace(T=jnp.full_like(state.T, 10.))
    params = params._replace(T_atm_3d=jnp.full_like(params.T_atm_3d, 12.), lambda_bulk=80.)
    _, ledger = make_budget_step(params)(state)
    area = np.sum(np.asarray(params.dx_2d) * params.dy * np.asarray(params.wet_mask))
    decay_rate = 80. / (RHO_0 * C_P * params.dz_surface)
    expected = 80. * 2. * params.dt * (1. - 0.5 * decay_rate * params.dt) * area
    recorded = ledger["source_inputs"][SOURCE_NAMES.index("bulk_heat"), 0]
    assert float(recorded) == pytest.approx(expected, rel=1e-12)
    assert float(recorded) != pytest.approx(80. * 2. * params.dt * area, rel=1e-5)


def test_surface_temperature_and_salinity_restore_are_counted():
    _, params, state, _ = _setup(heat=0.)
    params = params._replace(coastal_restore_coef_2d=jnp.full_like(params.wet_mask, 1e-5),
                             coastal_restore_T_2d=jnp.full_like(params.wet_mask, 0.),
                             restore_coef_S=2e-5, S_ref_2d=jnp.full_like(params.wet_mask, 36.))
    _, ledger = make_budget_step(params)(state)
    area = np.sum(np.asarray(params.dx_2d) * params.dy * np.asarray(params.wet_mask))
    heat = RHO_0 * C_P * params.dz_surface * 1e-5 * 1.8 * params.dt * (1. - 0.5 * 1e-5 * params.dt) * area
    salt = RHO_0 / 1000. * params.dz_surface * 2e-5 * params.dt * (1. - 0.5 * 2e-5 * params.dt) * area
    assert float(ledger["source_inputs"][SOURCE_NAMES.index("temperature_restore"), 0]) == pytest.approx(heat, rel=1e-12)
    assert float(ledger["source_inputs"][SOURCE_NAMES.index("salinity_restore"), 1]) == pytest.approx(salt, rel=1e-12)


def test_sponge_is_declared_separately_from_diffusion():
    _, params, state, _ = _setup(heat=0.)
    params = params._replace(sponge_rate=jnp.full_like(params.sponge_rate, 1e-5),
                             T_clim_3d=jnp.full_like(state.T, 0.), S_clim_3d=jnp.full_like(state.S, 36.))
    updated, ledger = make_budget_step(params)(state)
    expected = _inventory_change(state, updated, params)
    np.testing.assert_allclose(ledger["source_inputs"][SOURCE_NAMES.index("sponge")], expected, rtol=1e-11, atol=1.)
    np.testing.assert_array_equal(ledger["stage_changes"][STAGE_NAMES.index("linear_diffusion")], 0.)


@pytest.mark.parametrize("initial_ice,heat", [(0., -100.), (1e-6, 100.), (1., -100.), (1., 100.)])
def test_ice_source_closes_enthalpy_without_double_counting(initial_ice, heat):
    _, params, state, _ = _setup(depth=20., heat=heat, ice=True)
    state = state._replace(ice=jnp.full_like(state.ice, initial_ice))
    updated, ledger = make_budget_step(params)(state)
    area = np.sum(np.asarray(params.dx_2d) * params.dy * np.asarray(params.wet_mask))
    expected_heat = heat * params.dt * area / (1. + initial_ice / params.ice_insulation_scale_m)
    expected_salt = 917. * 30. / 1000. * np.sum(np.asarray(updated.ice - state.ice) * np.asarray(params.dx_2d) * params.dy)
    assert float(ledger["source_inputs"][SOURCE_NAMES.index("ice_atmosphere_heat"), 0]) == pytest.approx(expected_heat, rel=1e-12)
    assert float(ledger["source_inputs"][SOURCE_NAMES.index("ice_brine"), 1]) == pytest.approx(expected_salt, rel=1e-10)
    assert float(ledger["source_inputs"][SOURCE_NAMES.index("prescribed_heat"), 0]) == 0.
    assert abs(float(ledger["budget_residual"][0])) <= 1e-8 * abs(expected_heat)
    assert abs(float(ledger["budget_residual"][1])) <= 1e-7 * abs(expected_salt)


def test_internal_diffusion_source_is_detected_not_declared(monkeypatch):
    _, params, state, _ = _setup(heat=0.)
    original = solver_horizontal._horizontal_tracer_diffusion
    injected_rate = 1e-6
    import zhenmode.model.dynamics.processes as processes
    monkeypatch.setattr(processes, "_horizontal_tracer_diffusion",
                        lambda tracer, configured: original(tracer, configured) + injected_rate * configured.wet_mask_z)
    _, ledger = make_budget_step(params)(state)
    volume = np.asarray(params.dx_2d)[:, :, None] * params.dy * np.asarray(params.dz_node) * np.asarray(params.wet_mask_z)
    expected = injected_rate * params.dt * RHO_0 * C_P * volume.sum()
    assert float(ledger["budget_residual"][0]) == pytest.approx(expected, rel=1e-10)
    np.testing.assert_array_equal(ledger["source_inputs"], 0.)


def test_runtime_heat_and_air_targets_do_not_use_old_closure():
    _, params, state, _ = _setup(heat=0.)
    advance = make_budget_step(params)
    _, first = advance(state, forcing=(params.tau_x_2d, params.tau_y_2d, jnp.full_like(params.Q_heat_2d, 100.)))
    _, second = advance(state, forcing=(params.tau_x_2d, params.tau_y_2d, jnp.full_like(params.Q_heat_2d, 200.)))
    np.testing.assert_allclose(second["source_inputs"], 2. * first["source_inputs"], rtol=1e-12)
    params = params._replace(lambda_bulk=80.)
    advance = make_budget_step(params)
    _, first = advance(state, atmosphere=jnp.full_like(params.T_atm_3d, 0.))
    _, second = advance(state, atmosphere=jnp.full_like(params.T_atm_3d, -1.8))
    assert float(first["source_inputs"][SOURCE_NAMES.index("bulk_heat"), 0]) > 0.
    assert float(second["source_inputs"][SOURCE_NAMES.index("bulk_heat"), 0]) == 0.


def test_subquantum_float32_heat_is_a_nonzero_residual():
    _, params, state, _ = _setup(heat=1e-6)
    state = solver_types.JaxStateG(*(field.astype(jnp.float32) for field in state))
    params = params._replace(**{name: value.astype(jnp.float32) for name, value in params._asdict().items()
                              if isinstance(value, jnp.ndarray) and jnp.issubdtype(value.dtype, jnp.floating)})
    updated, ledger = make_budget_step(params)(state)
    np.testing.assert_array_equal(updated.T, state.T)
    assert float(ledger["source_inputs"][SOURCE_NAMES.index("prescribed_heat"), 0]) > 0.
    assert float(ledger["budget_residual"][0]) < 0.
