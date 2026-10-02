"""Restricted full-stage seam, not moving-geometry or historical qualification."""

from tests.support.paths import REPOSITORY_ROOT
import importlib.util
import json
from dataclasses import replace
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest
from tests.support.material.reference_geometry import _fixture

from config import C_P, RHO_0, PhysicsConfig

PATH = REPOSITORY_ROOT / 'research/experiments/material_top_band/integration.py'
spec = importlib.util.spec_from_file_location('material_top_band_integration', PATH)
integration = importlib.util.module_from_spec(spec)
spec.loader.exec_module(integration)


def configured():
    physics = replace(PhysicsConfig(), nu_h=2e6, nu_v=1e-4, nu_bi=0.,
                      kappa_h=100., kappa_v=1e-6, kappa_bi=2e14, kappa_conv=.01,
                      kappa_gm=0., kappa_redi=0., r_bot=.001)
    grid, (_, initialize, _, params, _) = _fixture(
        physics=physics, process_time_scheme='symmetric_fast_v3',
        match_barotropic_transport=True, fct_adv=True, use_scan=True,
        dt_bt=10. / 12)
    state = initialize()
    profile = jnp.array([15., 14., 12., 8.], dtype=jnp.float64)
    state = state._replace(T=jnp.broadcast_to(profile, state.T.shape),
                           S=jnp.full_like(state.S, 35.),
                           u=jnp.zeros_like(state.u), v=jnp.zeros_like(state.v),
                           eta=jnp.zeros_like(state.eta), ice=jnp.zeros_like(state.ice))
    params = params._replace(adv_nsub=2, conv_nsub=2, nu_nsub=2,
                             lambda_bulk=80., T_atm_3d=jnp.full_like(params.T_atm_3d, 17.),
                             tau_x_2d=jnp.zeros_like(params.tau_x_2d),
                             tau_y_2d=jnp.zeros_like(params.tau_y_2d),
                             Q_heat_2d=jnp.zeros_like(params.Q_heat_2d))
    return grid, params, state


def test_complete_declared_stages_execute_with_active_physics():
    grid, params, state = configured()
    before = tuple(np.asarray(value).tobytes() for value in state)
    result = integration.advance(state, params, grid)
    assert result['accepted'], result['report']
    report = result['report']
    assert report['executed_stages'] == list(integration.STAGES)
    assert report['fast_subcycles'] == 12
    assert report['all_declared_stages_executed']
    assert not report['moving_geometry_supported'] and not report['qualification_passed']
    assert params.kappa_bi > 0 and params.fct_adv and params.lambda_bulk == 80
    assert np.max(abs(np.asarray(result['state'].T) - np.asarray(state.T))) > 1e-6
    assert report['source_heat_J'] > 0
    assert report['cross_band_transport_max_m_s'] == 0
    assert report['pressure_work_J'] == 0
    np.testing.assert_array_equal(result['state'].eta, state.eta)
    assert tuple(np.asarray(value).tobytes() for value in state) == before
    area = np.asarray(params.dx_2d) * params.dy
    change = np.sum(area[..., None] * np.asarray(params.dz_node)
                    * (np.asarray(result['state'].T) - np.asarray(state.T))) * RHO_0 * C_P
    assert abs(change - report['source_heat_J']) <= report['heat_roundoff_bound_J']


def test_capacity_failure_rejects_before_any_stage(monkeypatch):
    grid, params, state = configured()
    params = params._replace(kappa_bi=1e30)
    def unexpected(*args, **kwargs):
        pytest.fail('unsupported capacity entered a numerical stage')
    monkeypatch.setattr(integration.material, '_material_step', unexpected)
    result = integration.advance(state, params, grid, max_subcycles=1)
    assert not result['accepted']
    assert 'original_linear_capacity' in result['report']['missing_contracts']
    assert result['report']['executed_stages'] == []


def test_finite_coefficient_derived_nonfinite_plan_structurally_rejects(monkeypatch):
    grid, params, state = configured()
    params = params._replace(kappa_bi=1e308)
    assert np.isfinite(params.kappa_bi)
    def unexpected(*args, **kwargs):
        pytest.fail('nonfinite derived capacity entered a numerical stage')
    monkeypatch.setattr(integration.material, '_material_step', unexpected)
    capability = integration.coverage(state, params, grid)
    assert 'nonfinite_original_linear_plan' in capability['missing_contracts']
    assert capability['preflight_required_subcycles']['linear'] is None
    json.dumps(capability, allow_nan=False)
    result = integration.advance(state, params, grid)
    assert not result['accepted'] and result['report']['executed_stages'] == []
    assert 'nonfinite_original_linear_plan' in result['report']['missing_contracts']
    assert result['band'] is None
    for actual, expected in zip(result['state'], state, strict=True):
        assert actual.shape == expected.shape and actual.dtype == expected.dtype
        assert np.asarray(actual).tobytes() == np.asarray(expected).tobytes()


@pytest.mark.parametrize('mutation', ['eta', 'velocity', 'horizontal_density', 'wind', 'coast'])
def test_unadapted_contracts_reject_before_any_original_stage(mutation, monkeypatch):
    grid, params, state = configured()
    if mutation == 'eta':
        state = state._replace(eta=jnp.full_like(state.eta, -2.49))
    elif mutation == 'velocity':
        state = state._replace(u=state.u.at[2, 2, 0].set(.1))
    elif mutation == 'horizontal_density':
        state = state._replace(T=state.T.at[2, 2, 0].set(16.))
    elif mutation == 'wind':
        params = params._replace(tau_x_2d=jnp.full_like(params.tau_x_2d, .01))
    else:
        grid = replace(grid, wet_mask_3d=grid.wet_mask_3d.copy())
        grid.wet_mask_3d[2, 2] = 0
    def unexpected(*args, **kwargs):
        pytest.fail('unadapted contract entered the original numerical stages')
    monkeypatch.setattr(integration.material, '_material_step', unexpected)
    result = integration.advance(state, params, grid)
    assert not result['accepted']
    assert result['report']['executed_stages'] == []
    assert result['report']['missing_contracts']
    for actual, original in zip(result['state'], state, strict=True):
        assert np.asarray(actual).tobytes() == np.asarray(original).tobytes()


def test_injected_midstage_failure_rolls_back_complete_state(monkeypatch):
    grid, params, state = configured()
    original = integration.material._barotropic_subcycle_transport
    def damaged(*args, **kwargs):
        value, faces, change = original(*args, **kwargs)
        return value._replace(eta=value.eta.at[0, 0].set(.01)), faces, change
    monkeypatch.setattr(integration.material, '_barotropic_subcycle_transport', damaged)
    result = integration.advance(state, params, grid)
    assert not result['accepted']
    assert not result['report']['all_declared_stages_executed']
    for actual, original_field in zip(result['state'], state, strict=True):
        assert np.asarray(actual).tobytes() == np.asarray(original_field).tobytes()


@pytest.mark.parametrize('field', ['inv_dx', 'dz_norm', 'f'])
def test_consumed_derived_metric_mismatch_rejects_before_stage(field):
    grid, params, state = configured()
    params = params._replace(**{field: getattr(params, field) + .001})
    result = integration.advance(state, params, grid)
    assert not result['accepted'] and result['report']['executed_stages'] == []
    assert 'derived_metric_binding:' + field in result['report']['missing_contracts']


@pytest.mark.parametrize('field', ['eta', 'kappa_h', 'dx_2d'])
def test_unknown_masked_inputs_do_not_become_supported(field):
    grid, params, state = configured()
    if field == 'eta':
        state = state._replace(eta=np.ma.array(np.asarray(state.eta), mask=True))
    else:
        params = params._replace(**{field: np.ma.array(np.asarray(getattr(params, field)), mask=True)})
    result = integration.advance(state, params, grid)
    assert not result['accepted'] and result['report']['executed_stages'] == []
    assert 'masked_unknown:' + field in result['report']['missing_contracts']
    if field == 'eta':
        np.testing.assert_array_equal(result['state'].eta.mask, state.eta.mask)


def test_observer_does_not_change_original_numerical_result():
    grid, params, state = configured()
    adapted = integration.advance(state, params, grid)
    original = integration.material._material_step(state, params, 'actual_geometry_v2', 256, 'joint_heun_v1')
    assert adapted['accepted'] and bool(original.valid)
    for actual, expected in zip(adapted['state'], original.state, strict=True):
        assert np.asarray(actual).tobytes() == np.asarray(expected).tobytes()


def test_masked_grid_rotation_rejects_unknown_binding():
    grid, params, state = configured()
    grid = replace(grid, f=np.ma.array(grid.f, mask=True))
    result = integration.advance(state, params, grid)
    assert not result['accepted'] and result['report']['executed_stages'] == []
    assert 'masked_grid_unknown:f' in result['report']['missing_contracts']


@pytest.mark.parametrize('field', ['T_atm_3d', 'Q_heat_2d', 'dealias_lon_mask'])
def test_bad_parameter_array_shape_rejects_before_stages(field):
    grid, params, state = configured()
    params = params._replace(**{field: jnp.ones((1,), dtype=jnp.float64)})
    result = integration.advance(state, params, grid)
    assert not result['accepted'] and result['report']['executed_stages'] == []
    assert result['report']['rejection_reason']


def test_nonfinite_scalar_reference_rejects_before_stages():
    grid, params, state = configured()
    params = params._replace(T_ref=float('nan'))
    result = integration.advance(state, params, grid)
    assert not result['accepted'] and result['report']['executed_stages'] == []
    assert 'nonfinite_input:T_ref' in result['report']['missing_contracts']


def test_tampered_production_heat_budget_cannot_supply_independent_inventory(monkeypatch):
    grid, params, state = configured()
    original = integration.material._material_step
    def damaged(*args, **kwargs):
        result = original(*args, **kwargs)
        # Forge mutually matching production budget fields while returned T is unchanged.
        budget = dict(result.budget)
        budget['source_inputs'] = result.budget['source_inputs'].at[0, 0].add(1e20)
        budget['observed_change'] = result.budget['observed_change'].at[0].add(1e20)
        return result._replace(budget=budget)
    monkeypatch.setattr(integration.material, '_material_step', damaged)
    result = integration.advance(state, params, grid)
    assert not result['accepted']
    assert 'independent full-step heat/source closure' in result['report']['rejection_reason']


@pytest.mark.parametrize('defect', ['nan_heat', 'nan_salt', 'inf', 'shape'])
def test_invalid_source_budget_rejects_nan_inf_and_shape(defect, monkeypatch):
    grid, params, state = configured()
    original = integration.material._material_step
    def damaged(*args, **kwargs):
        result = original(*args, **kwargs)
        budget = dict(result.budget)
        sources = result.budget['source_inputs']
        if defect == 'shape':
            sources = sources[:1]
        else:
            slot = 1 if defect == 'nan_salt' else 0
            sources = sources.at[0, slot].set(np.inf if defect == 'inf' else np.nan)
        budget['source_inputs'] = sources
        return result._replace(budget=budget)
    monkeypatch.setattr(integration.material, '_material_step', damaged)
    result = integration.advance(state, params, grid)
    assert not result['accepted']
    assert 'invalid complete-step source budget' in result['report']['rejection_reason']
    for actual, expected in zip(result['state'], state, strict=True):
        assert np.asarray(actual).tobytes() == np.asarray(expected).tobytes()
