"""Actual substep means and top-face transports, not snapshot-rate proxies."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ocean_solver.audit.schema import NONLINEAR_PROCESS_NAMES
from ocean_solver.audit.stages import _StageRecorder, make_budget_step
from ocean_solver.config.definitions import C_P, RHO_0
from ocean_solver.dynamics.processes import _compute_tracer_tendency
from ocean_solver.dynamics.transport import _advection_scalar, _vertical_transport_iface
from ocean_solver.physics.isopycnal import _isopycnal_closure
from ocean_solver.physics.vertical import _conv_flux_tendency, _convective_mask
from tests.support.fd.horizontal_diffusion import _parameters
from tests.support.fd.nonlinear_budgets import _state


@pytest.mark.parametrize("land", [False, True])
@pytest.mark.parametrize("scan", [False, True])
@pytest.mark.parametrize("substeps", [1, 3])
def test_actual_advection_and_boundary_means_match_independent_substeps(land, scan, substeps):
    _, params, volume = _parameters(65., land=land)
    params = params._replace(kappa_h=0., adv_nsub=substeps, use_scan=scan, fct_adv=True)
    state = _state(params)

    @jax.jit
    def probe(current):
        recorder = _StageRecorder(params)
        _compute_tracer_tendency(current, params, budget=recorder)
        return recorder.result(current, current)

    report = probe(state)
    transport = _vertical_transport_iface(state.u, state.v, params)
    assert float(jnp.min(transport[..., 0])) < 0. < float(jnp.max(transport[..., 0]))
    area = np.asarray(params.dx_2d) * params.dy * np.asarray(params.wet_mask)
    expected_rate = np.zeros(3)
    expected_boundary = np.zeros(3)
    current_temperature, current_salinity = state.T, state.S
    for _ in range(substeps):
        temperature_rate = _advection_scalar(current_temperature, state.u, state.v, transport, params)
        salinity_rate = _advection_scalar(current_salinity, state.u, state.v, transport, params)
        expected_rate[:2] += [RHO_0 * C_P * np.sum(np.asarray(temperature_rate) * volume),
                              RHO_0 / 1000. * np.sum(np.asarray(salinity_rate) * volume)]
        expected_boundary[:2] += [RHO_0 * C_P * np.sum(np.asarray(transport[..., 0] * current_temperature[..., 0]) * area),
                                  RHO_0 / 1000. * np.sum(np.asarray(transport[..., 0] * current_salinity[..., 0]) * area)]
        current_temperature = current_temperature + temperature_rate * (params.dt / substeps)
        current_salinity = current_salinity + salinity_rate * (params.dt / substeps)
    expected_rate *= params.dt / (2. * substeps)
    expected_boundary *= params.dt / (2. * substeps)
    actual = report["nonlinear_process_changes"][NONLINEAR_PROCESS_NAMES.index("advection")]
    np.testing.assert_allclose(actual, expected_rate, rtol=5e-12, atol=1e-3)
    np.testing.assert_allclose(report["advection_boundary_changes"], expected_boundary, rtol=5e-12, atol=1e-3)
    assert np.all(np.abs(report["advection_boundary_residual"]) <= 5e-12 * np.maximum(report["nonlinear_process_scale"], 1.))

def test_nonlinear_processes_reconstruct_actual_update_not_external_sources():
    _, params, volume = _parameters(65., land=True)
    params = params._replace(kappa_h=0., dt=1., fct_adv=True)
    state = _state(params)
    _, report = make_budget_step(params)(state)
    inventory_scale = np.array([RHO_0 * C_P * np.sum(np.abs(np.asarray(state.T)) * volume),
                                RHO_0 / 1000. * np.sum(np.abs(np.asarray(state.S)) * volume), 0.])
    roundoff = 16. * np.finfo(np.asarray(state.T).dtype).eps * inventory_scale
    tolerance = 5e-12 * np.maximum(report["change_scale"] + report["nonlinear_process_scale"], 1.)
    assert np.all(np.abs(report["nonlinear_accounting_residual"]) <= tolerance + roundoff)
    np.testing.assert_array_equal(report["source_inputs"], 0.)
    assert abs(float(report["budget_residual"][0])) > 1.
    np.testing.assert_allclose(report["budget_residual"][:2], report["advection_boundary_changes"][:2], rtol=5e-9)

def test_linearized_surface_inventory_is_separate_from_original_residual():
    _, params, _ = _parameters(65., land=True)
    params = params._replace(kappa_h=0., dt=1.)
    state = _state(params)._replace(eta=jnp.full_like(params.wet_mask, 0.1) * params.wet_mask)
    updated, report = make_budget_step(params)(state)
    area = np.asarray(params.dx_2d) * params.dy * np.asarray(params.wet_mask)
    heat = RHO_0 * C_P * np.sum(area * (np.asarray(updated.eta) * np.asarray(updated.T[..., 0])
                                       - np.asarray(state.eta) * np.asarray(state.T[..., 0])))
    salt = RHO_0 / 1000. * np.sum(area * (np.asarray(updated.eta) * np.asarray(updated.S[..., 0])
                                        - np.asarray(state.eta) * np.asarray(state.S[..., 0])))
    np.testing.assert_allclose(report["surface_displacement_tracer_change"], [heat, salt, 0.], rtol=5e-12, atol=1.)
    np.testing.assert_allclose(report["budget_residual"], report["observed_change"] - np.asarray(report["source_inputs"]).sum(axis=0))

@pytest.mark.parametrize("scan", [False, True])
@pytest.mark.parametrize("process", ["convection", "gm", "redi"])
def test_enabled_closures_record_actual_rates_not_external_sources(scan, process):
    _, params, volume = _parameters(65., land=True)
    params = params._replace(kappa_h=0., dt=1., use_scan=scan, localize_conv=True,
                             kappa_conv=0.01 if process == "convection" else 0.,
                             conv_nsub=3 if process == "convection" else 1,
                             kappa_gm=5. if process == "gm" else 0.,
                             kappa_redi=7. if process == "redi" else 0.)
    state = _state(params)

    @jax.jit
    def probe(current):
        recorder = _StageRecorder(params)
        _compute_tracer_tendency(current, params, budget=recorder)
        return recorder.result(current, current)

    report = probe(state)
    if process == "convection":
        conv_mask, gate = _convective_mask(state, params)
        current_temperature, current_salinity = state.T, state.S
        temperature_rates, salinity_rates = [], []
        for _ in range(params.conv_nsub):
            temperature = _conv_flux_tendency(current_temperature, conv_mask,
                                              params.kappa_conv / params.conv_nsub, params, gate)
            salinity = _conv_flux_tendency(current_salinity, conv_mask,
                                           params.kappa_conv / params.conv_nsub, params, gate)
            temperature_rates.append(np.asarray(temperature))
            salinity_rates.append(np.asarray(salinity))
            current_temperature += temperature * params.dt / params.conv_nsub
            current_salinity += salinity * params.dt / params.conv_nsub
        temperature = np.mean(temperature_rates, axis=0)
        salinity = np.mean(salinity_rates, axis=0)
    else:
        gm_temperature, gm_salinity, redi_temperature, redi_salinity = _isopycnal_closure(state, params)
        temperature, salinity = ((gm_temperature, gm_salinity) if process == "gm"
                                  else (redi_temperature, redi_salinity))
    assert np.max(np.abs(temperature)) > 0.
    expected = [RHO_0 * C_P * params.dt / 2. * np.sum(np.asarray(temperature) * volume),
                RHO_0 / 1000. * params.dt / 2. * np.sum(np.asarray(salinity) * volume), 0.]
    actual = report["nonlinear_process_changes"][NONLINEAR_PROCESS_NAMES.index(process)]
    assert np.all(np.abs(np.asarray(actual) - expected) <= 5e-12 * np.maximum(report["nonlinear_process_scale"], 1.))
    np.testing.assert_array_equal(report["source_inputs"], 0.)
