"""Original native small-grid stock and generic-clock handoff controls."""
import copy
import math

import jax.numpy as jnp
import numpy as np
import pytest

from ocean_solver.candidates.material import solver as material
from ocean_solver.configuration import G_EARTH, RHO_0
from research.experiments.material_top_band.original_native import (
    OriginalNativeAudit,
    capture_native,
    reference_pressure_kick,
)
from tests.support.original_native import (
    assert_identity,
    bound,
    independent_outflow,
    make_original_native_case,
    original_factory,
)


@pytest.fixture(scope="module")
def case():
    return make_original_native_case()


def unchanged(first, second):
    for a, b in zip(first, second, strict=True):
        assert np.asarray(a).dtype == np.asarray(b).dtype
        assert np.asarray(a).shape == np.asarray(b).shape
        assert np.asarray(a).tobytes() == np.asarray(b).tobytes()




def test_true_native_fields_stocks_metric_and_pressure(case):
    state, params, grid = case
    view = capture_native(state, params, grid)
    unchanged(view.state, state)
    expected_h = np.broadcast_to([2.5, 7.5, 12.5, 17.5, 25., 15.], state.u.shape)
    np.testing.assert_array_equal(view.h_ref, expected_h)
    expected_actual = expected_h.copy()
    expected_actual[..., 0] += state.eta
    np.testing.assert_array_equal(view.h_tracer, expected_actual)
    np.testing.assert_array_equal(view.tracer_stocks[..., 0], expected_actual*state.T)
    np.testing.assert_array_equal(view.reference_momentum[..., 0], RHO_0*expected_h*state.u)
    np.testing.assert_array_equal(view.mass, RHO_0*(grid.dx_2d*grid.dy)[..., None]*expected_h)
    rho = RHO_0*(-2e-4*(np.asarray(state.T)-15.)+7.6e-4*(np.asarray(state.S)-35.))
    pressure = np.zeros_like(rho)+RHO_0*G_EARTH*np.asarray(state.eta)[..., None]
    pressure[..., 1:] += G_EARTH*np.cumsum(.5*(rho[..., :-1]+rho[..., 1:])*(-np.diff(grid.z)), axis=-1)
    np.testing.assert_allclose(view.rho_prime, rho, rtol=0., atol=1e-15)
    np.testing.assert_allclose(view.pressure, pressure, rtol=0., atol=2e-12)
    assert not view.h_ref.flags.writeable
    assert view.authority == "FD_point_samples_material_top_mass_lumped_linear_momentum_v1"


def test_reference_pressure_kick_uses_actual_midpoint_and_transpose(case):
    state, params, grid = case
    view = capture_native(state, params, grid)
    kick = reference_pressure_kick(view, params, duration=.01)
    B = independent_outflow(grid)
    p = view.pressure.ravel()
    independently_forced = B.T@p
    scale = np.abs(B).T@np.abs(p)
    assert np.all(np.abs(kick.force.ravel()-independently_forced) <= bound(kick.force.ravel(), scale))
    native_velocity = np.stack((state.u, state.v))
    repeated_mass = np.broadcast_to(view.mass, native_velocity.shape)
    assert np.all(abs(kick.before_velocity-native_velocity) <= bound(kick.before_velocity, native_velocity))
    assert np.all(abs(kick.before_momentum-repeated_mass*native_velocity) <= bound(kick.before_momentum, repeated_mass*native_velocity))
    assert np.all(abs(kick.before_momentum-repeated_mass*kick.before_velocity) <= bound(kick.before_momentum, repeated_mass*kick.before_velocity))
    assert np.all(abs(kick.after_momentum-repeated_mass*kick.after_velocity) <= bound(kick.after_momentum, repeated_mass*kick.after_velocity))
    assert np.all(abs(kick.after_momentum-kick.before_momentum-.01*kick.force) <= bound(kick.before_momentum, kick.after_momentum, .01*kick.force))
    np.testing.assert_array_equal(kick.force, repeated_mass*view.acceleration)
    assert np.max(np.abs(kick.after_momentum-kick.before_momentum)) > 0.
    before, after = kick.before_velocity.ravel(), kick.after_velocity.ravel()
    mass = np.tile(view.mass.ravel(), 2)
    change = math.fsum(.5*mass*(after*after-before*before))
    direct_work = math.fsum(.01*.5*(before+after)*independently_forced)
    scale = math.fsum(.5*mass*(before*before+after*after))
    assert abs(change-direct_work) <= bound(scale, direct_work)
    transport_work = .01*math.fsum(p*(B@(.5*(before+after))))
    assert abs(direct_work-transport_work) <= bound(np.abs(p)@(np.abs(B)@np.abs(.01*.5*(before+after))))
    unchanged(view.state, state)
    assert not kick.physical_moving_pressure_qualified


def test_moving_metric_top_band_and_half_mean_counterexamples(case):
    state, params, grid = case
    view = capture_native(state, params, grid)
    decoded = view.reference_momentum[..., 0]/(RHO_0*view.h_tracer)
    expected = np.asarray(state.u)*view.h_ref/view.h_tracer
    np.testing.assert_allclose(decoded, expected, rtol=4e-16, atol=0.)
    assert abs(2.5/(2.5-.4)-25./21.) < 3e-16
    rates = np.array([1., 0., 0.])-np.array([1./9., 1./3., 5./9.])
    np.testing.assert_allclose(rates, [8./9., -1./3., -5./9.], rtol=0., atol=2e-16)
    assert abs(rates.sum()) < 1e-16 and np.max(abs(rates)) > .8
    native_mean = np.sum(view.h_ref*state.u, axis=-1)/80.
    actual_mean = np.sum(view.h_tracer*state.u, axis=-1)/(80.+state.eta)
    defect = state.eta*(state.u[..., 0]-native_mean)/(80.+state.eta)
    assert np.all(abs(actual_mean-native_mean-defect) <= bound(actual_mean, native_mean))
    np.testing.assert_array_equal(np.linalg.solve([[.75, .25], [.25, .75]], [0., 1.]), [-.5, 1.5])


@pytest.mark.parametrize("fault", ["eta", "inverse_metric", "clock", "grid_metric", "masked", "derived_count"])
def test_invalid_native_binding_rejects_without_changes(case, fault):
    state, params, grid = case
    saved = tuple(np.asarray(value).copy() for value in state)
    copied = copy.deepcopy(grid)
    bad_state, bad_params = state, params
    if fault == "eta":
        bad_state = state._replace(eta=jnp.full_like(state.eta, np.nan))
    elif fault == "inverse_metric":
        bad_params = params._replace(inv_dx=params.inv_dx*1.01)
    elif fault == "clock":
        bad_params = params._replace(dt_bt=.02)
    elif fault == "grid_metric":
        copied.dx_2d[0, 0] *= 1.01
    elif fault == "masked":
        bad_state = state._replace(T=np.ma.array(state.T, mask=np.ones(state.T.shape, bool)))
    else:
        bad_params = params._replace(adv_nsub=2)
    with pytest.raises(ValueError):
        capture_native(bad_state, bad_params, copied)
    unchanged(state, saved)


def test_joint_handoff_refuses_all_six_fields(case):
    state, params, grid = case
    saved = tuple(np.asarray(value).copy() for value in state)
    audit = OriginalNativeAudit(state, params, grid)
    for target in ("joint_moving_stocks", "half_prism_raw_means"):
        with pytest.raises(ValueError, match="incompatible"):
            audit.require_handoff(target)
        unchanged(audit.state, state)
    unchanged(state, saved)
    assert audit.accepted_joint_steps == 0 and audit.last_receipt is None


def test_original_ten_stages_twelve_generic_fast_calls_and_default_identity(case, monkeypatch):
    state, params, grid = case
    default = material._material_step(state, params, "actual_geometry_v2", 16, "joint_heun_v1")
    from ocean_solver.fd import barotropic
    actual_calls = []
    original_fast = barotropic._symmetric_free_surface_step
    def observed_original(*args, **kwargs):
        actual_calls.append(float(args[4]))
        return original_fast(*args, **kwargs)
    monkeypatch.setattr(barotropic, '_symmetric_free_surface_step', observed_original)
    audit = OriginalNativeAudit(state, params, grid)
    receipt = audit.diagnose()
    assert actual_calls == [.01]*12
    assert receipt.original_valid and bool(default.valid)
    unchanged(receipt.original_state, default.state)
    assert len(receipt.fast_calls) == 12
    assert [row["index"] for row in receipt.fast_calls] == list(range(12))
    assert len(receipt.stages) == 10
    assert receipt.stages[2]["name"] == "nonlinear_predictor"
    assert receipt.stages[6]["name"] == "accepted_tracer_replay"
    B = independent_outflow(grid)
    area = grid.dx_2d*grid.dy
    for row in receipt.fast_calls:
        eta_before, eta_after = row["before"][0], row["after"][0]
        fx, fy = row["mean_faces"]
        volume = grid.dy*(fx-np.roll(fx, 1, axis=0))
        incoming = np.roll(fy, 1, axis=1).copy()
        incoming[:, 0] = 0.
        volume += grid.dx_2d/grid.cos_lat[None, :]*(fy-incoming)
        predicted = eta_before-.01*volume/area+row["filter_change"]
        assert np.all(abs(eta_after-predicted) <= bound(eta_before, eta_after, .01*volume/area))
    assert np.max(abs(receipt.original_state.eta-state.eta)) > 0.
    assert np.max(abs(receipt.original_state.T-state.T)) > 0.
    assert np.max(abs(receipt.original_state.u-state.u)) > 0.
    from dataclasses import replace

    from tests.support.original_native import audit_native_receipt
    kick = reference_pressure_kick(audit.view,params)
    gates = audit_native_receipt(audit.view,kick,receipt,grid)
    assert max(gates["identity_roundoff_ratios"].values()) <= 1.
    with pytest.raises(ValueError):
        audit_native_receipt(audit.view,replace(kick,impulse_work=kick.impulse_work+1e6),receipt,grid)
    with pytest.raises(ValueError):
        audit_native_receipt(audit.view,replace(kick,transport_work=kick.transport_work+1e6),receipt,grid)
    broken = dict(receipt.fast_calls[0])
    fx,fy = (value.copy() for value in broken["mean_faces"])
    fx[0,1] += 10.
    broken["mean_faces"] = (fx,fy)
    corrupt = replace(receipt,fast_calls=(broken,*receipt.fast_calls[1:]))
    with pytest.raises(ValueError):
        audit_native_receipt(audit.view,kick,corrupt,grid)
    assert receipt.convection_rhs_abs_max > 0.
    assert receipt.pressure_impulse_abs_max > 0.
    assert B.shape == (192, 384)
    assert audit.accepted_joint_steps == 0
    unchanged(audit.state, state)


@pytest.mark.parametrize("late", [False, True])
def test_observer_failure_full_rollback(case, late):
    state, params, grid = case
    saved = tuple(np.asarray(value).copy() for value in state)
    audit = OriginalNativeAudit(state, params, grid)
    def fail(row):
        if row["index"] == (11 if late else 0):
            raise RuntimeError("injected observer failure")
    with pytest.raises(RuntimeError, match="observer failure"):
        audit.diagnose(fast_observer=fail)
    unchanged(audit.state, state)
    unchanged(state, saved)
    assert audit.accepted_joint_steps == 0 and audit.last_receipt is None


def test_scan_observation_rejects_before_stages(case):
    state, params, grid = case
    saved = tuple(np.asarray(value).copy() for value in state)
    seen = []
    with pytest.raises(ValueError, match="use_scan=False"):
        material._material_step(state, params._replace(use_scan=True), "actual_geometry_v2",
                                16, "joint_heun_v1", stage_observer=lambda *args: seen.append(args),
                                fast_observer=lambda *args: seen.append(args))
    assert not seen
    unchanged(state, saved)

def test_current_mass_only_pressure_reinterpretation_is_detected(case):
    state, params, grid = case
    view = capture_native(state, params, grid)
    B = independent_outflow(grid)
    force = B.T@view.pressure.ravel()
    acceleration = view.acceleration.reshape(2, -1).ravel()
    actual_mass = np.tile((RHO_0*view.area[..., None]*view.h_tracer).ravel(), 2)
    wrongly_forced = actual_mass*acceleration
    scale = np.abs(B).T@abs(view.pressure.ravel())+abs(wrongly_forced)
    blocked = scale == 0.
    assert np.all(force[blocked] == 0.) and np.all(wrongly_forced[blocked] == 0.)
    envelope = bound(scale)
    ratio = np.divide(abs(wrongly_forced-force), envelope, out=np.zeros_like(force), where=envelope > 0.)
    assert np.max(ratio) > 100.


def test_parameter_change_after_capture_refuses_before_observers(case):
    state, params, grid = case
    saved = tuple(np.asarray(value).copy() for value in state)
    audit = OriginalNativeAudit(state, params, grid)
    audit.params = params._replace(tau_x_2d=params.tau_x_2d+.001)
    seen = []
    with pytest.raises(ValueError, match="binding"):
        audit.diagnose(fast_observer=seen.append)
    assert not seen
    unchanged(state, saved)
    unchanged(audit.state, state)
    assert audit.accepted_joint_steps == 0 and audit.last_receipt is None


@pytest.mark.parametrize("duration", [True, 0., -.01, .02, np.nan, np.inf, 1j])
def test_invalid_pressure_probe_duration_refuses(case, duration):
    state, params, grid = case
    saved = tuple(np.asarray(value).copy() for value in state)
    view = capture_native(state, params, grid)
    with pytest.raises(ValueError):
        reference_pressure_kick(view, params, duration=duration)
    unchanged(state, saved)


def test_pressure_view_metric_tampering_refuses(case):
    from dataclasses import replace
    state, params, grid = case
    view = capture_native(state, params, grid)
    with pytest.raises(ValueError, match="binding"):
        reference_pressure_kick(replace(view, mass=view.mass*1.01), params)


@pytest.mark.parametrize("bad_area", [1.01, np.nan])
def test_pressure_view_area_tampering_refuses(case, bad_area):
    from dataclasses import replace
    state, params, grid = case
    view = capture_native(state, params, grid)
    with pytest.raises(ValueError):
        reference_pressure_kick(replace(view, area=view.area*bad_area), params)


def test_success_then_late_failure_retains_complete_diagnostic_receipt(case):
    state, params, grid = case
    saved = tuple(np.asarray(value).copy() for value in state)
    audit = OriginalNativeAudit(state, params, grid)
    previous = audit.diagnose()
    def fail(row):
        if row["index"] == 11:
            raise RuntimeError("late transaction failure")
    with pytest.raises(RuntimeError, match="transaction failure"):
        audit.diagnose(fast_observer=fail)
    assert audit.last_receipt is previous
    assert audit.accepted_joint_steps == 0
    unchanged(audit.state, saved)
    unchanged(state, saved)


@pytest.mark.parametrize("field", ["u", "eta"])
def test_large_finite_native_values_cannot_export_nonfinite_derived_state(case, field):
    state, params, grid = case
    bad = state._replace(**{field:jnp.full_like(getattr(state,field),1e308)})
    with pytest.raises(ValueError):
        capture_native(bad, params, grid)


def test_true_factory_with_inconsistent_spherical_metric_refuses(case):
    state, _, grid = case
    copied = copy.deepcopy(grid)
    copied.dx_2d[:,1] *= 1.01
    _, _, _, rebuilt, _ = original_factory(copied)
    with pytest.raises(ValueError, match="spherical geometry"):
        capture_native(state, rebuilt, copied)


@pytest.mark.parametrize("actual,expected,scale", [(1.,0.,0.),(np.nan,0.,1.),(0.,0.,np.inf)])
def test_frozen_identity_gate_refuses_invalid_or_zero_bound(actual, expected, scale):
    with pytest.raises(ValueError):
        assert_identity(actual,expected,scale)
