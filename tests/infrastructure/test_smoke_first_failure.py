"""Rejected smoke steps cannot advance accepted time, state or budgets."""

import sys

import jax.numpy as jnp
import numpy as np
import pytest

from tests.support.paths import REPOSITORY_ROOT

sys.path.insert(0, str(REPOSITORY_ROOT / "scripts"))

from verify_debug_integration import load_initial_fixture, make_monitored_advance

from jax_solver_global import JaxStateG
from stage_budgets import empty_budget


def _state():
    zero = jnp.zeros((2, 2, 2))
    return JaxStateG(zero, zero, zero + 15., zero + 35., jnp.zeros((2, 2)), jnp.zeros((2, 2)))


@pytest.mark.parametrize("count", [3, 6, 30])
def test_first_velocity_failure_stops_and_discards_its_state_and_budget(count):
    def step(state):
        ledger = empty_budget()
        ledger["budget_residual"] = jnp.array([1., 2., 3.])
        return state._replace(u=state.u + 4.), ledger

    actual = make_monitored_advance(step, empty_budget(), audited=True)(_state(), count)
    state, peak, _, finite, totals, accepted, attempted, failure, rejected, rejected_ledger = actual
    np.testing.assert_array_equal(state.u, 8.)
    np.testing.assert_array_equal(rejected.u, 12.)
    np.testing.assert_array_equal(totals["budget_residual"], [2., 4., 6.])
    np.testing.assert_array_equal(rejected_ledger["budget_residual"], [1., 2., 3.])
    assert int(accepted) == 2 and int(attempted) == 3 and int(failure) == 3
    assert float(peak) == 12. and bool(finite)


@pytest.mark.parametrize("reason", [1, 2, 4])
def test_nonfinite_state_ledger_and_eta_fail_on_first_attempt(reason):
    def step(state):
        ledger = empty_budget()
        if reason == 1:
            state = state._replace(T=state.T.at[0, 0, 0].set(jnp.nan))
        elif reason == 2:
            ledger["budget_residual"] = jnp.array([jnp.nan, 0., 0.])
        else:
            state = state._replace(eta=state.eta + 15.)
        return state, ledger

    original = _state()
    actual = make_monitored_advance(step, empty_budget(), audited=True)(original, 8)
    for before, after in zip(original, actual[0], strict=True):
        np.testing.assert_array_equal(before, after)
    assert int(actual[5]) == 0 and int(actual[6]) == 1 and int(actual[7]) == reason
    for values in actual[4].values():
        np.testing.assert_array_equal(values, 0.)


@pytest.mark.parametrize("initial,code", [("u", 3), ("eta", 4), ("T", 1)])
def test_invalid_incoming_state_is_not_stepped(initial, code):
    original = _state()
    if initial == "u":
        original = original._replace(u=original.u + 10.)
    elif initial == "eta":
        original = original._replace(eta=original.eta + 15.)
    else:
        original = original._replace(T=original.T.at[0, 0, 0].set(jnp.inf))
    actual = make_monitored_advance(lambda state: state._replace(T=state.T + 100.), empty_budget())(original, 9)
    assert int(actual[5]) == int(actual[6]) == 0 and int(actual[7]) == code
    np.testing.assert_array_equal(actual[0].T, original.T)


def test_batch_boundary_does_not_hide_first_failure():
    advance = make_monitored_advance(lambda state: state._replace(u=state.u + 4.), empty_budget())
    first = advance(_state(), 2)
    assert int(first[5]) == 2 and int(first[7]) == 0
    second = advance(first[0], 9)
    assert int(second[5]) == 0 and int(second[6]) == 1 and int(second[7]) == 3
    np.testing.assert_array_equal(second[0].u, first[0].u)


def test_cumulative_budget_overflow_is_not_accepted():
    def step(state):
        ledger = empty_budget()
        ledger["budget_residual"] = jnp.array([1e308, 0., 0.])
        return state, ledger

    actual = make_monitored_advance(step, empty_budget(), audited=True)(_state(), 4)
    assert int(actual[5]) == 1 and int(actual[6]) == 2 and int(actual[7]) == 2
    assert np.isfinite(np.asarray(actual[4]["budget_residual"])).all()
    assert np.isfinite(np.asarray(actual[9]["budget_residual"])).all()


@pytest.mark.parametrize("field", ["projection_relative_residual_max", "transport_consistency_max"])
def test_invalid_negative_infinite_metric_cannot_be_hidden_by_maximum_accumulation(field):
    def step(state):
        ledger = empty_budget()
        ledger[field] = jnp.full_like(ledger[field], -jnp.inf)
        return state, ledger

    actual = make_monitored_advance(step, empty_budget(), audited=True)(_state(), 3)
    assert int(actual[5]) == 0 and int(actual[6]) == 1 and int(actual[7]) == 2
    assert np.isfinite(np.asarray(actual[4][field])).all()


def test_healthy_actual_candidate_matches_unmonitored_steps_and_ledgers():
    from stage_budgets import accumulate_budget, make_budget_step
    from tests.support.material.process_time import _candidate

    _, (_, initialize, _, params, _) = _candidate(use_scan=True)
    step = make_budget_step(params)
    initial = initialize()
    expected, totals = initial, empty_budget()
    for _ in range(3):
        expected, ledger = step(expected)
        totals = accumulate_budget(totals, ledger)
    actual = make_monitored_advance(step, empty_budget(), audited=True)(initial, 3)
    assert int(actual[5]) == int(actual[6]) == 3 and int(actual[7]) == 0
    for before, after in zip(expected, actual[0], strict=True):
        np.testing.assert_array_equal(before, after)
    for name in totals:
        np.testing.assert_array_equal(totals[name], actual[4][name])


@pytest.mark.parametrize("metric,reason", [(0, 5), (2, 6)])
def test_transport_limits_stop_at_first_bad_actual_ledger_without_altering_it(metric, reason):
    def step(state):
        ledger = empty_budget()
        ledger["transport_consistency_max"] = ledger["transport_consistency_max"].at[metric].set(
            (state.T[0, 0, 0] - 15.) * 1e-6)
        ledger["observed_change"] = jnp.array([1., 0., 0.])
        return state._replace(T=state.T + 1.), ledger

    actual = make_monitored_advance(step, empty_budget(), audited=True,
                                    transport_tolerances=(1e-6, 1e-6))(_state(), 30)
    assert int(actual[5]) == 2 and int(actual[6]) == 3 and int(actual[7]) == reason
    np.testing.assert_array_equal(actual[0].T, 17.)
    np.testing.assert_array_equal(actual[8].T, 18.)
    assert float(actual[4]["transport_consistency_max"][metric]) == 1e-6
    assert float(actual[9]["transport_consistency_max"][metric]) == 2e-6
    np.testing.assert_array_equal(actual[4]["observed_change"], [2., 0., 0.])


@pytest.mark.parametrize("limits,audited", [((1e-6, 1e-6), False), ((0., 1.), True),
                                          ((1., np.nan), True), ((1., np.inf), True), ((1.,), True)])
def test_transport_limits_reject_unaudited_or_invalid_contracts(limits, audited):
    with pytest.raises(ValueError, match="transport_tolerances"):
        make_monitored_advance(lambda state: state, empty_budget(), audited=audited,
                               transport_tolerances=limits)


@pytest.mark.parametrize("fault", [None, "lon", "wet_mask_z", "T_initial", "S_initial"])
def test_real_initial_fixture_fails_closed_on_wrong_geometry_shape_or_nan(tmp_path, fault):
    from tests.support.material.process_time import _candidate

    grid, _ = _candidate()
    values = {"lon": grid.lon, "lat": grid.lat, "z": grid.z, "wet_mask_z": grid.wet_mask_3d,
              "T_initial": np.full((8, 8, 4), 15.), "S_initial": np.full((8, 8, 4), 35.)}
    if fault in {"lon", "wet_mask_z"}:
        values[fault] = np.asarray(values[fault]) + 1.
    elif fault == "T_initial":
        values[fault] = np.full((8, 8), 15.)
    elif fault == "S_initial":
        values[fault][0, 0, 0] = np.nan
    path = tmp_path / "initial.npz"
    np.savez(path, **values)
    if fault is not None:
        with pytest.raises(ValueError, match=fault):
            load_initial_fixture(path, grid)
    else:
        temperature, salinity = load_initial_fixture(path, grid)
        np.testing.assert_array_equal(temperature, values["T_initial"])
        np.testing.assert_array_equal(salinity, values["S_initial"])
