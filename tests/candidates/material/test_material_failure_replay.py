"""Failure reproduction is not a repaired real-world integration."""

import sys
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest

from tests.support.paths import REPOSITORY_ROOT

sys.path.insert(0, str(REPOSITORY_ROOT))

from material_top import make_material_top_step
from research.experiments.material_failure_replay.audit import (
    SCHEME,
    diagnose,
    geometry_witness,
    synthetic_fixture,
)


@pytest.fixture(scope="module")
def sequence():
    grid, params, state = synthetic_fixture()
    original = make_material_top_step(params, subcycle_scheme=SCHEME, max_subcycles=128)(state)
    step = make_material_top_step(params, subcycle_scheme=SCHEME, max_subcycles=256)
    lifted = step(state)
    following = step(lifted.state)
    return grid, params, state, original, lifted, following


def test_same_initial_state_capacity_then_negative_top_full_step(sequence):
    _, params, state, original, lifted, following = sequence
    reports = [diagnose(original, state, params, 128), diagnose(lifted, state, params, 256),
               diagnose(following, lifted.state, params, 256)]
    assert [report["diagnostic_blocker"] for report in reports] == [
        "subcycle_capacity", "accepted", "nonpositive_node_with_water_remaining"]
    assert [report["accepted"] for report in reports] == [False, True, False]
    assert 128 < reports[0]["checks"]["nonlinear_required_subcycles"] <= 256
    assert reports[1]["checks"]["nonlinear_active_subcycles"] == reports[0]["checks"]["nonlinear_required_subcycles"]
    assert reports[1]["checks"]["nonlinear_combined_fraction_max"] <= .5
    assert reports[1]["checks"]["local_inventory_roundoff_ratio_max"] <= 1.
    assert reports[2]["attempted_geometry"]["actual_column_at_minimum_m"] > 47.
    assert reports[2]["attempted_geometry"]["minimum_wet_thickness_m"] < 0.
    assert reports[2]["checks"]["velocity_abs_max_m_per_s"] < 1.01
    # Skipped tracer updates create a large secondary implementation residual;
    # keep it visible, but do not misdiagnose it as the initiating obstruction.
    assert "local_inventory_roundoff_ratio_max" in reports[0]["failed_scalar_checks"]
    assert "local_inventory_roundoff_ratio_max" in reports[2]["failed_scalar_checks"]
    for report in (reports[0], reports[2]):
        assert report["rejected_state_rollback_bytes_equal"]
        assert not report["attempt_budget_is_accepted_increment"]


def test_more_capacity_cannot_restore_positive_geometry(sequence):
    _, params, _, _, lifted, _ = sequence
    result = make_material_top_step(params, subcycle_scheme=SCHEME, max_subcycles=2048)(lifted.state)
    report = diagnose(result, lifted.state, params, 2048)
    assert report["capacity_exceeded_stages"] == []
    assert report["diagnostic_blocker"] == "nonpositive_node_with_water_remaining"
    assert not report["accepted"]
    assert report["rejected_state_rollback_bytes_equal"]


def test_diagnosis_does_not_mutate_state_checks_or_budget(sequence):
    _, params, state, original, _, _ = sequence
    values = list(original.state) + list(original.attempted_state) + list(original.checks.values()) + list(original.budget.values())
    before = [np.asarray(value).tobytes() for value in values]
    diagnose(original, state, params, 128)
    assert before == [np.asarray(value).tobytes() for value in values]


@pytest.mark.parametrize("eta", [-2.50508739, -2.50169047])
def test_documented_scalar_geometry_is_not_a_real_state_replay(eta):
    # Historical rounded eta values, NOT original restart/ETOPO/WOA arrays.
    params = SimpleNamespace(wet_mask_z=np.ones((1, 1, 3)),
                             dz_node=np.array([2.5, 10., 737.5]),
                             surface_mask=np.array([1., 0., 0.]))
    witness = geometry_witness(np.array([[eta]]), params)
    assert witness["minimum_wet_thickness_m"] == 2.5 + eta
    assert witness["actual_column_at_minimum_m"] == pytest.approx(750. + eta, abs=1e-12)
    assert witness["nonpositive_wet_nodes"] == 1
    assert witness["nonpositive_columns"] == 0


def test_geometry_excludes_dry_sentinels_and_preserves_column_dryout():
    params = SimpleNamespace(wet_mask_z=np.array([[[0., 0.]], [[1., 1.]]]),
                             dz_node=np.array([2.5, 10.]), surface_mask=np.array([1., 0.]))
    witness = geometry_witness(np.array([[np.nan], [-13.]]), params)
    assert witness["finite"]
    assert witness["minimum_index_xyz"] == [1, 0, 0]
    assert witness["nonpositive_columns"] == 1


@pytest.mark.parametrize("eta,blocker", [(-51., "nonpositive_column"), (np.nan, "nonfinite")])
def test_nonfinite_and_column_failure_precede_capacity_in_diagnosis(sequence, eta, blocker):
    _, params, state, original, _, _ = sequence
    attempted = original.attempted_state._replace(eta=jnp.full_like(state.eta, eta))
    # Corrupt a result only to test the offline classifier, not a solver run.
    report = diagnose(original._replace(attempted_state=attempted), state, params, 128)
    assert report["diagnostic_blocker"] == blocker
    assert not report["accepted"]


def test_missing_checks_cannot_be_silently_treated_as_zero(sequence):
    _, params, state, original, _, _ = sequence
    checks = dict(original.checks)
    del checks["nonlinear_required_subcycles"]
    with pytest.raises(ValueError, match="incomplete"):
        diagnose(original._replace(checks=checks), state, params, 128)
