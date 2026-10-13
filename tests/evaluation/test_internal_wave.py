"""Independent known truth, frozen declarations, corruptions and representation controls."""

import copy

import numpy as np
import pytest

from tests.support.internal_wave import planted
from zhenmode.benchmarks.internal_wave import (
    MODEL_NAMES,
    SELECTION_SHA256,
    contract,
    initial_native,
    matrix,
    selection,
    validate_contract,
)
from zhenmode.benchmarks.standing_wave import digest
from zhenmode.evaluation.internal_wave import reference, score


def test_frozen_matrix_and_selected_options():
    rows = matrix()
    assert len(rows) == len({digest(c) for c in rows}) == 21
    assert digest(selection()) == SELECTION_SHA256
    for c in rows:
        validate_contract(c)
        old = selection()["models"][c["frozen_model"]]
        assert c["dt"] == old["dt_s"]
        key = {"zhenmode": "ocean_options", "mom6": "mom_time_options", "oceananigans": "oceananigans_options"}[c["frozen_model"]]
        assert c[key] == old["options"]


@pytest.mark.parametrize("c", matrix(), ids=lambda c: f"{c['case']}-{c['control']}-{c['frozen_model']}")
def test_independent_planted_truth_initialization_and_full_oracle(c):
    a = planted(c)
    model = MODEL_NAMES[c["frozen_model"]]
    initial = initial_native(c, model)
    for q in ("eta", "u", "T", "S"):
        np.testing.assert_allclose(initial[q].ravel(), a[q][0].ravel(), rtol=0, atol=3e-13)
    exact = reference(c, a)
    for q in exact:
        np.testing.assert_allclose(exact[q], a[q], rtol=0, atol=3e-13)
    r = score(c, a)
    assert r["engineering_screen_pass"]
    assert max(r["metrics"][q] for q in ("eta_error", "u_error", "v_error")) < 1e-11
    assert r["metrics"]["T_error_C"] < 3e-13


@pytest.mark.parametrize("field", ("dt", "thresholds", "internal_wave", "ocean_options", "selected_index", "selection_sha256", "frozen_model"))
def test_contract_changes_rejected(field):
    c = contract("zhenmode")
    c[field] = False
    with pytest.raises(ValueError):
        validate_contract(c)


@pytest.mark.parametrize("field", ("u", "v"))
def test_reversed_phase_is_measured_as_failed_screen(field):
    c = contract("mom6")
    a = planted(c)
    a[field][1:] *= -1
    assert field + "_error" in score(c, a)["failed_metrics"]


@pytest.mark.parametrize("damage", ("clock", "model", "options", "initial", "missing", "nonfinite", "eta_geometry"))
def test_corrupted_native_output_rejected(damage):
    c = contract("oceananigans")
    a = planted(c)
    if damage == "clock":
        a["time"][1] += 1
    elif damage == "model":
        a["metadata"]["model"] = "MOM6"
    elif damage == "options":
        a["metadata"]["recorded_numerics"]["resolved_options"]["tracer_advection"] = "Centered2"
    elif damage == "initial":
        a["T"][0, 3, 2] += 1e-6
    elif damage == "missing":
        a.pop("h")
    elif damage == "eta_geometry":
        a["eta"][1:] *= -1
    else:
        a["v"][2, 3, 1] = np.nan
    with pytest.raises(ValueError):
        score(c, a)


def test_null_detects_small_drift_relative_to_stratified_initial_state():
    c = contract("zhenmode", control="null")
    a = planted(c)
    assert np.min(a["T"]) > 12
    assert score(c, a)["engineering_screen_pass"]
    a["T"][1:] += 1e-8
    r = score(c, a)
    assert "null_T" in r["failed_metrics"]
    assert "T_error_C" not in r["failed_metrics"]


def test_contract_and_independent_fixture_do_not_alias():
    c = contract("mom6")
    untouched = copy.deepcopy(c)
    a = planted(c)
    a["metadata"]["recorded_numerics"]["resolved_options"]["BEBT"] = 0.2
    assert c == untouched
