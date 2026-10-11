"""Known wave truth, declaration corruptions and native-option plumbing controls."""

import copy
import json

import numpy as np
import pytest

from tests.support.standing_wave import fixture
from zhenmode.benchmarks.configuration_scan import contract, rows, validate_contract
from zhenmode.benchmarks.standing_wave import digest
from zhenmode.evaluation.standing_wave import score


def test_matrix_is_finite_unique_and_every_row_rebuilds():
    declared = rows()
    assert len(declared) == 65
    identities = [digest(contract(**r)) for r in declared]
    assert len(set(identities)) == len(declared)
    for r in declared:
        validate_contract(contract(**r))


@pytest.mark.parametrize("damage", ["physics", "threshold", "clock", "options", "extra", "row"])
def test_scan_rejects_undeclared_changes(damage):
    c = contract("thermal-wind", "mom6", "mom-ppm-h3")
    if damage == "physics":
        c["thermal_wind"]["front_width_m"] *= 2
    elif damage == "threshold":
        c["thresholds"]["error"] *= 2
    elif damage == "clock":
        c["dt"] = True
    elif damage == "options":
        c["mom_time_options"]["KV"] = 1
    elif damage == "extra":
        c["scan"]["hidden"] = True
    else:
        c["scan"]["model"] = "zhenmode"
    with pytest.raises(ValueError):
        validate_contract(c)


@pytest.mark.parametrize("setting", ["base", "dt-half", "dt-quarter", "zhen-second-order"])
def test_closed_form_wave_recovered_through_scan_scorer(setting):
    c = contract("standing-wave", "zhenmode", setting)
    _, a = fixture("fine")
    m = a["metadata"]
    m.update(schema=c["schema"], contract_sha256=digest(c), dt_s=c["dt"], steps=int(32000 / c["dt"]))
    m["recorded_numerics"]["resolved_options"] = c["ocean_options"].copy()
    assert score(c, a)["metrics"]["eta_error"] < 1e-14
    bad = copy.deepcopy(a)
    bad["u"][1:] *= -1
    assert "u_error" in score(c, bad)["failed_metrics"]
    bad = copy.deepcopy(a)
    bad["metadata"]["dt_s"] *= 2
    with pytest.raises(ValueError, match="timestep"):
        score(c, bad)
    wrong = contract("standing-wave", "mom6", "base")
    bad = copy.deepcopy(a)
    bad["metadata"]["contract_sha256"] = digest(wrong)
    with pytest.raises(ValueError, match="output model"):
        score(wrong, bad)


def test_vertical_scan_keeps_depth_and_end_half_capacities():
    c = contract("thermal-wind", "zhenmode", "vertical-double")
    assert c["nz"] == 32
    assert sum(c["initial_h_m"]) == pytest.approx(100)
    assert c["initial_h_m"][0] == c["initial_h_m"][-1]
    assert c["initial_h_m"][1] == 2 * c["initial_h_m"][0]
    np.testing.assert_equal(c["thermal_wind"], contract("thermal-wind", "zhenmode", "base")["thermal_wind"])


def test_refined_grid_followup_changes_only_the_external_step():
    from zhenmode.benchmarks.configuration_scan import followup_rows

    assert followup_rows()[0] not in rows()
    before = contract("thermal-wind", "mom6", "grid-double")
    after = contract(**followup_rows()[0])
    assert after["mom_time_options"]["DTBT"] == 12.5
    assert after["mom_time_options"]["SPLIT_RK2B"] is False
    before.pop("scan")
    after.pop("scan")
    after["mom_time_options"]["DTBT"] = 50.0
    assert digest(before) == digest(after)


def test_mom_input_applies_actual_substeps_and_transport(tmp_path, monkeypatch):
    from zhenmode.baselines.mom6 import standing_wave as mom

    c = contract("standing-wave", "mom6", "mom-fast4-rk2b")
    # Deliberate awkward non-default values, checked at the native input boundary.
    c["mom_time_options"]["TRACER_ADVECTION_SCHEME"] = "PPM:H3"
    exe = tmp_path / "MOM6"
    exe.write_bytes(b"manufactured native boundary control")
    config = tmp_path / "configuration.json"
    config.write_text(json.dumps(dict(contract=c, mom_executable=str(exe), mom_source=str(tmp_path))))

    def initial(_, directory):
        directory.mkdir()
        (directory / "standing_wave_initial.nc").write_bytes(b"controlled input")

    monkeypatch.setattr(mom, "prepare_wave_input", initial)
    monkeypatch.setattr(mom.subprocess, "check_output", lambda *a, **k: mom.MOM6)
    observed = []
    monkeypatch.setattr(mom.subprocess, "run", lambda *a, **k: observed.append((tmp_path / "MOM_input").read_text()))
    monkeypatch.chdir(tmp_path)
    mom.integrate(config)
    assert len(observed) == 1
    assert "DTBT = 6.25\n" in observed[0]
    assert "SPLIT_RK2B = True\n" in observed[0]
    assert 'TRACER_ADVECTION_SCHEME = "PPM:H3"\n' in observed[0]
    assert "KV = 0.0\n" in observed[0]


def test_cli_routes_one_row_and_never_hides_other_model(monkeypatch, tmp_path):
    from zhenmode.execution import configuration_scan as scan

    seen = []
    monkeypatch.setattr(scan, "execute", lambda c, *a, **k: seen.append((c, k)))
    scan.run("thermal-wind", "oceananigans", "oa-weno5", tmp_path)
    assert seen[0][1]["models"] == ("oceananigans",)
    assert seen[0][0]["oceananigans_options"]["tracer_advection"] == "WENO5"


@pytest.mark.parametrize("label", ["Centered2", "Centered4", "WENO5"])
def test_native_main_scheme_cannot_be_forged_by_a_boundary_buffer(label):
    from zhenmode.baselines.oceananigans.adapter import check_scan_advection

    descriptions = {"Centered2": "Centered(order=2)", "Centered4": "Centered(order=4)",
                    "WENO5": "WENO{3, Float64, BackendOptimizedDivision}(order=5)\n  buffer: Centered(order=2)\n  advecting velocity: Centered(order=4)"}
    check_scan_advection(dict.fromkeys(("T", "S", "physical_T", "physical_S"), descriptions[label]), label)
    for wrong in descriptions.keys() - {label}:
        with pytest.raises(ValueError, match="main native"):
            check_scan_advection({"T": descriptions[wrong]}, label)
