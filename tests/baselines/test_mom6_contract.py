"""Baseline adapter contracts; no MOM6 build or integration runs in pytest."""
import json
import os

import netCDF4
import numpy as np
import pytest

from zhenmode.baselines.mom6 import adapter as mom6
from zhenmode.baselines.mom6.adapter import (
    CASE_ID,
    convert,
    evaluate,
    native_parameters,
    prepare_wave_input,
    run,
)
from zhenmode.provenance.sources import sha256_file as file_digest


def test_installed_native_definitions_are_used_and_hashed():
    pins = mom6.RESOURCE_ROOT / "pins.json"
    case = mom6.RESOURCE_ROOT / "cases/tc1.json"
    assert mom6.PINS == {name: json.loads(pins.read_text())[name] for name in ("MOM6", "FMS", "CVMix", "GSW")}
    spec = json.loads(case.read_text(encoding="utf-8"))
    assert mom6.CASE_ID == spec["id"]
    assert mom6.INPUT_FILES == tuple(spec["input_files"])
    identity = mom6.adapter_identity()
    assert identity["baselines/mom6/pins.json"] == file_digest(pins)
    assert identity["baselines/mom6/cases/tc1.json"] == file_digest(case)


def test_adapter_identity_detects_changed_packaged_definition(tmp_path, monkeypatch):
    module = tmp_path / "zhenmode/baselines/mom6/adapter.py"
    module.parent.mkdir(parents=True)
    module.write_text("# independent adapter fixture\n")
    pins = module.parent / "pins.json"
    pins.write_text('{"MOM6": "original"}')
    monkeypatch.setattr(mom6, "__file__", str(module))
    before = mom6.adapter_identity()
    pins.write_text('{"MOM6": "changed"}')
    assert before != mom6.adapter_identity()


def native_fixture(tmp_path):
    manifest = {"run_id": "fixture", "case_id": CASE_ID, "execution_status": "completed",
                "expected_days": 0.25, "cost": {"status": "not_recorded"},
                "source_identity": {}, "executable": {}, "inputs": {}}
    (tmp_path/"run.json").write_text(json.dumps(manifest))
    with netCDF4.Dataset(tmp_path/"ocean.stats.nc", "w") as ds:
        ds.createDimension("Time", 3)
        for name in ("Time", "En", "Mass", "Heat", "Salt", "max_CFL_trans", "max_CFL_lin", "Ntrunc"):
            variable = ds.createVariable(name, "f8", ("Time",))
            variable.units = ("days" if name == "Time" else "Joules" if name in {"En", "Heat"}
                              else "kg" if name in {"Mass", "Salt"} else "Nondim")
            variable[:] = [0, .125, .25] if name == "Time" else (0 if name == "Ntrunc" else 0.1)
    manifest["native_output_receipt"] = {"ocean.stats.nc": file_digest(tmp_path/"ocean.stats.nc")}
    (tmp_path/"run.json").write_text(json.dumps(manifest))
    return manifest


def test_tc1_convert_report_and_tamper_rejection(tmp_path):
    native_fixture(tmp_path)
    convert(tmp_path)
    report = evaluate(tmp_path)
    assert report["acceptance"]["status"] == "passed"
    assert report["effect"]["rmse"] is None
    assert report["comparability"] == "not_comparable_to_global_case"
    assert report["industrial_qualified"] is False
    with pytest.raises(FileExistsError):
        convert(tmp_path)


def test_tc1_missing_native_stats_units_rejected(tmp_path):
    manifest = native_fixture(tmp_path)
    with netCDF4.Dataset(tmp_path/"ocean.stats.nc", "a") as ds:
        ds["Time"].units = "seconds"
    manifest["native_output_receipt"]["ocean.stats.nc"] = file_digest(tmp_path/"ocean.stats.nc")
    (tmp_path/"run.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="time units"):
        convert(tmp_path)


def test_conversion_hash_checked_independently_of_arrays(tmp_path):
    native_fixture(tmp_path)
    convert(tmp_path)
    with np.load(tmp_path/"native_diagnostics.npz") as values:
        arrays = {name: values[name] for name in values.files}
    arrays["Mass"] = arrays["Mass"] * 2
    np.savez(tmp_path/"native_diagnostics.npz", **arrays)
    with pytest.raises(ValueError, match="changed"):
        evaluate(tmp_path)


def test_failed_run_cannot_convert(tmp_path):
    manifest = native_fixture(tmp_path)
    manifest["execution_status"] = "failed"
    (tmp_path/"run.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="completed"):
        convert(tmp_path)


def test_native_output_cannot_be_replaced_after_completed_launch(tmp_path):
    native_fixture(tmp_path)
    with netCDF4.Dataset(tmp_path/"ocean.stats.nc", "a") as ds:
        ds["Mass"][:] = 2
    with pytest.raises(ValueError, match="launch output changed"):
        convert(tmp_path)


def test_existing_binary_without_build_receipt_is_unverified(tmp_path):
    evidence = mom6.build_provenance(tmp_path, {}, {"sha256": "a"*64}, {})
    assert evidence["status"] == "unverified"


def test_minimal_fake_build_receipt_cannot_claim_executed_commands(tmp_path):
    (tmp_path/"build_manifest.json").write_text(json.dumps({"sha256": "a"*64, "source_identity": {}, "binary_dependencies": {}}))
    evidence = mom6.build_provenance(tmp_path, {}, {"sha256": "a"*64}, {})
    assert evidence["status"] == "unverified"


@pytest.mark.skipif(os.name != "posix", reason="Linux process-group cleanup requires POSIX runner")
def test_monitor_failure_kills_and_waits_without_running_mom6(tmp_path, monkeypatch):
    executable = tmp_path/"MOM6"
    executable.write_bytes(b"mock executable")
    manifest = {"case_id": CASE_ID, "run_id": "mock", "execution_status": "proposed", "inputs": {},
                "executable": {"path": str(executable), "sha256": file_digest(executable)},
                "source_identity": {}, "cache": str(tmp_path), "environment": {"machine": "test"}}
    (tmp_path/"run.json").write_text(json.dumps(manifest))
    class FakeProcess:
        pid = 987654
        waited = False
        def poll(self):
            return None
        def wait(self, timeout=None):
            self.waited = True
            return -9
    process = FakeProcess()
    killed = []
    monkeypatch.setattr(mom6.subprocess, "Popen", lambda *args, **kwargs: process)
    monkeypatch.setattr(mom6.os, "killpg", lambda pid, sig: killed.append((pid, sig)))
    def fail_monitor(group):
        raise RuntimeError("monitor failure")
    monkeypatch.setattr(mom6, "_process_group_rss", fail_monitor)
    with pytest.raises(RuntimeError, match="monitor failure"):
        run(tmp_path)
    assert killed and killed[0][0] == process.pid and process.waited
    assert json.loads((tmp_path/"run.json").read_text())["execution_status"] == "failed"


def test_repeated_native_execution_rejected_without_launch(tmp_path):
    native_fixture(tmp_path)
    with pytest.raises(ValueError):
        run(tmp_path)


def test_native_ambiguous_overrides_rejected(tmp_path):
    (tmp_path/"MOM_input").write_text('GRID_CONFIG="spherical"\nDT=100\nDT=200\n')
    (tmp_path/"MOM_override").write_text('')
    with pytest.raises(ValueError, match="ambiguous"):
        native_parameters(tmp_path)


def test_standing_wave_cell_mean_preparation_uses_exact_spatial_integral(tmp_path):
    contract = {"schema": "standing-wave-v0", "nx": 64, "ny": 8, "nz": 4, "H_m": 100.,
                "gravity": 9.81, "rho0": 1025., "f": 0., "T_C": 15., "S_psu": 35.,
                "period_s": 32000., "Ly_m": 100000., "Lx_m": 32000*np.sqrt(981.),
                "amplitude_m": .01,
                "native_sampling": {"MOM6": {"eta": "cell_mean", "u": "node", "layout": "cgrid"}}}
    path = tmp_path/"contract.json"
    path.write_text(json.dumps(contract))
    receipt = prepare_wave_input(path, tmp_path/"prepared")
    dx, k = contract["Lx_m"]/64, 2*np.pi/contract["Lx_m"]
    faces = np.arange(65)*dx
    independent_mean = .01*(np.sin(k*faces[1:])-np.sin(k*faces[:-1]))/(k*dx)
    with netCDF4.Dataset(receipt["input_path"]) as ds:
        np.testing.assert_allclose(ds["eta"][0, 0, :], independent_mean, rtol=0, atol=2e-16)
        assert ds["x"].units == "m"
        np.testing.assert_array_equal(ds["eta"][-1, :, :], -100.)
    assert receipt["execution_status"] == "proposed"


def test_wave_native_option_guard_remains_active_with_optimized_python(tmp_path):
    import os
    import subprocess
    import sys

    (tmp_path / "native-run.json").write_text("{}")
    (tmp_path / "run.log").write_text("manufactured native log")
    (tmp_path / "MOM_parameter_doc.all").write_text("SPLIT = False ! deliberate mismatch\n")
    code = (
        "from zhenmode.benchmarks.standing_wave import contract; "
        "from zhenmode.baselines.mom6.standing_wave import convert; "
        "convert(contract(), " + repr(str(tmp_path)) + ", {})"
    )
    result = subprocess.run(
        [sys.executable, "-O", "-c", code],
        capture_output=True,
        text=True,
        env=dict(os.environ, PYTHONOPTIMIZE="1"),
        timeout=30,
    )
    assert result.returncode != 0
    assert "MOM6 resolved SPLIT" in result.stderr
