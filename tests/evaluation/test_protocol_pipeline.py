"""Protocol negative controls use declared toy data, never a model oracle."""
import copy
import json
from pathlib import Path

import netCDF4
import numpy as np
import pytest

from ocean_solver.evaluation.pipeline import compare, evaluate, import_historical
from ocean_solver.evaluation.protocols import digest, file_digest, load_json, validate_protocol
from ocean_solver.provenance.sources import PACKAGE_SOURCE_MODULES


@pytest.fixture
def bundle(tmp_path):
    protocol = {"schema_version": 1, "id": "toy-v2", "kind": "sst", "case_id": "toy",
                "metric_definition": "area_weighted_angular_box_v2",
                "reference": {"role": "saved_initial_surface_temperature", "version": "toy", "sha256": None},
                "window": {"start_day": 0, "end_day": 2},
                "spatial": {"coordinate_system": "geographic_degrees", "mask": "reference_wet",
                            "weighting": "wet_cell_area", "remapping": "none"},
                "temporal": {"weighting": "arithmetic_saved_records"}, "temperature_unit": "degC",
                "acceptance": {"max_raw_rmse_degC": None, "max_abs_heat_drift_percent": None,
                               "max_abs_salt_drift_percent": None}, "limitations": []}
    field = np.arange(16).reshape(4, 4) + 10.
    arrays = dict(days=np.array([0., 1., 2.]), T_top=np.stack([field]*3), T_init=field[:, :, None],
                  wet_mask=np.ones((4, 4), dtype=bool), lat=np.array([-60., -20., 20., 60.]),
                  lon=np.array([0., 90., 180., 270.]), verdict=np.array("PASS"),
                  max_u_peak=np.array(0.), max_eta=np.zeros(3), heat_content_J=np.ones(3),
                  salt_content_kg=np.ones(3))
    package = Path(__file__).resolve().parents[2]/"src/ocean_solver"
    full = {name.removeprefix("ocean_solver/")+".py": file_digest(package/(name.removeprefix("ocean_solver/")+".py"))
            for name in PACKAGE_SOURCE_MODULES}
    executed = {name: full[name] for name in ("runtime/application.py", "runtime/integration.py", "model/factory.py",
                                              "timestepping/integration.py", "dynamics/processes.py",
                                              "numerics/horizontal.py", "physics/vertical.py", "io/output.py")}
    manifest = dict(run_id="toy-repeat-1", case_id="toy", method="zhenmode", config_hash="a"*64,
                    execution_status="completed", physical_problem_sha256="b"*64, data_sha256="c"*64,
                    effective_physics_sha256="d"*64,
                    physical_grid_sha256="1"*64,
                    source_identity=full, executed_source_files=executed,
                    result_source_identity={"all_package_files": full},
                    cost={"status": "not_recorded"})
    arrays["source_identity_json"] = np.array(json.dumps({"all_package_files": full}))
    np.savez(tmp_path / "result.npz", **arrays)
    manifest.update(result_path=str((tmp_path/"result.npz").resolve()),
                    result_sha256=file_digest(tmp_path/"result.npz"))
    for name, value in (("protocol", protocol), ("manifest", manifest)):
        (tmp_path / (name + ".json")).write_text(json.dumps(value))
    return tmp_path, protocol, arrays, manifest


def run_bundle(bundle, output="report"):
    path, _, _, _ = bundle
    return evaluate(path/"result.npz", path/"protocol.json", path/"manifest.json", path/output)


def test_completed_is_not_unconfigured_acceptance(bundle):
    report = run_bundle(bundle)
    assert report["execution_status"] == "completed"
    assert report["acceptance"]["status"] == "not_declared"
    assert report["numerical"]["long_term_stability"] == "not_run"
    assert report["effect"]["reference_role"] == "initialization_field_not_independent_validation"
    assert report["industrial_qualified"] is False
    assert Path(bundle[0]/"report/report.md").is_file()


@pytest.mark.parametrize("change", [lambda p: p.update(temperature_unit="K"),
                                    lambda p: p.update(extra="unknown"),
                                    lambda p: p["window"].update(start_day=3),
                                    lambda p: p["spatial"].update(remapping="bilinear"),
                                    lambda p: p.update(metric_definition="legacy_equal_cell_index_box_v1")])
def test_invalid_protocol_rejected(bundle, change):
    protocol = copy.deepcopy(bundle[1])
    change(protocol)
    with pytest.raises(ValueError):
        validate_protocol(protocol)


def test_duplicate_fields_rejected(tmp_path):
    path = tmp_path/"ambiguous.json"
    path.write_text('{"window": 1, "window": 2}')
    with pytest.raises(ValueError, match="duplicate"):
        load_json(path)


@pytest.mark.parametrize("days", [np.array([0., 2., 1.]), np.array([0., 1., 3.]),
                                  np.array([0., np.nan, 2.])])
def test_bad_time_windows_rejected(bundle, days):
    bundle[2]["days"] = days
    np.savez(bundle[0]/"result.npz", **bundle[2])
    bundle[3]["result_sha256"] = file_digest(bundle[0]/"result.npz")
    (bundle[0]/"manifest.json").write_text(json.dumps(bundle[3]))
    with pytest.raises(ValueError):
        run_bundle(bundle)


def test_failed_run_and_changed_reference_rejected(bundle):
    path, protocol, _, manifest = bundle
    manifest["execution_status"] = "failed"
    (path/"manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="completed"):
        run_bundle(bundle)
    manifest["execution_status"] = "completed"
    (path/"manifest.json").write_text(json.dumps(manifest))
    protocol["reference"]["sha256"] = "f"*64
    (path/"protocol.json").write_text(json.dumps(protocol))
    with pytest.raises(ValueError, match="reference field"):
        run_bundle(bundle)


def test_comparison_refuses_missing_and_mixed_protocols(bundle):
    report = run_bundle(bundle)
    second = copy.deepcopy(report)
    second["run_id"] = "repeat-2"
    assert compare([report, second])["comparable"]
    assert compare([report, second])["cost_comparable"] is False
    second["comparison_identity"]["metric_definition"] = "legacy_equal_cell_index_box_v1"
    assert compare([report, second])["comparable"] is False
    second["comparison_identity"] = None
    assert compare([report, second])["comparable"] is False


def test_distinct_outputs_are_never_overwritten(bundle):
    run_bundle(bundle)
    with pytest.raises(FileExistsError):
        run_bundle(bundle)


def test_result_cannot_be_borrowed_from_another_run(bundle):
    bundle[2]["T_top"] = bundle[2]["T_top"] + 3
    np.savez(bundle[0]/"result.npz", **bundle[2])
    with pytest.raises(ValueError, match="actual hash-bound"):
        run_bundle(bundle)


def test_source_receipt_mismatch_rejected(bundle):
    bundle[3]["executed_source_files"]["dynamics/processes.py"] = "0"*64
    (bundle[0]/"manifest.json").write_text(json.dumps(bundle[3]))
    with pytest.raises(ValueError, match="executed source"):
        run_bundle(bundle)


@pytest.mark.parametrize("field,value", [("wet_mask", np.full((4, 4), np.nan)),
                                        ("wet_mask", np.full((4, 4), 2.)),
                                        ("max_u_peak", np.array(np.nan))])
def test_invalid_mask_or_peak_rejected(bundle, field, value):
    bundle[2][field] = value
    np.savez(bundle[0]/"result.npz", **bundle[2])
    bundle[3]["result_sha256"] = file_digest(bundle[0]/"result.npz")
    (bundle[0]/"manifest.json").write_text(json.dumps(bundle[3]))
    with pytest.raises(ValueError):
        run_bundle(bundle)


def test_protocol_same_id_content_change_rejected(bundle):
    bundle[3]["protocol_file_sha256"] = file_digest(bundle[0]/"protocol.json")
    (bundle[0]/"manifest.json").write_text(json.dumps(bundle[3]))
    bundle[1]["acceptance"]["max_raw_rmse_degC"] = 5
    (bundle[0]/"protocol.json").write_text(json.dumps(bundle[1]))
    with pytest.raises(ValueError, match="protocol content changed"):
        run_bundle(bundle)


@pytest.mark.parametrize("bad", ["grid", "wet", "unit", "unbound_geometry"])
def test_native_shared_grid_units_and_binding_rejected(bundle, bad):
    directory = bundle[0]
    prog, geometry = directory/"prog.nc", directory/"geometry.nc"
    with netCDF4.Dataset(prog, "w") as ds:
        for name, size in (("time", 3), ("y", 4), ("x", 4)):
            ds.createDimension(name, size)
        time = ds.createVariable("time", "f8", ("time",))
        time.units = "days since 2000-01-01"
        time[:] = [0, 1, 2]
        temp = ds.createVariable("temp", "f8", ("time", "y", "x"))
        temp.units = "K" if bad == "unit" else "degC"
        temp[:] = bundle[2]["T_top"].transpose(0, 2, 1)
    with netCDF4.Dataset(geometry, "w") as ds:
        ds.createDimension("lat", 4)
        ds.createDimension("lon", 4)
        ds.createVariable("lath", "f8", ("lat",))[:] = bundle[2]["lat"] + (1 if bad == "grid" else 0)
        ds.createVariable("lonh", "f8", ("lon",))[:] = bundle[2]["lon"]
        ds.createVariable("wet", "f8", ("lat", "lon"))[:] = 2 if bad == "wet" else 1
    bundle[3].update(result_path=str(prog), result_sha256=file_digest(prog),
                     reference_npz=str(directory/"result.npz"), reference_npz_sha256=file_digest(directory/"result.npz"),
                     geometry_path=str(geometry), geometry_sha256="0"*64 if bad == "unbound_geometry" else file_digest(geometry))
    (directory/"manifest.json").write_text(json.dumps(bundle[3]))
    with pytest.raises(ValueError):
        evaluate(prog, directory/"protocol.json", directory/"manifest.json", directory/"external-report",
                 format="mom6", reference=directory/"result.npz", geometry=geometry)


def test_two_facade_files_cannot_certify_production_source(bundle):
    bundle[3]["executed_source_files"] = {name: bundle[3]["source_identity"][name]
                                          for name in ("__init__.py", "runtime/entry.py")}
    (bundle[0]/"manifest.json").write_text(json.dumps(bundle[3]))
    report = run_bundle(bundle)
    assert report["comparability"]["status"] == "limited"
    assert "production_executed_core_source_envelope" in report["comparability"]["missing_identity"]
    assert compare([report, copy.deepcopy(report)])["comparable"] is False


def test_missing_source_envelope_stays_limited(bundle):
    bundle[3].pop("source_identity")
    (bundle[0]/"manifest.json").write_text(json.dumps(bundle[3]))
    report = run_bundle(bundle)
    assert report["comparability"]["status"] == "limited"
    assert compare([report, copy.deepcopy(report)])["comparable"] is False


def test_historical_v1_import_keeps_original_bytes(tmp_path):
    path = tmp_path/"old.json"
    original = b'{"global":{"raw_rmse":0.91364293},"global_a2_rmse_c":1.11257074}\n'
    path.write_bytes(original)
    report = import_historical(path)
    assert path.read_bytes() == original
    assert report["metric_definition"] == "legacy_equal_cell_index_box_v1"
    assert report["independent_rerun"] is False
    assert report["metrics"]["global"]["raw_rmse"] == 0.91364293
    assert digest(report["metrics"]) != report["source_sha256"]
