"""Raw weak-work witness verification refuses lies and corrupted arrays."""
import hashlib
import json

import numpy as np
import pytest

from research.experiments.material_rstar_coordinates.weak_witness_audit import (
    inspect_weak_witnesses,
)


def _fixture(folder):
    field = np.ones((1, 1, 2))
    plane = np.ones((1, 1))
    arrays = {"area_m2": plane, "thickness_m": field, "reference_density_kg_per_m3": np.array(1025.),
              "velocity_x_m_per_s": field, "velocity_y_m_per_s": field,
              "force_x_m_per_s2": 0. * field, "force_y_m_per_s2": 0. * field,
              "content_rate_kg_per_s": 0. * field, "content_conjugate_m2_per_s2": field,
              "surface_rate_m_per_s": 0. * plane, "surface_conjugate_J_per_m": plane,
              "pressure_Pa": field, "pressure_floor_reference_Pa": field, "dx_m": plane, "dy_m": np.array(1.),
              "rhs_absolute_scale_kg_per_s": field, "actual_content_rate_kg_per_s": 0. * field,
              "actual_surface_rate_m_per_s": 0. * plane, "eta_absolute_scale_m_per_s": plane,
              "geometry_mass_fraction": .5 * field,
              "registered_force_floor_m_per_s2": np.array(64. * np.finfo(float).eps / 1025.)}
    path = folder / "witness.npz"
    np.savez_compressed(path, **arrays)
    case = {"witness": path.name, "witness_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "truncate": False, "density": "constant", "rest_applicable": True,
            "rhs_numpy_passed": True, "eta_numpy_passed": True, "pressure_work_passed": True,
            "physical_affine_rest_passed": True, "constant_tracer_geometry_passed": True,
            "pressure_work_residual_watts": 0., "pressure_work_64eps_floor_watts": 0.,
            "maximum_force_m_per_s2": 0., "rest_force_64eps_floor_m_per_s2": float(arrays["registered_force_floor_m_per_s2"]),
            "global_inventory_residual_kg_per_s": 0., "global_volume_residual_m3_per_s": 0.}
    cases = []
    for stairs in (False, True):
        for flat in (False, True):
            for kind in ("zero", "constant", "affine", "random"):
                current = {**case, "stairs": stairs, "flat": flat, "density": kind}
                current["rest_applicable"] = flat and kind != "random"
                current["physical_affine_rest_passed"] = True if current["rest_applicable"] else None
                current["constant_tracer_geometry_passed"] = True if kind == "constant" else None
                cases.append(current)
    report = {"cases": cases, "completed_bed_representation": True, "original_domain_instantaneous_gates_passed": True}
    (folder / "report.json").write_text(json.dumps(report), encoding="utf-8")
    return arrays, report


def test_pure_weak_audit_serializes_and_verifies_all_saved_flags(tmp_path):
    _fixture(tmp_path)
    result = inspect_weak_witnesses(tmp_path)
    assert result["independent_witness_audit_passed"]
    assert result["original_domain_instantaneous_gates_passed"]
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("lie", ["rhs_numpy_passed", "pressure_work_passed", "physical_affine_rest_passed"])
def test_pure_weak_audit_rejects_report_flag_lies(tmp_path, lie):
    _, report = _fixture(tmp_path)
    selected = next(case for case in report["cases"] if case["rest_applicable"] and case["density"] == "constant")
    selected[lie] = False
    (tmp_path / "report.json").write_text(json.dumps(report), encoding="utf-8")
    assert not inspect_weak_witnesses(tmp_path)["independent_witness_audit_passed"]


def test_pure_weak_audit_rejects_modified_witness_bytes(tmp_path):
    _fixture(tmp_path)
    (tmp_path / "witness.npz").write_bytes(b"changed")
    with pytest.raises(ValueError, match="bytes changed"):
        inspect_weak_witnesses(tmp_path)


@pytest.mark.parametrize("kind", ["float32", "nan"])
def test_pure_weak_audit_rejects_nonfinite_or_wrong_dtype_even_with_updated_hash(tmp_path, kind):
    arrays, report = _fixture(tmp_path)
    arrays["force_x_m_per_s2"] = arrays["force_x_m_per_s2"].astype(np.float32) if kind == "float32" else arrays["force_x_m_per_s2"] + np.nan
    path = tmp_path / "witness.npz"
    np.savez_compressed(path, **arrays)
    for case in report["cases"]:
        case["witness_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    (tmp_path / "report.json").write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ValueError, match="finite float64"):
        inspect_weak_witnesses(tmp_path)


@pytest.mark.parametrize("kind", ["empty", "missing", "duplicate"])
def test_pure_weak_audit_rejects_incomplete_or_duplicate_registered_controls(tmp_path, kind):
    _, report = _fixture(tmp_path)
    if kind == "empty":
        report["cases"] = []
    elif kind == "missing":
        report["cases"].pop()
    else:
        report["cases"][-1] = report["cases"][0].copy()
    (tmp_path / "report.json").write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ValueError, match="complete unique"):
        inspect_weak_witnesses(tmp_path)
