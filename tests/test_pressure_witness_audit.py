"""Raw pressure witness identity, numeric gates and report negative controls."""
import hashlib
import json

import numpy as np
import pytest

from research.experiments.material_rstar_coordinates.pressure_witness_audit import (
    inspect_witness,
)


def _raw_pack(folder):
    shape = (2, 2, 2)
    density = 1025.
    arrays = {"reference_density_kg_per_m3": np.asarray(density), "area_m2": np.ones(shape[:2]),
              "thickness_m": np.ones(shape), "velocity_x_m_per_s": np.ones(shape), "velocity_y_m_per_s": np.zeros(shape),
              "independent_density_content_rate_kg_per_s": np.zeros(shape),
              "density_content_energy_conjugate_m2_per_s2": np.zeros(shape),
              "independent_surface_rate_m_per_s": np.ones(shape[:2]),
              "surface_energy_conjugate_joules_per_m": np.ones(shape[:2]),
              "pressure_pa": np.full(shape, 1000.), "dx_m": np.ones(shape[:2]), "dy_m": np.asarray(1.)}
    for prefix in ("existing", "width_weighted", "energy_adjoint"):
        arrays[f"{prefix}_force_x_m_per_s2"] = np.full(shape, -.5 / density if prefix == "energy_adjoint" else 0.)
        arrays[f"{prefix}_force_y_m_per_s2"] = np.zeros(shape)
    witness = folder / "affine_stair_witness.npz"
    np.savez_compressed(witness, **arrays)
    candidates = {}
    for name in ("existing_chain_rule", "actual_width_chain_rule", "energy_adjoint"):
        paired = name == "energy_adjoint"
        candidates[name] = {"work_residual_watts": 0. if paired else 4.,
                            "work_64eps_floor_watts": 64. * np.finfo(float).eps * (8. if paired else 4.),
                            "maximum_force_m_per_s2": .5 / density if paired else 0.,
                            "rest_force_64eps_floor_m_per_s2": 64. * np.finfo(float).eps * 1000. / density,
                            "pressure_work_passed": paired, "rest_gate_passed": not paired}
    report = {"witness_sha256": hashlib.sha256(witness.read_bytes()).hexdigest(),
              "cases": [{"stairs": True, "flat_eta": True, "density": "affine", "candidate_gates": candidates}]}
    (folder / "report.json").write_text(json.dumps(report), encoding="utf-8")
    return arrays, report


def test_raw_pressure_audit_verifies_failure_and_serializes_native_json_types(tmp_path):
    _raw_pack(tmp_path)
    result = inspect_witness(tmp_path)
    assert result["independent_witness_audit_passed"]
    assert not result["witness_physical_interface_passed"]
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("tamper", ["flag", "value"])
def test_raw_pressure_audit_detects_false_report_claims(tmp_path, tamper):
    _, report = _raw_pack(tmp_path)
    candidate = report["cases"][0]["candidate_gates"]["actual_width_chain_rule"]
    candidate["pressure_work_passed" if tamper == "flag" else "work_residual_watts"] = True if tamper == "flag" else 0.
    (tmp_path / "report.json").write_text(json.dumps(report), encoding="utf-8")
    assert not inspect_witness(tmp_path)["independent_witness_audit_passed"]


@pytest.mark.parametrize("tamper", ["bytes", "float32", "nan"])
def test_raw_pressure_audit_rejects_changed_bytes_or_invalid_numeric_arrays(tmp_path, tamper):
    arrays, report = _raw_pack(tmp_path)
    if tamper == "float32":
        arrays["thickness_m"] = arrays["thickness_m"].astype(np.float32)
    elif tamper == "nan":
        arrays["thickness_m"][0, 0, 0] = np.nan
    else:
        arrays["thickness_m"][0, 0, 0] = 2.
    witness = tmp_path / "affine_stair_witness.npz"
    np.savez_compressed(witness, **arrays)
    if tamper != "bytes":
        report["witness_sha256"] = hashlib.sha256(witness.read_bytes()).hexdigest()
        (tmp_path / "report.json").write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ValueError, match="bytes changed" if tamper == "bytes" else "finite float64"):
        inspect_witness(tmp_path)
