"""Pure NumPy verification of all saved weak work/rest witnesses, not trajectories."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def inspect_weak_witnesses(folder):
    folder = Path(folder)
    report = json.loads((folder / "report.json").read_text(encoding="utf-8"))
    completed = report.get("completed_bed_representation")
    if type(completed) is not bool:
        raise ValueError("weak witness audit requires explicit representation and control matrix")
    expected = {(truncate, stairs, flat, kind)
                for truncate in ((False,) if completed else (False, True))
                for stairs in (False, True) for flat in (False, True)
                for kind in ("zero", "constant", "affine", "random")}
    observed = [(case.get("truncate"), case.get("stairs"), case.get("flat"), case.get("density"))
                for case in report["cases"]]
    if len(observed) != len(expected) or set(observed) != expected:
        raise ValueError("weak witness audit requires the complete unique registered control matrix")
    cases = []
    epsilon = 64. * np.finfo(float).eps
    for case in report["cases"]:
        witness = folder / case["witness"]
        if hashlib.sha256(witness.read_bytes()).hexdigest() != case["witness_sha256"]:
            raise ValueError("weak witness bytes changed")
        with np.load(witness, allow_pickle=False) as saved:
            arrays = {name: saved[name].copy() for name in saved.files}
        if any(value.dtype != np.float64 or not np.all(np.isfinite(value)) for value in arrays.values()):
            raise ValueError("weak witnesses require finite float64 arrays")
        area, thickness = arrays["area_m2"], arrays["thickness_m"]
        rho0 = float(arrays["reference_density_kg_per_m3"])
        kinetic = tuple(rho0 * area[..., None] * thickness * arrays[f"velocity_{axis}_m_per_s"] * arrays[f"force_{axis}_m_per_s2"] for axis in ("x", "y"))
        potential = (arrays["content_rate_kg_per_s"] * arrays["content_conjugate_m2_per_s2"],
                     arrays["surface_rate_m_per_s"] * arrays["surface_conjugate_J_per_m"])
        work = sum(float(np.sum(value)) for value in (*kinetic, *potential))
        work_floor = epsilon * sum(float(np.sum(np.abs(value))) for value in (*kinetic, *potential))
        force = max(float(np.max(np.abs(arrays[f"force_{axis}_m_per_s2"]))) for axis in ("x", "y"))
        pressure = min(float(np.max(np.abs(arrays[name]))) for name in ("pressure_Pa", "pressure_floor_reference_Pa"))
        force_floor = epsilon * pressure / rho0 * max(1. / float(np.min(arrays["dx_m"])), 1. / float(arrays["dy_m"]))
        scale = arrays["rhs_absolute_scale_kg_per_s"]
        actual = arrays["actual_content_rate_kg_per_s"]
        eta = arrays["actual_surface_rate_m_per_s"]
        rhs_passed = bool(np.all(np.abs(actual - arrays["content_rate_kg_per_s"]) <= epsilon * (1. + scale)))
        eta_passed = bool(np.all(np.abs(eta - arrays["surface_rate_m_per_s"]) <= epsilon * (1e-20 + arrays["eta_absolute_scale_m_per_s"])))
        global_rhs, global_floor = float(np.sum(actual)), epsilon * float(np.sum(scale))
        volume, volume_floor = float(np.sum(area * eta)), epsilon * float(np.sum(np.abs(area * eta)))
        constant_passed = None
        if case["density"] == "constant":
            expected = 1.5 * area[..., None] * arrays["geometry_mass_fraction"] * eta[..., None]
            constant_passed = bool(np.all(np.abs(actual - expected) <= epsilon * (1. + scale + np.abs(expected))))
        flags = {"rhs_numpy_passed": rhs_passed, "eta_numpy_passed": eta_passed,
                 "pressure_work_passed": bool(abs(work) <= work_floor),
                 "physical_affine_rest_passed": bool(force <= force_floor) if case["rest_applicable"] else None,
                 "constant_tracer_geometry_passed": constant_passed}
        comparisons = ((work, case["pressure_work_residual_watts"], work_floor),
                       (work_floor, case["pressure_work_64eps_floor_watts"], epsilon * work_floor),
                       (force, case["maximum_force_m_per_s2"], force_floor),
                       (force_floor, case["rest_force_64eps_floor_m_per_s2"], epsilon * force_floor),
                       (force_floor, float(arrays["registered_force_floor_m_per_s2"]), epsilon * force_floor),
                       (global_rhs, case["global_inventory_residual_kg_per_s"], global_floor),
                       (volume, case["global_volume_residual_m3_per_s"], volume_floor))
        agreed = all(case[name] == value for name, value in flags.items()) and all(abs(actual_value - recorded) <= floor for actual_value, recorded, floor in comparisons)
        physical = bool(all(value is not False for value in flags.values()) and abs(global_rhs) <= global_floor and abs(volume) <= volume_floor)
        cases.append({"witness": case["witness"], "truncate": case["truncate"], "report_agrees": agreed,
                      "instantaneous_witness_gates_passed": physical, "pressure_work_residual_watts": work,
                      "pressure_work_floor_watts": work_floor, "maximum_force_m_per_s2": force, "flags": flags})
    original_passed = all(case["instantaneous_witness_gates_passed"] for case in cases if not case["truncate"])
    return {"scope": "all saved arrays: work/rest and kernel versus saved independent quadrature, not an independent rerun of quadrature or an ocean step",
            "cases": cases, "all_witness_hashes_valid": True,
            "registered_control_matrix_verified": True,
            "independent_witness_audit_passed": all(case["report_agrees"] for case in cases)
            and original_passed == report["original_domain_instantaneous_gates_passed"],
            "original_domain_instantaneous_gates_passed": original_passed, "production_promotion": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    if arguments.output.exists():
        raise ValueError("independent weak audit output must be new")
    result = inspect_weak_witnesses(arguments.input)
    result["auditor_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    arguments.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, allow_nan=False), flush=True)
    raise SystemExit(0 if result["independent_witness_audit_passed"] else 1)


if __name__ == "__main__":
    main()
