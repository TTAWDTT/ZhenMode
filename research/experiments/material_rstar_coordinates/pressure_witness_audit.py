"""Pure NumPy pressure witness check, without importing the model or JAX."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def inspect_witness(folder):
    report = json.loads((folder / "report.json").read_text(encoding="utf-8"))
    witness = folder / "affine_stair_witness.npz"
    if hashlib.sha256(witness.read_bytes()).hexdigest() != report["witness_sha256"]:
        raise ValueError("pressure witness bytes changed")
    with np.load(witness, allow_pickle=False) as saved:
        arrays = {name: saved[name].copy() for name in saved.files}
    if any(value.dtype != np.float64 or not np.all(np.isfinite(value)) for value in arrays.values()):
        raise ValueError("pressure witness requires finite float64 arrays")
    case = next(case for case in report["cases"] if case["stairs"] and case["flat_eta"] and case["density"] == "affine")
    rho0 = float(arrays["reference_density_kg_per_m3"])
    mass = arrays["area_m2"][..., None] * arrays["thickness_m"]
    velocities = arrays["velocity_x_m_per_s"], arrays["velocity_y_m_per_s"]
    potential = (arrays["independent_density_content_rate_kg_per_s"] * arrays["density_content_energy_conjugate_m2_per_s2"],
                 arrays["independent_surface_rate_m_per_s"] * arrays["surface_energy_conjugate_joules_per_m"])
    metric = max(1. / float(np.min(arrays["dx_m"])), 1. / float(arrays["dy_m"]))
    force_floor = 64. * np.finfo(float).eps * float(np.max(np.abs(arrays["pressure_pa"]))) / rho0 * metric
    names = {"existing": "existing_chain_rule", "width_weighted": "actual_width_chain_rule", "energy_adjoint": "energy_adjoint"}
    candidates = {}
    for prefix, name in names.items():
        forces = arrays[f"{prefix}_force_x_m_per_s2"], arrays[f"{prefix}_force_y_m_per_s2"]
        kinetic = tuple(rho0 * mass * velocity * force for velocity, force in zip(velocities, forces, strict=True))
        residual = sum(float(np.sum(value)) for value in (*kinetic, *potential))
        floor = 64. * np.finfo(float).eps * sum(float(np.sum(np.abs(value))) for value in (*kinetic, *potential))
        maximum = max(float(np.max(np.abs(force))) for force in forces)
        work_passed, rest_passed = abs(residual) <= floor, maximum <= force_floor
        original = case["candidate_gates"][name]
        flags_agree = work_passed == original["pressure_work_passed"] and rest_passed == original["rest_gate_passed"]
        values_agree = (abs(residual - original["work_residual_watts"]) <= floor
                        and abs(floor - original["work_64eps_floor_watts"]) <= 64. * np.finfo(float).eps * floor
                        and abs(maximum - original["maximum_force_m_per_s2"]) <= force_floor
                        and abs(force_floor - original["rest_force_64eps_floor_m_per_s2"]) <= 64. * np.finfo(float).eps * force_floor)
        candidates[name] = {"pressure_work_residual_watts": residual, "pressure_work_64eps_floor_watts": floor,
                            "maximum_rest_force_m_per_s2": maximum, "rest_force_64eps_floor_m_per_s2": force_floor,
                            "pressure_work_passed": bool(work_passed), "rest_passed": bool(rest_passed),
                            "report_flags_agree": bool(flags_agree), "report_values_agree": bool(values_agree)}
    return {"scope": "saved affine-stair witness only, not all16 controls or full ocean physics",
            "witness_hash_valid": True, "candidates": candidates,
            "independent_witness_audit_passed": all(value["report_flags_agree"] and value["report_values_agree"] for value in candidates.values()),
            "witness_physical_interface_passed": any(value["pressure_work_passed"] and value["rest_passed"] for value in candidates.values()),
            "production_promotion": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    if arguments.output.exists():
        raise ValueError("independent audit output must be new")
    result = inspect_witness(arguments.input)
    result["auditor_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    arguments.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, allow_nan=False), flush=True)
    raise SystemExit(0 if result["independent_witness_audit_passed"] else 1)


if __name__ == "__main__":
    main()
