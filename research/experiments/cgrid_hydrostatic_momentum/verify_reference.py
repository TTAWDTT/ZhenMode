"""Independent inventory/geometry checks of eight hashed linear momentum cases."""
import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]


def _path(relative):
    path = (ROOT / relative).resolve()
    if not path.is_relative_to(ROOT) or not path.is_file():
        raise ValueError(f"missing or outside-workspace evidence: {relative}")
    return path


def verify(report):
    if report["scope"] != "frozen_pressure_linear_3d_momentum_active_density_fct_not_full_ocean":
        raise ValueError("unexpected reference scope")
    if report.get("rotation_formulation") != "physical_wet_dual_rectangles_and_common_overlap":
        raise ValueError("physical rotation formulation not qualified by this verifier")
    if report.get("pressure_formulation") != "shared_face_force_over_physical_dual_mass":
        raise ValueError("physical pressure formulation not qualified by this verifier")
    if report["status"] != "PASS" or len(report["runs"]) != 8:
        raise ValueError("eight reference groups have not passed")
    required = {f"src/{name}" for name in ("finite_volume.py", "bounded_transport.py", "barotropic_transport.py",
                                         "cgrid_momentum.py", "grid.py", "config.py")}
    required |= {f"research/experiments/cgrid_hydrostatic_momentum/{name}" for name in ("run_reference.py", "protocol.md", "selection.md", "overlap_protocol.md", "pressure_work_protocol.md")}
    required.add("tests/test_cgrid_hydrostatic_momentum.py")
    required.add("tests/test_cgrid_pressure_work.py")
    manifest = report["provenance"]["source_sha256"]
    if not required.issubset(manifest):
        raise ValueError("incomplete runtime source manifest")
    for relative, expected in manifest.items():
        if hashlib.sha256(_path(relative).read_bytes()).hexdigest() != expected:
            raise ValueError(f"runtime source mismatch: {relative}")
    if report["reference_coefficients"] != [1., 1e-3, 1e-7]:
        raise ValueError("reference density profile changed")
    expected_groups = {(geometry, dtype, disturbed) for geometry in ("prior_smoothed", "unsmoothed")
                       for dtype in ("float64", "float32") for disturbed in (False, True)}
    groups, verified = set(), []
    for run in report["runs"]:
        group = (run["geometry"], run["velocity_dtype"], run["disturbed"])
        if group not in expected_groups or group in groups:
            raise ValueError("unknown or duplicate reference group")
        groups.add(group)
        if (run["completed_steps"] != 100 or run["dt_seconds"] != 60. or run["barotropic_substeps"] != 4
                or run["inventory_dtype"] != "float64" or run["status"] != "PASS" or len(run["history"]) != 100):
            raise ValueError("duration, precision or accepted-step history changed")
        for iteration, row in enumerate(run["history"], start=1):
            values = [value for key, value in row.items() if key not in ("stage_valid", "step")]
            if row["step"] != iteration or not all(row["stage_valid"]) or len(row["stage_valid"]) != 4:
                raise ValueError("rejected or incomplete stage history")
            if not all(value is not None and np.isfinite(value) for value in values):
                raise ValueError("nonfinite intermediate diagnostics")
            energy_gate = 2e-6 if run["velocity_dtype"] == "float32" else 1e-12
            if (row["rotation_energy_relative"] > energy_gate or row["rotation_solve_relative"] > 1e-12
                    or row["surface_error_m"] > row["surface_gate_m"] or row["surface_gate_m"] < 1e-12
                    or row["surface_gate_m"] > 1e-12 + 1e-12 * max(row["eta_max_m"], run["history"][max(iteration - 2, 0)]["eta_max_m"]) + 1e-24
                    or row["outflow_fraction"] > 1. or row["gravity_cfl_bound"] > 2.
                    or row["constant_error"] > 1e-12 or row["bound_excursion"] > 1e-12
                    or row["lower_height_relative_error"] > 1e-12
                    or (not run["disturbed"] and row["pressure_max_m_s2"] > 1e-12)):
                raise ValueError("recorded intermediate physical gate failed")
        snapshot = _path(run["snapshot_path"])
        if hashlib.sha256(snapshot.read_bytes()).hexdigest() != run["snapshot_sha256"]:
            raise ValueError("snapshot hash mismatch")
        with np.load(snapshot) as data:
            initial_volume, final_volume = data["initial_volume"], data["final_volume"]
            initial_content, final_content = data["initial_content"], data["final_content"]
            if initial_volume.shape != (180, 66, 14) or final_volume.shape != initial_volume.shape:
                raise ValueError("physical volume grid shape changed")
            if initial_content.shape != initial_volume.shape + (2,) or final_content.shape != initial_content.shape:
                raise ValueError("content shape mismatch")
            arrays = (initial_volume, final_volume, initial_content, final_content)
            if any(field.dtype != np.float64 or not np.all(np.isfinite(field)) for field in arrays):
                raise ValueError("inventory precision or finiteness failed")
            interfaces, depth, area = data["interfaces"], data["physical_depth"], data["area"]
            expected_area = 6.371e6 ** 2 * np.radians(np.diff(data["longitude_edges"]))[:, None] * np.diff(np.sin(np.radians(data["latitude_edges"])))[None, :]
            height = np.maximum(np.minimum(depth[..., None] - interfaces[:-1], np.diff(interfaces)), 0.)
            if interfaces[-1] != 8000. or not np.all(np.diff(interfaces) > 0.):
                raise ValueError("physical vertical interfaces changed")
            np.testing.assert_allclose(area, expected_area, rtol=1e-14)
            np.testing.assert_allclose(data["thickness"], height, rtol=1e-14)
            np.testing.assert_allclose(initial_volume, area[..., None] * height, rtol=1e-14)
            wet = initial_volume > 0.
            if np.any(final_volume[wet] <= 0.) or np.any(final_volume[~wet] != 0.) or np.any(final_content[~wet] != 0.):
                raise ValueError("material cells invalid or dry inventories changed")
            lower = np.max(np.abs(final_volume[..., 1:] / area[..., None] - height[..., 1:]) / np.maximum(height[..., 1:], 1.))
            if lower > 1e-12:
                raise ValueError("non-top volume/geometry changed")
            budgets = np.abs(np.sum(final_content - initial_content, axis=(0, 1, 2))) / np.sum(np.abs(initial_content), axis=(0, 1, 2))
            if np.any(budgets > 1e-12):
                raise ValueError("content budgets failed")
            concentration_initial = initial_content[wet] / initial_volume[wet, None]
            concentration_final = final_content[wet] / final_volume[wet, None]
            coefficients = data["reference_coefficients"]
            np.testing.assert_array_equal(coefficients, [1., 1e-3, 1e-7])
            centers = interfaces[:-1] + .5 * height
            density = coefficients[0] + coefficients[1] * centers + coefficients[2] * (centers ** 2 + height ** 2 / 12.)
            longitude = .5 * (data["longitude_edges"][:-1] + data["longitude_edges"][1:])
            if run["disturbed"]:
                density += .1 * np.sin(np.radians(longitude))[:, None, None]
            np.testing.assert_allclose(concentration_initial[:, 0], density[wet], rtol=1e-12, atol=1e-12)
            if (np.max(np.abs(concentration_final[:, 1] - 35.)) > 1e-12
                    or np.any(concentration_final.min(axis=0) < concentration_initial.min(axis=0) - 1e-12)
                    or np.any(concentration_final.max(axis=0) > concentration_initial.max(axis=0) + 1e-12)):
                raise ValueError("constant concentration or bounds failed")
            for component in ("east", "north"):
                field = data[f"final_{component}_velocity"]
                if field.dtype.name != run["velocity_dtype"] or field.shape != initial_volume.shape or not np.all(np.isfinite(field)):
                    raise ValueError("velocity precision, shape or finiteness failed")
            if run["disturbed"] and (run["maximum_speed_m_s"] <= 1e-9 or run["maximum_shear_m_s"] <= 0.):
                raise ValueError("manufactured momentum path inactive")
            volume_delta = float(np.sum(final_volume - initial_volume))
            np.testing.assert_allclose(volume_delta, run["volume_budget_delta_m3"], rtol=1e-12, atol=1e-12)
            np.testing.assert_allclose(budgets, run["content_budget_relative"], rtol=1e-12, atol=1e-30)
            verified.append({"group": group, "content_budget_relative": budgets.tolist(), "volume_delta_m3": volume_delta})
    if groups != expected_groups:
        raise ValueError("missing reference group")
    return verified


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", default="results/industrial_alignment/cgrid_momentum_mass_reference.json")
    parser.add_argument("--negative-controls", action="store_true")
    args = parser.parse_args()
    report = json.loads((ROOT / args.report).read_text(encoding="utf-8"))
    verified = verify(report)
    print(json.dumps({"scope": "inventory_and_geometry_snapshots_plus_recorded_stage_gates_not_independent_step_replay",
                      "verifier_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "verified": verified}, indent=2))
    if args.negative_controls:
        for label in ("duration", "dtype", "stage", "snapshot", "duplicate"):
            altered = copy.deepcopy(report)
            if label == "duration":
                altered["runs"][0]["dt_seconds"] = 600.
            elif label == "dtype":
                altered["runs"][0]["velocity_dtype"] = "float16"
            elif label == "stage":
                altered["runs"][0]["history"][0]["stage_valid"][0] = False
            elif label == "snapshot":
                altered["runs"][0]["snapshot_sha256"] = "0" * 64
            else:
                altered["runs"][1] = copy.deepcopy(altered["runs"][0])
            try:
                verify(altered)
            except (ValueError, AssertionError) as error:
                print(f"negative {label}: rejected ({error})")
            else:
                raise ValueError(f"negative control accepted: {label}")


if __name__ == "__main__":
    main()
