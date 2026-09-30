"""Reproduce dimensioned metric diagnostics with archived harness/source hashes."""
import argparse
import hashlib
import importlib.util
import json
import shutil
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    root = Path(__file__).resolve().parents[3]
    for folder in (root, root / "src", root / "tests"):
        sys.path.insert(0, str(folder))
    output = arguments.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    source_files = [root / "tests/test_rstar_metric_controls.py", root / "tests/_helpers.py",
                    root / "src/jax_solver_global.py", root / "src/config.py", root / "src/grid.py",
                    root / "research/experiments/material_rstar_coordinates/kernel.py",
                    root / "research/experiments/material_rstar_coordinates/dense_oracle.py",
                    root / "research/experiments/material_rstar_coordinates/protocol.json", Path(__file__).resolve()]
    hashes = {}
    for source in source_files:
        name = source.relative_to(root).as_posix()
        target = output / "sources" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        hashes[name] = hashlib.sha256(source.read_bytes()).hexdigest()
    (output / "source_hashes.json").write_text(json.dumps(hashes, indent=2) + "\n", encoding="utf-8")
    specification = importlib.util.spec_from_file_location("metric_controls", root / "tests/test_rstar_metric_controls.py")
    controls = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(controls)
    import jax
    import jax.numpy as jnp
    import numpy as np

    _, params, depths = controls._fixture()
    generator = np.random.default_rng(930004)
    eta = generator.uniform(-3., 2., params.wet_mask.shape) * np.asarray(params.wet_mask)
    geometry = controls._geometry(params, depths, eta)
    stencil = controls.make_reference_stencil(depths, params.wet_mask_z)
    dense = controls.assemble_diffusion(eta, depths, np.asarray(params.dz_node),
                                      np.asarray(params.wet_mask_z), np.asarray(params.dx_2d),
                                      float(params.dy), np.asarray(params.cos_lat), 1000., .001)
    field = generator.normal(size=params.wet_mask_z.shape)
    response = np.asarray(controls.diffusion_content_rhs(jnp.asarray(field), geometry, stencil, params, 1000., .001))[dense.wet]
    expected = dense.stiffness @ field[dense.wet] / dense.area
    floor = 64. * np.finfo(float).eps * np.max(np.abs(dense.stiffness) @ np.abs(field[dense.wet]) / dense.area)
    inverse_root_mass = 1. / np.sqrt(dense.mass)
    naive = dense.old_divergence_stiffness * inverse_root_mass[:, None] * inverse_root_mass[None, :]
    symmetric = .5 * (naive + naive.T)
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    eigen_floor = 64. * np.finfo(float).eps * np.linalg.norm(naive, ord=np.inf)
    witness = inverse_root_mass * eigenvectors[:, -1]
    naive_work = float(witness @ dense.old_divergence_stiffness @ witness)
    corrected_work = float(witness @ dense.horizontal_stiffness @ witness)
    normalized = dense.stiffness * inverse_root_mass[:, None] * inverse_root_mass[None, :]
    normalized_floor = 64. * np.finfo(float).eps * np.linalg.norm(normalized, ord=np.inf)
    corrected_max_eigenvalue = float(np.linalg.eigvalsh(.5 * (normalized + normalized.T))[-1])
    off_diagonal = dense.stiffness.copy()
    np.fill_diagonal(off_diagonal, 0.)
    minimum_off_diagonal = float(np.min(off_diagonal))
    row, column = np.unravel_index(np.argmin(off_diagonal), off_diagonal.shape)
    maximum_principle_witness = np.zeros(len(dense.mass))
    maximum_principle_witness[column] = 1.
    undershoot_rate = float((dense.stiffness @ maximum_principle_witness / dense.mass)[row])
    resting_eta = jnp.full(params.wet_mask.shape, -2.6) * params.wet_mask
    resting_geometry = controls._geometry(params, depths, resting_eta)
    rho = 1.5 + .002 * resting_geometry.node_depth
    pressure = controls.hydrostatic_pressure(rho, resting_eta, resting_geometry, params)
    force = controls.coordinate_pressure_gradient(pressure, rho, resting_geometry, params)
    false_force = controls._gradient_conservative_3d(pressure, params)
    pressure_bound = (64. * np.finfo(float).eps * float(jnp.max(jnp.abs(pressure)))
                      / controls.RHO_0 * max(float(jnp.max(params.inv_dx)), float(params.inv_dy)))
    errors = controls._pressure_coordinate_errors()
    ratios = [before / after for before, after in zip(errors[:-1], errors[1:], strict=True)]
    matrix_floor = 64. * np.finfo(float).eps * np.linalg.norm(dense.stiffness, ord=np.inf)
    constant_residual = float(np.max(np.abs(dense.stiffness.sum(axis=1))))
    inventory_residual = float(np.max(np.abs(dense.stiffness.sum(axis=0))))
    source_unchanged = all(hashlib.sha256((root / name).read_bytes()).hexdigest() == digest for name, digest in hashes.items())
    report = {
        "scope": "isolated metric control, not a qualified actual trajectory or industrial comparison",
        "backend": jax.default_backend(), "devices": [str(device) for device in jax.devices()],
        "jax_version": jax.__version__, "x64": bool(jax.config.jax_enable_x64),
        "wet_unknowns": int(dense.wet.sum()),
        "minimum_wet_thickness_m": float(np.min(dense.thickness[dense.wet])),
        "dense_rhs_error_content_per_second": float(np.max(np.abs(response - expected))),
        "dense_rhs_64eps_floor": float(floor),
        "constant_stiffness_residual": constant_residual, "inventory_stiffness_residual": inventory_residual,
        "stiffness_64eps_floor": float(matrix_floor),
        "old_divergence_positive_eigenvalue_per_second": float(eigenvalues[-1]),
        "old_divergence_eigenvalue_64eps_floor": float(eigen_floor),
        "old_divergence_witness_work_per_second": naive_work,
        "same_witness_corrected_horizontal_work_per_second": corrected_work,
        "corrected_full_maximum_eigenvalue_per_second": corrected_max_eigenvalue,
        "corrected_full_eigenvalue_64eps_floor": float(normalized_floor),
        "minimum_corrected_off_diagonal_stiffness": minimum_off_diagonal,
        "maximum_principle_unit_pulse_undershoot_rate_per_second": undershoot_rate,
        "maximum_principle_qualified": False,
        "corrected_resting_force_m_per_second_squared": max(float(jnp.max(jnp.abs(value))) for value in force),
        "uncorrected_resting_force_m_per_second_squared": max(float(jnp.max(jnp.abs(value))) / controls.RHO_0 for value in false_force),
        "resting_force_64eps_floor": float(pressure_bound),
        "quadratic_pressure_mms_errors_m_per_second_squared": errors,
        "quadratic_pressure_mms_refinement_ratios": ratios,
        "source_unchanged": source_unchanged,
    }
    report["metric_controls_passed"] = bool(
        source_unchanged and report["x64"] and report["dense_rhs_error_content_per_second"] <= floor
        and constant_residual <= matrix_floor and inventory_residual <= matrix_floor
        and eigenvalues[-1] > 1000. * eigen_floor and naive_work > 1000. * eigen_floor
        and corrected_work <= eigen_floor and corrected_max_eigenvalue <= normalized_floor
        and report["corrected_resting_force_m_per_second_squared"] <= pressure_bound
        and all(3.5 < ratio < 4.5 for ratio in ratios))
    np.savez(output / "dense_counterexamples.npz", eta=eta, wet=dense.wet, mass=dense.mass,
             corrected_stiffness=dense.stiffness, horizontal_stiffness=dense.horizontal_stiffness,
             old_divergence_stiffness=dense.old_divergence_stiffness, energy_witness=witness,
             unit_pulse=maximum_principle_witness, pulse_row=row, pulse_column=column)
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, allow_nan=False), flush=True)
    if not report["metric_controls_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
