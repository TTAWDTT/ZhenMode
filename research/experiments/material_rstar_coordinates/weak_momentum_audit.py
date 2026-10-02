"""Freeze consistent kinetic mass with independent work and original rest gates."""
import argparse
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    root = Path(__file__).resolve().parents[3]
    output = arguments.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    names = ["src/config.py", "src/grid.py", "src/jax_solver_global.py", "src/material_top.py", "pyproject.toml"]
    names.extend("tests/" + name for name in
                 ("_helpers.py", "test_rstar_weak_momentum.py", "test_rstar_weak_transport.py",
                  "test_rstar_metric_controls.py", "test_rstar_representation.py", "test_rstar_pressure_work.py"))
    names.extend("research/experiments/material_rstar_coordinates/" + name for name in
                 ("kernel.py", "nodal_mass.py", "pressure_work.py", "weak_transport.py", "weak_oracle.py", "bed_completion.py",
                  "weak_momentum.py", "consistent_velocity_protocol.json", "weak_momentum_audit.py",
                  "pressure_accuracy.py", "pressure_accuracy_protocol.json"))
    hashes = {}
    from ocean_solver.provenance.archives import current_source_files
    files = current_source_files(root, names)
    for name, source in files.items():
        target = output / "sources" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        hashes[name] = hashlib.sha256(source.read_bytes()).hexdigest()
    (output / "source_hashes.json").write_text(json.dumps(hashes, indent=2) + "\n", encoding="utf-8")
    for folder in (root, root / "src", root / "tests"):
        sys.path.insert(0, str(folder))
    import jax
    import numpy as np

    from tests.support.rstar.weak_momentum import _consistent_diagnostic

    cases = []
    for stairs in (False, True):
        for flat in (False, True):
            for kind in ("zero", "constant", "affine", "random"):
                result, arrays = _consistent_diagnostic(stairs, flat, kind)
                name = f"stairs{int(stairs)}_flat{int(flat)}_{kind}.npz"
                np.savez_compressed(output / name, **arrays)
                result.update(witness=name, witness_sha256=hashlib.sha256((output / name).read_bytes()).hexdigest())
                cases.append(result)
    unchanged = all(hashlib.sha256(files[name].read_bytes()).hexdigest() == digest for name, digest in hashes.items())
    passed = all(case["rhs_numpy_passed"] and case["eta_numpy_passed"] and case["pressure_work_passed"]
                 and case["physical_affine_rest_passed"] is not False and case["constant_tracer_geometry_passed"] is not False
                 and abs(case["global_inventory_residual_kg_per_s"]) <= case["global_inventory_floor_kg_per_s"]
                 and abs(case["global_volume_residual_m3_per_s"]) <= case["global_volume_floor_m3_per_s"] for case in cases)
    report = {"scope": "independently assembled consistent kinetic matrix and NumPy weak RHS; instantaneous only",
              "backend": jax.default_backend(), "devices": [str(device) for device in jax.devices()],
              "jax": jax.__version__, "numpy": np.__version__, "python": sys.version,
              "environment": {name: os.environ.get(name) for name in ("JAX_PLATFORMS", "XLA_FLAGS")},
              "x64_enabled": bool(jax.config.x64_enabled), "source_hashes_unchanged": unchanged,
              "cases": cases, "original_domain_instantaneous_gates_passed": bool(passed),
              "production_promotion": False, "accepted_ocean_steps": 0}
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, allow_nan=False), flush=True)
    raise SystemExit(0 if unchanged and passed and jax.config.x64_enabled else 1)


if __name__ == "__main__":
    main()
