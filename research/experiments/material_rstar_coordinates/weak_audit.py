"""Freeze joint weak-transport work/rest controls, including original-bed fails."""
import argparse
import hashlib
import json
import os
import platform
import shutil
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--complete-bed", action="store_true")
    arguments = parser.parse_args()
    root = Path(__file__).resolve().parents[3]
    output = arguments.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    names = ["tests/test_rstar_weak_transport.py", "tests/test_rstar_bed_completion.py", "tests/test_rstar_representation.py", "tests/test_rstar_pressure_work.py",
             "tests/test_rstar_metric_controls.py", "tests/_helpers.py", "src/config.py", "src/grid.py", "src/jax_solver_global.py", "pyproject.toml"]
    names.extend("research/experiments/material_rstar_coordinates/" + name for name in
                 ("kernel.py", "pressure_work.py", "nodal_mass.py", "weak_transport.py", "weak_oracle.py", "weak_protocol.json", "weak_audit.py",
                  "bed_completion.py", "bed_completion_protocol.json"))
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

    from tests.support.rstar.weak_transport import _weak_diagnostic

    cases = []
    for truncate in ((False,) if arguments.complete_bed else (False, True)):
        for stairs in (False, True):
            for flat in (False, True):
                for kind in ("zero", "constant", "affine", "random"):
                    result, arrays = _weak_diagnostic(truncate, stairs, flat, kind, arguments.complete_bed)
                    name = f"bed{int(truncate)}_stairs{int(stairs)}_flat{int(flat)}_{kind}.npz"
                    np.savez_compressed(output / name, **arrays)
                    result["witness"] = name
                    result["witness_sha256"] = hashlib.sha256((output / name).read_bytes()).hexdigest()
                    cases.append(result)
    unchanged = all(hashlib.sha256(files[name].read_bytes()).hexdigest() == digest for name, digest in hashes.items())
    original = [case for case in cases if not case["truncate"]]
    original_passed = all(case["rhs_numpy_passed"] and case["eta_numpy_passed"] and case["pressure_work_passed"]
                          and abs(case["global_inventory_residual_kg_per_s"]) <= case["global_inventory_floor_kg_per_s"]
                          and abs(case["global_volume_residual_m3_per_s"]) <= case["global_volume_floor_m3_per_s"]
                          and case["constant_tracer_geometry_passed"] is not False
                          and (not case["rest_applicable"] or case["physical_affine_rest_passed"]) for case in original)
    report = {"scope": "instantaneous joint weak transport and pressure, not a bounded ocean step",
              "backend": jax.default_backend(), "devices": [str(device) for device in jax.devices()],
              "runtime": {"python": sys.version, "platform": platform.platform(), "jax": jax.__version__, "numpy": np.__version__,
                          "environment": {name: os.environ.get(name) for name in ("JAX_PLATFORMS", "XLA_FLAGS")}},
              "x64_enabled": bool(jax.config.x64_enabled), "source_hashes_unchanged": unchanged, "cases": cases,
              "completed_bed_representation": arguments.complete_bed,
              "original_domain_instantaneous_gates_passed": bool(original_passed),
              "production_promotion": False, "accepted_ocean_steps": 0,
              "workspace": "dense local hat traces scale with horizontal columns times vertical-node-count squared; no global dense matrix, but full actual GPU working memory and throughput not qualified"}
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, allow_nan=False), flush=True)
    raise SystemExit(0 if unchanged and jax.config.x64_enabled and original_passed else 1)


if __name__ == "__main__":
    main()
