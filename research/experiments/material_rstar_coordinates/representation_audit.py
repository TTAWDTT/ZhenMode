"""Freeze the bounded bed/mass pressure-interface localization matrix."""
import argparse
import hashlib
import json
import platform
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
    names = ["tests/test_rstar_representation.py", "tests/test_rstar_nodal_mass.py", "tests/test_rstar_pressure_work.py", "tests/test_rstar_metric_controls.py",
             "tests/_helpers.py", "src/config.py", "src/grid.py", "src/jax_solver_global.py", "pyproject.toml",
             "research/experiments/material_rstar_coordinates/kernel.py",
             "research/experiments/material_rstar_coordinates/pressure_work.py",
             "research/experiments/material_rstar_coordinates/nodal_mass.py",
             "research/experiments/material_rstar_coordinates/representation_protocol.json",
             "research/experiments/material_rstar_coordinates/representation_audit.py"]
    hashes = {}
    from ocean_solver.provenance.archives import current_source_files
    files = current_source_files(root, names)
    for name, source in files.items():
        target = output / "sources" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        hashes[name] = hashlib.sha256(source.read_bytes()).hexdigest()
    (output / "source_hashes.json").write_text(json.dumps(hashes, indent=2) + "\n", encoding="utf-8")
    for directory in (root, root / "src", root / "tests"):
        sys.path.insert(0, str(directory))
    import jax
    import numpy as np

    from tests.support.rstar.representation import _representation_diagnostic

    cases = []
    for truncate in (False, True):
        for consistent in (False, True):
            for flat in (False, True):
                result, arrays = _representation_diagnostic(truncate, consistent, flat)
                name = f"bed{int(truncate)}_consistent{int(consistent)}_flat{int(flat)}.npz"
                np.savez_compressed(output / name, **arrays)
                result["witness"] = name
                result["witness_sha256"] = hashlib.sha256((output / name).read_bytes()).hexdigest()
                cases.append(result)
    unchanged = all(hashlib.sha256(files[name].read_bytes()).hexdigest() == digest for name, digest in hashes.items())
    report = {"scope": "representation-only instantaneous localization, not original-domain acceptance",
              "backend": jax.default_backend(), "devices": [str(device) for device in jax.devices()],
              "runtime": {"python": sys.version, "platform": platform.platform(), "jax": jax.__version__, "numpy": np.__version__},
              "x64_enabled": bool(jax.config.x64_enabled), "source_hashes_unchanged": unchanged,
              "cases": cases, "production_promotion": False, "accepted_ocean_steps": 0}
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, allow_nan=False), flush=True)
    raise SystemExit(0 if unchanged and jax.config.x64_enabled else 1)


if __name__ == "__main__":
    main()
