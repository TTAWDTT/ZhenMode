"""Freeze physical nodal pressure accuracy before any production integration."""
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
    parser.add_argument("--consistent-velocity", action="store_true")
    arguments = parser.parse_args()
    root = Path(__file__).resolve().parents[3]
    output = arguments.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    names = ["src/config.py", "src/jax_solver_global.py", "src/material_top.py", "pyproject.toml"]
    names.extend("research/experiments/material_rstar_coordinates/" + name for name in
                 ("kernel.py", "pressure_work.py", "nodal_mass.py", "weak_transport.py",
                  "pressure_accuracy.py", "pressure_accuracy_protocol.json", "pressure_accuracy_audit.py"))
    names.extend("research/experiments/material_rstar_coordinates/" + name for name in
                 ("weak_momentum.py", "consistent_velocity_protocol.json"))
    hashes = {}
    for name in names:
        destination = output / "sources" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(root / name, destination)
        hashes[name] = hashlib.sha256((root / name).read_bytes()).hexdigest()
    (output / "source_hashes.json").write_text(json.dumps(hashes, indent=2) + "\n", encoding="utf-8")
    sys.path.insert(0, str(root))
    sys.path.insert(0, str(root / "src"))
    import jax
    import numpy as np

    jax.config.update("jax_enable_x64", True)
    from research.experiments.material_rstar_coordinates.pressure_accuracy import (
        pressure_accuracy_diagnostic,
    )

    diagnostic, witnesses = pressure_accuracy_diagnostic(arguments.consistent_velocity)
    for name, arrays in witnesses.items():
        target = output / (name + ".npz")
        np.savez_compressed(target, **arrays)
        next(record for record in diagnostic["records"] if name == f"nx{record['longitude_count']}_nz{record['node_count']}").update(
            witness=target.name, witness_sha256=hashlib.sha256(target.read_bytes()).hexdigest())
    unchanged = all(hashlib.sha256((root / name).read_bytes()).hexdigest() == digest for name, digest in hashes.items())
    report = {"scope": "operator pressure accuracy only, not a full PDE or ocean step", **diagnostic,
              "backend": jax.default_backend(), "devices": [str(device) for device in jax.devices()],
              "runtime": {"python": sys.version, "platform": platform.platform(), "jax": jax.__version__, "numpy": np.__version__,
                          "environment": {name: os.environ.get(name) for name in ("JAX_PLATFORMS", "XLA_FLAGS")}},
              "source_hashes_unchanged": unchanged, "x64_enabled": bool(jax.config.x64_enabled)}
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, allow_nan=False), flush=True)
    passed = unchanged and jax.config.x64_enabled and all(report[field] for field in
                ("vertical_second_order_passed", "horizontal_second_order_passed", "independent_functionals_passed"))
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
