"""Freeze bounded consistent content controls, sources, raw arrays and failures."""
import argparse
import datetime
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
    arguments = parser.parse_args()
    root = Path(__file__).resolve().parents[3]
    output = arguments.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    started = datetime.datetime.now(datetime.timezone.utc).isoformat()
    names = ["src/config.py", "src/grid.py", "src/jax_solver_global.py", "src/material_top.py", "pyproject.toml"]
    names.extend("tests/" + name for name in ("_helpers.py", "test_rstar_weak_bounded.py", "test_rstar_weak_momentum.py",
                 "test_rstar_weak_transport.py", "test_rstar_metric_controls.py", "test_rstar_representation.py", "test_rstar_pressure_work.py"))
    names.extend("research/experiments/material_rstar_coordinates/" + name for name in
                 ("kernel.py", "nodal_mass.py", "pressure_work.py", "weak_transport.py", "weak_oracle.py", "bed_completion.py",
                  "weak_momentum.py", "pressure_accuracy.py", "sparse_diffusion.py", "weak_sparse.py", "weak_bounded.py",
                  "weak_bounded_protocol.json", "weak_bounded_audit.py"))
    hashes = {}
    for name in names:
        target = output / "sources" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(root / name, target)
        hashes[name] = hashlib.sha256((root / name).read_bytes()).hexdigest()
    (output / "source_hashes.json").write_text(json.dumps(hashes, indent=2) + "\n", encoding="utf-8")
    for folder in (root, root / "src", root / "tests"):
        sys.path.insert(0, str(folder))
    import jax
    import numpy as np
    from test_rstar_weak_bounded import _bounded_diagnostic

    cases = []
    for flat in (False, True):
        for kind in ("constant", "pulse", "signed", "heat"):
            record, arrays = _bounded_diagnostic(kind, flat)
            name = f"flat{int(flat)}_{kind}.npz"
            np.savez_compressed(output / name, **arrays)
            record.update(witness=name, witness_sha256=hashlib.sha256((output / name).read_bytes()).hexdigest())
            cases.append(record)
    unchanged = all(hashlib.sha256((root / name).read_bytes()).hexdigest() == digest for name, digest in hashes.items())
    passed = all(case["valid"] and case["independent_stage_passed"]
                 and abs(case["inventory_residual"]) <= case["inventory_64eps_floor"] for case in cases)
    report = {"scope": "prescribed nodal velocities: physical sparse transport, bounded consistent content and source controls; not a solved ocean step",
              "started_at_utc": started, "finished_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
              "backend": jax.default_backend(), "devices": [str(device) for device in jax.devices()],
              "python": sys.version, "platform": platform.platform(), "jax": jax.__version__, "numpy": np.__version__,
              "environment": {name: os.environ.get(name) for name in ("JAX_PLATFORMS", "XLA_FLAGS")},
              "x64_enabled": bool(jax.config.x64_enabled), "source_hashes_unchanged": unchanged,
              "cases": cases, "bounded_prescribed_controls_passed": bool(passed), "production_promotion": False,
              "accepted_ocean_steps": 0, "accepted_prescribed_tracer_stages": sum(case["valid"] for case in cases),
              "not_qualified": "full moving momentum/energy, complete source/diffusion suite, fast-slow or full-step PDE order/AD, restart, actual1degree integration, century/climate/forecast or industrial comparison"}
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, allow_nan=False), flush=True)
    raise SystemExit(0 if passed and unchanged and jax.config.x64_enabled else 1)


if __name__ == "__main__":
    main()
