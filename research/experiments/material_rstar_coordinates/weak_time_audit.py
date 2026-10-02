"""Freeze prescribed consistent-content time controls, including failed order gates."""
import argparse
import datetime
import hashlib
import json
import os
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[3]
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    names = ["src/config.py", "src/grid.py", "src/jax_solver_global.py", "src/material_top.py", "pyproject.toml"]
    names.extend("tests/" + name for name in ("_helpers.py", "test_rstar_weak_time.py", "test_rstar_weak_bounded.py",
                 "test_rstar_weak_momentum.py", "test_rstar_weak_transport.py", "test_rstar_metric_controls.py",
                 "test_rstar_representation.py", "test_rstar_pressure_work.py"))
    names.extend("research/experiments/material_rstar_coordinates/" + name for name in
                 ("kernel.py", "nodal_mass.py", "pressure_work.py", "weak_transport.py", "weak_oracle.py", "bed_completion.py",
                  "weak_momentum.py", "pressure_accuracy.py", "sparse_diffusion.py", "weak_sparse.py", "weak_bounded.py",
                  "weak_bounded_protocol.json", "weak_time.py", "weak_time_protocol.json", "weak_time_audit.py"))
    from ocean_solver.provenance.archives import current_source_files
    files = current_source_files(root, names)
    hashes = {name: hashlib.sha256(source.read_bytes()).hexdigest() for name, source in files.items()}
    for name, source in files.items():
        destination = output / "sources" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(files[name].read_bytes())
    (output / "source_hashes.json").write_text(json.dumps(hashes, indent=2) + "\n")
    for folder in (root, root / "src", root / "tests"):
        sys.path.insert(0, str(folder))

    import jax
    import numpy as np

    from tests.support.rstar.weak_time import _moving_time_diagnostic, _time_diagnostic

    report = {"started_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
              "backend": jax.default_backend(), "devices": [str(device) for device in jax.devices()],
              "x64_enabled": bool(jax.config.x64_enabled), "jax": jax.__version__, "numpy": np.__version__,
              "environment": {name: os.environ.get(name) for name in ("JAX_PLATFORMS", "XLA_FLAGS")},
              "scope": "prescribed fields time controls only; moving active limiter gates remain independently classified",
              "accepted_ocean_steps": 0, "production_promotion": False, "cases": []}
    for kind in ("fourier", "signed", "pulse"):
        if kind == "fourier":
            record, arrays = _time_diagnostic(with_arrays=True)
            passed = min(record["orders"]["heun"]) >= 1.9 and max(record["orders"]["euler"]) < 1.5
        else:
            record, arrays = _moving_time_diagnostic(kind)
            passed = record["reference_resolved"] and min(record["orders"]) >= 1.9
        witness = output / (kind + ".npz")
        np.savez_compressed(witness, **arrays)
        record.update(kind=kind, time_order_passed=bool(passed), witness=witness.name,
                      witness_sha256=hashlib.sha256(witness.read_bytes()).hexdigest())
        report["cases"].append(record)
        (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        print(json.dumps(record, allow_nan=False), flush=True)
    report["source_hashes_unchanged"] = all(hashlib.sha256(files[name].read_bytes()).hexdigest() == value for name, value in hashes.items())
    report["finished_at_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    report["all_registered_time_gates_passed"] = all(case["time_order_passed"] for case in report["cases"])
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, indent=2, allow_nan=False), flush=True)
    raise SystemExit(0 if report["all_registered_time_gates_passed"] and report["source_hashes_unchanged"] and report["x64_enabled"] else 1)


if __name__ == "__main__":
    main()
