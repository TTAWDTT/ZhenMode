"""Freeze and run sparse/moving controls, retaining failures and numeric properties."""
import argparse
import ast
import hashlib
import json
import os
import platform
import shutil
import sys
import xml.etree.ElementTree as element_tree
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    root = Path(__file__).resolve().parents[3]
    output = arguments.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    names = ["tests/test_rstar_sparse_diffusion.py", "tests/test_rstar_moving_content.py",
             "tests/test_rstar_metric_controls.py", "tests/_helpers.py", "pyproject.toml",
             "src/jax_solver_global.py", "src/config.py", "src/grid.py",
             "research/experiments/material_rstar_coordinates/kernel.py",
             "research/experiments/material_rstar_coordinates/dense_oracle.py",
             "research/experiments/material_rstar_coordinates/sparse_diffusion.py",
             "research/experiments/material_rstar_coordinates/moving_content.py",
             "research/experiments/material_rstar_coordinates/sparse_protocol.json",
             "research/experiments/material_rstar_coordinates/moving_protocol.json",
             "research/experiments/material_rstar_coordinates/sparse_audit.py"]
    hashes = {}
    for name in names:
        source = root / name
        target = output / "sources" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        hashes[name] = hashlib.sha256(source.read_bytes()).hexdigest()
    (output / "source_hashes.json").write_text(json.dumps(hashes, indent=2) + "\n", encoding="utf-8")
    sys.path.insert(0, str(root))
    import pytest

    result = int(pytest.main([str(root / names[0]), str(root / names[1]), "-q", "-o", "junit_family=legacy",
                              f"--junitxml={output / 'controls.xml'}"]))
    import jax
    import jaxlib
    import numpy as np

    cases = []
    for case in element_tree.parse(output / "controls.xml").findall(".//testcase"):
        properties = {}
        for item in case.findall("./properties/property"):
            value = item.attrib["value"]
            try:
                value = ast.literal_eval(value)
            except (ValueError, SyntaxError):
                pass
            properties[item.attrib["name"]] = value
        cases.append({"name": case.attrib["name"], "seconds": float(case.attrib["time"]),
                      "passed": not any(case.find(kind) is not None for kind in ("failure", "error", "skipped")),
                      "properties": properties})
    unchanged = all(hashlib.sha256((root / name).read_bytes()).hexdigest() == digest for name, digest in hashes.items())
    report = {"scope": "isolated sparse diffusion and prescribed-transport moving-content controls, not an ocean integration",
              "backend": jax.default_backend(), "devices": [str(device) for device in jax.devices()],
              "runtime": {"python": sys.version, "platform": platform.platform(), "jax": jax.__version__,
                          "jaxlib": jaxlib.__version__, "numpy": np.__version__,
                          "environment": {name: os.environ.get(name) for name in ("JAX_PLATFORMS", "JAX_ENABLE_X64", "XLA_FLAGS")}},
              "x64_enabled": bool(jax.config.x64_enabled), "pytest_exit_code": result,
              "source_hashes_unchanged": unchanged, "cases": cases,
              "control_gates_passed": bool(result == 0 and unchanged and jax.config.x64_enabled
                                           and len(cases) == 28 and all(case["passed"] for case in cases)),
              "production_promotion": False,
              "not_qualified": "moving-face overlap, momentum pressure-work/energy, hard convection gradients, ice, real1degree, restart byte gate, century/climate/forecast or industrial comparison"}
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, allow_nan=False), flush=True)
    raise SystemExit(0 if report["control_gates_passed"] else 1)


if __name__ == "__main__":
    main()
