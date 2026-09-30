"""Archive pressure-work versus equilibrium qualification, including failures."""
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
    arguments = parser.parse_args()
    root = Path(__file__).resolve().parents[3]
    output = arguments.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    names = ["tests/test_rstar_pressure_work.py", "tests/test_rstar_metric_controls.py", "tests/_helpers.py", "pyproject.toml",
             "src/jax_solver_global.py", "src/config.py", "src/grid.py",
             "research/experiments/material_rstar_coordinates/kernel.py",
             "research/experiments/material_rstar_coordinates/pressure_work.py",
             "research/experiments/material_rstar_coordinates/pressure_protocol.json",
             "research/experiments/material_rstar_coordinates/pressure_audit.py"]
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
    from test_rstar_pressure_work import (
        _candidate_diagnostics,
        _case,
        _numpy_rates,
        _vertical_refinement,
    )

    from config import G_EARTH, RHO_0
    from research.experiments.material_rstar_coordinates.kernel import (
        coordinate_pressure_gradient,
        hydrostatic_pressure,
    )
    from research.experiments.material_rstar_coordinates.pressure_work import (
        energy_adjoint_force,
        paired_chain_rule_force,
        potential_conjugates,
    )

    cases = [_candidate_diagnostics(stairs, flat, kind) for stairs in (False, True) for flat in (False, True)
             for kind in ("zero", "constant", "random", "affine")]
    candidate_status = {}
    for name in cases[0]["candidate_gates"]:
        work = all(case["candidate_gates"][name]["pressure_work_passed"] for case in cases)
        rest = all(case["candidate_gates"][name]["rest_gate_passed"] for case in cases if case["candidate_gates"][name]["rest_gate_applicable"])
        candidate_status[name] = {"all_pressure_work_gates_passed": work, "all_rest_gates_passed": rest,
                                  "both_interface_gates_passed": work and rest}
    params, _, geometry, basis, surface, density, content, velocities = _case(True, True, "affine")
    pressure = hydrostatic_pressure(density, surface, geometry, params)
    rates = _numpy_rates(np.asarray(density), velocities, geometry, params)
    conjugates = potential_conjugates(content, surface, geometry, basis, params)
    arrays = {"density": density, "density_content_kg": content, "surface_m": surface,
              "area_m2": params.dx_2d * params.dy, "thickness_m": geometry.thickness,
              "pressure_pa": pressure, "dx_m": params.dx_2d, "dy_m": params.dy,
              "reference_density_kg_per_m3": RHO_0, "gravity_m_per_s2": G_EARTH,
              "node_depth_m": geometry.node_depth, "reference_basis_mean_depth_m": basis.mean_depth,
              "velocity_x_m_per_s": velocities[0], "velocity_y_m_per_s": velocities[1],
              "independent_density_content_rate_kg_per_s": rates[0], "independent_surface_rate_m_per_s": rates[1],
              "density_content_energy_conjugate_m2_per_s2": conjugates[0], "surface_energy_conjugate_joules_per_m": conjugates[1]}
    for name, forces in (("existing", coordinate_pressure_gradient(pressure, density, geometry, params)),
                         ("width_weighted", paired_chain_rule_force(pressure, density, geometry, params)),
                         ("energy_adjoint", energy_adjoint_force(density, content, surface, geometry, basis, params))):
        arrays[f"{name}_force_x_m_per_s2"], arrays[f"{name}_force_y_m_per_s2"] = forces
    np.savez_compressed(output / "affine_stair_witness.npz", **{name: np.asarray(value) for name, value in arrays.items()})
    refinement = _vertical_refinement()
    unchanged = all(hashlib.sha256((root / name).read_bytes()).hexdigest() == digest for name, digest in hashes.items())
    report = {"scope": "instantaneous original-node rstar pressure and declared coordinate-band transport, not an ocean step",
              "backend": jax.default_backend(), "devices": [str(device) for device in jax.devices()],
              "runtime": {"python": sys.version, "platform": platform.platform(), "jax": jax.__version__, "numpy": np.__version__,
                          "environment": {name: os.environ.get(name) for name in ("JAX_PLATFORMS", "XLA_FLAGS")}},
              "x64_enabled": bool(jax.config.x64_enabled), "source_hashes_unchanged": unchanged,
              "candidate_status": candidate_status, "cases": cases, "production_promotion": False,
              "witness_sha256": hashlib.sha256((output / "affine_stair_witness.npz").read_bytes()).hexdigest(),
              "vertical_refinement": refinement,
              "pressure_interface_qualified": bool(unchanged and jax.config.x64_enabled and any(value["both_interface_gates_passed"] for value in candidate_status.values())),
              "not_qualified": "physical overlap, complete fast/slow and momentum advection/energy, stage time, ice, restart, actual1degree, century/climate/forecast or industrial comparison"}
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, allow_nan=False), flush=True)
    raise SystemExit(0 if report["pressure_interface_qualified"] else 1)


if __name__ == "__main__":
    main()
