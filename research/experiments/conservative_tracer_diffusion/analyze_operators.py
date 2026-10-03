"""Reproduce the registered diffusion budget fixture and spherical accuracy test."""
import argparse
import hashlib
import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from ocean_solver.provenance.archives import current_source_files

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tests")]

from ocean_solver.config.definitions import C_P, R_EARTH, RHO_0, PhysicsConfig
from ocean_solver.numerics.horizontal import _biharmonic_h
from ocean_solver.numerics.horizontal import _horizontal_biharmonic_tracer
from ocean_solver.numerics.horizontal import _horizontal_tracer_diffusion
from ocean_solver.numerics.horizontal import _laplacian_h
from ocean_solver.model.factory import make_solver_global
from tests.support.fd.horizontal_diffusion import _parameters
from tests.support.grid import all_wet_grid


def budget(tendency, volume):
    values = np.asarray(tendency)
    change = float(np.sum(values * volume))
    activity = float(np.sum(np.abs(values) * volume))
    return {"heat_source_W": change * RHO_0 * C_P,
            "signed_to_absolute_budget": change / activity if activity else None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="results/industrial_alignment/diffusion_operator_review.json")
    args = parser.parse_args()
    rows = []
    for land in (False, True):
        grid = all_wet_grid(nx=24, ny=32, nz=4)
        wet = grid.wet_mask_3d.copy()
        if land:
            wet[4:9, 15:20, :] = 0.
            wet[12:17, 8:13, 2:] = 0.
        grid = replace(grid, wet_mask_3d=wet, wet_mask=wet[:, :, 0],
                       ocean_mask=wet[:, :, 0].astype(bool), land_mask=1. - wet[:, :, 0])
        physics = replace(PhysicsConfig(), nu_h=0., nu_v=0., nu_bi=0., kappa_h=100.,
                          kappa_bi=0., kappa_v=0., kappa_conv=0.)
        forcing = tuple(np.zeros((grid.nx, grid.ny)) for _ in range(3))
        _, _, _, params, _ = make_solver_global(
            grid, physics, 60., forcing=forcing, lambda_bulk=0.,
            polar_cap_rows=0, polar_cap_taper=0, return_params=True)
        tracer = jnp.asarray(np.random.default_rng(2718).uniform(5., 25., wet.shape))
        volume = grid.dx_2d[:, :, None] * grid.dy * np.asarray(params.dz_node) * wet
        rows.append({
            "land_and_dry_bottom": land,
            "legacy_laplacian": budget(params.kappa_h * _laplacian_h(tracer, params), volume),
            "conservative_laplacian": budget(_horizontal_tracer_diffusion(tracer, params), volume),
            "legacy_biharmonic": budget(-1e12 * _biharmonic_h(tracer, params), volume),
            "conservative_biharmonic": budget(-1e12 * _horizontal_biharmonic_tracer(tracer, params), volume),
        })
    errors = []
    for ny in (24, 48, 96):
        grid, params, volume = _parameters(50., ny=ny, nx=2 * ny)
        longitude = 2. * np.pi * np.arange(grid.nx) / grid.nx
        tracer = np.broadcast_to(np.sin(longitude)[:, None, None]
                                 * grid.cos_lat[None, :, None], volume.shape)
        exact = -2. * params.kappa_h * tracer / R_EARTH ** 2
        computed = np.asarray(_horizontal_tracer_diffusion(jnp.asarray(tracer), params))
        interior = (slice(None), slice(4, -4), slice(None))
        errors.append(float(np.sqrt(np.sum((computed[interior] - exact[interior]) ** 2 * volume[interior])
                                    / np.sum(exact[interior] ** 2 * volume[interior]))))
    report = {
        "scope": "isolated_operator_not_climate_drift",
        "seed": 2718,
        "budget_fixture": "tests._helpers.all_wet_grid(24,32,4), registered land/dry masks",
        "budgets": rows,
        "spherical_relative_errors": errors,
        "spherical_refinement_ratios": [errors[index] / errors[index + 1] for index in range(2)],
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "source_sha256": {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in current_source_files(ROOT, [ROOT / "src/jax_solver_global.py", ROOT / "tests/_helpers.py",
                                       ROOT / "tests/test_horizontal_tracer_diffusion.py", Path(__file__).resolve()]).items()},
    }
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
