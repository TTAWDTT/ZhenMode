"""Independent geometry and actual barotropic time-average counterexamples."""
import argparse
import hashlib
import json
import subprocess
import sys
import types
from dataclasses import asdict, replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from ocean_solver.provenance.archives import current_source_files

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

from verify_debug_integration import make_smoke_fixture

from config import DEFAULT_CONFIG, R_EARTH, PhysicsConfig
from grid import GlobalOceanGrid

FROZEN_KERNEL = "2042ce1"
FROZEN_SOURCE = subprocess.check_output(["git", "show", f"{FROZEN_KERNEL}:src/jax_solver_global.py"], cwd=ROOT)
frozen = types.ModuleType("frozen_volume_invariants")
frozen.__file__ = str(ROOT / "src/jax_solver_global.py")
sys.modules[frozen.__name__] = frozen
exec(compile(FROZEN_SOURCE, frozen.__file__, "exec"), frozen.__dict__)


def fixture(stair=False):
    nx, ny = 24, 16
    latitude = np.linspace(-30., 30., ny)
    cosine = np.cos(np.radians(latitude))
    depth = np.full((nx, ny), 50.)
    if stair:
        depth[6:12, 4:12] = 15.
    nodes = np.array([0., -5., -20., -50.])
    wet = (-nodes[None, None, :] <= depth[..., None]).astype(float)
    grid = GlobalOceanGrid(
        lon=np.arange(nx) * 360. / nx, lat=latitude,
        dx_2d=np.broadcast_to(R_EARTH * np.radians(360. / nx) * cosine, (nx, ny)).copy(),
        dy=R_EARTH * np.radians(latitude[1] - latitude[0]), cos_lat=cosine,
        f=np.zeros((nx, ny)), z=nodes, dz=-np.diff(nodes), nz=len(nodes), depth=depth,
        wet_mask=np.ones((nx, ny)), ocean_mask=np.ones((nx, ny), dtype=bool),
        land_mask=np.zeros((nx, ny)), wet_mask_3d=wet, nx=nx, ny=ny)
    physics = replace(PhysicsConfig(), nu_h=0., nu_v=0., nu_bi=0., kappa_h=0.,
                      kappa_v=0., kappa_bi=0., kappa_conv=0., kappa_gm=0., kappa_redi=0., r_bot=0.)
    params = frozen.make_solver_global(grid, physics, 3600., lambda_bulk=0., mode_split=True,
              dt_bt=150., polar_cap_rows=0, polar_cap_taper=0, return_params=True)[3]
    return grid, physics, params


def weighted_norm(field, area):
    return float(np.sqrt(np.sum(np.asarray(field, dtype=np.float64) ** 2 * area)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bathy", default=DEFAULT_CONFIG.bathymetry_file)
    parser.add_argument("--out", default="results/industrial_alignment/volume_transport_invariants.json")
    args = parser.parse_args()
    sources = [Path(__file__).resolve(), Path(__file__).with_name("protocol.md"),
               ROOT / "scripts/verify_debug_integration.py", ROOT / "src/grid.py", ROOT / "src/config.py"]
    report = {"scope": "read_only_missing_invariants_not_a_physical_fix_or_climate_qualification", "status": "running",
              "provenance": {"frozen_kernel": FROZEN_KERNEL,
                  "frozen_kernel_sha256": hashlib.sha256(FROZEN_SOURCE).hexdigest(),
                  "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                  "git_status": subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).splitlines(),
                  "source_sha256": {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in current_source_files(ROOT, sources).items()},
                  "bathymetry_sha256": hashlib.sha256(Path(args.bathy).read_bytes()).hexdigest(),
                  "jax_version": jax.__version__, "backend": jax.default_backend()}, "ideal": []}
    grid, physics, params = fixture()
    np.testing.assert_allclose(np.asarray(params.dz_node).ravel(), [5., 10., 22.5, 30.], rtol=0., atol=1e-12)
    widths = np.asarray(grid.dz)
    trapezoid = np.concatenate([widths[:1] / 2., (widths[:-1] + widths[1:]) / 2., widths[-1:] / 2.])
    np.testing.assert_allclose(trapezoid, [2.5, 10., 22.5, 15.], rtol=0., atol=1e-12)
    report["analytic_weights"] = {"node_m": np.asarray(params.dz_node).ravel().tolist(),
        "barotropic_trapezoid_m": trapezoid.tolist(), "physical_depth_m": 50.,
        "node_column_depth_m": float(jnp.sum(params.dz_node)), "barotropic_depth_m": float(params.H_sw),
        "excess_node_column_m": float(jnp.sum(params.dz_node) - params.H_sw), "physics": asdict(physics)}
    for stair in (False, True):
        grid, _, params = fixture(stair)
        phase = 2. * np.pi * np.arange(grid.nx)[:, None, None] / grid.nx
        profile = np.array([1., .5, -.25, -1.])[None, None, :]
        velocity_x = jnp.asarray(np.sin(phase) * profile * grid.wet_mask_3d)
        velocity_y = jnp.zeros_like(velocity_x)
        native = np.asarray(frozen._column_divergence(velocity_x, velocity_y, params))
        averaged = frozen._barotropic_velocity(velocity_x, velocity_y, params)
        barotropic = np.asarray(params.H_sw * frozen._divergence_conservative(*averaged, params))
        area = grid.dx_2d * grid.dy
        if not stair:
            analytic_divergence = np.cos(phase[..., 0]) * np.sin(2. * np.pi / grid.nx) / grid.dx_2d
            np.testing.assert_allclose(native, -25.625 * analytic_divergence, rtol=5e-13, atol=1e-20)
            np.testing.assert_allclose(barotropic, -13.125 * analytic_divergence, rtol=5e-13, atol=1e-20)
        report["ideal"].append({"stair": stair, "native_column_divergence_units": "m/s",
            "relative_native_vs_barotropic_l2_difference": weighted_norm(native - barotropic, area) / weighted_norm(native, area),
            "maximum_local_difference_m_per_s": float(np.max(np.abs(native - barotropic))),
            "input_sha256": hashlib.sha256(np.asarray(velocity_x).tobytes() + np.asarray(velocity_y).tobytes() + grid.depth.tobytes()).hexdigest()})
    grid, _, params = fixture()
    phase = 2. * jnp.pi * jnp.arange(grid.nx)[:, None] / grid.nx
    initial_eta = jnp.broadcast_to(.2 * jnp.cos(phase), (grid.nx, grid.ny))
    initial_x = jnp.broadcast_to(.1 * jnp.sin(phase), initial_eta.shape)
    initial_y = jnp.zeros_like(initial_x)

    @jax.jit
    def subcycles():
        def advance(carry, unused):
            eta, velocity_x, velocity_y = carry
            updated = frozen._free_surface_step_fd(eta, velocity_x, velocity_y, params, dt_half=params.dt_bt)
            return updated, (velocity_x, velocity_y)

        return jax.lax.scan(advance, (initial_eta, initial_x, initial_y), None, length=params.n_subcyc)

    (eta, final_x, final_y), old_velocities = subcycles()
    mean_velocities = tuple(jnp.mean(field, axis=0) for field in old_velocities)
    actual_tendency = (eta - initial_eta) / params.dt
    matched = -params.H_sw * frozen._divergence_conservative(*mean_velocities, params)
    endpoint = -params.H_sw * frozen._divergence_conservative(final_x, final_y, params)
    area = grid.dx_2d * grid.dy
    norm = weighted_norm(actual_tendency, area)
    match_error = weighted_norm(actual_tendency - matched, area) / norm
    if match_error > 1e-12:
        raise ValueError(f"actual substep-mean identity failed: {match_error}")
    report["actual_subcycles"] = {"count": params.n_subcyc, "dt_bt_seconds": params.dt_bt,
        "substep_mean_continuity_relative_residual": match_error,
        "endpoint_continuity_relative_residual": weighted_norm(actual_tendency - endpoint, area) / norm,
        "disabled": ["coriolis", "wind", "bottom_drag", "sponge", "eta_restore", "polar_cap"],
        "input_sha256": hashlib.sha256(np.asarray(initial_eta).tobytes() + np.asarray(initial_x).tobytes()).hexdigest()}
    real_grid, real_physics, *_ = make_smoke_fixture(2., args.bathy, 2e14)
    real_params = frozen.make_solver_global(real_grid, real_physics, 600., return_params=True)[3]
    wet = np.asarray(real_params.wet_mask_z)
    intervals = np.asarray(real_grid.dz)
    weights = np.concatenate([intervals[:1] / 2., (intervals[:-1] + intervals[1:]) / 2., intervals[-1:] / 2.])
    depths = {"node_proxy": np.sum(np.asarray(real_params.dz_node) * wet, axis=-1),
              "barotropic_masked_node_trapezoid": np.sum(weights * wet, axis=-1),
              "density_pressure_both_wet_interfaces": np.sum(intervals * wet[..., :-1] * wet[..., 1:], axis=-1),
              "bathymetry": real_grid.depth}
    area = real_grid.dx_2d * real_grid.dy * real_grid.wet_mask
    active = np.asarray(real_grid.wet_mask) > .5
    depth_statistics = {}
    for name, depth in depths.items():
        difference = depth - depths["bathymetry"]
        depth_statistics[name] = {"domain_volume_m3": float(np.sum(depth * area)),
            "domain_volume_difference_m3": float(np.sum(difference * area)),
            "column_depth_difference_m_percentiles_0_5_50_95_100": np.percentile(difference[active], [0, 5, 50, 95, 100]).tolist()}
    report["real_geometry"] = {"grid": [real_grid.nx, real_grid.ny, real_grid.nz],
        "latitude_scope": "truncated_plus_minus_65_degrees_not_full_global_ocean",
        "columns_deeper_than_last_node": int(np.count_nonzero(active & (real_grid.depth > -real_grid.z[-1]))),
        "wet_columns": int(np.count_nonzero(active)), "depths": depth_statistics}
    report["status"] = "complete_counterexamples_not_repaired"
    destination = ROOT / args.out
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
