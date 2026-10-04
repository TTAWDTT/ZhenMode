"""Bounded CPU reproduction of material-top rejection, never a qualification run.

The host-side diagnosis reads the existing result without changing its gate,
state, budget or timestep. Geometry takes precedence over capacity in the
*diagnostic* ordering; this is not a claim about chronological operator failure.
"""
import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from ocean_solver.provenance.archives import current_source_files

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from ocean_solver.config.definitions import PhysicsConfig  # noqa: E402
from ocean_solver.geometry.types import GlobalOceanGrid
from ocean_solver.model.factory import make_solver_global  # noqa: E402
from zhenmode_research.candidates.material.solver import (  # noqa: E402
    CFL_LIMIT,
    CONTINUITY_TOLERANCE_M,
    FACE_TOLERANCE_M2_PER_S,
    make_material_top_step,
)

SCHEME = "actual_geometry_v2"
STAGES = ("linear_first", "nonlinear", "linear_second")
LIMITS = {
    "transport_outflow_fraction_max": CFL_LIMIT,
    "convective_fraction_max": CFL_LIMIT,
    "linear_diffusion_fraction_max": CFL_LIMIT,
    "nonlinear_combined_fraction_max": CFL_LIMIT,
    "local_continuity_residual_max_m": CONTINUITY_TOLERANCE_M,
    "tracer_face_mismatch_max_m2_per_s": FACE_TOLERANCE_M2_PER_S,
    "unpaired_eta_filter_max_m": CONTINUITY_TOLERANCE_M,
    "local_inventory_roundoff_ratio_max": 1.,
    "ice_abs_max_m": 0.,
    "velocity_abs_max_m_per_s": 10.,
    "eta_abs_max_m": 15.,
}


def geometry_witness(eta, params):
    """Independent NumPy reconstruction; do not call material_thickness."""
    wet = np.asarray(params.wet_mask_z)
    if not np.isin(wet, [0., 1.]).all() or not wet.any():
        raise ValueError("diagnosis requires a nonempty binary wet mask")
    active = wet > 0.
    widths = np.broadcast_to(np.asarray(params.dz_node), wet.shape)
    height = widths + np.asarray(eta)[..., None] * np.asarray(params.surface_mask)
    if not np.isfinite(height[active]).all():
        return {"finite": False}
    index = np.unravel_index(np.argmin(np.where(active, height, np.inf)), wet.shape)
    column = index[:2]
    reference = np.sum(np.where(active, widths, 0.), axis=-1)
    actual_column = np.sum(np.where(active, height, 0.), axis=-1)
    ocean = active.any(axis=-1)
    return {
        "finite": True,
        "minimum_wet_thickness_m": float(height[index]),
        "minimum_index_xyz": [int(i) for i in index],
        "eta_at_minimum_m": float(np.asarray(eta)[column]),
        "reference_column_at_minimum_m": float(reference[column]),
        "actual_column_at_minimum_m": float(actual_column[column]),
        "minimum_actual_column_m": float(actual_column[ocean].min()),
        "nonpositive_wet_nodes": int(np.sum(active & (height <= 0.))),
        "nonpositive_columns": int(np.sum(ocean & (actual_column <= 0.))),
    }


def diagnose(result, initial, params, maximum):
    """Explain a completed actual_geometry_v2 result; never authorize a step."""
    if isinstance(maximum, bool) or not isinstance(maximum, int) or maximum < 1:
        raise ValueError("maximum must be the positive integer used for this attempt")
    checks = {key: np.asarray(value).item() for key, value in result.checks.items()}
    stages = STAGES + (("momentum",) if any(key.startswith("momentum_") for key in checks) else ())
    limits = dict(LIMITS)
    if "momentum" in stages:
        limits["momentum_diffusion_fraction_max"] = CFL_LIMIT
    required_keys = set(limits) | {"finite", "minimum_wet_thickness_m"}
    required_keys |= {f"{stage}_{suffix}" for stage in stages
                      for suffix in ("required_subcycles", "active_subcycles", "schedule_supported")}
    if missing := required_keys - checks.keys():
        raise ValueError(f"incomplete actual_geometry_v2 checks: {sorted(missing)}")
    before = geometry_witness(initial.eta, params)
    attempted = geometry_witness(result.attempted_state.eta, params)
    finite = bool(checks["finite"]) and before["finite"] and attempted["finite"]
    capacity = [stage for stage in stages if checks[f"{stage}_required_subcycles"] > maximum]
    unsupported = [stage for stage in stages if not checks[f"{stage}_schedule_supported"]]
    failed = [key for key, limit in limits.items()
              if not np.isfinite(checks[key]) or checks[key] > limit]
    accepted = bool(result.valid)
    if accepted:
        blocker = "accepted"
    elif not finite:
        blocker = "nonfinite"
    elif before["nonpositive_columns"] or attempted["nonpositive_columns"]:
        blocker = "nonpositive_column"
    elif (before["nonpositive_wet_nodes"] or attempted["nonpositive_wet_nodes"]
          or checks["minimum_wet_thickness_m"] <= 0.):
        blocker = "nonpositive_node_with_water_remaining"
    elif capacity:
        blocker = "subcycle_capacity"
    elif unsupported:
        blocker = "unsupported_schedule"
    else:
        blocker = "other_rejection"
    rollback = all(np.asarray(old).dtype == np.asarray(new).dtype
                   and np.asarray(old).shape == np.asarray(new).shape
                   and np.asarray(old).tobytes() == np.asarray(new).tobytes()
                   for old, new in zip(initial, result.state, strict=True))
    return {
        "accepted": accepted, "diagnostic_blocker": blocker,
        "capacity_exceeded_stages": capacity, "unsupported_stages": unsupported,
        "failed_scalar_checks": failed, "checks": checks,
        "initial_geometry": before, "attempted_geometry": attempted,
        "rejected_state_rollback_bytes_equal": rollback if not accepted else None,
        "attempt_budget_is_accepted_increment": accepted,
    }


def synthetic_fixture():
    """8x8x4, all-wet 50 m column; deliberately thin top, prescribed zonal flow."""
    nx = ny = 8
    lat = np.linspace(-30., 30., ny)
    lon = (np.arange(nx) + .5) * 360. / nx
    cosine = np.cos(np.deg2rad(lat))
    grid = GlobalOceanGrid(
        lon=lon, lat=lat,
        dx_2d=np.broadcast_to(6.371e6 * cosine * np.deg2rad(360. / nx), (nx, ny)).copy(),
        dy=float(6.371e6 * np.deg2rad(lat[1] - lat[0])), cos_lat=cosine,
        f=np.zeros((nx, ny)), z=np.array([0., -5., -20., -50.]),
        dz=np.array([5., 15., 30.]), nz=4, nx=nx, ny=ny,
        depth=np.full((nx, ny), 50.), wet_mask=np.ones((nx, ny)),
        ocean_mask=np.ones((nx, ny), bool), land_mask=np.zeros((nx, ny), bool),
        wet_mask_3d=np.ones((nx, ny, 4)))
    physics = replace(PhysicsConfig(), nu_h=0., nu_v=0., nu_bi=0., kappa_h=0.,
                      kappa_v=0., kappa_bi=0., kappa_conv=.0015, kappa_gm=0.,
                      kappa_redi=0., r_bot=0.)
    _, initialize, _, params, _ = make_solver_global(
        grid, physics, 10., mode_split=True, dt_bt=5., dtype="float64", return_params=True,
        column_geometry="nodal_dual_v1", conservative_kv=True, localize_conv=True,
        polar_cap_rows=0, polar_cap_taper=0, process_time_scheme="symmetric_fast_v3",
        match_barotropic_transport=True, monotone_adv=True, use_scan=True)
    state = initialize()
    state = state._replace(
        T=jnp.full_like(state.T, 15.), S=jnp.full_like(state.S, 35.),
        eta=jnp.full_like(state.eta, -2.49988),
        u=jnp.broadcast_to(jnp.sin(2. * jnp.pi * jnp.arange(nx)[:, None, None] / nx), state.u.shape))
    return grid, params, state


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(output):
    import resource  # POSIX process receipt is required only by this CLI.

    if jax.default_backend() != "cpu":
        raise ValueError("this bounded reproduction requires JAX_PLATFORMS=cpu")
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    grid, params, state = synthetic_fixture()
    inputs = {f"state_{key}": np.asarray(value) for key, value in state._asdict().items()}
    inputs.update({f"grid_{key}": np.asarray(value) for key, value in vars(grid).items()})
    # Include all factory parameters, including defaults, in the frozen input packet.
    inputs.update({f"params_{key}": np.asarray(value) for key, value in params._asdict().items()
                   if value is not None})
    config = {"subcycle_scheme": SCHEME, "capacity_original": 128, "capacity_counterfactual": 256,
              "seed": None, "initialization": "deterministic analytic fields; no random sampling",
              "none_parameters": [key for key, value in params._asdict().items() if value is None],
              "scalar_gate_limits": LIMITS, "positive_thickness_gate_m": 0.}
    (output / "config.json").write_text(json.dumps(config, indent=2, allow_nan=False) + "\n")
    np.savez_compressed(output / "inputs.npz", **inputs)
    step128 = make_material_top_step(params, subcycle_scheme=SCHEME, max_subcycles=128)
    step256 = make_material_top_step(params, subcycle_scheme=SCHEME, max_subcycles=256)
    first = step128(state)
    lifted = step256(state)  # independent capacity diagnostic, not continuation of 128
    next_result = step256(lifted.state)
    cases = []
    for name, initial, result, maximum in (
        ("original_capacity", state, first, 128),
        ("capacity_only_counterfactual", state, lifted, 256),
        ("next_complete_attempt", lifted.state, next_result, 256),
    ):
        report = diagnose(result, initial, params, maximum)
        report.update(name=name, maximum_subcycles=maximum)
        arrays = {}
        for label, values in (("initial", initial._asdict()), ("attempted", result.attempted_state._asdict()),
                              ("returned", result.state._asdict()), ("budget", result.budget),
                              ("checks", result.checks)):
            arrays.update({f"{label}_{key}": np.asarray(value) for key, value in values.items()})
        np.savez_compressed(output / f"{name}.npz", **arrays)
        cases.append(report)
    reproduced = ([case["diagnostic_blocker"] for case in cases]
                  == ["subcycle_capacity", "accepted", "nonpositive_node_with_water_remaining"]
                  and cases[0]["rejected_state_rollback_bytes_equal"]
                  and cases[2]["rejected_state_rollback_bytes_equal"])
    source_paths = sorted((ROOT / "src").glob("*.py")) + [Path(__file__).resolve(),
                                                                  ROOT / "tests/test_material_failure_replay.py"]
    report = {
        "schema_version": 1, "synthetic_reproduction_matched": bool(reproduced),
        "qualification_passed": False, "real_1degree_replayed": False,
        "production_promotion": False, "real_ocean_steps_accepted": 0,
        "synthetic_full_step_attempts": 3, "synthetic_full_step_accepts": sum(c["accepted"] for c in cases),
        "checkout_head_at_execution": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "worktree_status_at_execution": subprocess.check_output(["git", "status", "--short"], cwd=ROOT, text=True),
        "source_sha256": {name: sha256(p) for name, p in current_source_files(ROOT, source_paths).items()},
        "config_sha256": sha256(output / "config.json"),
        "packet_sha256": {p.name: sha256(p) for p in sorted(output.glob("*.npz"))},
        "environment": {"python": platform.python_version(), "jax": jax.__version__,
                        "numpy": np.__version__, "backend": jax.default_backend(),
                        "devices": [str(device) for device in jax.devices()]},
        "wall_seconds_including_compilation": time.monotonic() - started,
        "process_peak_rss_kib_linux": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "cost_scope": "local CPU only; no paid API, GPU or ocean-data download; not a throughput benchmark",
        "cases": cases,
    }
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    # A successfully reproduced physical failure still must not exit as a qualification pass.
    return 1 if reproduced else 2


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="new evidence directory")
    args = parser.parse_args()
    raise SystemExit(run(args.output))
