"""Independently diagnose rejected float32 physical surface-volume identities."""
import hashlib
import json
import subprocess
import sys
import types
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

import jax
import jax.numpy as jnp
import numpy as np

from config import DEFAULT_CONFIG, GlobalGridConfig
from grid import global_grid_dims, make_global_grid

jax.config.update("jax_enable_x64", True)


def main():
    archived_path = ROOT / "results/industrial_alignment/extensive_transport_reference.json"
    archived = json.loads(archived_path.read_text(encoding="utf-8"))
    frozen_ref = "1d04acf"
    for name, digest in archived["provenance"]["source_sha256"].items():
        git_path = name.replace(chr(92), "/")
        source = subprocess.check_output(["git", "show", f"{frozen_ref}:{git_path}"], cwd=ROOT)
        if hashlib.sha256(source).hexdigest() != digest:
            raise ValueError(f"archived reference source changed: {name}")
    saved_module = sys.modules.get("finite_volume")
    frozen_volume = types.ModuleType("finite_volume")
    volume_source = subprocess.check_output(["git", "show", f"{frozen_ref}:src/finite_volume.py"], cwd=ROOT)
    exec(compile(volume_source, "frozen_finite_volume", "exec"), frozen_volume.__dict__)
    sys.modules["finite_volume"] = frozen_volume
    frozen_barotropic = types.ModuleType("frozen_barotropic_transport")
    barotropic_source = subprocess.check_output(["git", "show", f"{frozen_ref}:src/barotropic_transport.py"], cwd=ROOT)
    try:
        exec(compile(barotropic_source, "frozen_barotropic_transport", "exec"), frozen_barotropic.__dict__)
    finally:
        if saved_module is None:
            del sys.modules["finite_volume"]
        else:
            sys.modules["finite_volume"] = saved_module
    build_geometry = frozen_volume.build_geometry
    surface_volume = frozen_volume.surface_volume
    ExtensiveState = frozen_volume.ExtensiveState
    coupled_surface_step = frozen_barotropic.coupled_surface_step
    nx, ny = global_grid_dims(2., 65., remap="area")
    config = replace(GlobalGridConfig(), resolution=2., lat_max=65., nx=nx, ny=ny)
    rows = []
    for label, smooth, floor in (("prior_smoothed", 80, 500.), ("unsmoothed", 0, 10.)):
        grid = make_global_grid(config, DEFAULT_CONFIG.bathymetry_file,
                                smooth_passes=smooth, min_depth=floor, remap="area")
        geometry = build_geometry(np.r_[grid.lon - 1., grid.lon[-1] + 1.],
                                  np.r_[grid.lat - 1., grid.lat[-1] + 1.],
                                  np.r_[-grid.z, 8000.], grid.depth)
        geometry = jax.tree_util.tree_map(lambda value: jnp.asarray(value, jnp.float32), geometry)
        phase = 2. * np.pi * np.arange(nx)[:, None] / nx
        requested = jnp.asarray(np.broadcast_to(.2 * np.cos(phase), (nx, ny)) * grid.ocean_mask, jnp.float32)
        volume = surface_volume(geometry, requested)
        initial_eta = volume[..., 0] / geometry.area - geometry.thickness[..., 0]
        state = ExtensiveState(volume, volume[..., None] * jnp.asarray([12., 35.], jnp.float32))
        east = jnp.asarray(np.broadcast_to(.02 * np.sin(phase), (nx, ny)), jnp.float32)
        east = jnp.where(jnp.sum(geometry.east_area, axis=-1) > 0., east, 0.)
        north = jnp.zeros_like(east)
        layer_zero = jnp.zeros_like(volume)

        @jax.jit
        def step(current, eta, velocity_east):
            return coupled_surface_step(geometry, current, eta, velocity_east, north,
                                        layer_zero, layer_zero, 15., 4)

        result = step(state, initial_eta, east)
        top_volume = np.asarray(result.transport.state.volume[..., 0], dtype=np.float64)
        area = np.asarray(geometry.area, dtype=np.float64)
        height = np.asarray(geometry.thickness[..., 0], dtype=np.float64)
        predicted_eta = np.asarray(result.barotropic.eta, dtype=np.float64)
        reported_eta = np.asarray(result.transport.state.volume[..., 0] / geometry.area - geometry.thickness[..., 0], dtype=np.float64)
        physical_eta = top_volume / area - height
        scale = max(float(np.max(np.abs(initial_eta))), float(np.max(np.abs(result.barotropic.eta - initial_eta))), 1e-8)
        optimal_volume = (area * (height + predicted_eta)).astype(np.float32).astype(np.float64)
        optimal_eta = optimal_volume / area - height
        initial_physical_eta = np.asarray(volume[..., 0], dtype=np.float64) / area - height
        failed = next(run for run in archived["runs"] if run["geometry"] == label and run["dtype"] == "float32" and run["constant"])
        if abs(float(result.surface_error) - failed["maximum_surface_identity_error"]) > 1e-14:
            raise ValueError("original compiled first-failure residual did not reproduce")
        rows.append({"geometry": label, "original_rejected_ratio": float(result.surface_error),
                     "ratio_reproduces_archived_first_failure": True,
                     "double_diagnostic_ratio": float(np.max(np.abs(physical_eta - predicted_eta)) / scale),
                     "diagnostic_division_rounding_ratio": float(np.max(np.abs(reported_eta - physical_eta)) / scale),
                     "initial_volume_eta_representation_ratio": float(np.max(np.abs(initial_physical_eta - np.asarray(initial_eta))) / scale),
                     "best_representable_volume_ratio": float(np.max(np.abs(optimal_eta - predicted_eta)) / scale),
                     "maximum_half_volume_ulp_m": float(np.max(.5 * np.spacing(np.asarray(result.transport.state.volume[..., 0])) / area)),
                     "barotropic_valid": bool(result.barotropic.valid),
                     "transport_valid": bool(result.transport.valid),
                     "outflow_fraction": float(result.transport.max_outflow_fraction),
                     "wave_bound": float(result.barotropic.gravity_cfl_bound)})
    report = {"scope": "precision_attribution_not_a_changed_gate_or_runtime_fix",
              "frozen_ref": frozen_ref,
              "original_gate": 2e-6,
              "reference_report_sha256": hashlib.sha256(archived_path.read_bytes()).hexdigest(),
              "tool_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "rows": rows}
    output = ROOT / "results/industrial_alignment/extensive_transport_precision.json"
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
