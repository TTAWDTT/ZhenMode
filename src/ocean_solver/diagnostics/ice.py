"""Summarize the minimal sea-ice/mixed-layer closed-loop diagnostics.

The function reads an ocean_solver NPZ plus its standardized benchmark JSON and
returns one manifest fragment that records the Stage-I closed-loop items without
re-running the model.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ocean_solver.evaluation.metrics import cell_area

RHO_ICE = 917.0
LATENT_HEAT_FUSION = 3.34e5


def summarize_ice_closed_loop(npz_path: str | Path,
                              benchmark_path: str | Path,
                              experiment_id: str) -> dict:
    """Return a Stage-I manifest built from one saved dynamic-ice run."""
    data = np.load(npz_path, allow_pickle=True)
    days = np.asarray(data["days"], dtype=float)
    ice = np.asarray(data["ice_top"], dtype=float)
    wet = np.asarray(data["wet_mask"], dtype=bool)
    area = cell_area(np.asarray(data["lat"]), np.asarray(data["lon"]))
    growth_m3 = 0.0
    melt_m3 = 0.0
    for k in range(1, days.size):
        delta = ice[k] - ice[k - 1]
        growth_m3 += float(np.sum(np.maximum(delta, 0.0) * area * wet))
        melt_m3 += float(np.sum(np.maximum(-delta, 0.0) * area * wet))

    latent_growth_j = RHO_ICE * LATENT_HEAT_FUSION * growth_m3
    latent_melt_j = RHO_ICE * LATENT_HEAT_FUSION * melt_m3
    salt_change_kg = float(np.asarray(data["salt_content_kg"], dtype=float)[-1]
                           - np.asarray(data["salt_content_kg"], dtype=float)[0])
    heat_change_j = float(np.asarray(data["heat_content_J"], dtype=float)[-1]
                          - np.asarray(data["heat_content_J"], dtype=float)[0])
    surface_area = float(area[wet].sum())
    duration_s = float(days[-1] - days[0]) * 86400.0
    enthalpy_change_j = heat_change_j - latent_growth_j + latent_melt_j
    surface_input_j = (float(data["surface_heat_input_J"][-1] - data["surface_heat_input_J"][0])
                       if "surface_heat_input_J" in data else None)
    residual_j = enthalpy_change_j - surface_input_j if surface_input_j is not None else None
    salt_units = str(data["salt_content_units"].item()) if "salt_content_units" in data else "legacy_psu_mass"
    if salt_units == "legacy_psu_mass":
        salt_change_kg /= 1000.0
    data.close()

    benchmark = json.loads(Path(benchmark_path).read_text(encoding="utf-8"))
    return {
        "experiment_id": experiment_id,
        "status": "minimal_closed_loop_diagnostic",
        "source_npz": str(Path(npz_path)),
        "source_benchmark": str(Path(benchmark_path)),
        "duration_days": float(days[-1]),
        "climate_score": {
            "verdict": benchmark["verdict"],
            "global_a2_rmse_c": benchmark["global_a2_rmse_c"],
            "north_atlantic_40_60_raw_rmse_c":
                benchmark["north_atlantic_40_60"]["raw_rmse"],
            "north_atlantic_40_60_raw_bias_c":
                benchmark["north_atlantic_40_60"]["raw_bias"],
            "near_wall_55_60_raw_bias_c":
                benchmark["near_wall_55_60"]["raw_bias"],
            "heat_drift_percent": benchmark["heat_drift_percent"],
            "salt_drift_percent": benchmark["salt_drift_percent"],
        },
        "ice": {
            **benchmark["ice"],
            "growth_volume_m3": growth_m3,
            "melt_volume_m3": melt_m3,
            "latent_heat_growth_J": latent_growth_j,
            "latent_heat_melt_J": latent_melt_j,
            "growth_melt_sampling": "net_changes_between_snapshots_not_gross_timestep_fluxes",
        },
        "mixed_layer": benchmark["mld"],
        "surface_budget": {
            "heat_content_change_J": heat_change_j,
            "water_ice_enthalpy_change_J": enthalpy_change_j,
            "surface_heat_input_J": surface_input_j,
            "latent_heat_growth_J": latent_growth_j,
            "latent_heat_melt_J": latent_melt_j,
            "heat_budget_residual_J": residual_j,
            "mean_residual_W_m2": residual_j / (duration_s * surface_area)
            if residual_j is not None and duration_s > 0 and surface_area > 0 else None,
            "budget_scope": ("water_ice_enthalpy_minus_recorded_surface_heat_input"
                             if surface_input_j is not None
                             else "enthalpy_change_only_without_recorded_external_heat_input"),
        },
        "brine_salt_flux": {
            "salt_content_change_kg": salt_change_kg,
            "definition": "wet-column salt content change including all enabled salt terms; not an isolated brine flux",
        },
        "stability": {
            "verdict": benchmark["verdict"],
            "max_u_peak_m_s": benchmark["max_u_peak"],
            "max_eta_last_m": benchmark["max_eta_last"],
        },
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--npz", required=True)
    p.add_argument("--benchmark", required=True)
    p.add_argument("--experiment-id", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    result = summarize_ice_closed_loop(args.npz, args.benchmark, args.experiment_id)
    Path(args.out).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))




if __name__ == "__main__":
    main()
