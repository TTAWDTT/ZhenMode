"""
Long-integration driver for the global finite-difference solver (jax_solver_global).

Runs the global FD solver -- closed no-flux N/S walls at lat_max=60, nu_h=5e6
spin-up stabilizer, dt=60 -- with real seasonal NCEP wind and a bulk air-sea
heat flux. The point of the global domain is the non-circular A1/A2 zonal SST
skill test: a global non-periodic domain removes the periodic-BC crutch, so
large-scale SST structure has to emerge from
geometry + wind + bathymetry rather than from a prescribed meridional T_atm
clamp.

Stability: the no-flux wall + nu_h=5e6 hold both no-wind and wind-forced
(tau0=0.1) runs to 1000 steps with 0 NaN and max|T| bounded. This driver
extends that to a full annual integration. The pass/fail criteria below are
pre-registered -- do NOT move the bar after running.

Output:
  - results/global_<tag>.npz   (snapshots: eta/T/u maxima, KE, SSH_std, T_top)
  - logs/global_<tag>.log      (full monitor trace + verdict)
  - results/global_<tag>_3d/   (streamed 3D T/U/V snapshots, one .npy each)

Usage:
  python src/run_long_integration_global.py --days 365 --seasonal-wind --tag g365d
  python src/run_long_integration_global.py --days 200 --tag g200d_smoke   # shorter probe
"""
import argparse as argparse
import os
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataclasses import replace as replace

from air_reanalysis import load_annual_mean_air_temp as load_annual_mean_air_temp
from air_reanalysis import load_monthly_mean_air_temp as load_monthly_mean_air_temp
from benchmark_metrics import mixed_layer_depth as mixed_layer_depth
from config import DEFAULT_CONFIG as DEFAULT_CONFIG
from config import GlobalGridConfig as GlobalGridConfig
from config import PhysicsConfig as PhysicsConfig
from diagnostics import BudgetDiagnostics as BudgetDiagnostics
from diagnostics import compute_budget_diagnostics as compute_budget_diagnostics
from diagnostics import diagnostics_to_arrays as diagnostics_to_arrays
from forcing import BULK_LAMBDA_DEFAULT as BULK_LAMBDA_DEFAULT
from forcing import air_temp_profile as air_temp_profile
from forcing import heat_flux_meridional as heat_flux_meridional
from forcing import ocean_zonal_mean as ocean_zonal_mean
from grid import global_grid_dims as global_grid_dims
from grid import land_distance_from_land_mask as land_distance_from_land_mask
from grid import make_global_grid as make_global_grid
from integration_monitor import classify_state as classify_state
from jax_solver_global import JaxStateG as JaxStateG
from jax_solver_global import make_solver_global as make_solver_global
from jax_solver_global import projection_config as projection_config
from ocean_solver.fd.backend import jax as jax
from ocean_solver.fd.backend import jnp as jnp
from ocean_solver.fd.backend import np as np
from ocean_solver.runtime.application import run_main
from ocean_solver.runtime.cli import AMPLITUDE_CAP_C as AMPLITUDE_CAP_C
from ocean_solver.runtime.cli import DRIFT_TOL_C as DRIFT_TOL_C
from ocean_solver.runtime.cli import DT_BT_DEFAULT as DT_BT_DEFAULT
from ocean_solver.runtime.cli import DT_DEFAULT as DT_DEFAULT
from ocean_solver.runtime.cli import ETA_BLOWUP_M as ETA_BLOWUP_M
from ocean_solver.runtime.cli import LAMBDA_BULK_DEFAULT_G as LAMBDA_BULK_DEFAULT_G
from ocean_solver.runtime.cli import LAT_MAX_DEFAULT as LAT_MAX_DEFAULT
from ocean_solver.runtime.cli import MAX_U_BOUND as MAX_U_BOUND
from ocean_solver.runtime.cli import MIN_DEPTH_DEFAULT as MIN_DEPTH_DEFAULT
from ocean_solver.runtime.cli import NU_BI_DEFAULT as NU_BI_DEFAULT
from ocean_solver.runtime.cli import NU_BI_REF_1DEG as NU_BI_REF_1DEG
from ocean_solver.runtime.cli import NU_H_DEFAULT as NU_H_DEFAULT
from ocean_solver.runtime.cli import NU_H_REF_1DEG as NU_H_REF_1DEG
from ocean_solver.runtime.cli import NY_DEFAULT as NY_DEFAULT
from ocean_solver.runtime.cli import POLAR_CAP_ROWS_DEFAULT as POLAR_CAP_ROWS_DEFAULT
from ocean_solver.runtime.cli import POLAR_CAP_TAPER_DEFAULT as POLAR_CAP_TAPER_DEFAULT
from ocean_solver.runtime.cli import SMOOTH_PASSES_DEFAULT as SMOOTH_PASSES_DEFAULT
from ocean_solver.runtime.cli import SPONGE_DAYS_DEFAULT_G as SPONGE_DAYS_DEFAULT_G
from ocean_solver.runtime.cli import _validate_arguments as _validate_arguments
from ocean_solver.runtime.cli import scaled_physics_for_resolution as scaled_physics_for_resolution
from ocean_solver.runtime.identity import SOURCE_MODULES as SOURCE_MODULES
from ocean_solver.runtime.identity import _same_state_bytes as _same_state_bytes
from ocean_solver.runtime.identity import _source_identity as source_identity
from ocean_solver.runtime.identity import _state_identity as _state_identity
from ocean_solver.runtime.inputs import _input_files as resolve_input_files
from ocean_solver.runtime.inputs import _lat_band_mask as _lat_band_mask
from ocean_solver.runtime.metrics import state_is_finite as state_is_finite
from ocean_solver.runtime.metrics import total_kinetic_energy as total_kinetic_energy
from ocean_solver.runtime.paths import _Tee as _Tee
from ocean_solver.runtime.records import _save_snapshot_file as _save_snapshot_file
from ocean_solver.runtime.recovery import SNAPSHOT_FIELDS as SNAPSHOT_FIELDS
from ocean_solver.runtime.recovery import SNAPSHOT_SCALARS as SNAPSHOT_SCALARS
from ocean_solver.runtime.recovery import _validate_restart_history as _validate_restart_history
from ocean_solver.runtime.seasonal import build_seasonal_wind_global as build_seasonal_wind_global
from ocean_solver.runtime.seasonal import interp_monthly_field as interp_monthly_field
from ocean_solver.runtime.seasonal import interp_monthly_field_jit as interp_monthly_field_jit
from ocean_solver.runtime.seasonal import interp_seasonal_wind as interp_seasonal_wind
from ocean_solver.runtime.seasonal import interp_seasonal_wind_jit as interp_seasonal_wind_jit
from ocean_solver.runtime.seasonal import marine_smooth_2d as marine_smooth_2d
from ocean_solver.runtime.services import RunServices
from restart_contract import file_sha256 as file_sha256
from restart_contract import fingerprint as fingerprint
from restart_contract import load_restart as load_restart
from restart_contract import make_restart_contract as make_restart_contract
from restart_contract import save_restart as save_restart
from runtime_validation import finite_number as finite_number
from runtime_validation import integer_count as integer_count
from source_identity import production_source_modules as production_source_modules
from stage_budgets import METRIC_NAMES as METRIC_NAMES
from stage_budgets import NONLINEAR_PROCESS_NAMES as NONLINEAR_PROCESS_NAMES
from stage_budgets import SOURCE_NAMES as SOURCE_NAMES
from stage_budgets import STAGE_NAMES as STAGE_NAMES
from stage_budgets import accumulate_budget as accumulate_budget
from stage_budgets import empty_budget as empty_budget
from stage_budgets import make_budget_step as make_budget_step
from wind_reanalysis import real_wind_forcing as real_wind_forcing
from woa_data import get_initial_fields as get_initial_fields


def _input_files(args, *, seasonal=None, air=None):
    return resolve_input_files(args, seasonal=seasonal, air=air, default_config=DEFAULT_CONFIG)


def _source_identity():
    return source_identity(Path(__file__).resolve().parent)


def main():
    services = RunServices(
        default_config=DEFAULT_CONFIG,
        make_global_grid=make_global_grid,
        get_initial_fields=get_initial_fields,
        heat_flux_meridional=heat_flux_meridional,
        build_seasonal_wind_global=build_seasonal_wind_global,
        real_wind_forcing=real_wind_forcing,
        load_monthly_mean_air_temp=load_monthly_mean_air_temp,
        load_annual_mean_air_temp=load_annual_mean_air_temp,
        air_temp_profile=air_temp_profile,
        make_solver_global=make_solver_global,
        make_budget_step=make_budget_step,
        make_restart_contract=make_restart_contract,
        input_files=_input_files,
        source_identity=_source_identity,
    )
    return run_main(services, Path(__file__).resolve().parent)


if __name__ == "__main__":
    sys.exit(main())
