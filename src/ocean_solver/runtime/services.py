"""Invocation-bound loader, solver, audit, and identity dependencies."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from ocean_solver.config.definitions import Config


@dataclass
class RunServices:
    default_config: Config
    make_global_grid: Callable
    get_initial_fields: Callable
    heat_flux_meridional: Callable
    build_seasonal_wind_global: Callable
    real_wind_forcing: Callable
    load_monthly_mean_air_temp: Callable
    load_annual_mean_air_temp: Callable
    air_temp_profile: Callable
    make_solver_global: Callable
    make_budget_step: Callable
    make_restart_contract: Callable
    input_files: Callable
    source_identity: Callable
    resolve_grid_dimensions: Callable | None = None
