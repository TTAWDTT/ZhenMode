"""Assemble a validated production run and invoke accepted-step orchestration."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from functools import partial

from ocean_solver.audit import stages
from ocean_solver.config import definitions
from ocean_solver.config.definitions import Config
from ocean_solver.forcing import air, fields, seasonal, wind
from ocean_solver.io import climatology, grid, restart
from ocean_solver.io.paths import prepare_output_paths
from ocean_solver.io.recovery import prepare_recovery
from ocean_solver.model import factory
from ocean_solver.provenance.sources import source_root
from ocean_solver.runtime import identity, inputs
from ocean_solver.runtime.cli import parse_run_configuration
from ocean_solver.runtime.context import build_run_context
from ocean_solver.runtime.integration import run_integration


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


def default_services(default_config=None):
    """Bind actual loader/process owners for one invocation; callers may replace services."""
    config = definitions.DEFAULT_CONFIG if default_config is None else default_config
    return RunServices(
        default_config=config,
        make_global_grid=grid.make_global_grid,
        get_initial_fields=climatology.get_initial_fields,
        heat_flux_meridional=fields.heat_flux_meridional,
        build_seasonal_wind_global=seasonal.build_seasonal_wind_global,
        real_wind_forcing=wind.real_wind_forcing,
        load_monthly_mean_air_temp=air.load_monthly_mean_air_temp,
        load_annual_mean_air_temp=air.load_annual_mean_air_temp,
        air_temp_profile=fields.air_temp_profile,
        make_solver_global=factory.make_solver_global,
        make_budget_step=stages.make_budget_step,
        make_restart_contract=restart.make_restart_contract,
        input_files=partial(inputs._input_files, default_config=config),
        source_identity=partial(identity._source_identity, source_root(__file__)),
    )


def run_main(services, source_directory):
    configuration = parse_run_configuration()
    args = configuration.args
    paths = prepare_output_paths(args)
    if args.strict_forcing:
        missing = [str(path) for path in services.input_files(args).values() if not path.is_file()]
        if missing:
            raise ValueError(f"strict forcing preflight: missing local inputs {missing}")
    context = build_run_context(configuration, services)
    recovery = prepare_recovery(
        args, configuration.requested_steps, context, paths, services, source_directory
    )
    return run_integration(args, configuration.requested_steps, context, paths, recovery)
