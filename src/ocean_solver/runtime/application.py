"""Assemble a validated production run and invoke accepted-step orchestration."""

from __future__ import annotations

from ocean_solver.runtime.cli import parse_run_configuration
from ocean_solver.runtime.context import build_run_context
from ocean_solver.runtime.integration import run_integration
from ocean_solver.runtime.paths import prepare_output_paths
from ocean_solver.runtime.recovery import prepare_recovery


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
