"""Commands delegated by the installed ``zhenmode`` entry point."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .resolve import expand_experiment, expand_sweep, resource_estimate
from .runs import list_runs, run_experiment, write_json
from .schema import ConfigurationError


def main(argv=None):
    parser = argparse.ArgumentParser(prog="zhenmode")
    commands = parser.add_subparsers(dest="group", required=True)
    experiment = commands.add_parser("experiment").add_subparsers(dest="command", required=True)
    for name in ("validate", "expand", "run"):
        command = experiment.add_parser(name)
        command.add_argument("path", type=Path)
        command.add_argument("--root", type=Path, default=Path.cwd())
        if name == "expand":
            command.add_argument("--output", type=Path)
        if name == "run":
            command.add_argument("--outputs", type=Path, default=Path("outputs"))
            command.add_argument("--dry-run", action="store_true")
            command.add_argument("--evaluate", action="store_true", help="score a completed run with its frozen protocol")
            command.add_argument("--backend", choices=("cpu", "cuda"), default="cpu", help="CUDA is explicit: Linux/WSL2, one device, up to 3 h / 8192 MiB host RSS; arrange resources before running")
    sweep = commands.add_parser("sweep").add_subparsers(dest="command", required=True).add_parser("expand")
    sweep.add_argument("path", type=Path)
    sweep.add_argument("--root", type=Path, default=Path.cwd())
    sweep.add_argument("--output", type=Path)
    runs = commands.add_parser("runs").add_subparsers(dest="command", required=True).add_parser("list")
    runs.add_argument("--outputs", type=Path, default=Path("outputs"))
    args = parser.parse_args(argv)
    try:
        if args.group == "runs":
            result = list_runs(args.outputs)
        elif args.group == "sweep":
            result = expand_sweep(args.path, args.root)
        else:
            expanded = expand_experiment(args.path, args.root)
            if args.command == "validate":
                result = {"valid": True, "experiment_id": expanded["experiment_id"], "config_hash": expanded["config_hash"], "resource_estimate": resource_estimate(expanded)}
            elif args.command == "run":
                result = run_experiment(expanded, args.root, args.outputs, dry_run=args.dry_run, evaluate=args.evaluate, backend=args.backend)
            else:
                result = expanded
        if getattr(args, "output", None):
            write_json(args.output, result, create=True)
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False))
        failed = isinstance(result, dict) and (result.get("execution_status") == "failed" or result.get("evaluation", {}).get("status") == "failed" or result.get("acceptance", {}).get("status") == "failed")
        return 1 if failed else 0
    except (ConfigurationError, FileExistsError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    raise SystemExit(main())
