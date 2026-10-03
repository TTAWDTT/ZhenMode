"""External baseline commands, callable as `zhenmode baseline`."""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from ocean_solver.baselines import forcing, mom6


def main(argv=None):
    parser = argparse.ArgumentParser(prog="zhenmode baseline")
    methods = parser.add_subparsers(dest="method", required=True)
    model = methods.add_parser("mom6")
    commands = model.add_subparsers(dest="command", required=True)
    forcing.add_arguments(commands.add_parser("export-forcing", help="export shared forcing to MOM6 A-grid inputs"))
    for command in ("doctor", "fetch", "build", "prepare"):
        p = commands.add_parser(command)
        p.add_argument("--cache", default=str(Path.home()/".cache/zhenmode/mom6/f49a000"))
        if command == "build":
            p.add_argument("--wall-seconds", type=int, default=1800)
        elif command == "prepare":
            p.add_argument("--run-root")
    for command in ("run", "convert", "evaluate"):
        p = commands.add_parser(command)
        p.add_argument("--run-dir", required=True)
        if command == "run":
            p.add_argument("--wall-seconds", type=int, default=180)
    p = commands.add_parser("prepare-native", help="check and bind existing shared global native inputs")
    for name in ("case", "preset", "native-dir", "reference-npz", "root"):
        p.add_argument("--"+name, required=True)
    p.add_argument("--cache", default=str(Path.home()/".cache/zhenmode/mom6/f49a000"))
    p.add_argument("--run-root")
    p = commands.add_parser("prepare-wave-input", help="prepare frozen native standing-wave initial interfaces only")
    p.add_argument("--contract", required=True)
    p.add_argument("--out-dir", required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "export-forcing":
            result = forcing.export_forcing(args)
        elif args.command in ("doctor", "fetch"):
            result = getattr(mom6, args.command)(args.cache)
        elif args.command == "build":
            result = mom6.build(args.cache, wall_seconds=args.wall_seconds)
        elif args.command == "prepare":
            result = mom6.prepare(args.cache, run_root=args.run_root)
        elif args.command == "prepare-native":
            result = mom6.prepare_native(args.cache, case_path=args.case, preset_path=args.preset,
                                         native_dir=args.native_dir, reference_npz=args.reference_npz,
                                         root=args.root, run_root=args.run_root)
        elif args.command == "prepare-wave-input":
            result = mom6.prepare_wave_input(args.contract, args.out_dir)
        elif args.command == "run":
            result = mom6.run(args.run_dir, wall_seconds=args.wall_seconds)
        else:
            result = getattr(mom6, args.command)(args.run_dir)
        # Large actual source hash lists are retained on disk in run.json.
        summary = {name: value for name, value in result.items() if name != "source_identity"}
        print(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False))
        if result.get("execution_status") == "failed" or result.get("ready") is False:
            return 1
        return 0
    except (ValueError, RuntimeError, OSError, KeyError, subprocess.TimeoutExpired) as error:
        parser.error(str(error))


if __name__ == "__main__":
    raise SystemExit(main())
