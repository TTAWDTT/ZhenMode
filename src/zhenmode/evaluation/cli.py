"""Installed `zhenmode evaluate` subcommands."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from zhenmode.evaluation.pipeline import compare, evaluate, import_historical
from zhenmode.evaluation.protocols import load_json


def main(argv=None):
    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments and arguments[0] == "wave":
        from zhenmode.evaluation.standing_wave import main as wave
        return wave(arguments[1:])
    if arguments and arguments[0] == "wave-study":
        from zhenmode.evaluation.wave_study import main as wave_study
        return wave_study(arguments[1:])
    if arguments and arguments[0] == "channel":
        from zhenmode.evaluation.channel_dynamics import main as channel
        return channel(arguments[1:])
    parser = argparse.ArgumentParser(prog="zhenmode evaluate")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("wave", help="freeze, score or compare native standing-wave trajectories")
    commands.add_parser("wave-study", help="analyze five frozen separate space/time controls")
    commands.add_parser("channel", help="score native rotating/stratified channel trajectories")
    score = commands.add_parser("score", help="validate a protocol and score one completed run")
    for name in ("input", "protocol", "run-manifest", "out-dir"):
        score.add_argument("--" + name, required=True)
    score.add_argument("--format", choices=("npz", "mom6"), default="npz")
    for name in ("reference", "geometry"):
        score.add_argument("--" + name)
    comparison = commands.add_parser("compare", help="reject incompatible reports; never infer speedup")
    comparison.add_argument("reports", nargs="+")
    comparison.add_argument("--out", required=True)
    historical = commands.add_parser("import-historical", help="preserve old results and protocol identity")
    historical.add_argument("--input", required=True)
    historical.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "score":
            result = evaluate(args.input, args.protocol, args.run_manifest, args.out_dir,
                              format=args.format, reference=args.reference, geometry=args.geometry)
        else:
            result = (compare([load_json(path) for path in args.reports]) if args.command == "compare"
                      else import_historical(args.input))
            with Path(args.out).open("x", encoding="utf-8") as stream:
                stream.write(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
        print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
        return 1 if args.command == "compare" and not result["comparable"] else 0
    except (ValueError, RuntimeError, OSError, KeyError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    raise SystemExit(main())
