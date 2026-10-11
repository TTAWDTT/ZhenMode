"""Installed entry: run a single frozen internal-wave holdout row without tuning."""

import argparse
import json
from pathlib import Path

from zhenmode.benchmarks.internal_wave import MODELS, PROFILES, contract, validate_contract
from zhenmode.evaluation.internal_wave import score
from zhenmode.execution.channel_dynamics import write_report
from zhenmode.execution.native_channel import execute
from zhenmode.execution.runs import write_json


def run(model, output, profile="primary", control="main", **options):
    c = contract(model, profile, control)
    return execute(c, output, evaluator=score, definition_validator=validate_contract,
                   reporter=write_report, models=(model,), scope=c["reference"]["kind"],
                   limitations=["Frozen selections from thermal matrix; no holdout tuning.",
                                "Linear reference versus full nonlinear native dynamics; main/half amplitude controls.",
                                "Native points/layer means and moving coordinates are retained.",
                                "MOM6 cached binary history remains incomplete; no speed or global SOTA claim."], **options)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", choices=MODELS, required=True)
    p.add_argument("--profile", choices=tuple(PROFILES), default="primary")
    p.add_argument("--control", choices=("main", "half", "null"), default="main")
    outputs = p.add_mutually_exclusive_group(required=True)
    outputs.add_argument("--output", type=Path)
    outputs.add_argument("--freeze", type=Path)
    for name in ("mom-executable", "mom-source", "oceananigans-cache", "source-revision"):
        p.add_argument("--" + name)
    p.add_argument("--julia", default="julia")
    args = p.parse_args(argv)
    if args.freeze:
        write_json(args.freeze, contract(args.model, args.profile, args.control), create=True)
        return 0
    r = run(args.model, args.output, args.profile, args.control,
            mom_executable=args.mom_executable, mom_source=args.mom_source,
            oceananigans_cache=args.oceananigans_cache, julia=args.julia,
            source_revision=args.source_revision)
    print(json.dumps(r["models"], indent=2, allow_nan=False))
    return 0
