"""Installed entry for the two frozen rotating/stratified channel benchmarks."""

import argparse
import json
from pathlib import Path

from zhenmode.benchmarks.channel_dynamics import CASES, contract, validate_contract
from zhenmode.evaluation.channel_dynamics import score
from zhenmode.execution.native_channel import MODELS, execute
from zhenmode.execution.runs import write_json


def write_report(path, receipt):
    keys = (
        "eta_error",
        "u_error",
        "v_error",
        "T_error_C",
        "S_error_psu",
        "volume",
        "T_inventory",
        "S_inventory",
    )
    lines = [
        f"# Channel: {receipt['benchmark']} / {receipt['case']}",
        "",
        "| Model | " + " | ".join(keys) + " | Failed screen metrics |",
        "| --- | " + " | ".join("---:" for _ in keys) + " | --- |",
    ]
    for model, r in receipt["models"].items():
        lines.append(
            "| "
            + model
            + " | "
            + " | ".join(f"{r['metrics'][k]:.8g}" for k in keys)
            + " | "
            + (", ".join(r["failed_metrics"]) or "none")
            + " |"
        )
    lines += [
        "",
        "Velocity RMS is three-dimensional. T/S error units are degC/psu; other metrics are relative.",
        "",
        *["- " + value for value in receipt["limitations"]],
        "",
    ]
    with Path(path).open("x", encoding="utf-8") as stream:
        stream.write("\n".join(lines))


def run(case, output, level="coarse", method="baseline", **options):
    c = contract(case, level, method)
    return execute(
        c,
        output,
        evaluator=score,
        definition_validator=validate_contract,
        reporter=write_report,
        scope=c["reference"]["kind"],
        limitations=[
            "Frozen idealized channel variants; no industrial qualification or speed ranking.",
            "Rotating-wave reference is linear; native models retain nonlinear dynamics.",
            "Thermal-wind reference tests equilibrium and density-neutral transport, not unstable-jet turbulence.",
            "Native node/layer and closed-wall discretizations differ and are retained.",
            "MOM6 cached binary identity is measured; historical full build receipt is incomplete.",
        ],
        **options,
    )


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--case", choices=CASES, required=True)
    p.add_argument("--level", choices=("coarse", "medium", "fine"), default="coarse")
    p.add_argument("--method", choices=("baseline", "symmetric-external-mode", "symmetric-closed-faces"), default="baseline")
    output = p.add_mutually_exclusive_group(required=True)
    output.add_argument("--output", type=Path)
    output.add_argument("--freeze", type=Path)
    p.add_argument("--models", nargs="+", choices=MODELS, default=MODELS)
    for name in ("mom-executable", "mom-source", "oceananigans-cache", "source-revision"):
        p.add_argument("--" + name)
    p.add_argument("--julia", default="julia")
    args = p.parse_args(argv)
    if args.freeze:
        write_json(args.freeze, contract(args.case, args.level, args.method), create=True)
        return 0
    result = run(
        args.case,
        args.output,
        args.level,
        method=args.method,
        models=args.models,
        mom_executable=args.mom_executable,
        mom_source=args.mom_source,
        oceananigans_cache=args.oceananigans_cache,
        julia=args.julia,
        source_revision=args.source_revision,
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "models": {k: r["metrics"] for k, r in result["models"].items()},
            },
            indent=2,
        )
    )
    return 0
