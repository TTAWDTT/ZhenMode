"""Sequential wave runs and receipts; adapters own native model execution."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from zhenmode.benchmarks.standing_wave import (
    contract,
    method_contract,
    study_contract,
    validate_contract,
)
from zhenmode.evaluation.standing_wave import score
from zhenmode.execution.native_channel import execute

MODELS = ("zhenmode", "mom6", "oceananigans")
MODEL_IDS = {"zhenmode": "ocean-solver", "mom6": "MOM6", "oceananigans": "Oceananigans"}


def write_report(path, receipt):
    """Render measured values directly from the same JSON receipt, without ranking."""
    keys = ("eta_error", "u_error", "phase_rad", "amplitude", "energy", "volume", "T", "S")
    lines = [
        f"# Standing wave: {receipt['case']}",
        "",
        "| Model | " + " | ".join(keys) + " | Failed screening metrics |",
        "| --- | " + " | ".join("---:" for _ in keys) + " | --- |",
    ]
    for name, report in receipt["models"].items():
        values = " | ".join(f"{report['metrics'][key]:.8g}" for key in keys)
        failed = ", ".join(report["failed_metrics"]) or "none"
        lines.append(f"| {name} | {values} | {failed} |")
    lines += [
        "",
        "Velocity error is depth-mean; phase is in radians. Error, amplitude, energy and volume are relative; T/S errors are in degC/psu.",
        "",
        "Execution completion and engineering screening are separate. Industrial qualification and performance comparison are not claimed.",
        "",
        *["- " + value for value in receipt["limitations"]],
        "",
    ]
    with Path(path).open("x", encoding="utf8") as stream:
        stream.write("\n".join(lines))


def run(
    case,
    output,
    *,
    models=MODELS,
    mom_executable=None,
    mom_source=None,
    oceananigans_cache=None,
    julia="julia",
    source_revision=None,
    study=None,
    method="baseline",
    half=False,
):
    if method not in ("baseline", "symmetric-external-mode") or (study and method != "baseline"):
        raise ValueError("unsupported wave method/study combination")
    c = (
        (contract(case, half) if method == "baseline" else method_contract(case, half))
        if study is None
        else study_contract(*study)
    )
    return execute(
        c,
        output,
        evaluator=score,
        definition_validator=validate_contract,
        reporter=write_report,
        scope="linear small-amplitude wave vs full nonlinear native models",
        limitations=[
            "One flat unstratified wave; no climate qualification or speed ranking.",
            "MOM6 cached binary identity is measured; historical source-to-binary build receipt is incomplete.",
            "Native vertical coordinates and time integration differ; eta and velocity are scored at native locations.",
        ],
        models=models,
        mom_executable=mom_executable,
        mom_source=mom_source,
        oceananigans_cache=oceananigans_cache,
        julia=julia,
        source_revision=source_revision,
    )


def main(argv=None):
    parser = argparse.ArgumentParser(prog="zhenmode benchmark standing-wave")
    parser.add_argument("--case", choices=("coarse", "medium", "fine"), default="coarse")
    parser.add_argument("--output", required=True)
    parser.add_argument("--models", nargs="+", choices=MODELS, default=MODELS)
    parser.add_argument("--mom-executable")
    parser.add_argument("--mom-source")
    parser.add_argument("--oceananigans-cache")
    parser.add_argument("--julia", default="julia")
    parser.add_argument("--source-revision")
    parser.add_argument("--study", nargs=2, type=int, metavar=("NX", "DT_SECONDS"))
    parser.add_argument("--half", action="store_true")
    parser.add_argument(
        "--method", choices=("baseline", "symmetric-external-mode"), default="baseline"
    )
    args = parser.parse_args(argv)
    result = run(
        args.case,
        args.output,
        models=args.models,
        mom_executable=args.mom_executable,
        mom_source=args.mom_source,
        oceananigans_cache=args.oceananigans_cache,
        julia=args.julia,
        source_revision=args.source_revision,
        study=args.study,
        method=args.method,
        half=args.half,
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "models": {name: row["metrics"] for name, row in result["models"].items()},
            },
            indent=2,
        )
    )
    return 0
