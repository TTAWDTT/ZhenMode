"""Run real-weather prescribed-stress control through the existing native supervisor."""

import argparse
from pathlib import Path

from zhenmode.benchmarks.forced_channel import contract, validate_contract
from zhenmode.evaluation.forced_channel import score
from zhenmode.execution.native_channel import MODELS, execute


def write_report(path, receipt):
    lines = [
        "# Prescribed real-weather stress / controlled water",
        "",
        "| Model | " + " | ".join(next(iter(receipt["models"].values()))["metrics"]) + " |",
        "| --- |"
        + " | ".join(" ---: " for _ in next(iter(receipt["models"].values()))["metrics"])
        + " |",
    ]
    for model, r in receipt["models"].items():
        lines.append(
            "| " + model + " | " + " | ".join(f"{v:.8g}" for v in r["metrics"].values()) + " |"
        )
    lines += [
        "",
        "RMS describes response, not error. No model accuracy or speed ranking.",
        "",
        *["- " + v for v in receipt["limitations"]],
        "",
    ]
    with Path(path).open("x", encoding="utf-8") as stream:
        stream.write("\n".join(lines))


def run(stress_file, output, method="baseline", **options):
    c = contract(stress_file, method)
    if c["data_kind"] != "observed_weather_derived_stress":
        raise ValueError("real forcing execution requires observed source weather")
    return execute(
        c,
        output,
        evaluator=score,
        definition_validator=validate_contract,
        reporter=write_report,
        scope=c["reference"]["kind"],
        limitations=[
            "Only wind stress is applied; heat/freshwater/ice/mixing remain disabled.",
            "Controlled flat channel, not a realistic regional forecast or global OMIP case.",
            "Cached MOM6 binary has measured identity and incomplete historical build provenance.",
        ],
        **options,
    )


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stress-file", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--method", choices=("baseline", "symmetric-external-mode"), default="baseline")
    p.add_argument("--models", nargs="+", choices=MODELS, default=MODELS)
    for name in ("mom-executable", "mom-source", "oceananigans-cache", "source-revision"):
        p.add_argument("--" + name)
    p.add_argument("--julia", default="julia")
    a = p.parse_args(argv)
    run(
        a.stress_file,
        a.output,
        a.method,
        models=a.models,
        mom_executable=a.mom_executable,
        mom_source=a.mom_source,
        oceananigans_cache=a.oceananigans_cache,
        julia=a.julia,
        source_revision=a.source_revision,
    )
    return 0
