"""Run one declared setting through the existing bounded native benchmark path."""

import argparse
import json
from pathlib import Path

from zhenmode.benchmarks.configuration_scan import (
    PROBLEMS,
    contract,
    followup_rows,
    rows,
    validate_contract,
)
from zhenmode.execution.native_channel import MODELS, execute
from zhenmode.execution.runs import write_json


def run(problem, model, setting, output, **options):
    c = contract(problem, model, setting)
    if problem == "standing-wave":
        from zhenmode.evaluation.standing_wave import score
        from zhenmode.execution.standing_wave import write_report
    else:
        from zhenmode.evaluation.channel_dynamics import score
        from zhenmode.execution.channel_dynamics import write_report
    return execute(
        c, output, models=(model,), evaluator=score,
        definition_validator=validate_contract, reporter=write_report,
        scope="Declared numerical settings on the unchanged physical problem",
        limitations=[
            "Development scan; not independent certification or a global optimum search.",
            "Physical inputs and screens are unchanged; native node/layer representations remain different.",
            "MOM6 cached binary identity is recorded; historical full build provenance remains incomplete.",
            "GPU and CPU costs are reported separately; no cross-hardware speed ranking.",
        ], **options,
    )


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    target = p.add_mutually_exclusive_group(required=True)
    target.add_argument("--plan", type=Path)
    target.add_argument("--output", type=Path)
    p.add_argument("--problem", choices=PROBLEMS)
    p.add_argument("--model", choices=MODELS)
    p.add_argument("--setting")
    for name in ("mom-executable", "mom-source", "oceananigans-cache", "source-revision"):
        p.add_argument("--" + name)
    p.add_argument("--julia", default="julia")
    a = p.parse_args(argv)
    if a.plan:
        write_json(a.plan, dict(schema="native-scan-plan-v1", rows=rows(), followups=followup_rows(), execution="not_started"), create=True)
        return 0
    if not all((a.problem, a.model, a.setting)):
        p.error("--output requires --problem, --model and --setting")
    receipt = run(a.problem, a.model, a.setting, a.output,
                  mom_executable=a.mom_executable, mom_source=a.mom_source,
                  oceananigans_cache=a.oceananigans_cache, julia=a.julia,
                  source_revision=a.source_revision)
    print(json.dumps(dict(status=receipt["status"], models=receipt["models"]), indent=2))
    return 0
