"""Installed entry for the production model, experiments and evaluation."""
from __future__ import annotations

import sys

HELP = """ZhenMode — existing global finite-difference production method

Usage: zhenmode COMMAND [ARGS]
  model       Run the production FD model (same arguments as ocean-solver)
  experiment  Validate, expand, or run a declared experiment
  sweep       Expand a parameter sweep and estimate resources (no execution)
  runs        Inspect independent run manifests
  evaluate    Score results or check comparison eligibility
  baseline    Prepare and run a pinned external baseline (MOM6)
  mms         Run the independent manufactured-solution operator checks

Each command provides --help. Plain model flags also select the production FD
method. Local research prototypes are outside the installed production workflow.
"""


def main(argv=None):
    arguments = list(sys.argv[1:] if argv is None else argv)
    if not arguments or arguments in (["--help"], ["-h"]):
        print(HELP)
        return 0
    command, *remaining = arguments
    if command in {"experiment", "sweep", "runs"}:
        from ocean_solver.experiments.cli import main as execute
        return execute(arguments)
    if command == "evaluate":
        from ocean_solver.evaluation.cli import main as execute
        return execute(remaining)
    if command == "baseline":
        from ocean_solver.baselines.cli import main as execute
        return execute(remaining)
    if command == "mms":
        from ocean_solver.validation.mms import main as execute
        return execute()
    if command == "model" or command.startswith("-"):
        from ocean_solver.runtime.entry import main as execute
        previous = sys.argv
        sys.argv = ["ocean-solver", *(remaining if command == "model" else arguments)]
        try:
            return execute()
        finally:
            sys.argv = previous
    raise SystemExit("Unknown command: " + command + " (use zhenmode --help)")
