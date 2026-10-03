"""Public CLI for the existing global finite-difference production method."""
import sys

from ocean_solver.provenance.sources import source_root
from ocean_solver.runtime.application import default_services, run_main


def main():
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except (AttributeError, OSError):
        pass
    return run_main(default_services(), source_root(__file__))


if __name__ == "__main__":
    sys.exit(main())
