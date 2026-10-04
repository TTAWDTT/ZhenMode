"""Public CLI for the existing global finite-difference production method."""
import sys

from zhenmode.model.runtime.application import default_services, run_main
from zhenmode.provenance.sources import source_root


def main():
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except (AttributeError, OSError):
        pass
    return run_main(default_services(), source_root(__file__))


if __name__ == "__main__":
    sys.exit(main())
