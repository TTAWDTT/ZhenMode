"""Unified test runner — runs the pytest suite under tests/ and reports results."""
import os
import subprocess
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main() -> None:
    print("=" * 60)
    print("Ocean Solver — Test Suite")
    print("=" * 60)
    print()
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/", "-q"],
        cwd=PROJECT_ROOT,
    )
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
