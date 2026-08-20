"""Unified test runner — executes all standalone test scripts and reports results."""
import subprocess
import sys
import os
import time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TESTS = [
    ("test_wave_speed", os.path.join(PROJECT_ROOT, "src", "test_wave_speed.py")),
    ("test_forcing", os.path.join(PROJECT_ROOT, "src", "test_forcing.py")),
    ("compare_jax_numpy", os.path.join(PROJECT_ROOT, "src", "compare_jax_numpy.py")),
]


def run_test(name: str, path: str) -> tuple[bool, float, str]:
    """Run a single test script, return (passed, elapsed_s, last_line)."""
    start = time.time()
    try:
        result = subprocess.run(
            [sys.executable, path],
            capture_output=True,
            text=True,
            cwd=PROJECT_ROOT,
            timeout=120,
        )
        elapsed = time.time() - start
        lines = (result.stdout + result.stderr).strip().split("\n")
        last_line = lines[-1] if lines else "(no output)"
        passed = result.returncode == 0 and "FAIL" not in last_line.upper()
        return passed, elapsed, last_line
    except subprocess.TimeoutExpired:
        elapsed = time.time() - start
        return False, elapsed, "TIMEOUT (>120s)"
    except Exception as e:
        elapsed = time.time() - start
        return False, elapsed, f"ERROR: {e}"


def main():
    print("=" * 60)
    print("Ocean Solver — Test Suite")
    print("=" * 60)
    print()

    results = []
    for name, path in TESTS:
        print(f"[{name}] running...", end=" ", flush=True)
        passed, elapsed, last_line = run_test(name, path)
        status = "PASS" if passed else "FAIL"
        print(f"{status} ({elapsed:.1f}s)")
        print(f"  -> {last_line}")
        results.append((name, passed, elapsed, last_line))
        print()

    n_pass = sum(1 for _, p, _, _ in results if p)
    n_total = len(results)
    print("-" * 60)
    print(f"Results: {n_pass}/{n_total} passed")
    for name, passed, elapsed, _ in results:
        status = "PASS" if passed else "FAIL"
        print(f"  {status}  {name} ({elapsed:.1f}s)")

    sys.exit(0 if n_pass == n_total else 1)


if __name__ == "__main__":
    main()
