"""Explicit CUDA budgets and Linux process-group RSS/wall supervision."""

from __future__ import annotations

import os
import signal
import subprocess
import time
from pathlib import Path

from .schema import ConfigurationError


def validate_budget(resources, backend):
    if backend not in {"cpu", "cuda"}:
        raise ConfigurationError("execution backend must be cpu or cuda")
    wall, memory = (180, 4096) if backend == "cpu" else (10800, 8192)
    if resources["cpu"] != 1 or resources["wall_seconds"] > wall or resources["memory_mib"] > memory:
        raise ConfigurationError(f"managed {backend} run exceeds 1 CPU / {wall} seconds / {memory} MiB; use a separately reviewed execution plan")


def process_group_rss(group):
    """Sum resident sets, including children; shared pages count per process."""
    total = 0
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            if os.getpgid(int(entry.name)) != group:
                continue
            status = (entry / "status").read_text()
            total += next(int(line.split()[1]) * 1024 for line in status.splitlines() if line.startswith("VmRSS:"))
        except (OSError, StopIteration):
            continue
    return total


def run_process_group(command, *, cwd, env, stdout, resources):
    """Enforce one CPU, wall time and sampled aggregate RSS, not a cgroup cap.

    Own a process group so timeout/interruption also kills worker children.
    """
    if os.name != "posix" or not Path("/proc/self/status").is_file() or not hasattr(os, "sched_setaffinity"):
        raise ConfigurationError("managed process groups require Linux /proc and CPU affinity (including WSL2)")
    grace = resources.get('termination_grace_seconds', 0)
    if type(grace) is not int or not 0 <= grace < resources['wall_seconds']:
        raise ConfigurationError('termination grace must fit inside the wall budget')
    begin, peak, reason = time.monotonic(), 0, None
    child = subprocess.Popen(command, cwd=cwd, env=env, stdout=stdout, stderr=subprocess.STDOUT, start_new_session=True)
    try:
        os.sched_setaffinity(child.pid, {min(os.sched_getaffinity(0))})
        while child.poll() is None:
            peak = max(peak, process_group_rss(child.pid))
            if peak > resources["memory_mib"] * 1024**2:
                reason = "host_rss_limit"
                break
            if time.monotonic() - begin > resources["wall_seconds"] - grace:
                reason = "wall_limit"
                if grace:
                    try:
                        os.killpg(child.pid, signal.SIGTERM)
                    except ProcessLookupError:
                        pass
                    while child.poll() is None and time.monotonic()-begin < resources['wall_seconds']:
                        peak = max(peak, process_group_rss(child.pid))
                        if peak > resources['memory_mib'] * 1024**2:
                            reason = 'host_rss_limit'
                            break
                        time.sleep(0.1)
                break
            time.sleep(0.2)
    finally:
        # Also clean up descendants after the main worker exits.
        try:
            os.killpg(child.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        child.wait()
    return subprocess.CompletedProcess(command, child.returncode), {
        "cpu_affinity": 1, "memory_mib": resources["memory_mib"], "wall_seconds": resources["wall_seconds"],
        "peak_host_rss_bytes": peak, "stop_reason": reason,
        "termination_grace_seconds": grace,
        "elapsed_wall_seconds": time.monotonic() - begin,
        "host_memory_enforcement": "aggregate process-group /proc RSS sampled at 0.2 s; not a hard cgroup cap"}


def run_cuda_worker(command, *, cwd, env, stdout, resources):
    """CUDA virtual reservations cannot use CPU RLIMIT_AS; supervise host RSS."""
    result, receipt = run_process_group(command, cwd=cwd, env=env, stdout=stdout,
                                        resources=resources)
    return result, receipt | {
        "gpu_allocator_fraction": 0.40,
        "gpu_memory_enforcement": "JAX allocator fraction; not a whole-device memory limit"}
