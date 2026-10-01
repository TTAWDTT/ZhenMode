"""One bounded short-window worker; no scheduler or persistent manager."""

import argparse
import ctypes
import hashlib
import json
import shutil
import subprocess
import sys
import time
from ctypes import wintypes
from pathlib import Path

folder = Path(__file__).parent
parser = argparse.ArgumentParser()
parser.add_argument("--execute-reviewed", action="store_true")
parser.add_argument("--protocol-sha256", required=True)
parser.add_argument("--protocol", type=Path, required=True)
parser.add_argument("--output-root", type=Path, required=True)
parser.add_argument("--worker", type=Path, required=True)
parser.add_argument("--source-dir", type=Path, required=True)
parser.add_argument("--archive-dir", type=Path, required=True)
parser.add_argument("--evidence-dir", type=Path, required=True)
parser.add_argument("--donor-provenance", type=Path, required=True)
parser.add_argument("--raw-t", type=Path, required=True)
parser.add_argument("--raw-s", type=Path, required=True)
args = parser.parse_args()
started = time.perf_counter()
if not args.execute_reviewed:
    parser.error("Independent review execution gate is required")
protocol_bytes = args.protocol.read_bytes()
if hashlib.sha256(protocol_bytes).hexdigest() != args.protocol_sha256:
    parser.error("Protocol hash mismatch")
protocol = json.loads(protocol_bytes)
if args.worker.resolve() != (folder / protocol["worker_filename"]).resolve():
    parser.error("Worker identity mismatch")
for filename, digest in protocol["runner_hashes"].items():
    if hashlib.sha256((folder / filename).read_bytes()).hexdigest() != digest:
        parser.error("Runner hash mismatch")
resource = subprocess.check_output(
    [
        "powershell",
        "-NoProfile",
        "-Command",
        "(Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory",
    ],
    text=True,
).strip()
if int(resource) < 6 * 1024 * 1024:
    raise SystemExit("Insufficient free memory; no worker started")
if shutil.disk_usage(args.output_root).free < 8 * 1024**3:
    raise SystemExit("Insufficient private output storage; no worker started")
run = args.output_root / ("run-" + str(time.time_ns()))
run.mkdir()


class MemoryCounters(ctypes.Structure):
    _fields_ = [
        ("cb", wintypes.DWORD),
        ("PageFaultCount", wintypes.DWORD),
        ("PeakWorkingSetSize", ctypes.c_size_t),
        ("WorkingSetSize", ctypes.c_size_t),
        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
        ("PagefileUsage", ctypes.c_size_t),
        ("PeakPagefileUsage", ctypes.c_size_t),
    ]


kernel = ctypes.WinDLL("kernel32", use_last_error=True)
psapi = ctypes.WinDLL("psapi", use_last_error=True)
kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
kernel.OpenProcess.restype = wintypes.HANDLE
kernel.SetProcessAffinityMask.argtypes = [wintypes.HANDLE, ctypes.c_size_t]
kernel.CloseHandle.argtypes = [wintypes.HANDLE]
psapi.GetProcessMemoryInfo.argtypes = [
    wintypes.HANDLE,
    ctypes.POINTER(MemoryCounters),
    wintypes.DWORD,
]
log = (run / "worker.log").open("x", encoding="utf-8")
child = subprocess.Popen(
    [
        sys.executable,
        "-X",
        "utf8",
        str(args.worker),
        "--run-dir",
        str(run),
        "--protocol-sha256",
        args.protocol_sha256,
        "--protocol",
        str(args.protocol),
        "--source-dir",
        str(args.source_dir),
        "--archive-dir",
        str(args.archive_dir),
        "--evidence-dir",
        str(args.evidence_dir),
        "--donor-provenance",
        str(args.donor_provenance),
        "--raw-t",
        str(args.raw_t),
        "--raw-s",
        str(args.raw_s),
    ],
    stdout=log,
    stderr=subprocess.STDOUT,
    cwd=folder,
)
handle = kernel.OpenProcess(0x0400 | 0x0200 | 0x0010, False, child.pid)
peak = peak_commit = 0
reason = None
last_progress = {}
if not handle or not kernel.SetProcessAffinityMask(handle, 1):
    reason = "resource_guard_setup_failed"
try:
    while child.poll() is None:
        if handle:
            counter = MemoryCounters()
            counter.cb = ctypes.sizeof(counter)
            if psapi.GetProcessMemoryInfo(handle, ctypes.byref(counter), counter.cb):
                peak = max(peak, counter.PeakWorkingSetSize)
                peak_commit = max(peak_commit, counter.PeakPagefileUsage)
            else:
                reason = "memory_observation_failed"
        if peak > 4 * 1024**3 or peak_commit > 4 * 1024**3:
            reason = "global4GiB_guard"
        now = time.perf_counter()
        if now - started > protocol["limits"]["batch_wall_seconds"]:
            reason = "batch_wall_guard"
        progress = run / "progress.json"
        if progress.exists():
            try:
                last_progress = json.loads(progress.read_text())
            except (OSError, json.JSONDecodeError):
                pass
        case = last_progress.get("active_case")
        if case and last_progress.get("active_start_perf_counter") is not None:
            active = (
                last_progress["spent_case_seconds"][case]
                + now
                - last_progress["active_start_perf_counter"]
            )
            if active > protocol["limits"]["per_cell_cumulative_wall_seconds"]:
                reason = "case900s_guard:" + case
        if reason:
            child.terminate()
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
            break
        time.sleep(0.25)
finally:
    if handle:
        kernel.CloseHandle(handle)
    log.close()
result = {
    "run_directory": str(run),
    "pid": child.pid,
    "exit_code": child.returncode,
    "wall_seconds": time.perf_counter() - started,
    "peak_working_set_bytes": peak,
    "peak_commit_bytes": peak_commit,
    "cpu_affinity_mask": 1,
    "stop_reason": reason,
    "last_progress": last_progress,
    "budget_interrupted_attempt_availability": "Last committed and staged results retained; in-flight unreturned attempt unavailable if guard interrupted.",
    "protocol_sha256": args.protocol_sha256,
}
(run / "resources.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
print(json.dumps(result), flush=True)
