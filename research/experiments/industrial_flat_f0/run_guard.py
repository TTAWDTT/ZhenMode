"""Linux-only bounded launcher for the dedicated non-root experiment account.

The inherited run marker follows MPI children across sessions/process groups.
All same-UID processes are counted for a conservative RSS bound; unexpected
untagged processes stop the run. Only identified run descendants are signalled.
This account must be idle before launch. No benchmark is launched on import.
"""
import argparse
import fcntl
import hashlib
import json
import os
import resource
import signal
import subprocess
import time
import uuid
from pathlib import Path


def effective_uid(directory):
    # /proc directory ownership can become root for a non-dumpable ordinary
    # process. Use the kernel's effective-UID field instead of inode ownership.
    line = next(line for line in (directory / 'status').read_text().splitlines()
                if line.startswith('Uid:'))
    return int(line.split()[2])


def user_processes(uid, *, require_environment=True):
    """Return PID/start identity, parent, RSS and environment for this UID only."""
    result = {}
    for directory in Path('/proc').iterdir():
        if not directory.name.isdigit():
            continue
        try:
            if effective_uid(directory) != uid:
                continue
            fields = (directory / 'stat').read_text().rsplit(')', 1)[1].split()
            if fields[0] == 'Z':
                continue
            record = dict(
                start=int(fields[19]), parent=int(fields[1]),
                rss=int((directory / 'statm').read_text().split()[1])
                    * os.sysconf('SC_PAGE_SIZE'))
            try:
                record['environment'] = (directory / 'environ').read_bytes().split(b'\0')
            except PermissionError:
                if require_environment:
                    raise
                record['environment'] = None
            result[int(directory.name)] = record
        except FileNotFoundError:
            continue
        except PermissionError as error:
            raise RuntimeError('cannot monitor an experiment-account process') from error
    return result


def extend_identities(live, tracked, marker, launcher_identity):
    """Bind each lineage edge to the currently observed parent PID/start pair."""
    for _ in range(len(live) + 1):
        old_count = len(tracked)
        for pid, record in live.items():
            parent = live.get(record['parent'])
            known_parent = (parent is not None
                            and tracked.get(record['parent']) == parent['start'])
            tagged = record['environment'] is not None and marker in record['environment']
            if tagged or (pid, record['start']) == launcher_identity or known_parent:
                tracked[pid] = record['start']
        if len(tracked) == old_count:
            break


def output_bytes(directory):
    total = 0
    for base, directories, files in os.walk(directory, followlinks=False):
        for name in directories + files:
            path = Path(base) / name
            if path.is_symlink():
                raise ValueError('run output contains an unaccounted symlink')
            if path.is_file():
                total += path.stat().st_size
    return total


def signal_identified(identities, uid, signum):
    for pid, start in identities.items():
        try:
            directory = Path('/proc') / str(pid)
            fields = (directory / 'stat').read_text().rsplit(')', 1)[1].split()
            if effective_uid(directory) == uid and int(fields[19]) == start:
                os.kill(pid, signum)
        except (OSError, ValueError, StopIteration, IndexError):
            continue


def stop_identified(process, tracked, marker, uid, launcher_identity):
    """Refresh TERM-time descendants for at most two seconds, then confirm cleanup."""
    if process.poll() is None:
        process.terminate()
    deadline = time.monotonic() + 2.
    confirmed = False
    try:
        while True:
            live = user_processes(uid, require_environment=False)
            live.pop(os.getpid(), None)
            extend_identities(live, tracked, marker, launcher_identity)
            remaining = {pid: record['start'] for pid, record in live.items()
                         if tracked.get(pid) == record['start']}
            if not remaining:
                break
            signal_identified(remaining, uid, signal.SIGKILL)
            if time.monotonic() >= deadline:
                break
            time.sleep(.05)
        final = user_processes(uid, require_environment=False)
        final.pop(os.getpid(), None)
        extend_identities(final, tracked, marker, launcher_identity)
        confirmed = not final
    except Exception:
        # Monitoring cannot establish cleanup; never suppress child cleanup.
        confirmed = False
    finally:
        signal_identified(tracked, uid, signal.SIGKILL)
        if process.poll() is None:
            process.kill()
        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            confirmed = False
    return confirmed


def guarded_run(command, directory, *, wall_s=600, rss_bytes=4 * 1024**3,
                output_limit=128 * 1024**2, output_budget_directory=None):
    if os.getuid() == 0:
        raise ValueError('run guard requires an ordinary user; no root override')
    if (not command or '--allow-run-as-root' in command
            or any(key.startswith('OMPI_ALLOW_RUN_AS_ROOT') for key in os.environ)):
        raise ValueError('root overrides are forbidden')
    for value, maximum in ((wall_s, 600), (rss_bytes, 4 * 1024**3),
                           (output_limit, 128 * 1024**2)):
        if not isinstance(value, (int, float)) or not 0 < value <= maximum:
            raise ValueError('invalid or expanded run budget')
    directory = Path(directory).resolve(strict=True)
    if directory.stat().st_uid != os.getuid():
        raise ValueError('run directory must belong to the experiment account')
    budget_directory = (Path(output_budget_directory).resolve(strict=True)
                        if output_budget_directory is not None else directory)
    if (budget_directory.stat().st_uid != os.getuid()
            or not directory.is_relative_to(budget_directory)):
        raise ValueError('shared output budget must contain the owned run directory')
    uid = os.getuid()
    lock_path = Path.home() / '.ocean-flat-wave.lock'
    with lock_path.open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        before = user_processes(uid)
        if set(before) != {os.getpid()}:
            raise ValueError('experiment account is not idle; do not compete with other jobs')
        if output_bytes(budget_directory) >= output_limit:
            raise ValueError('initial files exceed output budget')
        report_path = directory / 'guard_result.json'
        if report_path.exists():
            raise ValueError('refusing to overwrite an existing run result')
        return _launch(command, directory, uid, wall_s, int(rss_bytes),
                       int(output_limit), report_path, budget_directory)


def _launch(command, directory, uid, wall_s, rss_bytes, output_limit, report_path,
            budget_directory):
    marker = ('OCEAN_BENCHMARK_RUN_ID=' + uuid.uuid4().hex).encode()
    environment = os.environ.copy()
    environment[marker.split(b'=', 1)[0].decode()] = marker.split(b'=', 1)[1].decode()
    environment.update(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')

    def child_limits():
        resource.setrlimit(resource.RLIMIT_AS, (rss_bytes, rss_bytes))
        resource.setrlimit(resource.RLIMIT_FSIZE, (output_limit, output_limit))
        os.nice(10)
        os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})

    start = time.monotonic()
    tracked = {}
    peak_rss = peak_output = 0
    reason = None
    with (directory / 'guard_stdout.log').open('xb') as log:
        process = subprocess.Popen(command, cwd=directory, env=environment,
                                   stdout=log, stderr=subprocess.STDOUT,
                                   start_new_session=True, preexec_fn=child_limits)
        try:
            fields = (Path('/proc') / str(process.pid) / 'stat').read_text().rsplit(')', 1)[1].split()
            launcher_identity = (process.pid, int(fields[19]))
            tracked[process.pid] = launcher_identity[1]
        except FileNotFoundError:
            launcher_identity = None
        try:
            while True:
                live = user_processes(uid)
                peak_rss = max(peak_rss, sum(record['rss'] for record in live.values()))
                live.pop(os.getpid(), None)
                extend_identities(live, tracked, marker, launcher_identity)
                unknown = [pid for pid, record in live.items()
                           if tracked.get(pid) != record['start']]
                peak_output = max(peak_output, output_bytes(budget_directory))
                if unknown:
                    reason = 'unexpected account process: monitoring identity uncertain'
                elif peak_rss > rss_bytes:
                    reason = 'aggregate RSS budget'
                elif peak_output >= output_limit:
                    reason = 'aggregate output budget'
                elif time.monotonic() - start >= wall_s:
                    reason = 'wall timeout'
                if reason:
                    signal_identified(tracked, uid, signal.SIGTERM)
                    break
                if process.poll() is not None:
                    remaining = [pid for pid in live if pid != process.pid]
                    if remaining:
                        reason = 'launcher exited with live identified descendants'
                    break
                time.sleep(.05)
        except Exception as error:
            reason = 'monitoring failure: ' + type(error).__name__
        finally:
            cleanup_confirmed = stop_identified(process, tracked, marker, uid, launcher_identity)
    if not cleanup_confirmed and reason is None:
        reason = 'descendant cleanup not confirmed'
    report = dict(schema='ocean.bounded_native_run.v1', uid=uid,
                  command=command, exit_code=process.returncode, stopped_reason=reason,
                  wall_s=time.monotonic() - start, sampled_aggregate_peak_rss_bytes=peak_rss,
                  sampled_peak_output_bytes=peak_output, sample_interval_s=.05,
                  descendants_identified=len(tracked), wall_limit_s=wall_s,
                  descendant_cleanup_confirmed=cleanup_confirmed,
                  rss_limit_bytes=rss_bytes, output_limit_bytes=output_limit,
                  output_budget_directory=str(budget_directory),
                  guard_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  qualification_passed=False,
                  limitation='Sampled RSS/output bounds can overshoot between samples; '
                  'per-process address/file limits additionally apply. No performance ranking.')
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-directory', type=Path, required=True)
    parser.add_argument('--wall-seconds', type=float, default=600)
    parser.add_argument('--output-limit-bytes', type=int, default=128 * 1024**2)
    parser.add_argument('--output-budget-directory', type=Path,
                        help='shared root containing outputs from both models')
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    report = guarded_run(command, args.run_directory, wall_s=args.wall_seconds,
                         output_limit=args.output_limit_bytes,
                         output_budget_directory=args.output_budget_directory)
    print(json.dumps(report, allow_nan=False))
    raise SystemExit(0 if report['exit_code'] == 0 and report['stopped_reason'] is None else 2)


if __name__ == '__main__':
    main()
