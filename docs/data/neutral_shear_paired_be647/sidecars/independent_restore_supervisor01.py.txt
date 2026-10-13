"""One byte-only restore supervisor: one CPU, 180 s, sampled group RSS4GiB."""
import hashlib
import json
import os
from pathlib import Path
import resource
import signal
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
CAP = 4 * 1024**3


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write_new(path, data):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(data, stream, indent=2, allow_nan=False)
        stream.write('\n')


def group_rss(pgid):
    total = 0
    for item in Path('/proc').iterdir():
        if not item.name.isdigit():
            continue
        try:
            fields = (item / 'stat').read_text().rsplit(')', 1)[1].split()
            if int(fields[2]) == pgid:
                total += int(fields[21]) * os.sysconf('SC_PAGE_SIZE')
        except (OSError, ValueError, IndexError):
            continue
    return total


def main():
    declaration_path = HERE / 'INDEPENDENT_RESTORE_DECLARATION01.json'
    declaration = json.loads(declaration_path.read_text(encoding='utf-8-sig'))
    if declaration['status'] != 'AUTHORIZED_ONE_BYTE_RESTORE_ROOT_DELEGATION':
        raise ValueError('no byte-only authorization')
    for relative, digest in declaration['bound_file_sha256'].items():
        if sha(HERE / relative) != digest:
            raise ValueError('bound file changed: ' + relative)
    if declaration['resources'] != {'cpu': 1, 'wall_seconds': 180, 'memory_mib': 4096, 'sample_seconds': 0.2}:
        raise ValueError('resource declaration mismatch')
    os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
    resource.setrlimit(resource.RLIMIT_AS, (CAP, CAP))
    command = [sys.executable, str(HERE / 'independent_restore01.py')]
    launch = {'status': 'STARTED_NOT_COMPLETION', 'argv': command, 'cwd': str(HERE),
              'declaration_sha256': sha(declaration_path), 'supervisor_sha256': sha(__file__),
              'worker_sha256': sha(HERE / 'independent_restore01.py'), 'interpreter': sys.executable,
              'interpreter_sha256': sha(sys.executable), 'cpu_affinity': sorted(os.sched_getaffinity(0)),
              'resources': declaration['resources'], 'model_calls': 0}
    write_new(HERE / 'independent-restore01-launch.json', launch)
    started, peak, samples, stop = time.monotonic(), 0, 0, None
    with (HERE / 'independent-restore01.log').open('x', encoding='utf-8') as log:
        process = subprocess.Popen(command, cwd=HERE, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        while process.poll() is None:
            rss = group_rss(process.pid)
            peak = max(peak, rss)
            samples += 1
            elapsed = time.monotonic() - started
            if rss > CAP or elapsed > 180:
                stop = 'sampled_process_group_rss_exceeded' if rss > CAP else 'wall_seconds_exceeded'
                os.killpg(process.pid, signal.SIGKILL)
                break
            time.sleep(0.2)
        returncode = process.wait()
    result = {'status': 'COMPLETED_BYTE_SUPERVISION' if returncode == 0 and stop is None else 'FAIL_PRESERVED_NO_RETRY',
              'returncode': returncode, 'stop_reason': stop, 'elapsed_wall_seconds': time.monotonic()-started,
              'peak_sampled_child_group_rss_bytes': peak, 'rss_samples': samples,
              'cpu_affinity': sorted(os.sched_getaffinity(0)), 'resources': declaration['resources'],
              'host_memory_policy': '0.2-second aggregate child-process-group RSS, not a hard cgroup; inherited per-process RLIMIT_AS4GiB',
              'supervisor_peak_rss_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
              'launch_sha256': sha(HERE / 'independent-restore01-launch.json'),
              'declaration_sha256': sha(declaration_path), 'model_calls': 0, 'npz_decoded': 0}
    verdict = HERE / 'INDEPENDENT_RESTORE_VERDICT01.json'
    if verdict.exists():
        result['worker_verdict_sha256'] = sha(verdict)
    write_new(HERE / 'independent-restore01-resources.json', result)
    print(json.dumps(result))
    raise SystemExit(returncode if returncode else (1 if stop else 0))


if __name__ == '__main__':
    main()
