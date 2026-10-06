"""Budget negatives and actual Linux process supervision controls."""

import os
import subprocess
import sys
import time

import pytest

from zhenmode.execution.resources import run_cuda_worker, run_process_group, validate_budget
from zhenmode.execution.schema import ConfigurationError


def test_gpu_is_explicit_and_still_bounded():
    annual = {"cpu": 1, "wall_seconds": 10800, "memory_mib": 8192}
    validate_budget(annual, "cuda")
    with pytest.raises(ConfigurationError):
        validate_budget(annual, "cpu")
    for name, value in (("cpu", 2), ("wall_seconds", 10801), ("memory_mib", 8193)):
        with pytest.raises(ConfigurationError):
            validate_budget(annual | {name: value}, "cuda")
    with pytest.raises(ConfigurationError):
        validate_budget(annual, "automatic")


@pytest.mark.skipif(os.name != "posix", reason="Linux /proc supervisor")
@pytest.mark.parametrize("kind", ["wall", "memory", "complete"])
def test_actual_supervision(tmp_path, kind):
    script = "import time; time.sleep(30)" if kind == "wall" else "allocation = bytearray(64*1024**2); import time; time.sleep(30)" if kind == "memory" else "print('finished')"
    resources = {"cpu": 1, "wall_seconds": 1 if kind == "wall" else 10, "memory_mib": 16 if kind == "memory" else 256}
    begin = time.monotonic()
    with (tmp_path / "log").open("w") as log:
        result, receipt = run_cuda_worker([sys.executable, "-c", script], cwd=tmp_path, env=os.environ, stdout=log, resources=resources)
    assert time.monotonic() - begin < 15
    assert receipt["stop_reason"] == {"wall": "wall_limit", "memory": "host_rss_limit", "complete": None}[kind]
    assert (result.returncode == 0) == (kind == "complete")
    assert isinstance(result, subprocess.CompletedProcess)


@pytest.mark.skipif(os.name != 'posix', reason='Linux /proc supervisor')
def test_build_cleanup_grace_stays_inside_wall_budget(tmp_path):
    script = ("import signal,time,sys; from pathlib import Path; "
              "signal.signal(signal.SIGTERM, lambda *args: (Path('cleanup').write_text('done'),sys.exit(42))); "
              "time.sleep(30)")
    with (tmp_path / 'log').open('w') as log:
        result, receipt = run_process_group([sys.executable, '-c', script], cwd=tmp_path,
                    env=os.environ, stdout=log,
                    resources={'cpu': 1, 'wall_seconds': 2, 'memory_mib': 256, 'termination_grace_seconds': 1})
    assert result.returncode == 42 and (tmp_path / 'cleanup').read_text() == 'done'
    assert receipt['stop_reason'] == 'wall_limit' and receipt['elapsed_wall_seconds'] < 3


@pytest.mark.skipif(os.name != 'posix', reason='Linux /proc supervisor')
def test_grace_cannot_extend_the_wall_budget(tmp_path):
    with (tmp_path / 'log').open('w') as log, pytest.raises(ConfigurationError, match='inside'):
        run_process_group(['invalid'], cwd=tmp_path, env=os.environ, stdout=log,
                          resources={'cpu': 1, 'wall_seconds': 2, 'memory_mib': 256, 'termination_grace_seconds': 2})
