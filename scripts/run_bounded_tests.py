"""Windows CPU pytest runner: 180 s, one CPU, 4 GiB for the owned job tree.

Invoke with the project-local venv Python. Launch the underlying interpreter
directly with that venv's sys.path; measuring the Windows venv launcher alone
would miss its real child interpreter's memory usage.
"""
import ctypes
import json
import os
import subprocess
import sys
import sysconfig
import time
from ctypes import wintypes


class BasicLimits(ctypes.Structure):
    _fields_ = [('process_time', ctypes.c_int64), ('job_time', ctypes.c_int64),
                ('flags', wintypes.DWORD), ('minimum_working_set', ctypes.c_size_t),
                ('maximum_working_set', ctypes.c_size_t), ('active_process_limit', wintypes.DWORD),
                ('affinity', ctypes.c_size_t), ('priority', wintypes.DWORD),
                ('scheduling', wintypes.DWORD)]


class IoCounters(ctypes.Structure):
    _fields_ = [(name, ctypes.c_uint64) for name in
                ('read_ops', 'write_ops', 'other_ops', 'read_bytes', 'write_bytes', 'other_bytes')]


class JobLimits(ctypes.Structure):
    _fields_ = [('basic', BasicLimits), ('io', IoCounters),
                ('process_memory', ctypes.c_size_t), ('job_memory', ctypes.c_size_t),
                ('peak_process_memory', ctypes.c_size_t), ('peak_job_memory', ctypes.c_size_t)]


class MemoryCounters(ctypes.Structure):
    _fields_ = [('size', wintypes.DWORD), ('faults', wintypes.DWORD)] + [
        (name, ctypes.c_size_t) for name in
        ('peak_rss', 'rss', 'peak_paged', 'paged', 'peak_nonpaged', 'nonpaged',
         'private', 'peak_private')]


def run(arguments):
    if os.name != 'nt':
        raise RuntimeError('This runner uses Windows Job Objects; use native limits elsewhere.')
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    psapi = ctypes.WinDLL('psapi', use_last_error=True)
    kernel.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
    kernel.CreateJobObjectW.restype = wintypes.HANDLE
    kernel.SetInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
    kernel.QueryInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p,
                                               wintypes.DWORD, ctypes.c_void_p]
    kernel.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
    kernel.TerminateJobObject.argtypes = [wintypes.HANDLE, wintypes.UINT]
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel.GetProcessAffinityMask.argtypes = [wintypes.HANDLE, ctypes.c_void_p, ctypes.c_void_p]
    kernel.GetCurrentProcess.restype = wintypes.HANDLE
    psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.c_void_p, wintypes.DWORD]

    def checked(result):
        if not result:
            raise ctypes.WinError(ctypes.get_last_error())
        return result

    job = checked(kernel.CreateJobObjectW(None, None))
    process = None
    started = time.monotonic()
    peak_rss = peak_job = 0
    bound_stop = False
    try:
        mask, system_mask = ctypes.c_size_t(), ctypes.c_size_t()
        checked(kernel.GetProcessAffinityMask(kernel.GetCurrentProcess(), ctypes.byref(mask), ctypes.byref(system_mask)))
        limits = JobLimits()
        limits.basic.flags = 0x2000 | 0x200 | 0x10  # kill on close, job memory, affinity
        limits.basic.affinity = mask.value & -mask.value
        limits.job_memory = 4 * 1024 ** 3
        checked(kernel.SetInformationJobObject(job, 9, ctypes.byref(limits), ctypes.sizeof(limits)))
        # Preserve the invoking venv for subprocesses, while measuring the real interpreter.
        runner_directory = os.path.dirname(os.path.abspath(__file__))
        paths = [os.getcwd(), *(path for path in sys.path if os.path.abspath(path) != runner_directory)]
        bootstrap = ('import sys, site; sys.executable=' + repr(sys.executable) + '; sys.path[:]='
                     + repr(paths) + '; site.addsitedir('
                     + repr(sysconfig.get_path('purelib')) + '); sys.stdin.read(1); ')
        if arguments and arguments[0] == '--script':
            if len(arguments) < 2:
                raise ValueError('--script requires an absolute script filename')
            script = os.path.abspath(arguments[1])
            arguments = [script, *arguments[2:]]
            bootstrap += 'import runpy; runpy.run_path(sys.argv.pop(1), run_name="__main__")'
        elif arguments and arguments[0] == '--module':
            if len(arguments) < 2:
                raise ValueError('--module requires a module name')
            module = arguments[1]
            arguments = [module, *arguments[2:]]
            bootstrap += 'import runpy; runpy.run_module(sys.argv.pop(1), run_name="__main__")'
        else:
            bootstrap += 'import pytest; raise SystemExit(pytest.main(sys.argv[1:]))'
        environment = dict(os.environ, JAX_PLATFORMS='cpu', OMP_NUM_THREADS='1',
                           OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1')
        process = subprocess.Popen([sys._base_executable, '-B', '-c', bootstrap, *arguments],
                                   stdin=subprocess.PIPE, env=environment)
        # Child waits for one byte until its whole descendant tree is bounded.
        checked(kernel.AssignProcessToJobObject(job, wintypes.HANDLE(int(process._handle))))
        process.stdin.write(b'\n')
        process.stdin.close()
        while process.poll() is None:
            counters = MemoryCounters()
            counters.size = ctypes.sizeof(counters)
            checked(psapi.GetProcessMemoryInfo(wintypes.HANDLE(int(process._handle)),
                                              ctypes.byref(counters), counters.size))
            peak_rss = max(peak_rss, counters.peak_rss)
            checked(kernel.QueryInformationJobObject(job, 9, ctypes.byref(limits), ctypes.sizeof(limits), None))
            peak_job = max(peak_job, limits.peak_job_memory)
            if time.monotonic() - started > 180:
                checked(kernel.TerminateJobObject(job, 124))
                bound_stop = True
                break
            time.sleep(.1)
        code = process.wait()
        checked(kernel.QueryInformationJobObject(job, 9, ctypes.byref(limits), ctypes.sizeof(limits), None))
        peak_job = max(peak_job, limits.peak_job_memory)
        print('BOUNDED_RUN ' + json.dumps({'wall_s': round(time.monotonic() - started, 3),
              'peak_interpreter_RSS_bytes': peak_rss, 'peak_job_private_bytes': peak_job,
              'single_CPU': True, 'memory_limit_bytes': limits.job_memory, 'wall_limit_s': 180,
              'bound_stop': bound_stop, 'exit_code': code}))
        return code
    finally:
        if process is not None and process.poll() is None:
            kernel.TerminateJobObject(job, 124)
            process.wait()
        kernel.CloseHandle(job)


if __name__ == '__main__':
    raise SystemExit(run(sys.argv[1:]))
