"""Bounded CPU full-step monitoring overhead control, never industrial evidence.

Each trial is a fresh subprocess. Baseline omits monitoring ONLY in this isolated
probe; it cannot qualify for the production gate. No driver/solver is patched.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import platform
import subprocess
import sys
import tempfile
from pathlib import Path
from statistics import median
from time import perf_counter

from contract import digest, file_digest

ROOT = Path(__file__).resolve().parents[2]
CONFIG = dict(nx=8, ny=8, nz=4, dt_s=10, dt_bt_s=5, steps=8, warmup_steps=2,
              precision='float64', mode_split=True, use_scan=True, polar_cap_rows=0,
              polar_cap_taper=0, initial='15 + 0.01*cos(2*pi*i/8) degC; S=35',
              forcing='PhysicsConfig defaults; zero imposed wind and heat',
              repeats=3, cache='fresh process, persistent cache disabled')


def worker(mode, destination):
    # Import/setup is included in total, outside cold compilation.
    start = perf_counter()
    sys.path.insert(0, str(ROOT / 'src'))
    import jax
    import jax.numpy as jnp
    import numpy as np

    from config import PhysicsConfig
    from grid import GlobalOceanGrid
    from integration_monitor import classify_state
    from jax_solver_global import make_solver_global

    if any(d.platform != 'cpu' for d in jax.devices()):
        raise RuntimeError('CPU-only probe')
    lat = np.linspace(-30, 30, 8)
    cos_lat = np.cos(np.radians(lat))
    shape = (8, 8)
    z = -np.linspace(50, 4000, 4)
    grid = GlobalOceanGrid(
        lon=np.linspace(.5, 359.5, 8), lat=lat,
        dx_2d=np.broadcast_to(6.371e6 * cos_lat * np.radians(45), shape).copy(),
        dy=float(6.371e6 * np.radians(lat[1] - lat[0])), cos_lat=cos_lat,
        f=np.broadcast_to(2 * 7.2921e-5 * np.sin(np.radians(lat)), shape).copy(),
        z=z, dz=-np.diff(z), nz=4, depth=np.full(shape, 4000.),
        wet_mask=np.ones(shape), ocean_mask=np.ones(shape, dtype=bool),
        land_mask=np.zeros(shape), wet_mask_3d=np.ones((8, 8, 4)), nx=8, ny=8)
    step, initialize, _ = make_solver_global(
        grid, PhysicsConfig(), dt=10., dt_bt=5., mode_split=True, use_scan=True,
        polar_cap_rows=0, polar_cap_taper=0, dtype='float64')
    temperature = np.broadcast_to(15 + .01 * np.cos(2 * np.pi * np.arange(8)[:, None, None] / 8),
                                  (8, 8, 4)).copy()
    state0 = initialize(T_init=temperature, S_init=np.full((8, 8, 4), 35.))
    jax.block_until_ready(state0)
    times = {'setup': perf_counter() - start}
    mark = perf_counter()
    compiled = step.lower(state0).compile()
    monitor = classify_state.lower(state0).compile() if mode == 'monitored' else None
    times['cold_compile'] = perf_counter() - mark

    def advance(state):
        proposed = compiled(state)
        if monitor is not None:
            metrics = monitor(proposed)
            # Match production host peak updates and first-rejection decision.
            peaks[0] = float(jnp.maximum(peaks[0], metrics.max_u))
            peaks[1] = float(jnp.maximum(peaks[1], metrics.max_velocity))
            if int(metrics.failure):
                raise RuntimeError('first-rejection monitor failed')
        return proposed

    peaks = [0., 0.]
    mark = perf_counter()
    state = state0
    for _ in range(CONFIG['warmup_steps']):
        state = advance(state)
    jax.block_until_ready(state)
    times['warmup'] = perf_counter() - mark
    # Both variants start measured integration from exactly the same initial bytes.
    peaks = [0., 0.]
    state = state0
    mark = perf_counter()
    if monitor is not None:
        incoming = monitor(state)
        peaks = [float(incoming.max_u), float(incoming.max_velocity)]
        if int(incoming.failure):
            raise RuntimeError('invalid incoming state')
    for _ in range(CONFIG['steps']):
        state = advance(state)
    jax.block_until_ready(state)
    times['integration'] = perf_counter() - mark
    mark = perf_counter()
    arrays = {name: np.asarray(value) for name, value in zip(state._fields, state)}
    if not all(np.isfinite(array).all() for array in arrays.values()):
        raise RuntimeError('nonfinite final state')
    np.savez(destination, **arrays)
    with open(destination, 'rb') as handle:
        os.fsync(handle.fileno())
    state_hashes = {name: hashlib.sha256(array.tobytes()).hexdigest()
                    for name, array in arrays.items()}
    times['diagnostics_io'] = perf_counter() - mark
    times['total'] = perf_counter() - start
    return dict(mode=mode, timings=times, state_sha256=state_hashes,
                device=str(jax.devices()), jax=jax.__version__, numpy=np.__version__,
                peak_memory={'status': 'unavailable', 'reason': 'no portable device peak allocator counter'})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True)
    parser.add_argument('--worker', choices=['unmonitored', 'monitored'])
    args = parser.parse_args()
    if args.worker:
        print(json.dumps(worker(args.worker, args.out), allow_nan=False))
        return 0
    if Path(args.out).exists():
        raise FileExistsError(args.out)
    sources = {str(path.relative_to(ROOT)): file_digest(path)
               for path in sorted((ROOT / 'src').glob('*.py'))}
    sources.update({str(path.relative_to(ROOT)): file_digest(path)
                    for path in sorted(Path(__file__).parent.glob('*.py'))})
    report = dict(status='BLOCKED', qualification_scope='synthetic_cpu_monitor_overhead_only',
                  industrial_qualified=False, config=CONFIG, config_sha256=digest(CONFIG),
                  source_sha=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                  source_clean=not bool(subprocess.check_output(
                      ['git', 'status', '--porcelain'], cwd=ROOT, text=True).strip()),
                  source_files=sources, source_tree_sha256=digest(sources),
                  platform=platform.platform(), python=sys.version, trials=[],
                  missing_dependencies=[name for name in ('jax', 'numpy', 'netCDF4')
                                        if importlib.util.find_spec(name) is None])
    if not report['missing_dependencies']:
        environment = dict(os.environ, JAX_PLATFORMS='cpu', JAX_ENABLE_X64='true',
                           JAX_ENABLE_COMPILATION_CACHE='false', OMP_NUM_THREADS='1',
                           OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
        try:
            with tempfile.TemporaryDirectory(prefix='ocean-monitor-') as temp:
                for repeat in range(CONFIG['repeats']):
                    modes = ['unmonitored', 'monitored']
                    if repeat % 2:
                        modes.reverse()
                    for mode in modes:
                        wall_start = perf_counter()
                        command = [sys.executable, str(Path(__file__).resolve()), '--worker', mode,
                                   '--out', str(Path(temp, f'{repeat}-{mode}.npz'))]
                        result = subprocess.run(command, cwd=ROOT, env=environment,
                                                text=True, capture_output=True, timeout=60)
                        if result.returncode:
                            raise RuntimeError(result.stderr[-4000:])
                        trial = json.loads(result.stdout.splitlines()[-1])
                        trial.update(pair_id=str(repeat), process_wall_s=perf_counter() - wall_start)
                        report['trials'].append(trial)
            hashes = [t['state_sha256'] for t in report['trials']]
            report['same_final_bytes'] = all(h == hashes[0] for h in hashes)
            report['status'] = 'MEASURED' if report['same_final_bytes'] else 'STATE_MISMATCH'
            report['integration_median_s'] = {
                mode: median(t['timings']['integration'] for t in report['trials'] if t['mode'] == mode)
                for mode in ('unmonitored', 'monitored')}
        except (RuntimeError, subprocess.TimeoutExpired, ValueError) as error:
            report.update(status='FAILED', error=str(error))
    with Path(args.out).open('x') as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
        handle.write('\n')
    print(json.dumps({key: report[key] for key in ('status', 'missing_dependencies', 'qualification_scope')}))
    return 0 if report['status'] == 'MEASURED' else 2


if __name__ == '__main__':
    raise SystemExit(main())
