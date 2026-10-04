"""Bounded synthetic production restart check.

Uses the existing 8x8x4 driver fixture; four independent CPU processes execute
8 steps continuously and 2+2+4 steps with two strict checkpoint continuations.
No downloads, external devices or historical data. Outputs are create-only.
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from zhenmode.provenance.sources import current_source_files

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT)]


def source_hashes():
    """Hash actual implementations, helpers and this producer."""
    files = current_source_files(ROOT, [Path(__file__).resolve()])
    return {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in files.items()}


def worker(directory, phase):
    import jax
    import numpy as np
    import pytest

    import zhenmode.model.audit.stages as owner_stages
    import zhenmode.model.runtime.integration as owner_monitor
    from tests.support.driver import run_controlled_driver
    from zhenmode.model.io.restart import fingerprint

    if jax.default_backend() != 'cpu':
        raise RuntimeError('CPU witness must run on CPU')
    original = owner_stages.make_budget_step
    hlo = {}
    def capture(params):
        advance = original(params)
        def step(*args, **kwargs):
            if not hlo:
                lowered = advance.lower(*args, **kwargs)
                text = str(lowered.compiler_ir(dialect='stablehlo'))
                hlo['stablehlo_sha256'] = hashlib.sha256(text.encode()).hexdigest()
                hlo['abstract_arguments'] = [{'shape': list(a.shape), 'dtype': str(a.dtype),
                                             'weak_type': bool(getattr(a, 'weak_type', False))}
                                            for a in jax.tree.leaves((args, kwargs))]
            return advance(*args, **kwargs)
        return step
    endpoint = {}
    classify = owner_monitor.classify_state
    def capture_state(state, *args, **kwargs):
        metrics = classify(state, *args, **kwargs)
        if not int(metrics.failure):
            endpoint.update({name: np.asarray(value).copy() for name, value in zip(state._fields, state, strict=True)})
        return metrics
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(owner_monitor, 'classify_state', capture_state)
        patch.setattr(owner_stages, 'make_budget_step', capture)
        grid, contract = run_controlled_driver(
            patch, directory, restart=directory / 'ckpt_controlled.npz' if phase in ('second', 'last') else None,
            crash_after=3 if phase in ('first', 'second') else None,
            options=('--budget-audit', '--wind-jit', '--use-scan'))
    with (directory / f"{phase}_endpoint.npz").open("xb") as stream:
        np.savez_compressed(stream, **endpoint)
    (directory / f'{phase}_identity.json').write_text(json.dumps({
        **hlo, 'contract': contract, 'grid_identity': fingerprint(grid),
        'synthetic_input': 'all wet 8x8x4; T=17 C, S=35; monthly zonal stress (month+1)*0.001 N/m2',
        'numpy': np.__version__}, indent=2, sort_keys=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--worker', choices=('continuous', 'first', 'second', 'last'))
    args = parser.parse_args()
    if args.worker:
        worker(args.output, args.worker)
        return
    args.output.mkdir(parents=True, exist_ok=False)
    env = {**os.environ, 'JAX_PLATFORMS': 'cpu', 'OPENBLAS_NUM_THREADS': '1',
           'OMP_NUM_THREADS': '1'}
    durations = {}
    for phase in ('continuous', 'first', 'second', 'last'):
        directory = args.output / ('continuous' if phase == 'continuous' else 'resumed')
        start = time.monotonic()
        with (args.output / f'{phase}.log').open('x') as log:
            subprocess.run([sys.executable, __file__, '--worker', phase, '--output', str(directory)],
                           env=env, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=120)
        durations[phase] = time.monotonic() - start
    import numpy as np
    with np.load(args.output / 'continuous/global_controlled.npz') as left, np.load(args.output / 'resumed/global_controlled.npz') as right:
        equal = {name: left[name].shape == right[name].shape and left[name].dtype == right[name].dtype
                 and left[name].tobytes() == right[name].tobytes() for name in left.files}
        assert set(left.files) == set(right.files)
    with np.load(args.output / 'continuous/continuous_endpoint.npz') as left, np.load(args.output / 'resumed/last_endpoint.npz') as right:
        for name in ('u', 'v', 'T', 'S', 'eta', 'ice'):
            equal['raw_endpoint_' + name] = (left[name].dtype == right[name].dtype
                                            and left[name].shape == right[name].shape
                                            and left[name].tobytes() == right[name].tobytes())
    files = {str(path.relative_to(args.output)): hashlib.sha256(path.read_bytes()).hexdigest()
             for path in sorted(args.output.rglob('*')) if path.is_file()}
    report = {'scope': 'synthetic CPU 8x8x4 80 seconds; not historical 30-day CUDA replay',
              'all_saved_fields_byte_equal': all(equal.values()), 'field_byte_equal': equal,
              'process_wall_seconds': durations, 'artifact_sha256': files,
              'accepted_ocean_steps': 16, 'industrial_qualification': False,
              'source_sha256': source_hashes()}
    (args.output / 'report.json').write_text(json.dumps(report, indent=2, sort_keys=True))
    print(json.dumps({'all_saved_fields_byte_equal': all(equal.values()), 'process_wall_seconds': durations}))
    if not all(equal.values()):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
