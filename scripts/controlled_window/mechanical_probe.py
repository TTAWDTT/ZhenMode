"""Two independent original dt300 attempts; private stage arrays, no trajectory."""
# ruff: noqa: E402
import os

os.environ.update(JAX_PLATFORMS='cpu', JAX_ENABLE_X64='true', OMP_NUM_THREADS='1',
                  OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
import argparse
import hashlib
import json
import sys
import time
from io import BytesIO
from pathlib import Path

import numpy as np


def sha(content):
    return hashlib.sha256(content).hexdigest()


def load_arrays(content):
    with np.load(BytesIO(content), allow_pickle=False) as values:
        return {key: values[key].copy() for key in values.files}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('source_dir', 'archive_dir', 'run_dir', 'output_dir', 'original_protocol', 'protocol'):
        parser.add_argument('--'+key.replace('_', '-'), type=Path, required=True)
    parser.add_argument('--phase', choices=['baseline', 'instrumented'], required=True)
    parser.add_argument('--protocol-sha256', required=True)
    args = parser.parse_args()
    protocol_bytes = args.protocol.read_bytes()
    if sha(protocol_bytes) != args.protocol_sha256:
        raise ValueError('probe protocol changed')
    protocol = json.loads(protocol_bytes)
    if sha(Path(__file__).read_bytes()) != protocol['runner_sha256']:
        raise ValueError('probe runner changed')
    original_bytes = args.original_protocol.read_bytes()
    if sha(original_bytes) != protocol['original_protocol_sha256']:
        raise ValueError('original protocol changed')
    original = json.loads(original_bytes)
    hashes = protocol['instrumented_source_hashes'] if args.phase == 'instrumented' else {
        Path(key).name: (original['instrumented_material_sha256'] if key == 'src/material_top.py' else value)
        for key, value in original['historical_source_hashes'].items()}
    for name, expected in hashes.items():
        if sha((args.source_dir / name).read_bytes()) != expected:
            raise ValueError('source changed: '+name)
    base = {}
    for name, expected in original['base_inputs'].items():
        content = (args.archive_dir / name).read_bytes()
        if sha(content) != expected:
            raise ValueError('base input changed: '+name)
        base[name] = content
    arrays = load_arrays(base['parameter_arrays_used.npz'])
    scalars = json.loads(base['manifest_start.json'])['scalar_params']
    sys.path.insert(0, str(args.source_dir))
    import jax
    import jax.numpy as jnp

    import jax_solver_global as solver
    from material_top import make_observed_material_top_step

    assert jax.default_backend() == 'cpu' and jax.config.jax_enable_x64
    params = solver.FDPhysParams(**{
        key: jnp.asarray(arrays[key]) if key in arrays else scalars[key]
        for key in solver.FDPhysParams._fields})
    projection_calls = []
    original_projection = solver._project_column_divergence

    def observed_projection(*values, **options):
        projection_calls.append('traced')
        return original_projection(*values, **options)

    solver._project_column_divergence = observed_projection
    step = make_observed_material_top_step(params)
    phase_dir = args.output_dir / args.phase
    phase_dir.mkdir(exist_ok=False)
    records = []
    for label, relative in [('t0', 'round_000/I0B0.npz'), ('6h', 'round_072/I0B0-attempted.npz')]:
        content = (args.run_dir / relative).read_bytes()
        if sha(content) != protocol['checkpoint_sha256'][label]:
            raise ValueError('checkpoint changed')
        state = solver.JaxStateG(**{key: jnp.asarray(value) for key, value in load_arrays(content).items()})
        started = time.perf_counter()
        result = step(state)
        jax.block_until_ready(result)
        wall = time.perf_counter() - started
        outcome = result[0]
        fields = {key: np.asarray(value) for key, value in outcome.attempted_state._asdict().items()}
        returned = {key: np.asarray(value) for key, value in outcome.state._asdict().items()}
        np.savez_compressed(phase_dir / (label+'-attempted.npz'), **fields)
        np.savez_compressed(phase_dir / (label+'-returned.npz'), **returned)
        checks = {key: np.asarray(value).item() for key, value in outcome.checks.items()}
        record = dict(label=label, valid=bool(outcome.valid), checks=checks,
                      step_wall_seconds=wall, projection_trace_calls=len(projection_calls),
                      checkpoint_sha256=sha(content))
        if args.phase == 'instrumented':
            probe = result[2]
            raw = {f'outer_{index}_{key}': np.asarray(value)
                   for index, stage in enumerate(probe['outer']) for key, value in stage.items()}
            raw.update({'fast_'+key: np.asarray(value) for key, value in probe['fast'].items()})
            np.savez_compressed(phase_dir / (label+'-stages-private.npz'), **raw)
            reference = load_arrays((args.output_dir / 'baseline' / (label+'-attempted.npz')).read_bytes())
            reference_returned = load_arrays((args.output_dir / 'baseline' / (label+'-returned.npz')).read_bytes())
            differences = {key: float(np.max(np.abs(value-reference[key]))) for key, value in fields.items()}
            returned_differences = {key: float(np.max(np.abs(value-reference_returned[key]))) for key, value in returned.items()}
            baseline = json.loads((args.output_dir / 'baseline' / 'records.json').read_text())
            matched = next(item for item in baseline if item['label'] == label)
            flags_equal = record['valid'] == matched['valid'] and all(
                checks[key] == value for key, value in matched['checks'].items() if isinstance(value, bool))
            arrays_finite = all(np.all(np.isfinite(value)) for collection in
                                (fields, returned, reference, reference_returned) for value in collection.values())
            differences_pass = all(np.isfinite(value) and value <= 1e-11
                                   for value in [*differences.values(), *returned_differences.values()])
            record['equivalence'] = dict(max_field_difference=differences, max_returned_difference=returned_differences,
                                         boolean_flags_equal=flags_equal, all_arrays_finite=bool(arrays_finite),
                                         passed=bool(flags_equal and arrays_finite and differences_pass))
            if not record['equivalence']['passed']:
                raise ValueError('instrumentation changed numerical result')
        records.append(record)
        (phase_dir / 'records.json').write_text(json.dumps(records, indent=2, allow_nan=False)+'\n', encoding='utf-8')
        print(json.dumps(record, allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
