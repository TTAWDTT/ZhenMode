"""Write an optional private quality sidecar; never invokes the ocean solver."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from input_quality import audit_woa_variable, enforce_strict_quality  # noqa: E402
from input_sources import (  # noqa: E402
    InputSnapshot,
    load_climatology_snapshot,
    load_snapshot_npz,
)


def file_identity(path):
    with path.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    return {'filename': path.name, 'bytes': path.stat().st_size, 'sha256': digest}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--temperature', type=Path, required=True)
    parser.add_argument('--salinity', type=Path, required=True)
    parser.add_argument('--grid', type=Path, required=True)
    parser.add_argument('--initial-state', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--strict', action='store_true')
    args = parser.parse_args(argv)
    # The existing reader prefers .npz twins; record the file actually consumed.
    sources = {label: (Path(str(path) + '.npz') if Path(str(path) + '.npz').exists()
                      else path).resolve()
               for label, path in [('T', args.temperature), ('S', args.salinity)]}
    masks_path = args.output.with_suffix('.masks.npz')
    protected = [*sources.values(), args.grid]
    if args.initial_state is not None:
        protected.append(args.initial_state)
    outputs = [args.output.resolve(), masks_path.resolve()]
    if len(set(outputs)) != 2 or any(p.resolve() in outputs for p in protected):
        parser.error('output paths must differ from all input paths')
    if args.output.exists() or masks_path.exists():
        parser.error('refusing to overwrite an existing report or mask artifact')
    try:
        snapshots = {label: InputSnapshot.read(path) for label, path in sources.items()}
        grid_snapshot = InputSnapshot.read(args.grid)
        initial_snapshot = (InputSnapshot.read(args.initial_state)
                            if args.initial_state is not None else None)
        return write_preflight(args, snapshots, grid_snapshot, initial_snapshot, masks_path)
    except (ValueError, OSError, KeyError) as error:
        print(f'Unsupported input preflight: {error}', file=sys.stderr)
        return 2


def write_preflight(args, snapshots, grid_snapshot, initial_snapshot, masks_path):
    grid = load_snapshot_npz(grid_snapshot)
    initial = {}
    if initial_snapshot is not None:
        loaded = load_snapshot_npz(initial_snapshot)
        initial = {key: loaded[key] for key in ('T', 'S')}
    report = {'schema': 'ocean.input_quality_bundle.v1', 'mode': 'strict' if args.strict
              else 'diagnostic', 'historical_raw_hashes_available': False, 'variables': {}}
    masks = {}
    for label in ('T', 'S'):
        raw = load_climatology_snapshot(snapshots[label], label)
        quality, variable_masks = audit_woa_variable(
            raw, grid, variable=label, initial_field=initial.get(label))
        quality['current_source_identity'] = snapshots[label].identity()
        report['variables'][label] = quality
        masks.update({f'{label}_{key}': value for key, value in variable_masks.items()})
    report['grid_identity'] = grid_snapshot.identity()
    report['processor_sha256'] = file_identity(Path(__file__).parents[1] /
                                             'src/input_quality.py')['sha256']
    report['reader_sha256'] = file_identity(Path(__file__).parents[1] /
                                          'src/input_sources.py')['sha256']
    report['entrypoint_sha256'] = file_identity(Path(__file__))['sha256']
    if initial_snapshot is not None:
        report['initial_state_identity'] = initial_snapshot.identity()
    for snapshot in [*snapshots.values(), grid_snapshot, initial_snapshot]:
        if snapshot is not None:
            snapshot.verify_unchanged()
    report['input_snapshots_unchanged_at_validation'] = True
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with masks_path.open('xb') as stream:
        np.savez_compressed(stream, **masks)
    report['private_mask_artifact'] = file_identity(masks_path)
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    for snapshot in [*snapshots.values(), grid_snapshot, initial_snapshot]:
        if snapshot is not None:
            snapshot.verify_unchanged()
    if args.strict:
        try:
            for quality in report['variables'].values():
                enforce_strict_quality(quality)
        except ValueError as error:
            print(str(error), file=sys.stderr)
            return 2
    print(f'Input quality sidecar: {args.output}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
