"""Write instantaneous native ocean fields, without stepping or averaging.

The FD model stores static nodal thickness. Any eta-adjusted geometric
thickness is labelled a derived diagnostic, never native moving-volume state.
"""
import hashlib
import json
import os
import tempfile
from pathlib import Path

import numpy as np


class NativeSnapshots:
    def __init__(self, directory, inputs, actual_config):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=False)
        self.inputs = inputs
        self.next_time_s = 0
        config = dict(actual_config)
        config.update(output='instantaneous native fields',
                      static_thickness_field='h0_m',
                      derived_thickness_field='h_geometric_m',
                      h_geometric_definition='h0 plus eta in the first dual cell; '
                      'not a native transported inventory', qualification_passed=False)
        (self.directory / 'actual_config.json').write_text(
            json.dumps(config, indent=2, allow_nan=False) + '\n', encoding='utf-8')
        np.savez(self.directory / 'native_metrics.npz', **{
            key: inputs[key] for key in ('x_m', 'y_m', 'z_m', 'h0_m', 'dx_2d_m',
                                       'dy_m', 'cos_metric', 'f_s_inverse',
                                       'area_m2', 'depth_m', 'wet')})

    def write(self, time_s, state, *, accepted, gate_evidence):
        if type(accepted) is not bool or not accepted:
            raise ValueError('refusing a rejected or unknown state')
        if not isinstance(gate_evidence, dict) or gate_evidence.get('passed') is not True:
            raise ValueError('acceptance gate evidence is required')
        if type(time_s) is not int or time_s != self.next_time_s or time_s > 32000:
            raise ValueError('expected t0 and successive instantaneous 1000s samples')
        values = {name: np.asarray(getattr(state, name))
                  for name in ('u', 'v', 'T', 'S', 'eta', 'ice')}
        for name, array in values.items():
            expected = (64, 8) if name in ('eta', 'ice') else (64, 8, 4)
            if array.shape != expected or not np.all(np.isfinite(array)):
                raise ValueError('native state has invalid shape or nonfinite values')
        h_geometry = np.broadcast_to(self.inputs['h0_m'], (64, 8, 4)).copy()
        h_geometry[..., 0] += values['eta']
        if np.any(h_geometry <= 0.):
            raise ValueError('geometric diagnostic lies outside the positive-thickness domain')
        payload = dict(values, time_s=np.array(time_s), h0_m=self.inputs['h0_m'],
                       h_geometric_m=h_geometry)
        destination = self.directory / f'instant_{time_s:05d}s.npz'
        with tempfile.TemporaryDirectory(prefix='.stage-', dir=self.directory) as temporary:
            staged = Path(temporary) / 'snapshot.npz'
            with staged.open('xb') as stream:
                np.savez(stream, **payload)
            os.link(staged, destination)
        evidence = dict(time_s=time_s, filename=destination.name,
                        bytes=destination.stat().st_size,
                        sha256=hashlib.sha256(destination.read_bytes()).hexdigest(),
                        gate=gate_evidence)
        # Append-only index; no snapshot or earlier evidence is overwritten.
        with (self.directory / 'snapshot_index.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(evidence, allow_nan=False) + '\n')
        self.next_time_s += 1000
        return evidence
