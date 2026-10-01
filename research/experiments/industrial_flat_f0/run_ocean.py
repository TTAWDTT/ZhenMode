"""One frozen full production solver trajectory; no surrogate evolution."""
# ruff: noqa: E402 -- CPU selection and import paths must precede JAX/core imports.
import argparse
import hashlib
import json
import os
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace

os.environ['JAX_PLATFORMS'] = 'cpu'
START = time.monotonic()
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'research/experiments/standing_wave_v0'))

import jax
import jax.numpy as jnp
import numpy as np
from native_output import NativeSnapshots
from prepare_inputs import native_arrays
from score import contract

from config import PhysicsConfig
from jax_solver_global import make_solver_global


def engineering_gate(state, inputs, frozen):
    values = {name: np.asarray(getattr(state, name)) for name in ('u', 'v', 'T', 'S', 'eta', 'ice')}
    if not all(np.all(np.isfinite(value)) for value in values.values()):
        raise ValueError('nonfinite native state')
    h = np.broadcast_to(inputs['h0_m'], (64, 8, 4)).copy()
    h[..., 0] += values['eta']
    volume0 = float(np.sum(inputs['area_m2']) * 100.)
    volume_error = abs(float(np.sum(values['eta'] * inputs['area_m2']))) / volume0
    tracer_error = max(float(np.max(np.abs(values['T'] - 15.))) / 15.,
                       float(np.max(np.abs(values['S'] - 35.))) / 35.)
    speed = .01 * np.sqrt(9.81 / 100.)
    v_ratio = float(np.max(np.abs(values['v']))) / speed
    if (np.any(h <= 0.) or volume_error > frozen['thresholds']['volume']
            or tracer_error > frozen['thresholds']['tracer']
            or v_ratio > frozen['thresholds']['v_over_U']
            or np.any(values['ice'] != 0.)):
        raise ValueError('frozen engineering gate rejected native state: ' + json.dumps(dict(
            min_h=float(h.min()), volume_error=volume_error, tracer_error=tracer_error, v_ratio=v_ratio)))
    return dict(passed=True, kind='explicit frozen engineering guard; not a native rollback flag',
                min_h_m=float(h.min()), volume_error=volume_error,
                tracer_error=tracer_error, v_over_U=v_ratio)


def run(directory, input_file):
    frozen = contract('coarse')
    expected = native_arrays(0.)
    with np.load(input_file, allow_pickle=False) as archive:
        if set(archive.files) != set(expected):
            raise ValueError('frozen input members differ')
        inputs = {key: archive[key] for key in archive.files}
    for name in expected:
        np.testing.assert_array_equal(inputs[name], expected[name])
    grid = SimpleNamespace(nx=64, ny=8, nz=4, x_m=inputs['x_m'], y_m=inputs['y_m'],
                           z=inputs['z_m'], dz=inputs['dz_m'], dx_2d=inputs['dx_2d_m'],
                           dy=float(inputs['dy_m']), cos_lat=inputs['cos_metric'],
                           f=inputs['f_s_inverse'], depth=inputs['depth_m'],
                           wet_mask_3d=inputs['wet'], wet_mask=inputs['wet'][..., 0],
                           ocean_mask=np.ones((64, 8), dtype=bool),
                           land_mask=np.zeros((64, 8), dtype=bool))
    physics = replace(PhysicsConfig(), nu_h=0., nu_v=0., nu_bi=0., kappa_h=0.,
                      kappa_v=0., kappa_bi=0., kappa_conv=0., kappa_gm=0.,
                      kappa_redi=0., r_bot=0., cd=0., T_ref=15., S_ref=35.)
    if any(device.platform != 'cpu' for device in jax.devices()):
        raise ValueError('CPU-only contract')
    step, initialize, _, params, _ = make_solver_global(
        grid, physics, 100., dtype='float64', return_params=True, **frozen['ocean_options'])
    np.testing.assert_allclose(np.asarray(params.dz_node).ravel(), inputs['h0_m'], rtol=0., atol=1.e-12)
    np.testing.assert_array_equal(np.asarray(params.H_sw), 100.)
    state = initialize(inputs['T_degC'], inputs['S_psu'])._replace(eta=jnp.asarray(inputs['eta_m']))
    jax.block_until_ready(state)
    # Lower/compile executes no numerical step. There is no warm-up trajectory.
    compiled = step.lower(state).compile()
    configuration = dict(options=frozen['ocean_options'], physics=asdict(physics),
                         input_sha256=hashlib.sha256(input_file.read_bytes()).hexdigest(),
                         source_module_sha256=hashlib.sha256((ROOT / 'src/jax_solver_global.py').read_bytes()).hexdigest(),
                         jax_version=jax.__version__, dt_s=100., steps=320,
                         devices=[str(device) for device in jax.devices()],
                         filters='native FFT 2/3 zonal and five-point meridional dealias filter',
                         numerical_scheme='unmodified full _step_impl legacy forward-backward surface',
                         thickness='static nodal dual; eta-adjusted geometry is a derived diagnostic')
    writer = NativeSnapshots(directory / 'native', inputs, configuration)
    writer.write(0, state, accepted=True, gate_evidence=engineering_gate(state, inputs, frozen))
    initialization_s = time.monotonic() - START
    integration_s = 0.
    for index in range(1, 321):
        began = time.monotonic()
        state = compiled(state)
        jax.block_until_ready(state)
        integration_s += time.monotonic() - began
        evidence = engineering_gate(state, inputs, frozen)
        if index % 10 == 0:
            writer.write(index * 100, state, accepted=True, gate_evidence=evidence)
    (directory / 'trajectory_receipt.json').write_text(json.dumps(dict(
        initialization_s=initialization_s, integration_s=integration_s,
        child_total_wall_s=time.monotonic() - START, completed_steps=320,
        native_configuration=configuration, qualification_passed=False), indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-directory', type=Path, required=True)
    parser.add_argument('--input-file', type=Path, required=True)
    args = parser.parse_args()
    run(args.run_directory, args.input_file)
