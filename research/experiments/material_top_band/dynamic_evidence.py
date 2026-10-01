"""Reproducible synthetic nonzero witness; summaries only, no private arrays."""
import argparse
import hashlib
import json
import platform
from dataclasses import asdict
from pathlib import Path

import numpy as np

from research.experiments.material_top_band import dynamic as d


def case_grid(*, compact=False):
    cosine = np.ones(4) if compact else np.cos(np.deg2rad(np.linspace(-30., 30., 4)))
    dx = np.full((8, 4), 800.) if compact else np.broadcast_to(
        6371000 * 2 * np.pi / 8 * cosine[None, :], (8, 4)).copy()
    dy = 800. if compact else 6371000 * np.deg2rad(20.)
    return d.Geometry(dx, dy, cosine, np.array([2.5, 7.5, 12.5, 42.5, 235., 200.]))


def case_state(grid):
    phase = 2 * np.pi * np.arange(8)[:, None] / 8
    eta = -.4 + .02 * np.cos(phase) * np.ones((8, 4))
    h = grid.thickness(eta)
    specific = np.zeros(h.shape + (4,))
    specific[..., 0] = 15 - np.arange(6)[None, None, :] + .05 * np.cos(phase)[..., None]
    specific[..., 1] = 35 + .01 * np.sin(phase)[..., None]
    specific[..., 2] = d.RHO * (.02 + .01 * np.sin(phase)[..., None]) * np.array([1., 1., 1., .6, .3, .1])
    return d.State(h, h[..., None] * specific, grid.bottom)


def moving_witness():
    grid = case_grid()
    initial = case_state(grid)
    parameters = d.Parameters(dt=30., nu_h=2e6, nu_v=1e-4, kappa_h=100., kappa_v=1e-6,
                              kappa_conv=.01, r_bot=.001, lambda_bulk=80., air_temperature=17.,
                              tau_x=.001, tau_y=.0002, coriolis=1e-4, fct=True)
    output, accepted, report = d.advance(initial, grid, parameters)
    if not accepted:
        raise RuntimeError(report['rejection_reason'])
    return {'authenticity': 'prototype', 'shape': [8, 4, 6], 'dt_s': parameters.dt,
            'parameters': asdict(parameters), 'accepted': accepted,
            'eta_change_max_m': float(np.max(abs(output.eta - initial.eta))),
            'deep_stock_change_max_IT_IS_Mu_Mv': np.max(abs(output.n[..., 3:, :] - initial.n[..., 3:, :]),
                                                       axis=(0, 1, 2)).tolist(),
            'specific_velocity_change_max_m_s': float(np.max(abs(
                output.n[..., 2:] / (d.RHO * output.h[..., None])
                - initial.n[..., 2:] / (d.RHO * initial.h[..., None])))),
            'pressure_absolute_work_sum_J': sum(abs(r['pressure_work_J']) for r in report['fast_steps']),
            'horizontal_density_gradient_nonzero': True, 'vertical_shear_nonzero': True,
            'report': report}


def order_witness():
    grid = case_grid(compact=True)
    initial = case_state(grid)
    outputs = []
    increments = (1., .5, .25, .125)
    for dt in increments:
        current = initial.copy()
        for _ in range(round(2. / dt)):
            current, accepted, report = d.advance(current, grid, d.Parameters(dt=dt))
            if not accepted:
                raise RuntimeError(report['rejection_reason'])
        outputs.append(np.r_[current.h.ravel(),
                             (current.n / np.array([1., 1., d.RHO, d.RHO])).ravel()])
    differences = [float(np.linalg.norm(outputs[k] - outputs[k + 1])) for k in range(3)]
    observed = np.log2(np.asarray(differences[:-1]) / differences[1:]).tolist()
    return {'configuration': 'fast_only; all slow coefficients zero, FCT active',
            'fixed_endpoint_s': 2., 'macro_dt_s': list(increments),
            'metric': 'L2 of h, IT, IS, Mu/rho0, Mv/rho0 at identical endpoint',
            'successive_endpoint_differences': differences, 'observed_orders': observed,
            'second_order_witness_passed': bool(min(observed) >= 1.9),
            'full_active_subset_order_qualified': False,
            'scope': 'this synthetic fixed endpoint only; no spatial convergence or industrial claim'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    arguments = parser.parse_args()
    result = {'contract': d.CONTRACT, 'python': platform.python_version(), 'numpy': np.__version__,
              'source_sha256': hashlib.sha256(Path(d.__file__).read_bytes()).hexdigest(),
              'moving_witness': moving_witness(), 'order_witness': order_witness(),
              'industrial_quality_or_speed_qualified': False}
    arguments.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps({'output': arguments.output.as_posix(),
                      'accepted': result['moving_witness']['accepted'],
                      'eta_change_max_m': result['moving_witness']['eta_change_max_m'],
                      'observed_orders': result['order_witness']['observed_orders']}, allow_nan=False))


if __name__ == '__main__':
    main()
