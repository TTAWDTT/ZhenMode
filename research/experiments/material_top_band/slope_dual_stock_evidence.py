"""Reproduce scalar physical projection gates without returning an accepted state."""
import argparse
import hashlib
import importlib.metadata
import json
import math
import platform
import subprocess
from dataclasses import replace
from pathlib import Path

import numpy as np

from . import affine_physical_oracle as physical
from . import slope_dual_stock_oracle as oracle
from .affine_physical import bound
from .affine_physical_cases import manufactured_case
from .affine_physical_evidence import Gate
from .slope_dual_stock import FrozenPhysicalProjection

ROOT = Path(__file__).resolve().parents[3]
PROTOCOL = 'docs/slope_dual_stock_protocol.json'


def fixture(*, eta=(-.2, .3), U=.03, strain=.008, shear=None, **parameters):
    profile, spec = manufactured_case(eta=eta, u0=U, strain=strain, **parameters)
    if shear is not None:
        state = profile.state
        if shear == 'zero_depth_flux':
            velocity = np.zeros((2, 14, 2))
            velocity[0, 0, 0] = 1.
            velocity[0, -1, 0] = -state.h[0, 0] / state.h[0, -1]
        elif shear == 'vertical':
            n = np.arange(28).reshape(2, 14)
            velocity = np.stack([.03 + .002 * n + .01 * np.cos(n), -.02 + .001 * np.sin(n)], axis=-1)
        else:
            raise ValueError('unknown declared shear fixture')
        stocks = state.stocks.copy()
        stocks[:, :, 2:] = 1025. * state.h[:, :, None] * velocity
        profile = replace(profile, state=replace(state, stocks=stocks))
    return FrozenPhysicalProjection(profile, distance_m=spec.distance, length_m=spec.length), spec


def arrays(gate, candidate, reference, scale):
    if candidate.shape != reference.shape or scale.shape != candidate.shape:
        raise ValueError('independent gate shape mismatch')
    for index in np.ndindex(candidate.shape):
        gate.check(candidate[index], reference[index], bound(scale[index]))


def case_evidence(epsilon_values, *, material=True, **parameters):
    projection, spec = fixture(**parameters)
    state = projection.profile.state
    mapped, ref = projection.evaluate(), oracle.mapped(spec, projection.chart, state)
    gates = {name: Gate() for name in ('physical_stock', 'physical_mass', 'physical_PE', 'mass_stock_binding', 'raw_inventory', 'physical_KE', 'fixed_W_work', 'pressure_force')}
    arrays(gates['physical_stock'], mapped.values, ref['values'], mapped.scale + ref['scale'])
    arrays(gates['physical_mass'], mapped.mass, ref['mass'], abs(mapped.mass) + abs(ref['mass']))
    gates['physical_PE'].check(mapped.PE, ref['PE'], bound(mapped.PE_scale + ref['PE_scale']))
    raw = oracle.raw_inventory(state)
    arrays(gates['raw_inventory'], mapped.raw_inventory, raw, abs(mapped.raw_inventory) + abs(raw))
    for j in range(2):
        momentum = mapped.values[:, :, 3 + j].ravel()
        arrays(gates['mass_stock_binding'], mapped.mass @ mapped.velocity[:, j], momentum,
               abs(mapped.mass) @ abs(mapped.velocity[:, j]) + abs(momentum))
    for side in range(2):
        for layer in range(14):
            force = physical.basis_force(spec, projection.chart, side, layer)
            gates['pressure_force'].check(projection.base.force_N[side, layer], force['value'],
                                          bound(projection.base.force_scale_N[side, layer] + force['scale']))
    K = oracle.kinetic(spec, projection.chart, mapped.velocity)
    matrix_K = .5 * math.fsum(float(mapped.velocity[:, j] @ mapped.mass @ mapped.velocity[:, j]) for j in range(2))
    gates['physical_KE'].check(matrix_K, K['value'], bound(K['scale'] + abs(matrix_K)))
    initial_power = physical.pressure_power(spec, projection.chart, mapped.velocity[:, 0].reshape(2, 14))
    initial_candidate_power = math.fsum(float(value) for value in (projection.base.force_N.ravel() * mapped.velocity[:, 0]))
    initial_power_scale = math.fsum(float(value) for value in (projection.base.force_scale_N.ravel() * abs(mapped.velocity[:, 0]))) + initial_power['scale']
    gates['pressure_force'].check(initial_candidate_power, initial_power['value'], bound(initial_power_scale))
    depth_terms = state.h * mapped.velocity[:, 0].reshape(2, 14)
    depth_flux = [math.fsum(float(value) for value in depth_terms[side]) for side in range(2)]
    depth_bounds = [bound(math.fsum(float(abs(value)) for value in depth_terms[side])) for side in range(2)]
    if parameters.get('shear') == 'zero_depth_flux':
        for flux, tolerance in zip(depth_flux, depth_bounds, strict=True):
            if abs(flux) > tolerance:
                raise AssertionError('initial endpoint zero-depth-flux construction failed')
        if abs(initial_power['value']) <= 1000. * bound(initial_power_scale):
            raise AssertionError('initial zero-depth-flux pressure power is not resolved')
    kick = projection.fixed_mass_impulse(.01)
    after = mapped.velocity.copy()
    after[:, 0] = kick['velocity_after']
    Kafter = oracle.kinetic(spec, projection.chart, after)
    midpoint = .5 * (mapped.velocity[:, 0] + after[:, 0])
    work = physical.pressure_power(spec, projection.chart, midpoint.reshape(2, 14))
    gates['fixed_W_work'].check(kick['kinetic_change'], Kafter['value'] - K['value'], bound(kick['scale'] + K['scale'] + Kafter['scale']))
    gates['fixed_W_work'].check(kick['kinetic_change'], .01 * work['value'], bound(kick['scale'] + .01 * work['scale']))
    defect = mapped.raw_inventory - mapped.values[:, :, :5].sum(axis=(0, 1))
    output = dict(strips=len(projection.chart), raw_minus_projected_H_IT_IS_Mu_Mv=defect.tolist(),
                  raw_PE_J=mapped.raw_PE, physical_PE_J=mapped.PE,
                  raw_minus_physical_PE_J=mapped.raw_PE - mapped.PE,
                  raw_minus_physical_free_PE_J=mapped.raw_free_PE - mapped.free_PE,
                  raw_minus_physical_anomaly_PE_J=(mapped.raw_PE - mapped.raw_free_PE) - (mapped.PE - mapped.free_PE),
                  physical_KE_J=K['value'], minimum_W_eigenvalue=kick['minimum_mass_eigenvalue'],
                  initial_endpoint_depth_flux_m2_s=depth_flux, initial_endpoint_depth_flux_roundoff_bounds_m2_s=depth_bounds,
                  initial_independent_pressure_power_W=initial_power['value'], initial_pressure_power_roundoff_bound_W=bound(initial_power_scale),
                  fixed_W_kinetic_change_J=kick['kinetic_change'], fixed_W_pressure_work_J=.01 * work['value'],
                  fixed_W_solve_max_residual=kick['solve_residual'], raw_conservative_remap=False,
                  maximum_abs_pressure_force_N=float(np.max(abs(projection.base.force_N))),
                  auxiliary_impulse_duration_s=.01, accepted_state_returned=False)
    if material:
        U, strain = parameters.get('U', .03), parameters.get('strain', .008)
        direction = projection.direction(U, strain)
        independent = oracle.direction(spec, projection.chart, state, U, strain)
        gates.update({name: Gate() for name in ('physical_Bdot', 'weighted_local_balance', 'physical_PE_dot', 'affine_curve_stock_FD', 'affine_curve_PE_FD', 'raw_defect', 'raw_defect_dot')})
        arrays(gates['physical_Bdot'], direction.values, independent['values'], direction.scale + independent['scale'])
        energy = physical.energy_direction(spec, U, strain)
        gates['physical_PE_dot'].check(direction.PE_dot, energy['PE_dot'], bound(direction.PE_scale + energy['scale']))
        omission_ratios = dict(shape=0., basis=0., gravity=0., virtual_cut=0.)
        weighted = oracle.weighted_balances(spec, projection.chart, state, U, strain)
        for row, independent in zip(direction.rows, weighted, strict=True):
            arrays(gates['weighted_local_balance'], row['value'] + independent['flux'], independent['source'], row['scale'] + independent['scale'])
            tolerance = bound(float(np.sum(row['scale'] + independent['scale'])))
            omission_ratios['shape'] = max(omission_ratios['shape'], float(np.max(abs(row['volume'] - independent['content_dot']))) / tolerance)
            omission_ratios['basis'] = max(omission_ratios['basis'], abs(row['value'][0] + independent['flux'][0]) / tolerance)
            omission_ratios['gravity'] = max(omission_ratios['gravity'], abs(row['value'][5] + independent['flux'][5] - independent['basis_source'][5]) / tolerance)
            if not row['strip'].actual_top:
                deleted_cut_residual = row['value'] + independent['flux'] - independent['top_flux'] - independent['source']
                omission_ratios['virtual_cut'] = max(omission_ratios['virtual_cut'], float(np.max(abs(deleted_cut_residual))) / tolerance)
        if not all(value > 1000. for value in omission_ratios.values()):
            raise AssertionError('declared local omission control did not fail')
        curve_rows = []
        for epsilon in epsilon_values:
            changed_state, _ = oracle.curve(spec, state, U, strain, epsilon)
            changed = projection.evaluate(replace(projection.profile, state=changed_state))
            envelope = oracle.forward_envelope(spec, projection.chart, state, U, strain, epsilon)
            for index in np.ndindex(direction.values.shape):
                gates['affine_curve_stock_FD'].check((changed.values[index] - mapped.values[index]) / epsilon, direction.values[index], envelope['values_bound'][index])
            observed_PE = (changed.PE - mapped.PE) / epsilon
            gates['affine_curve_PE_FD'].check(observed_PE, direction.PE_dot, envelope['PE_bound'])
            curve_rows.append(dict(epsilon_s=epsilon, observed_PE_direction_W=observed_PE,
                                   PE_truncation_bound_W=envelope['PE_truncation_bound'], PE_total_bound_W=envelope['PE_bound']))
        expected = oracle.analytic_defects(spec, U, strain)
        rate_defect = direction.raw_inventory_dot - direction.values[:, :, :5].sum(axis=(0, 1))
        arrays(gates['raw_defect'], defect, expected, abs(mapped.raw_inventory) + mapped.scale[:, :, :5].sum(axis=(0, 1)))
        arrays(gates['raw_defect_dot'], rate_defect, -3. * strain * expected, abs(direction.raw_inventory_dot) + direction.scale[:, :, :5].sum(axis=(0, 1)))
        output.update(physical_PE_dot_W=direction.PE_dot, raw_PE_dot_W=direction.raw_PE_dot,
                      raw_minus_physical_PE_dot_W=direction.raw_PE_dot - direction.PE_dot,
                      raw_minus_projected_inventory_dot=rate_defect.tolist(),
                      maximum_abs_deep_projected_IS_dot=float(np.max(abs(direction.values[:, 3:, 2]))),
                      maximum_abs_deep_projected_Mu_dot=float(np.max(abs(direction.values[:, 3:, 3]))),
                      local_omission_residual_bound_ratios=omission_ratios, affine_curve_FD=curve_rows)
    output['gates'] = {name: gate.receipt() for name, gate in gates.items()}
    return output


def build_evidence():
    protocol = json.loads((ROOT / PROTOCOL).read_text(encoding='utf-8'))
    if protocol['frozen_thresholds']['roundoff_eps_multiplier'] != 512 or protocol['qualification_passed']:
        raise ValueError('frozen protocol or qualification boundary changed')
    epsilon = protocol['frozen_thresholds']['epsilon_s']
    cases = dict(slope_positive=case_evidence(epsilon),
                 slope_negative=case_evidence(epsilon, U=-.03, strain=-.008),
                 flat_positive=case_evidence(epsilon, eta=(-.2, -.2)),
                 flat_negative=case_evidence(epsilon, eta=(-.2, -.2), U=-.03, strain=-.008),
                 zero_force=case_evidence(epsilon, material=False, eta=(-.2, -.2), density_gradient_x=0., external_pressure=(80., 80.)),
                 static_shear=case_evidence(epsilon, material=False, shear='vertical'),
                 zero_depth_flux=case_evidence(epsilon, material=False, shear='zero_depth_flux'))
    return dict(contract=protocol['contract'], restricted_physical_projection_probe_passed=True,
                qualification_passed=False, accepted_steps=0, real_archive_steps=0,
                production_force_consumption_qualified=False, cases=cases,
                numerical_scope='manufactured frozen-chart physical B/Bdot and fixed-W scratch; no raw conservative remap, accepted step, time order or speed qualification')


def provenance():
    from ocean_solver.provenance.archives import current_source_files
    if subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT, text=True).strip():
        raise ValueError('committed clean source checkout required before witness execution')
    selected = list((ROOT / 'research/experiments/material_top_band').glob('*.py'))
    selected += [ROOT / 'tests/research/contracts/test_slope_dual_stock.py', ROOT / PROTOCOL,
                 ROOT / 'scripts/run_bounded_research_tests.py',
                 ROOT / 'research/experiments/material_top_band/affine_requirements.lock']
    files = current_source_files(ROOT, selected)
    return dict(source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                source_sha256={label: hashlib.sha256(path.read_bytes()).hexdigest() for label, path in sorted(files.items())},
                Python=platform.python_version(), package_versions={name: importlib.metadata.version(name) for name in ('numpy', 'pytest', 'ruff')})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    source = provenance()
    receipt = dict(build_evidence(), provenance=source)
    text = json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + '\n'
    if args.output:
        args.output.write_text(text, encoding='utf-8')
    else:
        print(text, end='')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
