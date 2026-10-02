"""Reproduce public scalar gates for the restricted instantaneous physical dual."""
import argparse
import hashlib
import importlib.metadata
import json
import math
import platform
import subprocess
from pathlib import Path

import numpy as np

from . import affine_physical_oracle as oracle
from .affine_physical import AffinePhysicalDual, bound, total
from .affine_physical_cases import manufactured_case

ROOT = Path(__file__).resolve().parents[3]
PROTOCOL = 'docs/affine_physical_pressure_protocol.json'


class Gate:
    """Retain maxima with their own a-priori bound; any failed row aborts."""

    def __init__(self):
        self.count, self.max_ratio, self.max_residual, self.bound_at_max_residual = 0, 0., 0., 0.

    def check(self, candidate, reference, tolerance):
        residual = abs(candidate - reference)
        if not all(math.isfinite(float(v)) for v in (candidate, reference, tolerance)) or tolerance <= 0.:
            raise ValueError('finite positive independent gate envelope required')
        if residual > tolerance:
            raise AssertionError('frozen scientific gate failed')
        self.count += 1
        self.max_ratio = max(self.max_ratio, residual / tolerance)
        if residual >= self.max_residual:
            self.max_residual, self.bound_at_max_residual = residual, tolerance

    def receipt(self):
        if self.count == 0:
            raise ValueError('empty gate cannot pass')
        return dict(checks=self.count, maximum_residual=self.max_residual,
                    bound_at_maximum_residual=self.bound_at_max_residual,
                    maximum_residual_bound_ratio=self.max_ratio, passed=True)


def case_evidence(protocol, *, flow=None, **parameters):
    profile, spec = manufactured_case(**parameters)
    op = AffinePhysicalDual(profile, distance_m=spec.distance, length_m=spec.length)
    oracle.validate_strip_ownership(spec, op.state, op.segments)
    gates = {name: Gate() for name in ('contour_N', 'basis_N', 'geometric_variation_N')}
    for row in op.contours():
        ref = oracle.segment_contour(spec, row['segment'])
        gates['contour_N'].check(row['force_N'], ref['value'], bound(row['scale'] + ref['scale']))
    for side in range(2):
        for layer in range(14):
            ref = oracle.basis_force(spec, op.segments, side, layer)
            gates['basis_N'].check(op.force_N[side, layer], ref['value'], bound(op.force_scale_N[side, layer] + ref['scale']))
            for epsilon in protocol['frozen_thresholds']['central_geometry_epsilon_m']:
                varied = oracle.geometric_load_variation(spec, op.segments, side, layer, epsilon)
                gates['geometric_variation_N'].check(op.force_N[side, layer], varied['value'], varied['bound'] + bound(op.force_scale_N[side, layer]))
    result = dict(strips=len(op.segments), velocity_bases=28,
                  maximum_abs_basis_force_N=float(np.max(abs(op.force_N))),
                  physical_PE_J=op.physical_PE(), endpoint_PE_J=op.node_PE(),
                  endpoint_minus_physical_PE_J=op.node_PE() - op.physical_PE())
    if flow is not None:
        U, strain = flow
        rate = op.affine_rates(U, strain)
        energy = oracle.energy_direction(spec, U, strain)
        gates.update({name: Gate() for name in ('physical_PE_dot_W', 'pressure_energy_W', 'layer_stock_dot', 'global_ALE_stock_dot', 'local_ALE_stock_dot', 'stock_PE_forward_W', 'endpoint_map_gap', 'fixed_mass_work_J')})
        gates['physical_PE_dot_W'].check(rate['physical_PE_dot_W'], energy['PE_dot'], bound(rate['PE_scale'] + energy['scale']))
        gates['pressure_energy_W'].check(rate['pressure_power_W'] + energy['PE_dot'], energy['boundary_power'], bound(rate['pressure_power_scale'] + energy['scale']))
        for side in range(2):
            for layer in range(14):
                ref = oracle.layer_stock_direction(spec, op.state, side, layer, U, strain)
                for slot in range(4):
                    value = rate['stockdot'][side, layer, slot]
                    gates['layer_stock_dot'].check(value, ref[slot], bound(abs(value) + abs(ref[slot])))
        flux = oracle.ale_fluxes(spec, op.segments, U, strain)
        for slot, value in enumerate(op.physical_stock_direction(rate)):
            gates['global_ALE_stock_dot'].check(value, -flux['outer'][slot], bound(abs(value) + flux['scale'][slot]))
        for row in flux['local']:
            for slot in range(4):
                gates['local_ALE_stock_dot'].check(row['content_dot'][slot], -row['flux'][slot], bound(row['scale'][slot]))
        stock_differences = []
        for epsilon in protocol['frozen_thresholds']['one_sided_stock_epsilon_s']:
            op.validate_perturbation(rate, epsilon)
            ref = oracle.stock_PE_forward(op.state, rate, epsilon, op.node_area_m2)
            gates['stock_PE_forward_W'].check(rate['node_PE_dot_W'], ref['value'], ref['bound'])
            stock_differences.append(dict(epsilon_s=epsilon, observed_direction_W=ref['value'],
                                          truncation_bound_W=ref['truncation_bound'], total_bound_W=ref['bound']))
        gap = oracle.endpoint_PE_discrepancy(spec, U, strain)
        gates['endpoint_map_gap'].check(result['endpoint_minus_physical_PE_J'], gap['energy'], bound(op.PE_scale() + abs(gap['energy'])))
        gates['endpoint_map_gap'].check(rate['node_PE_dot_W'] - rate['physical_PE_dot_W'], gap['direction'], bound(rate['node_PE_scale'] + rate['PE_scale']))
        impulse = op.fixed_mass_impulse()
        work = oracle.pressure_power(spec, op.segments, impulse['midpoint_u'])
        gates['fixed_mass_work_J'].check(impulse['kinetic_change_J'], work['value'], bound(impulse['scale'] + work['scale']))
        omitted_slope = abs(rate['node_PE_dot_W'] - rate['slope_PE_dot_W'] - energy['PE_dot'])
        result.update(pressure_power_W=rate['pressure_power_W'], physical_PE_dot_W=energy['PE_dot'],
                      endpoint_PE_dot_W=rate['node_PE_dot_W'], open_boundary_power_W=energy['boundary_power'],
                      analytic_endpoint_PE_gap_J=gap['energy'], analytic_endpoint_PE_dot_gap_W=gap['direction'],
                      endpoint_pairing_passed=bool(rate['endpoint_pairing_passed']),
                      slope_PE_dot_W=rate['slope_PE_dot_W'], omitted_slope_residual_W=omitted_slope,
                      omitted_local_cut_residual=flux['omitted_local_residual'], omitted_local_cut_bound=flux['omitted_local_bound'],
                      maximum_abs_internal_cut_flux=float(np.max(abs(flux['internal']))),
                      maximum_abs_deep_IS_dot=float(np.max(abs(rate['stockdot'][:, 3:, 1]))),
                      maximum_abs_deep_Mu_dot=float(np.max(abs(rate['stockdot'][:, 3:, 2]))),
                      maximum_abs_band_R_m_s=float(np.max(abs(rate['relative_downward'][:, 3]))),
                      fixed_mass_kinetic_change_J=impulse['kinetic_change_J'],
                      fixed_mass_metric='auxiliary endpoint nodal lumped', stock_PE_differences=stock_differences)
        if spec.eta[0] == spec.eta[1]:
            result['nodal_minus_interpolated_volume_KE_J'] = oracle.nodal_minus_volume_KE(spec, U, strain)
            gates['endpoint_map_gap'].check(rate['node_PE_dot_W'], energy['PE_dot'], bound(rate['node_PE_scale'] + energy['scale']))
            if not rate['endpoint_pairing_passed']:
                raise AssertionError('restricted flat endpoint pairing must pass')
        elif rate['endpoint_pairing_passed']:
            raise AssertionError('unequal eta endpoint consumption must refuse')
        if flux['omitted_local_residual'] <= 1000. * flux['omitted_local_bound']:
            raise AssertionError('omitted local flux control must fail by frozen bound')
        if spec.eta[0] == spec.eta[1] and omitted_slope <= 1000. * bound(rate['node_PE_scale'] + energy['scale']):
            raise AssertionError('omitted slope PE chain control must fail by frozen bound')
    result['gates'] = {name: gate.receipt() for name, gate in gates.items()}
    return result


def build_evidence():
    protocol = json.loads((ROOT / PROTOCOL).read_text(encoding='utf-8'))
    if protocol['frozen_thresholds']['roundoff_eps_multiplier'] != 512 or protocol['qualification_passed']:
        raise ValueError('frozen protocol or qualification boundary changed')
    cases = dict(
        zero_spurious=case_evidence(protocol, density_gradient_x=0., external_pressure=(80., 80.)),
        flat_positive=case_evidence(protocol, flow=(.03, .008)),
        flat_negative=case_evidence(protocol, flow=(-.03, -.008), u0=-.03, strain=-.008),
        unequal_eta=case_evidence(protocol, flow=(.03, .008), eta=(-.2, .3)),
        nonlinear_TS_static=case_evidence(protocol, density_gradient_x=0., external_pressure=(80., 80.), nonlinear_TS=True),
        constant_density_static=case_evidence(protocol, density_gradient_x=0., density_slope=0., external_pressure=(80., 80.)),
    )
    return dict(contract=protocol['contract'], instantaneous_probe_passed=True,
                qualification_passed=False, production_force_consumption_qualified=False,
                accepted_steps=0, real_archive_steps=0, cases=cases,
                numerical_scope='manufactured instantaneous affine physical dual; no full-step or speed qualification')


def provenance():
    from ocean_solver.provenance.archives import current_source_files
    if subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT, text=True).strip():
        raise ValueError('committed clean source checkout required before witness execution')
    selected = list((ROOT / 'research/experiments/material_top_band').glob('*.py'))
    selected += [ROOT / 'tests/research/contracts/test_affine_physical_pressure.py', ROOT / PROTOCOL,
                 ROOT / 'research/experiments/material_top_band/affine_requirements.lock']
    files = current_source_files(ROOT, selected)
    hashes = {label: hashlib.sha256(path.read_bytes()).hexdigest() for label, path in sorted(files.items())}
    return dict(source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                source_sha256=hashes, Python=platform.python_version(),
                package_versions={name: importlib.metadata.version(name) for name in ('numpy', 'pytest', 'ruff')})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    source = provenance()
    receipt = dict(build_evidence(), provenance=source)
    total([gate['maximum_residual'] for case in receipt['cases'].values() for gate in case['gates'].values()])
    text = json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + '\n'
    if args.output:
        args.output.write_text(text, encoding='utf-8')
    else:
        print(text, end='')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
