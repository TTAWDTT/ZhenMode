"""Reproduce scalar receipts for real, restricted manufactured raw commits."""
import argparse
import hashlib
import importlib.metadata
import json
import math
import platform
import subprocess
from pathlib import Path

import numpy as np

from . import fixed_eta_raw_oracle as oracle
from .affine_physical_cases import manufactured_case
from .affine_physical_evidence import Gate
from .fixed_eta_raw import FixedEtaRawPressureSlice, variable_mass_identity

ROOT = Path(__file__).resolve().parents[3]
PROTOCOL = 'docs/fixed_eta_raw_protocol.json'


def compare(gate, value, reference, scale):
    value, reference, scale = map(np.asarray, (value, reference, scale))
    if value.shape != reference.shape:
        raise ValueError('independent receipt shape mismatch')
    scale = np.broadcast_to(scale, value.shape)
    if not all(np.isfinite(v).all() for v in (value, reference, scale)) or np.any(scale < 0.):
        raise ValueError('finite values and nonnegative primitive operation scale required')
    for index in np.ndindex(value.shape):
        gate.check(value[index], reference[index], 512. * np.finfo(float).eps * max(1., scale[index]))


def case_evidence(durations, **parameters):
    profile, spec = manufactured_case(density_gradient_x=0., strain=0., **parameters)
    stepper = FixedEtaRawPressureSlice(profile, distance_m=spec.distance, length_m=spec.length)
    gates = {key: Gate() for key in ('actual_raw_stock_increment', 'raw_force', 'row_stock_flux', 'face_stock_flux', 'face_pressure', 'face_energy_flux', 'row_energy_flux', 'actual_pressure_work')}
    rows = []
    area = spec.distance * spec.length / 2.
    for dt in durations:
        before = stepper.profile
        receipt = stepper.advance(dt)
        independent = oracle.audit(before, receipt, spec)
        compare(gates['actual_raw_stock_increment'], area * (receipt.state.stocks - before.state.stocks), independent['stock_increment'],
                area * (abs(receipt.state.stocks) + abs(before.state.stocks)) + dt * independent['force_scale'][:, :, None])
        compare(gates['raw_force'], receipt.raw_force, independent['force'], receipt.raw_force_scale + independent['force_scale'])
        compare(gates['row_stock_flux'], receipt.net_water, independent['net_water'], receipt.ledger_scale[:, :, 0] + independent['transport_scale'][:, :, 0])
        compare(gates['row_stock_flux'], receipt.net_stocks, independent['net_stocks'], receipt.ledger_scale[:, :, 1:] + independent['transport_scale'][:, :, 1:])
        compare(gates['row_energy_flux'], receipt.net_energy, independent['net_energy'], receipt.energy_flux_scale + independent['energy_flux_scale'])
        for face, reference in zip(receipt.faces, independent['faces'], strict=True):
            compare(gates['face_stock_flux'], face['transport'], reference['transport'], face['transport_scale'] + reference['transport_scale'])
            for side in range(2):
                compare(gates['face_stock_flux'], face['outer_transport'][side], reference['transport'], face['outer_transport_scale'][side] + reference['transport_scale'])
                compare(gates['face_energy_flux'], face['outer_energy'][side], np.array([reference['KE_transport'], reference['PE_transport']]),
                        face['outer_energy_scale'][side] + np.array([reference['KE_scale'], reference['PE_scale']]))
            compare(gates['face_pressure'], face['pressure'], reference['pressure'], face['pressure_scale'] + abs(reference['pressure']))
            for key, scale in (('KE_transport', 'KE_scale'), ('PE_transport', 'PE_scale')):
                compare(gates['face_energy_flux'], face[key], reference[key], face[scale] + reference[scale])
        for value in (receipt.raw_KE_change_J, receipt.physical_KE_change_J, receipt.pressure_work_J, receipt.boundary_pressure_work_J):
            compare(gates['actual_pressure_work'], value, independent['boundary_pressure_work'], receipt.energy_scale + independent['energy_scale'])
        unchanged = (np.array_equal(receipt.state.h, before.state.h) and np.array_equal(receipt.state.interfaces, before.state.interfaces)
                     and np.array_equal(receipt.state.stocks[:, :, :2], before.state.stocks[:, :, :2])
                     and np.array_equal(receipt.state.stocks[:, :, 3], before.state.stocks[:, :, 3]))
        if not unchanged:
            raise AssertionError('frozen geometry/thermodynamic/V stock byte preservation failed')
        impulse = area * math.fsum(float(v) for v in (receipt.state.stocks[:, :, 2] - before.state.stocks[:, :, 2]).ravel())
        rows.append(dict(duration_s=dt, actual_time_s=receipt.time_s, accepted_raw_steps=receipt.accepted_raw_steps,
                         shared_segments=len(receipt.faces), raw_Mu_impulse_change_kg_m_s=impulse,
                         maximum_actual_raw_Mu_change=float(np.max(abs(receipt.state.stocks[:, :, 2] - before.state.stocks[:, :, 2]))),
                         maximum_actual_deep_Mu_change=float(np.max(abs(receipt.state.stocks[:, 3:, 2] - before.state.stocks[:, 3:, 2]))),
                         maximum_shared_water_transport_m3=max(abs(face['transport'][0]) for face in receipt.faces),
                         maximum_shared_IS_transport=max(abs(face['transport'][2]) for face in receipt.faces),
                         maximum_shared_Mu_transport_kg_m_s=max(abs(face['transport'][3]) for face in receipt.faces),
                         maximum_shared_KE_transport_J=max(abs(face['KE_transport']) for face in receipt.faces),
                         maximum_shared_PE_transport_J=max(abs(face['PE_transport']) for face in receipt.faces),
                         raw_KE_change_J=receipt.raw_KE_change_J, physical_KE_change_J=receipt.physical_KE_change_J,
                         PE_change_J=receipt.PE_change_J, boundary_pressure_work_J=receipt.boundary_pressure_work_J,
                         actual_midpoint_pressure_work_J=receipt.pressure_work_J,
                         inverse_maximum_actual_raw_momentum_residual_kg_m_s=receipt.maximum_inverse_residual,
                         fixed_geometry_TS_Mv_byte_preserved=unchanged))
    return dict(accepted_raw_steps=stepper.accepted_raw_steps, steps=rows, gates={key: gate.receipt() for key, gate in gates.items()})


def build_evidence():
    protocol = json.loads((ROOT / PROTOCOL).read_text(encoding='utf-8'))
    if protocol['frozen_thresholds']['roundoff_eps_multiplier'] != 512 or protocol['qualification_passed']:
        raise ValueError('frozen threshold or qualification boundary changed')
    cases = dict(positive=case_evidence((.01, .02)), negative=case_evidence((.01, .02), u0=-.03, external_pressure=(92., 80.)),
                 zero_pressure=case_evidence((.01,), external_pressure=(80., 80.)),
                 sign_crossing=case_evidence((.02,), u0=(6. / 1025.) * .02 / 2.))
    snapshots = oracle.moving_snapshots()
    moving = {}
    for key in ('raw', 'physical'):
        row = snapshots[key]
        chain = variable_mass_identity(row['mass_before'], row['mass_after'], row['u_before'], row['u_after'])
        gate = Gate()
        compare(gate, chain['kinetic_change'], row['KE_after'] - row['KE_before'], chain['scale'])
        if abs(chain['mass_work']) <= 1000. * 512. * np.finfo(float).eps * max(1., chain['scale']):
            raise AssertionError('moving fixed-mass omission control unresolved')
        moving[key] = dict(chain, independent_actual_snapshot_KE_gate=gate.receipt())
    for key in ('omitted_Wdot_max_residual', 'omitted_Rdot_max_residual', 'moving_local_storage_change', 'moving_KE_gap_change'):
        if abs(snapshots[key]) <= 1e-5:
            raise AssertionError('declared moving obstruction unresolved')
        moving[key] = snapshots[key]
    return dict(contract=protocol['contract'], restricted_fixed_eta_raw_step_passed=True, qualification_passed=False,
                accepted_raw_steps=sum(row['accepted_raw_steps'] for row in cases.values()), accepted_moving_steps=0,
                real_archive_steps=0, original_global_CV_identified=False, production_force_consumption_qualified=False,
                cases=cases, obstructions=oracle.obstructions(), moving_snapshot_algebra=moving,
                numerical_scope='New manufactured fixed-eta uniform-velocity half-prisms only; actual raw commits. No original CV, moving adapter, Q/Psi, predict/fast/replay, time-order or speed qualification.')


def provenance():
    from ocean_solver.provenance.archives import current_source_files
    if subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT, text=True).strip():
        raise ValueError('committed clean checkout required before witness execution')
    selected = list((ROOT / 'research/experiments/material_top_band').glob('*.py'))
    selected += [ROOT / 'tests/research/contracts/test_fixed_eta_raw.py', ROOT / PROTOCOL,
                 ROOT / 'scripts/run_bounded_research_tests.py', ROOT / 'research/experiments/material_top_band/affine_requirements.lock']
    files = current_source_files(ROOT, selected)
    return dict(source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                source_sha256={label: hashlib.sha256(path.read_bytes()).hexdigest() for label, path in sorted(files.items())},
                Python=platform.python_version(), package_versions={name: importlib.metadata.version(name) for name in ('numpy', 'pytest', 'ruff')})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    source = provenance()
    result = dict(build_evidence(), provenance=source)
    text = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + '\n'
    if args.output:
        args.output.write_text(text, encoding='utf-8')
    else:
        print(text, end='')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
