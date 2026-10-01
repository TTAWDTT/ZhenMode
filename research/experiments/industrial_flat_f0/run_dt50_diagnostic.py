"""Register, run once, and read two dt50 time-refinement diagnostics.

No original input, baseline contract, solver source or threshold is overwritten.
Use `prepare`, commit the registration and code, then use `run` once.
"""
# ruff: noqa: E402 -- thread selection must precede NumPy imports.
import argparse
import csv
import hashlib
import json
import os
import re
import resource
import subprocess
import sys
import time
from pathlib import Path

os.environ.update(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')

import netCDF4
import numpy as np
from diagnostic_contract import time_refinement
from diagnostic_scoring import isolated_time_scorer
from export_score import mom_arrays, ocean_arrays
from prepare_mom import prepare
from run_guard import guarded_run, output_bytes, user_processes

REPO = Path(__file__).resolve().parents[3]
SCORER = REPO / 'research/experiments/standing_wave_v0/score.py'
SCORER_SHA256 = '29e97ecb6f18d79cc67cffe09b3ec51563e76819027fbb6a73474f7a40bc291b'
SOURCE_BASE = '8ad3552ff651cf45dfec7ae9a119ac75eef28890'
EXE_SHA256 = '35743c5cc1068b5d19f7ae36d358a8e83d3da43c12429211a61e545c9f4caac9'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def check_analysis_limits(root, started):
    require(time.monotonic()-started < 1200., 'total diagnostic deadline exceeded')
    require(output_bytes(root) < 128*1024**2, 'total diagnostic output budget exceeded')
    rss = sum(record['rss'] for record in user_processes(os.getuid()).values())
    require(rss <= 4*1024**3, 'analysis aggregate RSS budget exceeded')
    return rss


def write_new(path, value):
    with path.open('x', encoding='utf-8') as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + '\n')


def register(root, reference, executable):
    scorer = isolated_time_scorer(SCORER_SHA256)
    baseline = json.loads((reference / 'coarse-contract.json').read_text())
    require(baseline == scorer.contract('coarse'), 'baseline contract differs')
    candidate = time_refinement(baseline)
    root.mkdir(exist_ok=False)
    write_new(root / 'contract.json', candidate)
    prepare(root / 'mom', False, 50.)
    (root / 'ocean').mkdir()
    identity = sha(reference / 'frozen-inputs/ocean_native_inputs.npz')
    require(identity == '43ffb2c82f30638ac0b79b9199459c6fc1bb0922a76fee84b5fad7df8a0160f7', 'ocean input changed')
    require(sha(root / 'mom/INPUT/initial.nc') == 'f5fce7eb2f7ec76711942d3e29e0596093fe0608021885b480abd06a769b1318', 'MOM input changed')
    require(sha(executable) == EXE_SHA256, 'MOM executable changed')
    original = json.loads((reference / 'mom-coarse/requested_config.json').read_text())['settings']
    requested = json.loads((root / 'mom/requested_config.json').read_text())['settings']
    differences = {k: [original[k], requested[k]] for k in original if original[k] != requested[k]}
    require(set(differences) == {'DT', 'DT_THERM', 'DT_FORCING', 'DTBT'}, 'unregistered requested parameters')
    files = [SCORER, *sorted((REPO / 'src').glob('*.py')),
             *sorted(Path(__file__).parent.glob('*.py'))]
    protocol = dict(schema='ocean.dt50_time_refinement_diagnostic.v1',
        diagnostic_only=True, scientific_purpose='Temporal error and dissipation sensitivity; not paired speed qualification',
        grid=[64, 8, 4], dt_s=50., steps=640, duration_s=32000., output_s=1000.,
        complete_outputs=33, attempts_per_model=1,
        baseline_contract_sha256=sha(reference / 'coarse-contract.json'),
        baseline_score_sha256=sha(reference / 'mom-coarse-score.json'),
        baseline_output_sha256=sha(reference / 'mom-coarse.npz'),
        contract_byte_sha256=sha(root / 'contract.json'), contract_canonical_sha256=scorer.digest(candidate),
        input_sha256=identity, executable_sha256=EXE_SHA256,
        requested_mom_diff=differences,
        expected_resolved_mom_changes=['DT', 'DT_THERM', 'DT_FORCING', 'DTBT', 'DT_TRACER_ADVECT'],
        retained_mom_parameters=dict(BE=.6, BEGW=0., BEBT=.1, DT_BT_FILTER=-.25),
        fast_mode_prediction=dict(dtbt_s=50., dt_filt_s=6.25, nstep=1, nfilter=1,
            iterations_per_btstep=2, velocity_eta_weights=[8/9, 1/9], transport_acceleration_weights=[.9, .1]),
        scoring='Separate namespace overrides contract admission only; original scorer bytes and checks/formulas/thresholds retained',
        budget=dict(cpu=1, ranks=1, rss_bytes=4*1024**3, wall_per_model_s=600,
            wall_total_s=1200, new_output_total_bytes=128*1024**2),
        limitations=['MOM full-state nonfinite/positive-thickness check at saved outputs; native MAXTRUNC=0 retained',
            'Ocean engineering TS gate is relative each step; final scorer TS metric is absolute',
            'No fine grid, repeat, parameter repair, threshold relaxation, speedup or industrial qualification'],
        source_base=SOURCE_BASE, source_hashes={str(p.relative_to(REPO)): sha(p) for p in files})
    write_new(root / 'protocol.json', protocol)
    print(json.dumps(dict(protocol_sha256=sha(root / 'protocol.json'),
        contract_sha256=sha(root / 'contract.json'), root=str(root))))


def verify_registration(root, reference, executable):
    require(sha(root / 'protocol.json') == sha(Path(__file__).with_name('dt50_protocol.json')),
            'protocol differs from committed registration')
    p = json.loads((root / 'protocol.json').read_text())
    require(sha(root / 'contract.json') == p['contract_byte_sha256'], 'registered contract changed')
    require(sha(executable) == p['executable_sha256'], 'registered executable changed')
    for name, identity in p['source_hashes'].items():
        if sha(REPO / name) != identity:
            raise ValueError('registered source changed: ' + name)
    for name, identity in [('coarse-contract.json', p['baseline_contract_sha256']),
                           ('mom-coarse-score.json', p['baseline_score_sha256']),
                           ('mom-coarse.npz', p['baseline_output_sha256'])]:
        require(sha(reference / name) == identity, 'baseline artifact changed: ' + name)
    require(sha(reference / 'frozen-inputs/ocean_native_inputs.npz') == p['input_sha256'], 'registered input changed')
    return p


def summarize(root, name, arrays, guard, protocol, scorer, started):
    check_analysis_limits(root, started)
    c = json.loads((root / 'contract.json').read_text())
    scorer.validate(c, arrays)
    if np.any(arrays['h'] <= 0.):
        raise ValueError('nonpositive saved thickness')
    report = scorer.score(c, arrays)
    rho, g = c['rho0'], c['gravity']
    kinetic = .5*rho*sum(np.sum(arrays['volume_'+q]*arrays[q]**2, axis=(1, 2)) for q in ('u', 'v'))
    pe = .5*rho*g*np.sum(arrays['area']*arrays['eta']**2, axis=1)
    energy = kinetic + pe
    check_analysis_limits(root, started)
    t = arrays['time']
    amplitude = c['amplitude_m']
    speed = amplitude*np.sqrt(g/c['H_m'])
    metadata = arrays['metadata']
    eta_ref, _ = scorer.exact(c, t, arrays['x_eta'], metadata['sampling_eta'], c['Lx_m']/64)
    _, u_ref = scorer.exact(c, t, arrays['x_u'], metadata['sampling_u'], arrays['width_u'])
    ce, su = eta_ref[0]/amplitude, u_ref[8]/speed
    wu = arrays['volume_u'][0].sum(axis=1)
    ubar = np.sum(arrays['u']*arrays['volume_u'], axis=2)/arrays['volume_u'].sum(axis=2)
    ae = np.sum(arrays['eta']*ce*arrays['area'], axis=1)/(amplitude*np.sum(ce**2*arrays['area']))
    au = np.sum(ubar*su*wu, axis=1)/(speed*np.sum(su**2*wu))
    z = ae + 1j*au
    csv_path = root / (name+'-energy-modal.csv')
    with csv_path.open('x', newline='') as stream:
        writer = csv.writer(stream, lineterminator='\n')
        writer.writerow(['time_s', 'K_J', 'PE_J', 'E_J', 'signed_relative_change',
                         'eta_mode_a', 'u_mode_a', 'modal_amplitude_ratio', 'phase_error_rad'])
        writer.writerows(zip(t, kinetic, pe, energy, energy/energy[0]-1, ae, au, abs(z),
                             np.angle(z*np.exp(-2j*np.pi*t/c['period_s']))))
    check_analysis_limits(root, started)
    report.update(diagnostic_only=True, industrial_qualified=False, equal_error_speedup_qualified=False,
        protocol_sha256=sha(root / 'protocol.json'), energy_final_signed=float(energy[-1]/energy[0]-1),
        energy_worst_time_s=float(t[np.argmax(abs(energy/energy[0]-1))]),
        minimum_saved_thickness_m=float(np.min(arrays['h'])),
        nonfinite_saved_values=sum(int(np.count_nonzero(~np.isfinite(arrays[k]))) for k in ('eta','u','v','h','T','S')),
        complete_states=33, steps=640, dt_s=50.,
        guard={k: guard[k] for k in ('exit_code', 'stopped_reason', 'wall_s', 'sampled_aggregate_peak_rss_bytes',
                                    'sampled_peak_output_bytes', 'descendant_cleanup_confirmed')},
        energy_csv_sha256=sha(csv_path), guard_sha256=sha(root / name / 'guard_result.json'))
    if name == 'mom':
        directory = root / name
        current = json.loads((directory / 'native_inspection.json').read_text())['resolved_options']
        original = json.loads((root / 'baseline_mom_options.json').read_text())
        diff = {k: [original.get(k), current.get(k)] for k in set(current) | set(original)
                if original.get(k) != current.get(k)}
        if set(diff) != set(protocol['expected_resolved_mom_changes']):
            raise ValueError('unexpected resolved MOM diff: ' + str(diff))
        require(all(current[k] == value for k, value in protocol['retained_mom_parameters'].items()), 'MOM numerical defaults changed')
        report['resolved_mom_diff'] = diff
        report['effective_fast_mode'] = protocol['fast_mode_prediction']
        with netCDF4.Dataset(directory / 'ocean.stats.nc') as ds:
            raw = np.ma.asarray(ds['En'][:])
            if raw.shape != (33,) or np.any(np.ma.getmaskarray(raw)) or not np.all(np.isfinite(raw)):
                raise ValueError('invalid native energy statistics')
            en = np.asarray(raw)
            report['native_energy_max_abs'] = float(np.max(abs(en/en[0]-1)))
            report['native_energy_final_signed'] = float(en[-1]/en[0]-1)
        log = (directory / 'guard_stdout.log').read_text()
        native_io = {}
        for label in ('Ocean MOM save_restart', 'Ocean diagnostics framework'):
            matches = re.findall(r'^\(' + re.escape(label) + r'\)\s+(\d+)\s+(\d+\.\d+)', log, re.M)
            if len(matches) != 1:
                raise ValueError('missing native I/O clock: ' + label)
            native_io[label] = dict(calls=int(matches[0][0]), seconds=float(matches[0][1]))
        report['clocks'] = dict(initialization_s=metadata['initialization_s'],
            main_loop_including_io_s=metadata['integration_s'],
            cold_lower_compile_s=None, first_compiled_execution_s=None,
            warm_remaining_execution_s=None, native_io_clocks=native_io)
        report['timing_limitations'] = 'No JIT; native Initialization and Main loop clocks; Main loop includes diagnostics/checkpoint I/O. First/warm split unavailable.'
    else:
        receipt = json.loads((root / name / 'trajectory_receipt.json').read_text())
        report['clocks'] = {k: receipt[k] for k in ('initialization_s', 'cold_lower_compile_s',
            'first_compiled_execution_s', 'warm_remaining_execution_s', 'integration_s',
            'output_io_s', 'engineering_gate_s', 'child_total_wall_s')}
        report['timing_limitations'] = 'Initialization includes cold compile/t0 I/O; integration equals first+remaining execution. Categories not additive.'
        report['engineering_gate_steps'] = receipt['completed_steps']
    check_analysis_limits(root, started)
    write_new(root / (name+'-summary.json'), report)
    check_analysis_limits(root, started)
    print(json.dumps(report, indent=2))


def run_once(root, reference, executable):
    protocol = verify_registration(root, reference, executable)
    if subprocess.check_output(['git', 'status', '--porcelain'], cwd=REPO).strip():
        raise ValueError('commit code and registration before integration')
    if (root / 'run_started.json').exists():
        raise ValueError('refusing a second attempt')
    os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
    resource.setrlimit(resource.RLIMIT_AS, (4*1024**3, 4*1024**3))
    write_new(root / 'run_started.json', dict(protocol_sha256=sha(root / 'protocol.json'), attempts=1))
    baseline = json.loads((reference / 'mom-coarse/native_inspection.json').read_text())['resolved_options']
    write_new(root / 'baseline_mom_options.json', baseline)
    scorer = isolated_time_scorer(SCORER_SHA256)
    c = json.loads((root / 'contract.json').read_text())
    started = time.monotonic()
    check_analysis_limits(root, started)
    commands = [('mom', [str(executable)]), ('ocean', [sys.executable, str(Path(__file__).with_name('run_ocean.py')),
        '--run-directory', str(root / 'ocean'), '--input-file', str(reference / 'frozen-inputs/ocean_native_inputs.npz'),
        '--contract', str(root / 'contract.json')])]
    for name, command in commands:
        verify_registration(root, reference, executable)
        guard = guarded_run(command, root / name, wall_s=min(600., 1200.-(time.monotonic()-started)),
            rss_bytes=4*1024**3, output_limit=128*1024**2, output_budget_directory=root)
        if (guard['exit_code'] != 0 or guard['stopped_reason'] is not None
                or guard['descendant_cleanup_confirmed'] is not True):
            write_new(root / 'failure.json', dict(model=name, stage='native_run',
                guard_sha256=sha(root / name / 'guard_result.json'), diagnostic_only=True))
            raise RuntimeError('native diagnostic stopped; no repeat')
        check_analysis_limits(root, started)
        arrays = (mom_arrays(root / name, c, guard, executable) if name == 'mom' else
                  ocean_arrays(root / name, c, guard, SOURCE_BASE))
        check_analysis_limits(root, started)
        summarize(root, name, arrays, guard, protocol, scorer, started)
    verify_registration(root, reference, executable)
    check_analysis_limits(root, started)
    completion = dict(diagnostic_only=True, attempts_per_model=1,
        root_output_bytes=output_bytes(root), elapsed_including_analysis_s=time.monotonic()-started,
        protocol_sha256=sha(root / 'protocol.json'), status='budget_checked')
    pending = root / 'completion_pending.json'
    write_new(pending, completion)
    check_analysis_limits(root, started)
    pending.rename(root / 'completed.json')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare', 'run'])
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--reference-root', type=Path, required=True)
    parser.add_argument('--executable', type=Path, required=True)
    args = parser.parse_args()
    if args.action == 'prepare':
        register(args.root, args.reference_root, args.executable)
    else:
        run_once(args.root, args.reference_root, args.executable)


if __name__ == '__main__':
    main()
