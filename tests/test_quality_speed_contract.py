"""Synthetic acceptance-harness tests; no physical or speed qualification."""
import copy
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

TOOLS = Path(__file__).resolve().parents[1] / 'scripts' / 'quality_speed'
sys.path.insert(0, str(TOOLS))
from contract import HASH_FIELDS, PAIR_FIELDS, digest, evaluate
from gate import load
from manifest import attach_files, verify_files
from timing import measure_lifecycle


def fixtures():
    h = 'a' * 64
    policy = dict(schema_version=1, registration='test fixture, NOT physical thresholds',
                  minimum_speedup=1.1, minimum_repeats=3, metrics={
                      'heat_budget': dict(category='conservation', direction='max', limit=1,
                                   unit='W/m2', definition_sha256=h),
                      'solution_error': dict(category='error', direction='max', limit=1,
                                 unit='K', definition_sha256=h),
                      'convergence_order': dict(category='convergence', direction='min', limit=1.9,
                                    unit='1', definition_sha256=h)})
    for name, unit in [('salt_budget', 'kg/s'), ('volume_budget', 'm3/s')]:
        policy['metrics'][name] = dict(category='conservation', direction='max', limit=1,
                                       unit=unit, definition_sha256=h)
    run = dict(schema_version=1, model='test', scenario='test', algorithm='test',
               evidence_class='synthetic_control', source_sha='a' * 40, source_clean=True,
               source_tree_sha256=h, config_sha256=h, precision='float64',
               environment='stdlib test', command='test',
               hardware=dict(device='cpu', cpu='test', accelerator='none', runtime='test',
                             threads=1, ranks=1, memory_limit_bytes=1000),
               duration_s=100, dt_s=10, dt_bt_s=5, requested_steps=10, accepted_steps=10,
               attempted_steps=10, verdict='PASS', coverage_complete=True,
               timing_scope='full_integration_with_production_monitoring',
               synchronization='all_state_and_diagnostics', cold_cache=True,
               peak_memory={'status': 'unavailable', 'reason': 'test fixture'},
               quality={name: dict(value=2 if name == 'convergence_order' else .5,
                                   unit=rule['unit'], definition_sha256=h, evidence_sha256=h)
                        for name, rule in policy['metrics'].items()},
               trials=[dict(pair_id=str(i), setup=1, cold_compile=1, warmup=1,
                            integration=10, diagnostics_io=1, total=15) for i in range(3)])
    run.update({key: h for key in HASH_FIELDS})
    candidate = copy.deepcopy(run)
    for trial in candidate['trials']:
        trial.update(integration=5, total=10)
    policy['pair_contract'] = {key: copy.deepcopy(run[key]) for key in PAIR_FIELDS}
    policy['runs'] = {label: {key: item[key] for key in ('source_sha', 'config_sha256', 'algorithm')}
                      for label, item in (('control', run), ('candidate', candidate))}
    return run, candidate, policy


class GateTests(unittest.TestCase):
    def setUp(self):
        self.a, self.b, self.p = fixtures()

    def result(self):
        return evaluate(self.a, self.b, self.p, digest(self.p))

    def test_synthetic_pass_never_industrial(self):
        result = self.result()
        self.assertEqual(result['status'], 'PASS')
        self.assertFalse(result['industrial_qualified'])
        self.assertEqual(result['speed']['total']['median'], 1.5)

    def test_each_required_run_field_missing(self):
        for key in self.b:
            with self.subTest(key=key):
                bad = copy.deepcopy(self.b)
                del bad[key]
                self.assertEqual(evaluate(self.a, bad, self.p, digest(self.p))['status'], 'INCOMPLETE')

    def test_policy_freeze(self):
        frozen = digest(self.p)
        self.p['metrics']['heat_budget']['limit'] = 100
        self.assertEqual(evaluate(self.a, self.b, self.p, frozen)['status'], 'INCOMPLETE')

    def test_missing_categories(self):
        del self.p['metrics']['convergence_order']
        self.assertEqual(self.result()['status'], 'INCOMPLETE')

    def test_bad_numeric_metrics(self):
        for value in (None, False, '0', float('nan'), float('inf'), -1):
            with self.subTest(value=value):
                self.b['quality']['heat_budget']['value'] = value
                self.assertEqual(self.result()['status'], 'INCOMPLETE')

    def test_quality_failure_hides_speed(self):
        self.b['quality']['heat_budget']['value'] = .6
        result = self.result()
        self.assertEqual(result['status'], 'QUALITY_FAIL')
        self.assertIsNone(result['speed'])

    def test_control_also_must_pass(self):
        self.a['quality']['convergence_order']['value'] = 1.5
        self.assertEqual(self.result()['status'], 'QUALITY_FAIL')

    def test_convergence_failure(self):
        self.b['quality']['convergence_order']['value'] = 1.89
        self.assertEqual(self.result()['status'], 'QUALITY_FAIL')

    def test_hardware_forcing_dt_mismatch(self):
        for key, value in [('hardware', {**self.b['hardware'], 'threads': 2}),
                           ('forcing_sha256', 'b' * 64), ('precision', 'float32')]:
            bad = copy.deepcopy(self.b)
            bad[key] = value
            result = evaluate(self.a, bad, self.p, digest(self.p))
            self.assertEqual(result['status'], 'NOT_COMPARABLE')
            self.assertIsNone(result['speed'])
        self.b.update(dt_s=5, requested_steps=20, accepted_steps=20, attempted_steps=20)
        self.assertEqual(self.result()['status'], 'NOT_COMPARABLE')

    def test_truncated_duration(self):
        self.b['accepted_steps'] = 9
        self.assertEqual(self.result()['status'], 'INCOMPLETE')

    def test_missing_zero_or_invalid_timing(self):
        for key in ('setup', 'cold_compile', 'warmup', 'integration', 'diagnostics_io', 'total'):
            bad = copy.deepcopy(self.b)
            del bad['trials'][0][key]
            self.assertEqual(evaluate(self.a, bad, self.p, digest(self.p))['status'], 'INCOMPLETE')
        self.b['trials'][0]['integration'] = 0
        self.assertEqual(self.result()['status'], 'INCOMPLETE')

    def test_total_must_include_phases(self):
        self.b['trials'][0]['total'] = 5
        self.assertEqual(self.result()['status'], 'INCOMPLETE')

    def test_fast_kernel_slow_total(self):
        self.b['trials'][0]['total'] = 20
        self.assertEqual(self.result()['status'], 'SPEED_FAIL')

    def test_no_hardcoded_two_times(self):
        for t in self.b['trials']:
            t.update(integration=8, total=13)
        self.assertEqual(self.result()['status'], 'PASS')

    def test_repeats_and_pairing(self):
        self.b['trials'][0]['pair_id'] = 'other'
        self.assertEqual(self.result()['status'], 'NOT_COMPARABLE')
        self.b['trials'][0]['pair_id'] = '1'
        self.assertEqual(self.result()['status'], 'INCOMPLETE')

    def test_partial_payload_and_nonobject(self):
        for value in ({}, {'verdict': 'PASS'}, [], None):
            self.assertEqual(evaluate(self.a, value, self.p, digest(self.p))['status'], 'INCOMPLETE')

    def test_both_runs_cannot_replace_frozen_scenario(self):
        self.a['forcing_sha256'] = self.b['forcing_sha256'] = 'b' * 64
        self.assertEqual(self.result()['status'], 'NOT_COMPARABLE')

    def test_wrong_registered_source(self):
        self.b['source_sha'] = 'b' * 40
        self.assertEqual(self.result()['status'], 'NOT_COMPARABLE')

    def test_source_dirty(self):
        self.b['source_clean'] = False
        self.assertEqual(self.result()['status'], 'INCOMPLETE')

    def test_bad_metric_definition(self):
        self.b['quality']['heat_budget']['definition_sha256'] = 'b' * 64
        self.assertEqual(self.result()['status'], 'INCOMPLETE')


class EvidenceTests(unittest.TestCase):
    def test_artifact_tampering_and_escape(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'data'
            path.write_text('actual bytes')
            run = attach_files({}, temp, ['data'])
            self.assertEqual(verify_files(run, temp), [])
            path.write_text('changed')
            self.assertTrue(verify_files(run, temp))
            with self.assertRaises(ValueError):
                attach_files({}, temp, ['../missing'])

    def test_unattached_hash(self):
        with tempfile.TemporaryDirectory() as temp:
            Path(temp, 'data').write_text('x')
            run = attach_files({'forcing_sha256': 'a' * 64}, temp, ['data'])
            self.assertTrue(verify_files(run, temp))

    def test_duplicate_json_and_nan(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp, 'bad.json')
            for content in ('{"x":0,"x":1}', '{"x":NaN}'):
                path.write_text(content)
                with self.assertRaises(ValueError):
                    load(path)

    def test_cli_incomplete_nonzero(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp, 'empty.json')
            path.write_text('{}')
            completed = subprocess.run([sys.executable, str(TOOLS / 'gate.py'),
                                        '--control', str(path), '--candidate', str(path),
                                        '--policy', str(path), '--policy-sha256', 'a' * 64,
                                        '--out', str(Path(temp, 'result.json'))],
                                       capture_output=True, text=True, timeout=10)
            self.assertEqual(completed.returncode, 2)
            self.assertEqual(json.loads(completed.stdout)['status'], 'INCOMPLETE')

    def test_cli_all_verdicts_with_attached_bytes(self):
        for expected, code in [('PASS', 0), ('QUALITY_FAIL', 1), ('SPEED_FAIL', 1),
                               ('NOT_COMPARABLE', 3), ('INCOMPLETE', 2)]:
            with self.subTest(expected=expected), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                (root / 'fixture.txt').write_text('synthetic unit-test evidence only')
                a, b, policy = fixtures()
                attached = attach_files({}, root, ['fixture.txt'])['artifacts']
                h = attached['fixture.txt']
                for run in (a, b):
                    for key in list(run):
                        if key.endswith('_sha256'):
                            run[key] = h
                    for item in run['quality'].values():
                        item.update(definition_sha256=h, evidence_sha256=h)
                    run['artifacts'] = attached
                for rule in policy['metrics'].values():
                    rule['definition_sha256'] = h
                policy['pair_contract'] = {key: copy.deepcopy(a[key]) for key in PAIR_FIELDS}
                for item in policy['runs'].values():
                    item['config_sha256'] = h
                if expected == 'QUALITY_FAIL':
                    b['quality']['heat_budget']['value'] = .9
                if expected == 'SPEED_FAIL':
                    b['trials'][0]['total'] = 30
                if expected == 'NOT_COMPARABLE':
                    b['precision'] = 'float32'
                if expected == 'INCOMPLETE':
                    (root / 'fixture.txt').write_text('tampered')
                for name, value in [('control', a), ('candidate', b), ('policy', policy)]:
                    (root / f'{name}.json').write_text(json.dumps(value))
                completed = subprocess.run(
                    [sys.executable, str(TOOLS / 'gate.py'), '--control', str(root / 'control.json'),
                     '--candidate', str(root / 'candidate.json'), '--policy', str(root / 'policy.json'),
                     '--policy-sha256', digest(policy), '--out', str(root / 'result.json')],
                    text=True, capture_output=True, timeout=10)
                self.assertEqual(completed.returncode, code, completed.stderr)
                result = json.loads(completed.stdout)
                self.assertEqual(result['status'], expected)
                self.assertFalse(result['industrial_qualified'])

    def test_synchronized_phase_accounting(self):
        ticks = iter(range(12))
        synchronized = []
        def operation(previous):
            return 1 if previous is None else previous + 1
        timings, final = measure_lifecycle(
            setup=operation, compile_cold=operation, warmup=operation, integrate=operation,
            diagnostics_io=operation, synchronize=synchronized.append,
            pair_id='p', clock=lambda: next(ticks))
        self.assertEqual(synchronized, [1, 2, 3, 4, 5])
        self.assertEqual(final, 5)
        self.assertEqual(timings['total'], 11)
        self.assertEqual(timings['integration'], 1)

    def test_failed_sync_not_recorded_as_success(self):
        def fail(value):
            raise RuntimeError('device failed')
        with self.assertRaises(RuntimeError):
            measure_lifecycle(setup=lambda _: None, compile_cold=None, warmup=None,
                              integrate=None, diagnostics_io=None, synchronize=fail, pair_id='p')


if __name__ == '__main__':
    unittest.main()
