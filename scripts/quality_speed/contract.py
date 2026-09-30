"""Independent, fail-closed quality/speed contract. No solver imports."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from statistics import median

SCHEMA = 1
PHASES = ('setup', 'cold_compile', 'warmup', 'integration', 'diagnostics_io')
# Equal physical work and resources; algorithm/source may differ, but are recorded.
PAIR_FIELDS = ('scenario', 'evidence_class', 'hardware', 'precision', 'grid_sha256',
               'initial_sha256', 'forcing_sha256', 'boundary_sha256', 'physics_sha256',
               'scoring_sha256', 'output_sha256', 'duration_s', 'dt_s', 'dt_bt_s')
HASH_FIELDS = tuple(k for k in PAIR_FIELDS if k.endswith('_sha256'))
# Closed top-level schema: unrecognized status aliases must not be ignored.
RUN_FIELDS = frozenset(PAIR_FIELDS) | {
    'schema_version', 'model', 'algorithm', 'source_sha', 'source_clean',
    'source_tree_sha256', 'config_sha256', 'environment', 'command',
    'requested_steps', 'accepted_steps', 'attempted_steps', 'verdict',
    'coverage_complete', 'not_comparable', 'timing_scope', 'synchronization',
    'cold_cache', 'peak_memory', 'quality', 'trials', 'artifacts',
}
CATEGORIES = {'conservation', 'error', 'convergence'}
REQUIRED_METRICS = {'heat_budget': 'conservation', 'salt_budget': 'conservation',
                    'volume_budget': 'conservation', 'solution_error': 'error',
                    'convergence_order': 'convergence'}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def file_digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def number(value, positive=False):
    return (type(value) in (int, float) and math.isfinite(value)
            and (value > 0 if positive else value >= 0))


def sha(value, length=64):
    return (isinstance(value, str) and len(value) == length
            and all(c in '0123456789abcdef' for c in value))


def nonempty(value):
    return isinstance(value, str) and bool(value.strip())


def policy_errors(policy, expected_sha256):
    errors = []
    try:
        if not sha(expected_sha256) or digest(policy) != expected_sha256:
            errors.append('policy: pre-registered SHA256 mismatch')
    except (ValueError, TypeError):
        errors.append('policy: invalid JSON numbers/types')
    if (type(policy.get('schema_version')) is not int or policy['schema_version'] != SCHEMA
            or not nonempty(policy.get('registration'))):
        errors.append('policy: schema/registration missing')
    if not number(policy.get('minimum_speedup'), True) or policy['minimum_speedup'] <= 1:
        errors.append('policy: explicit significant speedup > 1 required')
    if type(policy.get('minimum_repeats')) is not int or policy['minimum_repeats'] < 3:
        errors.append('policy: at least three paired repeats required')
    pair = policy.get('pair_contract')
    if not isinstance(pair, dict) or any(key not in pair for key in PAIR_FIELDS):
        errors.append('policy: frozen pair_contract missing')
    versions = policy.get('runs')
    if not isinstance(versions, dict):
        errors.append('policy: frozen run versions missing')
    else:
        for label in ('control', 'candidate'):
            run = versions.get(label)
            if (not isinstance(run, dict) or not sha(run.get('source_sha'), 40)
                    or not sha(run.get('config_sha256')) or not nonempty(run.get('algorithm'))):
                errors.append(f'policy: frozen {label} source/config/algorithm missing')
    metrics = policy.get('metrics')
    if not isinstance(metrics, dict) or not metrics:
        return errors + ['policy: metrics missing']
    for name, category in REQUIRED_METRICS.items():
        if not isinstance(metrics.get(name), dict) or metrics[name].get('category') != category:
            errors.append(f'policy: mandatory {name}/{category} missing')
    categories = set()
    for name, rule in metrics.items():
        if not isinstance(rule, dict):
            errors.append(f'policy: {name} rule invalid')
            continue
        category = rule.get('category')
        if isinstance(category, str):
            categories.add(category)
        if (not isinstance(category, str) or category not in CATEGORIES or rule.get('direction') not in ('max', 'min')
                or not number(rule.get('limit')) or not nonempty(rule.get('unit'))
                or not sha(rule.get('definition_sha256'))):
            errors.append(f'policy: {name} incomplete rule')
        if category == 'convergence' and (rule.get('direction') != 'min'
                                          or not number(rule.get('limit'), True)):
            errors.append(f'policy: {name} requires positive minimum convergence order')
        if category in ('conservation', 'error') and rule.get('direction') != 'max':
            errors.append(f'policy: {name} must bound nonnegative absolute error')
    if not CATEGORIES <= categories:
        errors.append('policy: conservation, error and convergence all required')
    return errors


def run_errors(run, policy):
    errors = []
    for key in run.keys() - RUN_FIELDS:
        errors.append(f'unsupported run field: {key}')
    if type(run.get('not_comparable')) is not bool:
        errors.append('not_comparable: explicit boolean required')
    if type(run.get('schema_version')) is not int or run['schema_version'] != SCHEMA:
        errors.append('schema_version')
    for key in ('model', 'algorithm', 'scenario', 'precision', 'environment', 'command'):
        if not nonempty(run.get(key)):
            errors.append(key)
    if not sha(run.get('source_sha'), 40) or run.get('source_clean') is not True:
        errors.append('source_sha/source_clean')
    if not sha(run.get('config_sha256')) or not sha(run.get('source_tree_sha256')):
        errors.append('config/source tree hash')
    for key in HASH_FIELDS:
        if not sha(run.get(key)):
            errors.append(key)
    if run.get('evidence_class') not in ('synthetic_control', 'industrial_pair'):
        errors.append('evidence_class')
    hardware = run.get('hardware')
    if (not isinstance(hardware, dict)
            or any(not nonempty(hardware.get(k)) for k in ('device', 'cpu', 'accelerator', 'runtime'))
            or any(type(hardware.get(k)) is not int or hardware[k] <= 0
                   for k in ('threads', 'ranks', 'memory_limit_bytes'))):
        errors.append('hardware')
    for key in ('duration_s', 'dt_s', 'dt_bt_s'):
        if not number(run.get(key), True):
            errors.append(key)
    for key in ('requested_steps', 'accepted_steps', 'attempted_steps'):
        if type(run.get(key)) is not int or run[key] <= 0:
            errors.append(key)
    if not errors and (run['requested_steps'] != run['accepted_steps']
                       or run['attempted_steps'] != run['accepted_steps']
                       or not math.isclose(run['accepted_steps'] * run['dt_s'],
                                           run['duration_s'], rel_tol=1e-12, abs_tol=0)):
        errors.append('incomplete integration')
    if run.get('verdict') != 'PASS' or run.get('coverage_complete') is not True:
        errors.append('verdict/coverage_complete')
    if run.get('timing_scope') != 'full_integration_with_production_monitoring':
        errors.append('timing_scope')
    if run.get('synchronization') != 'all_state_and_diagnostics':
        errors.append('synchronization')
    if run.get('cold_cache') is not True:
        errors.append('cold_cache')
    memory = run.get('peak_memory')
    if not isinstance(memory, dict) or memory.get('status') not in ('measured', 'unavailable'):
        errors.append('peak_memory status')
    elif memory['status'] == 'measured':
        if not number(memory.get('bytes'), True) or not nonempty(memory.get('method')):
            errors.append('peak_memory measurement')
    elif not nonempty(memory.get('reason')):
        errors.append('peak_memory unavailable reason')
    measurements = run.get('quality')
    if not isinstance(measurements, dict):
        errors.append('quality')
    else:
        for name, rule in policy['metrics'].items():
            item = measurements.get(name)
            if (not isinstance(item, dict) or not number(item.get('value'))
                    or item.get('unit') != rule['unit']
                    or item.get('definition_sha256') != rule['definition_sha256']
                    or not sha(item.get('evidence_sha256'))):
                errors.append(f'quality.{name}')
    trials = run.get('trials')
    if not isinstance(trials, list) or len(trials) < policy['minimum_repeats']:
        errors.append('trials')
    else:
        ids = []
        for index, trial in enumerate(trials):
            if not isinstance(trial, dict):
                errors.append(f'trials.{index}')
                continue
            ids.append(trial.get('pair_id'))
            valid = all(number(trial.get(key)) for key in PHASES)
            if (not valid or not number(trial.get('total'), True)
                    or not number(trial.get('integration'), True)
                    or not nonempty(trial.get('pair_id'))):
                errors.append(f'trials.{index}: missing/invalid phase')
            elif trial['total'] < sum(trial[key] for key in PHASES):
                errors.append(f'trials.{index}: total excludes measured phases')
        if any(not nonempty(i) for i in ids) or len(set(map(str, ids))) != len(ids):
            errors.append('trials: duplicate/missing pair_id')
    return errors


def evaluate(control, candidate, policy, expected_policy_sha256):
    """Numbers are evidence summaries, not independently recomputed physics.

    The CLI verifies attached bytes. A trusted, frozen independent scorer must
    provide quality values; this contract cannot authenticate self-reported truth.
    """
    result = {'status': 'INCOMPLETE', 'pass': False, 'industrial_qualified': False,
              'quality': {}, 'speed': None, 'reasons': [], 'policy_sha256': expected_policy_sha256}
    if not all(isinstance(x, dict) for x in (control, candidate, policy)):
        result['reasons'] = ['inputs must be JSON objects']
        return result
    errors = policy_errors(policy, expected_policy_sha256)
    if not errors:
        errors = ([f'control: {e}' for e in run_errors(control, policy)]
                  + [f'candidate: {e}' for e in run_errors(candidate, policy)])
    if errors:
        result['reasons'] = errors
        return result
    result['evidence_class'] = control['evidence_class']
    mismatch = [f'{label}: declared not_comparable'
                for label, run in (('control', control), ('candidate', candidate))
                if run['not_comparable']]
    mismatch += [key for key in PAIR_FIELDS
                if digest(control[key]) != digest(candidate[key])
                or digest(control[key]) != digest(policy['pair_contract'][key])]
    for label, run in (('control', control), ('candidate', candidate)):
        for key in ('source_sha', 'config_sha256', 'algorithm'):
            if run[key] != policy['runs'][label][key]:
                mismatch.append(f'{label}: frozen {key}')
    a = {trial['pair_id']: trial for trial in control['trials']}
    b = {trial['pair_id']: trial for trial in candidate['trials']}
    if a.keys() != b.keys():
        mismatch.append('trial pair IDs')
    if mismatch:
        result.update(status='NOT_COMPARABLE', reasons=mismatch)
        return result
    for name, rule in policy['metrics'].items():
        base, new = control['quality'][name]['value'], candidate['quality'][name]['value']
        if rule['direction'] == 'max':
            passes = base <= rule['limit'] and new <= rule['limit'] and new <= base
        else:
            passes = base >= rule['limit'] and new >= rule['limit'] and new >= base
        result['quality'][name] = {'pass': passes, 'control': base, 'candidate': new,
                                   'limit': rule['limit'], 'unit': rule['unit']}
    if not all(item['pass'] for item in result['quality'].values()):
        result['status'] = 'QUALITY_FAIL'
        return result
    # Conservative paired envelope, not a statistical confidence interval.
    speed = {}
    for phase in ('integration', 'total'):
        ratios = [a[key][phase] / b[key][phase] for key in a]
        if not all(math.isfinite(ratio) and ratio > 0 for ratio in ratios):
            result['reasons'] = ['nonfinite/underflow speed ratio']
            return result
        speed[phase] = {'paired_speedups': ratios, 'median': median(ratios),
                        'min': min(ratios), 'max': max(ratios),
                        'control_median_s': median(t[phase] for t in a.values()),
                        'candidate_median_s': median(t[phase] for t in b.values())}
    passed = all(item['min'] >= policy['minimum_speedup'] for item in speed.values())
    result.update(status='PASS' if passed else 'SPEED_FAIL', speed=speed,
                  **{'pass': passed, 'industrial_qualified': passed and
                     control['evidence_class'] == 'industrial_pair'})
    return result
