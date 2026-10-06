"""Pinned MOM6+SIS2 candidate preparation; unresolved mechanisms fail qualification."""
from __future__ import annotations

import json
import re
from pathlib import Path

from zhenmode.baselines.mom6.adapter import _command, source_identity
from zhenmode.execution.benchmark import plan
from zhenmode.provenance.sources import load_json, sha256_file

PINS = load_json(Path(__file__).with_name('omip2-pins.json'))
GAPS = [
    'coupler NCAR uses LY2004 drag and two iterations; reviewed LY2009 change required',
    'Gill saturation and air properties plus 10m potential-temperature conversion required',
    'conservative radiation/precipitation/runoff mapping receipts required',
    'JRA55-do v1.4.0, WOA, bathymetry and observations actual data identities required',
    'TEOS10 variable conversions and nonlinear EOS reference checks required',
    'native ice/mixing/shortwave choices and independent water/ice/heat/salt budgets required',
    'monthly diagnostics, new scorer and source-to-binary build receipt required',
]


def preparation_plan(profile='integration-6h'):
    return plan(profile, 'mom6', 3600) | {
        'candidate': PINS,
        'mechanism_gaps': GAPS,
        'build_plan': {
            'platform': 'Linux/WSL with Fortran, MPI and NetCDF development dependencies',
            'target': 'ice_ocean_SIS2', 'fms_codebase': 'src/FMS2', 'jobs': 1,
            'command_template': ['make', '-j1', 'ice_ocean_SIS2', 'FMS_CODEBASE=src/FMS2',
                                 'BUILD=<isolated-absolute-build-directory>'],
            'status': 'not_started_requires_resource_approval_and_reviewed_source_changes',
            'flags': 'compiler, MPI, NetCDF and precision must be resolved and receipted',
        },
        'source_preparation': {
            'checkout': PINS['examples']['commit'],
            'initialize_submodules': list(PINS['components']) + list(PINS['nested']),
            'command_templates': [
                ['git', 'clone', '--no-checkout', '--filter=blob:none', PINS['examples']['url'], '<isolated-examples-cache>'],
                ['git', '-C', '<isolated-examples-cache>', 'checkout', '--detach', PINS['examples']['commit']],
                ['git', '-C', '<isolated-examples-cache>', 'submodule', 'update', '--init', '--recursive',
                 *PINS['components']],
            ],
            'policy': 'isolated cache; no download or build performed by planning',
        },
    }


def _replace_setting(text, name, value):
    pattern = rf'^(\s*{re.escape(name)}\s*=\s*)([^!\n]+)'
    result, count = re.subn(pattern, lambda match: match[1] + value + ' ', text, flags=re.MULTILINE)
    if count != 1:
        raise ValueError(f'expected one unambiguous setting: {name}')
    return result


def corrected_configuration(files, profile):
    """Concrete known corrections only; outstanding source/physical changes stay blocked."""
    if profile not in {'integration-6h', 'integration-30d', 'integration-1y'}:
        raise ValueError('candidate preparation currently supports bounded first-cycle profiles only')
    days, hours = {'integration-6h': (0, 6), 'integration-30d': (30, 0), 'integration-1y': (365, 0)}[profile]
    nml = _replace_setting(files['input.nml'], 'calendar', "'gregorian',")
    nml = _replace_setting(nml, 'days', f'{days},')
    nml = _replace_setting(nml, 'hours', f'{hours},')
    restore = _replace_setting(files['MOM_saltrestore'], 'FLUXCONST', f'{50 / 365:.17g}')
    restore = _replace_setting(restore, 'ADJUST_NET_FRESH_WATER_TO_ZERO', 'False')
    restore = _replace_setting(restore, 'MAX_DELTA_SRESTORE', '1.0e20')
    # WOA source file, EOS and remaining upstream options are not silently relabelled.
    return files | {'input.nml': nml, 'MOM_saltrestore': restore}


def prepare(examples_dir, output, *, profile='integration-6h'):
    examples_dir, output = Path(examples_dir).resolve(), Path(output).resolve()
    top = Path(_command(['git', '-C', examples_dir, 'rev-parse', '--show-toplevel'])).resolve()
    if top != examples_dir:
        raise ValueError('examples-dir must be its own pinned Git checkout')
    identities = {'examples': source_identity(examples_dir, PINS['examples']['commit'])}
    for path, commit in (PINS['components'] | PINS['nested']).items():
        directory = examples_dir / path
        top = Path(_command(['git', '-C', directory, 'rev-parse', '--show-toplevel'])).resolve()
        if top != directory:
            raise ValueError(f'uninitialized source component: {path}')
        identities[path] = source_identity(directory, commit)
    files = {}
    for path, identity in PINS['configuration_files'].items():
        actual = examples_dir / path
        if sha256_file(actual) != identity['sha256']:
            raise ValueError(f'upstream configuration identity mismatch: {path}')
        files[actual.name] = actual.read_text(encoding='utf-8')
    audit = PINS['coupler_audit']
    if sha256_file(examples_dir / audit['path']) != audit['sha256']:
        raise ValueError('audited coupler implementation changed')
    transformed = corrected_configuration(files, profile)
    output.mkdir(parents=True, exist_ok=False)
    for name, content in transformed.items():
        (output / name).write_text(content, encoding='utf-8')
    result = preparation_plan(profile) | {
        'execution_status': 'proposed', 'configuration_prepared': True,
        'execution_ready': False, 'qualification': 'blocked_by_listed_mechanism_gaps',
        'source_identity': identities,
        'staged_files': {name: sha256_file(output / name) for name in transformed},
        'changes': ['Gregorian calendar', 'profile duration', '50m/year SSS piston',
                    'no global freshwater adjustment', 'no physical-range SSS clipping'],
        'not_a_run_directory': 'INPUT, layouts, overrides, diag_table and build/data receipts remain required',
    }
    (output / 'preparation.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    return result
