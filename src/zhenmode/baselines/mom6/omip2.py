"""Pinned MOM6+SIS2 candidate preparation; unresolved mechanisms fail qualification."""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import uuid
from pathlib import Path

from zhenmode.baselines.mom6.adapter import _command, adapter_identity, source_identity
from zhenmode.baselines.mom6.coupled_sources import (
    corrected_latent_energy,
    expected_sources,
    source_tree,
    stage_sources,
)
from zhenmode.evaluation.protocols import digest
from zhenmode.execution.benchmark import plan
from zhenmode.execution.resources import run_process_group
from zhenmode.provenance.sources import load_json, sha256_file

PINS = load_json(Path(__file__).with_name('omip2-pins.json'))
GAPS = [
    'staged LY2009/Gill patch requires compiled reference checks and coupled build receipt',
    'explicit SIS2-to-MOM latent energy needs compiled coupling checks; atmosphere/ice stocks remain unaudited',
    'bounded mean fluxes must be held/integrated, not native FMS linear interpolation',
    'conservative radiation/precipitation/runoff mapping receipts required',
    'JRA55-do v1.4.0, WOA, bathymetry and observations actual data identities required',
    'TEOS10 variable conversions and nonlinear EOS reference checks required',
    'native ice/mixing/shortwave choices and independent water/ice/heat/salt budgets required',
    'monthly diagnostics, new scorer and source-to-binary build receipt required',
]


def corrected_coefficients(source):
    """Patch the actual pinned coupler, never an adapter facade or upstream cache.

    Scope is transfer coefficients only. Gill saturation/air properties,
    potential temperature, latent heat and coupling remain separate gates.
    The upstream LGPL notice and the unused historical branch are preserved.
    """
    if hashlib.sha256(source).hexdigest() != PINS['coupler_audit']['sha256']:
        raise ValueError('audited coupler implementation changed')
    text = source.decode('utf-8')
    first = text.index('subroutine ncar_ocean_fluxes (')
    last = text.index('end subroutine ncar_ocean_fluxes', first)
    routine = text[first:last]

    def replace(old, new, count):
        nonlocal routine
        if routine.count(old) != count:
            raise ValueError(f'ambiguous pinned coefficient patch: {old}')
        routine = routine.replace(old, new)

    replace('integer, parameter :: n_itts = 2',
            'integer, parameter :: n_itts = 5\n'
            '  ! Protocol constants are local: do not change native ice/ocean gravity.\n'
            '  real, parameter :: grav = 9.81, vonkarm = 0.4', 1)
    replace('real :: cd_rt', 'real :: previous_cd\n  real :: cd_rt', 1)
    replace('(2.7/u10+0.142+0.0764*u10)/1e3',
            'merge(2.34, 2.7/u10+0.142+0.0764*u10-3.14807e-10*u10**6, u10>=33.0)/1e3', 4)
    replace('tv = t(i)*(1+0.608*q(i));',
            'tv = t(i)*(1+(28.966/18.016-1)*q(i));', 2)
    replace('qstar/(q(i)+1/0.608)', 'qstar/(q(i)+1/(28.966/18.016-1))', 2)
    replace('cd_rt = sqrt(cd(i));', 'previous_cd = cd(i)\n                cd_rt = sqrt(cd(i));', 2)
    replace('u10 = u/(1+cd_n10_rt*(log(z(i)/10)-psi_m)/vonkarm);',
            'u10 = max(0.3, u/(1+cd_n10_rt*(log(z(i)/10)-psi_m)/vonkarm));', 2)
    # A convergence decision uses coefficients computed by this iteration;
    # it must not replace a diagnostic residual with its expected value.
    replace('             end do',
            '                if (abs(cd(i)-previous_cd)/(cd(i)+1e-8)<1e-4) exit\n'
            '             end do\n'
            '             ustar(i) = sqrt(cd(i))*u\n'
            '             tstar = ch(i)/sqrt(cd(i))*(t(i)-ts(i))\n'
            '             qstar = ce(i)/sqrt(cd(i))*(q(i)-qs(i))\n'
            '             bstar(i) = grav*(tstar/tv+qstar/(q(i)+1/(28.966/18.016-1)))', 2)
    return (text[:first] + '! ZhenMode OMIP2 coefficient adaptation; see preparation.json.\n'
            + routine + text[last:]).encode('utf-8')


def corrected_surface_exchange(source):
    """Gill open-water air properties in the actual coupler; ice retains native exchange.

    This does not fix SIS2 latent-energy transfer or qualify the radiation,
    freshwater and temporal mapping. Those gates must still be satisfied.
    """
    text = corrected_coefficients(source).decode('utf-8')
    first = text.index('subroutine surface_flux_1d (')
    last = text.index('end subroutine surface_flux_1d', first)
    routine = text[first:last]

    def replace(old, new):
        nonlocal routine
        if routine.count(old) != 1:
            raise ValueError(f'ambiguous pinned open-water exchange patch: {old}')
        routine = routine.replace(old, new)

    replace('integer :: i, nbad', 'real :: heat_capacity(size(t_atm)), celsius(size(t_atm)), pressure_hpa(size(t_atm))\n'
            '  real, parameter :: eps_gill = 18.016/28.966, virtual_gill = 28.966/18.016-1\n'
            '  integer :: i, nbad')
    replace('  ! initilaize surface air humidity according to surface type',
            '  ! Gill seawater saturation: Raoult factor acts on vapor pressure, not on q.\n'
            '  where (avail .and. seawater)\n'
            '    celsius = t_surf0 - 273.15\n'
            '    pressure_hpa = p_surf/100\n'
            '    e_sat = 0.98*10**((0.7859+0.03477*celsius)/(1+0.00412*celsius)) &\n'
            '            *(1+1e-6*pressure_hpa*(4.5+0.0006*celsius**2))\n'
            '    q_sat = eps_gill*e_sat/(pressure_hpa-(1-eps_gill)*e_sat)\n'
            '    celsius = t_surf1 - 273.15\n'
            '    e_sat1 = 0.98*10**((0.7859+0.03477*celsius)/(1+0.00412*celsius)) &\n'
            '             *(1+1e-6*pressure_hpa*(4.5+0.0006*celsius**2))\n'
            '    q_sat1 = eps_gill*e_sat1/(pressure_hpa-(1-eps_gill)*e_sat1)\n'
            '  endwhere\n\n  ! initilaize surface air humidity according to surface type')
    replace('  if (raoult_sat_vap) where (seawater) q_surf0 = 0.98 * q_surf0',
            '  ! Seawater Raoult correction was already applied to e_sat above.')
    replace('  if(alt_gustiness) then',
            '  heat_capacity = cp_air\n'
            '  where (avail .and. seawater)\n'
            '    heat_capacity = 1004.6*(1+0.8735*q_atm)\n'
            '    th_atm = t_atm + 9.81*10/heat_capacity\n'
            '    tv_atm = t_atm*(1+virtual_gill*q_atm)\n'
            '    thv_atm = th_atm*(1+virtual_gill*q_atm)\n'
            '    thv_surf = t_surf0*(1+virtual_gill*q_surf0)\n'
            '    p_ratio = 1\n'
            '  endwhere\n'
            '  if (any(avail .and. seawater .and. (abs(z_atm-10)>1e-10 .or. p_surf/=p_atm &\n'
            '      .or. rough_mom/=rough_scale))) &\n'
            '    call mpp_error(FATAL,"OMIP Gill exchange requires 10m height and common surface pressure")\n'
            '  if (.not.ncar_ocean_flux .or. ncar_ocean_flux_orig .or. .not.alt_gustiness &\n'
            '      .or. gust_const/=0.5 .or. use_mixing_ratio .or. do_simple) &\n'
            '    call mpp_error(FATAL,"OMIP Gill exchange configuration mismatch")\n\n'
            '  if(alt_gustiness) then')
    replace('     ! sensible heat flux',
            '     where (seawater) rho = p_atm/(287.04*tv_atm)\n\n     ! sensible heat flux')
    replace('rho_drag = cp_air * drag_t * rho', 'rho_drag = heat_capacity * drag_t * rho')
    replace('     ! stresses',
            '     where (seawater)\n'
            '       flux_r = 5.670374419e-8*t_surf**4\n'
            '       drdt_surf = 4*5.670374419e-8*t_surf**3\n'
            '     endwhere\n\n     ! stresses')
    return (text[:first] + '! ZhenMode OMIP2 Gill open-water adaptation; see preparation.json.\n'
            + routine + text[last:]).encode('utf-8')


def preparation_plan(profile='integration-6h'):
    return plan(profile, 'mom6', 3600) | {
        'candidate': PINS,
        'mechanism_gaps': GAPS,
        'build_plan': {
            'platform': 'Linux/WSL with Fortran, MPI and NetCDF development dependencies',
            'target': 'ice_ocean_SIS2', 'fms_codebase': 'src/FMS2', 'jobs': 1,
            'command_template': ['make', '-j1', 'ice_ocean_SIS2', 'FMS_CODEBASE=src/FMS2',
                                 'BUILD=<isolated-absolute-build-directory>'],
            'status': 'not_started_requires_resource_plan_and_reviewed_source_changes',
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
    controls = (' ncar_ocean_flux=.true., ncar_ocean_flux_orig=.false.,\n'
                ' raoult_sat_vap=.true., alt_gustiness=.true., gust_const=0.5,\n'
                ' use_virtual_temp=.true., no_neg_q=.false.,\n'
                ' use_mixing_ratio=.false., do_simple=.false.\n')
    nml, count = re.subn(r'(?im)^\s*&surface_flux_nml\b[^/]*^\s*/',
                        '&surface_flux_nml\n' + controls + '/', nml)
    if count != 1:
        raise ValueError('expected one unambiguous surface_flux_nml')
    restore = _replace_setting(files['MOM_saltrestore'], 'FLUXCONST', f'{50 / 365:.17g}')
    restore = _replace_setting(restore, 'ADJUST_NET_FRESH_WATER_TO_ZERO', 'False')
    restore = _replace_setting(restore, 'MAX_DELTA_SRESTORE', '1.0e20')
    # WOA source file, EOS and remaining upstream options are not silently relabelled.
    return files | {'input.nml': nml, 'MOM_saltrestore': restore}


def prepare(examples_dir, output, *, profile='integration-6h'):
    if os.name != 'posix':
        raise ValueError('prepare inside Linux/WSL using native Git symlink semantics')
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
    patched = corrected_surface_exchange((examples_dir / audit['path']).read_bytes())
    transformed = corrected_configuration(files, profile)
    output.mkdir(parents=True, exist_ok=False)
    for name, content in transformed.items():
        (output / name).write_text(content, encoding='utf-8')
    patches = {audit['path']: patched} | {
        path: corrected_latent_energy(path, (examples_dir / path).read_bytes())
        for path in PINS['latent_energy_sources']}
    for path, content in patches.items():
        staged_source = output / path
        staged_source.parent.mkdir(parents=True, exist_ok=True)
        staged_source.write_bytes(content)
    result = preparation_plan(profile) | {
        'examples_dir': str(examples_dir),
        'execution_status': 'proposed', 'configuration_prepared': True,
        'execution_ready': False, 'qualification': 'blocked_by_listed_mechanism_gaps',
        'source_identity': identities,
        'staged_files': {name: sha256_file(output / name) for name in transformed},
        'source_patch': {'path': audit['path'], 'upstream_sha256': audit['sha256'],
                         'staged_sha256': sha256_file(output / audit['path']),
                         'scope': 'LY2009_Gill_open_water_exchange_excludes_SIS2_latent_energy',
                         'upstream_cache_modified': False,
                         'required_configuration': {'ncar_ocean_flux': True,
                                                    'ncar_ocean_flux_orig': False,
                                                    'alt_gustiness': True, 'gust_const': 0.5,
                                                    'use_mixing_ratio': False, 'do_simple': False},
                         'compiled_reference_check': 'not_run_by_preparation'},
        'source_patches': {path: {'upstream_sha256': sha256_file(examples_dir / path),
                                  'staged_sha256': sha256_file(output / path)} for path in patches},
        'latent_energy_scope': 'explicit_bottom_energy_no_atmosphere_stock_or_complete_ice_qualification',
        'changes': ['Gregorian calendar', 'profile duration', '50m/year SSS piston',
                    'no global freshwater adjustment', 'no physical-range SSS clipping',
                    'LY2009/Gill open-water exchange configuration',
                    'Gill water latent heat and explicit energy through ice/ocean interface and stocks'],
        'not_a_run_directory': 'INPUT, layouts, overrides, diag_table and build/data receipts remain required',
    }
    (output / 'preparation.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    return result


def build_segment(preparation, output, *, wall_seconds=180):
    """Resumable single-CPU coupled build, still not permission to run/score.

    Every segment rechecks all pinned upstream files and the staged patch.
    Compiler flags enter as environment, so autoconf's required flags survive
    recursive make. A different source/tool recipe needs a separate output.
    """
    if os.name != 'posix' or type(wall_seconds) is not int or not 1 <= wall_seconds <= 180:
        raise ValueError('coupled build requires Linux/WSL and a 1-180s segment')
    tool_names = ('mpicc', 'mpif90', 'gfortran', 'gcc', 'ld', 'python3', 'make', 'autoreconf', 'nf-config')
    missing = [name for name in tool_names if shutil.which(name) is None]
    if missing:
        raise ValueError('missing coupled build tools: ' + ', '.join(missing))
    receipt = load_json(Path(preparation) / 'preparation.json')
    examples = Path(receipt['examples_dir'])
    if receipt['source_identity']['examples'] != source_identity(examples, PINS['examples']['commit']):
        raise ValueError('examples source identity changed since preparation')
    for path, commit in (PINS['components'] | PINS['nested']).items():
        if receipt['source_identity'][path] != source_identity(examples / path, commit):
            raise ValueError(f'component source identity changed: {path}')
    known = {examples / path / name for path in PINS['components'] | PINS['nested']
             for name in receipt['source_identity'][path]['tracked_files']}
    for path in PINS['components']:
        for actual in (examples / path).rglob('*'):
            if '.git' in actual.parts:
                continue
            if actual.is_file() and actual.suffix.lower() in {'.f90', '.f', '.f95', '.f03', '.f08', '.c', '.cc', '.cpp', '.cxx', '.h', '.inc'} and actual not in known:
                raise ValueError(f'unregistered compilable source: {actual}')
    patches = {PINS['coupler_audit']['path']: corrected_surface_exchange(
        (examples / PINS['coupler_audit']['path']).read_bytes())} | {
        path: corrected_latent_energy(path, (examples / path).read_bytes())
        for path in PINS['latent_energy_sources']}
    if set(receipt.get('source_patches', {})) != set(patches):
        raise ValueError('preparation lacks the current explicit energy patch set; prepare a new directory')
    for path, content in patches.items():
        patch = Path(preparation) / path
        if (patch.read_bytes() != content
                or sha256_file(patch) != receipt['source_patches'][path]['staged_sha256']):
            raise ValueError('staged patch differs from the checked transformation: ' + path)
    output = Path(output).resolve()
    environment = dict(os.environ, CC='mpicc', MPICC='mpicc', FC='mpif90', MPIFC='mpif90',
                       FCFLAGS='-O1 -g0 -fdefault-real-8 -fdefault-double-8 -I/usr/include',
                       CPPFLAGS='', CFLAGS='-O1 -g0', LDFLAGS='', LIBS='',
                       PYTHON='python3', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    executed = expected_sources(examples, receipt['source_identity'], patches)
    recipe = {'preparation_sha256': sha256_file(Path(preparation) / 'preparation.json'),
              'adapter_source_identity': adapter_identity(),
              'flags': {key: environment[key] for key in ('CC', 'FC', 'MPICC', 'MPIFC', 'FCFLAGS', 'CPPFLAGS', 'CFLAGS', 'LDFLAGS', 'LIBS', 'PYTHON')},
              'tools': {name: {'path': shutil.which(name),
                               'sha256': sha256_file(shutil.which(name)),
                               'version': _command([name, '--version'])}
                        for name in tool_names},
              'source_patches': receipt['source_patches'], 'target': 'ice_ocean_SIS2',
              'executed_source_tree_sha256': digest(executed),
              'source_staging': 'fresh_git_archives_exclude_untracked_build_controls'}
    if output.exists():
        if load_json(output / 'recipe.json') != recipe:
            raise ValueError('build output belongs to another source/flag recipe')
    else:
        output.mkdir(parents=True)
        (output / 'recipe.json').write_text(json.dumps(recipe, indent=2) + '\n')
        stage_sources(examples, output / 'sources', patches)
    if (load_json(output / 'executed-sources.json') != executed
            or source_tree(output / 'sources') != executed):
        raise ValueError('actual staged sources/build controls changed outside the recorded recipe')
    checkpoint = output / 'artifact-checkpoint.json'
    artifacts_before = _build_artifacts(output / 'build')
    if checkpoint.exists():
        if load_json(checkpoint)['artifacts'] != artifacts_before:
            raise ValueError('build intermediates changed outside the recorded segments; use a new output')
    elif artifacts_before:
        raise ValueError('existing build lacks an intermediate-file checkpoint; use a new output')
    recovery = _recover_dependencies(output)
    command = ['make', '-j1', 'ice_ocean_SIS2', 'FMS_CODEBASE=src/FMS2',
               f'BUILD={output}/build']
    segment = uuid.uuid4().hex
    with (output / f'{segment}.log').open('x') as log:
        result, limits = run_process_group(command, cwd=output / 'sources', env=environment, stdout=log,
                                          resources={'cpu': 1, 'wall_seconds': wall_seconds, 'memory_mib': 4096,
                                                     'termination_grace_seconds': min(5, wall_seconds-1)})
    artifacts = _build_artifacts(output / 'build')
    # Bind resumable objects, modules, archives and generated execution recipes,
    # not just the final facade/binary. Preserve each segment's immutable snapshot.
    snapshot = {'segment': segment, 'artifacts': artifacts}
    (output / f'{segment}-artifacts.json').write_text(json.dumps(snapshot, indent=2) + '\n')
    temporary = checkpoint.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(snapshot, indent=2) + '\n')
    temporary.replace(checkpoint)
    binary = output / 'build/ice_ocean_SIS2/coupler_main'
    completed = result.returncode == 0 and binary.is_file()
    failure_reason = None
    try:
        source_unchanged = (recipe['adapter_source_identity'] == adapter_identity()
                            and receipt['source_identity']['examples'] == source_identity(examples, PINS['examples']['commit'])
                            and all(receipt['source_identity'][path] == source_identity(examples / path, commit)
                                    for path, commit in (PINS['components'] | PINS['nested']).items())
                            and source_tree(output / 'sources') == executed
                            and all(sha256_file(Path(preparation) / path) == receipt['source_patches'][path]['staged_sha256']
                                    for path in patches))
    except (ValueError, OSError, RuntimeError) as error:
        source_unchanged, failure_reason = False, str(error)
    if not source_unchanged:
        completed = False
        failure_reason = failure_reason or 'execution source changed during build'
    dependencies = None
    if completed:
        try:
            dependencies = _linked_libraries(binary)
        except (ValueError, OSError, RuntimeError) as error:
            completed, failure_reason = False, str(error)
    report = {'execution_status': 'failed' if not source_unchanged else 'completed' if completed else
              'incomplete' if limits['stop_reason'] == 'wall_limit' else 'failed',
              'kind': 'coupled_build_segment', 'recipe': recipe, 'command': command,
              'resources': limits, 'exit_code': result.returncode,
              'binary_sha256': sha256_file(binary) if completed else None,
              'artifact_snapshot_sha256': sha256_file(output / f'{segment}-artifacts.json'),
              'executed_sources_sha256': sha256_file(output / 'executed-sources.json'),
              'dependency_recovery': recovery,
              'source_unchanged_after_segment': source_unchanged,
              'failure_reason': failure_reason,
              'linked_libraries': dependencies,
              'qualification': 'build_only_remaining_mechanism_data_diagnostic_gates',
              'execution_ready': False, 'climate_qualification': False}
    (output / f'{segment}.json').write_text(json.dumps(report, indent=2) + '\n')
    return report


def _build_artifacts(directory):
    """No unrecorded cached object may enter a resumed source-to-binary build."""
    return {path.relative_to(directory).as_posix(): sha256_file(path)
            for path in sorted(directory.rglob('*')) if path.is_file()}


def _recover_dependencies(output):
    """Retain then invalidate a truncated native makedep output, never model data.

    Caller must verify the preceding artifact checkpoint before invoking this.
    The pinned generator writes object rules before the executable link rule;
    interruption can leave a parseable file with no coupler_main target.
    """
    path = output / 'build/ice_ocean_SIS2/Makefile.dep'
    if not path.exists():
        return None
    if path.is_symlink() or not path.resolve().is_relative_to((output / 'build').resolve()):
        raise ValueError('dependency recovery target escapes the isolated build')
    text = path.read_text()
    if re.search(r'(?m)^coupler_main\s*:', text):
        return None
    if not text.startswith('# Makefile.dep created by makedep\n'):
        raise ValueError('unrecognized dependency file; refuse automatic recovery')
    retained = output / 'recovery' / uuid.uuid4().hex / 'Makefile.dep'
    retained.parent.mkdir(parents=True)
    shutil.copyfile(path, retained)
    report = {'kind': 'truncated_native_generated_dependencies',
              'original': path.relative_to(output).as_posix(),
              'retained': retained.relative_to(output).as_posix(),
              'sha256': sha256_file(retained), 'numerical_source_modified': False}
    (retained.parent / 'recovery.json').write_text(json.dumps(report, indent=2) + '\n')
    path.unlink()
    return report


def _linked_libraries(binary):
    libraries = {}
    for line in _command(['ldd', binary]).splitlines():
        if 'not found' in line:
            raise ValueError('coupled binary has an unresolved runtime library: ' + line.strip())
        match = re.search(r'(?:=>\s+)?(/\S+)', line)
        if match:
            library = Path(match[1]).resolve()
            libraries[str(library)] = sha256_file(library)
    if not libraries:
        raise ValueError('no inspected dynamic runtime libraries for the coupled binary')
    return libraries
