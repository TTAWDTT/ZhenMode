"""Compile actual pinned/patched MOM6 coefficients against the author oracle.

Run inside Linux under external single-CPU/180s/4GiB supervision. This is a
component experiment, not a coupled model or an ocean benchmark score.
"""
import argparse
import json
import math
import subprocess
from pathlib import Path

from zhenmode.baselines.mom6.adapter import adapter_identity
from zhenmode.baselines.mom6.omip2 import corrected_coefficients
from zhenmode.provenance.sources import load_json, sha256_file


def check(source_path, reference_path, output, fms_include):
    # Bind the existing author-generated fixture, not a result of our own port.
    if sha256_file(reference_path) != '5d43d440d4e4b32ccd2de4e0ece6db0e74abc728b97f13448b682dcff4c5b76d':
        raise ValueError('independent author reference identity changed')
    reference = load_json(reference_path)
    raw = Path(source_path).read_bytes()
    patched = corrected_coefficients(raw)
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    result = {'scope': 'transfer_coefficients_component_only',
              'harness_sha256': sha256_file(__file__), 'adapter_source_identity': adapter_identity(),
              'reference_source': reference['source_url'],
              'reference_fixture_sha256': sha256_file(reference_path),
              'source_sha256': sha256_file(source_path), 'climate_qualification': False,
              'rtol': 1e-7, 'atol': 1e-14, 'results': {}}
    flags = ['-O0', '-ffree-line-length-none', '-fdefault-real-8', '-fdefault-double-8']
    fms_include = Path(fms_include).resolve()
    modules = {name: sha256_file(fms_include / name) for name in ('constants_mod.mod', 'fmsconstants.mod')}
    flags += ['-I' + str(fms_include)]
    result['fms_constant_modules'] = modules
    result['compiler'] = subprocess.check_output(['gfortran', '--version'], text=True)
    result['flags'] = flags
    for name, source in [('upstream', raw), ('adapted', patched)]:
        text = source.decode('utf-8')
        first = text.index('subroutine ncar_ocean_fluxes (')
        last = text.index('end subroutine ncar_ocean_fluxes', first) + len('end subroutine ncar_ocean_fluxes')
        routine = text[first:last]
        driver = ['program coefficient_check', 'use constants_mod, only: grav, vonkarm', 'implicit none',
                  'logical :: ncar_ocean_flux_orig=.false.',
                  'real :: cd(1),ch(1),ce(1),us(1),bs(1)']
        for row in reference['cases']:
            values = [row[k] for k in ('scalar_wind', 'theta_air_k', 'temperature_surface_k',
                                      'humidity_air', 'humidity_surface')]
            args = ','.join(f'[{v:.17e}]' for v in values)
            driver += [f'call ncar_ocean_fluxes({args},[10.0],[.true.],cd,ch,ce,us,bs)',
                       'write(*,"(3(es25.17,1x))") cd(1),ch(1),ce(1)']
        driver += ['contains', routine, 'end program']
        filename = output / f'{name}.f90'
        filename.write_text('\n'.join(driver) + '\n')
        binary = output / name
        subprocess.run(['gfortran', *flags, str(filename), '-o', str(binary)],
                       check=True, timeout=60, cwd=output)
        values = subprocess.check_output([str(binary)], text=True, timeout=30)
        (output / f'{name}.txt').write_text(values)
        parsed = [[float(value) for value in line.split()] for line in values.splitlines()]
        if len(parsed) != len(reference['cases']) or any(len(row) != 3 for row in parsed):
            raise ValueError('compiled oracle harness returned incomplete results')
        errors, passed = [], True
        for actual, expected in zip(parsed, reference['cases'], strict=True):
            for value, key in zip(actual, ('cd', 'ch', 'ce'), strict=True):
                error = abs(value - expected[key])
                errors.append(error / abs(expected[key]))
                passed &= math.isfinite(value) and error <= 1e-14 + 1e-7 * abs(expected[key])
        result['results'][name] = {'pass': bool(passed), 'cases': len(parsed),
                                   'maximum_relative_error': max(errors),
                                   'compiled_source_sha256': sha256_file(filename),
                                   'binary_sha256': sha256_file(binary), 'values': parsed}
    result['status'] = 'PASS' if result['results']['adapted']['pass'] and not result['results']['upstream']['pass'] else 'FAIL'
    (output / 'report.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--source', required=True)
    parser.add_argument('--reference', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--fms-include', required=True,
                        help='actual built FMS modules; do not substitute host constants')
    args = parser.parse_args()
    result = check(args.source, args.reference, args.output, args.fms_include)
    print(json.dumps(result, indent=2, allow_nan=False))
    raise SystemExit(0 if result['status'] == 'PASS' else 1)
