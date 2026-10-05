"""Actual full coupler module + built FMS against independent component references.

Run with one CPU, 180s and 4GiB supervision. Latent energy transfer through
SIS2, ice, precipitation, absorbed radiation and ocean integration are excluded.
"""
import argparse
import json
import math
import shlex
import subprocess
from pathlib import Path

from zhenmode.baselines.mom6.adapter import adapter_identity
from zhenmode.baselines.mom6.omip2 import _linked_libraries, corrected_surface_exchange
from zhenmode.provenance.sources import load_json, sha256_file


def check(source, reference, fms_build, output, compare_zhenmode=False):
    if sha256_file(reference) != '5d43d440d4e4b32ccd2de4e0ece6db0e74abc728b97f13448b682dcff4c5b76d':
        raise ValueError('author coefficient reference changed')
    rows = load_json(reference)['cases']
    raw = Path(source).read_bytes()
    adapted = corrected_surface_exchange(raw)
    output, fms_build = Path(output).resolve(), Path(fms_build).resolve()
    output.mkdir(parents=True, exist_ok=False)
    # Invert the published Gill rational humidity equation to choose pressure
    # for each independent author coefficient fixture. No model function is
    # called to generate expected coefficients, humidity or fluxes.
    expected, inputs = [], []
    epsilon = 18.016 / 28.966
    for row in rows:
        q, qs, sst = row['humidity_air'], row['humidity_surface'], row['temperature_surface_k']
        celsius = sst - 273.15
        a = 0.98 * 10 ** ((0.7859 + 0.03477*celsius)/(1+0.00412*celsius))
        b = 1e-6 * (4.5+0.0006*celsius**2)
        pressure = 100*a*(qs*(1-epsilon)+epsilon)/(qs-a*b*(qs*(1-epsilon)+epsilon))
        cp = 1004.6*(1+0.8735*q)
        air_t = row['theta_air_k'] - 9.81*10/cp
        rho = pressure/(287.04*air_t*(1+(1/epsilon-1)*q))
        speed = row['scalar_wind']
        expected.append([rho*cp*speed*row['ch']*(sst-row['theta_air_k']),
                         rho*speed*row['ce']*(qs-q), 5.670374419e-8*sst**4,
                         -rho*speed**2*row['cd'], 0, row['cd'], row['ch'], row['ce'], qs])
        inputs.append([air_t, q, speed+0.3, 0, pressure, 10, pressure, sst, air_t])
    result = {'scope': 'open_water_coupler_component_not_SIS2_latent_energy_or_ocean',
              'harness_sha256': sha256_file(__file__), 'adapter_source_identity': adapter_identity(),
              'reference_sha256': sha256_file(reference), 'source_sha256': sha256_file(source),
              'fms_archive_sha256': sha256_file(fms_build / 'libFMS.a'),
              'climate_qualification': False, 'rtol': 1e-7, 'atol': 1e-12,
              'expected': expected, 'results': {}}
    flags = ['-cpp', '-O0', '-ffree-line-length-none', '-fdefault-real-8', '-fdefault-double-8',
             '-I' + str(fms_build)]
    libraries = shlex.split(subprocess.check_output(['nf-config', '--flibs'], text=True))
    for name, content in [('upstream', raw), ('adapted', adapted)]:
        directory = output / name
        directory.mkdir()
        (directory / 'surface_flux.F90').write_bytes(content)
        (directory / 'input.nml').write_text(
            '&surface_flux_nml\n ncar_ocean_flux=.true., ncar_ocean_flux_orig=.false.,\n'
            ' raoult_sat_vap=.true., alt_gustiness=.true., gust_const=0.5,\n'
            ' use_mixing_ratio=.false., do_simple=.false.\n/\n')
        driver = ['program check_surface', 'use fms_mod, only: fms_init,fms_end',
                  'use surface_flux_mod, only: surface_flux,surface_flux_init',
                  'use sat_vapor_pres_mod, only: sat_vapor_pres_init', 'implicit none',
                  'real :: result(1,20), qs(1)', 'call fms_init()',
                  'call sat_vapor_pres_init()', 'call surface_flux_init()']
        for values in inputs:
            args = ','.join(f'[{v:.17e}]' for v in values)
            outs = ','.join(f'result(:,{i})' for i in range(1, 21))
            driver += ['qs=0', f'call surface_flux({args},qs,[0.3],[0.0], &',
                       f' [1e-4],[1e-4],[1e-4],[1e-4],[0.0],{outs}, &',
                       ' 3600.0,[.false.],[.true.],[.true.])',
                       'write(*,"(a,9(es25.17,1x))") "CASE ",result(1,1:8),qs(1)']
        driver += ['call fms_end()', 'end program']
        (directory / 'driver.f90').write_text('\n'.join(driver) + '\n')
        binary = directory / 'check_surface'
        command = ['mpif90', *flags, 'surface_flux.F90', 'driver.f90',
                   str(fms_build / 'libFMS.a'), *libraries, '-o', str(binary)]
        compilation = subprocess.run(command, cwd=directory, capture_output=True, text=True, timeout=60)
        (directory / 'compile.txt').write_text(compilation.stdout + compilation.stderr)
        compilation.check_returncode()
        execution = subprocess.run([str(binary)], cwd=directory, capture_output=True, text=True, timeout=30)
        text = execution.stdout
        (directory / 'stdout.txt').write_text(text)
        (directory / 'stderr.txt').write_text(execution.stderr)
        execution.check_returncode()
        actual = [[float(v) for v in line[5:].split()] for line in text.splitlines() if line.startswith('CASE ')]
        if len(actual) != len(expected) or any(len(row) != 9 for row in actual):
            raise ValueError('full coupler harness did not return all component results')
        passed = all(math.isfinite(value) and abs(value-target) <= 1e-12+1e-7*abs(target)
                     for values, targets in zip(actual, expected, strict=True)
                     for value, target in zip(values, targets, strict=True))
        result['results'][name] = {'pass': passed, 'actual': actual, 'command': command,
                                   'linked_libraries': _linked_libraries(binary),
                                   'compiled_source_sha256': sha256_file(directory / 'surface_flux.F90'),
                                   'driver_sha256': sha256_file(directory / 'driver.f90'),
                                   'binary_sha256': sha256_file(binary)}
    if compare_zhenmode:
        import numpy as np

        from zhenmode.model.solver.numerics.backend import jax, jnp
        from zhenmode.model.solver.physics.air_sea import (
            AirState,
            open_water_fluxes,
            saturation_specific_humidity,
        )

        jax.config.update('jax_enable_x64', True)
        values = np.array(inputs).T
        fields = [jnp.asarray(values[i], dtype=jnp.float64) for i in (0, 1, 4, 2, 3)]
        zero = jnp.zeros(len(rows), dtype=jnp.float64)
        air = AirState(*fields, zero, zero, zero, zero, zero, zero)
        sst = jnp.asarray(values[7]-273.15, dtype=jnp.float64)
        flux = jax.jit(open_water_fluxes)(air, sst, jnp.full_like(sst, 0.3), zero)
        actual = np.array([-flux.sensible, flux.evaporation, -flux.longwave/0.98,
                           -flux.tau_x, -flux.tau_y, flux.cd, flux.ch, flux.ce,
                           saturation_specific_humidity(sst+273.15, air.pressure_pa)]).T
        passed = bool(np.all(np.isfinite(actual)) and
                      np.all(np.abs(actual-np.array(expected)) <= 1e-12+1e-7*np.abs(expected)))
        result['results']['zhenmode'] = {'pass': passed, 'actual': actual.tolist(),
                                        'precision': 'fp64', 'backend': jax.default_backend(),
                                        'coupled_ocean_integration': False}
    columns = [('sensible_up', 'W m-2'), ('evaporation_up', 'kg m-2 s-1'),
               ('longwave_up_unabsorbed', 'W m-2'), ('stress_u_out_of_ocean', 'Pa'),
               ('stress_v_out_of_ocean', 'Pa'), ('cd', '1'), ('ch', '1'), ('ce', '1'),
               ('saturation_specific_humidity', 'kg kg-1')]
    for entry in result['results'].values():
        entry['component_errors'] = {
            name: {'unit': unit, 'rmse': math.sqrt(sum((a[i]-e[i])**2 for a, e in
                                                      zip(entry['actual'], expected, strict=True))/len(expected)),
                   'maximum_absolute_error': max(abs(a[i]-e[i]) for a, e in
                                                 zip(entry['actual'], expected, strict=True)),
                   'samples': len(expected), 'sample_weighting': 'eight_fixture_cases_equal'}
            for i, (name, unit) in enumerate(columns)}
    result['status'] = ('PASS' if result['results']['adapted']['pass']
                        and not result['results']['upstream']['pass']
                        and (not compare_zhenmode or result['results']['zhenmode']['pass']) else 'FAIL')
    result['ocean_observation_score'] = None
    (output / 'report.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    for option in ('source', 'reference', 'fms-build', 'output'):
        parser.add_argument('--' + option, required=True)
    parser.add_argument('--compare-zhenmode', action='store_true',
                        help='also validate the JAX open-water component; never an ocean observation score')
    args = parser.parse_args()
    try:
        result = check(args.source, args.reference, args.fms_build, args.output, args.compare_zhenmode)
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        result = {'status': 'FAIL', 'scope': 'component_harness_failure',
                  'error': str(error), 'climate_qualification': False}
        failure = Path(args.output) / 'failure.json'
        if failure.parent.is_dir() and not failure.exists():
            failure.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'status': result['status'], 'scope': result['scope']}))
    raise SystemExit(0 if result['status'] == 'PASS' else 1)
