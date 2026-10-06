"""Independent pinned GSW/MOM6 thermodynamic components, not ocean integration.

Run in Linux under the external one-CPU/180s/4GiB supervisor. Original Fortran
files are copied byte-for-byte; no production implementation generates their
expected values. Keep all builds, values and receipts in a new local output.
"""
import argparse
import itertools
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from zhenmode.model.solver.numerics.backend import jax, jnp
from zhenmode.model.solver.physics import teos10
from zhenmode.provenance.sources import (
    production_source_modules,
    sha256_file,
    source_paths,
    source_root,
)

GSW_COMMIT = '29e64d652786e1d076a05128c920f394202bfe10'
SOURCES = {
    'modules/gsw_mod_kinds.f90': 'd4635f6d64daab6d264a489083660180ba178b589d1e4d9b227ccc76ca3a1722',
    'modules/gsw_mod_teos10_constants.f90': '4d4347a84d522d4e98cb2409a060809f9a0ed6206af0cf356ba1ad1a40484ddd',
    'modules/gsw_mod_specvol_coefficients.f90': '78ac3da86a288da5406e8a914ef278facae2c1c1d980b3fe8a31404e9ce99601',
    'modules/gsw_mod_toolbox.f90': '8169ecc9d9e3cc66160bf35eb856f2e69122253c0d741766ea1d61a2bed6b95d',
    'toolbox/gsw_specvol.f90': '5053329ece5ccd120785da2a7b136d777aeb9e1c6c5127b6d35c7835ff00155b',
    'toolbox/gsw_rho.f90': '1faab55267f0ca3bf7f67a26ebc558c454b2d0124a1c25f427f9703db6d113c2',
    'toolbox/gsw_rho_first_derivatives.f90': 'ee984aecbd33cf18782b7d887d03d3f87107375174f47cfd14ef910b5fccb6c3',
    'toolbox/gsw_sr_from_sp.f90': '04c154a390d30c60fd4322292f275e4fa6cefa7d8f2dd8615b28128d7907b97a',
}
TEMPERATURE_SOURCES = {
    'modules/gsw_mod_freezing_poly_coefficients.f90': '401876e7c4767c3f7be99484f13af27c3b1fe2ef230d38accb2a6992923e6345',
    'toolbox/gsw_ct_freezing_poly.f90': '9e717ac7a3b8646713454a9bb8e83ed8796d1c4735512abd0ce2f20ea5b03ecb',
    'toolbox/gsw_ct_from_pt.f90': 'f9b2b30cd520d105ebf391fcb0bdcfe25846b520838a1b77987034992c3a671a',
    'toolbox/gsw_pt_from_ct.f90': 'abe385c70f9f020bc220383b6d232e0ba3544fc2ef1fb6b12ce93061e26903bf',
    'toolbox/gsw_ct_from_t.f90': 'd263bdaa6b9814b6e5e0ccfac6e115dd9c4b64baa09b46d65261ec7574b16cf9',
    'toolbox/gsw_pt0_from_t.f90': '3857d997501afc48995ddd8ab2c078ab9695a87096578dbff93d530d3139b9db',
    'toolbox/gsw_entropy_part.f90': '9cf50e1260ddc7ec498476a88df45a3a9712f78b6ff7253edcc785b9da66ce6c',
    'toolbox/gsw_entropy_part_zerop.f90': 'a65a025721e751f290d4c139388cdeab8f2fc66304b254a00397aa8e5622986d',
    'toolbox/gsw_gibbs_pt0_pt0.f90': 'b190a4472d158bca7db92feb29471e94b0954c80ce647f1bf929522cd66e8a01',
    'toolbox/gsw_pt_from_t.f90': '24cc5fd8fbaf53cb9a60dc3997019978efe95fb659b16d49b34dd427b487788c',
    'toolbox/gsw_gibbs.f90': 'fdcfb399b15520e68c39fdec66f6c8041b881ff77783576cfdf6283106895543',
    'toolbox/gsw_t_from_ct.f90': '5b61a477573db2a10c758b58a79fe8dea0dcdea8cfb8254e57de1fd588125df0',
    'toolbox/gsw_ct_first_derivatives.f90': 'bdfac630ce3178e5335f9bd9c92bf16058c32cc50e0ed58d07bf8d07fa85f40c',
    'toolbox/gsw_rho_t_exact.f90': '9c336d82c01aa43aa603f3f19d6ad719ea8c80de3b8c30d758640df390bde466',
}
MOM_SHA = '73b1c73718d45c33bce819231f4a35543b3412c26e3aa4ef2863df7beadcfe19'
# Official gsw_rho example: a second, published check independent of our driver.
OFFICIAL = [(34.7118, 28.8099, 10, 1021.839935738108),
            (34.8915, 28.4392, 50, 1022.262457966867),
            (35.0256, 22.7862, 125, 1024.427195413316),
            (34.8472, 10.2262, 250, 1027.790152759127),
            (34.7366, 6.8272, 600, 1029.837779000189),
            (34.7324, 4.3236, 1000, 1032.002453224572)]
FIXTURE = [(0, 0, 0), (0, 40, 10), (5, -2, 0), (20, 10, 500),
           (30, 5, 1000), (35.16504, -2, 0), (35.16504, 0, 4000),
           (35.16504, 10, 500), (35.16504, 11, 500), (35.16504, 10, 0),
           (35.16504, 11, 1000), (42, 40, 8000)]


def driver(mom, *, temperatures=False):
    start = mom.index('real elemental function density_elem_TEOS10(')
    stop = mom.index('end function density_elem_TEOS10', start)
    routine = mom[start:stop + len('end function density_elem_TEOS10')]
    parameters = [line for line in mom.splitlines() if line.startswith('real, parameter :: Pa2db')]
    if len(parameters) != 1:
        raise ValueError('actual MOM6 pressure-conversion declaration is ambiguous')
    # Execute the actual MOM function and its Pa conversion. The empty type
    # only supplies its unused `this` argument; no complete MOM EOS is claimed.
    result = '''! MOM6 function below is from the Modular Ocean Model version 6.
! SPDX-License-Identifier: Apache-2.0
! The unused receiver type and driver are only a component harness.
program reference
use gsw_mod_toolbox, only: gsw_rho, gsw_rho_first_derivatives, gsw_sr_from_sp
implicit none
type TEOS10_EOS
end type
type(TEOS10_EOS) :: eos
MOM_PRESSURE_PARAMETER
real :: s,t,p,rs,rt,rp
integer :: ios
do
 read(*,*,iostat=ios) s,t,p
 if(ios /= 0) exit
 call gsw_rho_first_derivatives(s,t,p,rs,rt,rp)
 write(*,'(6(es26.18,1x))') gsw_rho(s,t,p),rs,rt,rp,gsw_sr_from_sp(s), &
   density_elem_TEOS10(eos,t,s,p*1.e4)
enddo
contains
'''.replace('MOM_PRESSURE_PARAMETER', parameters[0]) + routine + '\nend program\n'
    if temperatures:
        result = result.replace('implicit none', '''use gsw_mod_toolbox, only: gsw_ct_from_pt,gsw_pt_from_ct,gsw_ct_from_t, &
 gsw_pt0_from_t,gsw_t_from_ct,gsw_ct_first_derivatives,gsw_rho_t_exact,gsw_ct_freezing_poly
implicit none''', 1)
        result = result.replace('real :: s,t,p,rs,rt,rp', 'real :: s,t,p,rs,rt,rp,cts,ctt', 1)
        result = result.replace(" write(*,'(6(es26.18,1x))')", " call gsw_ct_first_derivatives(s,t,cts,ctt)\n write(*,'(15(es26.18,1x))')", 1)
        result = result.replace('density_elem_TEOS10(eos,t,s,p*1.e4)', '''density_elem_TEOS10(eos,t,s,p*1.e4), &
   gsw_ct_from_pt(s,t),gsw_pt_from_ct(s,t),gsw_ct_from_t(s,t,p),gsw_pt0_from_t(s,t,p), &
   gsw_t_from_ct(s,t,p),cts,ctt,gsw_rho_t_exact(s,gsw_t_from_ct(s,t,p),p), &
   gsw_ct_freezing_poly(s,0.,0.)''', 1)
    return result


def compare_temperatures(cases, reference):
    """Fixed predeclared thresholds; compare originals, never only round trips."""
    results = {}
    for precision, atol in [('float64', 1e-10), ('float32', 1e-4)]:
        sa, temperature, pressure = jnp.asarray(cases, dtype=getattr(jnp, precision)).T
        outputs = [jax.jit(function)(sa, temperature, *args) for function, args in (
            (teos10.conservative_from_potential, ()), (teos10.potential_from_conservative, ()),
            (teos10.conservative_from_in_situ, (pressure,)), (teos10.potential_from_in_situ, (pressure,)),
            (teos10.in_situ_from_conservative, (pressure,)))]
        derivatives = jax.jit(jax.grad(lambda s,t: jnp.sum(teos10.conservative_from_potential(s,t)),
                                      argnums=(0,1)))(sa, temperature)
        freezing=jax.jit(teos10.surface_freezing_ct)(sa)
        actual = np.asarray(jnp.stack((*outputs, *derivatives,freezing), axis=1))
        expected = np.column_stack((reference[:, 6:13],reference[:,14]))
        np.testing.assert_allclose(actual, expected, rtol=0, atol=atol)
        damaged = expected.copy()
        damaged[0,0] += .01
        try:
            np.testing.assert_allclose(actual, damaged, rtol=0, atol=atol)
        except AssertionError:
            pass
        else:
            raise AssertionError('temperature comparator accepted planted mismatch')
        results[precision] = {'maximum_absolute_errors': dict(zip(
            ('CT_from_pt','pt_from_CT','CT_from_t','pt_from_t','t_from_CT','CT_SA','CT_pt','surface_freezing_CT'),
            np.max(np.abs(actual-expected),axis=0).tolist(),strict=True)),
            'temperature_and_derivative_atol':atol,'planted_mismatch_rejected':True,
            'fresh_water_derivatives_finite':bool(np.isfinite(actual[sa == 0]).all())}
    return results


def check(source, mom_source, output, *, temperatures=False):
    source, mom_source, output = Path(source), Path(mom_source), Path(output)
    output.mkdir(parents=True, exist_ok=False)
    try:
        sources = SOURCES | TEMPERATURE_SOURCES if temperatures else SOURCES
        for name, expected in sources.items():
            original = source/name
            if sha256_file(original) != expected:
                raise ValueError('GSW source identity differs: '+name)
            copied = output/original.name
            copied.write_bytes(original.read_bytes())
            if sha256_file(copied) != expected or sha256_file(original) != expected:
                raise ValueError('actual compiled source identity differs: '+name)
        if sha256_file(mom_source) != MOM_SHA:
            raise ValueError('MOM6 TEOS10 source identity differs')
        (output/'MOM_EOS_TEOS10.F90').write_bytes(mom_source.read_bytes())
        if sha256_file(output/'MOM_EOS_TEOS10.F90') != MOM_SHA:
            raise ValueError('actual MOM density source changed during copy')
        (output/'driver.f90').write_text(driver((output/'MOM_EOS_TEOS10.F90').read_text(),
                                               temperatures=temperatures))
        command = ['gfortran', '-O0', '-ffree-line-length-none', '-fdefault-real-8',
                   '-fdefault-double-8', *(Path(name).name for name in sources),
                   'driver.f90', '-o', 'reference']
        build = subprocess.run(command, cwd=output, capture_output=True, text=True, timeout=60)
        (output/'compile.log').write_text(build.stdout+build.stderr)
        build.check_returncode()
        cases = FIXTURE + [row[:3] for row in OFFICIAL] + list(itertools.product(
            (0, 5, 20, 35, 42), (-2, 0, 5, 20, 35, 40), (0, 10, 500, 1000, 4000, 6000, 8000)))
        inputs = '\n'.join(' '.join(f'{float(v):.17e}' for v in row) for row in cases)+'\n'
        (output/'inputs.txt').write_text(inputs)
        run = subprocess.run([str((output/'reference').resolve())], input=inputs, cwd=output,
                             capture_output=True, text=True, timeout=30)
        (output/'reference.txt').write_text(run.stdout)
        (output/'reference.stderr').write_text(run.stderr)
        run.check_returncode()
        reference = np.loadtxt(output/'reference.txt')
        if reference.shape != (len(cases), 15 if temperatures else 6) or not np.isfinite(reference).all():
            raise ValueError('incomplete or nonfinite independent reference')
        # Convert the reference pressure derivative from per Pa to per dbar.
        reference[:, 3] *= 10000
        np.testing.assert_allclose(reference[12:18, 0], [row[3] for row in OFFICIAL],
                                   rtol=0, atol=2e-10)
        results = {}
        for precision, limits in [('float64', (2e-10, 2e-10)), ('float32', (5e-4, 1e-5))]:
            s, t, p = jnp.asarray(cases, dtype=getattr(jnp, precision)).T
            teos10.validate_state(s, t, p)
            rho = jax.jit(teos10.density)(s, t, p)
            derivatives = jax.jit(teos10.density_derivatives)(s, t, p)
            actual = np.asarray(jnp.stack((rho, *derivatives, teos10.reference_salinity(s)), axis=1))
            errors = np.max(np.abs(actual-reference[:, :5]), axis=0)
            np.testing.assert_allclose(actual[:, 0], reference[:, 0], rtol=0, atol=limits[0])
            np.testing.assert_allclose(actual[:, 1:4], reference[:, 1:4], rtol=0, atol=limits[1])
            np.testing.assert_allclose(actual[:, 4], reference[:, 4], rtol=0, atol=1e-5 if precision=='float32' else 2e-13)
            np.testing.assert_allclose(actual[:, 0], reference[:, 5], rtol=0, atol=limits[0])
            results[precision] = {'max_absolute_errors': dict(zip(
                ('rho_kg_m3','drho_dSA','drho_dCT','drho_dp_per_dbar','SR_g_kg'), errors.tolist(), strict=True)),
                'density_rmse_kg_m3': float(np.sqrt(np.mean((actual[:, 0]-reference[:, 0])**2))),
                'density_atol_kg_m3': limits[0], 'derivative_atol': limits[1]}
        # Independently show why in-situ pressure differences cannot determine convection.
        common_contrast = reference[7, 0] - reference[8, 0]
        different_contrast = reference[9, 0] - reference[10, 0]
        if not common_contrast > 0 > different_contrast:
            raise AssertionError('pressure negative control did not discriminate instability')
        report = {'status': 'PASS', 'scope': 'thermodynamic_polynomial_components_not_ocean',
                  'created_utc':datetime.now(timezone.utc).isoformat(),
                  'data_kind': 'manufactured_thermodynamic_states', 'cases':len(cases),
                  'gsw_commit':GSW_COMMIT,'source_sha256':sources,'mom_source_sha256':MOM_SHA,
                  'MOM_scope':'actual_density_function_with_empty_unused_this_type_not_full_module',
                  'compiler':subprocess.check_output(['gfortran','--version'],text=True),
                  'command':command,'driver_sha256':sha256_file(output/'driver.f90'),
                  'binary_sha256':sha256_file(output/'reference'),
                  'harness_sha256':sha256_file(__file__),
                  'implementation_sha256':sha256_file(teos10.__file__),
                  'backend_sha256':sha256_file(__import__('zhenmode.model.solver.numerics.backend',fromlist=['']).__file__),
                  'package_source_sha256':{name:sha256_file(path) for name,path in
                      source_paths(source_root(teos10.__file__),production_source_modules()).items()},
                  'environment':{'jax':jax.__version__,'numpy':np.__version__,
                                 'devices':[str(device) for device in jax.devices()]},
                  'reference_values_sha256':sha256_file(output/'reference.txt'),
                  'precision_results':results,
                  'pressure_control':{'common_pressure_contrast_kg_m3':common_contrast,
                                      'different_pressure_contrast_kg_m3':different_contrast},
                  'funnel_accuracy_assessed':False,'state_conversions_verified':False,
                  'production_enabled':False,'ocean_observation_score':None,
                  'execution_ready':False,'climate_qualification':False}
        if temperatures:
            report['temperature_results'] = compare_temperatures(cases,reference)
            report['temperature_conversion_scope'] = 'numeric_components_not_native_state_preparation'
            # Informative Gibbs comparison only. The broad rectangle is not a
            # funnel claim; do not silently certify its outside-funnel states.
            report['polynomial_vs_Gibbs'] = {
                'all_states_maximum_absolute_density_difference_kg_m3':float(np.max(np.abs(reference[:,0]-reference[:,13]))),
                'official_six_states_density_differences_kg_m3':(reference[12:18,0]-reference[12:18,13]).tolist(),
                'scope':'sample_differences_not_full_funnel_verification'}
            temperature_fixture = {'gsw_commit':GSW_COMMIT,'source_sha256':sources,
                'driver_sha256':report['driver_sha256'],'compiler':report['compiler'],'command':command,
                'columns':['SA_g_kg','temperature_number_degC','p_dbar','CT_from_pt','pt_from_CT',
                           'CT_from_t','pt_from_t','t_from_CT','CT_SA','CT_pt','surface_freezing_CT'],
                'rows':[list(inputs)+values[6:13].tolist()+[float(values[14])]
                        for inputs,values in zip(cases[:18],reference[:18],strict=True)],
                'scope':'each_temperature_number_interpreted_separately_per_named_conversion',
                'accuracy_scope':'agreement_with_fixed_original_software_not_physical_truth_certificate'}
            (output/'temperature-fixture.json').write_text(json.dumps(temperature_fixture,indent=2,allow_nan=False)+'\n')
        fixture = {'gsw_commit':GSW_COMMIT, 'source_sha256':SOURCES, 'mom_source_sha256':MOM_SHA,
                   'driver_sha256':report['driver_sha256'], 'compiler':report['compiler'],
                   'command':command,'columns':['SA_g_kg','CT_degC','p_dbar','rho','rho_SA','rho_CT','rho_p_dbar','SR'],
                   'rows':[list(inputs)+values[:5].tolist()
                           for inputs, values in zip(cases[:18],reference[:18],strict=True)],
                   'scope':report['scope'], 'funnel_accuracy_assessed':False}
        (output/'fixture.json').write_text(json.dumps(fixture,indent=2,allow_nan=False)+'\n')
        (output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
        return report
    except (Exception, KeyboardInterrupt) as error:
        (output/'failure.json').write_text(json.dumps({'status':'FAIL','reason_type':type(error).__name__,
                                                      'reason':str(error)},indent=2)+'\n')
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',required=True,help='fixed GSW-Fortran checkout root')
    parser.add_argument('--mom-source',required=True,help='fixed MOM_EOS_TEOS10.F90')
    parser.add_argument('--output',required=True)
    parser.add_argument('--temperatures',action='store_true',help='also check original GSW temperature conversions and derivatives')
    args = parser.parse_args()
    result = check(args.source,args.mom_source,args.output,temperatures=args.temperatures)
    print(json.dumps({'status':result['status'],'cases':result['cases'],
                      'precision_results':result['precision_results']},indent=2))
