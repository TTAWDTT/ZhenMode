"""Fail-closed checks of explicit resolved settings and native MOM geometry.

The caller must supply the frozen full configuration contract. These checks
do not establish complete physics equivalence or industrial qualification.
"""
import hashlib
from pathlib import Path
import re

import netCDF4
import numpy as np


BASE_MOM_CONTRACT = dict(
    GRID_CONFIG='cartesian', AXIS_UNITS='m', REENTRANT_X=True, REENTRANT_Y=False,
    NIGLOBAL=64, NJGLOBAL=8, NK=4, LENLON=1002269.4248554128, LENLAT=100000.,
    ROTATION='betaplane', F_0=0., BETA=0., TOPO_CONFIG='flat', MAXIMUM_DEPTH=100.,
    G_EARTH=9.81, RHO_0=1025., EQN_OF_STATE='LINEAR',
    RHO_REF_LINEAR_EOS=1025., T_REF_LINEAR_EOS=15., S_REF_LINEAR_EOS=35.,
    DRHO_DT=-.205, DRHO_DS=.779, DRHO_DP=0.,
    DO_DYNAMICS=True, ENABLE_THERMODYNAMICS=True, USE_EOS=True,
    OFFLINE_TRACER_MODE=False, WIND_CONFIG='zero', BUOY_CONFIG='zero')


def parse_resolved(text):
    settings = {}
    for line in text.splitlines():
        match = re.match(r'^([A-Z][A-Z0-9_]*)\s*=\s*(.*?)\s*(?:!.*)?$', line)
        if match is None:
            continue
        name, raw = match.groups()
        if raw.startswith('"') and raw.endswith('"'):
            value = raw[1:-1]
        elif raw.lower() in ('true', '.true.', 'false', '.false.'):
            value = raw.lower() in ('true', '.true.')
        else:
            try:
                value = float(raw.replace('D', 'E').replace('d', 'e'))
            except ValueError:
                value = raw
        if name in settings and settings[name] != value:
            raise ValueError('contradictory resolved parameter: ' + name)
        settings[name] = value
    return settings


def verify_resolved_settings(text, full_expected):
    settings = parse_resolved(text)
    expected = dict(BASE_MOM_CONTRACT)
    for name, value in full_expected.items():
        if name in expected and value != expected[name]:
            raise ValueError('caller changes frozen base parameter: ' + name)
        expected[name] = value
    failures = []
    for name, value in expected.items():
        actual = settings.get(name)
        if isinstance(value, bool):
            equal = type(actual) is bool and actual == value
        elif isinstance(value, (int, float)):
            equal = (type(actual) is float and np.isfinite(actual)
                     and np.isclose(actual, value, rtol=1.e-12, atol=0.))
        else:
            equal = actual == value
        if not equal:
            failures.append(name)
    if failures:
        raise ValueError('missing or mismatched resolved settings: ' + ', '.join(failures))
    if type(settings.get('USE_BT_CONT_TYPE')) is not bool:
        raise ValueError('missing or invalid resolved USE_BT_CONT_TYPE')
    if (settings['USE_BT_CONT_TYPE'] is False
            and type(settings.get('NONLINEAR_BT_CONTINUITY')) is not bool):
        raise ValueError('missing active NONLINEAR_BT_CONTINUITY')
    # The historical nonlinear flag is dormant when USE_BT_CONT_TYPE=True.
    free_surface = {name: settings.get(name) for name in (
        'SPLIT', 'SPLIT_RK2B', 'USE_BT_CONT_TYPE', 'NONLINEAR_BT_CONTINUITY',
        'BT_THICK_SCHEME', 'BT_PROJECT_VELOCITY', 'BEBT', 'DTBT')}
    return dict(checked_settings=expected, free_surface_resolved=free_surface,
                nonlinear_flag_applies=settings.get('USE_BT_CONT_TYPE') is False,
                qualification_passed=False)


def verify_mom_geometry(path):
    path = Path(path)
    expected_area = 1002269.4248554128 / 64. * 100000. / 8.
    with netCDF4.Dataset(path) as ds:
        for name in ('lonh', 'lath'):
            if ds[name].units not in ('m', 'meter', 'meters'):
                raise ValueError('Cartesian coordinate lacks metre identity: ' + name)
        for name, expected in [('D', 100.), ('wet', 1.), ('f', 0.),
                               ('Ah', expected_area), ('dxT', 1002269.4248554128 / 64.),
                               ('dyT', 100000. / 8.)]:
            values = np.ma.asarray(ds[name][:])
            if (values.shape != (8, 64) or np.any(np.ma.getmaskarray(values))
                    or not np.all(np.isfinite(values))
                    or not np.allclose(values, expected, rtol=1.e-12, atol=0.)):
                raise ValueError('native Cartesian geometry mismatch: ' + name)
    return dict(filename=path.name, sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                geometry_contract_passed=True, qualification_passed=False)
