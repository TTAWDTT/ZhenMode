"""Small configuration/IO witnesses; no production timestep is executed."""
import sys
from pathlib import Path

import netCDF4
import numpy as np
import pytest

DIRECTORY = Path(__file__).parents[1] / 'research/experiments/industrial_flat_f0'
sys.path.insert(0, str(DIRECTORY))
from native_mom import active_expected, read_field
from prepare_mom import prepare, settings


def test_positive_independent_preflight_and_coarse_duration():
    assert settings(True)['DAYMAX'] * 86400. == 100.
    assert settings(False)['DAYMAX'] * 86400. == 32000.
    assert settings(True)['DT'] == settings(False)['DT'] == 100.
    assert settings(True)['NIHALO'] == settings(True)['NJHALO'] == 4


def test_preflight_native_sampling_and_input_hashes(tmp_path):
    import hashlib
    import json
    directory = tmp_path / 'mom'
    prepare(directory, True)
    request = json.loads((directory / 'requested_config.json').read_text())
    for name, identity in request['input_identities'].items():
        assert hashlib.sha256((directory / name).read_bytes()).hexdigest() == identity['sha256']
    assert '"native",100,"seconds"' in (directory / 'diag_table').read_text()
    assert all(line.endswith(',1') for line in
               (directory / 'diag_table').read_text().splitlines()[3:])
    with netCDF4.Dataset(directory / 'INPUT/initial.nc') as ds:
        assert ds['x'].cartesian_axis == 'X'
        assert ds['Interface'].cartesian_axis == 'Z'
        expected = .01 * np.cos(2 * np.pi * (np.arange(64) + .5) / 64) * np.sinc(1 / 64)
        np.testing.assert_allclose(ds['eta'][0, 0], expected, rtol=0., atol=1.e-16)
    with pytest.raises(FileExistsError):
        prepare(directory, True)


def test_native_masked_coordinate_rejected(tmp_path):
    path = tmp_path / 'masked.nc'
    with netCDF4.Dataset(path, 'w') as ds:
        ds.createDimension('x', 2)
        ds.createVariable('x', 'f8', ('x',), fill_value=-999.)[:] = [1., -999.]
    with netCDF4.Dataset(path) as ds, pytest.raises(ValueError, match='invalid native field'):
        read_field(ds, 'x', ('x',))


def test_dormant_fields_require_explicit_runtime_disable_controls():
    requested = settings(True)
    resolved = dict(requested, LAPLACIAN=False)
    for name in ('ISOTROPIC', 'KH', 'KH_VEL_SCALE', 'SMAGORINSKY_KH', 'SMAG_BI_CONST', 'KD'):
        resolved.pop(name)
    expected, evidence = active_expected(requested, resolved)
    assert len(evidence) == 6
    assert expected['LAPLACIAN'] is False
    for control in ('GRID_CONFIG', 'LAPLACIAN', 'SMAGORINSKY_AH', 'ADIABATIC'):
        changed = dict(resolved)
        changed.pop(control)
        with pytest.raises(ValueError, match='unproven dormant'):
            active_expected(requested, changed)
