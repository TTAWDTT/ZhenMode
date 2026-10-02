"""Small input-support contracts, independent of real/private datasets."""

import importlib.util

import numpy as np

from tests.support.paths import REPOSITORY_ROOT


def example():
    raw = {
        'lon': np.array([0., 1., 2., 3.]),
        'lat': np.array([-1., 0., 1.]),
        'depth': np.array([0., 10.]),
        'data': np.full((2, 3, 4), 7.),
        'source_format': np.array('ocean.woa_twin.v1'),
        'variable': np.array('temperature'), 'units': np.array('degrees_celsius'),
        'missing_encoding': np.array('nan'),
        'longitude_units': np.array('degrees_east'),
        'latitude_units': np.array('degrees_north'),
        'depth_units': np.array('m'), 'depth_positive': np.array('down'),
    }
    grid = {
        'lon': raw['lon'].copy(), 'lat': raw['lat'].copy(),
        'z': -raw['depth'], 'depth': np.full((4, 3), 20.),
        'wet_mask_3d': np.ones((4, 3, 2)),
    }
    return raw, grid

def cli():
    path = REPOSITORY_ROOT / 'scripts/preflight_initial_inputs.py'
    spec = importlib.util.spec_from_file_location('preflight_initial_inputs', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
