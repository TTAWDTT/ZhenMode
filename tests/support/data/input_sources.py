"""Adversarial format/identity and deterministic read-change witnesses."""





import numpy as np
from netCDF4 import Dataset

from tests.support.data.input_quality import example


def write_inputs(folder):
    raw, grid = example()
    np.savez(folder / 'temp.npz', **raw)
    np.savez(folder / 'salt.npz', **{**raw, 'variable': np.array('salinity'),
                                  'units': np.array('1')})
    np.savez(folder / 'grid.npz', **grid)
    np.savez(folder / 'initial.npz', T=raw['data'].transpose(2, 1, 0),
             S=raw['data'].transpose(2, 1, 0))
    return ['--temperature', str(folder / 'temp.npz'), '--salinity',
            str(folder / 'salt.npz'), '--grid', str(folder / 'grid.npz'),
            '--initial-state', str(folder / 'initial.npz'),
            '--output', str(folder / 'quality.json'), '--strict']

def netcdf_source(path, *, units='degrees_celsius', fill=True, sentinel=False):
    with Dataset(path, 'w') as dataset:
        for name, values in [('time', [0]), ('depth', [0., 10.]),
                             ('lat', [-1., 0., 1.]), ('lon', [0., 1., 2., 3.])]:
            dataset.createDimension(name, len(values))
            coordinate = dataset.createVariable(name, 'f8', (name,))
            coordinate[:] = values
            if name != 'time':
                coordinate.units = {'depth': 'm', 'lat': 'degrees_north',
                                    'lon': 'degrees_east'}[name]
            if name == 'depth':
                coordinate.positive = 'down'
        variable = dataset.createVariable('t_an', 'f4', ('time', 'depth', 'lat', 'lon'),
                                          fill_value=np.float32(9.96921e36) if fill else False)
        variable.units = units
        variable[:] = np.full(variable.shape, 9.96921e36 if sentinel else 7., dtype=np.float32)
