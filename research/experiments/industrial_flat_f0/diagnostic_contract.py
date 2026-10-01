"""The only registered time refinement; preserve every other coarse setting."""
from copy import deepcopy


def time_refinement(baseline, dt_s=50.):
    if dt_s not in (50., 100.) or baseline['dt'] != 100.:
        raise ValueError('only the coarse dt50 diagnostic is registered')
    result = deepcopy(baseline)
    result['dt'] = dt_s
    for name in ('DT', 'DT_THERM', 'DT_FORCING', 'DTBT'):
        result['mom_time_options'][name] = dt_s
    return result


def validate_time_contract(candidate, baseline):
    if candidate != time_refinement(baseline, candidate['dt']):
        raise ValueError('diagnostic changes more than the registered timestep')
