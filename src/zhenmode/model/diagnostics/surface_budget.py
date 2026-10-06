"""Endpoint inventory versus separately supplied surface flux integrals.

No residual is reassigned to a source. This measures heat and Boussinesq virtual
salt in the open-water component scope; it does not certify full water/ice closure.
"""
import numpy as np

from zhenmode.model.config import C_P, RHO_0


def surface_budget(before, after, *, wet_volume, wet_area, heat_flux, salt_flux, dt_seconds):
    volume, area = np.asarray(wet_volume), np.asarray(wet_area)
    heat, salt = np.asarray(heat_flux), np.asarray(salt_flux)
    if (volume.shape != np.shape(before.T) or area.shape != volume.shape[:2] or
            heat.shape != area.shape or salt.shape != area.shape or
            np.shape(after.T) != volume.shape or np.shape(before.S) != volume.shape or
            np.shape(after.S) != volume.shape or not np.isfinite(dt_seconds) or dt_seconds <= 0 or
            any(not np.isfinite(a).all() for a in (volume, area, heat, salt, before.T, after.T, before.S, after.S)) or
            np.any(volume < 0) or np.any(area < 0)):
        raise ValueError('invalid surface budget geometry, fields or SI duration')
    observed = np.array([
        RHO_0 * C_P * np.sum(volume * (np.asarray(after.T) - np.asarray(before.T))),
        RHO_0 / 1000 * np.sum(volume * (np.asarray(after.S) - np.asarray(before.S))),
    ])
    expected = dt_seconds * np.array([np.sum(area * heat), RHO_0 / 1000 * np.sum(area * salt)])
    return dict(metric_names=['fixed_volume_water_heat_J', 'virtual_salt_kg'],
                observed_change=observed.tolist(), independent_surface_input=expected.tolist(),
                residual=(observed - expected).tolist(),
                scope='open_water_surface_component', climate_qualification=False)
