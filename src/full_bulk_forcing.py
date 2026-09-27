"""Monthly full-bulk surface forcing helpers for Stage-G.

This module keeps the Stage-G forcing contract in one place.  The NCEP files
provide monthly atmospheric states; the solver turns them into fluxes with live
SST so the feedback remains active during the integration.

Sign convention for net surface heat is positive into the ocean.
"""
from __future__ import annotations

import numpy as np
import jax.numpy as jnp


RHO_AIR = 1.225
CP_AIR = 1005.0
DRAG_COEFFICIENT = 1.3e-3
LATENT_HEAT_VAPORIZATION = 2.501e6
STEFAN_BOLTZMANN = 5.670374419e-8
SURFACE_EMISSIVITY = 0.98
SURFACE_ALBEDO = 0.07
SURFACE_PRESSURE_PA = 101325.0


def saturation_specific_humidity_kg_kg(temperature_c,
                                       pressure_pa=SURFACE_PRESSURE_PA):
    t_c = np.asarray(temperature_c, dtype=np.float64)
    es = 611.2 * np.exp(17.67 * t_c / (t_c + 243.5))
    return 0.622 * es / (pressure_pa - 0.378 * es)


def saturation_specific_humidity_kg_kg_jax(temperature_c,
                                           pressure_pa=SURFACE_PRESSURE_PA):
    t_c = jnp.asarray(temperature_c)
    es = 611.2 * jnp.exp(17.67 * t_c / (t_c + 243.5))
    return 0.622 * es / (pressure_pa - 0.378 * es)


def build_full_bulk_fields(air_temperature_c,
                           specific_humidity_kg_kg,
                           downward_longwave_w_m2,
                           downward_shortwave_w_m2,
                           precipitation_rate_kg_m2_s,
                           wind_speed_m_s,
                           transfer_coefficient=DRAG_COEFFICIENT):
    arrays = {
        "air_temperature_c": air_temperature_c,
        "specific_humidity_air_kg_kg": specific_humidity_kg_kg,
        "downward_longwave_w_m2": downward_longwave_w_m2,
        "downward_shortwave_w_m2": downward_shortwave_w_m2,
        "precipitation_rate_kg_m2_s": precipitation_rate_kg_m2_s,
        "wind_speed_m_s": wind_speed_m_s,
    }
    shape = None
    for name, field in arrays.items():
        field = np.asarray(field, dtype=np.float64)
        if field.ndim != 3 or field.shape[0] != 12:
            raise ValueError(f"monthly field {name} must have shape (12, nx, ny)")
        if not np.all(np.isfinite(field)):
            raise ValueError(f"monthly field {name} contains non-finite values")
        arrays[name] = field
        if shape is None:
            shape = field.shape[1:]
        elif field.shape[1:] != shape:
            raise ValueError("monthly Stage-G fields are on different grids")
    if np.any(arrays["wind_speed_m_s"] < 0.0):
        raise ValueError("NCEP wind speed is negative")
    if np.any(arrays["precipitation_rate_kg_m2_s"] < 0.0):
        raise ValueError("NCEP precipitation rate is negative")
    if np.any(arrays["specific_humidity_air_kg_kg"] < 0.0):
        raise ValueError("NCEP specific humidity is negative")
    wind = arrays.pop("wind_speed_m_s")
    arrays["sensible_transfer_w_m2_k"] = RHO_AIR * CP_AIR * transfer_coefficient * wind
    arrays["latent_transfer_w_m2_kgkg"] = (RHO_AIR * LATENT_HEAT_VAPORIZATION
                                           * transfer_coefficient * wind)
    arrays["wind_speed_m_s"] = wind
    return arrays


def net_surface_heat_flux(sst_c, air_temperature_c,
                          specific_humidity_air_kg_kg,
                          downward_longwave_w_m2,
                          downward_shortwave_w_m2,
                          sensible_transfer_w_m2_k,
                          latent_transfer_w_m2_kgkg,
                          sea_ice_fraction=None):
    t_air = jnp.asarray(air_temperature_c)
    t_sst = jnp.asarray(sst_c)
    q_air = jnp.asarray(specific_humidity_air_kg_kg)
    q_sat = saturation_specific_humidity_kg_kg_jax(t_sst)
    shortwave = (1.0 - SURFACE_ALBEDO) * jnp.asarray(downward_shortwave_w_m2)
    outgoing_longwave = SURFACE_EMISSIVITY * STEFAN_BOLTZMANN * (t_sst + 273.15) ** 4
    sensible = jnp.asarray(sensible_transfer_w_m2_k) * (t_sst - t_air)
    latent = (jnp.asarray(latent_transfer_w_m2_kgkg)
              * (q_sat - q_air))
    net = shortwave + jnp.asarray(downward_longwave_w_m2) \
        - outgoing_longwave - sensible - latent
    if sea_ice_fraction is not None:
        insulation = 1.0 - jnp.clip(jnp.asarray(sea_ice_fraction), 0.0, 1.0)
        net = net * insulation
    return net