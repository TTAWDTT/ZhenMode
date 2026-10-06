"""Open-water LY2009 exchange with Gill moist-air properties, in SI units.

Equations: Large & Yeager (2009), Appendix, and JRA55-do user manual
Appendix B (Gill branch). No IO or interpolation occurs in this module.
The coefficient iteration follows LY2004 Eqs 7-10 and LY2009 Eq 11,
with five iterations maximum and a 1e-4 relative drag convergence criterion.
This is an explicitly selected component, not a full ocean/ice qualification.
"""

from __future__ import annotations

from typing import NamedTuple

from zhenmode.model.config import C_P, RHO_0
from zhenmode.model.solver.numerics.backend import jax, jnp

EPSILON = 18.016 / 28.966
VIRTUAL_FACTOR = 1 / EPSILON - 1
GAS_CONSTANT = 287.04


class AirState(NamedTuple):
    """10m weather; K, kg/kg, Pa, m/s, W/m2 and kg/m2/s, respectively."""

    temperature_k: object
    specific_humidity: object
    pressure_pa: object
    wind_u: object
    wind_v: object
    shortwave_down: object
    longwave_down: object
    rain: object
    snow: object
    runoff: object
    calving: object


class SurfaceFluxes(NamedTuple):
    """Heat/stress positive into water; evaporation positive out of water."""

    tau_x: object
    tau_y: object
    sensible: object
    latent: object
    evaporation: object
    shortwave: object
    longwave: object
    cd: object
    ch: object
    ce: object


def saturation_specific_humidity(temperature_k, pressure_pa):
    """Gill seawater saturation: pressure correction and Raoult factor on vapor pressure."""
    celsius = temperature_k - 273.15
    hpa = pressure_pa / 100
    vapor_hpa = 0.98 * 10 ** ((0.7859 + 0.03477 * celsius) / (1 + 0.00412 * celsius))
    vapor_hpa *= 1 + 1e-6 * hpa * (4.5 + 0.0006 * celsius**2)
    return EPSILON * vapor_hpa / (hpa - (1 - EPSILON) * vapor_hpa)


def _neutral_drag(speed):
    return 1e-3 * jnp.where(speed >= 33, 2.34,
                           2.7 / speed + 0.142 + 0.0764 * speed - 3.14807e-10 * speed**6)


def _stability_profiles(zeta):
    # Both branches are real-valued, including when JAX evaluates an inactive branch.
    root2 = jnp.sqrt(jnp.maximum(1 - 16 * zeta, 1))
    root = jnp.sqrt(root2)
    momentum = (2 * jnp.log((1 + root) / 2) + jnp.log((1 + root2) / 2)
                - 2 * jnp.arctan(root) + jnp.pi / 2)
    heat = 2 * jnp.log((1 + root2) / 2)
    return jnp.where(zeta > 0, -5 * zeta, momentum), jnp.where(zeta > 0, -5 * zeta, heat)


def ncar_coefficients10m(theta_air_k, humidity_air, temperature_surface_k,
                         humidity_surface, scalar_wind):
    """Transfer coefficients for equal 10m heights; theta is referenced to the surface.

    Scalar wind floor 0.5m/s; neutral estimate floor 0.3m/s; z/L clipped to
    [-10,10]. These explicit regularizations are part of this component identity.
    No temperature/humidity floor or missing-weather substitution is performed.
    """
    speed = jnp.maximum(scalar_wind, 0.5)
    drag = _neutral_drag(speed)
    root = jnp.sqrt(drag)
    heat = 1e-3 * root * jnp.where(theta_air_k >= temperature_surface_k, 18, 32.7)
    moisture = 0.0346 * root
    temperature_jump = theta_air_k - temperature_surface_k
    humidity_jump = humidity_air - humidity_surface
    virtual_temperature = theta_air_k * (1 + VIRTUAL_FACTOR * humidity_air)

    def iteration(_, carry):
        cd, ch, ce, neutral_root, active = carry
        friction = jnp.sqrt(cd) * speed
        temperature_scale = ch / jnp.sqrt(cd) * temperature_jump
        humidity_scale = ce / jnp.sqrt(cd) * humidity_jump
        buoyancy = 9.81 * (temperature_scale / virtual_temperature
                          + humidity_scale / (humidity_air + 1 / VIRTUAL_FACTOR))
        zeta = jnp.clip(0.4 * 10 * buoyancy / friction**2, -10, 10)
        psi_m, psi_h = _stability_profiles(zeta)
        neutral_wind = jnp.maximum(speed / (1 - neutral_root * psi_m / 0.4), 0.3)
        neutral_cd = _neutral_drag(neutral_wind)
        next_root = jnp.sqrt(neutral_cd)
        next_cd = neutral_cd / (1 - next_root * psi_m / 0.4)**2
        neutral_ch = 1e-3 * next_root * jnp.where(zeta >= 0, 18, 32.7)
        neutral_ce = 0.0346 * next_root
        scale = jnp.sqrt(next_cd / neutral_cd)
        next_ch = neutral_ch * scale / (1 - neutral_ch * psi_h / (0.4 * next_root))
        next_ce = neutral_ce * scale / (1 - neutral_ce * psi_h / (0.4 * next_root))
        keep_iterating = jnp.abs(next_cd - cd) / (next_cd + 1e-8) >= 1e-4
        return (jnp.where(active, next_cd, cd), jnp.where(active, next_ch, ch),
                jnp.where(active, next_ce, ce), jnp.where(active, next_root, neutral_root),
                active & keep_iterating)

    drag, heat, moisture, _, _ = jax.lax.fori_loop(
        0, 5, iteration, (drag, heat, moisture, root, jnp.ones_like(drag, dtype=bool)))
    return drag, heat, moisture


def open_water_fluxes(air, sst_c, ocean_u, ocean_v, *, albedo=0.066, emissivity=0.98):
    """Use current bulk SST/current, without direct SST restoring or ice substitution.

    The caller must validate finite SI input. Air potential temperature at 10m
    is Ta+g*z/Cp (surface reference); density uses absolute Ta, not theta.
    Snow/calving energetics and freshwater are applied by separate operators.
    """
    humidity = air.specific_humidity
    cp_air = 1004.6 * (1 + 0.8735 * humidity)
    theta = air.temperature_k + 9.81 * 10 / cp_air
    density = air.pressure_pa / (GAS_CONSTANT * air.temperature_k * (1 + VIRTUAL_FACTOR * humidity))
    sst_k = sst_c + 273.15
    saturated = saturation_specific_humidity(sst_k, air.pressure_pa)
    relative_u, relative_v = air.wind_u - ocean_u, air.wind_v - ocean_v
    scalar = jnp.maximum(jnp.hypot(relative_u, relative_v), 0.5)
    cd, ch, ce = ncar_coefficients10m(theta, humidity, sst_k, saturated, scalar)
    evaporation = density * scalar * ce * (saturated - humidity)
    sensible = density * cp_air * scalar * ch * (theta - sst_k)
    latent = -(2.5008e6 - 2300 * sst_c) * evaporation
    return SurfaceFluxes(density * cd * scalar * relative_u,
                         density * cd * scalar * relative_v, sensible, latent, evaporation,
                         (1 - albedo) * air.shortwave_down,
                         emissivity * (air.longwave_down - 5.670374419e-8 * sst_k**4),
                         cd, ch, ce)


def apply_open_water_exchange(state, fluxes, air, sss_reference, weights, dt_seconds,
                              *, piston_m_s=50 / (365 * 86400), source_salinity=None,
                              heat_capacity_j_kg_k=C_P):
    """Explicit surface operator; fixed Boussinesq volume, virtual salt flux.

    weights are wet inverse-thickness deposition weights supplied by geometry.
    This first-order, held-flux component has no snow, ice or penetrating shortwave.
    Its caller must reject those unsupported processes. It is not a climate preset.
    Returns state and separate heat, virtual-salt, freshwater and restoring fluxes.
    """
    heat = fluxes.sensible + fluxes.latent + fluxes.shortwave + fluxes.longwave
    freshwater = air.rain + air.runoff - fluxes.evaporation  # kg/m2/s into water
    surface_salt = state.S[:, :, 0] if source_salinity is None else source_salinity
    restoring = piston_m_s * (sss_reference - surface_salt)  # PSU m/s
    salt = -surface_salt * freshwater / RHO_0 + restoring
    temperature = state.T + (dt_seconds * heat / (RHO_0 * heat_capacity_j_kg_k))[:, :, None] * weights
    salinity = state.S + (dt_seconds * salt)[:, :, None] * weights
    # Reader/exchange precision must not silently promote the prognostic state.
    return state._replace(T=temperature.astype(state.T.dtype), S=salinity.astype(state.S.dtype)), heat, salt, freshwater, restoring
