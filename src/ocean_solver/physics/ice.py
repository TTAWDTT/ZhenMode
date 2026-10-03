"""Minimal thermodynamic mixed-layer + sea-ice closure.

This is deliberately a prototype, not a full sea-ice model.  The point is to
close the loop:

    atmospheric forcing -> mixed-layer heat capacity
                         -> freezing / melting
                         -> ocean heat + salt flux
                         -> SST / mixed-layer response

The eventual solver integration can use these functions directly; the unit
tests here are the regression gate before that integration.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class MLIceConfig:
    """Parameters for the smallest defensible mixed-layer/sea-ice closure."""

    mixed_layer_depth_m: float = 50.0
    freeze_temp_c: float = -1.8
    bulk_lambda: float = 80.0
    rho_ocean: float = 1025.0
    rho_ice: float = 917.0
    cp_ocean: float = 3992.0
    latent_heat_fusion: float = 3.34e5
    ice_salt_diff: float = 30.0
    # Simple conductivity proxy: 1 m of ice weakens the atmospheric exchange
    # by about this factor.  Not intended as a full CICE-style thermodynamic
    # column; it only supplies the minimal feedback.
    ice_insulation_scale_m: float = 1.0


def surface_heat_flux(T_mld: float | np.ndarray, T_atm: float | np.ndarray,
                      ice_thickness_m: float | np.ndarray,
                      cfg: MLIceConfig) -> np.ndarray:
    """Atmosphere/ocean heat flux into the mixed layer, W/m2.

    Positive flux warms the ocean.  Ice reduces the exchange using a simple
    conductivity proxy; the sign remains the same, so melt/growth still has a
    feedback.
    """
    ice = np.maximum(np.asarray(ice_thickness_m, dtype=float), 0.0)
    insulation = 1.0 / (1.0 + ice / cfg.ice_insulation_scale_m)
    return (cfg.bulk_lambda
            * (np.asarray(T_atm, dtype=float)
               - np.asarray(T_mld, dtype=float))
            * insulation)


def ice_growth_rate_m_per_s(temperature_change_c: float | np.ndarray,
                            cfg: MLIceConfig) -> np.ndarray:
    """Convert a mixed-layer heat deficit/surplus to ice growth/melt rate."""
    heat_change_j = (cfg.rho_ocean * cfg.cp_ocean
                     * cfg.mixed_layer_depth_m
                     * np.asarray(temperature_change_c, dtype=float))
    latent = cfg.rho_ice * cfg.latent_heat_fusion
    return heat_change_j / latent


def mixed_layer_ice_step(T_mld: float | np.ndarray,
                         ice_thickness_m: float | np.ndarray,
                         T_atm: float | np.ndarray,
                         dt_s: float,
                         cfg: MLIceConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Advance the minimal mixed-layer/ice closure by one step.

    Returns:
        ``T_mld_new, ice_thickness_new, surface_heat_flux_wm2, salt_flux_psu_m_s``.

    The closure is intentionally explicit.  It is safe for small dt and is
    meant to be the unit-tested core of a future solver coupling.
    """
    if dt_s <= 0.0:
        raise ValueError("dt_s must be positive")
    T = np.asarray(T_mld, dtype=float)
    ice = np.maximum(np.asarray(ice_thickness_m, dtype=float), 0.0)
    T_atm = np.asarray(T_atm, dtype=float)
    heat_capacity = cfg.rho_ocean * cfg.cp_ocean * cfg.mixed_layer_depth_m

    q = surface_heat_flux(T, T_atm, ice, cfg)
    T_new = T + q * dt_s / heat_capacity

    # Convert heat deficit/surplus relative to freezing into ice growth/melt.
    deficit = cfg.freeze_temp_c - T_new
    latent_change = heat_capacity * np.abs(deficit) / (
        cfg.rho_ice * cfg.latent_heat_fusion)
    grows = (q < 0.0) & (T_new < cfg.freeze_temp_c)
    melt = (q > 0.0) & (ice > 0.0)
    active = grows | melt
    latent_change = np.where(active, latent_change, 0.0)
    ice_new = np.where(grows, ice + latent_change,
                       np.where(melt,
                                np.maximum(0.0, ice - latent_change),
                                ice))
    # If ice is present, the heat budget first changes phase, not temperature.
    T_new = np.where(active, cfg.freeze_temp_c, T_new)
    salt_flux = (cfg.rho_ice * cfg.ice_salt_diff
                 * np.asarray(latent_change) / cfg.rho_ocean)
    salt_flux = np.where(grows, salt_flux, -salt_flux)
    return T_new, ice_new, q, salt_flux
