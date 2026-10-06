"""Explicit opt-in JRA -> live bulk -> FD stress -> surface operator coupling.

The production default is unchanged. This component route rejects snow, calving,
ice and competing legacy surface sources, and cannot qualify a global climate run.
"""
import numpy as np

from zhenmode.model.config import RHO_0
from zhenmode.model.diagnostics.surface_budget import surface_budget
from zhenmode.model.solver.numerics.backend import jnp
from zhenmode.model.solver.physics.air_sea import apply_open_water_exchange, open_water_fluxes
from zhenmode.model.solver.physics.eos import heat_capacity
from zhenmode.model.solver.physics.surface import _surface_heat_weights


def bind_open_water_step(reader, step_dyn, params, *, sss_reference):
    """Point-sample weather at step start; hold exchange over one native FD step.

    FD receives stress only. After its step, apply heat and virtual salt once,
    using start-state exchange. Deposition uses the configured node/mixed-layer
    weights. This explicit split is first order; radiation penetration is absent.
    Returns a callable(state, forcing_seconds) -> (state, component_report).
    """
    if step_dyn is None:
        raise ValueError('online forcing requires dynamic_forcing=True')
    forbidden = ('lambda_bulk', 'restore_coef_S', 'ice_salt_flux', 'dynamic_ice',
                 'coastal_bulk_lambda_2d', 'coastal_restore_coef_2d', 'sponge_rate')
    if any(np.any(np.asarray(getattr(params, name)) != 0) for name in forbidden):
        raise ValueError('online component conflicts with legacy surface/ice/sponge sources')
    wet = np.asarray(params.wet_mask) > .5
    if hasattr(reader, 'wet') and not np.array_equal(reader.wet, wet):
        raise ValueError('forcing wet grid differs from solver geometry')
    reference = np.asarray(sss_reference)
    if reference.shape != wet.shape or not np.isfinite(reference[wet]).all() or np.any(reference[wet] <= 0):
        raise ValueError('SSS reference must be a finite positive native-grid field')
    reference = np.where(wet, reference, 0.)
    weights = _surface_heat_weights(params)
    dt = float(params.dt)
    cp = heat_capacity(params)
    area = np.asarray(params.dx_2d) * float(params.dy) * wet
    volume = area[:, :, None] * np.asarray(params.dz_node) * np.asarray(params.wet_mask_z)
    if not np.allclose(np.sum(np.asarray(weights) * np.asarray(params.dz_node), axis=-1)[wet], 1):
        raise ValueError('surface deposition weights do not conserve column input')

    def advance(state, forcing_seconds):
        sst = state.T[:, :, 0]
        phase_temperature = sst
        freezing_temperature = params.ice_freeze_temp_c
        if params.thermodynamics == 'teos10_reference':
            from zhenmode.model.solver.physics.teos10 import (
                potential_from_conservative,
                surface_freezing_ct,
                validate_state,
            )
            validate_state(state.S,state.T,params.eos_pressure_dbar)
            sst = potential_from_conservative(state.S[:, :, 0],sst)
            freezing_temperature = surface_freezing_ct(state.S[:, :, 0])
        if (any(not np.isfinite(np.asarray(a)).all() for a in state) or
                np.any(np.asarray(state.ice) != 0) or
                np.any(np.asarray(phase_temperature)[wet] <= np.broadcast_to(freezing_temperature,wet.shape)[wet])):
            raise ValueError('open-water component requires finite ice-free, unfrozen state')
        weather = reader.sample(forcing_seconds, interval_end_seconds=forcing_seconds + dt)
        if np.any(np.asarray(weather.snow)[wet] != 0) or np.any(np.asarray(weather.calving)[wet] != 0):
            raise ValueError('snow/calving energetics require the full ice surface route')
        # Dry geometry is outside this component, not a missing-weather fallback.
        air = weather._replace(temperature_k=jnp.where(wet, weather.temperature_k, 273.15),
                               pressure_pa=jnp.where(wet, weather.pressure_pa, 101325.))
        raw = open_water_fluxes(air, sst, state.u[:, :, 0], state.v[:, :, 0])
        fluxes = raw._make(jnp.where(wet, value, 0.) for value in raw)
        dynamics = step_dyn(state, fluxes.tau_x, fluxes.tau_y, jnp.zeros(wet.shape, dtype=state.T.dtype))
        updated, heat, salt, freshwater, restoring = apply_open_water_exchange(
            dynamics, fluxes, air, jnp.asarray(reference), weights, dt,
            source_salinity=state.S[:, :, 0],heat_capacity_j_kg_k=cp)
        final_temperature = updated.T[:, :, 0]
        if params.thermodynamics == 'teos10_reference':
            freezing_temperature = surface_freezing_ct(updated.S[:, :, 0])
            validate_state(updated.S,updated.T,params.eos_pressure_dbar)
        if np.any(np.asarray(final_temperature)[wet] <= np.broadcast_to(freezing_temperature,wet.shape)[wet]):
            raise ValueError('surface cooling requires ice physics; component step rejected')
        report = surface_budget(dynamics, updated, wet_volume=volume, wet_area=area,
                                heat_flux=heat, salt_flux=salt, dt_seconds=dt,heat_capacity_j_kg_k=cp)
        report.update(data_kind=reader.data_kind, forcing_seconds=float(forcing_seconds),
                      interval_seconds=dt, coupling='start_state_held_first_order_split',
                      state_precision=str(updated.T.dtype), exchange_precision=str(fluxes.sensible.dtype),
                      freshwater_kg=float(dt * np.sum(area * np.asarray(freshwater))),
                      restoring_salt_kg=float(dt * RHO_0 / 1000 * np.sum(area * np.asarray(restoring))),
                      surface_integrals={name: float(dt * np.sum(area * np.asarray(getattr(fluxes, name))))
                                         for name in ('tau_x', 'tau_y', 'sensible', 'latent', 'evaporation',
                                                      'shortwave', 'longwave')},
                      integral_units={'tau_x': 'N s', 'tau_y': 'N s', 'sensible': 'J', 'latent': 'J',
                                      'evaporation': 'kg', 'shortwave': 'J', 'longwave': 'J'},
                      water_volume_update='none_virtual_salt',
                      thermodynamics=params.thermodynamics,
                      temperature_definition='CT' if params.thermodynamics=='teos10_reference' else 'legacy_temperature',
                      salinity_definition='SR_approximates_SA' if params.thermodynamics=='teos10_reference' else 'legacy_salinity',
                      execution_ready_for_full_protocol=False)
        return updated, report
    return advance
