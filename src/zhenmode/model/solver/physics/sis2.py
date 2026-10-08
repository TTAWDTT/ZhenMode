"""GPU surface exchanges with native SIS2; no duplicated sea-ice physics.

The fixed reference cells stay fixed. Surface inventories include eta times
the surface point value, consistent with the existing displacement ledger.
The full native ice state is external; JaxStateG.ice is not its restart state.
"""

import math
from numbers import Real

from zhenmode.model.config import CP0_TEOS10, RHO_0
from zhenmode.model.solver.numerics.backend import jnp
from zhenmode.model.solver.physics.teos10 import surface_freezing_ct


def extract_frazil(state, params):
    """Raise supercooled CT to TEOS freezing; send the energy deficit to SIS2.

    Frazil is a positive J/m² heat deficit. The opposite water enthalpy change
    is recorded explicitly and must cancel the native ice input, including
    any unused frazil returned by SIS2. This is a surface-node operation.
    """
    wet = params.wet_mask > 0.5
    height = params.dz_node[..., 0] + state.eta
    freezing = surface_freezing_ct(state.S[..., 0])
    deficit = jnp.where(wet, jnp.maximum(freezing - state.T[..., 0], 0), 0)
    energy = RHO_0 * CP0_TEOS10 * height * deficit
    return state._replace(T=state.T.at[..., 0].add(deficit)), energy


def apply_sis2_exchange(state, fields, params, duration=None):
    """Apply held native fluxes once and return per-area water/salt/heat inputs.

    SIS2 salt is kg/m²/s out of ocean; SR is g/kg. Native mass enthalpy terms
    are interval J/m² INTO ice and hence enter ocean with the opposite sign.
    Liquid precipitation carries zero enthalpy at the chosen 0°C reference;
    frozen precipitation/calving also removes latent fusion energy. Shortwave
    is deposited at the surface in this component, pending penetration wiring.
    """
    interval = fields["interval_seconds"]
    if jnp.ndim(interval) != 0:
        raise ValueError("native exchange interval must be a scalar")
    if isinstance(interval, Real) and (interval <= 0 or not math.isfinite(interval)):
        raise ValueError("native exchange interval must be positive and finite")
    if duration is None:
        duration = interval
    if isinstance(interval, Real) and isinstance(duration, Real) and interval != duration:
        raise ValueError("ocean exchange duration differs from native SIS2 interval")
    valid_interval = jnp.isfinite(interval) & (interval > 0) & (interval == duration)
    wet = params.wet_mask > 0.5
    mass = interval * (
        fields["lprec"] + fields["fprec"] + fields["runoff"] + fields["calving"] - fields["flux_q"]
    )
    salt = -interval * fields["flux_salt"]
    fusion = fields["thermo_constants"][0]
    shortwave = sum(fields[n] for n in ("sw_vis_dir", "sw_vis_dif", "sw_nir_dir", "sw_nir_dif"))
    heat = interval * (
        shortwave
        + fields["flux_lw"]
        - fields["flux_t"]
        - fields["flux_lh"]
        - fusion * (fields["fprec"] + fields["calving"])
        + fields["runoff_hflx"]
        + fields["calving_hflx"]
    ) - (fields["enth_mass_in_ocn"] + fields["enth_mass_out_ocn"])
    # Restore any unused heat deficit; extraction must not manufacture energy.
    heat = heat - fields["frazil_left"]
    mass, salt, heat = (jnp.where(wet, value, 0) for value in (mass, salt, heat))
    old_height = params.dz_node[..., 0] + state.eta
    new_eta = state.eta + mass / RHO_0
    new_height = params.dz_node[..., 0] + new_eta
    valid = valid_interval & (old_height > 0) & (new_height > 0)
    safe_height = jnp.where(wet & valid, new_height, 1)
    temperature = (old_height * state.T[..., 0] + heat / (RHO_0 * CP0_TEOS10)) / safe_height
    salinity = (old_height * state.S[..., 0] + 1000 * salt / RHO_0) / safe_height
    # A traced mismatch or exhausted cell cannot become a finite accepted state.
    temperature, salinity = (jnp.where(valid, value, jnp.nan) for value in (temperature, salinity))
    updated = state._replace(
        T=state.T.at[..., 0].set(jnp.where(wet, temperature, state.T[..., 0])),
        S=state.S.at[..., 0].set(jnp.where(wet, salinity, state.S[..., 0])),
        eta=jnp.where(wet, new_eta, state.eta),
    )
    return updated, (mass, salt, heat), jnp.where(wet, new_height, 1)
