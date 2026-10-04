"""Canonical sources definitions; legacy operations unchanged."""
from zhenmode.model.config.definitions import C_P, RHO_0
from zhenmode.model.numerics.backend import jnp
from zhenmode.model.state.types import JaxStateG


def _surface_heat_weights(p):
    """Per-node heat deposition; sum(weights * wet node thickness) is one."""
    mixed_depth = float(p.mixed_layer_depth_m or 0.)
    if mixed_depth <= 0. and p.mixed_layer_depth_2d is None:
        return p.surface_mask * p.wet_mask_z / p.dz_surface
    requested_depth = (p.mixed_layer_depth_2d if p.mixed_layer_depth_2d is not None
                       else jnp.full_like(p.wet_mask, mixed_depth))
    depth = jnp.where(p.mixed_layer_mask_2d > 0.5, requested_depth, p.dz_surface)
    layer_top = jnp.cumsum(p.dz_node, axis=-1) - p.dz_node
    overlap = jnp.clip(depth[:, :, None] - layer_top, 0., p.dz_node) * p.wet_mask_z
    wet_depth = jnp.maximum(jnp.sum(overlap, axis=-1, keepdims=True), 1e-12)
    return overlap / (wet_depth * p.dz_node)

def _dynamic_ice_closure(state, p, budget=None):
    """Advance the minimal stateful ice closure after one dynamics step.

    This opt-in first-order surface operator applies atmospheric heat once.
    Water sensible heat minus ice latent heat is conserved through phase change.
    Existing ice exchanges heat at the surface node; excess melt energy and
    open-water flux are distributed over the wet mixed-layer overlap. There is
    no ice dynamics, entrainment or resolved ice thermodynamic column.
    """
    if not getattr(p, 'dynamic_ice', False):
        return state

    rho_ice = 917.0
    latent_heat = 3.34e5
    ice_salt_diff = 30.0
    dt = float(p.dt)

    T_sst = state.T[:, :, 0]
    ice = jnp.maximum(jnp.broadcast_to(
        jnp.asarray(state.ice, dtype=state.T.dtype), (p.nx, p.ny)), 0.0)

    insulation = 1.0 / (1.0 + ice / p.ice_insulation_scale_m)
    air_minus_sst = p.T_atm_3d[:, :, 0] - T_sst
    q = (p.Q_heat_2d
         + p.lambda_bulk * air_minus_sst
         + p.coastal_bulk_lambda_2d * air_minus_sst)
    q = q * insulation * p.wet_mask

    weights = _surface_heat_weights(p)
    heat_capacity = RHO_0 * C_P * p.dz_surface
    energy = q * dt
    projected = state.T + energy[:, :, None] * weights / (RHO_0 * C_P)
    freeze_deficit = heat_capacity * jnp.maximum(
        p.ice_freeze_temp_c - projected[:, :, 0], 0.)
    open_ice = freeze_deficit / (rho_ice * latent_heat)
    open_temperature = projected.at[:, :, 0].set(
        jnp.maximum(projected[:, :, 0], p.ice_freeze_temp_c))

    enthalpy = (heat_capacity * (T_sst - p.ice_freeze_temp_c)
                - rho_ice * latent_heat * ice + energy)
    existing_ice = jnp.maximum(-enthalpy, 0.) / (rho_ice * latent_heat)
    existing_temperature = state.T.at[:, :, 0].set(p.ice_freeze_temp_c)
    existing_temperature = (existing_temperature
                             + jnp.maximum(enthalpy, 0.)[:, :, None]
                             * weights / (RHO_0 * C_P))
    ice_new = jnp.where(ice > 0., existing_ice, open_ice) * p.wet_mask
    temperature = jnp.where((ice > 0.)[:, :, None], existing_temperature, open_temperature)
    salt_change = (rho_ice * ice_salt_diff / RHO_0
                   * (ice_new - ice)[:, :, None] * weights)
    T = jnp.where(p.wet_mask_z > 0.5, temperature, state.T)
    S = jnp.where(p.wet_mask_z > 0.5, state.S + salt_change, state.S)
    if budget is not None:
        budget.ice_sources(q, ice_new - ice, rho_ice, ice_salt_diff)
    return JaxStateG(state.u, state.v, T, S, state.eta,
                     ice_new * p.wet_mask)
