"""Read-only actual-stage audit, not a complete physical moving-volume budget.

Heat is fixed-node water sensible heat minus ice latent heat. Salt is nominal
water salt mass; eta displacement is separate from the fixed reference volume.
Decomposition closure does not imply that the independent source budget closes.
"""
import jax
import jax.numpy as jnp

from config import C_P, RHO_0
from jax_solver_global import _step_impl, _vertical_transport_iface

METRIC_NAMES = ("fixed_node_water_ice_enthalpy_J", "water_salt_kg", "eta_volume_m3")
STAGE_NAMES = ("linear_diffusion", "sponge", "nonlinear", "free_surface",
               "polar_cap_and_masks", "dynamic_ice")
SOURCE_NAMES = ("prescribed_heat", "bulk_heat", "coastal_bulk_heat", "temperature_restore",
                "salinity_restore", "prescribed_brine", "sponge", "ice_atmosphere_heat", "ice_brine")
NONLINEAR_PROCESS_NAMES = ("advection", "convection", "gm", "redi")


def empty_budget():
    """Zero for interval accumulation with accumulate_budget; not a closure claim."""
    return {"observed_change": jnp.zeros(3, dtype=jnp.float64),
            "stage_changes": jnp.zeros((len(STAGE_NAMES), 3), dtype=jnp.float64),
            "source_inputs": jnp.zeros((len(SOURCE_NAMES), 3), dtype=jnp.float64),
            "decomposition_residual": jnp.zeros(3, dtype=jnp.float64),
            "budget_residual": jnp.zeros(3, dtype=jnp.float64),
            "absolute_decomposition_residual": jnp.zeros(3, dtype=jnp.float64),
            "absolute_budget_residual": jnp.zeros(3, dtype=jnp.float64),
            "change_scale": jnp.zeros(3, dtype=jnp.float64),
            "nonlinear_process_changes": jnp.zeros((len(NONLINEAR_PROCESS_NAMES), 3), dtype=jnp.float64),
            "nonlinear_process_scale": jnp.zeros(3, dtype=jnp.float64),
            "advection_boundary_changes": jnp.zeros(3, dtype=jnp.float64),
            "advection_boundary_residual": jnp.zeros(3, dtype=jnp.float64),
            "absolute_advection_boundary_residual": jnp.zeros(3, dtype=jnp.float64),
            "nonlinear_accounting_residual": jnp.zeros(3, dtype=jnp.float64),
            "absolute_nonlinear_accounting_residual": jnp.zeros(3, dtype=jnp.float64),
            "surface_displacement_tracer_change": jnp.zeros(3, dtype=jnp.float64),
            "projection_transport_norm_squared": jnp.zeros(2, dtype=jnp.float64),
            "projection_relative_residual_max": jnp.asarray(0., dtype=jnp.float64)}


def accumulate_budget(totals, interval):
    """Sum inventories and squared norms, but preserve the worst residual ratio."""
    return {name: (jnp.maximum(totals[name], interval[name])
                   if name == "projection_relative_residual_max" else totals[name] + interval[name])
            for name in totals}


class _StageRecorder:
    """A trace-local recorder; no Python state survives an evaluated JIT step."""

    def __init__(self, params):
        self.params = params
        self.area = jnp.asarray(params.dx_2d, dtype=jnp.float64) * jnp.asarray(params.dy, dtype=jnp.float64)
        self.surface_area = self.area * params.wet_mask
        self.volume = self.area[:, :, None] * jnp.asarray(params.dz_node, dtype=jnp.float64) * params.wet_mask_z
        self.stages = {name: jnp.zeros(3, dtype=jnp.float64) for name in STAGE_NAMES}
        self.sources = {name: jnp.zeros(3, dtype=jnp.float64) for name in SOURCE_NAMES}
        self.change_scale = jnp.zeros(3, dtype=jnp.float64)
        self.processes = {name: jnp.zeros(3, dtype=jnp.float64) for name in NONLINEAR_PROCESS_NAMES}
        self.process_scale = jnp.zeros(3, dtype=jnp.float64)
        self.advection_boundary = jnp.zeros(3, dtype=jnp.float64)
        self.projection_norm_squared = jnp.zeros(2, dtype=jnp.float64)
        self.projection_relative_residual_max = jnp.asarray(0., dtype=jnp.float64)

    def difference(self, before, after):
        heat = RHO_0 * C_P * (jnp.asarray(after.T, dtype=jnp.float64) - jnp.asarray(before.T, dtype=jnp.float64)) * self.volume
        latent = 917. * 3.34e5 * (jnp.asarray(after.ice, dtype=jnp.float64)
                                 - jnp.asarray(before.ice, dtype=jnp.float64)) * self.surface_area
        salt = RHO_0 / 1000. * (jnp.asarray(after.S, dtype=jnp.float64) - jnp.asarray(before.S, dtype=jnp.float64)) * self.volume
        displacement = (jnp.asarray(after.eta, dtype=jnp.float64)
                        - jnp.asarray(before.eta, dtype=jnp.float64)) * self.surface_area
        change = jnp.stack((jnp.sum(heat) - jnp.sum(latent), jnp.sum(salt), jnp.sum(displacement)))
        scale = jnp.stack((jnp.sum(jnp.abs(heat)) + jnp.sum(jnp.abs(latent)),
                           jnp.sum(jnp.abs(salt)), jnp.sum(jnp.abs(displacement))))
        return change, scale

    def stage(self, name, before, after):
        change, scale = self.difference(before, after)
        self.stages[name] = self.stages[name] + change
        self.change_scale = self.change_scale + scale

    def surface_sources(self, tendencies):
        interval = self.params.dt / 2.
        for index, tendency in enumerate(tendencies):
            factor = RHO_0 * C_P if index < 4 else RHO_0 / 1000.
            component = 0 if index < 4 else 1
            amount = jnp.sum(jnp.asarray(tendency, dtype=jnp.float64) * self.volume) * factor * interval
            self.sources[SOURCE_NAMES[index]] = self.sources[SOURCE_NAMES[index]].at[component].add(amount)

    def nonlinear_terms(self, tendencies):
        interval = self.params.dt / 2.
        for name, (temperature, salinity) in zip(NONLINEAR_PROCESS_NAMES, tendencies, strict=True):
            heat = RHO_0 * C_P * interval * jnp.asarray(temperature, dtype=jnp.float64) * self.volume
            salt = RHO_0 / 1000. * interval * jnp.asarray(salinity, dtype=jnp.float64) * self.volume
            self.processes[name] = self.processes[name] + jnp.stack((jnp.sum(heat), jnp.sum(salt), jnp.asarray(0.)))
            self.process_scale = self.process_scale + jnp.stack((jnp.sum(jnp.abs(heat)),
                                                                jnp.sum(jnp.abs(salt)), jnp.asarray(0.)))

    def advection_boundary_fluxes(self, temperature_flux, salinity_flux):
        interval = self.params.dt / 2.
        heat = RHO_0 * C_P * interval * jnp.sum(jnp.asarray(temperature_flux, dtype=jnp.float64) * self.surface_area)
        salt = RHO_0 / 1000. * interval * jnp.sum(jnp.asarray(salinity_flux, dtype=jnp.float64) * self.surface_area)
        self.advection_boundary = self.advection_boundary + jnp.stack((heat, salt, jnp.asarray(0.)))

    def column_projection(self, velocity_x, velocity_y, corrected_x, corrected_y):
        before = jnp.asarray(_vertical_transport_iface(velocity_x, velocity_y, self.params)[..., 0], dtype=jnp.float64)
        after = jnp.asarray(_vertical_transport_iface(corrected_x, corrected_y, self.params)[..., 0], dtype=jnp.float64)
        norms = jnp.stack((jnp.sum(before ** 2 * self.surface_area),
                           jnp.sum(after ** 2 * self.surface_area)))
        self.projection_norm_squared = self.projection_norm_squared + norms
        relative = jnp.where(norms[0] > 0., jnp.sqrt(norms[1] / jnp.where(norms[0] > 0., norms[0], 1.)),
                             jnp.where(norms[1] == 0., 0., jnp.inf))
        self.projection_relative_residual_max = jnp.maximum(self.projection_relative_residual_max, relative)

    def surface_displacement_change(self, before, after):
        eta_before = jnp.asarray(before.eta, dtype=jnp.float64)
        eta_after = jnp.asarray(after.eta, dtype=jnp.float64)
        temperature = (eta_after * jnp.asarray(after.T[..., 0], dtype=jnp.float64)
                       - eta_before * jnp.asarray(before.T[..., 0], dtype=jnp.float64))
        salinity = (eta_after * jnp.asarray(after.S[..., 0], dtype=jnp.float64)
                    - eta_before * jnp.asarray(before.S[..., 0], dtype=jnp.float64))
        return jnp.stack((RHO_0 * C_P * jnp.sum(temperature * self.surface_area),
                          RHO_0 / 1000. * jnp.sum(salinity * self.surface_area), jnp.asarray(0.)))

    def sponge_sources(self, before, decay):
        fraction = 1. - jnp.asarray(decay, dtype=jnp.float64)
        temperature = (jnp.asarray(self.params.T_clim_3d, dtype=jnp.float64)
                       - jnp.asarray(before.T, dtype=jnp.float64)) * fraction
        salinity = (jnp.asarray(self.params.S_clim_3d, dtype=jnp.float64)
                    - jnp.asarray(before.S, dtype=jnp.float64)) * fraction
        source = jnp.stack((RHO_0 * C_P * jnp.sum(temperature * self.volume),
                            RHO_0 / 1000. * jnp.sum(salinity * self.volume), jnp.asarray(0.)))
        self.sources["sponge"] = self.sources["sponge"] + source

    def ice_sources(self, heat_flux, thickness_change, rho_ice, salt_difference):
        heat = jnp.sum(jnp.asarray(heat_flux, dtype=jnp.float64) * self.surface_area) * self.params.dt
        salt = (rho_ice * salt_difference / 1000.
                * jnp.sum(jnp.asarray(thickness_change, dtype=jnp.float64) * self.surface_area))
        self.sources["ice_atmosphere_heat"] = self.sources["ice_atmosphere_heat"].at[0].add(heat)
        self.sources["ice_brine"] = self.sources["ice_brine"].at[1].add(salt)

    def result(self, before, after):
        observed, _ = self.difference(before, after)
        stages = jnp.stack([self.stages[name] for name in STAGE_NAMES])
        sources = jnp.stack([self.sources[name] for name in SOURCE_NAMES])
        decomposition_residual = observed - jnp.sum(stages, axis=0)
        budget_residual = observed - jnp.sum(sources, axis=0)
        processes = jnp.stack([self.processes[name] for name in NONLINEAR_PROCESS_NAMES])
        boundary_residual = self.processes["advection"] - self.advection_boundary
        nonlinear_residual = self.stages["nonlinear"] - jnp.sum(sources[:6], axis=0) - jnp.sum(processes, axis=0)
        return {"observed_change": observed, "stage_changes": stages, "source_inputs": sources,
                "decomposition_residual": decomposition_residual,
                "budget_residual": budget_residual,
                "absolute_decomposition_residual": jnp.abs(decomposition_residual),
                "absolute_budget_residual": jnp.abs(budget_residual), "change_scale": self.change_scale,
                "nonlinear_process_changes": processes, "nonlinear_process_scale": self.process_scale,
                "advection_boundary_changes": self.advection_boundary,
                "advection_boundary_residual": boundary_residual,
                "absolute_advection_boundary_residual": jnp.abs(boundary_residual),
                "nonlinear_accounting_residual": nonlinear_residual,
                "absolute_nonlinear_accounting_residual": jnp.abs(nonlinear_residual),
                "surface_displacement_tracer_change": self.surface_displacement_change(before, after),
                "projection_transport_norm_squared": self.projection_norm_squared,
                "projection_relative_residual_max": self.projection_relative_residual_max}


def make_budget_step(params):
    """Return a JIT step yielding (unchanged solver state, per-step audit).

    Runtime forcing is (tau_x, tau_y, Q_heat); atmosphere has the solver's
    (nx, ny, 1) shape. Both are applied to the actual core and its source audit.
    Tables use METRIC_NAMES, STAGE_NAMES, SOURCE_NAMES and NONLINEAR_PROCESS_NAMES.
    Internal transport and linearized eta*C inventory are diagnostics, not sources.
    No conservation PASS is manufactured from the stage decomposition identity.
    """
    @jax.jit
    def advance(state, forcing=None, atmosphere=None):
        updates = {}
        if forcing is not None:
            if len(forcing) != 3:
                raise ValueError("forcing must contain tau_x, tau_y and Q_heat")
            updates.update(tau_x_2d=forcing[0], tau_y_2d=forcing[1], Q_heat_2d=forcing[2])
        if atmosphere is not None:
            updates["T_atm_3d"] = atmosphere
        current_params = params._replace(**updates)
        recorder = _StageRecorder(current_params)
        updated = _step_impl(state, current_params, budget=recorder)
        return updated, recorder.result(state, updated)

    return advance
