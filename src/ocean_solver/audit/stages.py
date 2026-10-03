"""Read-only actual-stage audit, not a complete physical moving-volume budget.

Heat is fixed-node water sensible heat minus ice latent heat. Salt is nominal
water salt mass; eta displacement is separate from the fixed reference volume.
Decomposition closure does not imply that the independent source budget closes.
Bottom-drag reference kinetic loss is a separate audit, not a heat source or
whole momentum/energy closure. A zero audited-half count means uninstrumented,
not necessarily zero physical drag in older schemes.
"""
import jax
import jax.numpy as jnp

from ocean_solver.audit.schema import MAXIMUM_BUDGET_FIELDS as MAXIMUM_BUDGET_FIELDS
from ocean_solver.audit.schema import METRIC_NAMES as METRIC_NAMES
from ocean_solver.audit.schema import NONLINEAR_PROCESS_NAMES as NONLINEAR_PROCESS_NAMES
from ocean_solver.audit.schema import SOURCE_NAMES as SOURCE_NAMES
from ocean_solver.audit.schema import STAGE_NAMES as STAGE_NAMES
from ocean_solver.audit.schema import TRANSPORT_METRIC_NAMES as TRANSPORT_METRIC_NAMES
from ocean_solver.audit.schema import accumulate_budget as accumulate_budget
from ocean_solver.audit.schema import empty_budget as empty_budget
from ocean_solver.config.definitions import C_P, RHO_0
from ocean_solver.dynamics.transport import (
    _face_transport_divergence,
    _layer_face_transports,
    _vertical_transport_iface,
)
from ocean_solver.timestepping.integration import (
    _step_impl,
)


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
        self.tracer_face_mean = (jnp.zeros_like(self.surface_area), jnp.zeros_like(self.surface_area))
        self.fast_face_mean = None
        self.transport_consistency_max = jnp.zeros(4, dtype=jnp.float64)
        self.transport_audited_steps = jnp.asarray(0., dtype=jnp.float64)
        self.prefilter_faces = None
        self.drag_energy_loss = jnp.asarray(0., dtype=jnp.float64)
        self.drag_face_change = jnp.asarray(0., dtype=jnp.float64)
        self.drag_halves = jnp.asarray(0., dtype=jnp.float64)

    def bottom_drag(self, before, after):
        velocity_before = jnp.asarray(before.u, dtype=jnp.float64) ** 2 + jnp.asarray(before.v, dtype=jnp.float64) ** 2
        velocity_after = jnp.asarray(after.u, dtype=jnp.float64) ** 2 + jnp.asarray(after.v, dtype=jnp.float64) ** 2
        self.drag_energy_loss = self.drag_energy_loss + 0.5 * RHO_0 * jnp.sum((velocity_before - velocity_after) * self.volume)
        before_faces = _layer_face_transports(before.u, before.v, self.params)
        after_faces = _layer_face_transports(after.u, after.v, self.params)
        difference = jnp.max(jnp.stack([jnp.max(jnp.abs(jnp.sum(new - old, axis=-1)))
                                        for new, old in zip(after_faces, before_faces, strict=True)]))
        self.drag_face_change = jnp.maximum(self.drag_face_change, difference)
        self.drag_halves = self.drag_halves + 1.

    def tracer_transport(self, layer_transport, weight=0.5):
        self.tracer_face_mean = tuple(total + weight * jnp.sum(flux, axis=-1)
                                      for total, flux in zip(self.tracer_face_mean, layer_transport, strict=True))

    def barotropic_transport(self, eta_before, eta_after, face_mean, filter_change):
        self.fast_face_mean = face_mean
        continuity = (eta_after - eta_before + self.params.dt * _face_transport_divergence(*face_mean, self.params)
                      - filter_change)
        self.transport_consistency_max = self.transport_consistency_max.at[:2].set(
            jnp.stack((jnp.max(jnp.abs(continuity)), jnp.max(jnp.abs(filter_change)))))
        self.transport_audited_steps = jnp.asarray(1., dtype=jnp.float64)

    def before_transport_filter(self, state):
        self.prefilter_faces = tuple(jnp.sum(flux, axis=-1)
                                     for flux in _layer_face_transports(state.u, state.v, self.params))

    def after_transport_filter(self, state):
        if self.prefilter_faces is not None:
            final_faces = tuple(jnp.sum(flux, axis=-1)
                                for flux in _layer_face_transports(state.u, state.v, self.params))
            mismatch = jnp.max(jnp.stack([jnp.max(jnp.abs(final - before))
                                          for final, before in zip(final_faces, self.prefilter_faces, strict=True)]))
            self.transport_consistency_max = self.transport_consistency_max.at[3].set(mismatch)

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

    def surface_sources(self, tendencies, duration=None):
        interval = self.params.dt / 2. if duration is None else duration
        for index, tendency in enumerate(tendencies):
            factor = RHO_0 * C_P if index < 4 else RHO_0 / 1000.
            component = 0 if index < 4 else 1
            amount = jnp.sum(jnp.asarray(tendency, dtype=jnp.float64) * self.volume) * factor * interval
            self.sources[SOURCE_NAMES[index]] = self.sources[SOURCE_NAMES[index]].at[component].add(amount)

    def nonlinear_terms(self, tendencies, duration=None, absolute_tendencies=None):
        interval = self.params.dt / 2. if duration is None else duration
        if absolute_tendencies is None:
            absolute_tendencies = jax.tree.map(jnp.abs, tendencies)
        for name, (temperature, salinity), (absolute_temperature, absolute_salinity) in zip(
                NONLINEAR_PROCESS_NAMES, tendencies, absolute_tendencies, strict=True):
            heat = RHO_0 * C_P * interval * jnp.asarray(temperature, dtype=jnp.float64) * self.volume
            salt = RHO_0 / 1000. * interval * jnp.asarray(salinity, dtype=jnp.float64) * self.volume
            self.processes[name] = self.processes[name] + jnp.stack((jnp.sum(heat), jnp.sum(salt), jnp.asarray(0.)))
            if duration is None:
                scale_heat, scale_salt = jnp.abs(heat), jnp.abs(salt)
            else:
                scale_heat = RHO_0 * C_P * interval * absolute_temperature * self.volume
                scale_salt = RHO_0 / 1000. * interval * absolute_salinity * self.volume
            self.process_scale = self.process_scale + jnp.stack((jnp.sum(scale_heat),
                                                                jnp.sum(scale_salt), jnp.asarray(0.)))

    def advection_boundary_fluxes(self, temperature_flux, salinity_flux, duration=None):
        interval = self.params.dt / 2. if duration is None else duration
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
        if self.fast_face_mean is not None:
            mismatch = jnp.max(jnp.stack([jnp.max(jnp.abs(tracer - fast))
                                          for tracer, fast in zip(self.tracer_face_mean, self.fast_face_mean, strict=True)]))
            self.transport_consistency_max = self.transport_consistency_max.at[2].set(mismatch)
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
                "projection_relative_residual_max": self.projection_relative_residual_max,
                "transport_consistency_max": self.transport_consistency_max,
                "transport_audited_steps": self.transport_audited_steps,
                "bottom_drag_reference_energy_loss_J": self.drag_energy_loss,
                "bottom_drag_face_change_max_m2_per_s": self.drag_face_change,
                "bottom_drag_audited_halves": self.drag_halves}

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
