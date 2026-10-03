"""Budget shapes and accumulation; independent of solver assembly and audit."""
from ocean_solver.numerics.backend import jnp

METRIC_NAMES = ("fixed_node_water_ice_enthalpy_J", "water_salt_kg", "eta_volume_m3")

STAGE_NAMES = ("linear_diffusion", "sponge", "nonlinear", "free_surface",
               "polar_cap_and_masks", "dynamic_ice")

SOURCE_NAMES = ("prescribed_heat", "bulk_heat", "coastal_bulk_heat", "temperature_restore",
                "salinity_restore", "prescribed_brine", "sponge", "ice_atmosphere_heat", "ice_brine")

NONLINEAR_PROCESS_NAMES = ("advection", "convection", "gm", "redi")

TRANSPORT_METRIC_NAMES = ("local_continuity_residual_m", "nontransport_eta_change_m",
                          "tracer_face_mean_mismatch_m2_per_s", "final_filter_face_change_m2_per_s")

MAXIMUM_BUDGET_FIELDS = ("projection_relative_residual_max", "transport_consistency_max",
                         "bottom_drag_face_change_max_m2_per_s")

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
            "projection_relative_residual_max": jnp.asarray(0., dtype=jnp.float64),
            "transport_consistency_max": jnp.zeros(4, dtype=jnp.float64),
            "transport_audited_steps": jnp.asarray(0., dtype=jnp.float64),
            "bottom_drag_reference_energy_loss_J": jnp.asarray(0., dtype=jnp.float64),
            "bottom_drag_face_change_max_m2_per_s": jnp.asarray(0., dtype=jnp.float64),
            "bottom_drag_audited_halves": jnp.asarray(0., dtype=jnp.float64)}

def accumulate_budget(totals, interval):
    """Sum inventories and squared norms, but preserve the worst residual ratio."""
    return {name: (jnp.maximum(totals[name], interval[name])
                   if name in MAXIMUM_BUDGET_FIELDS else totals[name] + interval[name])
            for name in totals}
