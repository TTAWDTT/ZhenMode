"""Build solver assembly and bind forcing with an independent shadow audit."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from ocean_solver.fd.backend import jnp
from ocean_solver.fd.types import FDPhysParams, JaxStateG
from ocean_solver.runtime.forcing import ForcingBundle, load_forcing
from ocean_solver.runtime.inputs import GridInputs, _lat_band_mask, load_grid_inputs


@dataclass
class SolverAssembly:
    params: FDPhysParams
    terms_fn: Callable
    do_step: Callable
    state: JaxStateG


@dataclass
class RunContext:
    inputs: GridInputs
    forcing: ForcingBundle
    solver: SolverAssembly


def assemble_solver(args, inputs, forcing, services):
    _ret = services.make_solver_global(
        inputs.grid,
        inputs.physics,
        args.dt,
        forcing=forcing.forcing_baked,
        T_atm=forcing.T_atm,
        lambda_bulk=forcing.lambda_bulk,
        S_ref_surf=forcing.S_ref_surf,
        sss_restore_days=args.sss_restore_days,
        coastal_restore_mask=inputs.coastal_restore_mask,
        coastal_restore_days=args.coastal_restore_days,
        coastal_restore_T=inputs.T_init[:, :, 0],
        coastal_bulk_mask=inputs.coastal_bulk_mask,
        coastal_bulk_lambda=args.coastal_bulk_lambda,
        coastal_kappa_h_mask=inputs.coastal_kappa_h_mask,
        coastal_kappa_h=args.coastal_kappa_h,
        coastal_kappa_v_mask=inputs.coastal_kappa_v_mask,
        coastal_kappa_v=args.coastal_kappa_v,
        sponge_days=args.sponge_days,
        sponge_cells=args.sponge_cells,
        T_init=inputs.T_init,
        S_init=inputs.S_init,
        polar_cap_rows=args.polar_cap_rows,
        polar_cap_taper=args.polar_cap_taper,
        return_params=True,
        eta_relax_days=args.eta_relax_days,
        eta_relax_box=args.eta_relax_box,
        eta_relax_buffer=args.eta_relax_buffer,
        dynamic_forcing=forcing.seasonal,
        mode_split=args.mode_split,
        dt_bt=inputs.dt_bt,
        nu_nsub=None
        if args.nu_nsub is None
        else "cfl"
        if args.nu_nsub == "cfl"
        else int(args.nu_nsub),
        dtype=args.dtype,
        use_scan=args.use_scan,
        freeze_adv_vel=args.freeze_adv_vel,
        conservative_kv=args.conservative_kv,
        project_adv_vel=args.project_adv_vel,
        projection_niter=args.projection_niter,
        projection_rtol=args.projection_rtol,
        projection_preconditioner=args.projection_preconditioner,
        projection_max_refinements=args.projection_max_refinements,
        localize_conv=args.localize_conv,
        monotone_adv=args.monotone_adv,
        fct_adv=args.fct_adv,
        mixed_layer_depth_m=args.mixed_layer_depth,
        mixed_layer_mask=_lat_band_mask(inputs.grid, args.mixed_layer_lat_band)
        if args.mixed_layer_lat_band
        else None,
        mixed_layer_depth_2d=inputs.stratification_mld,
        ice_freeze_temp_c=args.ice_freeze_temp,
        ice_salt_flux=args.ice_salt_flux,
        dynamic_ice=args.dynamic_ice,
        ice_insulation_scale_m=args.ice_insulation_scale_m,
    )
    if forcing.seasonal:
        step, init_state_global, _, _params, terms_fn, step_dyn = _ret
    else:
        step, init_state_global, _, _params, terms_fn = _ret
    if args.budget_audit:
        audited_step = services.make_budget_step(_params)
        ordinary_step = step

        def step(state):
            updated = ordinary_step(state)
            audited, ledger = audited_step(state)
            return (updated, audited, ledger)

        if forcing.seasonal:
            ordinary_dynamic_step = step_dyn

            def step_dyn(state, tx, ty, heat, T_atm_3d=None):
                updated = ordinary_dynamic_step(state, tx, ty, heat, T_atm_3d=T_atm_3d)
                audited, ledger = audited_step(state, (tx, ty, heat), T_atm_3d)
                return (updated, audited, ledger)

    do_step = forcing.bind_step(args, step, step_dyn if forcing.seasonal else None)
    state = init_state_global(T_init=jnp.array(inputs.T_init), S_init=jnp.array(inputs.S_init))
    return SolverAssembly(_params, terms_fn, do_step, state)


def build_run_context(configuration, services):
    args = configuration.args
    inputs = load_grid_inputs(args, configuration.parser, services)
    forcing = load_forcing(args, inputs, services)
    solver = assemble_solver(args, inputs, forcing, services)
    return RunContext(inputs, forcing, solver)
