"""Assemble inputs, solver and verified continuation, then start the production run."""

from __future__ import annotations

import os
import sys
from collections.abc import Callable
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from functools import partial
from pathlib import Path

import zhenmode.model.diagnostics.budgets as budgets
import zhenmode.model.inputs.bathymetry as bathymetry
import zhenmode.model.inputs.forcing.idealized as idealized
import zhenmode.model.inputs.forcing.reanalysis as reanalysis
import zhenmode.model.inputs.forcing.seasonal as seasonal
import zhenmode.model.inputs.initial_conditions as initial_conditions
import zhenmode.model.inputs.prepare as inputs
import zhenmode.model.solver.factory as factory
from zhenmode.model.config import DEFAULT_CONFIG, Config
from zhenmode.model.inputs.forcing.bundle import ForcingBundle, load_forcing
from zhenmode.model.inputs.prepare import GridInputs, _lat_band_mask, load_grid_inputs
from zhenmode.model.io import restart
from zhenmode.model.io.output import prepare_output_paths
from zhenmode.model.io.records import _validate_restart_history
from zhenmode.model.io.restart import RestartRecord, load_restart
from zhenmode.model.runtime.cli import ETA_BLOWUP_M, MAX_U_BOUND, parse_run_configuration
from zhenmode.model.runtime.run_loop import run_integration
from zhenmode.model.solver.numerics.backend import jnp
from zhenmode.model.solver.state import FDPhysParams, JaxStateG
from zhenmode.provenance.sources import _source_identity as source_identity
from zhenmode.provenance.sources import production_source_modules, source_paths, source_root


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

@dataclass
class OutputManifest:
    files: dict = dataclass_field(default_factory=dict)

@dataclass
class RecoveryContext:
    state: JaxStateG
    n_total: int
    n_snap: int
    ckpt_path: str
    checkpoint_contract: dict | None
    restored: RestartRecord | None
    output_manifest: OutputManifest
    start_step: int
    n_3d_snaps: int
    n_ckpt: int | None

def prepare_recovery(args, requested_steps, context, paths, services, source_directory):
    state = context.solver.state
    n_ckpt = None
    n_total = requested_steps
    if args.max_steps > 0:
        n_total = min(n_total, args.max_steps)
    n_snap = max(1, int(round(args.snap_days * 86400.0 / args.dt)))
    ckpt_path = os.path.join(args.out_dir, f"ckpt_{paths.tag}.npz")
    checkpoint_contract = None
    restored = None
    output_manifest = OutputManifest()
    if args.restart_from or args.checkpoint_days > 0:
        source_dir = Path(source_directory)
        source_names = production_source_modules()
        checkpoint_contract = services.make_restart_contract(
            context.inputs.grid,
            context.solver.params,
            dtype=args.dtype,
            forcing={
                "initial_T": context.inputs.T_init,
                "initial_S": context.inputs.S_init,
                "baked": context.forcing.forcing_baked,
                "wind_months": context.forcing.wind_months,
                "air_months": context.forcing.T_atm_months,
            },
            controls={
                "calendar": "repeating_360_day_30_day_months",
                "seasonal": context.forcing.seasonal,
                "wind_blend_days": args.wind_blend_days,
                "wind_jit": args.wind_jit,
                "n_snap": n_snap,
                "save_3d": args.save_3d,
                "save_3d_terms": args.save_3d_terms,
                "budget_kind": "strict_shadow_accepted_stage_ledger_v1"
                if args.budget_audit
                else "snapshot_inventory_only_no_flux_ledger",
                "forcing_provenance": context.forcing.forcing_provenance,
                "production_schema_version": 4,
                "monitor_schema_version": 1,
                "monitor_comparison": ">",
                "velocity_limit": MAX_U_BOUND,
                "eta_limit": ETA_BLOWUP_M,
            },
            code_paths=source_paths(source_dir, source_names),
            execution=context.forcing.execution_identity,
        )
    start_step = 0
    n_3d_snaps = 0
    if args.restart_from:
        restored = load_restart(args.restart_from, checkpoint_contract)
        _validate_restart_history(
            restored,
            context.inputs.grid,
            n_snap,
            args.dt,
            args.save_3d,
            args.save_3d_terms,
            {"3d": paths.three_d_dir, "terms": paths.three_d_terms_dir},
            args.budget_audit,
        )
        state = JaxStateG(**{name: jnp.asarray(value) for name, value in restored.state.items()})
        start_step = restored.step
        if start_step > n_total:
            raise ValueError("restart step exceeds the requested final step")
        n_3d_snaps = restored.counters["n_3d_snaps"]
        output_manifest = OutputManifest(dict(restored.outputs))
        print(
            f"RESUME from {args.restart_from}: step {start_step} (day {start_step * args.dt / 86400.0:.1f}), {len(restored.history['days'])} diagnostic rows restored, {n_3d_snaps} verified 3D snapshots kept"
        )
    if args.checkpoint_days > 0:
        n_ckpt = max(1, int(round(args.checkpoint_days * 86400.0 / args.dt)))
        if n_ckpt % n_snap:
            raise ValueError("checkpoint cadence must be an integer multiple of snapshot cadence")
    return RecoveryContext(
        state=state,
        n_total=n_total,
        n_snap=n_snap,
        ckpt_path=ckpt_path,
        checkpoint_contract=checkpoint_contract,
        restored=restored,
        output_manifest=output_manifest,
        start_step=start_step,
        n_3d_snaps=n_3d_snaps,
        n_ckpt=n_ckpt,
    )

@dataclass
class RunServices:
    default_config: Config
    make_global_grid: Callable
    get_initial_fields: Callable
    heat_flux_meridional: Callable
    build_seasonal_wind_global: Callable
    real_wind_forcing: Callable
    load_monthly_mean_air_temp: Callable
    load_annual_mean_air_temp: Callable
    air_temp_profile: Callable
    make_solver_global: Callable
    make_budget_step: Callable
    make_restart_contract: Callable
    input_files: Callable
    source_identity: Callable
    resolve_grid_dimensions: Callable | None = None

def default_services(default_config=None):
    """Bind actual loader/process owners for one invocation; callers may replace services."""
    config = DEFAULT_CONFIG if default_config is None else default_config
    return RunServices(
        default_config=config,
        make_global_grid=bathymetry.make_global_grid,
        get_initial_fields=initial_conditions.get_initial_fields,
        heat_flux_meridional=idealized.heat_flux_meridional,
        build_seasonal_wind_global=seasonal.build_seasonal_wind_global,
        real_wind_forcing=reanalysis.real_wind_forcing,
        load_monthly_mean_air_temp=reanalysis.load_monthly_mean_air_temp,
        load_annual_mean_air_temp=reanalysis.load_annual_mean_air_temp,
        air_temp_profile=idealized.air_temp_profile,
        make_solver_global=factory.make_solver_global,
        make_budget_step=budgets.make_budget_step,
        make_restart_contract=restart.make_restart_contract,
        input_files=partial(inputs._input_files, default_config=config),
        source_identity=partial(source_identity, source_root(__file__)),
    )

def run_main(services, source_directory):
    configuration = parse_run_configuration()
    args = configuration.args
    paths = prepare_output_paths(args)
    if args.strict_forcing:
        missing = [str(path) for path in services.input_files(args).values() if not path.is_file()]
        if missing:
            raise ValueError(f"strict forcing preflight: missing local inputs {missing}")
    context = build_run_context(configuration, services)
    recovery = prepare_recovery(
        args, configuration.requested_steps, context, paths, services, source_directory
    )
    return run_integration(args, configuration.requested_steps, context, paths, recovery)

def main():
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except (AttributeError, OSError):
        pass
    return run_main(default_services(), source_root(__file__))


if __name__ == "__main__":
    sys.exit(main())
