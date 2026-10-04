"""Production stdout log, progress and run description."""

from __future__ import annotations

import sys

from zhenmode.model.runtime.cli import AMPLITUDE_CAP_C, DRIFT_TOL_C, ETA_BLOWUP_M, MAX_U_BOUND


def print_run_header(args, context, recovery):
    header = []
    header.append("=" * 70)
    header.append(f"GLOBAL FD LONG INTEGRATION ({args.days:.0f} days)")
    header.append("=" * 70)
    header.append(
        f"grid: {context.inputs.grid.nx}x{context.inputs.grid.ny}x{context.inputs.grid.nz}  res={context.inputs.gcfg.resolution:g}°  dx_eq={context.inputs.dx_eq:.0f}m  lon[{context.inputs.grid.lon[0]:.1f},{context.inputs.grid.lon[-1]:.1f}]E lat[{context.inputs.grid.lat[0]:.1f},{context.inputs.grid.lat[-1]:.1f}]N"
    )
    header.append(
        f"dt={args.dt:.0f}s  steps={recovery.n_total}  snap every {recovery.n_snap} steps ({args.snap_days:.0f}d)"
    )
    if args.mode_split:
        ns = context.solver.params.n_subcyc
        _nnu = context.solver.params.nu_nsub if context.solver.params.nu_nsub is not None else ns
        header.append(
            f"MODE SPLIT: baroclinic dt={args.dt:.0f}s, barotropic subcycle {ns} x {args.dt / ns:.1f}s (--dt-bt {context.inputs.dt_bt:.0f}s), conv_nsub={context.solver.params.conv_nsub}, nu_nsub={_nnu}, scan={('ON' if context.solver.params.use_scan else 'py')}"
        )
    header.append(
        f"physics: nu_h={context.inputs.physics.nu_h:g}  nu_bi={context.inputs.physics.nu_bi:g}  kappa_conv={context.inputs.physics.kappa_conv}  kappa_v={context.inputs.physics.kappa_v:g}"
    )
    if (
        args.freeze_adv_vel
        or args.conservative_kv
        or args.project_adv_vel
        or args.localize_conv
        or args.monotone_adv
        or args.fct_adv
    ):
        header.append(
            f"RK2 flags: freeze_adv_vel={args.freeze_adv_vel}  conservative_kv={args.conservative_kv}  project_adv_vel={args.project_adv_vel}  localize_conv={args.localize_conv}  monotone_adv={args.monotone_adv}  fct_adv={args.fct_adv}"
        )
    if args.dtype != "float64":
        header.append(f"DTYPE: {args.dtype} (compute/checkpoint; snapshots float64)")
    if args.bulk_lambda_mult != 1.0:
        header.append(
            f"distorted physics: bulk-lambda x{args.bulk_lambda_mult:g} (accelerated-spinup phase A)"
        )
    if context.inputs.physics.kappa_gm > 0 or context.inputs.physics.kappa_redi > 0:
        header.append(
            f"sub-grid closure: kappa_gm={context.inputs.physics.kappa_gm:g} m^2/s  kappa_redi={context.inputs.physics.kappa_redi:g} m^2/s  gm_slope_max={context.inputs.physics.gm_slope_max:g}"
        )
    else:
        header.append("sub-grid closure: NONE (kappa_gm=0, kappa_redi=0)")
    header.append(
        f"bulk_flux={context.forcing.lambda_bulk:g} W/m^2/K"
        + (
            f" (T_atm={context.forcing.T_atm_source})"
            if context.forcing.lambda_bulk > 0.0
            else " (off)"
        )
    )
    if args.sss_restore_days > 0.0:
        header.append(
            f"sss_restore: tau={args.sss_restore_days:g}d  target={('zonal WOA SSS' if args.sss_restore_zonal else 'full 2D WOA SSS')}"
        )
    else:
        header.append("sss_restore: NONE (no surface salt flux)")
    header.append("wall: no-flux N/S (v=0 at boundary rows, mirror-ghost dy)")
    header.append(f"wind: {context.forcing.wind_src}")
    if context.forcing.seasonal and args.wind_blend_days > 0:
        header.append(f"wind blend: {args.wind_blend_days:g}d linear window at month boundaries")
        if context.forcing.T_atm_months is not None:
            header.append(
                f"air-temp blend: {args.wind_blend_days:g}d linear window at month boundaries"
            )
    elif context.forcing.seasonal:
        header.append("wind blend: NONE (step at month boundaries)")
    if args.sponge_days > 0 and args.sponge_cells > 0:
        header.append(f"sponge: {args.sponge_cells}-cell band, tau={args.sponge_days:g}d")
    else:
        header.append("sponge: NONE")
    if args.polar_cap_rows > 0:
        header.append(
            f"polar cap: {args.polar_cap_rows} rows + {args.polar_cap_taper}-row cos^2 taper"
        )
    else:
        header.append("polar cap: NONE")
    if args.eta_relax_days > 0 and args.eta_relax_box is not None:
        b = args.eta_relax_box
        header.append(
            f"eta_relax: tau={args.eta_relax_days:g}d  box=[{b[0]:.1f},{b[1]:.1f}]E x [{b[2]:.1f},{b[3]:.1f}]N  buffer={args.eta_relax_buffer:g}deg (mass-conserving)"
        )
    else:
        header.append("eta_relax: NONE")
    header.append(
        f"init: T_init_max={context.inputs.T_init_max:.2f}C  amplitude_cap={context.inputs.T_init_max + AMPLITUDE_CAP_C:.2f}C"
    )
    header.append(
        f"criteria: max|u|<{MAX_U_BOUND}  drift_tol={DRIFT_TOL_C}C  watchdog |eta|>{ETA_BLOWUP_M}m"
    )
    header.append("")
    header.append(
        f"{'day':>6} {'step':>7} {'max|u|':>9} {'max|T|':>8} {'max|eta|':>9} {'SSH_std':>9} {'KE':>12} {'NaN':>6}"
    )
    header.append("-" * 78)
    for line in header:
        print(line)

class _Tee:
    """Duplicate writes to the original stdout and a log file.

    Long runs are typically launched detached (nohup / taskset), where the
    console scrollback is lost; out_log preserves the progress table and the
    VERDICT block for later inspection.
    """

    def __init__(self, path):
        self.file = open(path, "a", buffering=1)
        self.stdout = sys.stdout

    def write(self, s):
        self.stdout.write(s)
        self.file.write(s)

    def flush(self):
        self.stdout.flush()
        self.file.flush()
