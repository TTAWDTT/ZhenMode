"""Frozen CLI defaults, validation, and resolution-dependent configuration."""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np

from zhenmode.model.audit.validation import finite_number, integer_count
from zhenmode.model.forcing.fields import BULK_LAMBDA_DEFAULT

DT_DEFAULT = 60.0

DT_BT_DEFAULT = 150.0

NU_H_REF_1DEG = 5e6

NU_BI_REF_1DEG = 2e14

MAX_U_BOUND = 10.0

DRIFT_TOL_C = 2.0

AMPLITUDE_CAP_C = 12.0

ETA_BLOWUP_M = 15.0

LAT_MAX_DEFAULT = 60.0

NY_DEFAULT = 120

SMOOTH_PASSES_DEFAULT = 30

MIN_DEPTH_DEFAULT = 100.0

NU_H_DEFAULT = 5e6

NU_BI_DEFAULT = 2e14

POLAR_CAP_ROWS_DEFAULT = 2

POLAR_CAP_TAPER_DEFAULT = 3

LAMBDA_BULK_DEFAULT_G = BULK_LAMBDA_DEFAULT

SPONGE_DAYS_DEFAULT_G = 0.0


def scaled_physics_for_resolution(res, dt_bt, nu_h, nu_bi):
    """Scale dt_bt/nu_h/nu_bi from their 1° values by the CFL power of dx.

    Each parameter that is None (not user-overridden) is replaced by its 1°
    reference scaled by (res/1°)^p. Explicit values pass through untouched.
    """
    f = float(res) / 1.0
    return (
        DT_BT_DEFAULT * f if dt_bt is None else dt_bt,
        NU_H_REF_1DEG * f**2 if nu_h is None else nu_h,
        NU_BI_REF_1DEG * f**4 if nu_bi is None else nu_bi,
    )


def _validate_arguments(args):
    for name in (
        "days",
        "dt",
        "snap_days",
        "min_depth",
        "lat_max",
        "gm_slope_max",
        "mixed_layer_depth_min",
        "mixed_layer_depth_max",
    ):
        finite_number(name, getattr(args, name), positive=True)
    if args.lat_max >= 90.0:
        raise ValueError("lat_max must be below 90 degrees")
    for name in ("dt_bt", "resolution", "projection_rtol"):
        if getattr(args, name) is not None:
            finite_number(name, getattr(args, name), positive=True)
    for name in (
        "nu_h",
        "nu_bi",
        "kappa_v",
        "kappa_conv",
        "kappa_gm",
        "kappa_redi",
        "lambda_bulk",
        "bulk_lambda_mult",
        "sss_restore_days",
        "coastal_restore_days",
        "global_sst_restore_days",
        "coastal_bulk_lambda",
        "coastal_kappa_h",
        "coastal_kappa_v",
        "sponge_days",
        "eta_relax_days",
        "eta_relax_buffer",
        "checkpoint_days",
        "wind_blend_days",
        "mixed_layer_depth",
        "mld_density_delta",
    ):
        if getattr(args, name) is not None:
            finite_number(name, getattr(args, name), nonnegative=True)
    for name in ("ice_air_floor_temp", "ice_freeze_temp", "ice_salt_flux"):
        finite_number(name, getattr(args, name))
    finite_number("ice_insulation_scale_m", args.ice_insulation_scale_m, positive=True)
    for name in (
        "max_steps",
        "smooth_passes",
        "air_marine_smooth_passes",
        "polar_cap_rows",
        "polar_cap_taper",
        "sponge_cells",
        "coastal_restore_cells",
        "coastal_bulk_cells",
        "coastal_kappa_h_cells",
        "coastal_kappa_v_cells",
    ):
        integer_count(name, getattr(args, name))
    if args.ny is not None:
        integer_count("ny", args.ny, minimum=4)
    if args.projection_niter is not None:
        integer_count("projection_niter", args.projection_niter, minimum=1)
    if args.nu_nsub is not None and args.nu_nsub != "cfl":
        try:
            value = int(args.nu_nsub)
        except ValueError as error:
            raise ValueError("nu_nsub must be cfl or a positive integer") from error
        integer_count("nu_nsub", value, minimum=1)
    if args.mixed_layer_depth_min > args.mixed_layer_depth_max:
        raise ValueError("mixed_layer_depth_min exceeds mixed_layer_depth_max")
    if args.save_3d_terms and not args.save_3d:
        raise ValueError("save_3d_terms requires save_3d")
    for name in ("mixed_layer_lat_band", "eta_relax_box"):
        values = getattr(args, name)
        if values is not None:
            for value in values:
                finite_number(name, value)
            if not -90.0 <= values[-2] <= values[-1] <= 90.0:
                raise ValueError(
                    f"{name} must contain an ordered latitude interval within [-90, 90]"
                )
    if args.z_levels:
        nodes = np.asarray([float(value) for value in args.z_levels.split(",")])
        if (
            len(nodes) < 3
            or not np.isfinite(nodes).all()
            or nodes[0] != 0.0
            or not np.all(np.diff(nodes) < 0.0)
        ):
            raise ValueError(
                "z_levels must start at zero and strictly descend through finite depths"
            )
    duration_seconds = finite_number("duration_seconds", args.days * 86400.0, positive=True)
    step_ratio = finite_number("requested_steps", duration_seconds / args.dt, positive=True)
    for name in ("snap_days", "checkpoint_days"):
        if getattr(args, name) > 0.0:
            finite_number(
                f"{name} step ratio", getattr(args, name) * 86400.0 / args.dt, positive=True
            )
    requested_steps = int(round(step_ratio))
    if requested_steps < 1:
        raise ValueError("requested duration rounds to zero integration steps")
    return requested_steps


@dataclass
class RunConfiguration:
    args: argparse.Namespace
    requested_steps: int
    parser: argparse.ArgumentParser


def build_run_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=float, default=365.0)
    ap.add_argument("--dt", type=float, default=DT_DEFAULT)
    ap.add_argument(
        "--mode-split",
        action="store_true",
        help="baroclinic/barotropic mode split: free surface runs in n_subcyc barotropic subcycles of --dt-bt each per baroclinic step, lifting the external-gravity-wave CFL; nu_h moves into the subcycle, kappa_conv is subcycled",
    )
    ap.add_argument(
        "--dt-bt",
        type=float,
        default=None,
        help=f"barotropic subcycle dt [s] (mode split); rounded so n_subcyc*dt_bt exactly fills the baroclinic dt. Default {DT_BT_DEFAULT:g} at 1°; with --resolution it auto-scales as dx (external-gravity-wave CFL).",
    )
    ap.add_argument(
        "--nu-nsub",
        default=None,
        help="nu_h subcycle count in split L half-steps: default = legacy (n_subcyc=24); 'cfl' right-sizes from the explicit-diffusion CFL (15 at 1 deg, 1.6x fewer laplacians); int = verbatim",
    )
    ap.add_argument(
        "--dtype",
        default="float64",
        choices=["float64", "float32"],
        help="compute dtype (default float64 = legacy bit-exact; float32 is ~2-3x faster on Ada GPUs, needs revalidation of drift criteria)",
    )
    ap.add_argument(
        "--use-scan",
        action="store_true",
        help="run the barotropic subcycle as lax.scan (numerically identical; smaller XLA graph, fewer host launches)",
    )
    ap.add_argument(
        "--wind-jit",
        action="store_true",
        help="compile the seasonal-wind month blend into the XLA graph (stack of 12 monthly fields on device, day as a traced scalar). Numerically identical to the Python blend; removes the per-step Python interp + host-to-device copy (measured ~1.2-1.4x on split runs)",
    )
    ap.add_argument("--lat-max", type=float, default=LAT_MAX_DEFAULT)
    ap.add_argument(
        "--resolution",
        type=float,
        default=None,
        help="horizontal grid spacing in degrees. With the default --resolution-remap legacy this must be a multiple of the 0.1° ETOPO source grid; with 'area' it may be any positive value and is conservatively remapped. When given, nx and ny are DERIVED from it and --ny is ignored. Default: None = legacy 1° grid (--ny decides). KEY: on finer grids the 1°-calibrated --dt-bt/--nu-h/--nu-bi all violate their CFL and the run diverges; this flag auto-scales them by dx^1/2/4 respectively (see scaled_physics_for_resolution). Pass any of those explicitly to override. At 1.0° the scaling is a no-op. See docs/resolution_cfl_limits.md.",
    )
    ap.add_argument(
        "--resolution-remap",
        choices=("legacy", "area"),
        default="legacy",
        help="'legacy' preserves integer 0.1° block averaging; 'area' uses conservative spherical-area overlap and supports arbitrary positive resolutions",
    )
    ap.add_argument(
        "--ny",
        type=int,
        default=None,
        help=f"meridional grid points at the default 1° resolution (default {NY_DEFAULT}); ignored when --resolution is given",
    )
    ap.add_argument(
        "--z-levels",
        default=None,
        help="vertical grid override: comma-separated node depths [m, negative down] e.g. '0,-5,-15,...' — replaces the default 14-level grid (used by the E2 AMOC experiment)",
    )
    ap.add_argument("--smooth-passes", type=int, default=SMOOTH_PASSES_DEFAULT)
    ap.add_argument("--min-depth", type=float, default=MIN_DEPTH_DEFAULT)
    ap.add_argument(
        "--nu-h",
        type=float,
        default=None,
        help=f"horizontal Laplacian viscosity [m^2/s]. Default {NU_H_DEFAULT:g} at 1°; with --resolution it auto-scales as dx^2 (explicit-diffusion CFL).",
    )
    ap.add_argument(
        "--nu-bi",
        type=float,
        default=None,
        help=f"biharmonic hyperviscosity [m^4/s]. Default {NU_BI_DEFAULT:g} at 1°; with --resolution it auto-scales as dx^4 (biharmonic CFL).",
    )
    ap.add_argument(
        "--kappa-v",
        type=float,
        default=None,
        help="vertical diffusivity override [m^2/s]; default keeps PhysicsConfig (1e-5). Accelerated-spinup phase A uses 2e-4 (x20) to speed deep-ocean tracer adjustment",
    )
    ap.add_argument(
        "--kappa-conv",
        type=float,
        default=0.05,
        help="convective vertical diffusivity override [m^2/s] (default 0.05; CFL at the 5 m top layer caps dt <= 0.5*dz^2/kappa_conv = 250 s at 0.05)",
    )
    ap.add_argument(
        "--bulk-lambda-mult",
        type=float,
        default=1.0,
        help="multiplier on the bulk heat-flux transfer coefficient (stronger surface restoring; phase-A distortion)",
    )
    ap.add_argument("--lambda-bulk", type=float, default=LAMBDA_BULK_DEFAULT_G)
    ap.add_argument("--no-bulk-flux", action="store_true")
    ap.add_argument(
        "--no-meridional-heat-flux",
        action="store_true",
        help="disable idealized meridional Q_heat for wind-only external comparison; combine with --no-bulk-flux",
    )
    ap.add_argument(
        "--real-air-temp",
        action="store_true",
        help="use annual-mean NCEP R1 2-m air temperature as the bulk-flux target instead of the zonal WOA SST profile. This is an observed, spatially varying atmospheric forcing field, not a WOA SST restore.",
    )
    ap.add_argument(
        "--real-air-temp-monthly",
        action="store_true",
        help="use 12 monthly NCEP R1 2-m air fields with the same seasonal blending as wind; requires --seasonal-wind. This takes precedence over --real-air-temp.",
    )
    ap.add_argument(
        "--ice-air-floor",
        action="store_true",
        help="simple sea-ice proxy: floor the bulk target at the freezing point so sub-freezing air does not force open water below freezing",
    )
    ap.add_argument(
        "--ice-air-floor-temp",
        type=float,
        default=-1.8,
        help="freezing-point floor for --ice-air-floor [C]",
    )
    ap.add_argument(
        "--ice-salt-flux",
        type=float,
        default=0.0,
        help="minimal sea-ice brine-rejection salt flux [psu/s] applied where the live SST is at or below the freezing point; 0 = off",
    )
    ap.add_argument(
        "--ice-freeze-temp",
        type=float,
        default=-1.8,
        help="freezing-point threshold for --ice-salt-flux [C]",
    )
    ap.add_argument(
        "--dynamic-ice",
        action="store_true",
        help="carry a minimal ice-thickness state with latent growth/melt, conductivity insulation, and brine salt flux; overrides --ice-salt-flux",
    )
    ap.add_argument(
        "--ice-insulation-scale-m",
        type=float,
        default=1.0,
        help="1m of ice reduces surface exchange by this scale; 1/(1+h/scale)",
    )
    ap.add_argument(
        "--air-marine-smooth-passes",
        type=int,
        default=0,
        help="number of wet-cell-only smoothing passes for the bulk air-temperature target; 0 keeps the raw target",
    )
    ap.add_argument(
        "--sss-restore-days",
        type=float,
        default=0.0,
        help="surface salinity restoring timescale [days]; 0 = off. Haney relaxation of SSS to the WOA surface climatology (v0.1 had NO surface salt flux — SSS drifted ~-0.9 psu/kyr non-converging)",
    )
    ap.add_argument(
        "--sss-restore-zonal",
        action="store_true",
        help="relax to the ZONAL MEAN of WOA SSS instead of the full 2D field (keeps zonal SSS structure predicted; analogous to the T_atm non-circularity rule)",
    )
    ap.add_argument(
        "--coastal-restore-days",
        type=float,
        default=0.0,
        help="diagnostic surface-temperature restoring timescale [days] in the land-adjacent band; 0 = off",
    )
    ap.add_argument(
        "--global-sst-restore-days",
        type=float,
        default=0.0,
        help="shared external-benchmark SST restoring timescale [days] over all wet cells; 0 = off",
    )
    ap.add_argument(
        "--coastal-restore-cells",
        type=int,
        default=0,
        help="width of the land-adjacent restoring band in cells",
    )
    ap.add_argument(
        "--coastal-restore-taper",
        choices=("none", "linear", "cos"),
        default="none",
        help="weight the coastal restoring band away from land; none keeps the current hard mask",
    )
    ap.add_argument(
        "--mixed-layer-lat-band",
        type=float,
        nargs=2,
        default=None,
        metavar=("LAT_MIN", "LAT_MAX"),
        help="restrict --mixed-layer-depth to a latitude band",
    )
    ap.add_argument(
        "--mixed-layer-depth",
        type=float,
        default=None,
        help="optional mixed-layer heat-capacity depth [m]. When set, the same surface heat flux is spread over this slab instead of the top grid-cell thickness.",
    )
    ap.add_argument(
        "--mixed-layer-mode",
        choices=("constant", "stratification"),
        default="constant",
        help="constant uses --mixed-layer-depth everywhere in the selected mask; stratification derives a 2D depth from the WOA density threshold",
    )
    ap.add_argument(
        "--mld-density-delta",
        type=float,
        default=0.03,
        help="density threshold for --mixed-layer-mode stratification [kg/m^3]",
    )
    ap.add_argument(
        "--mixed-layer-depth-min",
        type=float,
        default=10.0,
        help="lower clamp for the stratification-derived MLD [m]",
    )
    ap.add_argument(
        "--mixed-layer-depth-max",
        type=float,
        default=100.0,
        help="upper clamp for the stratification-derived MLD [m]",
    )
    ap.add_argument(
        "--coastal-bulk-lambda",
        type=float,
        default=0.0,
        help="extra bulk heat-exchange coefficient [W/m^2/K] in the land-adjacent band; 0 = off",
    )
    ap.add_argument(
        "--coastal-bulk-cells",
        type=int,
        default=0,
        help="width of the land-adjacent extra bulk-flux band",
    )
    ap.add_argument(
        "--coastal-kappa-h",
        type=float,
        default=0.0,
        help="extra horizontal tracer diffusivity [m^2/s] in the land-adjacent band; 0 = off",
    )
    ap.add_argument(
        "--coastal-kappa-h-cells",
        type=int,
        default=0,
        help="width of the land-adjacent horizontal-diffusion band",
    )
    ap.add_argument(
        "--coastal-kappa-v",
        type=float,
        default=0.0,
        help="extra vertical tracer diffusivity [m^2/s] in the land-adjacent band; 0 = off",
    )
    ap.add_argument(
        "--coastal-kappa-v-cells",
        type=int,
        default=0,
        help="width of the land-adjacent vertical-diffusion band",
    )
    ap.add_argument(
        "--kappa-gm",
        type=float,
        default=0.0,
        help="GM eddy diffusivity [m^2/s] (bolus transport); 0=off",
    )
    ap.add_argument(
        "--kappa-redi", type=float, default=0.0, help="Redi isopycnal diffusivity [m^2/s]; 0=off"
    )
    ap.add_argument(
        "--gm-slope-max", type=float, default=0.01, help="isopycnal slope limiter (dimensionless)"
    )
    ap.add_argument(
        "--freeze-adv-vel",
        action="store_true",
        help="RK2 tracer stage-2: advect with the old velocity instead of the freshly advected predictor (default off)",
    )
    ap.add_argument(
        "--conservative-kv",
        action="store_true",
        help="use interface-flux vertical diffusion (default off)",
    )
    ap.add_argument(
        "--project-adv-vel",
        action="store_true",
        help="project the RK2 stage-2 tracer velocity onto the native column-divergence-free space; convergence and moving-volume budget must be checked separately; default off",
    )
    ap.add_argument(
        "--projection-niter",
        type=int,
        default=None,
        help="CG cap; explicit value overrides OCEAN_PAV_NITER (default 150)",
    )
    ap.add_argument(
        "--projection-rtol",
        type=float,
        default=None,
        help="CG relative tolerance, floored at 32*dtype epsilon",
    )
    ap.add_argument(
        "--projection-preconditioner",
        choices=["none", "jacobi"],
        default="none",
        help="native wet-face Poisson preconditioner (default none)",
    )
    ap.add_argument(
        "--projection-max-refinements",
        type=int,
        choices=[0, 1, 2],
        default=2,
        help="bounded actual-transport correction passes (default 2; 0 disables)",
    )
    ap.add_argument(
        "--localize-conv",
        action="store_true",
        help="gate convective adjustment PER-INTERFACE (mix only across unstable interfaces) instead of the historical column-wide mask that mixes the whole column whenever any interface is unstable; default off",
    )
    ap.add_argument(
        "--monotone-adv",
        action="store_true",
        help="use first-order donor-cell horizontal tracer fluxes (vertical flux is already donor-cell). More diffusive than centered flux, but conservative and monotone under the combined tracer CFL; default off keeps the historical centered path bit-exact",
    )
    ap.add_argument(
        "--fct-adv",
        action="store_true",
        help="experimental TVD/MUSCL flux-limited horizontal tracer transport: reconstruct the face value from the two donor cells with a minmod slope and choose the state consistent with the face velocity. This is a first bounded-flux prototype, not a full Zalesak 3D FCT limiter; it takes precedence over --monotone-adv",
    )
    ap.add_argument("--sponge-days", type=float, default=SPONGE_DAYS_DEFAULT_G)
    ap.add_argument("--sponge-cells", type=int, default=0)
    ap.add_argument("--polar-cap-rows", type=int, default=POLAR_CAP_ROWS_DEFAULT)
    ap.add_argument("--polar-cap-taper", type=int, default=POLAR_CAP_TAPER_DEFAULT)
    ap.add_argument(
        "--eta-relax-days",
        type=float,
        default=0.0,
        help="semi-enclosed-sea SSH relaxation timescale [days]; 0 = off (Mediterranean strait artifact)",
    )
    ap.add_argument(
        "--eta-relax-box",
        type=float,
        nargs=4,
        default=None,
        metavar=("LON0", "LON1", "LAT0", "LAT1"),
        help="relaxation box in degrees E/N (core mask = 1 inside)",
    )
    ap.add_argument(
        "--eta-relax-buffer",
        type=float,
        default=1.0,
        help="cos-taper buffer width [degrees] around the box",
    )
    ap.add_argument("--snap-days", type=float, default=10.0)
    ap.add_argument("--seasonal-wind", action="store_true")
    ap.add_argument("--wind-year", type=int, default=2023)
    ap.add_argument("--wind-blend-days", type=float, default=5.0)
    ap.add_argument(
        "--month", default="2023-01", help="fixed wind month if not --seasonal-wind (YYYY-MM)"
    )
    ap.add_argument("--tag", default=None)
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--log-dir", default="logs")
    ap.add_argument(
        "--init-from",
        default=None,
        help="npz with T_init/S_init (precomputed WOA on this grid); skips woa_data (offline nodes)",
    )
    ap.add_argument("--save-3d", action="store_true")
    ap.add_argument(
        "--save-3d-terms",
        action="store_true",
        help="additionally save the per-term dT/dt decomposition [adv, diff_h, diff_v, conv, gm, redi] at each 3D snap (offline blowup attribution; needs --save-3d)",
    )
    ap.add_argument(
        "--max-steps", type=int, default=0, help="hard cap on steps (0 = no cap); for short probes"
    )
    ap.add_argument(
        "--checkpoint-days",
        type=float,
        default=0.0,
        help="save full state checkpoint every N days (0 = off); enables --restart-from resume after container kill",
    )
    ap.add_argument(
        "--restart-from",
        default=None,
        help="versioned checkpoint from --checkpoint-days; requires identical code/effective parameters/grid/forcing/dtype/backend, restores absolute time and histories; old state-only files need migration",
    )
    ap.add_argument(
        "--budget-audit",
        action="store_true",
        help="strict accepted-step ledger; shadow audit must match ordinary state bytes (extra cost)",
    )
    ap.add_argument(
        "--allow-forcing-fallback",
        action="store_true",
        help="exploration only: explicitly allow unavailable NCEP forcing to fall back",
    )
    ap.add_argument(
        "--strict-forcing",
        action="store_true",
        help="require all selected local input/cache files before loading; prohibit fallback/download",
    )
    return ap


def parse_run_configuration(argv=None):
    ap = build_run_parser()
    args = ap.parse_args() if argv is None else ap.parse_args(argv)
    if args.strict_forcing and args.allow_forcing_fallback:
        ap.error("--strict-forcing forbids --allow-forcing-fallback")
    try:
        requested_steps = _validate_arguments(args)
    except ValueError as error:
        ap.error(str(error))
    return RunConfiguration(args, requested_steps, ap)
