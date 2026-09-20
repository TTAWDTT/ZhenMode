"""
Long-integration driver for the global finite-difference solver (jax_solver_global).

Runs the global FD solver -- closed no-flux N/S walls at lat_max=60, nu_h=5e6
spin-up stabilizer, dt=60 -- with real seasonal NCEP wind and a bulk air-sea
heat flux. The point of the global domain is the non-circular A1/A2 zonal SST
skill test: a global non-periodic domain removes the periodic-BC crutch, so
large-scale SST structure has to emerge from
geometry + wind + bathymetry rather than from a prescribed meridional T_atm
clamp.

Stability: the no-flux wall + nu_h=5e6 hold both no-wind and wind-forced
(tau0=0.1) runs to 1000 steps with 0 NaN and max|T| bounded. This driver
extends that to a full annual integration. The pass/fail criteria below are
pre-registered -- do NOT move the bar after running.

Output:
  - results/global_<tag>.npz   (snapshots: eta/T/u maxima, KE, SSH_std, T_top)
  - logs/global_<tag>.log      (full monitor trace + verdict)
  - results/global_<tag>_3d/   (streamed 3D T/U/V snapshots, one .npy each)

Usage:
  python src/run_long_integration_global.py --days 365 --seasonal-wind --tag g365d
  python src/run_long_integration_global.py --days 200 --tag g200d_smoke   # shorter probe
"""
import argparse
import os
import sys
import time
from dataclasses import replace

try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import jax

jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
import numpy as np

from config import DEFAULT_CONFIG, GlobalGridConfig, PhysicsConfig
from forcing import (
    BULK_LAMBDA_DEFAULT,
    air_temp_profile,
    heat_flux_meridional,
    ocean_zonal_mean,
)
from grid import global_grid_dims, make_global_grid
from jax_solver_global import JaxStateG, make_solver_global
from wind_reanalysis import real_wind_forcing
from woa_data import get_initial_fields

# ── Pre-registered criteria (frozen; do not tune to a result) ──────────
# The gate is about stability + bounded drift, not about matching any
# particular magnitude, so MAX_U_BOUND and the drift tolerances are the same
# at every resolution.
DT_DEFAULT = 60.0              # s — explicit free-surface + FD CFL-safe at 1°
# 1°-grid reference values for --resolution auto-scaling. When --resolution is
# given, dt_bt/nu_h/nu_bi default to these scaled by (res/1°)^p, p = 1/2/4
# respectively (external-gravity-wave CFL, Laplacian CFL, biharmonic CFL).
# Measurements at 0.5° (docs/resolution_cfl_limits.md): dt_bt=300 diverges
# (barotropic CFL ~283 s), nu_h=5e6 diverges, nu_bi=2e14 diverges — all three
# are 1°-calibrated and must shrink. At res=1.0 the scaling is a no-op, so the
# legacy defaults are preserved exactly.
DT_BT_DEFAULT = 150.0          # s — barotropic subcycle dt at 1°
NU_H_REF_1DEG = 5e6            # m²/s — nu_h at 1° (see NU_H_DEFAULT)
NU_BI_REF_1DEG = 2e14          # m⁴/s — nu_bi at 1° (see NU_BI_DEFAULT)


def scaled_physics_for_resolution(res, dt_bt, nu_h, nu_bi):
    """Scale dt_bt/nu_h/nu_bi from their 1° values by the CFL power of dx.

    Each parameter that is None (not user-overridden) is replaced by its 1°
    reference scaled by (res/1°)^p. Explicit values pass through untouched.
    """
    f = float(res) / 1.0
    return (
        DT_BT_DEFAULT * f if dt_bt is None else dt_bt,
        NU_H_REF_1DEG * f ** 2 if nu_h is None else nu_h,
        NU_BI_REF_1DEG * f ** 4 if nu_bi is None else nu_bi,
    )

MAX_U_BOUND = 10.0            # m/s
DRIFT_TOL_C = 2.0             # C, second-half climb tolerance (monotonic drift)
AMPLITUDE_CAP_C = 12.0        # C, absolute ceiling above init max
ETA_BLOWUP_M = 15.0           # m, divergence watchdog. The grid under
                              # real NCEP wind builds a wind-driven barotropic
                              # setup whose tropical pile-up reaches ~5m while
                              # the mean stays ~0 (mass-conserved). 15m still
                              # catches true divergence while allowing the
                              # physical spin-up barotropic mode.
    # ── Global FD stable config ──────────────────────────────────────
LAT_MAX_DEFAULT = 60.0        # truncate poleward (cos=0.5, no metric singularity)
NY_DEFAULT = 120              # 1° resolution at lat_max=60 -> 120 rows
SMOOTH_PASSES_DEFAULT = 30    # bathymetry smoothing (steep topographic PGF)
MIN_DEPTH_DEFAULT = 100.0     # floor shallow coastal columns (bad WOA extrapolation)
NU_H_DEFAULT = 5e6            # m²/s spin-up stabilizer (CFL_edge=0.094, safe;
                              # production OGCMs use ~1e3-1e4; can lower post-spinup)
NU_BI_DEFAULT = 2e14           # biharmonic hyperviscosity (∇⁴), scale-selective
                              # damping of 2-3 cell grid-scale noise. CFL is NOT
                              # violated: nu_bi*dt/dx⁴ ≈ 0.02 at 1°/dt=60 and
                              # ≈0.04 at dt=120, both below the ~0.05 explicit-Euler
                              # limit (the earlier "CFL-violating" note was wrong —
                              # the real reason biharmonic had no effect was a
                              # sign-cancellation bug in the Strang residual, fixed
                              # in jax_solver_global.py). Calibrated empirically:
                              # 5e13 suppresses the 10d hotspot but a stronger
                              # coastal hotspot re-nucleates by day 60; 2e14 keeps
                              # 20d max|T|≈30 and is the candidate for 90/365d.
POLAR_CAP_ROWS_DEFAULT = 2    # ON: zonally average poleward rows to kill the
                              # cos(lat)->0 metric blow-up at the pole wall
                              # (the j=0 single-gridpoint divergence).
POLAR_CAP_TAPER_DEFAULT = 3   # cos^2-taper the cap edge over this many extra
                              # rows; a hard cutoff creates a meridional cliff
                              # at the cap inner edge that blows up in ~12 steps.
LAMBDA_BULK_DEFAULT_G = BULK_LAMBDA_DEFAULT
SPONGE_DAYS_DEFAULT_G = 0.0   # OFF (no residual instability at lat_max=60; the
                              # no-flux wall + nu_h sufficed. Available if a longer
                              # run re-nucleates poleward-row instability — option B.)


def state_is_finite(state):
    for f in (state.u, state.v, state.T, state.S, state.eta):
        a = np.asarray(f)
        if not np.isfinite(a).all():
            return False
    return True


def total_kinetic_energy(state, ocean_mask):
    u = np.asarray(state.u)
    v = np.asarray(state.v)
    ke = 0.5 * np.sum((u ** 2 + v ** 2) * ocean_mask[:, :, None])
    return float(ke)


def build_seasonal_wind_global(grid, year=2023):
    """12 monthly NCEP wind-stress snapshots interpolated to the global grid.

    real_wind_forcing is grid-agnostic (bilinear interp on grid.lat/lon);
    the global grid's ±lat_max range is within NCEP lat coverage. The y-edge
    taper is applied per snapshot (taper_2d_y) — compatible with the no-flux
    wall (smooths the boundary anomaly, doesn't conflict with v=0 at the wall).
    """
    months = []
    for m in range(1, 13):
        month_idx = (year - 1948) * 12 + (m - 1)
        tau_x, tau_y = real_wind_forcing(month_idx=month_idx, grid=grid)
        months.append((tau_x, tau_y))
    return months


def interp_seasonal_wind(wind_months, day, blend_days=5.0):
    """Linearly blend monthly wind snapshots near 30-day month boundaries.

    Removes the artificial
    step discontinuity at month transitions that excited boundary instabilities.
    """
    month_len = 30.0
    mpos = day % month_len
    mi = int(day // month_len) % 12
    half = blend_days / 2.0
    if blend_days <= 0.0 or (mpos >= half and mpos <= month_len - half):
        return wind_months[mi]
    if mpos < half:
        w = (half + mpos) / blend_days
        prev = wind_months[(mi - 1) % 12]
        cur = wind_months[mi]
        return ((1.0 - w) * prev[0] + w * cur[0],
                (1.0 - w) * prev[1] + w * cur[1])
    w = (mpos - (month_len - half)) / blend_days
    cur = wind_months[mi]
    nxt = wind_months[(mi + 1) % 12]
    return ((1.0 - w) * cur[0] + w * nxt[0],
            (1.0 - w) * cur[1] + w * nxt[1])


def interp_seasonal_wind_jit(wind_stack, day, blend_days=5.0):
    """JIT-traceable version of interp_seasonal_wind.

    wind_stack: (12, nx, ny) constant on device; day is a traced scalar.
    Reproduces the same 30-day month grid + blend_days tanh-free linear
    crossfade as interp_seasonal_wind, without Python branching on `day`
    (which would force a retrace per step and a fresh H2D copy of the
    blended field).
    """
    month_len = 30.0
    mpos = day % month_len
    mi = jnp.floor(day / month_len).astype(jnp.int32) % 12
    prev_i = (mi - 1) % 12
    nxt_i = (mi + 1) % 12
    half = blend_days / 2.0
    # Blend windows: w=0 -> pure current month; rises to 1 at the edges.
    w_after = jnp.clip((mpos - (month_len - half)) / blend_days, 0.0, 1.0)
    w_before = jnp.clip((half - mpos) / blend_days, 0.0, 1.0)
    cur = wind_stack[mi]
    nxt = wind_stack[nxt_i]
    prev = wind_stack[prev_i]
    # Blend: pure current month in the interior, linear crossfade to the
    # previous month across the leading edge and to the next month across
    # the trailing edge (weights are 0 outside the blend windows).
    tx = (1.0 - w_after - w_before) * cur[0] + w_after * nxt[0] + w_before * prev[0]
    ty = (1.0 - w_after - w_before) * cur[1] + w_after * nxt[1] + w_before * prev[1]
    return tx, ty


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=float, default=365.0)
    ap.add_argument("--dt", type=float, default=DT_DEFAULT)
    ap.add_argument("--mode-split", action="store_true",
                    help="baroclinic/barotropic mode split: free surface runs in "
                         "n_subcyc barotropic subcycles of --dt-bt each per "
                         "baroclinic step, lifting the external-gravity-wave CFL; "
                         "nu_h moves into the subcycle, kappa_conv is subcycled")
    ap.add_argument("--dt-bt", type=float, default=None,
                    help="barotropic subcycle dt [s] (mode split); rounded so "
                         "n_subcyc*dt_bt exactly fills the baroclinic dt. "
                         f"Default {DT_BT_DEFAULT:g} at 1°; with --resolution it "
                         "auto-scales as dx (external-gravity-wave CFL).")
    ap.add_argument("--nu-nsub", default=None,
                    help="nu_h subcycle count in split L half-steps: default "
                         "= legacy (n_subcyc=24); 'cfl' right-sizes from the "
                         "explicit-diffusion CFL (~10, 2.4x fewer laplacians); "
                         "int = verbatim")
    ap.add_argument("--dtype", default="float64", choices=["float64", "float32"],
                    help="compute dtype (default float64 = legacy bit-exact; "
                         "float32 is ~2-3x faster on Ada GPUs, needs revalidation "
                         "of drift criteria)")
    ap.add_argument("--use-scan", action="store_true",
                    help="run the barotropic subcycle as lax.scan (numerically "
                         "identical; smaller XLA graph, fewer host launches)")
    ap.add_argument("--wind-jit", action="store_true",
                    help="compile the seasonal-wind month blend into the XLA "
                         "graph (stack of 12 monthly fields on device, day as "
                         "a traced scalar). Numerically identical to the "
                         "Python blend; removes the per-step Python interp + "
                         "host-to-device copy (measured ~1.2-1.4x on split runs)")
    ap.add_argument("--lat-max", type=float, default=LAT_MAX_DEFAULT)
    ap.add_argument("--resolution", type=float, default=None,
                    help="horizontal grid spacing in degrees. With the default "
                         "--resolution-remap legacy this must be a multiple of "
                         "the 0.1° ETOPO source grid; with 'area' it may be "
                         "any positive value and is conservatively remapped. "
                         "When given, nx and ny are DERIVED from it and --ny "
                         "is ignored. Default: None = legacy 1° grid (--ny "
                         "decides). KEY: on finer grids the 1°-calibrated "
                         "--dt-bt/--nu-h/--nu-bi all violate their CFL and the "
                         "run diverges; this flag auto-scales them by dx^1/2/4 "
                         "respectively (see scaled_physics_for_resolution). "
                         "Pass any of those explicitly to override. At 1.0° "
                         "the scaling is a no-op. See "
                         "docs/resolution_cfl_limits.md.")
    ap.add_argument("--resolution-remap", choices=("legacy", "area"),
                    default="legacy",
                    help="'legacy' preserves integer 0.1° block averaging; "
                         "'area' uses conservative spherical-area overlap and "
                         "supports arbitrary positive resolutions")
    ap.add_argument("--ny", type=int, default=None,
                    help=f"meridional grid points at the default 1° resolution "
                         f"(default {NY_DEFAULT}); ignored when --resolution "
                         f"is given")
    ap.add_argument("--z-levels", default=None,
                    help="vertical grid override: comma-separated node depths "
                         "[m, negative down] e.g. '0,-5,-15,...' — replaces the "
                         "default 14-level grid (used by the E2 AMOC experiment)")
    ap.add_argument("--smooth-passes", type=int, default=SMOOTH_PASSES_DEFAULT)
    ap.add_argument("--min-depth", type=float, default=MIN_DEPTH_DEFAULT)
    ap.add_argument("--nu-h", type=float, default=None,
                    help=f"horizontal Laplacian viscosity [m^2/s]. Default "
                         f"{NU_H_DEFAULT:g} at 1°; with --resolution it "
                         f"auto-scales as dx^2 (explicit-diffusion CFL).")
    ap.add_argument("--nu-bi", type=float, default=None,
                    help=f"biharmonic hyperviscosity [m^4/s]. Default "
                         f"{NU_BI_DEFAULT:g} at 1°; with --resolution it "
                         f"auto-scales as dx^4 (biharmonic CFL).")
    ap.add_argument("--kappa-v", type=float, default=None,
                    help="vertical diffusivity override [m^2/s]; default keeps "
                         "PhysicsConfig (1e-5). Accelerated-spinup phase A uses "
                         "2e-4 (x20) to speed deep-ocean tracer adjustment")
    ap.add_argument("--kappa-conv", type=float, default=0.05,
                    help="convective vertical diffusivity override [m^2/s] "
                         "(default 0.05; CFL at the 5 m top layer caps "
                         "dt <= 0.5*dz^2/kappa_conv = 250 s at 0.05)")
    ap.add_argument("--bulk-lambda-mult", type=float, default=1.0,
                    help="multiplier on the bulk heat-flux transfer coefficient "
                         "(stronger surface restoring; phase-A distortion)")
    ap.add_argument("--lambda-bulk", type=float, default=LAMBDA_BULK_DEFAULT_G)
    ap.add_argument("--no-bulk-flux", action="store_true")
    ap.add_argument("--sss-restore-days", type=float, default=0.0,
                    help="surface salinity restoring timescale [days]; "
                         "0 = off. Haney relaxation of SSS to the WOA "
                         "surface climatology (v0.1 had NO surface salt "
                         "flux — SSS drifted ~-0.9 psu/kyr non-converging)")
    ap.add_argument("--sss-restore-zonal", action="store_true",
                    help="relax to the ZONAL MEAN of WOA SSS instead of the "
                         "full 2D field (keeps zonal SSS structure predicted; "
                         "analogous to the T_atm non-circularity rule)")
    ap.add_argument("--kappa-gm", type=float, default=0.0,
                    help="GM eddy diffusivity [m^2/s] (bolus transport); 0=off")
    ap.add_argument("--kappa-redi", type=float, default=0.0,
                    help="Redi isopycnal diffusivity [m^2/s]; 0=off")
    ap.add_argument("--gm-slope-max", type=float, default=0.01,
                    help="isopycnal slope limiter (dimensionless)")
    ap.add_argument("--freeze-adv-vel", action="store_true",
                    help="RK2 tracer stage-2: advect with the old velocity "
                         "instead of the freshly advected predictor (default off)")
    ap.add_argument("--conservative-kv", action="store_true",
                    help="use interface-flux vertical diffusion (default off)")
    ap.add_argument("--project-adv-vel", action="store_true",
                    help="project the RK2 stage-2 tracer velocity onto the "
                         "column-divergence-free space (removes the O(dt) "
                         "interior heat leak; default off)")
    ap.add_argument("--localize-conv", action="store_true",
                    help="gate convective adjustment PER-INTERFACE (mix only "
                         "across unstable interfaces) instead of the historical "
                         "column-wide mask that mixes the whole column whenever "
                         "any interface is unstable; default off")
    ap.add_argument("--monotone-adv", action="store_true",
                    help="use first-order donor-cell horizontal tracer fluxes "
                         "(vertical flux is already donor-cell). More diffusive "
                         "than centered flux, but conservative and monotone "
                         "under the combined tracer CFL; default off keeps the "
                         "historical centered path bit-exact")
    ap.add_argument("--sponge-days", type=float, default=SPONGE_DAYS_DEFAULT_G)
    ap.add_argument("--sponge-cells", type=int, default=0)
    ap.add_argument("--polar-cap-rows", type=int, default=POLAR_CAP_ROWS_DEFAULT)
    ap.add_argument("--polar-cap-taper", type=int, default=POLAR_CAP_TAPER_DEFAULT)
    ap.add_argument("--eta-relax-days", type=float, default=0.0,
                    help="semi-enclosed-sea SSH relaxation timescale [days]; "
                         "0 = off (Mediterranean strait artifact)")
    ap.add_argument("--eta-relax-box", type=float, nargs=4, default=None,
                    metavar=("LON0", "LON1", "LAT0", "LAT1"),
                    help="relaxation box in degrees E/N (core mask = 1 inside)")
    ap.add_argument("--eta-relax-buffer", type=float, default=1.0,
                    help="cos-taper buffer width [degrees] around the box")
    ap.add_argument("--snap-days", type=float, default=10.0)
    ap.add_argument("--seasonal-wind", action="store_true")
    ap.add_argument("--wind-year", type=int, default=2023)
    ap.add_argument("--wind-blend-days", type=float, default=5.0)
    ap.add_argument("--month", default="2023-01",
                    help="fixed wind month if not --seasonal-wind (YYYY-MM)")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--log-dir", default="logs")
    ap.add_argument("--init-from", default=None,
                    help="npz with T_init/S_init (precomputed WOA on this "
                         "grid); skips woa_data (offline nodes)")
    ap.add_argument("--save-3d", action="store_true")
    ap.add_argument("--save-3d-terms", action="store_true",
                    help="additionally save the per-term dT/dt decomposition "
                         "[adv, diff_h, diff_v, conv, gm, redi] at each 3D snap "
                         "(offline blowup attribution; needs --save-3d)")
    ap.add_argument("--max-steps", type=int, default=0,
                    help="hard cap on steps (0 = no cap); for short probes")
    ap.add_argument("--checkpoint-days", type=float, default=0.0,
                    help="save full state checkpoint every N days (0 = off); "
                         "enables --restart-from resume after container kill")
    ap.add_argument("--restart-from", default=None,
                    help="checkpoint npz to resume from (produced by "
                         "--checkpoint-days); integration continues from the "
                         "saved step, prior snapshots stay valid")
    args = ap.parse_args()

    tag = args.tag or f"g{int(args.days)}d"
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    out_npz = os.path.join(args.out_dir, f"global_{tag}.npz")
    out_log = os.path.join(args.log_dir, f"global_{tag}.log")
    sys.stdout = _Tee(out_log)
    print(f"# {time.strftime('%Y-%m-%d %H:%M:%S')}  {sys.executable}")
    print(f"# {' '.join(sys.argv)}")
    three_d_dir = None
    if args.save_3d:
        three_d_dir = os.path.join(args.out_dir, f"global_{tag}_3d")
        os.makedirs(three_d_dir, exist_ok=True)
    three_d_terms_dir = None
    if args.save_3d_terms:
        assert args.save_3d, "--save-3d-terms requires --save-3d"
        three_d_terms_dir = os.path.join(args.out_dir, f"global_{tag}_terms")
        os.makedirs(three_d_terms_dir, exist_ok=True)

    # ── Build global grid ──
    bathy = DEFAULT_CONFIG.bathymetry_file
    # Horizontal resolution: --resolution derives nx/ny from the ETOPO 0.1°
    # source; otherwise the legacy 1° grid with --ny meridional rows. The three
    # must stay mutually consistent (grid.py asserts the ETOPO read produces
    # exactly gc.nx x gc.ny), so never let the user set them independently.
    if args.resolution is not None:
        res = float(args.resolution)
        if res <= 0.0:
            ap.error(f"--resolution must be positive (got {res})")
        if args.resolution_remap == "legacy":
            step = res / 0.1
            if abs(step - round(step)) > 1e-9:
                ap.error(f"--resolution must be a multiple of the 0.1° ETOPO "
                         f"source grid when --resolution-remap=legacy "
                         f"(got {res}; use --resolution-remap=area)")
        nx, ny = global_grid_dims(res, args.lat_max,
                                  remap=args.resolution_remap)
        if nx < 4 or ny < 4:
            ap.error(f"--resolution {res} at lat_max={args.lat_max} gives a "
                     f"{nx}x{ny} grid; too coarse")
        gcfg_kwargs = {"lat_max": args.lat_max, "ny": ny, "nx": nx,
                       "resolution": res}
        if args.ny is not None:
            print(f"  NOTE: --ny {args.ny} ignored (--resolution {res} derives "
                  f"ny={ny})")
    else:
        ny = NY_DEFAULT if args.ny is None else args.ny
        gcfg_kwargs = {"lat_max": args.lat_max, "ny": ny}

    # ── Resolution-dependent physics auto-scaling ──
    # The 1°-calibrated dt_bt/nu_h/nu_bi all violate their CFL at finer grids
    # (measured: 0.5° diverges with any of them at production values). Scale
    # each by the power of dx its CFL demands, unless the user overrode it.
    # At --resolution 1.0 (or no --resolution) this is a no-op, preserving the
    # legacy defaults bit-for-bit.
    if args.resolution is not None:
        dt_bt, nu_h, nu_bi = scaled_physics_for_resolution(
            args.resolution, args.dt_bt, args.nu_h, args.nu_bi)
        overrides = []
        if args.dt_bt is None:
            overrides.append(f"dt_bt={dt_bt:.0f}s")
        if args.nu_h is None:
            overrides.append(f"nu_h={nu_h:.3g}")
        if args.nu_bi is None:
            overrides.append(f"nu_bi={nu_bi:.3g}")
        if overrides:
            print(f"  resolution {args.resolution:g}° auto-scales "
                  f"{', '.join(overrides)} (1° CFL values x dx^p)")
    else:
        dt_bt = DT_BT_DEFAULT if args.dt_bt is None else args.dt_bt
        nu_h = NU_H_DEFAULT if args.nu_h is None else args.nu_h
        nu_bi = NU_BI_DEFAULT if args.nu_bi is None else args.nu_bi
    if args.z_levels:
        zl = tuple(float(v) for v in args.z_levels.split(","))
        assert len(zl) >= 3 and zl[0] == 0.0 and all(
            zl[i] > zl[i + 1] for i in range(len(zl) - 1)), "bad --z-levels"
        gcfg_kwargs["z_levels"] = zl
        gcfg_kwargs["nz"] = len(zl)
        print(f"  vertical override: {len(zl)} levels, z={zl}")
    gcfg = replace(GlobalGridConfig(), **gcfg_kwargs)
    print(f"Building global FD grid (lat_max={args.lat_max}, "
          f"resolution={gcfg.resolution}°, ny={gcfg.ny}, "
          f"smooth={args.smooth_passes}, min_depth={args.min_depth})...")
    grid = make_global_grid(gcfg, bathy,
                            smooth_passes=args.smooth_passes,
                            min_depth=args.min_depth,
                            remap=args.resolution_remap)
    ocean = np.asarray(grid.ocean_mask, dtype=bool)
    dx_eq = float(grid.dx_2d[0, grid.ny // 2])
    print(f"  grid {grid.nx}x{grid.ny}x{grid.nz}, ocean {float(grid.wet_mask.mean()):.1%}, "
          f"dx_eq={dx_eq:.0f}m, lat[{grid.lat[0]:.1f},{grid.lat[-1]:.1f}]")

    physics = replace(PhysicsConfig(),
                      nu_h=nu_h, nu_bi=nu_bi, kappa_bi=nu_bi,
                      kappa_gm=args.kappa_gm, kappa_redi=args.kappa_redi,
                      kappa_v=(args.kappa_v if args.kappa_v is not None
                               else 1.0e-5),
                      kappa_conv=args.kappa_conv,
                      gm_slope_max=args.gm_slope_max)
    Q_heat = heat_flux_meridional(grid, Q0=50.0)

    # ── Initial fields (WOA2023, or precomputed npz via --init-from) ──
    if args.init_from:
        print(f"Loading precomputed init fields from {args.init_from}...")
        zf = np.load(args.init_from)
        T_init = np.array(zf["T_init"], dtype=np.float64)
        S_init = np.array(zf["S_init"], dtype=np.float64)
    else:
        print("Loading WOA2023 climatology for initial T/S...")
        T_init, S_init = get_initial_fields(grid)
    T_init = np.array(T_init)
    S_init = np.array(S_init)
    T_init_max = float(np.max(T_init))
    nan_init = int(np.isnan(T_init).sum() + np.isnan(S_init).sum())
    if nan_init > 0:
        print(f"  WARNING: {nan_init} NaN in initial fields (should be 0 after fill)")
    print(f"  T_init range=[{T_init.min():.2f}, {T_init_max:.2f}] C")

    # ── Wind forcing ──
    seasonal = args.seasonal_wind
    wind_months = None
    if seasonal:
        print(f"Loading 12 monthly NCEP wind snapshots (year {args.wind_year})...")
        try:
            wind_months = build_seasonal_wind_global(grid, year=args.wind_year)
            wind_src = f"seasonal cycle {args.wind_year} (12 monthly NCEP snapshots)"
        except Exception as e:
            print(f"  seasonal wind fetch failed ({e!r}); fallback fixed-month")
            seasonal = False
    if not seasonal:
        y, m = int(args.month[:4]), int(args.month[5:7])
        month_idx = (y - 1948) * 12 + (m - 1)
        tau_x, tau_y = real_wind_forcing(month_idx=month_idx, grid=grid)
        wind_src = f"fixed {args.month} (NCEP R1)"

    # ── Bulk air-sea heat flux ──
    # T_atm = zonally-uniform WOA SST meridional profile. Only the large-scale
    # meridional gradient is prescribed (the forced part); zonal SST structure
    # is genuinely PREDICTED by the model's advection/mixing. This is what
    # keeps the A1/A2 climatology comparison NON-CIRCULAR — the whole point
    # of the global non-periodic domain.
    T_sst = T_init[:, :, 0]
    lambda_bulk = 0.0 if args.no_bulk_flux else args.lambda_bulk * args.bulk_lambda_mult
    T_atm = air_temp_profile(grid, T_sst) if lambda_bulk > 0.0 else None
    if lambda_bulk > 0.0:
        print(f"  bulk air-sea flux: lambda={lambda_bulk:.1f} W/m^2/K, "
              f"T_atm=zonal WOA SST profile "
              f"({float(np.nanmin(T_atm)):.2f}..{float(np.nanmax(T_atm)):.2f} C)")

    # ── Surface salinity restoring target ──
    # Default: full 2D WOA SSS (real ocean SSS has strong zonal structure —
    # Atlantic 36.5 vs Pacific 34.5 — that a zonal-mean target would erase).
    # --sss-restore-zonal: relax to the ocean-only zonal mean, leaving zonal
    # SSS contrast for the model to predict (non-circular, like T_atm).
    S_sss = S_init[:, :, 0]
    S_ref_surf = None
    if args.sss_restore_days > 0.0:
        if args.sss_restore_zonal:
            prof = ocean_zonal_mean(grid, S_sss)
            S_ref_surf = np.broadcast_to(prof[None, :], (grid.nx, grid.ny)).copy()
            print(f"  SSS restoring: tau={args.sss_restore_days:g}d, "
                  f"target=ZONAL WOA SSS "
                  f"({float(np.nanmin(prof)):.2f}..{float(np.nanmax(prof)):.2f} psu)")
        else:
            S_ref_surf = S_sss
            print(f"  SSS restoring: tau={args.sss_restore_days:g}d, "
                  f"target=full 2D WOA SSS "
                  f"({float(np.nanmin(S_sss)):.2f}..{float(np.nanmax(S_sss)):.2f} psu)")

    # ── Build solver ──
    # Seasonal wind uses the DYNAMIC-FORCING path: step_dyn(state, tau_x,
    # tau_y, Q_heat) traces the 2D forcing as runtime arguments (single XLA
    # graph shared by all 12 monthly snapshots — no 12× memory). With
    # --seasonal-wind off, the baked-forcing step() is used, bit-exact to
    # all previous runs. (In seasonal mode the baked forcing is unused —
    # pass zeros as a placeholder since tau_x/tau_y are not loaded.)
    if not seasonal:
        forcing_baked = (tau_x, tau_y, Q_heat)
    else:
        forcing_baked = (np.zeros_like(Q_heat), np.zeros_like(Q_heat), Q_heat)
    _ret = make_solver_global(
        grid, physics, args.dt,
        forcing=forcing_baked,
        T_atm=T_atm, lambda_bulk=lambda_bulk,
        S_ref_surf=S_ref_surf, sss_restore_days=args.sss_restore_days,
        sponge_days=args.sponge_days, sponge_cells=args.sponge_cells,
        T_init=T_init, S_init=S_init,
        polar_cap_rows=args.polar_cap_rows,
        polar_cap_taper=args.polar_cap_taper, return_params=True,
        eta_relax_days=args.eta_relax_days, eta_relax_box=args.eta_relax_box,
        eta_relax_buffer=args.eta_relax_buffer,
        dynamic_forcing=seasonal,
        mode_split=args.mode_split, dt_bt=dt_bt,
        nu_nsub=(None if args.nu_nsub is None
                 else ('cfl' if args.nu_nsub == 'cfl' else int(args.nu_nsub))),
        dtype=args.dtype, use_scan=args.use_scan,
        freeze_adv_vel=args.freeze_adv_vel,
        conservative_kv=args.conservative_kv,
        project_adv_vel=args.project_adv_vel,
        localize_conv=args.localize_conv,
        monotone_adv=args.monotone_adv)
    if seasonal:
        step, init_state_global, _, _params, terms_fn, step_dyn = _ret
    else:
        step, init_state_global, _, _params, terms_fn = _ret
    # Runtime forcing must be cast to the compute dtype up front. Left in
    # float64 they promote every downstream tensor (a f32 state + f64 flux ->
    # f64), which under lax.scan is a hard carry-dtype error and under the
    # python loop is a silent mixed-precision run that forfeits the fp32
    # speedup. state_dtype is defined below for the checkpoint logic; hoisted
    # here so the forcing path can use it.
    _fdtype = jnp.float32 if args.dtype == "float32" else jnp.float64
    if seasonal:
        Q_heat_2d = jnp.array(Q_heat, dtype=_fdtype)
        if args.wind_jit:
            # (12, 2, nx, ny) on device; blend happens inside the graph.
            wind_stack = jnp.array(np.stack([np.stack(m) for m in wind_months]),
                                   dtype=_fdtype)
            def do_step(state, month_day):
                tx, ty = interp_seasonal_wind_jit(
                    wind_stack, month_day, blend_days=args.wind_blend_days)
                return step_dyn(state, tx, ty, Q_heat_2d)
        else:
            def do_step(state, month_day):
                tx, ty = interp_seasonal_wind(wind_months, month_day,
                                              blend_days=args.wind_blend_days)
                return step_dyn(state, jnp.array(tx, dtype=_fdtype),
                                jnp.array(ty, dtype=_fdtype), Q_heat_2d)
    else:
        def do_step(state, month_day):
            return step(state)

    state = init_state_global(T_init=jnp.array(T_init), S_init=jnp.array(S_init))

    # Compute dtype for checkpoint round-trips: fp32 runs keep the device
    # state in fp32 (I/O casts to float64 at the npz boundary, so checkpoint
    # files stay grid-version-agnostic and readable by float64 runs).
    state_dtype = _fdtype

    n_total = int(round(args.days * 86400.0 / args.dt))
    if args.max_steps > 0:
        n_total = min(n_total, args.max_steps)
    n_snap = max(1, int(round(args.snap_days * 86400.0 / args.dt)))

    # ── Checkpoint resume: restore u,v,T,S,eta and fast-forward the snapshot
    # counters so existing snap_*.npy files and npz table rows stay aligned.
    # Seasonal-wind phase is a pure function of the step index, so restoring
    # (u,v,T,S,eta) at a snap boundary is bit-consistent with an unbroken run.
    ckpt_path = os.path.join(args.out_dir, f"ckpt_{tag}.npz")
    start_step = 0
    n_3d_snaps = 0
    if args.restart_from:
        ck = np.load(args.restart_from, allow_pickle=True)
        assert int(ck["grid_nx"]) == grid.nx and int(ck["grid_ny"]) == grid.ny \
            and int(ck["grid_nz"]) == grid.nz, "grid size mismatch with checkpoint"
        state = JaxStateG(
            u=jnp.array(ck["u"], dtype=state_dtype),
            v=jnp.array(ck["v"], dtype=state_dtype),
            T=jnp.array(ck["T"], dtype=state_dtype),
            S=jnp.array(ck["S"], dtype=state_dtype),
            eta=jnp.array(ck["eta"], dtype=state_dtype))
        start_step = int(ck["cur_step"])
        assert start_step % n_snap == 0, "checkpoint must land on a snap boundary"
        # Re-align snapshot bookkeeping with what already exists on disk.
        n_prev_snaps = start_step // n_snap
        n_3d_snaps = n_prev_snaps
        print(f"RESUME from {args.restart_from}: step {start_step} "
              f"(day {start_step * args.dt / 86400.0:.1f}), "
              f"{n_prev_snaps} prior snapshots kept")
    if args.checkpoint_days > 0:
        n_ckpt = max(1, int(round(args.checkpoint_days * 86400.0 / args.dt)))
        if args.restart_from:
            assert n_ckpt % n_snap == 0 or args.checkpoint_days <= args.snap_days, \
                "checkpoint cadence must align with snap cadence on resume"

    # ── Header ──
    header = []
    header.append("=" * 70)
    header.append(f"GLOBAL FD LONG INTEGRATION ({args.days:.0f} days)")
    header.append("=" * 70)
    header.append(f"grid: {grid.nx}x{grid.ny}x{grid.nz}  "
                  f"res={gcfg.resolution:g}°  dx_eq={dx_eq:.0f}m  "
                  f"lon[{grid.lon[0]:.1f},{grid.lon[-1]:.1f}]E "
                  f"lat[{grid.lat[0]:.1f},{grid.lat[-1]:.1f}]N")
    header.append(f"dt={args.dt:.0f}s  steps={n_total}  snap every {n_snap} steps "
                  f"({args.snap_days:.0f}d)")
    if args.mode_split:
        ns = _params.n_subcyc
        _nnu = _params.nu_nsub if _params.nu_nsub is not None else ns
        header.append(f"MODE SPLIT: baroclinic dt={args.dt:.0f}s, barotropic "
                      f"subcycle {ns} x {args.dt / ns:.1f}s "
                      f"(--dt-bt {dt_bt:.0f}s), conv_nsub={_params.conv_nsub}, "
                      f"nu_nsub={_nnu}, scan={'ON' if _params.use_scan else 'py'}")
    header.append(f"physics: nu_h={physics.nu_h:g}  nu_bi={physics.nu_bi:g}  "
                  f"kappa_conv={physics.kappa_conv}  kappa_v={physics.kappa_v:g}")
    if args.freeze_adv_vel or args.conservative_kv or args.project_adv_vel \
            or args.localize_conv or args.monotone_adv:
        header.append(f"RK2 flags: freeze_adv_vel={args.freeze_adv_vel}  "
                      f"conservative_kv={args.conservative_kv}  "
                      f"project_adv_vel={args.project_adv_vel}  "
                      f"localize_conv={args.localize_conv}  "
                      f"monotone_adv={args.monotone_adv}")
    if args.dtype != "float64":
        header.append(f"DTYPE: {args.dtype} (compute; I/O stays float64)")
    if args.bulk_lambda_mult != 1.0:
        header.append(f"distorted physics: bulk-lambda x{args.bulk_lambda_mult:g} "
                      f"(accelerated-spinup phase A)")
    if physics.kappa_gm > 0 or physics.kappa_redi > 0:
        header.append(f"sub-grid closure: kappa_gm={physics.kappa_gm:g} m^2/s  "
                      f"kappa_redi={physics.kappa_redi:g} m^2/s  "
                      f"gm_slope_max={physics.gm_slope_max:g}")
    else:
        header.append("sub-grid closure: NONE (kappa_gm=0, kappa_redi=0)")
    header.append(f"bulk_flux={lambda_bulk:g} W/m^2/K"
                  + (" (T_atm=zonal WOA, non-circular)" if lambda_bulk > 0.0 else " (off)"))
    if args.sss_restore_days > 0.0:
        header.append(f"sss_restore: tau={args.sss_restore_days:g}d  "
                      f"target={'zonal WOA SSS' if args.sss_restore_zonal else 'full 2D WOA SSS'}")
    else:
        header.append("sss_restore: NONE (no surface salt flux)")
    header.append("wall: no-flux N/S (v=0 at boundary rows, mirror-ghost dy)")
    header.append(f"wind: {wind_src}")
    if seasonal and args.wind_blend_days > 0:
        header.append(f"wind blend: {args.wind_blend_days:g}d linear window at month boundaries")
    elif seasonal:
        header.append("wind blend: NONE (step at month boundaries)")
    if args.sponge_days > 0 and args.sponge_cells > 0:
        header.append(f"sponge: {args.sponge_cells}-cell band, tau={args.sponge_days:g}d")
    else:
        header.append("sponge: NONE")
    if args.polar_cap_rows > 0:
        header.append(f"polar cap: {args.polar_cap_rows} rows + {args.polar_cap_taper}-row cos^2 taper")
    else:
        header.append("polar cap: NONE")
    if args.eta_relax_days > 0 and args.eta_relax_box is not None:
        b = args.eta_relax_box
        header.append(f"eta_relax: tau={args.eta_relax_days:g}d  "
                      f"box=[{b[0]:.1f},{b[1]:.1f}]E x [{b[2]:.1f},{b[3]:.1f}]N  "
                      f"buffer={args.eta_relax_buffer:g}deg (mass-conserving)")
    else:
        header.append("eta_relax: NONE")
    header.append(f"init: T_init_max={T_init_max:.2f}C  "
                  f"amplitude_cap={T_init_max + AMPLITUDE_CAP_C:.2f}C")
    header.append(f"criteria: max|u|<{MAX_U_BOUND}  drift_tol={DRIFT_TOL_C}C  "
                  f"watchdog |eta|>{ETA_BLOWUP_M}m")
    header.append("")
    header.append(f"{'day':>6} {'step':>7} {'max|u|':>9} {'max|T|':>8} "
                  f"{'max|eta|':>9} {'SSH_std':>9} {'KE':>12} {'NaN':>6}")
    header.append("-" * 78)
    for line in header:
        print(line)

    # ── Integration loop ──
    snap_days = []
    snap_maxu = []
    snap_maxT = []
    snap_maxeta = []
    snap_sshstd = []
    snap_ke = []
    snap_eta = []
    snap_T_top = []
    maxT_history = []
    max_u_peak = 0.0
    diverged_at = None
    diverge_reason = ""
    cur = 0
    t0 = time.time()

    def snapshot(cur_step):
        nonlocal n_3d_snaps
        day = cur_step * args.dt / 86400.0
        # fp32 runs: cast every snapshot to float64 so the npz/npy artifact
        # layout is identical to legacy runs (all analyses read float64).
        f64 = (lambda a: np.asarray(a, dtype=np.float64)) \
            if args.dtype == "float32" else np.asarray
        eta = f64(state.eta)
        maxu = float(np.max(np.abs(f64(state.u))))
        maxT = float(np.max(np.abs(f64(state.T))))
        maxeta = float(np.nanmax(np.abs(eta))) if np.isfinite(eta).any() else float('nan')
        sshstd = float(np.std(eta[ocean])) if ocean.any() else float('nan')
        ke = total_kinetic_energy(state, ocean)
        nan = int(np.sum(~np.isfinite(f64(state.u))))
        snap_days.append(day)
        snap_maxu.append(maxu)
        snap_maxT.append(maxT)
        snap_maxeta.append(maxeta)
        snap_sshstd.append(sshstd)
        snap_ke.append(ke)
        snap_eta.append(eta.copy())
        snap_T_top.append(f64(state.T[:, :, 0]).copy())
        if args.save_3d:
            # 4-field snapshot: T,u,v,S each (nx,ny,nz). eta is NOT stacked —
            # it is 2D while these are 3D, and it is already saved per-frame in
            # the npz `eta` table (snap_eta) which all analyses read.
            snap3d = np.stack([
                f64(state.T).copy(),
                f64(state.u).copy(),
                f64(state.v).copy(),
                f64(state.S).copy(),
            ], axis=0)
            np.save(os.path.join(three_d_dir, f"snap_{n_3d_snaps:05d}.npy"), snap3d)
            del snap3d
            if args.save_3d_terms:
                # Per-term dT/dt decomposition at this snap: [adv, diff_h,
                # diff_v, conv, gm, redi], each (nx,ny,nz). Term sum equals
                # the N-step tracer tendency (no bulk/sponge/heat — those are
                # surface-only; the deep runaway attribution only needs these).
                tstack = np.asarray(terms_fn(state), dtype=np.float64) \
                    if args.dtype == "float32" else np.asarray(terms_fn(state))
                np.save(os.path.join(three_d_terms_dir,
                                     f"terms_{n_3d_snaps:05d}.npy"), tstack)
                del tstack
            n_3d_snaps += 1
            # Give the XLA async dispatch queue a chance to drain and free its
            # scratch buffers before the next 7200-step block (the 4.3 GB of
            # live snapshot arrays + ComfyUI sharing 31 GB of RAM pushed a
            # 380 MB device->host transfer into a transient OOM otherwise).
            import gc
            gc.collect()
        maxT_history.append(maxT)
        print(f"{day:7.1f} {cur_step:8d} {maxu:9.3f} {maxT:8.3f} "
              f"{maxeta:9.3f} {sshstd:9.4f} {ke:12.4e} {nan:6d}", flush=True)
        return maxu, maxT, maxeta, nan

    # Resume path: record the boundary row too, so the table has the day-0-of-
    # this-era state and the 3D snap index stays aligned with day (i*snap_days).
    maxu, maxT, maxeta, nan = snapshot(0) if not args.restart_from \
        else snapshot(start_step)

    cur = start_step

    while cur < n_total:
        take = min(n_snap, n_total - cur)
        for _ in range(take):
            state = do_step(state, cur * args.dt / 86400.0)
            cur += 1
        maxu, maxT, maxeta, nan = snapshot(cur)
        max_u_peak = max(max_u_peak, maxu)
        if (args.checkpoint_days > 0 and cur % n_ckpt == 0
                and cur < n_total and cur > start_step):
            np.savez_compressed(
                ckpt_path,
                u=np.asarray(state.u, dtype=np.float64),
                v=np.asarray(state.v, dtype=np.float64),
                T=np.asarray(state.T, dtype=np.float64),
                S=np.asarray(state.S, dtype=np.float64),
                eta=np.asarray(state.eta, dtype=np.float64), cur_step=cur,
                grid_nx=grid.nx, grid_ny=grid.ny, grid_nz=grid.nz,
                n_3d_snaps=n_3d_snaps)
            print(f"  [ckpt] saved {ckpt_path} at step {cur}", flush=True)
        if not state_is_finite(state):
            diverged_at = cur * args.dt / 86400.0
            diverge_reason = "non-finite field (NaN/Inf)"
            break
        if maxu > MAX_U_BOUND:
            diverged_at = cur * args.dt / 86400.0
            diverge_reason = f"max|u| {maxu:.2f} > bound {MAX_U_BOUND}"
            break
        if np.isfinite(maxeta) and maxeta > ETA_BLOWUP_M:
            diverged_at = cur * args.dt / 86400.0
            diverge_reason = f"|eta| {maxeta:.2f} > watchdog {ETA_BLOWUP_M}"
            break

    wall = time.time() - t0

    # ── Verdict ──
    monotonic_drift = False
    amplitude_bounded = True
    if len(maxT_history) >= 4 and diverged_at is None:
        half = len(maxT_history) // 2
        first_half_max = max(maxT_history[:half])
        final_quarter = maxT_history[-max(1, len(maxT_history) // 4):]
        if final_quarter and max(final_quarter) > first_half_max + DRIFT_TOL_C:
            monotonic_drift = True
        if final_quarter and max(final_quarter) > T_init_max + AMPLITUDE_CAP_C:
            amplitude_bounded = False

    if diverged_at is not None:
        verdict = "FAIL_BLOWUP"
    elif monotonic_drift or not amplitude_bounded:
        verdict = "FAIL_DRIFT"
    else:
        verdict = "PASS"

    print("=" * 78)
    print(f"VERDICT: {verdict}  (wall {wall/60:.1f} min, {cur} steps, "
          f"max_u_peak={max_u_peak:.3f})")
    if diverged_at is not None:
        print(f"  diverged at day {diverged_at:.1f}: {diverge_reason}")
    print(f"  monotonic_drift={monotonic_drift}  amplitude_bounded={amplitude_bounded}")
    print(f"  final max|u|={maxu:.3f}  max|T|={maxT:.3f}  max|eta|={maxeta:.3f}")

    # ── Save ──
    config_dict = {
        'lat_max': args.lat_max, 'ny': grid.ny, 'nx': grid.nx, 'nz': grid.nz,
        'resolution': float(gcfg.resolution),
        'dt': args.dt, 'nu_h': physics.nu_h, 'nu_bi': physics.nu_bi,
        'lambda_bulk': lambda_bulk, 'seasonal_wind': seasonal,
        'wind_blend_days': args.wind_blend_days, 'sponge_days': args.sponge_days,
        'sponge_cells': args.sponge_cells, 'polar_cap_rows': args.polar_cap_rows,
        'polar_cap_taper': args.polar_cap_taper,
        'eta_relax_days': args.eta_relax_days, 'eta_relax_box': args.eta_relax_box,
        'eta_relax_buffer': args.eta_relax_buffer,
        'smooth_passes': args.smooth_passes, 'min_depth': args.min_depth,
        'kappa_v': physics.kappa_v, 'bulk_lambda_mult': args.bulk_lambda_mult,
        'freeze_adv_vel': args.freeze_adv_vel,
        'conservative_kv': args.conservative_kv,
        'project_adv_vel': args.project_adv_vel,
        'localize_conv': args.localize_conv,
        'sss_restore_days': args.sss_restore_days,
        'sss_restore_zonal': bool(args.sss_restore_zonal),
    }
    np.savez_compressed(out_npz,
                        days=np.array(snap_days),
                        max_u=np.array(snap_maxu),
                        max_T=np.array(snap_maxT),
                        max_eta=np.array(snap_maxeta),
                        ssh_std=np.array(snap_sshstd),
                        ke=np.array(snap_ke),
                        eta=np.array(snap_eta),
                        T_top=np.array(snap_T_top),
                        T_init=T_init,
                        S_init=S_init,
                        wet_mask=np.asarray(grid.wet_mask),
                        lat=np.asarray(grid.lat),
                        lon=np.asarray(grid.lon),
                        z=np.asarray(grid.z),
                        verdict=verdict,
                        diverged_at=(diverged_at if diverged_at is not None else -1.0),
                        monotonic_drift=monotonic_drift,
                        amplitude_bounded=amplitude_bounded,
                        max_u_peak=max_u_peak,
                        n_3d_snaps=n_3d_snaps,
                        three_d_dir=(three_d_dir or ""),
                        config=str(config_dict))
    print(f"  saved {out_npz}")
    if three_d_dir:
        print(f"  3D snapshots: {n_3d_snaps} files in {three_d_dir}")


if __name__ == "__main__":
    main()
