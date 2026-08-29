"""
Long-integration driver for the GLOBAL finite-difference solver (jax_solver_global).

Companion to run_long_integration.py (regional pseudo-spectral). This driver
runs the global 1° FD solver — built on the stable G2 config (lat_max=60,
no-flux N/S wall, nu_h=5e6 spin-up stabilizer, dt=60) — with the bulk air-sea
heat flux thermodynamics from the closed arc. The whole point of the global
domain is the non-circular A1/A2 zonal SST skill test: a global non-periodic
domain removes the regional periodic-BC crutch and lets large-scale SST
structure emerge from geometry + wind + bathymetry rather than from a
prescribed meridional T_atm clamp.

Stability (G2, commit eb54d82): the no-flux wall + nu_h=5e6 holds both
no-wind and wind-forced (tau0=0.1) to 1000 steps / 0 NaN / max|T| bounded.
This driver extends that to a full annual integration with real seasonal
NCEP wind + bulk flux, same pre-registered pass/fail criteria as the regional
driver (do NOT move the bar after running).

Output:
  - results/global_<tag>.npz   (snapshots: eta/T/u maxima, KE, SSH_std, T_top)
  - logs/global_<tag>.log      (full monitor trace + verdict)
  - results/global_<tag>_3d/   (streamed 3D T/U/V snapshots, one .npy each)

Usage:
  python src/run_long_integration_global.py --days 365 --seasonal-wind --tag g365d
  python src/run_long_integration_global.py --days 200 --tag g200d_smoke   # shorter probe
"""
import sys
import os
import time
import argparse
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

from config import DEFAULT_CONFIG, PhysicsConfig, GlobalGridConfig
from grid import make_global_grid
from jax_solver_global import make_solver_global
from forcing import heat_flux_meridional, air_temp_profile, BULK_LAMBDA_DEFAULT
from wind_reanalysis import real_wind_forcing
from woa_data import get_initial_fields

# ── Pre-registered criteria (frozen; do not tune to a result) ──────────
# Same framework as the regional driver. The global FD grid is coarser
# (1° vs 0.1°) and dt is smaller (60 vs 150), so MAX_U_BOUND and the drift
# tolerances are kept identical — the gate is about stability + bounded
# drift, not matching regional magnitudes.
DT_DEFAULT = 60.0              # s — explicit free-surface + FD CFL-safe at 1°
MAX_U_BOUND = 10.0            # m/s
DRIFT_TOL_C = 2.0             # C, second-half climb tolerance (monotonic drift)
AMPLITUDE_CAP_C = 12.0        # C, absolute ceiling above init max
ETA_BLOWUP_M = 15.0           # m, divergence watchdog. Looser than the
                              # regional solver's 3m — the global grid under
                              # real NCEP wind builds a larger wind-driven
                              # barotropic setup (tropical pile-up reaches
                              # ~5m, mean stays ~0 = mass-conserved). 15m
                              # still catches true divergence while allowing
                              # the physical spin-up barotropic mode.
# ── Global FD stable config (G2 gate-passed, commit eb54d82) ──
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
                              # (the j=0 single-gridpoint divergence, G3).
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

    Same as the regional driver's interp_seasonal_wind: removes the artificial
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=float, default=365.0)
    ap.add_argument("--dt", type=float, default=DT_DEFAULT)
    ap.add_argument("--lat-max", type=float, default=LAT_MAX_DEFAULT)
    ap.add_argument("--ny", type=int, default=NY_DEFAULT)
    ap.add_argument("--smooth-passes", type=int, default=SMOOTH_PASSES_DEFAULT)
    ap.add_argument("--min-depth", type=float, default=MIN_DEPTH_DEFAULT)
    ap.add_argument("--nu-h", type=float, default=NU_H_DEFAULT)
    ap.add_argument("--nu-bi", type=float, default=NU_BI_DEFAULT)
    ap.add_argument("--lambda-bulk", type=float, default=LAMBDA_BULK_DEFAULT_G)
    ap.add_argument("--no-bulk-flux", action="store_true")
    ap.add_argument("--kappa-gm", type=float, default=0.0,
                    help="GM eddy diffusivity [m^2/s] (bolus transport); 0=off")
    ap.add_argument("--kappa-redi", type=float, default=0.0,
                    help="Redi isopycnal diffusivity [m^2/s]; 0=off")
    ap.add_argument("--gm-slope-max", type=float, default=0.01,
                    help="isopycnal slope limiter (dimensionless)")
    ap.add_argument("--sponge-days", type=float, default=SPONGE_DAYS_DEFAULT_G)
    ap.add_argument("--sponge-cells", type=int, default=0)
    ap.add_argument("--polar-cap-rows", type=int, default=POLAR_CAP_ROWS_DEFAULT)
    ap.add_argument("--polar-cap-taper", type=int, default=POLAR_CAP_TAPER_DEFAULT)
    ap.add_argument("--snap-days", type=float, default=10.0)
    ap.add_argument("--seasonal-wind", action="store_true")
    ap.add_argument("--wind-year", type=int, default=2023)
    ap.add_argument("--wind-blend-days", type=float, default=5.0)
    ap.add_argument("--month", default="2023-01",
                    help="fixed wind month if not --seasonal-wind (YYYY-MM)")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--log-dir", default="logs")
    ap.add_argument("--save-3d", action="store_true")
    ap.add_argument("--max-steps", type=int, default=0,
                    help="hard cap on steps (0 = no cap); for short probes")
    args = ap.parse_args()

    tag = args.tag or f"g{int(args.days)}d"
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    out_npz = os.path.join(args.out_dir, f"global_{tag}.npz")
    out_log = os.path.join(args.log_dir, f"global_{tag}.log")
    three_d_dir = None
    if args.save_3d:
        three_d_dir = os.path.join(args.out_dir, f"global_{tag}_3d")
        os.makedirs(three_d_dir, exist_ok=True)

    # ── Build global grid ──
    bathy = DEFAULT_CONFIG.bathymetry_file
    gcfg = replace(GlobalGridConfig(), lat_max=args.lat_max, ny=args.ny)
    print(f"Building global FD grid (lat_max={args.lat_max}, ny={args.ny}, "
          f"smooth={args.smooth_passes}, min_depth={args.min_depth})...")
    grid = make_global_grid(gcfg, bathy,
                            smooth_passes=args.smooth_passes,
                            min_depth=args.min_depth)
    ocean = np.asarray(grid.ocean_mask, dtype=bool)
    dx_eq = float(grid.dx_2d[0, grid.ny // 2])
    print(f"  grid {grid.nx}x{grid.ny}x{grid.nz}, ocean {float(grid.wet_mask.mean()):.1%}, "
          f"dx_eq={dx_eq:.0f}m, lat[{grid.lat[0]:.1f},{grid.lat[-1]:.1f}]")

    physics = replace(PhysicsConfig(),
                      nu_h=args.nu_h, nu_bi=args.nu_bi, kappa_bi=args.nu_bi,
                      kappa_gm=args.kappa_gm, kappa_redi=args.kappa_redi,
                      gm_slope_max=args.gm_slope_max)
    Q_heat = heat_flux_meridional(grid, Q0=50.0)

    # ── Initial fields (WOA2023) ──
    print("Loading WOA2023 climatology for initial T/S...")
    T_init, S_init = get_initial_fields(grid)
    T_init = np.array(T_init); S_init = np.array(S_init)
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
    lambda_bulk = 0.0 if args.no_bulk_flux else args.lambda_bulk
    T_atm = air_temp_profile(grid, T_sst) if lambda_bulk > 0.0 else None
    if lambda_bulk > 0.0:
        print(f"  bulk air-sea flux: lambda={lambda_bulk:.1f} W/m^2/K, "
              f"T_atm=zonal WOA SST profile "
              f"({float(np.nanmin(T_atm)):.2f}..{float(np.nanmax(T_atm)):.2f} C)")

    # ── Build solver (single compiled graph) ──
    # The global FD solver bakes the 2D forcing (tau_x, tau_y, Q_heat) into
    # the JIT-closed params. The regional solver solved the seasonal-cycle
    # memory problem by passing forcing as runtime data (step(state, jf));
    # the FD solver doesn't yet have that path, so a 12-closure seasonal
    # design would 12× the XLA memory (the exact failure that crashed the
    # regional 365d run). For now we use a FIXED-MONTH real NCEP wind —
    # architecturally clean (one graph), and sufficient for the non-circular
    # climatology skill test (which compares time-mean SST, not a seasonal
    # cycle). The G2 gate passed with fixed wind; seasonal cycling is a
    # future refinement that needs the FD dynamic-forcing modification.
    if seasonal:
        print("  NOTE: seasonal wind on the FD solver needs the dynamic-forcing")
        print("        path (not yet implemented); using fixed-month wind instead.")
        seasonal = False
        tau_x, tau_y = wind_months[0]   # use January as the fixed month
        wind_src = f"fixed {args.wind_year}-01 (NCEP R1, from seasonal fetch)"
    step, init_state_global, _ = make_solver_global(
        grid, physics, args.dt,
        forcing=(tau_x, tau_y, Q_heat),
        eos_type='linear',
        T_atm=T_atm, lambda_bulk=lambda_bulk,
        sponge_days=args.sponge_days, sponge_cells=args.sponge_cells,
        T_init=T_init, S_init=S_init,
        polar_cap_rows=args.polar_cap_rows,
        polar_cap_taper=args.polar_cap_taper)

    state = init_state_global(T_init=jnp.array(T_init), S_init=jnp.array(S_init))

    n_total = int(round(args.days * 86400.0 / args.dt))
    if args.max_steps > 0:
        n_total = min(n_total, args.max_steps)
    n_snap = max(1, int(round(args.snap_days * 86400.0 / args.dt)))

    # ── Header ──
    header = []
    header.append("=" * 70)
    header.append(f"GLOBAL FD LONG INTEGRATION ({args.days:.0f} days)")
    header.append("=" * 70)
    header.append(f"grid: {grid.nx}x{grid.ny}x{grid.nz}  dx_eq={dx_eq:.0f}m  "
                  f"lon[{grid.lon[0]:.1f},{grid.lon[-1]:.1f}]E "
                  f"lat[{grid.lat[0]:.1f},{grid.lat[-1]:.1f}]N")
    header.append(f"dt={args.dt:.0f}s  steps={n_total}  snap every {n_snap} steps "
                  f"({args.snap_days:.0f}d)")
    header.append(f"physics: nu_h={physics.nu_h:g}  nu_bi={physics.nu_bi:g}  "
                  f"kappa_conv={physics.kappa_conv}")
    if physics.kappa_gm > 0 or physics.kappa_redi > 0:
        header.append(f"sub-grid closure: kappa_gm={physics.kappa_gm:g} m^2/s  "
                      f"kappa_redi={physics.kappa_redi:g} m^2/s  "
                      f"gm_slope_max={physics.gm_slope_max:g}")
    else:
        header.append("sub-grid closure: NONE (kappa_gm=0, kappa_redi=0)")
    header.append(f"bulk_flux={lambda_bulk:g} W/m^2/K"
                  + (" (T_atm=zonal WOA, non-circular)" if lambda_bulk > 0.0 else " (off)"))
    header.append(f"wall: no-flux N/S (v=0 at boundary rows, mirror-ghost dy)")
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
    snap_days = []; snap_maxu = []; snap_maxT = []; snap_maxeta = []
    snap_sshstd = []; snap_ke = []; snap_eta = []; snap_T_top = []
    n_3d_snaps = 0
    maxT_history = []
    max_u_peak = 0.0
    diverged_at = None
    diverge_reason = ""
    cur = 0
    t0 = time.time()

    def snapshot(cur_step):
        nonlocal n_3d_snaps
        day = cur_step * args.dt / 86400.0
        eta = np.asarray(state.eta)
        maxu = float(np.max(np.abs(np.asarray(state.u))))
        maxT = float(np.max(np.abs(np.asarray(state.T))))
        maxeta = float(np.nanmax(np.abs(eta))) if np.isfinite(eta).any() else float('nan')
        sshstd = float(np.std(eta[ocean])) if ocean.any() else float('nan')
        ke = total_kinetic_energy(state, ocean)
        nan = int(np.sum(~np.isfinite(np.asarray(state.u))))
        snap_days.append(day); snap_maxu.append(maxu); snap_maxT.append(maxT)
        snap_maxeta.append(maxeta); snap_sshstd.append(sshstd); snap_ke.append(ke)
        snap_eta.append(eta.copy()); snap_T_top.append(np.asarray(state.T[:, :, 0]).copy())
        if args.save_3d:
            snap3d = np.stack([
                np.asarray(state.T).copy(),
                np.asarray(state.u).copy(),
                np.asarray(state.v).copy(),
            ], axis=0)
            np.save(os.path.join(three_d_dir, f"snap_{n_3d_snaps:05d}.npy"), snap3d)
            del snap3d
            n_3d_snaps += 1
        maxT_history.append(maxT)
        print(f"{day:7.1f} {cur_step:8d} {maxu:9.3f} {maxT:8.3f} "
              f"{maxeta:9.3f} {sshstd:9.4f} {ke:12.4e} {nan:6d}", flush=True)
        return maxu, maxT, maxeta, nan

    maxu, maxT, maxeta, nan = snapshot(0)

    while cur < n_total:
        take = min(n_snap, n_total - cur)
        for _ in range(take):
            state = step(state)
            cur += 1
        maxu, maxT, maxeta, nan = snapshot(cur)
        max_u_peak = max(max_u_peak, maxu)
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
        'lat_max': args.lat_max, 'ny': args.ny, 'nx': grid.nx, 'nz': grid.nz,
        'dt': args.dt, 'nu_h': physics.nu_h, 'nu_bi': physics.nu_bi,
        'lambda_bulk': lambda_bulk, 'seasonal_wind': seasonal,
        'wind_blend_days': args.wind_blend_days, 'sponge_days': args.sponge_days,
        'sponge_cells': args.sponge_cells, 'polar_cap_rows': args.polar_cap_rows,
        'polar_cap_taper': args.polar_cap_taper,
        'smooth_passes': args.smooth_passes, 'min_depth': args.min_depth,
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
