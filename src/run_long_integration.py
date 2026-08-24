"""
Long-integration driver — stage 1/2 of the long-run + climatology plan
(docs/long_run_climatology_plan_zh.md).

Purpose (boss directive): run a long integration, watch for blowup; if it
blows up, diagnose the cause; if it stays stable, compute a climatology and
compare to real climatology / statistics (stage 3, separate script).

This script implements STAGES 1 and 2:
  - Stage 1: 90-day seasonal stability probe (dt=150s, WOA stratified init,
    real NCEP monthly wind, Haney SST restoring tau=5d, divergence watchdog).
  - Stage 2: 365-day annual integration with seasonal-cycle wind (12 monthly
    snapshots cycled), only run after stage 1 passes.

Pre-registered pass/fail criteria (do NOT move the bar after running):
  PASS:
    - no NaN/Inf in u, v, T, S, eta over the whole run
    - max|u| stays < MAX_U_BOUND (10 m/s) throughout
    - monotonic_drift False: final-quarter max|T| does NOT climb > DRIFT_TOL_C
      above the first-half max (convergence, not accumulation)
    - amplitude_bounded True: final-quarter max|T| < init_max + AMPLITUDE_CAP_C
  FAIL (blowup): NaN/Inf or max|u| breach or watchdog (|eta|>ETA_BLOWUP_M) fires
  FAIL (drift): monotonic_drift True but not blown

On FAIL the script records the divergence point + a hotspot snapshot so the
root-cause analysis can proceed (locate -> classify seam vs real instability ->
find mechanism -> fix at test layer only, never production PhysicsConfig).

Output:
  - results/long_run_<tag>.npz   (snapshots: eta/T/u maxima, KE, SSH_std)
  - logs/long_run_<tag>.log      (full monitor trace + verdict)

Usage:
  python src/run_long_integration.py --stage 1 --days 90 --tag s1_90d
  python src/run_long_integration.py --stage 2 --days 365 --tag s2_365d --seasonal-wind
"""
import sys
import os
import time
import argparse
from dataclasses import replace

# Unbuffered stdout so progress is visible immediately when redirected to a
# log file (Python defaults to block-buffering when stdout is not a tty).
try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
import numpy as np

from config import DEFAULT_CONFIG
from grid import make_grid
from jax_solver import make_solver, make_forcing
from forcing import heat_flux_meridional
from wind_reanalysis import real_wind_forcing
from woa_data import get_initial_fields

# ── Pre-registered criteria (frozen; do not tune to a result) ──────────
DT_DEFAULT = 150.0          # s — spin_evo verified stable for 90d at dt=150
MAX_U_BOUND = 10.0          # m/s, below the DEGRADED (>10) threshold
DRIFT_TOL_C = 2.0           # C, second-half climb tolerance (monotonic drift)
AMPLITUDE_CAP_C = 12.0      # C, absolute ceiling above init max (true runaway)
ETA_BLOWUP_M = 3.0          # m, divergence watchdog (real eddies << 1.5 m)
RESTORE_DAYS_DEFAULT = 5.0  # Haney SST restoring (3rd-blowup-fix anchor)
NU_BI_DEFAULT = 1e12        # production default (do not change in experiments)


def state_is_finite(state):
    """True if every prognostic field is finite (catches divergence early)."""
    for f in (state.u, state.v, state.T, state.S, state.eta):
        a = np.asarray(f)
        if not np.isfinite(a).all():
            return False
    return True


def total_kinetic_energy(state, ocean_mask):
    """Domain-integrated KE per unit mass: 0.5 * sum(u^2 + v^2) over ocean."""
    u = np.asarray(state.u)
    v = np.asarray(state.v)
    # Sum over all levels; ocean_mask is 2D (nx,ny), broadcast over z.
    ke = 0.5 * np.sum((u ** 2 + v ** 2) * ocean_mask[:, :, None])
    return float(ke)


def build_seasonal_wind(grid, year=2023, taper_cells=8):
    """Load 12 monthly NCEP wind-stress snapshots for a seasonal cycle.

    Returns a list of (tau_x, tau_y) tuples, one per month (Jan..Dec), each
    y-tapered. Used by stage 2 to cycle the forcing through the seasons.
    """
    from forcing import taper_2d_y
    months = []
    for m in range(1, 13):
        month_idx = (year - 1948) * 12 + (m - 1)
        tau_x, tau_y = real_wind_forcing(month_idx=month_idx, grid=grid,
                                         taper_cells=taper_cells)
        months.append((tau_x, tau_y))
    return months


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", type=int, default=1, choices=[1, 2],
                    help="1 = 90d seasonal probe; 2 = 365d annual w/ seasonal wind")
    ap.add_argument("--days", type=float, default=None,
                    help="integration length in days (default: 90 stage1, 365 stage2)")
    ap.add_argument("--dt", type=float, default=DT_DEFAULT)
    ap.add_argument("--nu-bi", type=float, default=NU_BI_DEFAULT)
    ap.add_argument("--restore-days", type=float, default=RESTORE_DAYS_DEFAULT)
    ap.add_argument("--snap-days", type=float, default=10.0,
                    help="snapshot/monitor interval in days")
    ap.add_argument("--seasonal-wind", action="store_true",
                    help="cycle 12 monthly NCEP wind snapshots (stage 2)")
    ap.add_argument("--wind-year", type=int, default=2023)
    ap.add_argument("--month", default="2023-01",
                    help="fixed wind month for stage 1 (YYYY-MM)")
    ap.add_argument("--tag", default=None, help="output file tag")
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--log-dir", default="logs")
    ap.add_argument("--save-3d", action="store_true",
                    help="save 3D T + u,v snapshots (larger npz; needed for "
                         "full zonal-mean profile + Sverdrup in stage 3)")
    args = ap.parse_args()

    if args.days is None:
        args.days = 90.0 if args.stage == 1 else 365.0
    if args.stage == 2 and not args.seasonal_wind:
        # stage 2 defaults to seasonal wind; allow override but warn
        print("  NOTE: stage 2 without --seasonal-wind uses fixed-month wind")
    tag = args.tag or f"s{args.stage}_{int(args.days)}d"
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    out_npz = os.path.join(args.out_dir, f"long_run_{tag}.npz")
    out_log = os.path.join(args.log_dir, f"long_run_{tag}.log")

    grid = make_grid(DEFAULT_CONFIG.grid, DEFAULT_CONFIG.bathymetry_file)
    ocean = np.asarray(grid.ocean_mask, dtype=bool)
    dx_m = grid.dx
    physics = replace(DEFAULT_CONFIG.physics, nu_bi=args.nu_bi, kappa_bi=args.nu_bi)
    Q_heat = heat_flux_meridional(grid, Q0=50.0)

    # ── Initial fields ──
    print("Loading WOA2023 climatology for initial T/S...")
    T_init, S_init = get_initial_fields(grid)
    T_init_max = float(np.max(T_init))
    nan_init = int(np.isnan(T_init).sum() + np.isnan(S_init).sum())
    if nan_init > 0:
        print(f"  WARNING: {nan_init} NaN in initial fields (should be 0 after fill)")
    print(f"  T_init range=[{T_init.min():.2f}, {T_init_max:.2f}] C  "
          f"(init max used for amplitude cap)")

    # ── Wind forcing ──
    seasonal = args.seasonal_wind
    if seasonal:
        print(f"Loading 12 monthly NCEP wind snapshots (year {args.wind_year})...")
        try:
            wind_months = build_seasonal_wind(grid, year=args.wind_year)
            wind_src = f"seasonal cycle {args.wind_year} (12 monthly snapshots)"
        except Exception as e:
            print(f"  seasonal wind fetch failed ({e!r}); fallback fixed-month")
            seasonal = False
    if not seasonal:
        y, m = int(args.month[:4]), int(args.month[5:7])
        month_idx = (y - 1948) * 12 + (m - 1)
        tau_x, tau_y = real_wind_forcing(month_idx=month_idx, grid=grid)
        wind_src = f"fixed {args.month} (NCEP R1)"

    # ── Build solver ──
    # A SINGLE compiled step graph is used for the whole run, whether the
    # wind is fixed or seasonal. For seasonal wind the 12 monthly snapshots
    # become JaxForcing data objects swapped at runtime (step(state, jf)),
    # not 12 separate JIT closures — the earlier 12-closure design multiplied
    # XLA memory 12× and silently crashed the process mid-run (see
    # docs/long_run_climatology_report_zh.md, stage 2). Physically identical:
    # verified to round-off against the baked-in path.
    T_sst = T_init[:, :, 0]
    if seasonal:
        step, init_state, _ = make_solver(grid, physics, args.dt,
                                          forcing=None,
                                          T_sst=T_sst,
                                          tau_restore_days=args.restore_days)
        wind_forcings = [make_forcing(grid, tx, ty, Q_heat)
                         for (tx, ty) in wind_months]
    else:
        step, init_state, _ = make_solver(grid, physics, args.dt,
                                          forcing=(tau_x, tau_y, Q_heat),
                                          T_sst=T_sst,
                                          tau_restore_days=args.restore_days)
    state = init_state(T_init=jnp.array(T_init), S_init=jnp.array(S_init))

    n_total = int(round(args.days * 86400.0 / args.dt))
    n_snap = max(1, int(round(args.snap_days * 86400.0 / args.dt)))

    # ── Header ──
    header = []
    header.append("=" * 70)
    header.append(f"LONG INTEGRATION — stage {args.stage} ({args.days:.0f} days)")
    header.append("=" * 70)
    header.append(f"grid: {grid.nx}x{grid.ny}x{grid.nz}  dx={dx_m:.0f}m  "
                  f"domain lon[{grid.lon[0]:.1f},{grid.lon[-1]:.1f}]E "
                  f"lat[{grid.lat[0]:.1f},{grid.lat[-1]:.1f}]N")
    header.append(f"dt={args.dt:.0f}s  steps={n_total}  snap every {n_snap} steps "
                  f"({args.snap_days:.0f}d)")
    header.append(f"physics: nu_bi={args.nu_bi:g}  kappa_conv={physics.kappa_conv}  "
                  f"restore={args.restore_days:g}d")
    header.append(f"wind: {wind_src}")
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

    # ── Integration loop with monitoring ──
    snap_days = []
    snap_maxu = []
    snap_maxT = []
    snap_maxeta = []
    snap_sshstd = []
    snap_ke = []
    snap_eta = []          # full eta field at each snapshot (for climatology)
    snap_T_top = []        # surface T field at each snapshot
    snap_T3d = []          # full 3D T field (only if --save-3d)
    snap_U = []            # full 3D u field (only if --save-3d)
    snap_V = []            # full 3D v field (only if --save-3d)
    maxT_history = []      # for monotonic_drift criterion
    max_u_peak = 0.0
    diverged_at = None
    diverge_reason = ""
    cur = 0
    t0 = time.time()

    # initial snapshot
    def snapshot(cur_step):
        day = cur_step * args.dt / 86400.0
        eta = np.asarray(state.eta)
        maxu = float(np.max(np.abs(np.asarray(state.u))))
        maxT = float(np.max(np.abs(np.asarray(state.T))))
        maxeta = float(np.nanmax(np.abs(eta))) if np.isfinite(eta).any() else float('nan')
        sshstd = float(np.std(eta[ocean])) if ocean.any() else float('nan')
        ke = total_kinetic_energy(state, ocean)
        nan = int(np.sum(~np.isfinite(np.asarray(state.u))))
        snap_days.append(day)
        snap_maxu.append(maxu)
        snap_maxT.append(maxT)
        snap_maxeta.append(maxeta)
        snap_sshstd.append(sshstd)
        snap_ke.append(ke)
        snap_eta.append(eta.copy())
        snap_T_top.append(np.asarray(state.T[:, :, 0]).copy())
        if args.save_3d:
            snap_T3d.append(np.asarray(state.T).copy())
            snap_U.append(np.asarray(state.u).copy())
            snap_V.append(np.asarray(state.v).copy())
        maxT_history.append(maxT)
        print(f"{day:7.1f} {cur_step:8d} {maxu:9.3f} {maxT:8.3f} "
              f"{maxeta:9.3f} {sshstd:9.4f} {ke:12.4e} {nan:6d}")
        return maxu, maxeta, nan

    maxu, maxeta, nan = snapshot(0)

    while cur < n_total:
        take = min(n_snap, n_total - cur)
        for _ in range(take):
            if seasonal:
                # cycle monthly wind: which month does the current step fall in?
                day_idx = int((cur * args.dt) / 86400.0)
                mi = (day_idx // 30) % 12
                state = step(state, wind_forcings[mi])
            else:
                state = step(state)
            cur += 1
        maxu, maxeta, nan = snapshot(cur)
        max_u_peak = max(max_u_peak, maxu)
        # ── watchdog / divergence checks ──
        if not state_is_finite(state):
            diverged_at = cur * args.dt / 86400.0
            diverge_reason = "non-finite field (NaN/Inf)"
            print(f"  *** DIVERGED: {diverge_reason} at day {diverged_at:.1f} (step {cur})")
            break
        if maxeta > ETA_BLOWUP_M:
            diverged_at = cur * args.dt / 86400.0
            diverge_reason = f"|eta| max {maxeta:.2f}m > {ETA_BLOWUP_M}m (blow-up precursor)"
            print(f"  *** DIVERGED: {diverge_reason} at day {diverged_at:.1f} (step {cur})")
            break
        if nan > 0:
            diverged_at = cur * args.dt / 86400.0
            diverge_reason = f"{nan} NaN in u at day {diverged_at:.1f}"
            print(f"  *** DIVERGED: {diverge_reason}")
            break

    wall = time.time() - t0
    print(f"\n  wall time {wall:.0f}s ({wall/3600:.2f}h)")

    # ── Verdict (pre-registered criteria) ──
    has_nan = not state_is_finite(state)
    max_u_final = float(np.max(np.abs(np.asarray(state.u))))
    max_T_final = float(np.max(np.abs(np.asarray(state.T))))
    n_hist = len(maxT_history)
    first_half_maxT = max(maxT_history[:max(1, n_hist // 2)])
    final_quarter_maxT = max(maxT_history[max(1, 3 * n_hist // 4):])
    monotonic_drift = (final_quarter_maxT - first_half_maxT) > DRIFT_TOL_C
    amplitude_bounded = final_quarter_maxT < (T_init_max + AMPLITUDE_CAP_C)
    drift_up = (final_quarter_maxT - T_init_max) > DRIFT_TOL_C  # informational

    print()
    print("=" * 70)
    print(f"  diverged_at      = {diverged_at}")
    print(f"  diverge_reason   = {diverge_reason or 'none'}")
    print(f"  max|u| peak      = {max_u_peak:.3f} m/s (bound {MAX_U_BOUND})")
    print(f"  max|u| final     = {max_u_final:.3f} m/s")
    print(f"  max|T| final     = {max_T_final:.3f} C (init max {T_init_max:.2f})")
    print(f"  maxT first-half  = {first_half_maxT:.3f} C")
    print(f"  maxT final-quarter = {final_quarter_maxT:.3f} C")
    print(f"  drift above init ({DRIFT_TOL_C}C tol) = {drift_up}  (informational)")
    print(f"  monotonic_drift  = {monotonic_drift}  (second-half climb > {DRIFT_TOL_C}C)")
    print(f"  amplitude_bounded= {amplitude_bounded}  "
          f"(final < {T_init_max + AMPLITUDE_CAP_C:.2f}C)")

    if diverged_at is not None:
        verdict = "FAIL_BLOWUP"
    elif has_nan or max_u_final >= MAX_U_BOUND or not amplitude_bounded:
        verdict = "FAIL_BLOWUP"
    elif monotonic_drift:
        verdict = "FAIL_DRIFT"
    else:
        verdict = "PASS"
    print(f"  VERDICT          = {verdict}")
    print("=" * 70)

    # ── If diverged, emit a hotspot snapshot for root-cause analysis ──
    hotspot = {}
    if diverged_at is not None:
        T_arr = np.asarray(state.T)
        eta_arr = np.asarray(state.eta)
        # where is the temperature extreme? T_arr is (nx, ny, nz)
        flat_idx = int(np.nanargmax(np.abs(T_arr)))
        ix, iy, iz = np.unravel_index(flat_idx, T_arr.shape)
        hotspot = dict(
            T_max_loc=(int(ix), int(iy), int(iz)),
            T_max_val=float(np.nanmax(np.abs(T_arr))),
            eta_max_loc=tuple(int(a) for a in np.unravel_index(
                int(np.nanargmax(np.abs(eta_arr))), eta_arr.shape)),
            lon_at_Tmax=float(grid.lon[ix]),
            lat_at_Tmax=float(grid.lat[iy]),
            z_at_Tmax=float(grid.z[iz]),
        )
        print(f"  HOTSPOT: T max at (i={ix},j={iy},k={iz}) "
              f"({grid.lon[ix]:.1f}E,{grid.lat[iy]:.1f}N,z={grid.z[iz]:.0f}m) "
              f"= {hotspot['T_max_val']:.3f}C")

    # ── Save ──
    save_dict = dict(
             days=np.array(snap_days),
             max_u=np.array(snap_maxu),
             max_T=np.array(snap_maxT),
             max_eta=np.array(snap_maxeta),
             ssh_std=np.array(snap_sshstd),
             ke=np.array(snap_ke),
             eta=np.stack([np.asarray(x) for x in snap_eta], 0),
             T_top=np.stack([np.asarray(x) for x in snap_T_top], 0),
             diverged_at=np.float64(diverged_at if diverged_at is not None else -1.0),
             diverge_reason=np.array(diverge_reason or ""),
             verdict=np.array(verdict),
             first_half_maxT=np.float64(first_half_maxT),
             final_quarter_maxT=np.float64(final_quarter_maxT),
             monotonic_drift=np.array(monotonic_drift),
             amplitude_bounded=np.array(amplitude_bounded),
             max_u_peak=np.float64(max_u_peak),
             T_init_max=np.float64(T_init_max),
             hotspot_T_max_loc=np.array(hotspot.get('T_max_loc', (-1, -1, -1))),
             hotspot_T_max_val=np.float64(hotspot.get('T_max_val', 0.0)),
             hotspot_lon=np.float64(hotspot.get('lon_at_Tmax', 0.0)),
             hotspot_lat=np.float64(hotspot.get('lat_at_Tmax', 0.0)),
             hotspot_z=np.float64(hotspot.get('z_at_Tmax', 0.0)),
             config=dict(stage=args.stage, days=args.days, dt=args.dt,
                         nu_bi=args.nu_bi, restore_days=args.restore_days,
                         snap_days=args.snap_days, wind=wind_src))
    if args.save_3d and snap_T3d:
        save_dict['T3d'] = np.stack([np.asarray(x) for x in snap_T3d], 0)
        save_dict['U'] = np.stack([np.asarray(x) for x in snap_U], 0)
        save_dict['V'] = np.stack([np.asarray(x) for x in snap_V], 0)
    np.savez(out_npz, **save_dict)
    print(f"  saved {out_npz}")

    # also tee the verdict to the log by re-printing (caller redirects stdout)
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
