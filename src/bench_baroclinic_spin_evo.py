"""Spin-up evolution diagnostic: does baroclinic instability develop over time?

Step 5 diagnostic (bench_baroclinic_step5_diag.py) found the model SSH has
essentially NO mesoscale power below ~460 km — energy sits at the ~550-777 km
gyre scale and never cascades to eddy scales at 9-km / 30-day / wind+Q0 config.
Hypothesis H1 (the cheapest, most diagnostic one): 30 days is far too short a
spin-up for baroclinic instability to develop from a smooth WOA initial state
(real eddy e-folding is weeks-to-months).

This script tests H1 by integrating the SAME stable config for a long time and
recording the SSH field (and its mesoscale band-power fraction) at regular
checkpoints. If the eddy-band fraction rises from ~0 toward a finite value over
60-180 days, H1 is supported and a longer run is justified before re-scoring
real-data correlation. If it stays flat at ~0, H1 is falsified and the problem
is forcing/dissipation instead.

Band-power metric uses the radial power spectrum at each checkpoint:
    mesoscale fraction = P(50-400 km) / P(total),
where wavelengths < 460 km are the true eddy band the Step 5 diagnostic found
to be empty. (50-600 km is contaminated by the gyre skirt, so 50-400 km is the
cleaner eddy selection.)

Usage:
    python src/bench_baroclinic_spin_evo.py --spinup-days 90 --snap-days 10 \
        --out results/spin_evo_90d.npz
"""
import sys, os, time, argparse
import numpy as np
from dataclasses import replace

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp

from config import DEFAULT_CONFIG
from grid import make_grid
from jax_solver import make_solver
from forcing import heat_flux_meridional
from wind_reanalysis import real_wind_forcing, daily_wind_forcing
from woa_data import get_initial_fields

DT = 300.0
# True mesoscale eddy band (km) — the band the Step 5 diagnostic found empty.
EDDY_LMIN, EDDY_LMAX = 50.0, 400.0
# Blow-up precursor: normalized SSH amplitude beyond which we stop and record
# a DIVERGED verdict instead of reporting a misleading NaN table. Real mesoscale
# eddies stay well under ~1.5 m in this domain; exponential divergence explodes
# past it in a few steps (day-50 max was 0.998 m before the day-60 NaN).
ETA_BLOWUP_M = 3.0


def state_is_finite(state, eta):
    """True if every model field is finite (catches divergence that starts in
    u/v/T/S and only later poisons eta)."""
    fields = (state.u, state.v, state.T, state.S, state.eta)
    for f in fields:
        a = np.asarray(f)
        if not np.isfinite(a).all():
            return False
    return np.isfinite(eta).all()


def radial_power_frac(field, l_lo_km, l_hi_km, dx_m, ocean):
    """Fraction of power in wavelengths [l_lo, l_hi] km via radial spectrum."""
    f = np.nan_to_num(field, nan=0.0).astype(np.float64)
    f = f - f[ocean].mean()
    nx, ny = f.shape
    wx = np.hanning(nx); wy = np.hanning(ny)
    ft = f * wx[:, None] * wy[None, :]
    F = np.fft.fftshift(np.fft.fft2(ft))
    PSD = np.abs(F) ** 2
    kx = np.fft.fftshift(np.fft.fftfreq(nx, d=dx_m))   # [1/m]
    ky = np.fft.fftshift(np.fft.fftfreq(ny, d=dx_m))
    KX, KY = np.meshgrid(kx, ky, indexing='ij')
    K = np.sqrt(KX ** 2 + KY ** 2)                      # [1/m]
    L = 2 * np.pi / np.where(K > 0, K, np.nan)          # [m] wavelength
    Lkm = L / 1e3
    band = (Lkm >= l_lo_km) & (Lkm <= l_hi_km)
    PSDw = PSD * wx[:, None] * wy[None, :]
    tot = np.nansum(PSDw[np.isfinite(L)])
    sel = np.nansum(PSDw[np.isfinite(L) & band])
    return float(sel / tot) if tot > 0 else float('nan')


def bandlimited_noise_2d(nx, ny, rng, l_lo_km, l_hi_km, dx_m):
    """White-noise field band-passed to wavelengths [l_lo, l_hi] km, RMS=1.

    Used to seed mesoscale instability exactly where the Step 5 diagnostic
    found the model empty (the 50-400 km eddy band). Returns an (nx, ny)
    field with zero mean and unit RMS confined to the requested band.
    """
    n = rng.standard_normal((nx, ny))
    n = n - n.mean()
    N = np.fft.fftshift(np.fft.fft2(n))
    kx = np.fft.fftshift(np.fft.fftfreq(nx, d=dx_m))    # [1/m]
    ky = np.fft.fftshift(np.fft.fftfreq(ny, d=dx_m))
    KX, KY = np.meshgrid(kx, ky, indexing='ij')
    K = np.sqrt(KX ** 2 + KY ** 2)
    Lkm = (2 * np.pi / np.where(K > 0, K, 1.0)) / 1e3
    mask = (Lkm >= l_lo_km) & (Lkm <= l_hi_km)
    N = N * mask
    out = np.real(np.fft.ifft2(np.fft.ifftshift(N)))
    out = out - out.mean()
    rms = np.sqrt((out ** 2).mean())
    return out / rms if rms > 0 else np.zeros((nx, ny))


def inject_perturbation(state, physics, grid, amp_degC, seed, dx_m):
    """Add a surface-intensified, band-limited mesoscale T/S perturbation to
    the initial state. amp_degC<=0 returns the state unchanged. Perturbing the
    density field seeds baroclinic instability, discriminating 'no instability
    seed' from 'over-dissipation' in one run."""
    if amp_degC <= 0:
        return state
    rng = np.random.default_rng(seed)
    nx, ny, nz = grid.nx, grid.ny, grid.nz
    bl = bandlimited_noise_2d(nx, ny, rng, EDDY_LMIN, EDDY_LMAX, dx_m)
    # Surface-intensified vertical structure (real eddies decay over the
    # pycnocline, ~a quarter of the model depth).
    vert = np.exp(-np.arange(nz, dtype=np.float64) / (nz / 4.0))
    Tp = amp_degC * bl[:, :, None] * vert[None, None, :]
    Sp = 0.1 * amp_degC * bl[:, :, None] * vert[None, None, :]
    return state._replace(
        T=state.T + jnp.array(Tp),
        S=state.S + jnp.array(Sp),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--month", default="2023-01")
    ap.add_argument("--spinup-days", type=float, default=90.0)
    ap.add_argument("--snap-days", type=float, default=10.0)
    ap.add_argument("--dt", type=float, default=300.0)
    ap.add_argument("--nu-bi", type=float, default=1e12)
    ap.add_argument("--tau-restore-days", type=float, default=30.0)
    ap.add_argument("--pert-amp", type=float, default=0.0,
                    help="mesoscale T perturb (degC) to seed instability; 0 = off")
    ap.add_argument("--pert-seed", type=int, default=0)
    ap.add_argument("--daily-wind", action="store_true",
                    help="use daily-climatology wind instead of monthly mean")
    ap.add_argument("--out", default="results/spin_evo_90d.npz")
    args = ap.parse_args()

    grid = make_grid(DEFAULT_CONFIG.grid, DEFAULT_CONFIG.bathymetry_file)
    ocean = np.asarray(grid.ocean_mask, dtype=bool)
    dx_m = grid.dx

    y, m = int(args.month[:4]), int(args.month[5:7])
    month_idx = (y - 1948) * 12 + (m - 1)
    Q_heat = heat_flux_meridional(grid, Q0=50.0)

    tau_x = tau_y = tau_x_d = tau_y_d = None
    use_daily_steps = False
    if args.daily_wind:
        try:
            tau_x_d, tau_y_d = daily_wind_forcing(month=m, grid=grid)
            wind_src = f"real daily clim {args.month}"
            use_daily_steps = True
        except Exception as e:
            print(f"  daily wind fetch failed ({e!r}); fallback monthly")
            tau_x, tau_y = real_wind_forcing(month_idx=month_idx, grid=grid)
            wind_src = f"real {args.month}"
    else:
        try:
            tau_x, tau_y = real_wind_forcing(month_idx=month_idx, grid=grid)
            wind_src = f"real {args.month}"
        except Exception as e:
            print(f"  real wind fetch failed ({e!r}); fallback Stommel gyre")
            from forcing import wind_stress_gyre
            tau_x, tau_y = wind_stress_gyre(grid, tau0=0.1)
            wind_src = "stommel"

    physics = replace(DEFAULT_CONFIG.physics, nu_bi=args.nu_bi,
                      kappa_bi=args.nu_bi)
    T_init, S_init = get_initial_fields(grid)
    DT_eff = args.dt

    if use_daily_steps:
        # Pre-build 31 daily step functions. Each is JIT-compiled for its own
        # wind snapshot; startup cost is 31 x compile but keeps the core solver
        # unchanged. We only keep the step closures; init_state comes from the
        # first make_solver call.
        print("  building 31 daily-wind step functions (this may take a minute)...")
        steps_d = []
        init_state = None
        for d in range(tau_x_d.shape[0]):
            step_d, init_state_d, _ = make_solver(grid, physics, DT_eff,
                                                  forcing=(tau_x_d[d], tau_y_d[d], Q_heat),
                                                  T_sst=T_init[:, :, 0],
                                                  tau_restore_days=args.tau_restore_days)
            if init_state is None:
                init_state = init_state_d
            steps_d.append(step_d)
        step = steps_d[0]
    else:
        step, init_state, _ = make_solver(grid, physics, DT_eff,
                                          forcing=(tau_x, tau_y, Q_heat),
                                          T_sst=T_init[:, :, 0],
                                          tau_restore_days=args.tau_restore_days)
    state = init_state(T_init=jnp.array(T_init), S_init=jnp.array(S_init))
    state = inject_perturbation(state, physics, grid, args.pert_amp,
                                args.pert_seed, dx_m)

    n_total = int(round(args.spinup_days * 86400.0 / DT_eff))
    n_snap = max(1, int(round(args.snap_days * 86400.0 / DT_eff)))

    print("=" * 68)
    print("SPIN-UP EVOLUTION — does baroclinic instability develop?")
    print("=" * 68)
    print(f"grid {grid.nx}x{grid.ny}x{grid.nz} nu_bi={args.nu_bi:g} "
          f"restore={args.tau_restore_days:g}d wind={wind_src} dt={DT_eff:.0f}s")
    print(f"eddy band {EDDY_LMIN:.0f}-{EDDY_LMAX:.0f} km "
          f"(spinup={args.spinup_days:.0f}d, snap={args.snap_days:.0f}d)")
    if args.pert_amp > 0:
        print(f"perturbation: mesoscale T amp {args.pert_amp:.2f} degC, "
              f"seed {args.pert_seed}")
    print(f"{'day':>6} {'SSH_std(m)':>10} {'SSH_max(m)':>10} "
          f"{'eddy_frac(50-400km)':>21}")
    print("-" * 74)
    snap_days = []; snap_eta = []; snap_std = []; snap_frac = []
    t0 = time.time()
    cur = 0                      # absolute step count completed
    diverged_at = None
    while cur < n_total:
        take = min(n_snap, n_total - cur)
        for _ in range(take):
            if use_daily_steps:
                day_idx = int((cur * DT_eff) / 86400.0) % tau_x_d.shape[0]
                state = steps_d[day_idx](state)
            else:
                state = step(state)
        cur += take
        day = cur * DT_eff / 86400.0
        eta = np.asarray(state.eta)
        std = float(np.std(eta[ocean]))
        etamax = float(np.nanmax(np.abs(eta))) if np.isfinite(eta).any() else float('nan')
        frac = radial_power_frac(eta, EDDY_LMIN, EDDY_LMAX, dx_m, ocean)
        snap_days.append(day); snap_eta.append(eta)
        snap_std.append(std); snap_frac.append(frac)
        print(f"{day:7.1f} {std:10.4f} {etamax:10.3f} {frac:21.4f}")
        # Watchdog: stop cleanly on divergence / blow-up precursor instead of
        # recording NaN rows that mislead the verdict.
        if not state_is_finite(state, eta):
            diverged_at = float(day)
            print(f"  *** DIVERGED: non-finite field at day {day:.1f} (step {cur})")
            break
        if etamax > ETA_BLOWUP_M:
            diverged_at = float(day)
            print(f"  *** DIVERGED: |eta| max {etamax:.2f} m > {ETA_BLOWUP_M} m "
                  f"at day {day:.1f} (blow-up precursor)")
            break
    wall = time.time() - t0
    print(f"\n  wall time {wall:.0f}s")
    np.savez(args.out,
             days=np.array(snap_days), std=np.array(snap_std),
             eddy_frac=np.array(snap_frac),
             eta=np.stack([np.asarray(x) for x in snap_eta], 0),
             diverged_at=np.float64(diverged_at),
             config=dict(spinup_days=args.spinup_days, snap_days=args.snap_days,
                         dt=DT_eff, nu_bi=args.nu_bi,
                         tau_restore_days=args.tau_restore_days, month=args.month,
                         pert_amp=args.pert_amp, pert_seed=args.pert_seed))
    print(f"  saved {args.out}")

    # ── Verdict ──
    f0 = snap_frac[0]; f1 = snap_frac[-1]
    print("\n  VERDICT:")
    if diverged_at is not None:
        print(f"  DIVERGED at day {diverged_at:.0f} (before clean end): "
              f"eddy frac {f0*100:.1f}% -> {f1*100:.1f}%. Numeric stability "
              f"ceiling hit before instability could be confirmed — reduce dt "
              f"or damp before judging H1.")
    elif f1 - f0 > 0.05:
        print(f"  eddy frac {f0*100:.1f}% -> {f1*100:.1f}% over {snap_days[-1]:.0f}d: "
              "INSTABILITY DEVELOPING — longer spin-up is justified; re-score "
              "real-data after eddies equilibrate.")
    elif f1 < 0.05:
        print(f"  eddy frac stays ~{f1*100:.1f}%: H1 FALSIFIED — spin-up length "
              "is not the blocker; problem is forcing/dissipation/initial-noise.")
    else:
        print(f"  eddy frac {f0*100:.1f}% -> {f1*100:.1f}%: WEAK/INCONCLUSIVE — "
              "inspect full curve.")


if __name__ == "__main__":
    main()
