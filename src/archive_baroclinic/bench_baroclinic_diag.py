"""Spatial-structure diagnostic for the Step 2 stable run (nu_bi=1e12 + restore).

Step 2 established that nu_bi=1e12 (with surface-T restoring) is the only
STABLE stratified configuration, reaching an SSH-variance plateau of ~0.49 m.
But a plateau on its own does not tell us whether that variance is:
  (a) mesoscale eddy variance (good — resolvable turbulence, basis for a
      defensible real-data SSH metric), or
  (b) large-scale quasi-steady gyre 'setup' (the wind-driven SSH tilt, low
      wavenumber, NOT eddies).

This diagnostic integrates to the stable state, saves the final SSH field,
and decomposes its variance by horizontal wavenumber into eddy vs setup bands.

Usage:
    python src/bench_baroclinic_diag.py --spinup-days 10 --nu-bi 1e12 --tau-restore-days 30
"""
import sys, os, time, argparse
import numpy as np
from dataclasses import replace

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import jax
import jax.numpy as jnp

from config import DEFAULT_CONFIG
from grid import make_grid
from jax_solver import make_solver
from forcing import wind_stress_gyre, heat_flux_meridional
from woa_data import get_initial_fields

DT = 300.0
# Mesoscale eddy band (wavelength lambda in km): eddies of ~50-300 km are the
# mesoscale signal. Setup is lambda > ~600 km (basin/gyre scale).
EDDY_LMIN, EDDY_LMAX = 50.0, 600.0     # km
SETUP_LMIN = 600.0                      # km (larger than this = gyre setup)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spinup-days", type=float, default=10.0)
    ap.add_argument("--nu-bi", type=float, default=1e12)
    ap.add_argument("--tau-restore-days", type=float, default=30.0)
    args = ap.parse_args()

    grid = make_grid(DEFAULT_CONFIG.grid, DEFAULT_CONFIG.bathymetry_file)
    ocean = np.asarray(grid.ocean_mask, dtype=bool)
    dx = grid.dx  # [m]
    dy = grid.dy
    nx, ny = grid.nx, grid.ny
    physics = replace(DEFAULT_CONFIG.physics, nu_bi=args.nu_bi,
                      kappa_bi=args.nu_bi)
    print("=" * 64)
    print("SPATIAL-STRUCTURE DIAGNOSTIC (stable stratified SSH)")
    print("=" * 64)
    print(f"grid {nx}x{ny}  dx={dx:.0f}m dy={dy:.0f}m  "
          f"nu_bi={args.nu_bi:g} restore={args.tau_restore_days:g}d")

    T_init, S_init = get_initial_fields(grid)
    tau_x, tau_y = wind_stress_gyre(grid, tau0=0.1)
    Q_heat = heat_flux_meridional(grid, Q0=50.0)
    T_sst = T_init[:, :, 0]
    step, init_state, _ = make_solver(grid, physics, DT,
                                      forcing=(tau_x, tau_y, Q_heat),
                                      T_sst=T_sst,
                                      tau_restore_days=args.tau_restore_days)
    state = init_state(T_init=jnp.array(T_init), S_init=jnp.array(S_init))
    n_spin = int(round(args.spinup_days * 86400.0 / DT))
    t0 = time.time()
    for n in range(n_spin):
        state = step(state)
    wall = time.time() - t0
    eta = np.asarray(state.eta)
    print(f"  integrated {args.spinup_days:.0f}d ({wall:.0f}s)")
    print(f"  SSH ocean std = {np.std(eta[ocean]):.4f} m")

    # ── 2D spectral variance decomposition ──
    # Use periodic FFT. To avoid land/NaN contaminating the spectrum, NaN-fill
    # land then taper; report relative band contributions (qualitative).
    eta_f = np.nan_to_num(eta, nan=0.0).astype(np.float64)
    # remove domain mean
    eta_f = eta_f - eta_f[ocean].mean()
    # Hann taper to reduce leakage
    wx = np.hanning(nx)
    wy = np.hanning(ny)
    eta_t = eta_f * wx[:, None] * wy[None, :]
    F = np.fft.fft2(eta_t)
    PSD = np.abs(F) ** 2
    kx_1d = 2 * np.pi * np.fft.fftfreq(nx, d=dx)
    ky_1d = 2 * np.pi * np.fft.fftfreq(ny, d=dy)
    KX, KY = np.meshgrid(kx_1d, ky_1d, indexing='ij')
    K = np.sqrt(KX ** 2 + KY ** 2)
    L = 2 * np.pi / np.where(K > 0, K, np.nan)  # wavelength [m]

    band_eddy = (L >= EDDY_LMIN * 1e3) & (L <= EDDY_LMAX * 1e3)
    band_setup = L > SETUP_LMIN * 1e3
    band_sub = L < EDDY_LMIN * 1e3      # < 50 km (sub-mesoscale / damped)
    PSD = PSD * wx[:, None] * wy[None, :]  # matches taper (approx) for weights

    # weights: PSD symmetric under real input; count each k-mode once.
    def band_frac(mask, psd):
        return float(np.sum(psd[np.isfinite(L) & mask]) / max(1e-30, np.sum(psd[np.isfinite(L)])))

    frac_eddy = band_frac(band_eddy, PSD)
    frac_setup = band_frac(band_setup, PSD)
    frac_sub = band_frac(band_sub, PSD)

    print("\n  SSH variance by wavelength band (2D spectrum, tapered):")
    print(f"    mesoscale eddy   {EDDY_LMIN:.0f}-{EDDY_LMAX:.0f} km : "
          f"{frac_eddy*100:5.1f} %")
    print(f"    gyre setup  >{SETUP_LMIN:.0f} km        : {frac_setup*100:5.1f} %")
    print(f"    sub-meso   <{EDDY_LMIN:.0f} km         : {frac_sub*100:5.1f} %")

    # ── Interpretation ──
    if frac_eddy > 0.3:
        verdict = "MESOSCALE EDDIES DOMINATE variance — strong basis for a defensible real-data metric"
    elif frac_setup > 0.5:
        verdict = "GYRE SETUP DOMINATES (low wavenumber tilt) — weakly supports real-data eddy comparison"
    else:
        verdict = "MIXED / INCONCLUSIVE — inspect SSH field directly"
    print(f"\n  INTERPRETATION: {verdict}")


if __name__ == "__main__":
    main()
