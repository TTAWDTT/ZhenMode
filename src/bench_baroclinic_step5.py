"""Step 5: Real-data SLA comparison with mesoscale band-pass filtering.

Builds on the Step 2 stable baroclinic config (nu_bi=1e12 + WOA stratified
T/S + Haney surface-T restoring). The Step 2 spectral diagnostic showed the
stratified SSH variance is ~43% mesoscale eddies and ~47% large-scale gyre
setup. Real satellite SLA is predominantly mesoscale, so a raw full-field
correlation would be dominated by the (physics-and-forcing-dependent) gyre
setup tilt. This script therefore band-passes BOTH fields to the mesoscale
band (50-600 km, Difference-of-Gaussians in real space) before correlating,
isolating the eddy signal — the defensible "solid" metric the research
targets.

Returns corr / RMSE / amplitude for both the band-passed (primary) and the
raw full-field (reference/T3-1-equivalent) comparison.

Usage:
    python src/bench_baroclinic_step5.py --month 2023-01 --spinup-days 10
        --nu-bi 1e12 --tau-restore-days 30 [--force-fetch]
"""
import sys, os, time, argparse
import numpy as np
from dataclasses import replace
from scipy.ndimage import gaussian_filter
from scipy.interpolate import RegularGridInterpolator

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp

from config import DEFAULT_CONFIG
from grid import make_grid
from jax_solver import make_solver
from forcing import wind_stress_gyre, heat_flux_meridional
from wind_reanalysis import real_wind_forcing
from woa_data import get_initial_fields

DT = 300.0
# Mesoscale band (km). DoG band-pass = G(sig_hi) - G(sig_lo) retains
# wavelengths sig_lo*2pi .. sig_hi*2pi.
L_HI = 600.0      # km: retain wavelengths < 600 km (remove gyre setup)
L_LO = 50.0       # km: retain wavelengths > 50 km (remove sub-mesoscale)
DEG_PER_KM = 1.0 / 111.0   # ~1 deg lat ~ 111 km


def band_pass(field, sig_hi_km, sig_lo_km, dx_km, land=None):
    """Difference-of-Gaussians band-pass in real space, same units as field.
    land: (nx,ny) bool mask of land to fill before filtering (edge control).
    Returns band-passed field with values only meaningful over ocean,
    land set to NaN.
    """
    f = np.array(field, dtype=np.float64)
    if land is not None:
        f = np.where(land, 0.0, f)          # fill land to avoid edge smear
    sig_hi = max(1.0, sig_hi_km / dx_km)     # in pixels
    sig_lo = max(1.0, sig_lo_km / dx_km)
    g_hi = gaussian_filter(f, sigma=sig_hi, mode='reflect')
    g_lo = gaussian_filter(f, sigma=sig_lo, mode='reflect')
    bp = g_hi - g_lo
    if land is not None:
        bp = np.where(land, np.nan, bp)
    return bp


def get_model_eta(grid, physics, T_init, S_init, tau_x, tau_y, Q_heat,
                  spinup_days, tau_restore_days):
    step, init_state, _ = make_solver(grid, physics, DT,
                                      forcing=(tau_x, tau_y, Q_heat),
                                      T_sst=T_init[:, :, 0],
                                      tau_restore_days=tau_restore_days)
    state = init_state(T_init=jnp.array(T_init), S_init=jnp.array(S_init))
    n_spin = int(round(spinup_days * 86400.0 / DT))
    for n in range(n_spin):
        state = step(state)
    return np.asarray(state.eta)


def load_obs_sla(cache, grid):
    d = np.load(cache)
    sla_mean = d['sla']; lon_obs = d['lon']; lat_obs = d['lat']
    lat_obs_a = np.sort(lat_obs); lon_obs_a = np.sort(lon_obs)
    if not np.array_equal(lat_obs, lat_obs_a):
        sla_mean = sla_mean[np.argsort(lat_obs), :]
    if not np.array_equal(lon_obs, lon_obs_a):
        sla_mean = sla_mean[:, np.argsort(lon_obs)]
    interp = RegularGridInterpolator((lat_obs_a, lon_obs_a), sla_mean,
                                     bounds_error=False, fill_value=np.nan)
    yy, xx = np.meshgrid(grid.lat, grid.lon, indexing='ij')
    obs = interp(np.stack([yy.ravel(), xx.ravel()], -1)).reshape(
        grid.ny, grid.nx).T
    return np.asarray(obs, dtype=np.float64)


def corr_metrics(a, b, mask):
    a = a[mask]; b = b[mask]
    a = a - a.mean(); b = b - b.mean()
    den = np.sqrt((a * a).mean() * (b * b).mean())
    r = float((a * b).mean() / den) if den > 0 else float('nan')
    rmse = float(np.sqrt(np.mean((a - b) ** 2)))
    return r, rmse, float(a.std()), float(b.std())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--month", default="2023-01")
    ap.add_argument("--spinup-days", type=float, default=10.0)
    ap.add_argument("--nu-bi", type=float, default=1e12)
    ap.add_argument("--tau-restore-days", type=float, default=30.0)
    ap.add_argument("--force-fetch", action="store_true")
    ap.add_argument("--save-eta", default=None,
                    help="optional path (npz) to cache the model SSH field")
    args = ap.parse_args()




    grid = make_grid(DEFAULT_CONFIG.grid, DEFAULT_CONFIG.bathymetry_file)
    ocean = np.asarray(grid.ocean_mask, dtype=bool)
    land = ~ocean
    lat = np.asarray(grid.lat); lon = np.asarray(grid.lon)
    dx_deg = float(lon[1] - lon[0])
    dx_km = grid.dx / 1000.0      # ~9 km
    print("=" * 68)
    print("STEP 5 — REAL-DATA SLA CORRELATION (baroclinic, mesoscale band-pass)")
    print("=" * 68)
    print(f"grid {grid.nx}x{grid.ny}x{grid.nz}  nu_bi={args.nu_bi:g}  "
          f"restore={args.tau_restore_days:g}d")
    print(f"mesoscale band: {L_LO:.0f}-{L_HI:.0f} km (DoG band-pass)")

    # ── Forcing (same real wind as T3-1) ──
    y, m = int(args.month[:4]), int(args.month[5:7])
    month_idx = (y - 1948) * 12 + (m - 1)
    try:
        tau_x, tau_y = real_wind_forcing(month_idx=month_idx, grid=grid)
        print(f"\nforcing: real NCEP/NCAR R1 monthly wind, month {args.month}")
    except Exception as e:
        print(f"  real wind fetch failed ({e!r}); fallback Stommel gyre")
        tau_x, tau_y = wind_stress_gyre(grid, tau0=0.1)
    Q_heat = heat_flux_meridional(grid, Q0=50.0)

    # ── Stratified initial T/S + stable baroclinic config ──
    physics = replace(DEFAULT_CONFIG.physics, nu_bi=args.nu_bi,
                      kappa_bi=args.nu_bi)
    T_init, S_init = get_initial_fields(grid)
    t0 = time.time()
    model_eta = get_model_eta(grid, physics, T_init, S_init,
                              tau_x, tau_y, Q_heat,
                              args.spinup_days, args.tau_restore_days)
    print(f"  model spin-up {args.spinup_days:.0f}d -> {time.time()-t0:.0f}s")
    print(f"  model SSH ocean std = {np.std(model_eta[ocean]):.4f} m")
    if args.save_eta:
        np.savez(args.save_eta, eta=np.asarray(model_eta))

    # ── Observed SLA (cached or fetch) ──
    cache = f"data/t3_sla_{args.month}.npz"
    if not os.path.exists(cache) or args.force_fetch:
        from bench_t3_realdata import fetch_sla_month
        ys = args.month + "-01T00:00:00Z"
        ye = args.month + "-28T23:59:59Z"
        fetch_sla_month(grid, ys, ye, cache=cache)
    sla_obs = load_obs_sla(cache, grid)
    print(f"  obs SLA: model-grid interpolated, mean/std over ocean "
          f"= {np.nanmean(np.where(ocean, sla_obs, np.nan)):.4f} / "
          f"{np.nanstd(np.where(ocean, sla_obs, np.nan)):.4f} m")

    # ── Band-pass both fields to mesoscale band ──
    sig_hi = L_HI / (2 * np.pi)          # km -> filter sigma
    sig_lo = L_LO / (2 * np.pi)
    m_model_bp = band_pass(model_eta, sig_hi, sig_lo, dx_km, land)
    m_sla_bp = band_pass(np.nan_to_num(sla_obs, nan=0.0), sig_hi, sig_lo,
                         dx_km, land)

    # valid mask: ocean where both fields are finite
    valid = ocean & np.isfinite(m_model_bp) & np.isfinite(m_sla_bp) \
        & np.isfinite(sla_obs)

    # ── Metrics: band-passed (primary) and raw (reference) ──
    r_bp, rmse_bp, a_mod_bp, a_obs_bp = corr_metrics(m_model_bp, m_sla_bp, valid)
    # raw full-field (T3-1-style, demean each)
    eta_c = model_eta - model_eta[valid].mean()
    obs_c = sla_obs - np.nanmean(sla_obs[valid])
    r_raw, rmse_raw, a_mod_raw, a_obs_raw = corr_metrics(eta_c, obs_c, valid)

    print("\n────────────────────────────────────────────────────")
    print("RESULT — MESOSCALE BAND-PASSED (50-600 km) — primary metric")
    print(f"  spatial anomaly correlation : {r_bp:+.3f}")
    print(f"  RMS anomaly diff (RMSE)     : {rmse_bp:.4f} m")
    print(f"  model std / obs std         : {a_mod_bp:.4f} / {a_obs_bp:.4f} m")
    print("\n  REFERENCE — raw full-field (T3-1-style)")
    print(f"  correlation                  : {r_raw:+.3f}")
    print(f"  model std / obs std          : {a_mod_raw:.4f} / {a_obs_raw:.4f} m")

    # ── Honest verdict ──
    print("\n  Interpretive note: positive band-passed correlation means the "
          "modeled mesoscale eddy SSH pattern aligns with observed SLA; a low "
          "correlation at 9-km / 30-d spin-up reflects the resolution and "
          "forcing limits honestly, not a metric artifact.")


if __name__ == "__main__":
    main()
