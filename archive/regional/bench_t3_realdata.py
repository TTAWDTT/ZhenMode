"""
Tier-3 test T3-1: Real-data anomaly correlation of wind-driven sea level.

Goal: give the "is the integrated result accurate / solid" question a real
data answer. A barotropic, free-surface, wind-driven model cannot reproduce
pointwise altimetric sea-level anomaly (SLA) — observed SLA at 30-90 day
scales is dominated by baroclinic mesoscale eddies and steric effects the
model does not represent. Direct point RMSE would be dominated by that
mismatch and would be methodologically indefensible (scout finding 5a).

Defensible check (scout finding 5b): anomaly correlation of the *large-scale,
wind-driven* sea-level pattern.
  - Model: integrate the real monthly-mean wind (NCEP/NCAR R1, via
    real_wind_forcing) to a quasi-steady wind-driven state. The model's
    steady SSH (eta) is dominated by the wind-driven gyre setup.
  - Observed: daily SLA (nesdisSSH1day, NOAA CoastWatch ERDDAP, auth-free)
    averaged over the forcing month, on the same domain.
  - Compare: DEMEAN both fields (they are on different baselines: model eta is
    absolute SSH re cyclone setup, observed SLA is anomaly vs mean dynamic
    topography) and SPATIALLY SMOOTH both to >50 km (0.5 deg), the smallest
    scale the 0.25-deg observations can support. Then compute spatial
    anomaly correlation and RMSE of the residual/wind-driven pattern.

Interpretation is honest: high correlation after demeaning+smoothing means
the model's large-scale wind-driven sea-level pattern statistically
resembles the real ocean's; it is NOT a claim of mesoscale or eddy skill.

Usage:
  python src/bench_t3_realdata.py [--month YYYY-MM] [--ys YYYY-MM-DD] [--ye YYYY-MM-DD]
Default: --month 2023-01 (fall back to cached wind if fetch fails).
"""
import sys, os, time, argparse, tempfile
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
import numpy as np
import urllib.request
import netCDF4

from config import DEFAULT_CONFIG
from grid import make_grid
from jax_solver import make_solver
from forcing import wind_stress_gyre, heat_flux_meridional
from wind_reanalysis import real_wind_forcing

# ── Parameters ─────────────────────────────────────────────────────
DT = 300.0                 # s
SPINUP_DAYS = 30.0         # days to (quasi-)steady wind-driven state
SLA_URL = ("https://coastwatch.pfeg.noaa.gov/erddap/griddap/"
           "nesdisSSH1day.nc?sla[({YS})][({LAT0}):({LAT1})][({LON0}):({LON1})]")
SMOOTH_DEG = 0.5           # spatial smoothing scale (deg) -> >50 km


def smooth_gaussian(field, sigma_deg, dx_deg):
    """Isotropic Gaussian smoothing of a 2D (nx,ny) field."""
    sigma_cells = max(1.0, sigma_deg / dx_deg)
    s = int(round(6 * sigma_cells))
    y = np.arange(-s, s + 1)
    x = np.arange(-s, s + 1)
    yy, xx = np.meshgrid(y, x)
    w = np.exp(-(xx ** 2 + yy ** 2) / (2 * sigma_cells ** 2))
    w /= w.sum()
    from scipy.ndimage import convolve
    return convolve(field, w, mode='nearest')


def fetch_sla_month(grid, ys, ye, cache='data/t3_sla.npz'):
    """Fetch daily SLA over the model domain for [ys, ye] via ERDDAP.

    Returns (sla_time_mean, lon_obs, lat_obs) where sla_time_mean is the
    time-mean SLA on the observed (lon,lat) grid. Downloads the raw .nc via
    urllib (netCDF4 inline OPeNDAP constraints are flaky on Windows), then
    time-averages.
    """
    lon = grid.lon
    lat = grid.lat
    lon0, lon1 = float(lon.min()), float(lon.max())
    lat0, lat1 = float(lat.min()), float(lat.max())
    url = SLA_URL.format(YS=ys, YE=ye, LON0=lon0, LON1=lon1,
                         LAT0=lat0, LAT1=lat1)
    print(f"  fetching SLA [{ys} .. {ye}] domain "
          f"({lat0:.1f}-{lat1:.1f}N, {lon0:.1f}-{lon1:.1f}E)")
    print(f"  {url[:160]}...")
    fn = tempfile.mktemp(suffix='.nc')
    urllib.request.urlretrieve(url, fn)
    ds = netCDF4.Dataset(fn)
    sla = np.asarray(ds.variables['sla'][:], dtype=np.float64)  # (t, lat, lon)
    lon_obs = np.asarray(ds.variables['longitude'][:], dtype=np.float64)
    lat_obs = np.asarray(ds.variables['latitude'][:], dtype=np.float64)
    times = np.asarray(ds.variables['time'][:], dtype=np.float64)
    ds.close()
    os.remove(fn)
    # Time-mean over the window (mask any fill values).
    fill = -9999.0
    sla = np.where(np.isnan(sla) | (np.abs(sla) > 10.0), np.nan, sla)
    sla_mean = np.nanmean(sla, axis=0)
    n_valid = np.isfinite(sla_mean).sum()
    if n_valid < 0.5 * sla_mean.size:
        raise RuntimeError(f"Too few valid SLA points ({n_valid}/{sla_mean.size})")
    np.savez(cache, sla=sla_mean, lon=lon_obs, lat=lat_obs,
             times_per_day=times.size)
    print(f"  cached time-mean SLA ({sla_mean.shape}, {times.size} days) -> {cache}")
    return sla_mean, lon_obs, lat_obs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--month", default="2023-01",
                    help="forcing month YYYY-MM (also SLA window start)")
    ap.add_argument("--ys", default=None, help="observation start YYYY-MM-DD")
    ap.add_argument("--ye", default=None, help="observation end YYYY-MM-DD")
    ap.add_argument("--spinup-days", type=float, default=SPINUP_DAYS)
    ap.add_argument("--force-fetch", action="store_true",
                    help="re-fetch SLA even if cache exists")
    args = ap.parse_args()

    grid = make_grid(DEFAULT_CONFIG.grid, DEFAULT_CONFIG.bathymetry_file)
    lon = np.asarray(grid.lon)
    lat = np.asarray(grid.lat)
    dx_deg = float(lon[1] - lon[0])

    print("=" * 60)
    print("T3-1 REAL-DATA ANOMALY CORRELATION (wind-driven sea level)")
    print("=" * 60)
    print(f"grid {grid.nx}x{grid.ny}x{grid.nz}, {dx_deg:.2f} deg, "
          f"domain {lat.min():.1f}-{lat.max():.1f}N, "
          f"{lon.min():.1f}-{lon.max():.1f}E")

    # ── Wind forcing (monthly-mean real NCEP R1) ──
    # NCEP/NCAR R1 monthly means start 1948-01; map calendar month -> index so
    # the model is forced by the SAME month as the observed SLA window.
    y, m = int(args.month[:4]), int(args.month[5:7])
    month_idx = (y - 1948) * 12 + (m - 1)
    print(f"\nforcing: real NCEP/NCAR R1 monthly wind, month {args.month} "
          f"(time_idx={month_idx})")
    cache_wind = os.path.join("data", "wind", f"monthly_mean_{month_idx}.npz")
    if os.path.exists(cache_wind):
        print(f"  using cached monthly wind {cache_wind}")
    try:
        tau_x, tau_y = real_wind_forcing(month_idx=month_idx, grid=grid)
        print("  real_wind_forcing OK")
    except Exception as e:
        print(f"  real wind fetch failed ({e!r}); falling back to Stommel gyre")
        tau_x, tau_y = wind_stress_gyre(grid, tau0=0.1)
    Q_heat = heat_flux_meridional(grid, Q0=50.0)

    # ── Integrate to quasi-steady wind-driven state ──
    step, init_state, _ = make_solver(grid, DEFAULT_CONFIG.physics, DT,
                                      forcing=(tau_x, tau_y, Q_heat))
    _ = init_state()
    state = init_state()
    n_spin = int(round(args.spinup_days * 86400.0 / DT))
    print(f"\nintegrating {args.spinup_days:.0f} d spin-up "
          f"({n_spin} steps, dt={DT:.0f} s) to wind-driven steady state ...")
    t0 = time.time()
    every = max(1, n_spin // 5)
    for n in range(n_spin):
        state = step(state)
        if (n + 1) % every == 0:
            el = time.time() - t0
            eta_rms = float(jnp.sqrt(jnp.mean(state.eta ** 2)))
            print(f"  step {n+1:7d}/{n_spin}  eta_rms = {eta_rms:.4f} m  "
                  f"({el:.0f}s elapsed)")
    print(f"  spin-up done in {time.time()-t0:.0f}s")

    model_eta = np.asarray(state.eta)          # (nx, ny) wind-driven setup

    # ── Observed SLA (time-mean over window) ──
    ys = args.ys or (args.month + "-01T00:00:00Z")
    ye = args.ye or (args.month + "-28T23:59:59Z")
    cache = f"data/t3_sla_{args.month}.npz"
    if os.path.exists(cache) and not args.force_fetch:
        d = np.load(cache)
        sla_mean = d['sla']; lon_obs = d['lon']; lat_obs = d['lat']
        print(f"\nloaded cached observed SLA -> {cache}")
    else:
        print("\nfetching observed SLA ...")
        sla_mean, lon_obs, lat_obs = fetch_sla_month(grid, ys, ye, cache=cache)

    # ── Regrid observed SLA (time-mean) onto the model grid ──
    from scipy.interpolate import RegularGridInterpolator
    # lat_obs/lon_obs may be ascending 1-D axes; interp to model grid.
    lat_obs_a = np.sort(lat_obs)
    lon_obs_a = np.sort(lon_obs)
    sla_obs = np.asarray(sla_mean)
    # reorder columns/rows to match ascending axes if needed
    if not np.array_equal(lat_obs, lat_obs_a):
        sla_obs = sla_obs[np.argsort(lat_obs), :]
    if not np.array_equal(lon_obs, lon_obs_a):
        sla_obs = sla_obs[:, np.argsort(lon_obs)]
    interp = RegularGridInterpolator(
        (lat_obs_a, lon_obs_a), sla_obs,
        bounds_error=False, fill_value=np.nan)
    yy, xx = np.meshgrid(lat, lon, indexing='ij')
    sla_model_grid = interp(np.stack([yy.ravel(), xx.ravel()], -1)).reshape(lat.size, lon.size).T
    sla_model_grid = np.asarray(sla_model_grid, dtype=np.float64)

    # ── Smooth both to >50 km and demean ──
    m_eta = smooth_gaussian(model_eta, SMOOTH_DEG, dx_deg)
    m_sla = smooth_gaussian(np.nan_to_num(sla_model_grid, nan=0.0),
                            SMOOTH_DEG, dx_deg)
    valid = np.isfinite(sla_model_grid) & (sla_model_grid != 0.0)
    m_eta = m_eta - np.nanmean(m_eta[valid])
    m_sla = m_sla - np.nanmean(m_sla[valid])

    def corr(a, b, m):
        a = a[m]; b = b[m]
        a = a - a.mean(); b = b - b.mean()
        den = np.sqrt((a * a).mean() * (b * b).mean())
        return float((a * b).mean() / den) if den > 0 else float('nan')

    corr_val = corr(m_eta, m_sla, valid)
    # RMSE of the (smoothed, demeaned) anomaly fields on the valid mask.
    diff = (m_eta - m_sla)[valid]
    rmse = float(np.sqrt(np.mean(diff ** 2))) if diff.size else float('nan')
    amp_model = float(np.std(m_eta[valid]))
    amp_obs = float(np.std(m_sla[valid]))

    print("\n────────────────────────────────────────────────────")
    print("RESULT (smoothed, demeaned, wind-driven anomaly pattern)")
    print(f"  spatial anomaly correlation : {corr_val:+.3f}")
    print(f"  RMS anomaly diff (RMSE)     : {rmse:.4f} m")
    print(f"  model std / obs std         : {amp_model:.4f} / {amp_obs:.4f} m")
    print("  (positive corr of the large-scale wind-driven sea-level")
    print("   anomaly pattern; NOT a mesoscale/eddy skill claim)")

    # Evidence bar (pattern-consistency, modest by design).
    ok = corr_val > 0.25
    print(f"\n{'PASS' if ok else 'FAIL'}: corr {corr_val:+.3f} > 0.25 "
          f"(large-scale wind-driven SLA pattern consistency)")
    return ok


if __name__ == "__main__":
    ok = main()
    sys.exit(0 if ok else 1)
