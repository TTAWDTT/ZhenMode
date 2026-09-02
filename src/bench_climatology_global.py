"""
Stage-3 climatology comparison — GLOBAL FD solver edition.

Sister to bench_climatology_compare.py (regional). Reads a global run npz
from run_long_integration_global.py and scores the model climatology against
WOA2023, using the SAME pre-registered A1/A2 criteria (do NOT move the bar):

  A1: zonal-mean SST(y) model vs WOA    — pattern corr > 0.3, RMSE < 2.0 C
  A2: SST large-scale pattern (>2deg smoothed, demeaned) — corr > 0.3, RMSE < 2.0 C

NON-CIRCULARITY (the whole point of the global non-periodic domain): T_atm
(the bulk-flux restoring target) is a ZONALLY-UNIFORM meridional profile
built from the zonal-mean WOA SST. Only the meridional gradient is
prescribed. Any ZONAL structure in the model SST (the A2 pattern skill)
is genuinely PREDICTED by advection + mixing + wind — it cannot be read
off the forcing. A1 (zonal-mean) is partly restored (it tests whether the
meridional SST gradient equilibrates correctly under the bulk flux, which
is a real model question, not a tautology — a wrong diffusivity / wind
gyre gives the wrong heat-transport-equivalent gradient even with the same
T_atm). A2 is the strict non-circular skill.

B-class (dynamical plausibility, informational, no pass/fail):
  B1: SST variance distribution  B2: SSH spectrum slope  B3: KE drift

Excluded (T3-1 lesson — do NOT compute as pass/fail):
  - pointwise SLA/SSH spatial correlation (structural mismatch)
  - mesoscale eddy-by-eddy matching (phase unpredictable)
  - mesoscale SLA variance absolute value (resolution-limited)

Usage:
  python src/bench_climatology_global.py --npz results/global_g365d_prod.npz \
      [--steady-days 90] [--out-dir results/climatology_g]
"""
import sys
import os
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

from dataclasses import replace
from config import DEFAULT_CONFIG, GlobalGridConfig
from grid import make_global_grid
from woa_data import get_initial_fields


EXCLUDED = [
    "pointwise SLA/SSH spatial correlation (T3-1 structural mismatch)",
    "mesoscale eddy-by-eddy matching (phase unpredictable)",
    "mesoscale SLA variance absolute value (resolution-limited, H1-H4 falsified)",
]


def zonal_mean(field):
    """Mean over x (axis 0, periodic lon) -> (ny,) for 2D (nx,ny) or (ny,nz) for 3D."""
    return np.mean(field, axis=0)


def smooth_2d_global(field, lat, deg=2.0):
    """Spatially smooth a 2D (nx,ny) field to >deg degrees (large-scale only).

    Global grid: lon is periodic (wrap axis 0), lat is bounded (clamp axis 1).
    1 deg of latitude ~ 111 km; at 1 deg resolution a 2-deg window ~ 2 cells.
    Box-mean smoother isolating the large-scale pattern from mesoscale noise.
    """
    f = np.asarray(field, float)
    nx, ny = f.shape
    win = max(1, int(round(deg)))   # window half-width in cells (1 deg resolution)
    # periodic lon pad (axis 0), edge-clamp lat pad (axis 1)
    p = np.empty((nx + 2 * win, ny + 2 * win), dtype=float)
    p[win:win + nx, win:win + ny] = f
    # lon periodic: wrap
    p[:win, win:win + ny] = f[-win:, :]
    p[win + nx:, win:win + ny] = f[:win, :]
    # lat edge-clamp (no wrap at the no-flux wall)
    p[:, :win] = p[:, win:win + 1]
    p[:, win + ny:] = p[:, win + ny - 1:win + ny]
    # moving box mean via cumulative sum (fields are finite — no NaN here)
    cs = np.cumsum(np.cumsum(p, axis=0), axis=1)
    cs = np.pad(cs, ((1, 0), (1, 0)), mode='constant')   # so i0,j0 indexing works
    out = np.empty_like(f)
    w2 = 2 * win + 1
    for i in range(nx):
        for j in range(ny):
            i0, j0 = i, j
            i1, j1 = i + w2, j + w2
            out[i, j] = (cs[i1, j1] - cs[i0, j1] - cs[i1, j0] + cs[i0, j0]) / (w2 * w2)
    return out


def radial_spectrum(field, dx_m):
    """Azimuthally-averaged (radial) power spectrum of a 2D field.

    Returns (k_1m, P_k) for the KE/SSH spectrum vs k^-3 scaling check.
    """
    f = np.nan_to_num(field, nan=0.0).astype(np.float64)
    f = f - f.mean()
    nx, ny = f.shape
    wx = np.hanning(nx); wy = np.hanning(ny)
    ft = f * wx[:, None] * wy[None, :]
    F = np.fft.fftshift(np.fft.fft2(ft))
    PSD = np.abs(F) ** 2
    kx = np.fft.fftshift(np.fft.fftfreq(nx, d=dx_m))
    ky = np.fft.fftshift(np.fft.fftfreq(ny, d=dx_m))
    KX, KY = np.meshgrid(kx, ky, indexing='ij')
    K = np.sqrt(KX ** 2 + KY ** 2)
    kflat = K.ravel()
    pflat = PSD.ravel()
    kmax = min(kx.max(), ky.max())
    nb = 40
    bins = np.linspace(1e-6, kmax, nb + 1)
    idx = np.digitize(kflat, bins) - 1
    k_cent = 0.5 * (bins[:-1] + bins[1:])
    P = np.array([pflat[idx == b].mean() if (idx == b).any() else 0.0
                  for b in range(nb)])
    return k_cent, P


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True,
                    help="global_*.npz from run_long_integration_global.py")
    ap.add_argument("--steady-days", type=float, default=90.0,
                    help="use the last N days of snapshots as the steady-state climatology")
    ap.add_argument("--out-dir", default="results/climatology_g")
    ap.add_argument("--smooth-passes", type=int, default=30,
                    help="bathymetry smoothing passes (must match the run)")
    ap.add_argument("--min-depth", type=float, default=100.0)
    ap.add_argument("--lat-max", type=float, default=60.0)
    ap.add_argument("--ny", type=int, default=120)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # ── Load model run snapshots ──
    z = np.load(args.npz, allow_pickle=True)
    days = z['days']
    eta_snaps = z['eta']          # (n_snap, nx, ny)
    T_top_snaps = z['T_top']      # (n_snap, nx, ny)
    max_u = z['max_u']
    ke = z['ke']
    max_eta = z['max_eta']
    verdict = str(z['verdict'])
    print("=" * 70)
    print("CLIMATOLOGY COMPARISON — GLOBAL FD (stage 3)")
    print("=" * 70)
    print(f"source: {args.npz}")
    print(f"run verdict: {verdict}")
    print(f"snapshots: {len(days)} over days {days[0]:.1f}..{days[-1]:.1f}")

    if verdict != "PASS":
        print(f"  WARNING: source run did not PASS ({verdict}). Climatology may "
              f"be meaningless if it blew up. Proceeding on whatever steady "
              f"window exists, but flag results as UNRELIABLE.")

    # ── Rebuild the global grid (must match the run config) ──
    bathy = DEFAULT_CONFIG.bathymetry_file
    gcfg = replace(GlobalGridConfig(), lat_max=args.lat_max, ny=args.ny)
    grid = make_global_grid(gcfg, bathy, smooth_passes=args.smooth_passes,
                            min_depth=args.min_depth)
    ocean = np.asarray(grid.ocean_mask, dtype=bool)
    lat = np.asarray(grid.lat)
    lon = np.asarray(grid.lon)
    dx_eq = float(grid.dx_2d[0, grid.ny // 2])
    print(f"  grid {grid.nx}x{grid.ny}x{grid.nz}, ocean {float(grid.wet_mask.mean()):.1%}, "
          f"dx_eq={dx_eq:.0f}m, lat[{lat[0]:.1f},{lat[-1]:.1f}]N")

    # steady-state window: last steady-days
    t_end = days[-1]
    t_start = t_end - args.steady_days
    mask = days >= t_start
    if mask.sum() < 2:
        print(f"  only {mask.sum()} snapshots in steady window; using all")
        mask = np.ones_like(days, dtype=bool)
    print(f"steady window: day {t_start:.1f}..{t_end:.1f} ({mask.sum()} snapshots)")

    # ── Model climatology (time-mean over steady window) ──
    eta_clim = np.mean(eta_snaps[mask], axis=0)         # (nx,ny) mean SSH
    sst_clim = np.mean(T_top_snaps[mask], axis=0)       # (nx,ny) mean SST
    sst_anom = T_top_snaps[mask] - sst_clim[None, :, :]
    sst_var = np.mean(sst_anom ** 2, axis=0)            # SST variance

    # ── Reference climatology (WOA2023) on the SAME global grid ──
    print("Loading WOA2023 reference climatology...")
    T_init, S_init = get_initial_fields(grid)   # (nx,ny,nz) interpolated to grid
    woa_sst = np.asarray(T_init)[:, :, 0]       # WOA surface T = SST climatology

    # ============================================================
    # A-class: large-scale, matchable
    # ============================================================
    print("\n--- A-CLASS (large-scale, primary criteria) ---")

    # A1: zonal-mean SST(y) model vs WOA
    sst_zonal_model = np.array([sst_clim[ocean[:, j], j].mean()
                                if ocean[:, j].any() else np.nan
                                for j in range(grid.ny)])
    sst_zonal_woa = np.array([woa_sst[ocean[:, j], j].mean()
                              if ocean[:, j].any() else np.nan
                              for j in range(grid.ny)])
    good = np.isfinite(sst_zonal_model) & np.isfinite(sst_zonal_woa)
    corr_zonal = float(np.corrcoef(sst_zonal_model[good], sst_zonal_woa[good])[0, 1])
    rmse_zonal = float(np.sqrt(np.mean((sst_zonal_model[good] - sst_zonal_woa[good]) ** 2)))
    print(f"  [A1] zonal-mean SST(y) model vs WOA (ocean-mean at each lat):")
    print(f"       pattern corr = {corr_zonal:.3f}  (target > 0.3)")
    print(f"       RMSE         = {rmse_zonal:.3f} C  (target < 2.0)")
    a1_pass = corr_zonal > 0.3 and rmse_zonal < 2.0

    # A2: SST large-scale pattern (>2deg smoothed) model vs WOA — demeaned,
    # ocean-only. This is the NON-CIRCULAR skill (zonal structure is predicted).
    sst_model_sm = smooth_2d_global(sst_clim, lat, deg=2.0)
    sst_woa_sm = smooth_2d_global(woa_sst, lat, deg=2.0)
    m = ocean
    a_model = sst_model_sm[m] - sst_model_sm[m].mean()
    a_woa = sst_woa_sm[m] - sst_woa_sm[m].mean()
    corr_pat = (float(np.corrcoef(a_model, a_woa)[0, 1])
                if a_model.std() > 0 else float('nan'))
    rmse_pat = float(np.sqrt(np.mean((sst_model_sm[m] - sst_woa_sm[m]) ** 2)))
    print(f"  [A2] SST large-scale pattern (>2deg smoothed, demeaned) vs WOA:")
    print(f"       pattern corr = {corr_pat:.3f}  (target > 0.3)")
    print(f"       RMSE         = {rmse_pat:.3f} C  (target < 2.0)")
    print(f"       (non-circular: T_atm is zonally uniform; zonal SST structure")
    print(f"        is genuinely predicted by advection/mixing/wind)")
    a2_pass = ((not np.isnan(corr_pat)) and corr_pat > 0.3 and rmse_pat < 2.0)

    # ============================================================
    # B-class: dynamical plausibility (no phase match required)
    # ============================================================
    print("\n--- B-CLASS (dynamical plausibility, informational) ---")

    # B1: SST variance spatial distribution
    print(f"  [B1] SST variance (steady window):")
    print(f"       mean var = {np.mean(sst_var[ocean]):.4f} C^2")
    print(f"       max var  = {np.max(sst_var[ocean]):.4f} C^2")
    print(f"       (informational: non-zero variance = active variability)")

    # B2: SSH spectrum vs k^-3
    eta_anom = eta_clim - eta_clim[ocean].mean()
    k_cent, P = radial_spectrum(eta_anom, dx_eq)
    valid = (k_cent > 0) & (P > 0) & (k_cent < k_cent.max() * 0.8)
    if valid.sum() > 4:
        lk = np.log10(k_cent[valid]); lP = np.log10(P[valid])
        slope = float(np.polyfit(lk, lP, 1)[0])
        print(f"  [B2] SSH radial spectrum log-log slope = {slope:.2f} "
              f"(geostrophic turbulence expects ~-3 to -5)")
    else:
        slope = float('nan')
        print(f"  [B2] SSH spectrum: insufficient resolved band for slope fit")

    # B3: KE trend + max|eta| trend over steady window
    ke_steady = ke[mask]
    if len(ke_steady) > 2:
        ke_drift = float((ke_steady[-1] - ke_steady[0]) / (ke_steady[0] + 1e-30))
        print(f"  [B3] KE drift over steady window = {ke_drift*100:.2f}% "
              f"(target |drift| < ~50% over the window = roughly steady)")
    else:
        ke_drift = float('nan')
        print(f"  [B3] KE drift: too few points")
    eta_steady = max_eta[mask]
    print(f"       max|eta| over steady window: {eta_steady.min():.3f}..{eta_steady.max():.3f} m")

    # ============================================================
    # Excluded metrics (explicitly NOT computed)
    # ============================================================
    print("\n--- EXCLUDED (by design, T3-1 lesson) ---")
    for ex in EXCLUDED:
        print(f"  [X] {ex}")

    # ── Summary verdict ──
    print("\n" + "=" * 70)
    print("  A-CLASS SUMMARY:")
    print(f"    A1 zonal-mean SST:  {'PASS' if a1_pass else 'FAIL'} "
          f"(corr={corr_zonal:.3f}, rmse={rmse_zonal:.3f})")
    print(f"    A2 SST pattern:     {'PASS' if a2_pass else 'FAIL'} "
          f"(corr={corr_pat:.3f}, rmse={rmse_pat:.3f})  [non-circular]")
    print("  B-CLASS (informational, no pass/fail):")
    print(f"    B1 SST var mean={np.mean(sst_var[ocean]):.4f}, "
          f"B2 slope={slope:.2f}, B3 KE drift={ke_drift*100:.2f}%")
    overall = a1_pass and a2_pass
    print(f"  OVERALL: {'PASS' if overall else 'FAIL'}  "
          f"(A1 AND A2; B-class informational)")
    print("=" * 70)

    # ── Save figures + data ──
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        # Fig 1: zonal-mean SST profile
        fig, ax = plt.subplots(1, 1, figsize=(6, 5))
        ax.plot(sst_zonal_model, lat, 'b-', label='model climatology')
        ax.plot(sst_zonal_woa, lat, 'r--', label='WOA2023')
        ax.set_xlabel('zonal-mean SST (degC)'); ax.set_ylabel('latitude (N)')
        ax.set_title(f'Zonal-mean SST  (corr={corr_zonal:.3f}, RMSE={rmse_zonal:.2f})')
        ax.legend(); ax.grid(True)
        fig.tight_layout(); fig.savefig(os.path.join(args.out_dir, 'zonal_sst.png'), dpi=120)
        plt.close(fig)

        # Fig 2: SST pattern (smoothed)
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        for ax, fld, title in zip(axes,
                                  [sst_model_sm, sst_woa_sm, sst_model_sm - sst_woa_sm],
                                  ['model (smoothed)', 'WOA (smoothed)', 'model - WOA']):
            im = ax.pcolormesh(lon, lat, fld.T, shading='auto')
            ax.set_title(title); ax.set_xlabel('lon E'); ax.set_ylabel('lat N')
            fig.colorbar(im, ax=ax)
        fig.suptitle(f'SST large-scale pattern (corr={corr_pat:.3f}, RMSE={rmse_pat:.2f})')
        fig.tight_layout(); fig.savefig(os.path.join(args.out_dir, 'sst_pattern.png'), dpi=120)
        plt.close(fig)

        # Fig 3: SSH climatology + spectrum
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        im = axes[0].pcolormesh(lon, lat, eta_clim.T, shading='auto')
        axes[0].set_title('model SSH climatology (m)'); axes[0].set_xlabel('lon E')
        axes[0].set_ylabel('lat N'); fig.colorbar(im, ax=axes[0])
        axes[1].loglog(k_cent[valid], P[valid], 'b-')
        k_ref = k_cent[valid]; axes[1].loglog(k_ref, P[valid][0]*(k_ref/k_ref[0])**-3, 'r--', label='k^-3')
        axes[1].set_title(f'SSH radial spectrum (slope={slope:.2f})')
        axes[1].set_xlabel('k (1/m)'); axes[1].set_ylabel('PSD'); axes[1].legend(); axes[1].grid(True)
        fig.tight_layout(); fig.savefig(os.path.join(args.out_dir, 'ssh_spectrum.png'), dpi=120)
        plt.close(fig)
        print(f"\nfigures saved to {args.out_dir}/")
    except Exception as e:
        print(f"\n(plotting skipped: {e!r})")

    # ── Save comparison data ──
    np.savez(os.path.join(args.out_dir, 'climatology_compare_g.npz'),
             sst_zonal_model=sst_zonal_model, sst_zonal_woa=sst_zonal_woa,
             sst_model_sm=sst_model_sm, sst_woa_sm=sst_woa_sm,
             eta_clim=eta_clim, sst_var=sst_var,
             k_cent=k_cent, P=P, slope=np.array(slope),
             corr_zonal=np.array(corr_zonal), rmse_zonal=np.array(rmse_zonal),
             corr_pat=np.array(corr_pat), rmse_pat=np.array(rmse_pat),
             a1_pass=np.array(a1_pass), a2_pass=np.array(a2_pass),
             overall_pass=np.array(overall),
             ke_drift=np.array(ke_drift),
             lat=lat, lon=lon)
    print(f"data saved to {args.out_dir}/climatology_compare_g.npz")

    return 0 if overall else 1


if __name__ == "__main__":
    sys.exit(main())
