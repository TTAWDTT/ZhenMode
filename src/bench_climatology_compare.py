"""
Stage-3 climatology comparison — compare model climatology to real
climatology / statistics (docs/long_run_climatology_plan_zh.md).

CORE DISCIPLINE: statistical-feature comparison, NOT pointwise field
correlation. The T3-1 lesson (pointwise SLA corr against eddy-dominated obs
= structural mismatch, misleading FAIL) is hardcoded here: the excluded
metrics list is explicit and the script refuses to compute them.

Reads a long_run_*.npz from run_long_integration.py (snapshots of eta /
surface T / maxima / KE), computes the time-mean of the LATTER portion
(the statistical steady state), and compares:

  A-class (large-scale, matchable — primary criteria):
    - zonal-mean T/S profile  <T>(y,z) vs WOA2023
    - SST large-scale pattern  (model <SST> vs WOA surface T), >2deg smoothed
    - Sverdrup balance         integral(v*dz) vs curl(tau)/(rho*beta)
  B-class (dynamical plausibility, no phase match required):
    - EKE spatial distribution  <u'2+v'2>/2
    - KE power spectrum         radial spectrum, k^-3 scaling check
    - conservation              mass/energy long-term trend

Excluded (will NOT compute, by design):
    - pointwise SLA/SSH spatial correlation  (T3-1 structural mismatch)
    - mesoscale eddy-by-eddy matching        (phase unpredictable)
    - mesoscale SLA variance absolute value  (resolution-limited, H1-H4)

Usage:
  python src/bench_climatology_compare.py --npz results/long_run_s2_365d.npz \
      [--steady-days 90] [--out-dir results/climatology]
"""
import sys
import os
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

from config import DEFAULT_CONFIG, RHO_0, G_EARTH, OMEGA, R_EARTH
from grid import make_grid
from woa_data import get_initial_fields, load_woa_climatology


# ── Excluded metrics (T3-1 lesson — do not compute these as pass/fail) ──
EXCLUDED = [
    "pointwise SLA/SSH spatial correlation (T3-1 structural mismatch)",
    "mesoscale eddy-by-eddy matching (phase unpredictable)",
    "mesoscale SLA variance absolute value (resolution-limited, H1-H4 falsified)",
]


def zonal_mean(field):
    """Mean over x (axis 0) -> (ny, nz) for a 3D (nx,ny,nz) field."""
    return np.mean(field, axis=0)


def smooth_2d(field, grid, deg=2.0):
    """Spatially smooth a 2D (nx,ny) field to >deg degrees (large-scale only).

    Box-mean smoother over a window of ~deg/0.1 = 20 cells. Used to isolate
    the large-scale pattern from mesoscale noise before pattern comparison.
    """
    win = max(1, int(round(deg / grid.etopo_resolution)))
    f = np.asarray(field, float)
    nx, ny = f.shape
    out = np.empty_like(f)
    for i in range(nx):
        for j in range(ny):
            i0, i1 = max(0, i - win // 2), min(nx, i + win // 2 + 1)
            j0, j1 = max(0, j - win // 2), min(ny, j + win // 2 + 1)
            out[i, j] = np.nanmean(f[i0:i1, j0:j1])
    return out


def radial_spectrum(field, dx_m):
    """Azimuthally-averaged (radial) power spectrum of a 2D field.

    Returns (k_1m, P_k) for plotting the KE spectrum vs k^-3 scaling.
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
    # radial bin
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
    ap.add_argument("--npz", required=True, help="long_run_*.npz from run_long_integration")
    ap.add_argument("--steady-days", type=float, default=90.0,
                    help="use the last N days of snapshots as the steady-state climatology")
    ap.add_argument("--out-dir", default="results/climatology")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # ── Load model run snapshots ──
    z = np.load(args.npz, allow_pickle=True)
    days = z['days']
    eta_snaps = z['eta']          # (n_snap, nx, ny)
    T_top_snaps = z['T_top']      # (n_snap, nx, ny)
    max_u = z['max_u']
    ke = z['ke']
    verdict = str(z['verdict'])
    print("=" * 70)
    print("CLIMATOLOGY COMPARISON (stage 3)")
    print("=" * 70)
    print(f"source: {args.npz}")
    print(f"run verdict: {verdict}")
    print(f"snapshots: {len(days)} over days {days[0]:.1f}..{days[-1]:.1f}")

    if verdict != "PASS":
        print(f"  WARNING: source run did not PASS ({verdict}). Climatology may "
              f"be meaningless if it blew up. Proceeding on whatever steady "
              f"window exists, but flag results as unreliable.")

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
    # anomalies (for EKE-style variance on SST)
    sst_anom = T_top_snaps[mask] - sst_clim[None, :, :]
    sst_var = np.mean(sst_anom ** 2, axis=0)            # SST variance

    grid = make_grid(DEFAULT_CONFIG.grid, DEFAULT_CONFIG.bathymetry_file)
    ocean = np.asarray(grid.ocean_mask, dtype=bool)

    # ── Reference climatology (WOA2023) ──
    print("Loading WOA2023 reference climatology...")
    T_init, S_init = get_initial_fields(grid)   # (nx,ny,nz) interpolated to grid
    woa_sst = T_init[:, :, 0]                   # WOA surface T = SST climatology

    # ============================================================
    # A-class: large-scale, matchable
    # ============================================================
    print("\n--- A-CLASS (large-scale, primary criteria) ---")

    # A1: zonal-mean T profile  <T>(y,z) model vs WOA
    # NOTE: we only saved surface T snapshots (T_top), so the full 3D zonal
    # profile comparison needs 3D snapshots. For now compare the surface
    # zonal-mean <SST>(y) vs WOA surface <T>(y). Full-depth comparison is
    # deferred until run_long_integration saves 3D T snapshots.
    sst_zonal_model = np.mean(sst_clim, axis=0)        # (ny,)
    sst_zonal_woa = np.mean(woa_sst, axis=0)           # (ny,)
    # pattern correlation of the large-scale meridional SST gradient
    corr_zonal = float(np.corrcoef(sst_zonal_model, sst_zonal_woa)[0, 1])
    rmse_zonal = float(np.sqrt(np.mean((sst_zonal_model - sst_zonal_woa) ** 2)))
    print(f"  [A1] zonal-mean SST(y) model vs WOA:")
    print(f"       pattern corr = {corr_zonal:.3f}  (target > 0.3)")
    print(f"       RMSE         = {rmse_zonal:.3f} C  (target < 2.0)")
    a1_pass = corr_zonal > 0.3 and rmse_zonal < 2.0

    # A2: SST large-scale pattern (>2deg smoothed) model vs WOA
    sst_model_sm = smooth_2d(sst_clim, grid, deg=2.0)
    sst_woa_sm = smooth_2d(woa_sst, grid, deg=2.0)
    # demean (anomaly) then correlate — pattern consistency, not amplitude
    m = ocean
    a_model = sst_model_sm[m] - sst_model_sm[m].mean()
    a_woa = sst_woa_sm[m] - sst_woa_sm[m].mean()
    corr_pat = float(np.corrcoef(a_model, a_woa)[0, 1]) if a_model.std() > 0 else float('nan')
    rmse_pat = float(np.sqrt(np.mean((sst_model_sm[m] - sst_woa_sm[m]) ** 2)))
    print(f"  [A2] SST large-scale pattern (>2deg smoothed, demeaned) vs WOA:")
    print(f"       pattern corr = {corr_pat:.3f}  (target > 0.3)")
    print(f"       RMSE         = {rmse_pat:.3f} C  (target < 2.0)")
    a2_pass = (not np.isnan(corr_pat)) and corr_pat > 0.3 and rmse_pat < 2.0

    # A3: Sverdrup balance — qualitative (wind sign / gyre structure)
    # We don't have 3D velocity snapshots to integrate v*dz, so this is a
    # qualitative check on the SSH response sign vs wind-stress curl. Recorded
    # as informational; full quantitative Sverdrup needs 3D snapshots (deferred).
    print(f"  [A3] Sverdrup balance: deferred (needs 3D velocity snapshots; "
          f"current run saved 2D only). Qualitative sign check: model SSH "
          f"std={np.std(eta_clim[ocean]):.4f} m, should be O(cm) for wind-driven.")
    a3_pass = None  # deferred

    # ============================================================
    # B-class: dynamical plausibility (no phase match required)
    # ============================================================
    print("\n--- B-CLASS (dynamical plausibility, informational) ---")

    # B1: EKE / SST variance spatial distribution
    print(f"  [B1] SST variance (steady window):")
    print(f"       mean var = {np.mean(sst_var[ocean]):.4f} C^2")
    print(f"       max var  = {np.max(sst_var[ocean]):.4f} C^2")
    print(f"       (informational: non-zero variance = active variability)")

    # B2: KE power spectrum vs k^-3
    # use SSH anomaly field as proxy (KE spectrum needs 3D u,v; use eta spectrum)
    eta_anom = eta_clim - eta_clim[ocean].mean()
    k_cent, P = radial_spectrum(eta_anom, grid.dx)
    # fit slope in log-log over the resolved (non-tail) band
    valid = (k_cent > 0) & (P > 0) & (k_cent < k_cent.max() * 0.8)
    if valid.sum() > 4:
        lk = np.log10(k_cent[valid]); lP = np.log10(P[valid])
        slope = float(np.polyfit(lk, lP, 1)[0])
        print(f"  [B2] SSH radial spectrum log-log slope = {slope:.2f} "
              f"(geostrophic turbulence expects ~-3 to -5)")
    else:
        slope = float('nan')
        print(f"  [B2] SSH spectrum: insufficient resolved band for slope fit")

    # B3: conservation — KE trend over steady window
    ke_steady = ke[mask]
    if len(ke_steady) > 2:
        ke_drift = float((ke_steady[-1] - ke_steady[0]) / (ke_steady[0] + 1e-30))
        print(f"  [B3] KE drift over steady window = {ke_drift*100:.2f}% "
              f"(target |drift| < ~50% over the window = roughly steady)")
    else:
        ke_drift = float('nan')
        print(f"  [B3] KE drift: too few points")

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
          f"(corr={corr_pat:.3f}, rmse={rmse_pat:.3f})")
    print(f"    A3 Sverdrup:        DEFERRED (needs 3D snapshots)")
    print("  B-CLASS (informational, no pass/fail):")
    print(f"    B1 SST var mean={np.mean(sst_var[ocean]):.4f}, "
          f"B2 slope={slope:.2f}, B3 KE drift={ke_drift*100:.2f}%")
    print("=" * 70)

    # ── Save figures + data ──
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        # Fig 1: zonal-mean SST profile
        fig, ax = plt.subplots(1, 1, figsize=(6, 5))
        ax.plot(sst_zonal_model, grid.lat, 'b-', label='model climatology')
        ax.plot(sst_zonal_woa, grid.lat, 'r--', label='WOA2023')
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
            im = ax.pcolormesh(grid.lon, grid.lat, fld.T, shading='auto')
            ax.set_title(title); ax.set_xlabel('lon E'); ax.set_ylabel('lat N')
            fig.colorbar(im, ax=ax)
        fig.suptitle(f'SST large-scale pattern (corr={corr_pat:.3f}, RMSE={rmse_pat:.2f})')
        fig.tight_layout(); fig.savefig(os.path.join(args.out_dir, 'sst_pattern.png'), dpi=120)
        plt.close(fig)

        # Fig 3: SSH climatology + spectrum
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        im = axes[0].pcolormesh(grid.lon, grid.lat, eta_clim.T, shading='auto')
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
    np.savez(os.path.join(args.out_dir, 'climatology_compare.npz'),
             sst_zonal_model=sst_zonal_model, sst_zonal_woa=sst_zonal_woa,
             sst_model_sm=sst_model_sm, sst_woa_sm=sst_woa_sm,
             eta_clim=eta_clim, sst_var=sst_var,
             k_cent=k_cent, P=P, slope=np.array(slope),
             corr_zonal=np.array(corr_zonal), rmse_zonal=np.array(rmse_zonal),
             corr_pat=np.array(corr_pat), rmse_pat=np.array(rmse_pat),
             a1_pass=np.array(a1_pass), a2_pass=np.array(a2_pass),
             ke_drift=np.array(ke_drift))
    print(f"data saved to {args.out_dir}/climatology_compare.npz")

    return 0


if __name__ == "__main__":
    sys.exit(main())
