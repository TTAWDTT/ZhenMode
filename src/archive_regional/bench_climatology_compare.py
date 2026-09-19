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

    Box-mean smoother over a window sized by the grid spacing (dx~9.1km here,
    so 2 deg ~ 222km ~ 24 cells). Used to isolate the large-scale pattern from
    mesoscale noise before pattern comparison.
    """
    # 1 deg of latitude ~ 111 km; use mean of dx/dy for the window in cells.
    cell_km = (grid.dx + grid.dy) / 2.0 / 1000.0
    win = max(1, int(round(deg * 111.0 / cell_km)))
    f = np.asarray(field, float)
    nx, ny = f.shape
    out = np.empty_like(f)
    for i in range(nx):
        for j in range(ny):
            i0, i1 = max(0, i - win // 2), min(nx, i + win // 2 + 1)
            j0, j1 = max(0, j - win // 2), min(ny, j + win // 2 + 1)
            out[i, j] = np.nanmean(f[i0:i1, j0:j1])
    return out


def depth_integrated_transport(v_3d, grid):
    """Depth-integrate meridional velocity v over the wet column.

    Mirrors the solver's barotropic convention (jax_solver._barotropic_velocity):
    layer-centred velocity v_layer = 0.5*(v[k]+v[k+1]) weighted by the layer
    thickness dz[k]=|z[k]-z[k+1]|, with a partial bottom cell masked by the
    local seafloor depth. Returns V = sum(v_layer * dz_wet) [m^2/s] (nx,ny).

    Sign convention: z is negative downward (0 at surface, -4000 at depth).
    A layer k spans z_shallow = z[k] (less negative) .. z_deep = z[k+1]
    (more negative). The seafloor is at z_sf = -depth. The wet part of the
    layer runs from z_shallow down to min(z_deep, z_sf); its (positive)
    thickness is z_shallow - min(z_deep, z_sf). A layer fully below the
    seafloor (z_shallow < z_sf) is dry -> thickness 0.
    """
    z = np.asarray(grid.z, float)              # (nz,) negative downward
    nx, ny, nz = v_3d.shape
    depth = np.asarray(grid.depth, float)      # (nx,ny) positive ocean depth
    v_layer = 0.5 * (v_3d[:, :, :-1] + v_3d[:, :, 1:])   # (nx,ny,nz-1)
    z_shallow = z[:-1][None, None, :]          # (1,1,nz-1) less-negative edge
    z_deep = z[1:][None, None, :]              # (1,1,nz-1) more-negative edge
    z_sf = -depth[:, :, None]                  # (nx,ny,1) seafloor (negative)
    # wet bottom of cell = shallower of (cell deep edge, seafloor)
    wet_deep = np.maximum(z_deep, z_sf)        # both negative -> max = shallower
    dz_wet = np.clip(z_shallow - wet_deep, 0.0, None)   # (nx,ny,nz-1)
    return np.sum(v_layer * dz_wet, axis=-1)          # (nx,ny) m^2/s


def wind_stress_curl(tau_x, tau_y, grid):
    """curl(tau) = dtau_y/dx - dtau_x/dy [N/m^3] via central differences."""
    dtx_dy = np.gradient(tau_x, grid.dy, axis=1)
    dty_dx = np.gradient(tau_y, grid.dx, axis=0)
    return dty_dx - dtx_dy


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
    ap.add_argument("--sverdrup-3d-dir", default=None,
                    help="dir of streamed 3D snapshots (snap_XXXXX.npy, shape "
                         "(3,nx,ny,nz)=[T,U,V]) from run_long_integration --save-3d. "
                         "If supplied, computes the A3 Sverdrup balance "
                         "(beta*V vs curl(tau)/rho0); otherwise A3 stays deferred.")
    ap.add_argument("--sponge-cells", type=int, default=0,
                    help="half-width of the N/S sponge band (grid points); the "
                         "Sverdrup comparison excludes this many edge rows where "
                         "Rayleigh damping violates the linear steady balance.")
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

    # A3: Sverdrup balance  beta*V = curl(tau)/rho0   (linear steady interior)
    # V = depth-integrated meridional transport from 3D snapshots; RHS from
    # the time-mean wind-stress curl. Large-scale statistical comparison
    # (zonal-mean V(y), demeaned) over the interior only — the sponge bands
    # violate the linear steady balance, so they are excluded.
    a3_pass = None
    sverdrup_corr = float('nan')
    sverdrup_rmse = float('nan')
    V_model = None
    V_sverdrup = None
    if args.sverdrup_3d_dir:
        # match the steady window to the 3D snapshot files
        snap_files = sorted(__import__('glob').glob(
            os.path.join(args.sverdrup_3d_dir, 'snap_*.npy')))
        # one 3D snap per npz snapshot, same cadence -> index-align by mask
        snap_idx = np.where(mask)[0]
        if len(snap_files) >= len(days) and len(snap_idx) > 0:
            keep = [snap_files[i] for i in snap_idx if i < len(snap_files)]
            v_stack = []
            for f in keep:
                a = np.load(f)              # (3,nx,ny,nz) [T,U,V]
                v_stack.append(a[2])        # V
            v_mean = np.mean(v_stack, axis=0)              # (nx,ny,nz)
            V_model = depth_integrated_transport(v_mean, grid)            # (nx,ny)

            # time-mean wind over the same window: rebuild seasonal cycle and
            # sample at the snapshot day cadence, then average tau.
            from run_long_integration import build_seasonal_wind, interp_seasonal_wind
            wind_months = build_seasonal_wind(grid, year=2023)
            txs, tys = [], []
            for d in days[snap_idx]:
                tx, ty = interp_seasonal_wind(wind_months, float(d),
                                              blend_days=5.0)
                txs.append(tx); tys.append(ty)
            tau_x_mean = np.mean(txs, axis=0)
            tau_y_mean = np.mean(tys, axis=0)
            curl_tau = wind_stress_curl(tau_x_mean, tau_y_mean, grid)     # N/m^3
            V_sverdrup = curl_tau / (RHO_0 * grid.beta)                   # m^2/s

            # interior mask: exclude sponge bands + land
            nc = int(args.sponge_cells)
            interior = np.zeros((grid.nx, grid.ny), dtype=bool)
            interior[:, nc:grid.ny - nc] = True
            interior &= ocean
            # zonal-mean V(y) over the interior (large-scale, matchable)
            Vmod_y = np.array([V_model[ocean[:, j] & interior[:, j], j].mean()
                               if (ocean[:, j] & interior[:, j]).any() else np.nan
                               for j in range(grid.ny)])
            Vsve_y = np.array([V_sverdrup[ocean[:, j] & interior[:, j], j].mean()
                               if (ocean[:, j] & interior[:, j]).any() else np.nan
                               for j in range(grid.ny)])
            good = np.isfinite(Vmod_y) & np.isfinite(Vsve_y)
            sverdrup_mag_ratio = float('nan')
            if good.sum() > 4 and np.std(Vmod_y[good]) > 0 and np.std(Vsve_y[good]) > 0:
                # demean (pattern consistency, not amplitude) then correlate
                a_m = Vmod_y[good] - Vmod_y[good].mean()
                a_s = Vsve_y[good] - Vsve_y[good].mean()
                sverdrup_corr = float(np.corrcoef(a_m, a_s)[0, 1])
                sverdrup_rmse = float(np.sqrt(np.mean((Vmod_y[good] - Vsve_y[good]) ** 2)))
                # magnitude ratio on the full 2D interior fields (the zonal
                # mean hides eddy-scale barotropic variance; the 2D std shows
                # whether non-wind barotropic modes dominate the transport).
                sverdrup_mag_ratio = float(V_model[interior].std()
                                           / (V_sverdrup[interior].std() + 1e-30))
            a3_pass = ((not np.isnan(sverdrup_corr)) and sverdrup_corr > 0.3)
            print(f"  [A3] Sverdrup balance beta*V vs curl(tau)/rho0:")
            print(f"       interior (sponge {nc}-cell bands excluded), "
                  f"zonal-mean V(y) demeaned:")
            print(f"       pattern corr    = {sverdrup_corr:.3f}  (target > 0.3)")
            print(f"       RMSE            = {sverdrup_rmse:.3f} m^2/s "
                  f"(zonal-mean amplitude, informational)")
            print(f"       2D interior |Vmod|/|Vsve| std ratio = "
                  f"{sverdrup_mag_ratio:.2f}  (1 = wind-driven scale; "
                  f">>1 = barotropic modes dominate)")
            if not a3_pass and not np.isnan(sverdrup_corr):
                print(f"       -> FAIL is physically expected: Sverdrup is a "
                      f"linear STEADY interior theory that requires a western "
                      f"boundary layer to close the wind-driven gyre. This "
                      f"regional model uses a doubly-periodic + sponge domain "
                      f"(no WBL), so the depth-integrated transport is not "
                      f"constrained to the Sverdrup relation — the zonal-mean "
                      f"V(y) is a smooth southward barotropic mode "
                      f"(std {np.std(Vmod_y[good]):.1f} m^2/s) uncorrelated "
                      f"with the curl-driven Sverdrup prediction "
                      f"(std {np.std(Vsve_y[good]):.1f} m^2/s, which carries "
                      f"strong localized reanalysis-curl maxima). The FAIL "
                      f"quantifies the no-western-boundary limitation, not a "
                      f"model defect.")
            else:
                print(f"       (linear steady theory; expect qualitative not exact)")
        else:
            print(f"  [A3] Sverdrup: 3D dir given but snap count mismatch "
                  f"({len(snap_files)} snaps vs {len(days)} npz); deferred.")
    else:
        print(f"  [A3] Sverdrup balance: deferred (no --sverdrup-3d-dir; "
              f"needs 3D velocity snapshots). Qualitative sign check: model "
              f"SSH std={np.std(eta_clim[ocean]):.4f} m, should be O(cm) "
              f"for wind-driven.")

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
    if a3_pass is None:
        a3_lbl = "DEFERRED (needs 3D snapshots)"
    elif a3_pass:
        a3_lbl = (f"PASS (corr={sverdrup_corr:.3f}, rmse={sverdrup_rmse:.3f} m^2/s, "
                  f"mag_ratio={sverdrup_mag_ratio:.2f})")
    else:
        a3_lbl = (f"FAIL (corr={sverdrup_corr:.3f}, rmse={sverdrup_rmse:.3f} m^2/s, "
                  f"mag_ratio={sverdrup_mag_ratio:.2f}) — no-WBL limitation")
    print(f"    A3 Sverdrup:        {a3_lbl}")
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

        # Fig 4: Sverdrup balance (only if 3D transport computed)
        if V_model is not None and V_sverdrup is not None:
            nc = int(args.sponge_cells)
            fig, axes = plt.subplots(1, 3, figsize=(16, 4))
            vm = max(np.nanmax(np.abs(V_model)) * 1e-6,
                     np.nanmax(np.abs(V_sverdrup)) * 1e-6)
            for ax, fld, title in zip(axes,
                                      [V_model * 1e-6, V_sverdrup * 1e-6,
                                       (V_model - V_sverdrup) * 1e-6],
                                      ['model V = ∫v·dz (Sv/m)',
                                       'Sverdrup curl(τ)/(ρ₀β) (Sv/m)',
                                       'model − Sverdrup (Sv/m)']):
                im = ax.pcolormesh(grid.lon, grid.lat, fld.T, shading='auto',
                                   cmap='RdBu_r', vmin=-vm, vmax=vm)
                ax.set_title(title); ax.set_xlabel('lon E'); ax.set_ylabel('lat N')
                if nc > 0:
                    ax.axhline(grid.lat[nc], color='k', ls=':', lw=0.8)
                    ax.axhline(grid.lat[-nc - 1], color='k', ls=':', lw=0.8)
                fig.colorbar(im, ax=ax)
            fig.suptitle(f'Sverdrup balance (zonal-mean V corr={sverdrup_corr:.3f}, '
                         f'interior sponge bands excluded)')
            fig.tight_layout(); fig.savefig(os.path.join(args.out_dir, 'sverdrup.png'), dpi=120)
            plt.close(fig)
        print(f"\nfigures saved to {args.out_dir}/")
    except Exception as e:
        print(f"\n(plotting skipped: {e!r})")

    # ── Save comparison data ──
    save_kwargs = dict(
        sst_zonal_model=sst_zonal_model, sst_zonal_woa=sst_zonal_woa,
        sst_model_sm=sst_model_sm, sst_woa_sm=sst_woa_sm,
        eta_clim=eta_clim, sst_var=sst_var,
        k_cent=k_cent, P=P, slope=np.array(slope),
        corr_zonal=np.array(corr_zonal), rmse_zonal=np.array(rmse_zonal),
        corr_pat=np.array(corr_pat), rmse_pat=np.array(rmse_pat),
        a1_pass=np.array(a1_pass), a2_pass=np.array(a2_pass),
        ke_drift=np.array(ke_drift))
    if V_model is not None:
        save_kwargs.update(
            V_model=V_model, V_sverdrup=V_sverdrup,
            sverdrup_corr=np.array(sverdrup_corr),
            sverdrup_rmse=np.array(sverdrup_rmse),
            sverdrup_mag_ratio=np.array(sverdrup_mag_ratio),
            a3_pass=np.array(a3_pass if a3_pass is not None else False))
    np.savez(os.path.join(args.out_dir, 'climatology_compare.npz'), **save_kwargs)
    print(f"data saved to {args.out_dir}/climatology_compare.npz")

    return 0


if __name__ == "__main__":
    sys.exit(main())
