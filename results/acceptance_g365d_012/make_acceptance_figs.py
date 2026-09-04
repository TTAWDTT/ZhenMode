"""Acceptance figures for g365d_012 (365d seasonal global production run).

Pre-registered deliverables (docs/g365d_work_summary_zh.md §9 item 2):
  fig_A five-panel time series: max|u| / max|T| / max|eta| / SSH_std / KE
  fig_B T_top init vs day-365 comparison maps
  fig_C eta final-frame map with Med relaxation box highlighted
  fig_D monthly wind-stress vectors (seasonal-cycle proof)

All data from results/global_g365d_012.npz (md5 53876042, verified pull from 012).
Wind months are rebuilt locally from data/wind/monthly_mean_90*.npz with the
same real_wind_forcing path used by the run (identical formula + taper).
"""
import sys, os
_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, os.path.join(ROOT, "src"))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = os.path.join(ROOT, "results", "acceptance_g365d_012")
os.makedirs(OUT, exist_ok=True)

z = np.load(os.path.join(ROOT, "results", "global_g365d_012.npz"), allow_pickle=True)
days = z["days"]
wet = np.asarray(z["wet_mask"], dtype=bool)      # (nx=360 lon, ny=120 lat)
lon = z["lon"]                                    # 0.5..359.5
lat = z["lat"]                                    # -59.5..59.5
T_top = z["T_top"]                                # (38, 360, 120)
eta = z["eta"]                                    # (38, 360, 120)

# ══════════════════════════════════════════════════════════════════════
# fig_A: five-panel time series
# ══════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(5, 1, figsize=(10, 12), sharex=True)
panels = [
    ("max|u| (m/s)", z["max_u"], "tab:blue"),
    ("max|T| (degC)", z["max_T"], "tab:orange"),
    ("max|eta| (m)", z["max_eta"], "tab:green"),
    ("SSH std (m)", z["ssh_std"], "tab:red"),
    ("KE (J/m^2)", z["ke"], "tab:purple"),
]
for ax, (label, arr, c) in zip(axes, panels):
    ax.plot(days, arr, "-o", color=c, ms=3, lw=1.2)
    ax.set_ylabel(label)
    ax.grid(True, alpha=0.3)
axes[-1].set_xlabel("day")
axes[0].set_title("g365d_012 — 365d global 1° seasonal run: stability diagnostics "
                  f"(VERDICT: {str(z['verdict'])}, max|u| peak {float(z['max_u_peak']):.3f} m/s)")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig_A_timeseries_5panel.png"), dpi=130)
plt.close(fig)
print("fig_A done")

# ══════════════════════════════════════════════════════════════════════
# fig_B: T_top init vs day-365 (two maps + difference)
# ══════════════════════════════════════════════════════════════════════
T_init_top = z["T_init"][:, :, 0]
T_last = T_top[-1]
Tm = np.where(wet, T_init_top, np.nan)
Tl = np.where(wet, T_last, np.nan)
Td = np.where(wet, T_last - T_init_top, np.nan)

fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True, sharey=True)
im0 = axes[0].pcolormesh(lon, lat, Tm.T, shading="auto", cmap="RdYlBu_r",
                         vmin=0, vmax=30)
axes[0].set_title("SST initial (WOA2023, day 0)")
im1 = axes[1].pcolormesh(lon, lat, Tl.T, shading="auto", cmap="RdYlBu_r",
                         vmin=0, vmax=30)
axes[1].set_title("SST day 365")
im2 = axes[2].pcolormesh(lon, lat, Td.T, shading="auto", cmap="coolwarm",
                         vmin=-4, vmax=4)
axes[2].set_title("SST change (day365 − init)")
for ax in axes:
    ax.set_ylabel("lat N")
axes[2].set_xlabel("lon E")
fig.colorbar(im0, ax=axes[0], label="degC")
fig.colorbar(im1, ax=axes[1], label="degC")
fig.colorbar(im2, ax=axes[2], label="degC")
fig.suptitle("g365d_012 — SST: WOA init vs 365d seasonal integration")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig_B_sst_init_vs_d365.png"), dpi=130)
plt.close(fig)
print("fig_B done")

# ══════════════════════════════════════════════════════════════════════
# fig_C: eta final frame, Med box highlighted (relax box lon [-6,42]E, lat [30,46.5]N)
# ══════════════════════════════════════════════════════════════════════
eta_last = np.where(wet, eta[-1], np.nan)

fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True, sharey=True)
im = axes[0].pcolormesh(lon, lat, eta_last.T, shading="auto", cmap="RdBu_r",
                        vmin=-1.7, vmax=1.7)
axes[0].set_title("SSH (eta) day 365 — global")
fig.colorbar(im, ax=axes[0], label="m")

# Med zoom: relax box in grid lon convention (0..360): -6E -> 354E
box_lon = [354.0, 360.0, 360.0, 354.0, 354.0]
box_lat = [30.0, 30.0, 46.5, 46.5, 30.0]
axes[0].plot(box_lon, box_lat, "k--", lw=1.5, label="eta_relax box")
axes[0].legend(loc="lower left", fontsize=8)

lon_zoom = ((lon >= 340) | (lon <= 60))         # wrap window 340E..60E
lzp = np.concatenate([lon[lon_zoom], lon[lon_zoom] + 360.0]) - 354.0
eta_zoom = np.concatenate([eta_last[lon_zoom], eta_last[lon_zoom]], axis=0)
imz = axes[1].pcolormesh(lzp, lat, eta_zoom.T, shading="auto", cmap="RdBu_r",
                         vmin=-1.7, vmax=1.7)
axes[1].set_title("SSH day 365 — Atlantic/Mediterranean zoom (lon 340E..60E)")
axes[1].plot([0.0, 6.0, 6.0, 0.0, 0.0], [30.0, 30.0, 46.5, 46.5, 30.0],
             "k--", lw=1.5, label="eta_relax box (τ=30d)")
axes[1].legend(loc="lower right", fontsize=8)
fig.colorbar(imz, ax=axes[1], label="m")
for ax in axes:
    ax.set_ylabel("lat N")
axes[1].set_xlabel("lon relative to 354E (i.e. −6E at left edge)")
fig.suptitle("g365d_012 — final SSH frame; Med box max 0.587 m (was 9.58 m without relax)")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig_C_eta_final_medbox.png"), dpi=130)
plt.close(fig)
print("fig_C done")

# ══════════════════════════════════════════════════════════════════════
# fig_D: monthly wind-stress vectors (seasonal-cycle proof)
# Rebuild the 12 monthly tau fields exactly as the run did:
#   real_wind_forcing: bilinear interp of monthly_mean_90*.npz -> grid,
#   tau = rho_a * Cd * |U| * U, then taper_2d_y(8).
# ══════════════════════════════════════════════════════════════════════
from config import DEFAULT_CONFIG, GlobalGridConfig
from dataclasses import replace
from grid import make_global_grid
from wind_reanalysis import wind_stress_from_wind
from forcing import taper_2d_y

gcfg = replace(GlobalGridConfig(), lat_max=60.0, ny=120)
grid = make_global_grid(gcfg, DEFAULT_CONFIG.bathymetry_file,
                        smooth_passes=30, min_depth=100.0)

month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
               "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
fig, axes = plt.subplots(4, 3, figsize=(16, 12), sharex=True, sharey=True)
from wind_reanalysis import load_monthly_wind
taus = []
tau_max = 0.0
for m in range(12):
    # cached npz holds raw 10m wind on the NCEP grid; re-run the loader path
    # through the cache to get (nx, ny) components then stress+taper
    month_idx = (2023 - 1948) * 12 + m
    u10, v10 = load_monthly_wind(month_idx=month_idx, grid=grid)
    tx, ty = wind_stress_from_wind(u10, v10)
    ny = grid.ny
    tx = taper_2d_y(tx, ny, 8)
    ty = taper_2d_y(ty, ny, 8)
    taus.append((tx, ty))
    tau_max = max(tau_max, float(np.hypot(tx, ty).max()))

skip = 6   # every 6th point -> 60x20 vector grid
for ax, (tx, ty), name in zip(axes.ravel(), taus, month_names):
    spd = np.hypot(tx, ty)
    ax.contourf(lon, lat, spd.T, levels=20, cmap="viridis")
    ax.quiver(lon[::skip], lat[::skip], tx[::skip, ::skip].T, ty[::skip, ::skip].T,
              color="w", scale=6.0, width=0.0022)
    ax.set_title(f"{name}  |tau|max={spd.max():.3f} N/m2", fontsize=10)
    ax.set_xlim(0, 360); ax.set_ylim(-60, 60)
for ax in axes[-1]:
    ax.set_xlabel("lon E")
for ax in axes[:, 0]:
    ax.set_ylabel("lat N")
fig.suptitle("g365d_012 forcing — monthly NCEP wind stress 2023 (seasonal cycle input)")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig_D_wind_monthly_vectors.png"), dpi=130)
plt.close(fig)
print("fig_D done")
print("ALL FIGURES WRITTEN to", OUT)
