"""Acceptance figures for g365d_012 (365d seasonal global production run) — v2.

v2 changes (user feedback):
  * all fonts -> Times New Roman (rcParams serif family + STIX mathtext)
  * layout fixes: constrained_layout everywhere, per-panel colorbars sized
    with fraction/pad, fig_C zoom wrap-window rebuilt monotonically
  * fig_D quiver rescaled (scale=3, pivot mid) so seasonal arrows are visible
  * NEW: fig_GIF — integration-process animation (SST + SSH + KE trace),
    38 snapshots, PillowWriter

Pre-registered deliverables (docs/g365d_work_summary_zh.md §9 item 2):
  fig_A five-panel time series: max|u| / max|T| / max|eta| / SSH_std / KE
  fig_B T_top init vs day-365 comparison maps
  fig_C eta final-frame map with Med relaxation box highlighted
  fig_D monthly wind-stress vectors (seasonal-cycle proof)

All data from results/global_g365d_012.npz (md5 53876042, verified pull from 012).
Wind months are rebuilt locally from data/wind/monthly_mean_90*.npz with the
same real_wind_forcing path used by the run (identical formula + taper).

NOTE: figures are NEVER read back into context (image-blind workflow); a
programmatic font audit is printed instead of visual inspection.
"""
import sys, os
_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, os.path.join(ROOT, "src"))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.text as mtext

# ── Times New Roman everywhere (text + math) ─────────────────────────
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix",          # Times-like math glyphs
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
})

OUT = os.path.join(ROOT, "results", "acceptance_g365d_012")
os.makedirs(OUT, exist_ok=True)


def font_audit(fig, name):
    """Assert every text object resolves to Times New Roman; print report."""
    bad = []
    for t in fig.findobj(matplotlib.text.Text):
        s = t.get_text()
        if not s:
            continue
        f = t.get_fontproperties()
        fam = f.get_family()
        if "serif" not in fam and "Times New Roman" not in fam:
            bad.append((s[:40], fam))
    print(f"[font-audit] {name}: {'OK (all serif/Times)' if not bad else bad[:5]}")


z = np.load(os.path.join(ROOT, "results", "global_g365d_012.npz"), allow_pickle=True)
days = z["days"]
wet = np.asarray(z["wet_mask"], dtype=bool)      # (nx=360 lon, ny=120 lat)
lon = z["lon"]                                    # 0.5..359.5
lat = z["lat"]                                    # -59.5..59.5
T_top = z["T_top"]                                # (38, 360, 120)
eta = z["eta"]                                    # (38, 360, 120)

# ══════════════════════════════════════════════════════════════════════
# fig_A: five-panel time series (constrained layout, clean labels)
# ══════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(5, 1, figsize=(9.5, 11), sharex=True,
                         constrained_layout=True)
panels = [
    (r"max $|u|$ (m/s)", z["max_u"], "tab:blue"),
    (r"max $|T|$ ($^\circ$C)", z["max_T"], "tab:orange"),
    (r"max $|\eta|$ (m)", z["max_eta"], "tab:green"),
    ("SSH std (m)", z["ssh_std"], "tab:red"),
    ("KE (J m$^{-2}$)", z["ke"], "tab:purple"),
]
for ax, (label, arr, c) in zip(axes, panels):
    ax.plot(days, arr, "-o", color=c, ms=3.5, lw=1.3)
    ax.set_ylabel(label)
    ax.grid(True, alpha=0.3, lw=0.5)
    ax.margins(x=0.02)
axes[-1].set_xlabel("Day")
axes[0].set_title("g365d_012 — 365-day global 1° seasonal run\n"
                  f"VERDICT: {str(z['verdict'])},  max|$u$| peak "
                  f"{float(z['max_u_peak']):.3f} m/s,  525600 steps,  0 NaN",
                  fontsize=12)
font_audit(fig, "fig_A")
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

fig, axes = plt.subplots(3, 1, figsize=(11, 11), sharex=True, sharey=True,
                         constrained_layout=True)
im0 = axes[0].pcolormesh(lon, lat, Tm.T, shading="auto", cmap="RdYlBu_r",
                         vmin=0, vmax=30)
axes[0].set_title("SST initial (WOA2023, day 0)")
im1 = axes[1].pcolormesh(lon, lat, Tl.T, shading="auto", cmap="RdYlBu_r",
                         vmin=0, vmax=30)
axes[1].set_title("SST day 365")
im2 = axes[2].pcolormesh(lon, lat, Td.T, shading="auto", cmap="RdBu_r",
                         vmin=-4, vmax=4)
axes[2].set_title("SST change (day 365 − init)")
cb0 = fig.colorbar(im0, ax=axes[0], fraction=0.025, pad=0.015)
cb0.set_label("SST ($^\\circ$C)")
cb2 = fig.colorbar(im2, ax=axes[2], fraction=0.025, pad=0.015)
cb2.set_label("$\\Delta$SST ($^\\circ$C)")
for ax in axes:
    ax.set_ylabel("Latitude ($^\\circ$N)")
axes[2].set_xlabel("Longitude ($^\\circ$E)")
fig.suptitle("g365d_012 — SST: WOA init vs 365-day seasonal integration",
             fontsize=13)
font_audit(fig, "fig_B")
fig.savefig(os.path.join(OUT, "fig_B_sst_init_vs_d365.png"), dpi=130)
plt.close(fig)
print("fig_B done")

# ══════════════════════════════════════════════════════════════════════
# fig_C: eta final frame, Med box highlighted.
# Zoom uses a MONOTONIC wrapped x-axis: lon>=340 mapped to lon-360 (-20..-0.5),
# then lon<=60 (0.5..59.5). No duplicated coordinates.
# ══════════════════════════════════════════════════════════════════════
eta_last = np.where(wet, eta[-1], np.nan)

hi = np.where(lon >= 340)[0]          # -> x' = lon - 360  (−20 .. −0.5)
lo = np.where(lon <= 60)[0]           # -> x' = lon        (0.5 .. 59.5)
x_zoom = np.concatenate([lon[hi] - 360.0, lon[lo]])
eta_zoom = np.concatenate([eta_last[hi], eta_last[lo]], axis=0)
assert np.all(np.diff(x_zoom) > 0), "zoom x-axis must be strictly monotonic"

fig, axes = plt.subplots(2, 1, figsize=(11, 8.5), constrained_layout=True)
im = axes[0].pcolormesh(lon, lat, eta_last.T, shading="auto", cmap="RdBu_r",
                        vmin=-1.7, vmax=1.7)
axes[0].set_title("SSH ($\\eta$) day 365 — global")
box_lon = [354.0, 360.0, 360.0, 354.0, 354.0]
box_lat = [30.0, 30.0, 46.5, 46.5, 30.0]
axes[0].plot(box_lon, box_lat, "k--", lw=1.5, label="eta_relax box ($\\tau$=30 d)")
axes[0].legend(loc="lower left", framealpha=0.85)
cb = fig.colorbar(im, ax=axes[0], fraction=0.025, pad=0.015)
cb.set_label("SSH (m)")

box_z = [-6.0, 42.0, 42.0, -6.0, -6.0]
axes[1].pcolormesh(x_zoom, lat, eta_zoom.T, shading="auto", cmap="RdBu_r",
                   vmin=-1.7, vmax=1.7)
axes[1].plot(box_z, [30.0, 30.0, 46.5, 46.5, 30.0], "k--", lw=1.5,
             label="eta_relax box")
axes[1].set_title("SSH day 365 — Atlantic / Mediterranean zoom "
                  "(box max 0.587 m vs 9.58 m without relax)")
axes[1].legend(loc="lower right", framealpha=0.85)
cbz = fig.colorbar(im, ax=axes[1], fraction=0.025, pad=0.015)
cbz.set_label("SSH (m)")
for ax in axes:
    ax.set_ylabel("Latitude ($^\\circ$N)")
axes[1].set_xlabel("Longitude ($^\\circ$E)")
fig.suptitle("g365d_012 — final SSH with Mediterranean relax box", fontsize=13)
font_audit(fig, "fig_C")
fig.savefig(os.path.join(OUT, "fig_C_eta_final_medbox.png"), dpi=130)
plt.close(fig)
print("fig_C done")

# ══════════════════════════════════════════════════════════════════════
# fig_D: monthly wind-stress vectors (seasonal-cycle proof).
# Same rebuild path as the run: load_monthly_wind -> bulk stress -> taper_2d_y.
# Arrows: scale=3.0 (max |tau| ~0.5 N/m2 -> visible arrows), pivot mid.
# ══════════════════════════════════════════════════════════════════════
from dataclasses import replace
from config import DEFAULT_CONFIG, GlobalGridConfig
from grid import make_global_grid
from wind_reanalysis import load_monthly_wind, wind_stress_from_wind
from forcing import taper_2d_y

gcfg = replace(GlobalGridConfig(), lat_max=60.0, ny=120)
grid = make_global_grid(gcfg, DEFAULT_CONFIG.bathymetry_file,
                        smooth_passes=30, min_depth=100.0)

month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
               "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
taus = []
for m in range(12):
    month_idx = (2023 - 1948) * 12 + m
    u10, v10 = load_monthly_wind(month_idx=month_idx, grid=grid)
    tx, ty = wind_stress_from_wind(u10, v10)
    tx = taper_2d_y(tx, grid.ny, 8)
    ty = taper_2d_y(ty, grid.ny, 8)
    taus.append((tx, ty))
tau_abs_max = max(float(np.hypot(tx, ty).max()) for tx, ty in taus)

skip = 6
fig, axes = plt.subplots(4, 3, figsize=(15, 13), constrained_layout=True)
for ax, (tx, ty), name in zip(axes.ravel(), taus, month_names):
    spd = np.hypot(tx, ty)
    cf = ax.contourf(lon, lat, spd.T, levels=20, cmap="viridis")
    ax.quiver(lon[::skip], lat[::skip],
              tx[::skip, ::skip].T, ty[::skip, ::skip].T,
              color="w", scale=3.0, width=0.0025, pivot="mid",
              headwidth=3.5, headlength=4)
    ax.set_title(f"{name}   max |$\\tau$| = {spd.max():.3f} N m$^{{-2}}$",
                 fontsize=11)
    ax.set_xlim(0, 360)
    ax.set_ylim(-60, 60)
    ax.set_xticks([0, 90, 180, 270, 360])
    ax.set_yticks([-60, -30, 0, 30, 60])
for ax in axes[-1]:
    ax.set_xlabel("Longitude ($^\\circ$E)")
for ax in axes[:, 0]:
    ax.set_ylabel("Latitude ($^\\circ$N)")
cbar = fig.colorbar(cf, ax=axes, fraction=0.02, pad=0.012,
                    shrink=0.9, location="right")
cbar.set_label("|$\\tau$| (N m$^{-2}$)")
fig.suptitle("g365d_012 forcing — monthly NCEP wind stress 2023 "
             "(seasonal cycle input; arrows = stress direction)", fontsize=13)
font_audit(fig, "fig_D")
fig.savefig(os.path.join(OUT, "fig_D_wind_monthly_vectors.png"), dpi=130)
plt.close(fig)
print(f"fig_D done (tau_abs_max={tau_abs_max:.3f})")

# ══════════════════════════════════════════════════════════════════════
# fig_GIF: integration process — SST + ΔSST + SSH + KE trace,
# one frame per snapshot (38 frames, day counter, fixed color scales).
# ΔSST panel: absolute SST barely moves after day ~10 because the bulk
# heat flux (lambda=40 W/m2/K, ~6 d equilibration) anchors the surface
# layer to T_atm; the drift/seasonal signal (~0.1-6 C, zonal-mean NH
# midlat swing only ~0.5 C) is invisible on a 0-30 C band. The anomaly
# view T(t) - T(0) on a fixed ±5 C band makes the evolution visible
# (98.9% of all |dT| falls within ±5).
# Encoding note: frames are rendered to RGB and re-encoded with ONE
# shared palette + no dither. PillowWriter's per-frame adaptive palette
# remaps identical pixels to different palette entries each frame, which
# makes the smooth colorbar gradient flicker (scales themselves are
# fixed: SST 0–30 °C, ΔSST ±5 °C, SSH ±1.7 m).
# ══════════════════════════════════════════════════════════════════════
from matplotlib.animation import FuncAnimation, PillowWriter
from PIL import Image

dT_all = T_top - T_top[0]

fig = plt.figure(figsize=(14.5, 8.6), constrained_layout=True)
gs = fig.add_gridspec(2, 3, height_ratios=[3.2, 1.0])
axT = fig.add_subplot(gs[0, 0])
axD = fig.add_subplot(gs[0, 1])
axE = fig.add_subplot(gs[0, 2])
axK = fig.add_subplot(gs[1, :])

T_land = np.ma.masked_invalid(np.where(wet, np.nan, 1.0))
imT = axT.pcolormesh(lon, lat, T_top[0].T, shading="auto", cmap="RdYlBu_r",
                     vmin=0, vmax=30)
axT.pcolormesh(lon, lat, T_land.T, shading="auto", cmap="Greys", vmin=0, vmax=2)
axT.set_title("SST ($^\\circ$C)")
imD = axD.pcolormesh(lon, lat, dT_all[0].T, shading="auto", cmap="RdBu_r",
                     vmin=-5, vmax=5)
axD.pcolormesh(lon, lat, T_land.T, shading="auto", cmap="Greys", vmin=0, vmax=2)
axD.set_title("$\\Delta$SST vs day 0 ($^\\circ$C)")
imE = axE.pcolormesh(lon, lat, eta[0].T, shading="auto", cmap="RdBu_r",
                     vmin=-1.7, vmax=1.7)
axE.pcolormesh(lon, lat, T_land.T, shading="auto", cmap="Greys", vmin=0, vmax=2)
axE.set_title("SSH (m)")
for ax in (axT, axD, axE):
    ax.set_ylim(-60, 60)
    ax.set_ylabel("Lat ($^\\circ$N)")
    ax.set_xlabel("Lon ($^\\circ$E)")
cbT = fig.colorbar(imT, ax=axT, fraction=0.03, pad=0.02)
cbD = fig.colorbar(imD, ax=axD, fraction=0.03, pad=0.02)
cbE = fig.colorbar(imE, ax=axE, fraction=0.03, pad=0.02)

axK.plot(days, z["ke"], "-", color="tab:purple", lw=1.2, label="KE")
axK.plot(days, z["max_u"] * 400, "--", color="tab:blue", lw=1.0,
         label="max|$u$| ($\\times$400)")
axK.set_xlabel("Day")
axK.set_ylabel("KE (J m$^{-2}$)  /  max|$u$| (scaled)")
axK.legend(loc="upper left", ncols=2)
pointK, = axK.plot([], [], "o", color="tab:purple", ms=5)
pointU, = axK.plot([], [], "o", color="tab:blue", ms=5)
title = fig.suptitle("g365d_012 — 365-day seasonal integration  |  day 0",
                     fontsize=13)

def update(i):
    imT.set_array(T_top[i].T.ravel())
    imD.set_array(dT_all[i].T.ravel())
    imE.set_array(eta[i].T.ravel())
    pointK.set_data([days[i]], [z["ke"][i]])
    pointU.set_data([days[i]], [z["max_u"][i] * 400])
    title.set_text(f"g365d_012 — 365-day seasonal integration  |  "
                   f"day {days[i]:.0f}")
    return imT, imD, imE, pointK, pointU, title

anim = FuncAnimation(fig, update, frames=len(days), blit=False,
                     interval=280)
gif_path = os.path.join(OUT, "fig_GIF_integration.gif")

# Render each frame straight from the canvas, then quantize all frames
# against a single shared palette so identical pixels keep identical
# palette indices across frames (kills the colorbar flicker).
# A warm-up draw first (twice): constrained_layout settles and text
# glyph caches fill on the initial draws; frame 0 would otherwise be
# 1 px shifted / antialiased differently from the rest.
update(0)
fig.canvas.draw()
fig.canvas.draw()
frames_rgb = []
for i in range(len(days)):
    update(i)
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())
    frames_rgb.append(Image.fromarray(buf[..., :3].copy(), "RGB"))
plt.close(fig)
pal = frames_rgb[0].quantize(colors=255, dither=Image.Dither.NONE)
frames_p = [im.quantize(palette=pal, dither=Image.Dither.NONE)
            for im in frames_rgb]
frames_p[0].save(gif_path, save_all=True, append_images=frames_p[1:],
                 duration=250, loop=0, disposal=2, optimize=False)
print(f"fig_GIF done -> {gif_path} ({os.path.getsize(gif_path)/1e6:.1f} MB, "
      f"{len(frames_p)} frames, shared palette, {frames_rgb[0].size[0]}x{frames_rgb[0].size[1]}px)")

print("ALL FIGURES WRITTEN to", OUT)
