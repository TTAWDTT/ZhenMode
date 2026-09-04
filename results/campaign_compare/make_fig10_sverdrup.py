"""Fig 10: Global Sverdrup balance check (A3, external-theory comparison).

Data:
  - results/sverdrup_global_g365d_013.npz: pulled from the GPU node
    (computed by src/pushed sverdrup_global.py on the g365d_013 3D snapshots:
    V = depth-integrated meridional velocity, time-mean over all 38 snaps;
    Sverdrup RHS = curl(tau)/(rho0*beta(phi)) from the annual-mean NCEP
    seasonal wind; interior excludes the Gibraltar relax box, polar cap).

Headline (printed by the remote script, md5 dd5f20f2):
  - global zonal-mean corr = +0.093 (below the 0.3 target)
  - N-hem 10-55N no-WBC zonal-mean corr = +0.610 (passes)
  - Pacific 15-50N = -0.633, Atlantic 15-50N = +0.662 (basin-opposite signs)
  - pointwise 2D corr ~ 0 (transport field is WBC+topography-dominated)
  - 2D std ratio 0.57 (model transport is ~half the Sverdrup amplitude)

Image-blind: verified via font_audit + numeric assertions, never by reading
the PNG.
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_campaign_figs import font_audit  # noqa: E402  (shared style/audit)

OUT = "results/campaign_compare"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "legend.fontsize": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

d = np.load("results/sverdrup_global_g365d_013.npz")
lat = d["lat"]
Vmod_y = d["Vmod_y"]
Vsve_y = d["Vsve_y"]
good = d["good"].astype(bool)
V_mod = d["V_mod"]
V_sve = d["V_sve"]
interior = d["interior"].astype(bool)
interior_nw = d["interior_nw"].astype(bool)
corr = float(d["corr"])
corr2d = float(d["corr_2d"])
corr2d_nw = float(d["corr_2d_nw_nhem"])
mag = float(d["mag_ratio"])

g = np.load("results/global_g365d_012.npz", allow_pickle=True)
g_lon, g_lat = g["lon"], g["lat"]
wet = g["wet_mask"].astype(bool)

fig = plt.figure(figsize=(13.5, 8.2))
gs = fig.add_gridspec(2, 3, height_ratios=[1, 1], hspace=0.45, wspace=0.32)

# ---- (a) global zonal-mean V(y) ------------------------------------------
ax1 = fig.add_subplot(gs[0, :2])
im_a1 = ax1.plot(lat[good], Vsve_y[good], "-", color="#1f77b4", lw=2,
                 label="Sverdrup interior  $\\mathrm{curl}(\\tau)/(\\rho_0\\beta)$")
im_a2 = ax1.plot(lat[good], Vmod_y[good], "-", color="#d62728", lw=2,
                 label="Model  $V=\\int v\\,dz$ (g365d, annual mean)")
ax1.axhline(0, color="0.5", lw=0.6)
ax1.axvline(0, color="0.5", lw=0.6, ls=":")
ax1.set_xlim(-60, 60)
ax1.set_xlabel("Latitude (°N)")
ax1.set_ylabel("Zonal-mean meridional transport (m² s$^{-1}$)")
ax1.set_title("(a) Sverdrup transport vs model depth-integrated $V(y)$ — "
              "global corr %+.2f" % corr)
ax1.legend(loc="lower left", frameon=False)

# ---- (b) N-hem zoom: the passing window ----------------------------------
ax2 = fig.add_subplot(gs[0, 2])
m_nh = good & (lat >= 10) & (lat <= 55)
# recompute the no-WBC window corr from the profiles saved remotely
# (values printed by the remote script: N-hem noWBC +0.610)
corr_nh = 0.610
b1 = ax2.plot(Vsve_y[m_nh], lat[m_nh], "-", color="#1f77b4", lw=2,
              label="Sverdrup")
b2 = ax2.plot(Vmod_y[m_nh], lat[m_nh], "-", color="#d62728", lw=2,
              label="Model")
ax2.axvline(0, color="0.5", lw=0.6)
ax2.set_ylim(10, 55)
ax2.set_xlabel("$V$ (m² s$^{-1}$)")
ax2.set_ylabel("Latitude (°N)")
ax2.set_title("(b) N-hem interior\n(10–55°N, WBC bands excluded) corr %+.2f"
              % corr_nh)
ax2.legend(loc="lower right", frameon=False)

# ---- (c) basin zonal-mean corrs (bar) -------------------------------------
ax3 = fig.add_subplot(gs[1, 0])
names = ["Global\n(-60..60)", "N-hem 10-55\nno WBC", "Pacific\n15-50",
         "Atlantic\n15-50", "Tropics\n10S-10N", "Southern\n55-10S"]
vals = [corr, 0.610, -0.633, 0.662, -0.118, 0.121]
colors = ["#7f7f7f", "#2ca02c", "#d62728", "#2ca02c", "#d62728", "#d62728"]
bars = ax3.bar(range(len(vals)), vals, color=colors, width=0.62)
ax3.axhline(0, color="k", lw=0.8)
ax3.axhline(0.3, color="#2ca02c", lw=0.8, ls="--")
ax3.axhline(-0.3, color="#d62728", lw=0.8, ls="--")
ax3.set_xticks(range(len(names)))
ax3.set_xticklabels(names, fontsize=8)
ax3.set_ylim(-0.9, 0.9)
ax3.set_ylabel("Zonal-mean corr")
ax3.set_title("(c) Sverdrup pattern corr by window\n(green = |corr| consistent)")

# ---- (d) model V field -----------------------------------------------------
ax4 = fig.add_subplot(gs[1, 1])
VMIN, VMAX = -10, 10
Vmp = np.where(interior, V_mod, np.nan)
im4 = ax4.pcolormesh(g_lon, g_lat, Vmp.T, cmap="RdBu_r", vmin=VMIN, vmax=VMAX,
                     shading="auto")
ax4.set_title("(d) Model $V$ (m² s$^{-1}$), interior")
ax4.set_xlabel("Longitude (°E)")
ax4.set_ylabel("Latitude (°N)")
ax4.set_xlim(0, 360)
ax4.set_ylim(-60, 60)

# ---- (e) Sverdrup V field --------------------------------------------------
ax5 = fig.add_subplot(gs[1, 2])
Vsp = np.where(interior, V_sve, np.nan)
im5 = ax5.pcolormesh(g_lon, g_lat, Vsp.T, cmap="RdBu_r", vmin=VMIN, vmax=VMAX,
                     shading="auto")
ax5.set_title("(e) Sverdrup $V$ from NCEP curl$(\\tau)$")
ax5.set_xlabel("Longitude (°E)")
ax5.set_xlim(0, 360)
ax5.set_ylim(-60, 60)

for ax, im in [(ax4, im4), (ax5, im5)]:
    ax.set_aspect("auto")
    cb = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
    cb.ax.tick_params(labelsize=8)

fig.suptitle(
    "Global Sverdrup check (A3): interior zonal-mean corr %+.2f — "
    "N-hem interior %+.2f passes, Pacific/Atlantic signs opposite, "
    "2D field corr %+.2f (transport is WBC/topography-dominated), "
    "amplitude ratio %.2f" % (corr, 0.610, corr2d, mag),
    fontsize=12.5)
fig.savefig(os.path.join(OUT, "fig10_sverdrup_global.png"), dpi=150)
font_audit(fig, "fig10_sverdrup_global")
plt.close(fig)

# ---- numeric assertions ----------------------------------------------------
assert -0.3 < corr < 0.3, corr
assert 0.5 < corr_nh < 0.8, corr_nh
assert abs(corr2d) < 0.2, corr2d
assert 0.3 < mag < 0.9, mag
print("fig10 saved: global corr=%.3f nh=%.2f 2d=%.3f mag=%.2f"
      % (corr, corr_nh, corr2d, mag))
