"""Fig 9: External absolute-SSH verification (model eta vs satellite altimetry).

Data sources:
  - model: results/global_g365d_012.npz eta (production PASS run), time-mean
    over d250-365, surface.
  - obs absolute SSH: data/sla_npac/ssh_abs_npac_monthly_2012.npz
    (erdTAssh1day, TOPEX/Jason, 12 mid-month snapshots of 2012).
  - obs SLA anomaly: data/sla_npac/sla_npac_monthly_2023.npz (nesdisSSH1day).
  - obs geostrophic currents: data/sla_npac/geostrophic_obs_npac_monthly_2023.npz.

Headline (verified in analysis):
  - 2deg-smoothed annual-mean pattern corr = +0.928, demeaned RMSE 0.199 m,
    amplitude ratio 0.70.
  - KE front latitude: model 37.5N vs obs 35.8N.
  - WBC max speed: model 0.24 m/s (Kuroshio TC) vs obs jet core 0.28 m/s.
  - SLA (anomaly) pointwise corr ~ 0 -> expected: model has no mesoscale
    eddies at 1 deg; SLA is eddy-dominated. Included as the honest negative.

Image-blind: verified via font_audit + numeric assertions, never by reading
the PNG.
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import gaussian_filter

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

g = np.load("results/global_g365d_012.npz", allow_pickle=True)
g_lon, g_lat = g["lon"], g["lat"]
wet = g["wet_mask"].astype(bool)
em = g["eta"][g["days"] >= 250].mean(0)

# ---- absolute SSH comparison fields ------------------------------------
c = np.load("results/_ssh_abs_compare.npz")
EMw, SSHm, valid = c["model"], c["obs"], c["valid"].astype(bool)
GLON, GLAT = np.meshgrid(c["g_lon"], c["g_lat"], indexing="ij")
corr = float(c["corr"])
rmse = float(c["rmse"])

# Panels (a)/(b) share a demeaned scale: the model's eta is a relative
# surface displacement (its own mean ~ 0.14 m) while altimetry SSH is on
# the geoid-referenced absolute datum (mean ~ 0.91 m). The +0.77 m datum
# offset carries no physics — the comparable quantity is the spatial
# PATTERN, so both fields are demeaned over the valid window and plotted
# on one common (anomaly) scale. Panel (c) then shows the demeaned
# difference (pattern error, the thing corr=0.928 speaks to), not the raw
# offset-dominated difference.
_mu_m = EMw[valid].mean()
_mu_o = SSHm[valid].mean()
DATUM_OFFSET = _mu_o - _mu_m          # informational only
EMw_d = EMw - _mu_m
SSHm_d = SSHm - _mu_o

# ---- SLA anomaly panel (honest negative) --------------------------------
s = np.load("data/sla_npac/sla_npac_monthly_2023.npz")
sl = s["sla"] - np.nanmean(s["sla"], axis=(1, 2), keepdims=True)
sl_sm = gaussian_filter(np.nan_to_num(sl[0], nan=0.0), 8)  # Jan, 2-deg smoothed
sl_on_model = RegularGridInterpolator(
    (s["lat"], s["lon"]), sl_sm, bounds_error=False, fill_value=np.nan)(
    pts := np.stack([GLAT.ravel(), GLON.ravel()], -1)).reshape(GLON.shape)

# ---- geostrophic currents -----------------------------------------------
R, OM, GE = 6.371e6, 7.2921e-5, 9.81
f = 2 * OM * np.sin(np.deg2rad(g_lat))
dy = np.deg2rad(1.0) * R
dxr = np.deg2rad(1.0) * R * np.cos(np.deg2rad(g_lat))
dmg = np.gradient(em, axis=1) / dy
dmx = np.gradient(em, axis=0) / dxr[None, :]
ug = np.zeros_like(em)
vg = np.zeros_like(em)
ug[1:-1, 1:-1] = -(GE / f[None, 1:-1]) * dmg[1:-1, 1:-1]
vg[1:-1, 1:-1] = (GE / f[None, 1:-1]) * dmx[1:-1, 1:-1]
spd_m = np.where(wet, np.sqrt(ug**2 + vg**2), np.nan)
spd_ms = np.where(wet, gaussian_filter(np.nan_to_num(spd_m, nan=0.0), 2), np.nan)

go = np.load("data/sla_npac/geostrophic_obs_npac_monthly_2023.npz")
u_o = go["ugos"].mean(0)
v_o = go["vgos"].mean(0)
spd_o = np.sqrt(u_o**2 + v_o**2)
spd_os = gaussian_filter(np.nan_to_num(spd_o, nan=0.0), 8)

fig = plt.figure(figsize=(13.5, 8.6))
gs = fig.add_gridspec(2, 3, height_ratios=[1, 1], hspace=0.42, wspace=0.30)

# (a) model eta (demeaned anomaly)
ax1 = fig.add_subplot(gs[0, 0])
VMIN, VMAX = -0.8, 0.8
im1 = ax1.pcolormesh(GLON, GLAT, np.where(valid, EMw_d, np.nan),
                     cmap="RdYlBu_r", vmin=VMIN, vmax=VMAX, shading="auto")
ax1.set_title("(a) Model $\\eta$ anomaly (d250–365 mean)")
ax1.set_xlabel("Longitude (°E)")
ax1.set_ylabel("Latitude (°N)")

# (b) obs absolute SSH (same demeaned anomaly scale)
ax2 = fig.add_subplot(gs[0, 1])
im2 = ax2.pcolormesh(GLON, GLAT, np.where(valid, SSHm_d, np.nan),
                     cmap="RdYlBu_r", vmin=VMIN, vmax=VMAX, shading="auto")
ax2.set_title("(b) Observed SSH anomaly (2012 mean)")
ax2.set_xlabel("Longitude (°E)")

# (c) demeaned difference (pattern error)
ax3 = fig.add_subplot(gs[0, 2])
diff = np.where(valid, EMw_d - SSHm_d, np.nan)
im3 = ax3.pcolormesh(GLON, GLAT, diff, cmap="RdBu_r", vmin=-0.4, vmax=0.4,
                     shading="auto")
ax3.set_title("(c) Model − obs anomalies (corr %+.2f)" % corr)
ax3.set_xlabel("Longitude (°E)")

# (d) SLA anomaly (the honest negative)
ax4 = fig.add_subplot(gs[1, 0])
im4 = ax4.pcolormesh(GLON, GLAT, np.where(np.isfinite(sl_on_model), sl_on_model, np.nan),
                     cmap="RdBu_r", vmin=-0.15, vmax=0.15, shading="auto")
ax4.set_title("(d) Observed SLA anomaly, Jan 2023 (eddy-dominated)")
ax4.set_xlabel("Longitude (°E)")
ax4.set_ylabel("Latitude (°N)")

# (e) model geostrophic speed
ax5 = fig.add_subplot(gs[1, 1])
gi_w = np.where((g_lon >= 120) & (g_lon <= 180))[0]
gj_w = np.where((g_lat >= 15) & (g_lat <= 55))[0]
im5 = ax5.pcolormesh(GLON, GLAT, np.where(valid, spd_ms[np.ix_(gi_w, gj_w)], np.nan),
                     cmap="viridis", vmin=0, vmax=0.15, shading="auto")
ax5.set_title("(e) Model geostrophic speed")
ax5.set_xlabel("Longitude (°E)")

# (f) obs geostrophic speed (native 0.25-deg grid, smoothed to 2 deg)
ax6 = fig.add_subplot(gs[1, 2])
im6 = ax6.pcolormesh(go["lon"], go["lat"],
                     np.where(np.isfinite(spd_os), spd_os, np.nan),
                     cmap="viridis", vmin=0, vmax=0.15, shading="auto")
ax6.set_title("(f) Observed geostrophic speed (2023)")
ax6.set_xlabel("Longitude (°E)")

for ax, im in [(ax1, im1), (ax2, im2), (ax3, im3), (ax4, im4),
               (ax5, im5), (ax6, im6)]:
    ax.set_aspect("auto")
    cb = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
    cb.ax.tick_params(labelsize=8)

fig.suptitle(
    "External SSH verification (pattern comparison, datum offset %.2f m removed): "
    "2°-smoothed corr %+.2f, demeaned RMSE %.2f m, amplitude ratio 0.70 — "
    "eddy-band (d) is outside the model's remit" % (DATUM_OFFSET, corr, rmse),
    fontsize=13)
fig.savefig(os.path.join(OUT, "fig9_ssh_abs_verification.png"), dpi=150)
font_audit(fig, "fig9_ssh_abs_verification")
plt.close(fig)

# ---- numeric assertions --------------------------------------------------
assert corr > 0.9, corr
assert 0.15 < rmse < 0.25, rmse
# pattern-error fraction: demeaned diff mostly inside the ±0.4 scale
_frac = float(np.mean(np.abs(diff[valid]) > 0.4))
assert _frac < 0.15, _frac
print("fig9 saved: corr=%.3f rmse=%.3f datum_offset=%.2f |diff|>0.4: %.0f%%"
      % (corr, rmse, DATUM_OFFSET, _frac * 100))
