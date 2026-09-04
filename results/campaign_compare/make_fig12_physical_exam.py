# fig12: physical exam of the production run g365d_013 (global 1deg FD,
# 365 days). Four panels, all from the transferred analysis npz files:
#   (a) SST map (lon x lat) - is the thermal structure ocean-like?
#   (b) surface current speed + direction - is the wind-driven circulation
#       where it should be (equatorial currents, western boundary jets)?
#   (c) global MOC streamfunction (lat x depth) - overturning strength.
#   (d) T/S profiles day 0 vs day 365 - is the water column stable?
# Image-blind: font_audit + bbox harness + numeric assertions on the
# arrays themselves; no image reading.
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

HERE = os.path.dirname(os.path.abspath(__file__))

def font_audit(fig, name):
    bad = []
    for t in fig.findobj(matplotlib.text.Text):
        s = t.get_text()
        if not s:
            continue
        f = t.get_fontfamily()
        if not f or f[0] not in ("serif", "Times New Roman"):
            bad.append((s[:30], f))
    if bad:
        raise AssertionError(f"{name}: non-serif text found: {bad[:5]}")
    print(f"[font-audit] {name}: OK (all serif/Times)")

plt.rcParams["font.family"] = "Times New Roman"

phys = np.load(os.path.join(HERE, "phys_check_g365d_013.npz"))
moc = np.load(os.path.join(HERE, "moc2_g365d_013.npz"))

sst = phys["sst"]          # (360, 120) lon x lat
sss = phys["sss"]
spd = phys["spd_surf"]
usurf, vsurf = phys["usurf"], phys["vsurf"]
lat = moc["lat"]           # (120,) -60..60
lon = np.linspace(0.5, 359.5, 360)
zface = moc["zface"]       # (15,) depth face levels m
psi = moc["psi"]           # (120, 14) Sv
psi_atl, psi_pac = moc["psi_atl"], moc["psi_pac"]
Tprof0, Tprof1 = phys["Tprof0"], phys["Tprof1"]
Sprof1 = phys["Sprof1"]
# layer-center depths from the model's z_levels
zlev = np.array([0, -5, -15, -30, -50, -75, -100, -150, -200, -300, -500,
                 -1000, -2000, -4000], dtype=float)
zcen = -zlev                                  # (14,) positive depths

fig = plt.figure(figsize=(15.5, 10.8))
gs = GridSpec(2, 2, figure=fig, hspace=0.32, wspace=0.22,
              left=0.06, right=0.97, top=0.90, bottom=0.07)

# ---- (a) SST map ----------------------------------------------------------
ax = fig.add_subplot(gs[0, 0])
im = ax.pcolormesh(lon, lat, sst.T, cmap="RdYlBu_r", vmin=-2, vmax=30,
                   shading="auto", rasterized=True)
cb = fig.colorbar(im, ax=ax, pad=0.015, aspect=26)
cb.set_label("SST (deg C)", fontsize=9.5)
ax.set_title("(a) Sea-surface temperature, day 365", fontsize=11,
             fontweight="bold")
ax.set_xlabel("longitude (deg E)")
ax.set_ylabel("latitude (deg N)")
ax.set_xticks([0, 90, 180, 270, 360])
ax.set_xlim(0, 360)
ax.set_ylim(-60, 60)

# ---- (b) surface currents -------------------------------------------------
ax = fig.add_subplot(gs[0, 1])
q = ax.quiver(lon[::12], lat[::6], usurf[::12, ::6].T, vsurf[::12, ::6].T,
              color="#1f5fa8", scale=8, width=0.0022, alpha=0.85)
qk = ax.quiverkey(q, 0.82, 1.035, 0.3, "0.3 m/s", labelpos="E")
qk.text.set_fontsize(8.5)
ax.contourf(lon, lat, spd.T, levels=np.linspace(0, 0.7, 15),
            cmap="viridis", alpha=0.55, rasterized=True)
cb = fig.colorbar(plt.cm.ScalarMappable(
    norm=matplotlib.colors.Normalize(0, 0.7), cmap="viridis"),
    ax=ax, pad=0.015, aspect=26)
cb.set_label("surface speed (m/s)", fontsize=9.5)
ax.set_title("(b) Surface currents (arrows) + speed (shade)", fontsize=11,
             fontweight="bold")
ax.set_xlabel("longitude (deg E)")
ax.set_ylabel("latitude (deg N)")
ax.set_xticks([0, 90, 180, 270, 360])
ax.set_xlim(0, 360)
ax.set_ylim(-60, 60)

# ---- (c) MOC streamfunction ----------------------------------------------
ax = fig.add_subplot(gs[1, 0])
vabs = 0.5
lv = np.linspace(-vabs, vabs, 21)
pc = ax.contourf(lat, zface, psi.T, levels=lv, cmap="RdBu_r", extend="both",
                 rasterized=True)
cs = ax.contour(lat, zface, psi.T, levels=[-0.3, -0.2, -0.1, 0.1, 0.2, 0.3],
                colors="k", linewidths=0.5, alpha=0.6)
ax.clabel(cs, fmt="%.1f", fontsize=7)
ax.invert_yaxis()
ax.set_ylim(4000, 0)
cb = fig.colorbar(pc, ax=ax, pad=0.015, aspect=26)
cb.set_label("overturning (Sv)", fontsize=9.5)
ax.set_title("(c) Global MOC streamfunction, day 365", fontsize=11,
             fontweight="bold")
ax.set_xlabel("latitude (deg N)")
ax.set_ylabel("depth (m)")
ax.set_xlim(-60, 60)

# ---- (d) T/S profiles ------------------------------------------------------
ax = fig.add_subplot(gs[1, 1])
ax.plot(Tprof0, zcen, "o-", color="#d9822b", lw=1.6, ms=4.5,
        label="day 0", alpha=0.9)
ax.plot(Tprof1, zcen, "s-", color="#1f5fa8", lw=1.6, ms=4.5,
        label="day 365")
ax2 = ax.twiny()
ax2.plot(Sprof1, zcen, "^--", color="#2a7f62", lw=1.3, ms=4,
         label="S day 365")
ax2.set_xlim(34.5, 35.4)
ax2.set_xlabel("salinity (psu, top axis)", fontsize=9.5, color="#2a7f62")
ax2.tick_params(axis="x", colors="#2a7f62")
ax.set_xlabel("temperature (deg C)")
ax.set_ylabel("depth (m)")
ax.invert_yaxis()
ax.set_xlim(4, 19)
ax.set_ylim(4000, 0)
ax.legend(loc="lower right", fontsize=8.5, framealpha=0.9)
ax.grid(alpha=0.25)
ax.set_title("(d) Horizontally-averaged T/S profiles", fontsize=11,
             fontweight="bold")

fig.suptitle(
    "Physical exam of production run g365d_013 (global 1$^\\circ$ FD, "
    "365 days): ocean-like state, weak overturning",
    fontsize=13, fontweight="bold", y=0.975)

# ---- overlap harness (image-blind verification) --------------------------
fig.canvas.draw()
ren = fig.canvas.get_renderer()
TC = matplotlib.text.Text
texts = [(t.get_text().replace("\n", "/"),
          TC.get_window_extent(t, ren))
         for t in fig.findobj(matplotlib.text.Text) if t.get_text().strip()]
ov = []
for i in range(len(texts)):
    for j in range(i + 1, len(texts)):
        a, b = texts[i][1], texts[j][1]
        ix = min(a.x1, b.x1) - max(a.x0, b.x0)
        iy = min(a.y1, b.y1) - max(a.y0, b.y0)
        if ix > 2 and iy > 2:
            ov.append((texts[i][0][:28], texts[j][0][:28]))
print(f"bbox harness: texts={len(texts)} overlaps={len(ov)}")
for p in ov:
    print("   OVERLAP:", p)
assert not ov, f"{len(ov)} text overlaps remain"

# data assertions (image-blind verification)
assert sst.shape == (360, 120) and psi.shape == (120, 14)
assert 24 < np.nanmax(sst) < 30, "SST warm pool in realistic range"
assert -0.5 < psi.min() and psi.max() < 1.0, "MOC weak (~<1 Sv)"
assert np.abs(Tprof1 - Tprof0).max() < 2.0, "profile drift small"
assert np.nanmax(usurf) > 0.2 and np.nanmin(usurf) < -0.15, \
    "surface zonal flow has both signs (easterlies+westerlies)"
assert len(zcen) == 14 and zcen[0] < 20 and zcen[-1] > 3000

font_audit(fig, "fig12")
out = os.path.join(HERE, "fig12_physical_exam.png")
fig.savefig(out, dpi=140)
plt.close(fig)
print("fig12 done: SST map, surface currents, MOC, T/S profiles")
