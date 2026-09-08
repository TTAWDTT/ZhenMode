# -*- coding: utf-8 -*-
"""AMOC + deep-T drift analysis of the 10-yr mode-split GM/Redi run.

Reads results/global_tenyr_ms_gm.npz (2D tables) and every 3D snapshot in
results/global_tenyr_ms_gm_3d/ (T,u,v,S per snap). Computes:
  1. Atlantic MOC streamfunction psi(lat, z) per snap -> time series of
     AMOC strength (max psi at 30-60N, 300-3000 m, Atlantic sector).
  2. Global MOC strength (Antarctic cell sanity).
  3. Deep-T drift: mean T over WET-AT-DEPTH cells below 1000 m, first vs
     last snap (the g3650d metric: +1.15 -> +0.02 C/10yr after the
     conv/w-init fixes). The 3-D wet mask (layer-resolved, below-floor
     ghost layers excluded) is rebuilt from ETOPO exactly as grid.py does
     (results/_wet3_g360x120.npy); a column-only mask would include the
     T_ref/no-flux ghost fill and overstate deep-T by up to +6.6 C at the
     bottom level.
Writes results/moc_tenyr_ms_gm.npz and results/moc_tenyr_ms_gm.png.
"""
import glob
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.stdout.reconfigure(encoding="utf-8")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NPZ = os.path.join(ROOT, "results", "global_tenyr_ms_gm.npz")
D3 = os.path.join(ROOT, "results", "global_tenyr_ms_gm_3d")
OUT = os.path.join(ROOT, "results", "moc_tenyr_ms_gm.npz")
PNG = os.path.join(ROOT, "results", "moc_tenyr_ms_gm.png")

R_EARTH = 6.371e6

d = np.load(NPZ, allow_pickle=True)
lat = d["lat"]          # (120,) deg
lon = d["lon"]          # (360,) deg
z = d["z"]              # (14,) m, negative down, index 0 = surface
wet = d["wet_mask"]     # (360, 120)
days = d["days"]
nz = z.size
nlat, nlon = lat.size, lon.size

# cell thicknesses from node depths (midpoint rule; surface/bottom one-sided)
dz = np.empty(nz)
dz[0] = z[0] - z[1]
dz[-1] = z[-2] - z[-1]
for k in range(1, nz - 1):
    dz[k] = 0.5 * (z[k - 1] - z[k + 1])

# zonal grid spacing per latitude row
dx = 2.0 * np.pi * R_EARTH * np.cos(np.deg2rad(lat)) / nlon   # (120,) m

# Atlantic sector mask (0-360 lon): 280E-360E plus 0-25E
atl_i = np.zeros(nlon, dtype=bool)
atl_i[(lon >= 280.0)] = True
atl_i[(lon < 25.0)] = True
# global MOC uses all wet columns; deep-T uses the 3-D layer-resolved wet
# mask rebuilt from ETOPO (results/_wet3_g360x120.npy, see grid.py) so the
# below-seafloor ghost fill in the saved snaps is excluded.
wet3 = (wet > 0.5)[:, :, None]   # (360,120,1) broadcast over z
W3_PATH = os.path.join(ROOT, "results", "_wet3_g360x120.npy")
if os.path.exists(W3_PATH):
    wet3_3d = np.load(W3_PATH) > 0.5     # (360,120,nz)
else:
    # Fallback: column-uniform mask (overstates deep-T via ghost fill)
    wet3_3d = np.broadcast_to((wet > 0.5)[:, :, None], (nlon, nlat, nz)).copy()
wet3_deep = wet3_3d & (z <= -1000.0)[None, None, :]
wet3_deep_atl = wet3_deep & atl_i[:, None, None]
wet3_2k = wet3_3d & (z >= -2000.0)[None, None, :]


def moc_psi(v, lons=None):
    """MOC streamfunction psi(j, k) [Sv], cumulative from surface down.

    psi(j,k) = sum_{k'<=k} sum_i v[i,j,k'] dx(j) dz(k')  / 1e6.
    Positive = northward transport above depth z(k) (upper overturning cell).
    """
    if lons is None:
        vI = np.where(wet3, v, 0.0)
    else:
        vI = np.where(lons[:, :, None] & wet3, v, 0.0)
    V = vI.sum(axis=0) * dx[:, None] * dz[None, :]     # (120, 14) m^3/s
    return np.cumsum(V, axis=1) / 1.0e6               # Sv


snap_files = sorted(glob.glob(os.path.join(D3, "snap_*.npy")))
print(f"{len(snap_files)} 3D snaps, days {days[0]:.0f}..{days[-1]:.0f}")

atl2 = np.broadcast_to(atl_i[:, None], (nlon, nlat)).copy()   # (360,120)

amoc_t = np.zeros(len(snap_files))      # Atlantic upper-cell max
amoc_lat = np.zeros(len(snap_files))
amoc_z = np.zeros(len(snap_files))
gmoc_t = np.zeros(len(snap_files))      # global upper cell
deepT = np.zeros(len(snap_files))
deepT_atl = np.zeros(len(snap_files))
T2k = np.zeros(len(snap_files))

deep_k = z <= -1000.0
psi_final = None
v_final = None

for n, f in enumerate(snap_files):
    s = np.load(f)                     # (4, nx, ny, nz) = T,u,v,S
    v = s[2]
    psi_a = moc_psi(v, lons=atl2)
    psi_g = moc_psi(v, lons=None)

    # AMOC index: max psi over 30-60N and depths 300-3000 m, Atlantic
    jband = (lat >= 30.0) & (lat <= 60.0)
    kband = (z <= -300.0) & (z >= -3000.0)
    sub = psi_a[np.ix_(jband, kband)]
    m = np.unravel_index(np.argmax(sub), sub.shape)
    amoc_t[n] = sub[m]
    amoc_lat[n] = lat[jband][m[0]]
    amoc_z[n] = z[kband][m[1]]

    jg = (lat >= -60.0) & (lat <= 60.0)
    kg = (z <= -100.0) & (z >= -4000.0)
    gmoc_t[n] = psi_g[np.ix_(jg, kg)].max()

    T = s[0]
    deepT[n] = np.nanmean(np.where(wet3_deep, T, np.nan))
    deepT_atl[n] = np.nanmean(np.where(wet3_deep_atl, T, np.nan))
    T2k[n] = np.nanmean(np.where(wet3_2k, T, np.nan))
    if n == len(snap_files) - 1:
        psi_final = psi_a
        v_final = v

# deep-T drift (C over the integration = per 10 yr)
drift = deepT[-1] - deepT[0]
drift_atl = deepT_atl[-1] - deepT_atl[0]
print(f"AMOC strength: day0 {amoc_t[0]:.2f} Sv -> final {amoc_t[-1]:.2f} Sv "
      f"(peak {amoc_t.max():.2f} Sv at day {days[amoc_t.argmax()]:.0f}, "
      f"lat {amoc_lat[amoc_t.argmax()]:.0f}N, z {amoc_z[amoc_t.argmax()]:.0f} m)")
print(f"  time-mean (yr 2-10): {amoc_t[days >= 730].mean():.2f} Sv")
print(f"Global upper MOC: final {gmoc_t[-1]:.2f} Sv (mean yr2-10 "
      f"{gmoc_t[days >= 730].mean():.2f} Sv)")
print(f"deep-T (z<=1000m) global: {deepT[0]:.3f} -> {deepT[-1]:.3f} C "
      f"(drift {drift:+.3f} C/10yr)")
print(f"deep-T Atlantic: {deepT_atl[0]:.3f} -> {deepT_atl[-1]:.3f} C "
      f"(drift {drift_atl:+.3f} C/10yr)")

np.savez(OUT, days=days, amoc=amoc_t, amoc_lat=amoc_lat, amoc_z=amoc_z,
         gmoc=gmoc_t, deepT=deepT, deepT_atl=deepT_atl, T02k=T2k,
         psi_final=psi_final, lat=lat, z=z)

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
ax = axes[0]
ax.plot(days / 365.0, amoc_t)
ax.set_xlabel("year"); ax.set_ylabel("Sv")
ax.set_title("AMOC strength (Atl 30-60N, 0.3-3 km)")
ax.grid(alpha=0.3)
ax = axes[1]
pc = ax.contourf(lat, z, psi_final.T, levels=np.arange(-24, 25, 2.0),
                 cmap="RdBu_r", extend="both")
plt.colorbar(pc, ax=ax, label="Sv")
ax.set_title("Atlantic MOC psi, day 3660")
ax.set_xlabel("lat"); ax.set_ylabel("depth (m)")
ax.set_ylim(-4000, 0)
ax = axes[2]
ax.plot(days / 365.0, deepT, label="global")
ax.plot(days / 365.0, deepT_atl, label="Atlantic")
ax.plot(days / 365.0, T2k, label="0-2 km")
ax.set_xlabel("year"); ax.set_ylabel("C")
ax.set_title("mean T, wet-at-depth cells")
ax.legend(); ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(PNG, dpi=110)
print(f"saved {OUT}")
print(f"saved {PNG}")
