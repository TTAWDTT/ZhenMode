
import numpy as np, glob, os, sys
sys.path.insert(0, "/data/tmp/ocean/src")
from dataclasses import replace
from config import DEFAULT_CONFIG, GlobalGridConfig
from grid import make_global_grid
from wind_reanalysis import real_wind_forcing, wind_stress_from_wind
from run_long_integration_global import build_seasonal_wind_global, interp_seasonal_wind
def depth_integrated_transport(v_3d, grid):
    """Project bench convention: layer-centred v, partial bottom cell."""
    z = np.asarray(grid.z, float)
    depth = np.asarray(grid.depth, float)
    v_layer = 0.5 * (v_3d[:, :, :-1] + v_3d[:, :, 1:])
    z_shallow = z[:-1][None, None, :]
    z_deep = z[1:][None, None, :]
    z_sf = -depth[:, :, None]
    wet_deep = np.maximum(z_deep, z_sf)
    dz_wet = np.clip(z_shallow - wet_deep, 0.0, None)
    return np.sum(v_layer * dz_wet, axis=-1)



# --- rebuild the production grid (defaults: lat_max=60, ny=120, smooth=30, min_depth=100) ---
gcfg = replace(GlobalGridConfig(), lat_max=60.0, ny=120)
grid = make_global_grid(gcfg, DEFAULT_CONFIG.bathymetry_file,
                        smooth_passes=30, min_depth=100.0)
ocean = np.asarray(grid.ocean_mask, dtype=bool)
print("grid", grid.nx, grid.ny, "ocean %.3f" % ocean.mean())

# --- time-mean wind stress (ANNUAL mean of the 38 snapshot days) ---
dnpz = np.load("/data/tmp/ocean/results/global_g365d_013.npz")
days = np.asarray(dnpz["days"], dtype=float)
print("npz days:", days[:5], "...", days[-3:], "n=", len(days))
wind_months = build_seasonal_wind_global(grid, year=2023)
txs, tys = [], []
for d in days:
    tx, ty = interp_seasonal_wind(wind_months, float(d), blend_days=5.0)
    txs.append(tx); tys.append(ty)
tau_x_mean = np.mean(txs, axis=0)
tau_y_mean = np.mean(tys, axis=0)
print("tau_x range %.4f..%.4f  tau_y range %.4f..%.4f"
      % (tau_x_mean.min(), tau_x_mean.max(), tau_y_mean.min(), tau_y_mean.max()))

# --- curl(tau) on the spherical grid ---
dty_dx = (np.roll(tau_y_mean, -1, axis=0) - np.roll(tau_y_mean, 1, axis=0))          / (2.0 * grid.dx_2d)
dtx_dy = np.gradient(tau_x_mean, float(grid.dy), axis=1)
curl_tau = dty_dx - dtx_dy
OMEGA = 2 * np.pi / 86164.0
R_E = 6.371e6
beta_y = 2.0 * OMEGA * np.cos(np.radians(grid.lat)) / R_E
rho0 = 1025.0
V_sve = curl_tau / (rho0 * beta_y[None, :])
print("curl range %.3e..%.3e N/m^3" % (curl_tau.min(), curl_tau.max()))
print("V_sve range %.1f..%.1f m2/s" % (V_sve.min(), V_sve.max()))

# --- V_model: ANNUAL mean of all 38 snaps, project integration convention ---
fs = sorted(glob.glob("/data/tmp/ocean/results/global_g365d_013_3d/snap_*.npy"))
print("snaps:", len(fs))
v_stack = []
for f in fs:
    a = np.load(f, mmap_mode="r")
    v_stack.append(np.asarray(a[2], dtype=np.float64))
v_mean = np.mean(v_stack, axis=0)
V_mod = depth_integrated_transport(v_mean, grid)
print("V_mod range %.2f..%.2f m2/s" % (V_mod.min(), V_mod.max()))

# --- masks: exclude Gibraltar relax box, polar cap, and optionally WBC bands ---
lon2d, lat2d = np.meshgrid(grid.lon, grid.lat, indexing="ij")
box = (lat2d >= 29.0) & (lat2d <= 47.5)     & ((lon2d >= 353.0) | (lon2d <= 43.5))
cap = (lat2d > 55.0) | (lat2d < -55.0)
interior = ocean & (~box) & (~cap)
# WBC eastward-extension bands (Kuroshio 130-175E / Gulf Stream 280-330E at
# 30-45N): Sverdrup interior theory does not apply there.
wbc = np.zeros_like(interior)
wbm = (lat2d >= 28.0) & (lat2d <= 48.0)
wbc |= wbm & (lon2d >= 128.0) & (lon2d <= 178.0)   # Kuroshio
wbc |= wbm & (lon2d >= 278.0) & (lon2d <= 333.0)   # Gulf Stream
wbc |= wbm & (lon2d >= 300.0) & (lon2d <= 360.0) & (lon2d < 20.0)  # (wrap safety)
interior_nw = interior & (~wbc)
print("interior cells:", int(interior.sum()), "no-WBC:", int(interior_nw.sum()))

# --- zonal-mean profiles + corr helper ---
def zprof(V, m):
    return np.array([V[m[:, j], j].mean() if m[:, j].any() else np.nan
                     for j in range(grid.ny)])
lat_y = grid.lat
good = np.ones(grid.ny, dtype=bool)

Vmod_y = zprof(V_mod, interior)
Vsve_y = zprof(V_sve, interior)
good = np.isfinite(Vmod_y) & np.isfinite(Vsve_y)
a_m = Vmod_y[good] - Vmod_y[good].mean()
a_s = Vsve_y[good] - Vsve_y[good].mean()
corr = float(np.corrcoef(a_m, a_s)[0, 1])
rmse = float(np.sqrt(np.mean((Vmod_y[good] - Vsve_y[good]) ** 2)))
mag = float(V_mod[interior].std() / (V_sve[interior].std() + 1e-30))
print("A3 GLOBAL Sverdrup (annual): corr=%.3f  rmse=%.2f  2D std ratio=%.2f"
      % (corr, rmse, mag))
print("Vmod_y std=%.2f  Vsve_y std=%.2f" % (a_m.std(), a_s.std()))

def wcorr(mask_lat, lon_lo, lon_hi, m):
    sub = m & (lon2d >= lon_lo) & (lon2d < lon_hi)
    vm = zprof(V_mod, sub); vs = zprof(V_sve, sub)
    g = np.isfinite(vm) & np.isfinite(vs) & mask_lat
    if g.sum() <= 4: return float('nan'), int(g.sum())
    am = vm[g] - vm[g].mean(); as_ = vs[g] - vs[g].mean()
    if am.std() <= 0 or as_.std() <= 0: return float('nan'), int(g.sum())
    return float(np.corrcoef(am, as_)[0, 1]), int(g.sum())

nh = (lat_y >= 10) & (lat_y <= 55)
print("windowed zonal-mean corr (annual, interior excluding relax box + cap):")
c, n = wcorr((lat_y >= 20) & (lat_y <= 50), 0, 360, interior); print("  20-50N all basins:   %+.3f (n=%d)" % (c, n))
c, n = wcorr((lat_y >= 15) & (lat_y <= 50), 120, 245, interior); print("  Pacific 15-50N:      %+.3f (n=%d)" % (c, n))
c, n = wcorr((lat_y >= 15) & (lat_y <= 50), 260, 361, interior); print("  Atlantic 15-50N:     %+.3f (n=%d)" % (c, n))
# interior-only (WBC bands removed) -- the real Sverdrup test
c, n = wcorr((lat_y >= 15) & (lat_y <= 50), 120, 245, interior_nw); print("  Pacific 15-50N noWBC: %+.3f (n=%d)" % (c, n))
c, n = wcorr((lat_y >= 15) & (lat_y <= 50), 260, 361, interior_nw); print("  Atlantic 15-50N noWBC: %+ .3f (n=%d)" % (c, n))
c, n = wcorr(nh, 0, 360, interior_nw); print("  N-hem 10-55N noWBC:  %+.3f (n=%d)" % (c, n))

# --- pointwise 2D ---
vm = V_mod[interior]; vs = V_sve[interior]
am = vm - vm.mean(); as_ = vs - vs.mean()
c2d = float(np.corrcoef(am, as_)[0, 1])
print("pointwise 2D interior corr: %+.3f (std_m=%.2f std_s=%.2f)" % (c2d, am.std(), as_.std()))
vmn = V_mod[interior_nw & (lat2d >= 10) & (lat2d <= 55)]
vsn = V_sve[interior_nw & (lat2d >= 10) & (lat2d <= 55)]
amn = vmn - vmn.mean(); asn = vsn - vsn.mean()
c2dn = float(np.corrcoef(amn, asn)[0, 1])
print("pointwise 2D N-hem noWBC corr: %+.3f (std_m=%.2f std_s=%.2f)" % (c2dn, amn.std(), asn.std()))

np.savez("/data/tmp/ocean/results/sverdrup_global_g365d_013.npz",
         lat=lat_y, Vmod_y=Vmod_y, Vsve_y=Vsve_y, corr=corr, rmse=rmse,
         mag_ratio=mag, good=good, V_mod=V_mod, V_sve=V_sve,
         interior=interior, interior_nw=interior_nw,
         corr_2d=c2d, corr_2d_nw_nhem=c2dn)
print("saved /data/tmp/ocean/results/sverdrup_global_g365d_013.npz")
import hashlib
h = hashlib.md5(open("/data/tmp/ocean/results/sverdrup_global_g365d_013.npz","rb").read()).hexdigest()
print("MD5", h)
print("DONE")
