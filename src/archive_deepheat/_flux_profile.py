"""Vertical heat-flux profile from the local 10-yr snapshots.

Pure budget arithmetic: for level k, dOHC_k/dt = F[k-1/2] - F[k+1/2] (F>0
downward). Cumulating from the surface gives the flux at every interface,
with F[-1/2] = surface input. A conservative interior must return F -> ~0
at the seafloor. The cumulative overshoot above the surface flux is the
interior source/sink.
"""
import glob
import numpy as np

RHO_0, C_P = 1025.0, 3992.0
Z = np.array([0,-5,-15,-30,-50,-75,-100,-150,-200,-300,-500,-1000,-2000,-4000], float)
DZ = np.array([5,7.5,12.5,17.5,22.5,25,37.5,50,75,150,350,750,1500,2000], float)
ZC = np.array([-2.5,-10,-22.5,-40,-62.5,-87.5,-125,-175,-250,-400,-750,-1500,-3000,-4000.0])

snaps = sorted(glob.glob("results/global_tenyr_ms_gm_3d/*.npy"))
NT = len(snaps)
print(f"{NT} snapshots (monthly)")

import sys; sys.path.insert(0, "src")
from grid import make_global_grid, GlobalGridConfig
from config import DEFAULT_CONFIG
g = make_global_grid(GlobalGridConfig(lat_max=60.0, ny=120),
                     DEFAULT_CONFIG.bathymetry_file, smooth_passes=30, min_depth=100.0)
AREA = np.asarray(g.dx_2d, float) * float(g.dy)

snap0 = np.load(snaps[0])
T0 = snap0[0]
wet = (np.abs(T0 - 15.0) > 1e-9).astype(float)
vol = wet * AREA[:, :, None] * DZ[None, None, :]
V_lev = vol.sum(axis=(0, 1))
print("ocean vol %.3f e15 m3   wet3 %d" % (vol.sum()/1e15, int(wet.sum())))

# per-level OHC time series
OHC = np.zeros((NT, 14))
for i, f in enumerate(snaps):
    T = np.load(f)[0]
    OHC[i] = (T * vol).sum(axis=(0, 1)) * RHO_0 * C_P / 1e21
print("OHC[t0] %.1f ZJ  OHC[last] %.1f ZJ" % (OHC[0].sum(), OHC[-1].sum()))

# monthly tendency, smoothed over a year (12 months) to kill seasonal noise
d = np.zeros_like(OHC)
for i in range(NT):
    a, b = max(0, i-6), min(NT, i+7)
    d[i] = (OHC[b-1] - OHC[a]) / (b-1-a) * 12.0        # ZJ/yr

YRS = [(1, 4), (4, 7), (7, 11)]
print()
print("Per-level dOHC/dt [ZJ/yr], averaged over year windows:")
hdr = "  level  z(m)   " + "".join("%14s" % f"yr{a}-{b}" for a, b in YRS)
print(hdr)
for k in range(14):
    row = "  %5d %7.0f " % (k, ZC[k])
    for a, b in YRS:
        ia = min(NT-1, a*12); ib = min(NT, b*12)
        row += "%14.2f" % d[ia:ib, k].mean()
    print(row)

print()
print("Cumulative vertical heat flux F(z) [ZJ/yr, positive = DOWNWARD]")
print("  F[k+1/2] = sum_{j<=k} dOHC_j/dt   (relative to the surface input F[-1/2])")
for a, b in YRS:
    ia = min(NT-1, a*12); ib = min(NT, b*12)
    dm = d[ia:ib].mean(axis=0)
    F = np.concatenate([[0.0], np.cumsum(dm)])       # F at interfaces 0..14
    print(f"  --- years {a}-{b}: band totals {np.round(dm, 1)}")
    for k in range(15):
        zz = 0.0 if k == 0 else 0.5*(ZC[k-1] + (ZC[k] if k < 14 else -4200.0))
        print("      iface %2d  z %7.0f   F = %+9.2f ZJ/yr" % (k, zz, F[k]))
    print("      TOTAL column tendency = %+.2f ZJ/yr (excess over surface input)" % F[-1])

# What kappa_v and Redi would predict: F = -rho*cp*kappa*dT/dz
print()
print("Diffusive heat flux through each interface, from the snapshot T profile:")
Ta, Tb = np.load(snaps[12])[0], np.load(snaps[-1])[0]
pa = np.array([Ta[:, :, k][wet[:, :, k] > 0.5].mean() for k in range(14)])
pb = np.array([Tb[:, :, k][wet[:, :, k] > 0.5].mean() for k in range(14)])
for k in range(13):
    dTdz = (pb[k+1] - pb[k]) / ((ZC[k+1] - ZC[k]))
    F_kv = -RHO_0*C_P*1e-5*dTdz          # W/m2
    F_redi = -RHO_0*C_P*1.2e-3*dTdz
    A = AREA.sum()
    print("  iface %2d z %7.0f  dT/dz %+8.5f C/m   F(kv=1e-5) %+8.2f ZJ/yr   F(Redi Dv~1.2e-3) %+9.2f"
          % (k, 0.5*(ZC[k]+ZC[k+1]), dTdz,
             F_kv*A*3.15576e7/1e21, F_redi*A*3.15576e7/1e21))
