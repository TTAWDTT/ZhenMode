"""Close the surface heat budget for the local 10-yr run.

From the flux-profile diagnostic the total ocean tendency at yr 7-11 is
+153 ZJ/yr, and the seafloor no-flux BC makes that equal the surface input.
Here we compute what the two surface terms actually supply:
  bulk  = lambda_bulk * (T_atm - SST)   [W/m2]
  Q_heat (meridional)                    [W/m2]
and compare against the observed column tendency.
"""
import glob, sys
import numpy as np

sys.path.insert(0, "src")
from grid import make_global_grid, GlobalGridConfig
from config import DEFAULT_CONFIG
import forcing as F

RHO_0, C_P = 1025.0, 3992.0
SEC_YR = 3.15576e7
W2ZJ = 1.0 / 1e21 * SEC_YR          # W over 1 m2 -> ZJ/yr

g = make_global_grid(GlobalGridConfig(lat_max=60.0, ny=120),
                     DEFAULT_CONFIG.bathymetry_file, smooth_passes=30, min_depth=100.0)
AREA = np.asarray(g.dx_2d, float) * float(g.dy)
surf2d = np.asarray(g.wet_mask, float)          # (nx, ny) 2D surface ocean mask
print("ocean area %.3f e12 m2   frac %.4f" % (AREA.sum()/1e12, surf2d.mean()))

init = np.load("init_fields_g360x120.npz")
print("init keys", list(init.keys()))
T_init = init["T_init"].astype(np.float64)
print("T_init", T_init.shape, "surf mean over ocean %.4f" % T_init[:, :, 0][surf2d > 0.5].mean())

T_atm = F.air_temp_profile(g, T_init[:, :, 0])
print("T_atm shape", T_atm.shape, " range %.3f .. %.3f  ocean-mean %.4f"
      % (T_atm.min(), T_atm.max(), T_atm[surf2d > 0.5].mean()))

Q = F.heat_flux_meridional(g, Q0=50.0)
w = surf2d * AREA
print("Q_heat: domain mean %.6f  OCEAN mean %+.6f W/m2 = %+.2f ZJ/yr"
      % (Q.mean(), (Q * w).sum()/w.sum(), (Q * w).sum()*W2ZJ))

lam = F.BULK_LAMBDA_DEFAULT
print("BULK_LAMBDA_DEFAULT =", lam)

snaps = sorted(glob.glob("results/global_tenyr_ms_gm_3d/*.npy"))
print()
print("  yr    SST_ocean   dT=Tatm-SST   bulk[W/m2]   bulk[ZJ/yr]   Qheat[ZJ/yr]   total[ZJ/yr]   dOHC[ZJ/yr]")
OHC = np.zeros(len(snaps))
DZ = np.array([5,7.5,12.5,17.5,22.5,25,37.5,50,75,150,350,750,1500,2000], float)
T0 = np.load(snaps[0])[0]
wet3 = (np.abs(T0 - 15.0) > 1e-9).astype(float)
vol = wet3 * AREA[:, :, None] * DZ[None, None, :]
for i, f in enumerate(snaps):
    T = np.load(f)[0]
    OHC[i] = (T*vol).sum()*RHO_0*C_P/1e21
    if i % 12 == 0 or i == len(snaps)-1:
        sst = T[:, :, 0]
        dT = T_atm - sst
        bulk_Wm2 = lam * dT * surf2d
        bulk_ZJ = (bulk_Wm2 * AREA).sum() * W2ZJ
        qh_ZJ = (Q * w).sum() * W2ZJ
        a, b = max(0, i-6), min(len(snaps), i+7)
        rate = (OHC[b-1]-OHC[a])/(b-1-a)*12.0
        print("  %3d  %9.4f  %+10.4f  %+10.3f  %+12.2f  %+12.2f  %+12.2f  %+12.2f"
              % (i//12, sst[surf2d > 0.5].mean(), dT[surf2d > 0.5].mean(),
                 bulk_Wm2[surf2d > 0.5].mean(), bulk_ZJ, qh_ZJ, bulk_ZJ+qh_ZJ, rate))

print()
print("implied lambda needed to supply 153 ZJ/yr at the final state:")
sst = np.load(snaps[-1])[0][:, :, 0]
num = 153.0/W2ZJ                       # W total
den = ((T_atm - sst)*surf2d*AREA).sum()
print("   lambda_implied = %.3f W/m2/K  (configured %.1f)" % (num/den, lam))
