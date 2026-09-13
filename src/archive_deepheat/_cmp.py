import numpy as np, sys
sys.path.insert(0, "src")
from grid import make_global_grid, GlobalGridConfig
from config import DEFAULT_CONFIG

g = make_global_grid(GlobalGridConfig(lat_max=60.0, ny=120),
                     DEFAULT_CONFIG.bathymetry_file, smooth_passes=30, min_depth=100.0)
AREA = np.asarray(g.dx_2d, float) * float(g.dy)
DZN = np.array([5, 7.5, 12.5, 17.5, 22.5, 25, 37.5, 50, 75, 150, 350, 750, 1500, 2000], float)
RHO_0, C_P = 1025.0, 3992.0

d = np.load("results/global_tenyr_ms_gm.npz")
wet2 = np.asarray(d["wet_mask"], float)
Tck = np.asarray(np.load("results/ckpt_tenyr_ms_gm.npz")["T"], float)
Ts0 = np.load("results/global_tenyr_ms_gm_3d/snap_00000.npy")[0].astype(float)
Ts1 = np.load("results/global_tenyr_ms_gm_3d/snap_00122.npy")[0].astype(float)

vol = wet2[:, :, None] * AREA[:, :, None] * DZN[None, None, :]
ZJ = RHO_0 * C_P / 1e21
vm = lambda T: (T * vol).sum() / vol.sum()
print("ckpt        volmeanT %.4f  OHC %.1f ZJ  max %.3f" % (vm(Tck), (Tck * vol).sum() * ZJ, Tck.max()))
print("snap00000   volmeanT %.4f  OHC %.1f ZJ  max %.3f" % (vm(Ts0), (Ts0 * vol).sum() * ZJ, Ts0.max()))
print("snap00122   volmeanT %.4f  OHC %.1f ZJ  max %.3f" % (vm(Ts1), (Ts1 * vol).sum() * ZJ, Ts1.max()))
print()
print("max|ckpt - snap122| = %.5f" % np.abs(Tck - Ts1).max())
print("max|ckpt - snap0|   = %.5f" % np.abs(Tck - Ts0).max())
print()
# is ckpt maybe on a DIFFERENT vertical or masked differently? check depths
below = np.asarray(g.wet_mask, float)[:, :, None] * 0 + 1.0
print("ckpt k13 mean  %.4f   snap122 k13 mean %.4f" % (Tck[:, :, 13].mean(), Ts1[:, :, 13].mean()))
print("ckpt k0  mean  %.4f   snap122 k0  mean %.4f" % (Tck[:, :, 0].mean(), Ts1[:, :, 0].mean()))
print("Tck range", Tck.min(), Tck.max())
