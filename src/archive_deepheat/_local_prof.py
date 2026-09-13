import glob, numpy as np
d = "results/global_tenyr_ms_gm_3d"
snaps = sorted(glob.glob(d + "/*.npy"))
NT = len(snaps)
DZ = np.array([5,7.5,12.5,17.5,22.5,25,37.5,50,75,150,350,750,1500,2000], float)
ZB = np.concatenate([[0.0], -np.cumsum(DZ)])   # interface depths, 15 values
ZC = (ZB[:-1] + ZB[1:]) / 2.0                  # level centers

t0 = np.load(snaps[0])[0]
ghost = np.abs(t0 - 15.0) < 1e-9
wet3 = ~ghost
print("wet3 cells", wet3.sum(), "of", wet3.size)

# Need area. Rebuild grid locally? try: use equal-area approx from a global 360x120
# If local grid module imports without jax, use it.
try:
    import sys; sys.path.insert(0, "src")
    from grid import make_global_grid, GlobalGridConfig
    from config import DEFAULT_CONFIG
    g = make_global_grid(GlobalGridConfig(lat_max=60.0, ny=120),
                         DEFAULT_CONFIG.bathymetry_file, smooth_passes=30, min_depth=100.0)
    AREA = np.asarray(g.dx_2d, float) * float(g.dy)
    print("AREA ok", AREA.shape, AREA.sum()/1e12, "e12 m2")
except Exception as e:
    print("grid import failed:", e)
    AREA = None

def profile(T):
    return np.array([T[:, :, k][wet3[:, :, k]].mean() if wet3[:, :, k].any() else np.nan
                     for k in range(14)])

print()
print(" idx  zc(m)   T[t0]   T[last]   dT")
Tl = np.load(snaps[-1])[0]
p0, pl = profile(t0), profile(Tl)
for k in range(14):
    print("  %2d  %6.0f  %7.3f  %8.3f  %+7.3f" % (k, ZC[k], p0[k], pl[k], pl[k]-p0[k]))

if AREA is not None:
    vol = wet3 * AREA[:, :, None] * DZ[None, None, :]
    RC = 1025.0 * 3992.0
    print()
    print("  year   total_ZJ   0-100m   100-700m  700-2000m  2000-4000m")
    for i in range(0, NT, 12):
        T = np.load(snaps[i])[0]
        o = (T * vol).sum(axis=(0, 1)) * RC / 1e21
        print("  %5d  %8.2f  %8.2f  %8.2f  %9.2f  %9.2f" % (
            i//12, o.sum(), o[0:4].sum(), o[4:8].sum(), o[8:11].sum(), o[11:14].sum()))
