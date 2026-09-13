"""OHC time series from a run's 3D snapshots (runs inside the container).

The PASS criteria in run_long_integration_global.py check max|u|, a T-drift
tolerance and |eta| -- NOT the ocean heat content.  Compute the actual OHC
series so a "PASS" run cannot hide a monotonic warm drift.

UNITS NOTE (bug fixed): the previous version fit OHC[ZJ] against t in DAYS but
labelled the slope ZJ/yr, understating every trend by 365x.  Everything here is
now in years, and the slope is cross-checked against the integrated rise.
"""
import sys, glob, json
import numpy as np

sys.path.insert(0, "/data/tmp/ocean/src")
from grid import make_global_grid, GlobalGridConfig
from config import DEFAULT_CONFIG

tag = sys.argv[1] if len(sys.argv) > 1 else "spinH_qfix"
g = make_global_grid(GlobalGridConfig(lat_max=60.0, ny=120),
                     DEFAULT_CONFIG.bathymetry_file, smooth_passes=30, min_depth=100.0)
AREA = np.asarray(g.dx_2d, float) * float(g.dy)
DZN = np.array([5, 7.5, 12.5, 17.5, 22.5, 25, 37.5, 50, 75, 150, 350, 750, 1500, 2000], float)
vol = np.asarray(g.wet_mask_3d, float) * AREA[:, :, None] * DZN[None, None, :]
CP = 1025.0 * 3992.0
print("tag=%s  ocean vol 1e18 = %.6f" % (tag, vol.sum() / 1e18))

# --- figure out the real snapshot spacing from the run's own metadata ---
days_between = None
try:
    d = np.load("/data/tmp/ocean/results/global_%s.npz" % tag)
    ks = list(d.keys())
    print("global npz keys:", ks)
    for k in ("days", "day", "t_days", "snap_days", "times"):
        if k in ks:
            arr = np.asarray(d[k], float).ravel()
            if arr.size > 2:
                days_between = float(np.median(np.diff(arr)))
                print("  using %s -> spacing %.2f days" % (k, days_between))
                break
    for k in ks:
        v = d[k]
        try:
            if getattr(v, "size", 0) in (1, 2, 3) and v.dtype.kind in "iuf":
                print("  meta %-16s = %s" % (k, np.asarray(v).ravel()[:3]))
        except Exception:
            pass
except Exception as e:
    print("no global npz metadata (%s)" % e)

files = sorted(glob.glob("/data/tmp/ocean/results/global_%s_3d/snap_*.npy" % tag))
print("n_snaps=%d" % len(files))
if not files:
    sys.exit(0)
if days_between is None:
    # fall back: spinH_qfix style runs write 365-day snaps
    days_between = 365.0
    print("  WARNING: spacing unknown, assuming %.0f days" % days_between)

rows = []
for i, f in enumerate(files):
    T = np.asarray(np.load(f)[0], float)
    yr = i * days_between / 365.0
    ohc = float(((T - 15.0) * vol).sum() * CP / 1e21)   # Tref=15, matches earlier work
    deep = np.array([(T[:, :, k] * vol[:, :, k]).sum() / vol[:, :, k].sum()
                     for k in range(14)])
    Tm = float((T * vol).sum() / vol.sum())
    rows.append((yr, ohc, Tm, deep))

Y = np.array([r[0] for r in rows]); O = np.array([r[1] for r in rows])
print("\n  i    yr      OHC/ZJ     dOHC/yr    vol-meanT   T@2000m  T@4000m")
for i in range(0, len(rows), max(1, len(rows) // 14)):
    yr, ohc, Tm, deep = rows[i]
    rate = ((O[i] - O[i - 1]) / (Y[i] - Y[i - 1])) if i else np.nan
    print("  %3d %7.2f  %+11.2f  %+9.2f   %8.4f  %7.4f  %7.4f"
          % (i, yr, ohc, rate, Tm, deep[12], deep[13]))
yr, ohc, Tm, deep = rows[-1]
print("  %3d %7.2f  %+11.2f  %+9.2f   %8.4f  %7.4f  %7.4f"
      % (len(rows) - 1, yr, ohc, (O[-1] - O[-2]) / (Y[-1] - Y[-2]), Tm, deep[12], deep[13]))

print("\nINTEGRATED: OHC %.2f -> %.2f  rise %+.2f ZJ over %.2f yr  => mean %+.3f ZJ/yr"
      % (O[0], O[-1], O[-1] - O[0], Y[-1] - Y[0], (O[-1] - O[0]) / (Y[-1] - Y[0])))

print("\ntrend fits (ZJ/yr) -- slope of OHC vs time-in-YEARS:")
for lab, a, b in [("full record", 0, len(Y)), ("last 50%", len(Y) // 2, len(Y)),
                  ("last 20%", int(0.8 * len(Y)), len(Y)),
                  ("last 10%", int(0.9 * len(Y)), len(Y)),
                  ("last 5snap", -5, len(Y))]:
    if b - a >= 3:
        print("  %-12s n=%3d  %+9.3f" % (lab, b - a, np.polyfit(Y[a:b], O[a:b], 1)[0]))
    else:
        print("  %-12s n=%3d  (too few)" % (lab, b - a))

print("\nT@2000m / T@4000m trend (C/yr):")
for k in (12, 13):
    y = np.array([r[3][k] for r in rows])
    for lab, a, b in [("full", 0, len(Y)), ("last10%", int(0.9 * len(Y)), len(Y))]:
        if b - a >= 3:
            print("  k=%2d %-8s %+10.6f" % (k, lab, np.polyfit(Y[a:b], y[a:b], 1)[0]))
    print("  k=%2d  first %.4f -> last %.4f  (delta %+.4f C)"
          % (k, y[0], y[-1], y[-1] - y[0]))
