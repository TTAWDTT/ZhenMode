"""Local: compare the SOUTHERN-OCEAN deep-T budget story — aabw strength and
deep heat content evolution for relax15 vs slope005, whole-era series. This
characterizes the equilibrium basin: is slope005 cooling toward a deep
equilibrium (good) or draining toward a new runaway (bad)?
"""
import numpy as np

for t in ["relax15", "slope007", "slope005"]:
    d = np.load(f"C:/Users/zhen.luo/ocean_solver/_ana_tmp/spinC_{t}_analysis.npz")
    yrs, deepT, ohc = d["yrs"], d["deepT"], d["ohc"]
    # whole-era polyfit (yr2..end) and decade-by-decade slope to see curvature
    m = yrs >= 2.0
    full = np.polyfit(yrs[m], deepT[m], 1)[0]
    print(f"\n{t}:  whole-era deepT slope {full:+.5f} C/yr")
    print("  decade slopes (C/yr):", end=" ")
    for y0 in range(5, 200, 15):
        mm = (yrs >= y0) & (yrs < y0 + 15)
        if mm.sum() >= 4:
            print(f"{y0}-{y0+15}:{np.polyfit(yrs[mm], deepT[mm], 1)[0]:+.4f}", end="  ")
    print()
    print(f"  deepT path: yr5={np.interp(5, yrs, deepT):.3f}  yr50={np.interp(50, yrs, deepT):.3f}  "
          f"yr100={np.interp(100, yrs, deepT):.3f}  yr150={np.interp(150, yrs, deepT):.3f}  "
          f"end={deepT[-1]:.3f}")
    # OHC same curvature check
    print(f"  OHC decade slopes (ZJ/yr):", end=" ")
    for y0 in range(5, 200, 15):
        mm = (yrs >= y0) & (yrs < y0 + 15)
        if mm.sum() >= 4:
            print(f"{y0}-{y0+15}:{np.polyfit(yrs[mm], ohc[mm], 1)[0]:+.2f}", end="  ")
    print()
