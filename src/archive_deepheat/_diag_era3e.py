"""Era-3 conclusion data: trend table on matched windows with the instability
excluded, so the 0.005-optimum claim is airtight. Windows:
  slope003: yr 100-122 (last sane; AMOC<100 before onset)
  slope004: yr 100-138 (last sane before AMOC runaway)
  era-2 tags: yr 100-133 for strict comparability, plus 161-190 late era.
"""
import numpy as np

def series(tag):
    return np.load(f"C:/Users/zhen.luo/ocean_solver/_ana_tmp/spinC_{tag}_analysis.npz")

def trend(d, y0, y1, key):
    yrs = d["yrs"]
    sel = (yrs >= y0) & (yrs <= y1)
    return np.polyfit(yrs[sel], d[key][sel], 1)[0] if sel.sum() >= 4 else float("nan")

WIN = {
    "relax15":  (100, 133, 161, 190),
    "slope007": (100, 133, 161, 190),
    "slope005": (100, 133, 161, 190),
    "slope004": (100, 138, None, None),
    "slope003": (100, 122, None, None),
}

print(f"{'tag':<10} {'w':<9} {'deepT C/yr':>11} {'OHC ZJ/yr':>11} {'note'}")
for t, (a, b, c, dd) in WIN.items():
    d = series(t)
    print(f"{t:<10} {f'{a}-{b}':<9} {trend(d, a, b, 'deepT'):>11.5f} "
          f"{trend(d, a, b, 'ohc'):>11.2f}   "
          f"{'PRE-INSTABILITY (sane snaps only)' if c is None else 'stable whole era'}")
    if c:
        print(f"{t:<10} {f'{c}-{dd}':<9} {trend(d, c, dd, 'deepT'):>11.5f} "
              f"{trend(d, c, dd, 'ohc'):>11.2f}   late era")
