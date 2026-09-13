"""Era-3 forensics: the naive window comparison above is polluted — slope003/004
both had barotropic-mode instabilities late in their runs (AMOC 1737 Sv?!).
Check the AMOC trend BEFORE the instability onset for slope004, and see when
the ψ values diverge from the physical range.
"""
import numpy as np

def series(tag):
    return np.load(f"C:/Users/zhen.luo/ocean_solver/_ana_tmp/spinC_{tag}_analysis.npz")

d4 = series("slope004")
yrs = d4["yrs"]
print("slope004 full AMOC trajectory (5-yr sampling):")
for y in range(100, 190, 5):
    i = int(np.argmin(np.abs(yrs - y)))
    print(f"  yr{yrs[i]:6.1f}  amoc={d4['amoc'][i]:8.1f}  26n={d4['amoc_26n'][i]:8.1f}  "
          f"maxu={d4['maxu'][i]:5.2f}  ke={d4['ke'][i]:9.1f}")

print("\nAMOC@30-60N first crosses 100 Sv at yr:",
      yrs[np.argmax(d4["amoc"] > 100)])
print("AMOC series sanity: how many snaps < 100 Sv:", int((d4["amoc"] < 100).sum()),
      "of", len(d4["amoc"]))

# pre-onset trend: use only snaps with amoc < 100 Sv
ok = d4["amoc"] < 100
sel = ok & (yrs >= 100.0)
print("\nslope004 PRE-ONSET trends (yr 100 -> last-sane, n=%d):" % sel.sum())
for k, unit in [("deepT", "C/yr"), ("ohc", "ZJ/yr")]:
    print(f"  {k}: {np.polyfit(yrs[sel], d4[k][sel], 1)[0]:+.5f} {unit}")
print(f"  last sane snap: yr {yrs[ok][-1]:.1f}  amoc={d4['amoc'][ok][-1]:.1f}  "
      f"26n={d4['amoc_26n'][ok][-1]:.1f}  ratio={d4['amoc_26n'][ok][-1]/d4['amoc'][ok][-1]:.3f}")

d3 = series("slope003")
ok3 = d3["amoc"] < 100
sel3 = ok3 & (d3["yrs"] >= 100.0)
print(f"\nslope003 PRE-ONSET trends (yr 100 -> last-sane, n={sel3.sum()}):")
for k in ("deepT", "ohc"):
    print(f"  {k}: {np.polyfit(d3['yrs'][sel3], d3[k][sel3], 1)[0]:+.5f}")
print(f"  last sane snap: yr {d3['yrs'][ok3][-1]:.1f}  amoc={d3['amoc'][ok3][-1]:.1f}")
