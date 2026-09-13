"""Era-3 slope003 forensics: its blowup started EARLIER (yr~122-126). Check
the aabw/drake/pmoc around onset — Southern-Ocean origin?
"""
import numpy as np

d = np.load("C:/Users/zhen.luo/ocean_solver/_ana_tmp/spinC_slope003_analysis.npz")
yrs = d["yrs"]
print("slope003 yr 115-133 (yrly):")
for y in range(115, 134):
    i = int(np.argmin(np.abs(yrs - y)))
    print(f"  yr{yrs[i]:6.1f}  amoc={d['amoc'][i]:8.1f} 26n={d['amoc_26n'][i]:7.1f} "
          f"aabw={d['aabw'][i]:8.1f} pmoc={d['pmoc'][i]:7.1f} drake={d['drake'][i]:8.1f} "
          f"maxu={d['maxu'][i]:5.2f} sst={d['sst'][i]:6.3f} deepT={d['deepT'][i]:6.3f}")
