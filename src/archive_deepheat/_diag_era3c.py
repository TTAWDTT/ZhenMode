"""Era-3 forensics 2: where does slope004's instability live? Look at the
drake + aabw + pmoc series right around onset (yr 135-150), and compare the
Southern-Ocean cell — is this a Southern-Ocean barotropic-mode instability
(AABW sector) or a Northern one?
"""
import numpy as np

d = np.load("C:/Users/zhen.luo/ocean_solver/_ana_tmp/spinC_slope004_analysis.npz")
yrs = d["yrs"]
print("slope004 yr 130-152 (yrly):")
for y in range(130, 153):
    i = int(np.argmin(np.abs(yrs - y)))
    print(f"  yr{yrs[i]:6.1f}  amoc={d['amoc'][i]:8.1f} 26n={d['amoc_26n'][i]:7.1f} "
          f"aabw={d['aabw'][i]:8.1f} pmoc={d['pmoc'][i]:7.1f} drake={d['drake'][i]:8.1f} "
          f"maxu={d['maxu'][i]:5.2f} sst={d['sst'][i]:6.3f}")

d5 = np.load("C:/Users/zhen.luo/ocean_solver/_ana_tmp/spinC_slope005_analysis.npz")
y5 = d5["yrs"]
print("\nslope005 same window (for contrast):")
for y in range(130, 153, 3):
    i = int(np.argmin(np.abs(y5 - y)))
    print(f"  yr{y5[i]:6.1f}  amoc={d5['amoc'][i]:8.1f} 26n={d5['amoc_26n'][i]:7.1f} "
          f"aabw={d5['aabw'][i]:8.1f} pmoc={d5['pmoc'][i]:7.1f} drake={d5['drake'][i]:8.1f} "
          f"maxu={d5['maxu'][i]:5.2f} sst={d5['sst'][i]:6.3f}")
