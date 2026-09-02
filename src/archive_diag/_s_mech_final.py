import numpy as np, sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
# The ITCZ S dipole: stationarity + growth = local source, not advection.
# Candidates: (1) adv_S numerical (unphysical dS at near-zero u),
# (2) GM skew flux with huge S slopes at S_dipole (rho' formula uses S),
# (3) conv_S (kappa_conv * laplacian on unstable column).
# Check the actual density: does S=-30 PSU at 25C give UNSTABLE column?
# Linear EOS: rho' = a_T*(T-T_ref) + a_S*(S-S_ref)
import numpy as np
lat = np.linspace(-59.5, 59.5, 120)
arr = np.load('../results/global_gm365d_3d/snap_00020.npy')
T, u, v, S = arr
# T_ref, S_ref from solver
import re
src = open('jax_solver_global.py', encoding='utf-8').read()
mT = re.search(r'T_ref:\s*float\s*=\s*([-\d.e+]+)', src)
mS = re.search(r'S_ref:\s*float\s*=\s*([-\d.e+]+)', src)
print('T_ref,S_ref defaults:', mT.group(1) if mT else '?', mS.group(1) if mS else '?')
# find density_anomaly coefficients
ma = re.search(r'_density_anomaly.*?(?=def )', src, re.S)
print(ma.group(0)[:600] if ma else 'not found')
