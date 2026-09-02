import numpy as np, sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
# rho = [-0.000923, -0.001092, ...]: rho[0] > rho[1] -> unstable at iface 0!
# rho[0]=-0.000923 > rho[1]=-0.001092. Surface layer DENSER than layer below.
# d2S/dz2 at top: (S[2]-2*S[1]+S[0]) with S=[34.87,34.61,34.84]:
# = 34.84 - 69.22 + 34.87 = +0.49 PSU. Positive -> conv_S>0 -> S INCREASES at surface.
# Hmm that's stabilizing for the dipole. But the dipole cell was (302,70).
# Check (302,70) conv mask + all unstable cells in the ITCZ:
arr = np.load('../results/global_gm365d_3d/snap_00022.npy')
T, u, v, S = arr
def rho(T, S): return -2.0e-4*(T-15.0) + 7.6e-4*(S-35.0)
lat = np.linspace(-59.5, 59.5, 120)
r302 = rho(T[302,70,:], S[302,70,:])
print('rho(302,70,:):', np.round(r302, 6)[:8])
print('unstable ifaces at (302,70):', np.where(r302[:-1] > r302[1:])[0])
# count surface-convective cells in the TROPICAL band and check their S tendency sign
r_all = rho(T, S)
wm = np.ones_like(r_all)
unstable0 = (r_all[:,:,:-1] > r_all[:,:,1:])
print('cells unstable at iface0 (surface):', unstable0[:,:,0].sum())
# For each, S[0] vs S[1]: if S[0] < S[1] (fresh cap), conv_S at k=0 is
# +kappa*(S[2]-2S[1]+S[0])/dz2 — mixes DOWN fresh water? No: positive d2 means
# S0 is LOW relative to mean -> conv_S>0 raises S0. Stabilizing.
# BUT WAIT: d2z at top boundary: (S[2] - 2*S[1] + S[0])/d2z_h0_bot^2
# For the DIPOLE cell (302,70): S=[25.5, 33.5, 35.0]:
d2S = S[302,70,2] - 2*S[302,70,1] + S[302,70,0]
print('dipole d2S:', d2S, '-> conv_S/dt =', 0.05*d2S/331.8*86400, 'PSU/day')
# positive: S0 rises toward S1. This DESTROYS the dipole -> stabilizing.
# So conv at the dipole cell damps. The 84.5/day at (167,67) is a different cell.
# Check (167,67): S=[34.87,34.61,34.84]: d2S = 34.84-69.22+34.87 = +0.49
d2S2 = S[167,67,2] - 2*S[167,67,1] + S[167,67,0]
print('cell167 d2S:', d2S2, '-> conv_S =', 0.05*d2S2/331.8*86400, 'PSU/day')
