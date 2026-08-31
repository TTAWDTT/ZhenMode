import numpy as np, sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
# With the gate: iface-12 has 5341 REAL unstable cells at d50 (was 0 initially).
# (302,70) is one of them (bottom-wet = level 12, i.e. ~2000-4000m depth:
# level 12 = 2000m. wmz[302,70] = [1]*13+[0] -> bottom wet level = 12 (2000m).
# So iface 12 = level12 (2000m, WET) over level13 (GHOST).
# rho'(level12) = -3.38e-4 ... wait rho(302,70,12) vs rho(302,70,13):
# rho(302,70) printed: [-9.2e-3, -2.5e-3, -6.3e-4, -3.4e-4, -2.5e-4, -1.6e-4,
#   -8.8e-5, +2.1e-5, ...] only 8 values shown. iface 12 flagged unstable
# means rho[12] > rho[13]. Level 13 is GHOST: rho[13] = 0 (masked).
# rho[12] = ? If negative (light water), rho[12] > 0 = false. So rho[12]>0?
# Let me print the full profile:
arr = np.load('../results/global_gm365d_3d/snap_00022.npy')
T, u, v, S = arr
def rho(T, S): return -2.0e-4*(T-15.0) + 7.6e-4*(S-35.0)
r = rho(T[302,70,:], S[302,70,:])
print('full rho(302,70):', np.round(r, 7))
print('T:', np.round(T[302,70,:], 2))
print('S:', np.round(S[302,70,:], 3))
# rho[12] vs rho[13]: with masking rho[13]=0.
# unstable = rho[12] > 0? But rho[12] masked by wet too... rho[12] is wet
# (level 12 wet). rho[13]=0 (ghost). If rho[12] > 0 -> flagged.
# rho[12] > 0 means the 2000m water is DENSER than T_ref/S_ref reference.
