import numpy as np, sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
arr = np.load('../results/global_gm365d_3d/snap_00022.npy')
T, u, v, S = arr
def rho(T, S): return -2.0e-4*(T-15.0) + 7.6e-4*(S-35.0)
r = rho(T[302,70,:], S[302,70,:])
for kk in range(14):
    print(f'  k={kk:2d}: T={T[302,70,kk]:6.2f} S={S[302,70,kk]:7.3f} rho={r[kk]:+.3e}')
