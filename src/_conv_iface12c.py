import numpy as np, sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
# Initial WOA is stable at iface 12 (0 cells). By d50, 17086 cells unstable
# there. This is MODEL-GENERATED. What process densifies 2000m relative to
# 4000m? Look at when it crosses: check a mid-depth cell (S/T evolution).
# Actually easier: at d50 check WHERE the iface-12 instabilities are:
arr = np.load('../results/global_gm365d_3d/snap_00010.npy')
T, u, v, S = arr
wmz3 = np.asarray(jnp.load if False else None)
def rho(T, S): return -2.0e-4*(T-15.0) + 7.6e-4*(S-35.0)
r = rho(T, S)
unst12 = r[:,:,12] > r[:,:,13]
import collections
lat = np.linspace(-59.5, 59.5, 120)
print('d50 iface-12 unstable:', unst12.sum())
# zonal band histogram
rows = unst12.sum(axis=0)
for j in range(0, 120, 10):
    print(f'  lat {lat[j]:6.1f}..{lat[min(j+9,119)]:6.1f}: {rows[j:j+10].sum()}')
# What do T and S look like at 2000 vs 4000 in these cells?
ii, jj = np.where(unst12)
sel = ii < 100  # first 100
print('sample cells (i, j, T12, T13, S12, S13):')
for a, b in list(zip(ii, jj))[::2000][:10]:
    print(f'  ({a},{b}) lat{lat[b]:6.1f}: T {T[a,b,12]:.2f}->{T[a,b,13]:.2f}  S {S[a,b,12]:.3f}->{S[a,b,13]:.3f}  rho {r[a,b,12]:.2e} -> {r[a,b,13]:.2e}')
