import numpy as np, sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
# Mechanism chain complete. Now: WHEN does the runaway start and does the
# conv mask persist at the dipole? Track the ITCZ dipole amplitude + conv
# column count over time:
lat = np.linspace(-59.5, 59.5, 120)
for k in [10, 12, 14, 16, 18, 20, 22, 24, 26]:
    arr = np.load(f'../results/global_gm365d_3d/snap_{k:05d}.npy')
    T, u, v, S = arr
    def rho(T, S): return -2.0e-4*(T-15.0) + 7.6e-4*(S-35.0)
    r = rho(T, S)
    # real surface instabilities (both layers wet, z<200m: k=0..2)
    n_sfc = int(((r[:,:,:-1] > r[:,:,1:]) & (r[:,:,:-1]!=0) & (r[:,:,1:]!=0))[:,:,:3].sum())
    # ITCZ dipole amplitude: min S in tropical band
    m = S[:, 55:85, 0]
    print(f'd{k*5:3d}: sfc unstable(ifaces 0-2)={n_sfc:6d}  tropical S_min={m.min():8.2f}  tropical S_var={((m-m.mean())**2).mean():7.2f}')
