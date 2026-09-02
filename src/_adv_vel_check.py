import numpy as np, sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
# Drift di=-3 cells/5d = 3*111 km / 432000 s = 0.00077 m/s westward
# That is NOT advection by the mean flow (u~0.2-1.9 m/s). The dipole is
# stationary-ish, drifting 0.8 mm/s. So the S field is NOT being advected;
# it is being GENERATED locally.
# Key test: is the total surface S variance growing? And where's the source?
lat = np.linspace(-59.5, 59.5, 120)
for k in [16, 18, 20, 22]:
    arr = np.load(f'../results/global_gm365d_3d/snap_{k:05d}.npy')
    T, u, v, S = arr
    m = S[:, :, 0]
    var = ((m - m.mean())**2).mean()
    print(f'd{k*5}: S_surf var={var:.2f} std={np.sqrt(var):.2f}')
