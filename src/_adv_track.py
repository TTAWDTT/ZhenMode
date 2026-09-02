import numpy as np, sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
lat = np.linspace(-59.5, 59.5, 120)
print('surface S_min track (whole domain, k=0 only):')
for k in range(0, 28):
    try:
        arr = np.load(f'../results/global_gm365d_3d/snap_{k:05d}.npy')
    except FileNotFoundError:
        break
    T, u, v, S = arr
    m = S[:, :, 0]
    imin = np.unravel_index(np.argmin(m), m.shape)
    print(f'  d{k*5:3d}: S_min={m.min():9.2f} at lon={imin[0]:3d} lat={lat[imin[1]]:6.1f}')
