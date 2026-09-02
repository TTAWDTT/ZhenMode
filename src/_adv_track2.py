import numpy as np, sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
# Full variance track + full-field NaN onset
lat = np.linspace(-59.5, 59.5, 120)
for k in range(10, 28):
    arr = np.load(f'../results/global_gm365d_3d/snap_{k:05d}.npy')
    T, u, v, S = arr
    m = S[:, :, 0]
    var = ((m - m.mean())**2).mean()
    imin = np.unravel_index(np.argmin(m), m.shape)
    print(f'd{k*5:3d}: S var={var:8.2f} min={m.min():9.2f} at lon={imin[0]} lat={lat[imin[1]]:6.1f}')
