import numpy as np, sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
lat = np.linspace(-59.5, 59.5, 120); lon = np.arange(360) + 0.5
print('S_min location by snapshot (5d cadence):')
for k in [20,22,23,24,25,26]:
    arr = np.load(f'../results/global_gm365d_3d/snap_{k:05d}.npy')
    T, u, v, S = arr
    imin = np.unravel_index(np.argmin(S), S.shape)
    print(f'  d{k*5:3d}: S_min={S.min():9.2f} at i={imin[0]} lon={lon[imin[0]]:.0f} j={imin[1]} lat={lat[imin[1]]:.1f} k={imin[2]} | T there={T[imin]:.2f}')
    # time series at that cell
    print('        S[t, cell] =', ' '.join(f'{S[i,imin[1],imin[2]]:8.2f}' for i in range(max(0,k-4), k+1)))
