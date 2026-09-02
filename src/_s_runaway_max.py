import numpy as np, sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
lat = np.linspace(-59.5, 59.5, 120); lon = np.arange(360) + 0.5
# Follow the runaway cell (167,67) over time + its neighbors + wet pattern
print('S surf (i=162..172, j=64..70) at d90/d100/d110:')
for k in [18, 20, 22]:
    arr = np.load(f'../results/global_gm365d_3d/snap_{k:05d}.npy')
    T, u, v, S = arr
    print(f'--- d{k*5} ---')
    for j in range(70,63,-1):
        print(f'  lat{lat[j]:5.1f}: ' + ' '.join(f'{S[i,j,0]:8.2f}' for i in range(162,173)))
    imin = np.unravel_index(np.argmin(S[:,:,:3]), S.shape)
    print(f'  min S={S[imin]:.2f} at i={imin[0]} j={imin[1]} k={imin[2]}')
