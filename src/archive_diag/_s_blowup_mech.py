import numpy as np, sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
lat = np.linspace(-59.5, 59.5, 120); lon = np.arange(360) + 0.5
# vertical S profiles at runaway cell over time
print('S(z) profile at cell (302,70) over days:')
for k in [20,22,23,24,25,26]:
    arr = np.load(f'../results/global_gm365d_3d/snap_{k:05d}.npy')
    T, u, v, S = arr
    print(f'  d{k*5:3d}: ' + ' '.join(f'{S[302,70,kk]:8.2f}' for kk in range(6)))
print('S(z) profile at cell (165,68):')
for k in [18,20,22,23,24,25,26]:
    arr = np.load(f'../results/global_gm365d_3d/snap_{k:05d}.npy')
    T, u, v, S = arr
    print(f'  d{k*5:3d}: ' + ' '.join(f'{S[165,68,kk]:8.2f}' for kk in range(6)))
# horizontal S at k=0 in d120 snap near (302,70)
arr = np.load('../results/global_gm365d_3d/snap_00024.npy')
T, u, v, S = arr
print('S[k=0] (i=297..307, j=67..74) at d120:')
for j in range(74,66,-1):
    print(f'  lat{lat[j]:5.1f}: ' + ' '.join(f'{S[i,j,0]:7.1f}' for i in range(297,308)))
