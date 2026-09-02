import numpy as np, sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
d = np.load('../results/global_gm365d.npz', allow_pickle=True)
eta = d['eta']; days = d['days']
lat = np.linspace(-59.5, 59.5, 120); lon = np.arange(360) + 0.5
# eta history at jump cell (302, 11) across all frames
print('eta at (302,10..13) by day:')
for k in range(eta.shape[0]):
    if days[k] % 10 == 0 or days[k] >= 120:
        vals = ' '.join(f'{eta[k,i,j]:+.3f}' for i in (300,301,302,303) for j in (10,11,12))
        print(f'  d{days[k]:5.1f}: {vals}')
# growth fit
e_cell = eta[:,302,11]
mask = days >= 60
import numpy as np
p = np.polyfit(days[mask], np.log(np.abs(e_cell[mask])+1e-6), 1)
print(f'log-fit d60+: slope={p[0]:.4f}/d -> e-folding={1/p[0]:.1f} d (if exp)')
# wet mask + depth at region
wet = d['wet_mask']
print('wet(298..306, 7..15):')
for j in range(15,6,-1):
    print(f'  lat{lat[j]:5.1f}: ' + ''.join('#' if wet[i,j]>0.5 else '.' for i in range(297,307)))
T_init = d['T_init']
print('T_init surf (297..306, j=9..14):')
for j in range(14,8,-1):
    print(f'  lat{lat[j]:5.1f}: ' + ' '.join(f'{T_init[i,j,0]:5.1f}' for i in range(297,307)))
