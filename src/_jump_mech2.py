import numpy as np, sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
d = np.load('../results/global_gm365d.npz', allow_pickle=True)
eta = d['eta']; days = d['days']
lat = np.linspace(-59.5, 59.5, 120); lon = np.arange(360) + 0.5
# correct j: lat 10.5N -> j=70
print('eta at lon 300-303, lat j=69..73 (9.5-13.5N) by day:')
for k in range(eta.shape[0]):
    if days[k] % 10 == 0 or days[k] >= 115:
        vals = ' '.join(f'{eta[k,i,j]:+.3f}' for j in (69,70,71,72,73) for i in (301,302))
        print(f'  d{days[k]:5.1f}: {vals}')
wet = d['wet_mask']
T_init = d['T_init']; S_init = d['S_init']
print('wet (i=297..307, j=67..75), # = ocean:')
for j in range(75,66,-1):
    print(f'  lat{lat[j]:5.1f}: ' + ''.join('#' if wet[i,j]>0.5 else '.' for i in range(297,308)))
print('T_init surf:')
for j in range(75,66,-1):
    print(f'  lat{lat[j]:5.1f}: ' + ' '.join(f'{T_init[i,j,0]:5.1f}' for i in range(297,308)))
