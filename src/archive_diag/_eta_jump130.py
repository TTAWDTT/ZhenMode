import numpy as np, sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
d = np.load('../results/global_gm365d.npz', allow_pickle=True)
print('keys:', list(d.keys()))
days = d['days']; eta = d['eta']
print('eta shape', eta.shape, 'last days:', days[-4:])
lat = np.linspace(-59.5, 59.5, 120)
lon = np.arange(360) + 0.5
for k in [-3, -2]:
    e = eta[k]
    af = np.abs(e).ravel(); idx = np.argsort(af)[::-1][:5]
    print(f'day {days[k]:.0f}: top-5 |eta|:')
    for ix in idx:
        i0, j0 = np.unravel_index(ix, e.shape)
        print(f'   |eta|={af[ix]:.4f} val={e[i0,j0]:+.4f} at lon={lon[i0]:.0f} lat={lat[j0]:.1f}')
de = eta[-1] - eta[-2]
af = np.abs(de).ravel(); idx = np.argsort(af)[::-1][:8]
print(f'delta eta d{days[-2]:.0f}->d{days[-1]:.0f} top-8:')
for ix in idx:
    i0, j0 = np.unravel_index(ix, de.shape)
    print(f'   d|eta|={de[i0,j0]:+.4f} at lon={lon[i0]:.0f} lat={lat[j0]:.1f}')
# also compare d125->d130 jump
de2 = eta[-2] - eta[-3]
af2 = np.abs(de2).ravel(); idx2 = np.argsort(af2)[::-1][:8]
print(f'delta eta d{days[-3]:.0f}->d{days[-2]:.0f} top-8:')
for ix in idx2:
    i0, j0 = np.unravel_index(ix, de2.shape)
    print(f'   d|eta|={de2[i0,j0]:+.4f} at lon={lon[i0]:.0f} lat={lat[j0]:.1f}')
