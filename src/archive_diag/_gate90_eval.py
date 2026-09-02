import numpy as np, sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
d = np.load('results/global_gate90_fix.npz')
days = d['days']; eta = d['eta']
e = np.abs(eta).reshape(len(days), -1).max(axis=1)
print(f'max|eta| over run = {e.max():.3f} m  (criterion < 3.0)')
def idx(dd): return int(np.argmin(np.abs(days-dd)))
i40, i65, i90 = idx(40), idx(65), idx(days[-1])
r_early = (e[i65]-e[i40])/(days[i65]-days[i40])
r_late = (e[i90]-e[i65])/(days[i90]-days[i65])
print(f'd(eta)/dt d40-65: {r_early:+.5f} m/day; d65-90: {r_late:+.5f} m/day')
ar = abs(r_late/max(r_early,1e-12))
print(f'accel ratio = {ar:.4f}  (criterion <= 3)')
print('GATE:', 'PASS' if e.max() < 3.0 and ar <= 3 else 'FAIL')
