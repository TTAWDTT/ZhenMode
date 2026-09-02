import numpy as np, sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
# Density check at the runaway cell: T=25.5, S=25.5 vs neighbors
# rho' = -2e-4*(T-15) + 7.6e-4*(S-35)
def rho(T, S): return -2.0e-4*(T-15.0) + 7.6e-4*(S-35.0)
# d100 cell (165,68): T=25.47 S=-33.23; neighbor (166,68): T~25 S=35
# rho(dipole cell) = -2e-4*10.47 + 7.6e-4*(-68.23) = -2.09e-3 - 5.19e-2 = -5.4e-2
print('rho dipole cell:', rho(25.47, -33.23))
print('rho neighbor (S=35):', rho(25.47, 35.0))
# drho/dz at dipole cell d110: S profile [25.53, 33.52, 35.02...], T profile?
arr = np.load('../results/global_gm365d_3d/snap_00022.npy')
T, u, v, S = arr
print('T(302,70,:6):', T[302,70,:6])
print('S(302,70,:6):', S[302,70,:6])
print('rho(302,70,:6):', rho(T[302,70,:6], S[302,70,:6]))
# The huge negative rho' at surface (light water) + smaller negative below
# = STABLE stratification. drho/dz = 6.8e-3 > 0 = stable.
# BUT: S=-265 at d130, T=25: rho = -2e-4*10 + 7.6e-4*(-300) = -0.2298
# vs layer below S=-116: rho = -2e-4*10 + 7.6e-4*(-151) = -0.1155
# STILL stable (light over dense). So it's not convective instability.
# What about GM: slope = -drho_dx/drho_dz. drho_dx at the dipole edge:
# S goes -39 -> -20 -> 0 -> +12 over ~4 cells. drho_dx ~ 7.6e-4 * 15 PSU/cell
# = 1.14e-2 per 111 km. drho_dz ~ 6.8e-3 / 5 m = 1.36e-3 per m.
# slope = 1.14e-2/111e3 / 1.36e-3 = 7.5e-5?? No wait:
drho_dx = 7.6e-4 * 15 / 111191    # rho per meter
drho_dz = 6.8e-3 / 5.0            # rho per meter
print(f'slope raw = {drho_dx/drho_dz:.2e}')   # tanh-limited to 0.01
# Fz = -k*(SdotGradC + S2*dC_dz). With S=0.01, dS_dz=(S[k+1]-S[k])/5m
dS_dz = (33.52-25.53)/5.0
print(f'GM Fz S2*dC_dz term: 1000*{0.01**2:.0e}*{dS_dz:.2f} = {1000*0.01**2*dS_dz:.2e} PSU/s')
print(f'per day: {1000*0.01**2*dS_dz*86400:.2f} PSU/day')
