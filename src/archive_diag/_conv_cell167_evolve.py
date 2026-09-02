import numpy as np, sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
# Where does the ITCZ dipole START? First appearance of fresh cell:
# from track: d50 lon=173 lat=-5.5 S=31.77. d55: lon=165 lat=6.5.
# Wait -5.5 lat? That's south. The dipole seems to start in the far west
# Pacific warm pool / Indonesian region. What drives it there?
# The fresh water appears where HEAVY RAIN would be... but there's no
# freshwater forcing in the model! dS/dt terms: adv, diff_h, diff_v,
# conv, gm. No P-E. So the dipole must be advectively/convectively formed.
# KEY INSIGHT: the conv boundary-stencil feedback only AMPLIFIES an
# existing anomaly. The seed: bulk heat flux + WOA S structure.
# Surface S pattern d50 at the seed region:
arr = np.load('../results/global_gm365d_3d/snap_00010.npy')
T, u, v, S = arr
lat = np.linspace(-59.5, 59.5, 120)
print('S[k=0] (i=160..185, j=60..75) at d50:')
for j in range(75,59,-1):
    print(f'  lat{lat[j]:5.1f}: ' + ' '.join(f'{S[i,j,0]:5.1f}' for i in range(160,186)))
print('T[k=0] same region:')
for j in range(75,59,-1):
    print(f'  lat{lat[j]:5.1f}: ' + ' '.join(f'{T[i,j,0]:5.1f}' for i in range(160,186)))
