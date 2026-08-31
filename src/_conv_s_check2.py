import numpy as np, sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
# conv_S max = 84.5 PSU/day at (167,67,0) matches the runaway growth rate!
# conv_S = kappa_conv * conv_mask * d2S/dz2. kappa_conv=0.05 m^2/s.
# With S=[25.5, 33.5, 35.0] at z=[0,5,15]m:
d2z_h0 = 5.0  # d2z_h0_top ~ (5+10)/2? Verify from runner grid
# conv_S = 0.05 * (35.02 - 2*33.52 + 25.53) / 25 = 0.05*(-6.49)/25 = -0.013 PSU/s = -1122/day?
# No: -6.49/625 = -0.0104 PSU/s = -898/day. Too big. Measured 84.5.
# d2z_h0_top must be larger: d2z_h0_top^2 = 0.05*|d2S|/tendency
t = 84.5/86400
print('d2z_h0_top^2 =', 0.05*6.49/t, '-> d2z_h0_top =', (0.05*6.49/t)**0.5)
# The unstable interface is where rho[k] > rho[k+1] (light over dense OK; the
# mask flags rho ABOVE > rho BELOW i.e. unstable).
# At the dipole cell: rho' = [-0.0092, -0.0025, ...]. rho[0] < rho[1]:
# -0.0092 < -0.0025 -> STABLE at k=0..1 interface. Where's the unstable?
# Print rho profile full depth at (167,67):
arr = np.load('../results/global_gm365d_3d/snap_00022.npy')
T, u, v, S = arr
def rho(T, S): return -2.0e-4*(T-15.0) + 7.6e-4*(S-35.0)
r = rho(T[167,67,:], S[167,67,:])
print('rho(167,67,:):', np.round(r, 6))
print('T(167,67,:):', np.round(T[167,67,:], 2))
print('S(167,67,:):', np.round(S[167,67,:], 2))
# interface flags
print('unstable ifaces:', np.where((r[:-1] > r[1:]))[0])
