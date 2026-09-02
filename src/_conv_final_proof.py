import numpy as np, sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
# d2u_top = (u[2] - 2*u[1] + u[0]) / (z1-z0)^2
# This is a SECOND-ORDER stencil evaluated at node k=1 (interior), NOT at
# the boundary node k=0! (S2-2S1+S0)/h^2 is the centered d2/dz2 at z1.
# Applied as the tendency OF k=0 (the boundary value), it's the WRONG NODE:
# the tendency of S1 would be kappa*(S2-2S1+S0)/h^2 = correct mixing at k=1.
# But _d2_dz2 returns this as element 0 of the output, so dS[0]/dt gets
# the curvature at z1 — a one-gridpoint OFFSET. If S0 > S1 (salty surface,
# unstable), curvature at z1 = (S2-2S1+S0)/25 > 0 -> applied to S0 -> S0
# rises. Anti-diffusive: pushes S0 FARTHER from S1.
# MEASURED: conv_S at (167,67,0) = +84.5/day, growing the S dipole.
# Also dT at same cell = -18.4 K/day: T=[19.1,18.96,18.76]: d2T = T2-2T1+T0
# = 18.76-37.92+19.1 = -0.06 -> conv_T = 0.05*(-0.06)/25*86400 = -10.4 K/day
# (measured -18.4, same order). T0 drops FURTHER below T1. Also wrong sign!
# True convective mixing should WARM T0 toward T1 (T0 < T1? 19.1 > 18.96 —
# stable for T; the instability is salinity-driven: S0=34.87 > S1=34.61).
# A real convective adjustment would mix T and S between layers 0,1 (and
# deeper if needed) toward neutral. Instead the FD boundary stencil
# AMPLIFIES the anomaly: T0 -> colder, S0 -> saltier -> rho0 even denser ->
# conv mask never clears -> runaway. THE S/T DIPOLES ARE THIS FEEDBACK.
# Timeline matches: ITCZ dipole forms ~d50 when surface instabilities
# (salt-fingering-type in linear EOS) first appear (2000+ cells at d50),
# then amplifies exponentially.
# Where does the FIRST instability come from? WOA surface: subtropical
# salinity max (S>36.5) over colder subsurface = salt fingering. In the
# real ocean salt fingering is a real (weak) process; here the linear EOS
# + kappa_conv=0.05 makes the adjustment itself the amplifier.
# Check: does the conv mask at (167,67) persist? Which ifaces:
arr = np.load('../results/global_gm365d_3d/snap_00022.npy')
T, u, v, S = arr
def rho(T, S): return -2.0e-4*(T-15.0) + 7.6e-4*(S-35.0)
r = rho(T[167,67,:], S[167,67,:])
print('rho(167,67):', np.round(r, 7))
print('unstable ifaces:', np.where(r[:-1] > r[1:])[0])
