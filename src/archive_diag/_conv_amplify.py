import numpy as np, sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
# (167,67): conv_S = +84.5/day AT THE SURFACE. Sign analysis:
# S(167,67) = [34.87, 34.61, 34.84, ...]. d2S at k=0 = S2-2S1+S0 = +0.49.
# conv_S = +6.37/day (positive, raises S0 toward S1 -> DESTROYS the
# vertical salt contrast). But measured conv_S = +84.5/day. 13x bigger!
# My hand calc used d2z scale 18.2m; the real d2z_h0_top must be smaller.
# d2z_h0_top^2 = 331.8 -> 18.2m. 0.05*0.49/331.8*86400 = 6.37. But solver
# gives 84.5. Factor 13.3. Hmm — 331.8/13.3 = 25. So the effective d2z
# scale = 5m not 18.2m? d2z_h0_top = 5? That gives 0.05*0.49/25*86400=84.7.
# YES: d2z_h0_top = 5.0 (the SURFACE layer thickness). So conv_S = +84.7/day.
# Now the CRITICAL question: why does this AMPLIFY the dipole instead of
# damping it? conv_S(+) at k=0 raises S0 (34.87 -> higher, toward 34.61?).
# Wait, S0=34.87 > S1=34.61: surface is SALTY. d2S = S2-2S1+S0 = +0.49 > 0
# -> conv_S > 0 -> S0 INCREASES (saltier). That makes rho[0] even LARGER
# vs rho[1] -> MORE unstable -> conv mask stays on, and S0 keeps growing.
# THE CONVECTIVE ADJUSTMENT IS A POSITIVE FEEDBACK on S:
# kappa_conv*d2S/dz2 with one-sided d2 at the boundary moves S0 AWAY from
# S1 instead of toward it! The one-sided d2z stencil at the boundary:
# (S[2] - 2*S[1] + S[0])/d2z_h0^2 evaluates curvature using S0 as a NODE,
# so if S0 is high, d2 is positive and S0 rises MORE. A true mixing scheme
# would move S0 toward the mean of S1 (down-gradient), i.e. dS0/dt ~
# -(S0-S1)*kappa/dz^2 (negative when S0 > S1). The FD d2 with non-uniform
# weights actually has the right sign for the INTERIOR, but at the
# TOP boundary with S[0] > S[1] ~ S[2]:
S = np.array([34.87, 34.61, 34.84])
# true curvature (non-uniform z=0,5,15): d2S/dz2 = 2*[ (S2-S1)/10 - (S1-S0)/5 ] / (5+15)/2?
z = np.array([0., 5., 15.])
# second derivative via finite differences for non-uniform grid:
# d2S/dz2 ≈ 2 * [ (S2-S1)/(z2-z1) - (S1-S0)/(z1-z0) ] / (z2-z0)
d2 = 2 * ( (S[2]-S[1])/(z[2]-z[1]) - (S[1]-S[0])/(z[1]-z[0]) ) / (z[2]-z[0])
print('correct non-uniform d2S/dz2 at k=0:', d2, '-> conv_S =', 0.05*d2*86400)
# = 2*((0.23/10) - (-0.26/5))/15 = 2*(0.023+0.052)/15 = 0.01 PSU/m^2
# conv_S = +864/day?? That's positive too. Hmm: S0=34.87, S1=34.61:
# (S1-S0)/(5) = -0.052 (negative flux gradient), (S2-S1)/10 = +0.023.
# The surface flux-gradient is negative = S decreasing downward = UNSTABLE
# salt fingering configuration (salty over fresh). Convective mixing should
# homogenize S0,S1 -> S0 drops, S1 rises. d2 at the TOP NODE with the
# standard formula gives +0.01 -> S0 rises. WRONG DIRECTION.
# THE BUG: the top-boundary d2 stencil has S0 as a node with the curvature
# extrapolated from the INTERIOR gradient — it is NOT a mixing operator at
# the boundary. It amplifies boundary anomalies.
# Verify with the interior form: for uniform interior d2, S1 is the node:
# d2S1 = (S2-2S1+S0)/25: S=[34.87,34.61,34.84] node k=1: (34.84-69.22+34.87)/25
# = +0.49/25 > 0 -> S1 rises toward... S1=34.61 is the LOW; d2>0 raises S1.
# Correct mixing direction at k=1 (raises the low). But at k=0 the same
# formula raises the HIGH. Sign flip at the boundary node.
print('boundary d2 uses S0 as node: dS0/dt = kappa*(S2-2S1+S0)/d2z_h0^2')
print('This is a WAVE EQUATION term (anti-diffusive at boundaries), not mixing.')
