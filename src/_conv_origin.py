import numpy as np, sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
# CONFIRMED: conv_S at the dipole cell = -84.6 PSU/day at d110 = EXACTLY
# the measured growth rate (85.9). The dipole cell is flagged unstable at
# iface 12 (BOTTOM), and the whole column gets conv_mask -> kappa_conv *
# d2S/dz2 acts on the SURFACE layer with the huge dipole curvature:
# d2S = -6.49 PSU over ~18m scale -> -84.6 PSU/day. This is a POSITIVE
# FEEDBACK: dipole -> rho' surf goes very light -> wait, no.
# rho(302,70) = [-0.0092, -0.0025, ...]. rho[0] < rho[1]: STABLE at top.
# The instability flag is at iface 12 (bottom, near-seafloor).
# Question: WHY does the whole column convect because the BOTTOM is unstable?
# conv_mask = any(unstable_iface, axis=-1) -> whole column convects.
# kappa_conv * d2S/dz2 applied EVERYWHERE in the column, including the
# surface dipole. d2S at surface is huge (-6.49), so conv_S = -84 PSU/day,
# which DEEPENS the dipole (S0 drops further). POSITIVE FEEDBACK.
# Root cause chain:
# 1. SOME bottom interface becomes unstable (deep S cross-front diffusion)
# 2. conv_mask = any(...) flags the ENTIRE column
# 3. conv_S = kappa_conv * d2S/dz2 applied at ALL depths incl surface
# 4. Surface S dipole gets amplified (d2S<0 where dipole center is)
# 5. Amplified dipole -> more light surface water -> more columns triggered
# Wait but the dipole itself has STABLE surface (rho[0]<rho[1]). The trigger
# is the bottom instability. Check when the bottom instability started.
print('Bottom instability hunt: which cells had iface-12 unstable at d50/d70/d90?')
for k in [10, 14, 18, 22]:
    arr = np.load(f'../results/global_gm365d_3d/snap_{k:05d}.npy')
    T, u, v, S = arr
    def rho(T, S): return -2.0e-4*(T-15.0) + 7.6e-4*(S-35.0)
    r = rho(T, S)
    unst = (r[:,:,:-1] > r[:,:,1:])
    n_iface = unstable_counts = unst.sum(axis=(0,1))
    print(f'd{k*5:3d}: unstable iface counts k=0..13: {unstable_counts}')
