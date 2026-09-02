import numpy as np, sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
# horizontal advective CFL for S at the runaway cell, dt=120s, dx~1.1e5 m
# dt adv CFL = |u|*dt/dx
print('CFL check: max|u| ~ 1.87 m/s -> CFL =', 1.87*120/111191, '(2D advective, stable <~1)')
# but what matters for the dipole: the S dipole width grows?
lat = np.linspace(-59.5, 59.5, 120)
arr0 = np.load('../results/global_gm365d_3d/snap_00018.npy')
arr1 = np.load('../results/global_gm365d_3d/snap_00019.npy')
arr2 = np.load('../results/global_gm365d_3d/snap_00020.npy')
for k, arr in [(18, arr0), (19, arr1), (20, arr2)]:
    T, u, v, S = arr
    m = S[:, :, 0]
    # dipole: contiguous region below 30 PSU
    low = m < 30
    print(f'd{k*5}: lowS cells={low.sum()}, extent_lon={np.where(low.any(axis=1))[0].max()-np.where(low.any(axis=1))[0].min() if low.any() else 0}')
    # check: does the pattern MOVE day to day?
    if k > 18:
        # cross-correlate m with previous snapshot
        a = m - m.mean(); b = prev - prev.mean()
        cc = np.fft.ifft2(np.fft.fft2(a) * np.conj(np.fft.fft2(b))).real
        cc /= (np.linalg.norm(a)*np.linalg.norm(b))
        ij = np.unravel_index(np.argmax(cc), cc.shape)
        # wrap longitude
        di = ij[0] if ij[0] < 180 else ij[0]-360
        print(f'   best correlation shift vs prev: di={di} dj={ij[1]} corr={cc.max():.3f}')
    prev = m
