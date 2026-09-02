import numpy as np, sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp

# Is the S dipole advecting coherently (physical-like) or stationary & growing?
# Track the S_min location + value in each snapshot (k=0)
print('snapshot | S_min surf | location')
for k in range(0, 27):
    arr = np.load(f'../results/global_gm365d_3d/snap_{k:05d}.npy')
    T, u, v, S = arr
    m = S[:, :, 0]
    imin = np.unravel_index(np.argmin(m), m.shape)
    print(f'  d{k*5:3d}: S_min={m.min():9.2f} at i={imin[0]} j={imin[1]} (lon{imin[0]} lat{lat[imin[1]]:.1f})')
