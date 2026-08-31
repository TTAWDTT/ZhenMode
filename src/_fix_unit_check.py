import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import jax, jax.numpy as jnp
jax.config.update('jax_enable_x64', True)
import jax_solver_global as jsg

# 6 levels -> interior stencils at nodes 1..4 (4 coefs each), top at node 0, bot at node 5
class P:
    d2z_h0_top = 5.0; d2z_h0_bot = 2000.0
    hm = jnp.array([5.0, 15.0, 20.0, 25.0, 50.0])  # |z[k-1]-z[k]| nodes 1..4
    hp = jnp.array([15.0, 20.0, 25.0, 50.0, 200.0])
    d2z_hm = hm.reshape(1, 1, -1)
    d2z_hp = hp.reshape(1, 1, -1)
    d2z_denom = (hm * hp * (hm + hp) / 2.0).reshape(1, 1, -1)

# Runaway profile at (167,67,0) day 110: S = [34.87, 34.61, 34.84, ...]
S_prof = jnp.array([[[34.87, 34.61, 34.84, 34.9, 34.9, 34.9, 34.9]]])
lap = jsg._d2_dz2(S_prof, P)
d2S0 = float(lap[0, 0, 0])
conv_S = 0.05 * d2S0 * 86400
print(f'new top stencil d2S0 = {d2S0:+.4f}  ->  conv_S = {conv_S:+.1f} PSU/day')
old = (34.84 - 2 * 34.61 + 34.87) / 25
print(f'old stencil: {old:+.4f} -> {0.05*old*86400:+.1f} PSU/day  (run measured +84.5)')
assert conv_S < 0, 'FAIL: boundary still amplifies'
print('PASS: salty boundary node relaxes toward interior (sign flipped)')

S_stable = jnp[-1:] if False else jnp.array([[[34.0, 34.6, 34.84, 34.9, 34.9, 34.9, 34.9]]])
lap2 = jsg._d2_dz2(S_stable, P)
c2 = 0.05 * float(lap2[0, 0, 0]) * 86400
print(f'stable profile: conv_S = {c2:+.1f} PSU/day (must be > 0)')
assert c2 > 0
print('PASS: fresh boundary node gets saltier -- operator diffusive both ways')
