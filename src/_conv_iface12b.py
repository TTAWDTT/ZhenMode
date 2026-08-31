import numpy as np, sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
from dataclasses import replace
from config import DEFAULT_CONFIG, GlobalGridConfig
from grid import make_global_grid
import jax_solver_global as jsg

gcfg = replace(GlobalGridConfig(), lat_max=60.0, ny=120)
grid = make_global_grid(gcfg, DEFAULT_CONFIG.bathymetry_file, smooth_passes=30, min_depth=100.0)
base = jsg.make_fd_params(grid)
wmz = np.asarray(base.wet_mask_z)
# dz levels: interfaces at 5,15,30,50,75,100,150,200,300,500,1000,2000,4000
# 13 layers, 14 LEVELS (0..13). wet_mask_z has 14 entries (levels).
# iface k connects level k and k+1.
# For a column with bottom at level 13 (deepest, 4000m): ifaces 0..11 wet?
# levels wet: 0..13 all. iface 12 connects level 12 (2000m) & 13 (4000m): WET.
# iface-12 unstable = rho(2000m) > rho(4000m) = deep water denser below? That's
# STABLE in the usual sense (dense over light is stable... wait z is DOWNWARD
# index: level 13 = deeper = 4000m. rho[12] > rho[13] means 2000m water denser
# than 4000m water = UNSTABLE (dense over light).
# 17086 cells: check how many have BOTH levels 12,13 wet:
wet12 = (wmz[:,:,12]>0.5)&(wmz[:,:,13]>0.5)
print('cells with levels 12+13 both wet:', wet12.sum())
# histogram said 15527 bottom at level 13 + 11745 at level 12 = 27272 candidates
# So 17k of 27k deep columns have rho(2000m) > rho(4000m).
# WHY? WOA S at 2000m vs 4000m: deep Atlantic/Indian freshening?
# T contribution: -2e-4*(T-15). At 2000m T~2.5, at 4000m T~1.5:
# rho_T(2000) = -2e-4*(-12.5) = +2.5e-3; rho_T(4000) = +2.7e-3
# S contribution: +7.6e-4*(S-35). At 2000m S~34.9: -7.6e-5; 4000m S~34.7: -2.3e-4
# rho(2000) = 2.5e-3 - 0.76e-4 = 2.42e-3; rho(4000) = 2.7e-3 - 2.3e-4 = 2.47e-3
# Marginal. With real WOA values let me just measure:
T_init, S_init = None, None
d = np.load('../results/global_gm365d.npz', allow_pickle=True)
T_init = np.asarray(d['T_init']); S_init = np.asarray(d['S_init'])
def rho(T, S): return -2.0e-4*(T-15.0) + 7.6e-4*(S-35.0)
r0 = rho(T_init, S_init) * wmz
unst12_init = (r0[:,:,:-1] > r0[:,:,1:]) & (wmz[:,:,:-1]>0.5) & (wmz[:,:,1:]>0.5)
print('INITIAL iface-12 unstable count:', unst12_init[:,:,12].sum())
print('INITIAL all-iface unstable counts:', unst12_init.sum(axis=(0,1)))
