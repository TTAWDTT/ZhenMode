import numpy as np, sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
# THERE IT IS. rho[12]=+5.8e-4 (wet, 2000m) vs rho[13]=0 (ghost, masked).
# rho[12] > rho[13]=0 -> iface 12 "unstable". The wet_iface gate:
# wet_mask_z[12]=1 (wet), wet_mask_z[13]=0 (ghost) -> gate=0. GATED OUT.
# BUT the docstring in the solver says this was fixed... let me verify what
# the ACTUAL gate does for this column. wmz[302,70] = [1]*13+[0].
# wet_iface[12] = (wmz[12]>0.5)&(wmz[13]>0.5) = 1 & 0 = 0. GATED.
# So (302,70) should NOT be conv-flagged by iface 12. But measured
# unstable ifaces at (302,70) = [12] (from my UNGATED diagnostic!).
# With the gate, is (302,70) flagged at ALL?
arr = np.load('../results/global_gm365d_3d/snap_00022.npy')
T, u, v, S = arr
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
def rho(T, S): return -2.0e-4*(T-15.0) + 7.6e-4*(S-35.0)
r = rho(T, S) * wmz
gate = (wmz[:,:,:-1]>0.5) & (wmz[:,:,1:]>0.5)
unst = (r[:,:,:-1] > r[:,:,1:]) & gate
anycol = unst.any(axis=-1)
print('is (302,70) conv-flagged (gated)?', anycol[302,70])
print('total conv columns (gated) at d110:', anycol.sum())
# The measured conv_S at (302,70,0) = -84.6 PSU/day from the ACTUAL solver
# tendency decomposition. If the column is NOT flagged, conv_S=0 there.
# But my earlier decomposition measured conv_S = -84.6/day there. So the
# column IS flagged by the solver. CONTRADICTION -> the solver's gate
# differs from mine. Check the solver's actual lines 868-871 again:
import inspect
src = inspect.getsource(jsg._compute_tracer_tendency)
i = src.find('rho_prime')
print(src[src.find('rho_prime ='):src.find('heat_factor')])
