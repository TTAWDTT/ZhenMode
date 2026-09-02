import numpy as np, sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
# 17000 cells with iface-12 "unstable" at ALL times = 25% of ocean!
# This is the GHOST-WATER artifact again: iface 12 is the LAST WET interface
# for shallow columns. For a column with bottom at k=5, iface 12 is below
# the seafloor -> wet_iface gate should exclude it. But 17086 cells...
# wet_iface gate: BOTH adjacent layers wet. For a 100m-deep column, layers
# 0..2 wet, 3..13 ghost. iface 12: layer 12 vs 13 both GHOST -> gate=0.
# So the 17k count means these columns have wet layer 12 AND 13? Only deep
# columns do. Ocean 71.6% of 43200 = 30900 cells. Deep (>4000m) fraction?
# The 14-level grid: dz = [5,10,15,25,40,60,80,100,125,155,190,230,270,310]?
# cumulative: 5,15,30,55,95,155,235,335,460,615,805,1035,1305,1615? That's only 1615m.
# Hmm min_depth=100. Let me read the dz from the npz or grid.
import numpy as np, sys
sys.path.insert(0, '.')
from config import DEFAULT_CONFIG, GlobalGridConfig
from grid import make_global_grid
from dataclasses import replace
gcfg = replace(GlobalGridConfig(), lat_max=60.0, ny=120)
grid = make_global_grid(gcfg, DEFAULT_CONFIG.bathymetry_file, smooth_passes=30, min_depth=100.0)
print('dz:', np.asarray(grid.dz))
print('cumsum:', np.cumsum(np.asarray(grid.dz)))
# depth distribution
dep = np.asarray(grid.depth) if hasattr(grid, 'depth') else None
print('depth attr?', dep is not None)
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
import jax_solver_global as jsg
base = jsg.make_fd_params(grid)
wmz = np.asarray(base.wet_mask_z)
# count cells by bottom-wet-layer
bot_layer = wmz.shape[2] - 1 - np.argmax(wmz[:, :, ::-1] > 0.5, axis=2)
import collections
cnt = collections.Counter(bot_layer[wmz[:,:,0]>0.5].ravel())
print('bottom-wet-layer histogram:', dict(sorted(cnt.items())))
