"""Cache regional init fields (128x128x14) to avoid scipy-interpolate OOM."""
import os, sys
os.environ.setdefault('JAX_ENABLE_X64', '1')
sys.path.insert(0, 'src')
import numpy as np
from config import DEFAULT_CONFIG
from grid import make_grid
from woa_data import get_initial_fields

grid = make_grid(DEFAULT_CONFIG.grid, DEFAULT_CONFIG.bathymetry_file)
print(f"regional grid {grid.nx}x{grid.ny}x{grid.nz}, dx={grid.dx:.0f} dy={grid.dy:.0f} f0={grid.f0:.4e}")
T_init, S_init = get_initial_fields(grid)
T_init = np.array(T_init); S_init = np.array(S_init)
wet_mask = np.array(grid.ocean_mask, dtype=np.float64); dz = np.array(grid.dz)
f = np.array(grid.f)
np.savez('src/_cache_regional_init.npz',
         T=T_init, S=S_init, wet_mask=wet_mask, dz=dz, f=f,
         nx=grid.nx, ny=grid.ny, nz=grid.nz, f0=grid.f0, dx=grid.dx, dy=grid.dy)
print(f"cached regional ({T_init.shape})")
