"""Probe the exact blow-up nucleation point (+79.5, +124.5) on the global grid.

Questions:
  1. What is the ocean depth there? (ultra-shallow single-column?)
  2. What is dx there (cos(lat) metric)? Is the free-surface CFL violated?
  3. Is the polar-cap filter row (last 2 rows) interacting with a land boundary?
  4. Is there a land-sea step (neighbor depth jump) there -> topographic PGF?
"""
import os
os.environ.setdefault('JAX_PLATFORMS', 'cpu')
import numpy as np
from dataclasses import replace
from config import GlobalGridConfig
from grid import make_global_grid

BATHY = r"C:\Users\zhen.luo\Desktop\ETOPO_2022_v1_r3600x1800_surface.nc"

gcfg = replace(GlobalGridConfig(), lat_max=80.0, ny=160)
grid = make_global_grid(gcfg, BATHY, smooth_passes=20, min_depth=25.0)

lat = np.array(grid.lat)
lon = np.array(grid.lon)
depth = np.array(grid.depth)
wet = np.array(grid.wet_mask)

# target (+79.5, +124.5)
iy = int(np.argmin(np.abs(lat - 79.5)))
ix = int(np.argmin(np.abs(lon - 124.5)))
print(f"target lat {lat[iy]:.2f} (iy={iy}, ny={len(lat)})  lon {lon[ix]:.2f} (ix={ix})")
print(f"  depth = {depth[ix, iy]:.1f} m   wet = {wet[ix, iy]}")

# dx at this lat
R = 6371000.0
dlon = np.deg2rad(1.0)
dx = R * np.cos(np.deg2rad(lat[iy])) * dlon
print(f"  dx at lat {lat[iy]:.1f} = {dx/1e3:.2f} km")
print(f"  free-surface CFL dt < dx/sqrt(g*H) = {dx/np.sqrt(9.81*4000.0):.1f} s  (dt=60)")
print(f"  Laplacian CFL nu_h*dt/dx^2 = {5e5*60.0/dx**2:.3e}  (must < ~0.25)")

# neighbors
print(f"\n  3x3 depth neighborhood around ({lat[iy]:.1f},{lon[ix]:.1f}):")
for dj in [-1, 0, 1]:
    row = []
    for di in [-1, 0, 1]:
        j = iy + dj; i = (ix + di) % len(lon)
        row.append(f"{depth[i,j]:5.0f}({int(wet[i,j])})")
    print(f"    {' '.join(row)}")

# How many wet points in the last 5 rows? (polar cap structure)
print(f"\n  ocean fraction by poleward row band:")
for j in range(len(lat)-8, len(lat)):
    print(f"    lat {lat[j]:5.1f} (j={j}): wet frac {wet[:,j].mean():.2f}  "
          f"n_wet {int(wet[:,j].sum())}  mean depth {depth[wet[:,j].astype(bool), j].mean() if wet[:,j].any() else 0:.0f}")

# Check: is (+79.5,124.5) a single isolated wet column?
nb = 0
for di, dj in [(1,0),(-1,0),(0,1),(0,-1)]:
    j = iy+dj; i = (ix+di) % len(lon)
    if 0 <= j < len(lat):
        nb += int(wet[i, j])
print(f"\n  wet cardinal neighbors of target: {nb}/4  "
      f"({'ISOLATED' if nb==0 else 'edge' if nb<2 else 'interior'})")
