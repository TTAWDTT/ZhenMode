"""Shared helpers for the solver unit tests.

Only genuinely identical helpers live here; a test that needs a grid with
land, a shallow patch or a different shape keeps its own builder next to the
assertions that explain it.
"""
import numpy as np

from ocean_solver.geometry.types import GlobalOceanGrid


def all_wet_grid(nx=32, ny=32, nz=8):
    """Small, fully-wet global grid: uniform 4000 m depth, no land.

    Every cell is active, so a test probing a single operator does not have to
    reason about masks or ghost layers.
    """
    lat = np.linspace(-30.0, 30.0, ny)
    lon = np.linspace(0.5, 359.5, nx)
    R = 6.371e6
    dlon = 360.0 / nx
    cos_lat = np.cos(np.radians(lat))
    dx_2d = np.broadcast_to(R * cos_lat * np.radians(dlon), (nx, ny)).copy()
    dy = R * np.radians(abs(lat[1] - lat[0]))
    f = np.broadcast_to(2 * 7.2921e-5 * np.sin(np.radians(lat)), (nx, ny)).copy()
    z = -np.linspace(50.0, 4000.0, nz)          # negative downward
    dz = -np.diff(z)                            # positive thicknesses
    return GlobalOceanGrid(
        lon=lon, lat=lat, dx_2d=dx_2d, dy=float(dy), cos_lat=cos_lat, f=f,
        z=z, dz=dz, nz=nz, depth=np.full((nx, ny), 4000.0),
        wet_mask=np.ones((nx, ny)),
        ocean_mask=np.ones((nx, ny), dtype=bool),
        land_mask=np.zeros((nx, ny)),
        wet_mask_3d=np.ones((nx, ny, nz)),
        nx=nx, ny=ny)
