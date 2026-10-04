"""Identical passive-advection fixture shared by the two limiter suites."""
import numpy as np

from zhenmode.model.geometry.types import GlobalOceanGrid


def _synth_grid():
    """Small, fully wet periodic grid suitable for CPU operator tests."""
    nx, ny, nz = 24, 8, 6
    dlon = 360.0 / nx
    lon = np.arange(nx) * dlon
    lat = np.linspace(-30.0, 30.0, ny)
    cos_lat = np.cos(np.radians(lat))
    dx_2d = np.broadcast_to(
        6.371e6 * cos_lat[None, :] * np.radians(dlon), (nx, ny)).copy()
    dy = 6.371e6 * np.radians(abs(lat[1] - lat[0]))
    f = np.broadcast_to(2.0 * 7.2921e-5 * np.sin(np.radians(lat)),
                        (nx, ny)).copy()
    z = -np.linspace(50.0, 4000.0, nz)
    dz = -np.diff(z)
    return GlobalOceanGrid(
        lon=lon, lat=lat, dx_2d=dx_2d, dy=float(dy), cos_lat=cos_lat, f=f,
        z=z, dz=dz, nz=nz,
        depth=np.full((nx, ny), 4000.0),
        wet_mask=np.ones((nx, ny)),
        ocean_mask=np.ones((nx, ny), dtype=bool),
        land_mask=np.zeros((nx, ny), dtype=bool),
        wet_mask_3d=np.ones((nx, ny, nz)),
        nx=nx, ny=ny)
