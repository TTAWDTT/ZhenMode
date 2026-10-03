"""Tests for conservative arbitrary-resolution ETOPO remapping.

``--resolution`` historically required integer multiples of the 0.1° ETOPO
source grid.  The ``area`` remap keeps the legacy mode for old runs but lets
the runner accept a continuous target spacing (0.37°, 0.85°, ...).  The key
invariants are that the dimension helper and reader agree, and that the reader
closes exactly 360° of longitude rather than relying on integer block counts.

Run: python -m pytest tests/test_area_resolution.py -v
"""
import os

import numpy as np
import pytest

from ocean_solver.config.definitions import DEFAULT_CONFIG, GlobalGridConfig
from ocean_solver.io.grid import _read_etopo_global, global_grid_dims, make_global_grid

BATHY = DEFAULT_CONFIG.bathymetry_file
HAVE_BATHY = os.path.exists(BATHY) or os.path.exists(BATHY + ".npz")
requires_bathy = pytest.mark.skipif(
    not HAVE_BATHY, reason=f"bathymetry not found at {BATHY}")


@pytest.mark.parametrize("res,lat_max", [
    (20.0, 45.0),     # tiny smoke case
    (2.0, 60.0),
    (1.0, 60.0),
    (0.85, 60.0),
    (0.37, 60.0),
])
@requires_bathy
def test_area_dims_match_reader(res, lat_max):
    nx, ny = global_grid_dims(res, lat_max, remap="area")
    depth, lon, lat = _read_etopo_global(
        BATHY, resolution=res, lat_max=lat_max, remap="area")
    assert depth.shape == (nx, ny)
    assert len(lon) == nx
    assert len(lat) == ny
    assert bool(np.all(np.isfinite(depth)))
    assert bool(np.all(np.isfinite(lon))) and bool(np.all(np.isfinite(lat)))


@requires_bathy
def test_area_mode_closes_periodic_longitude_and_latitude():
    """0.85° does not divide 360/180, but the remapped grid must close."""
    nx, _ = global_grid_dims(0.85, 60.0, remap="area")
    assert nx == int(round(360.0 / 0.85))
    depth, lon, lat = _read_etopo_global(
        BATHY, resolution=0.85, lat_max=60.0, remap="area")
    dlon_eff = 360.0 / nx
    np.testing.assert_allclose(lon, dlon_eff * (0.5 + np.arange(nx)),
                               rtol=0.0, atol=1e-12)
    # Successive centers use the effective (not requested) spacing.
    np.testing.assert_allclose(np.diff(lat), 180.0 / int(round(180.0 / 0.85)),
                               rtol=1e-12, atol=1e-12)


@requires_bathy
def test_make_global_grid_area_mode():
    gc = GlobalGridConfig(nx=18, ny=5, resolution=20.0, lat_max=45.0)
    grid = make_global_grid(gc, BATHY, remap="area")
    assert (grid.nx, grid.ny) == (18, 5)
    assert grid.lon.shape == (18,)
    assert grid.lat.shape == (5,)
    assert bool(np.all(np.isfinite(grid.depth)))
    # Area mode uses the exact periodic spacing for the zonal metric.
    expected_dx0 = 6.371e6 * np.cos(np.radians(grid.lat[0])) * np.radians(20.0)
    np.testing.assert_allclose(grid.dx_2d[0, 0], expected_dx0, rtol=1e-12)
