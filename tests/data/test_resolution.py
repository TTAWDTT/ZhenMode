"""Tests for the --resolution interface (grid resolution control).

The runner derives the horizontal grid from a single --resolution flag:
    nx = etopo_nlon // step
    ny = #{lat centers with |lat| <= lat_max}
with step = round(resolution / 0.1). These must match EXACTLY the dimensions
_read_etopo_global produces, because make_global_grid asserts the read matches
gc.nx / gc.ny. The integer-floor division (// rather than round()) is the
subtle part: at resolutions whose step does not divide the 3600x1800 source
(e.g. 1.3 deg -> step=13 -> 276 columns, not 360/1.3 = 277; lat_max=45 at
2 deg -> 46 rows, not 2*45/2 = 45) the naive round() formula silently
disagrees and the grid build is rejected.

Run:  python -m pytest tests/test_resolution.py -v
"""
import os
import sys
import warnings
from dataclasses import replace

import numpy as np
import pytest

from config import DEFAULT_CONFIG, GlobalGridConfig
from grid import _read_etopo_global, global_grid_dims, make_global_grid
from tests.support.paths import REPOSITORY_ROOT

BATHY = DEFAULT_CONFIG.bathymetry_file
HAVE_BATHY = os.path.exists(BATHY) or os.path.exists(BATHY + ".npz")

requires_bathy = pytest.mark.skipif(
    not HAVE_BATHY, reason=f"bathymetry not found at {BATHY}")


def _dt_cfl_at(res_, lat_max=60.0):
    """External-gravity-wave CFL limit [s] for the FD free surface.

    ``0.5*min(dx, dy)/sqrt(g*H)``; the limiting dx is the poleward-most
    column (cos(lat) smallest), so this is NOT the equatorial value -- but
    every dx scales with the resolution, so the limit does too.
    """
    nx, ny = global_grid_dims(res_, lat_max)
    gc = replace(GlobalGridConfig(), lat_max=lat_max, ny=ny, nx=nx,
                 resolution=res_)
    g = make_global_grid(gc, BATHY, smooth_passes=30, min_depth=100.0)
    return 0.5 * min(float(np.min(g.dx_2d)), g.dy) / np.sqrt(9.81 * 4000.0)


# ── 1. Dimension derivation matches the ETOPO reader exactly ─────────
# The core invariant: global_grid_dims() predicts _read_etopo_global()'s
# output shape for every (resolution, lat_max) the interface accepts.

@pytest.mark.parametrize("step", [1, 2, 3, 4, 5, 7, 10, 13, 17, 20, 30])
@pytest.mark.parametrize("lat_max", [20.0, 30.0, 45.0, 60.0, 66.5, 75.0, 85.0])
@requires_bathy
def test_dims_match_reader(step, lat_max):
    res = round(step * 0.1, 10)
    nx, ny = global_grid_dims(res, lat_max)
    depth, _, _ = _read_etopo_global(BATHY, resolution=res, lat_max=lat_max)
    assert (nx, ny) == depth.shape, (
        f"res={res} lat_max={lat_max}: derived ({nx},{ny}) "
        f"!= reader {depth.shape}")


@requires_bathy
def test_bathymetry_reads_as_ocean():
    """The relief must be read with ETOPO's sign convention: ocean is NEGATIVE.

    Every dimension test above passes on an all-land grid, so a stand-in
    written with positive depths (elevation = depth) silently yields a world
    with no ocean at all: the grid builds, the shapes match, and every metric
    assertion still holds. Pin the sign, and the presence of both land and
    ocean, here.
    """
    depth, _, _ = _read_etopo_global(BATHY, resolution=1.0, lat_max=60.0)
    ocean = depth > 0.0
    assert 0.3 < ocean.mean() < 0.95, (
        f"ocean fraction {ocean.mean():.1%} is not a plausible world -- the "
        f"relief was probably written with the wrong sign convention")
    assert ocean.any() and (~ocean).any(), "expected both land and ocean"
    assert depth.max() > 1000.0, f"deepest point is only {depth.max():.0f} m"


def test_npz_twin_does_not_shadow_the_real_relief(tmp_path):
    """With both files present the netCDF relief must win.

    The .npz twin exists so nodes without netCDF4 can still read a relief; if
    it wins whenever it is present, a leftover twin silently replaces the real
    bathymetry. Nothing downstream can tell -- both paths return the same
    shape and both are "valid" depth fields.
    """
    try:
        import netCDF4
    except ImportError:
        pytest.skip("netCDF4 not installed; the twin is the only readable path")

    nlon, nlat = 3600, 1800
    lon = np.arange(nlon, dtype=np.float64) * 0.1
    lat = -90.0 + (np.arange(nlat, dtype=np.float64) + 0.5) * 0.1
    nc_path = str(tmp_path / "relief.nc")
    # netCDF4's writer trips a numpy 2.5 DeprecationWarning about setting
    # .shape; it is inside the library, not this test.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        with netCDF4.Dataset(nc_path, "w") as ds:
            ds.createDimension("lon", nlon)
            ds.createDimension("lat", nlat)
            ds.createVariable("lon", "f8", ("lon",))[:] = lon
            ds.createVariable("lat", "f8", ("lat",))[:] = lat
            # ETOPO sign convention: ocean is negative. Real 3000 m, twin 1000 m.
            ds.createVariable("z", "i2", ("lat", "lon"))[:, :] = np.full(
                (nlat, nlon), -3000, dtype=np.int16)
    np.savez(nc_path + ".npz",
             z=np.full((nlat, nlon), -1000, dtype=np.int16), lon=lon, lat=lat)

    depth, _, _ = _read_etopo_global(nc_path, resolution=1.0, lat_max=60.0)
    assert np.allclose(depth[depth > 0], 3000.0), (
        "the .npz twin shadowed the real relief file")


def test_dims_use_integer_floor_not_round():
    """The regression that motivated the helper: round() gives the wrong nx
    whenever step does not divide the 3600-point source."""
    # 1.3 deg -> step=13 -> 3600 // 13 = 276, but 360 / 1.3 = 276.9 -> 277.
    nx, _ = global_grid_dims(1.3, lat_max=60.0)
    assert nx == 276, f"expected integer-floor 276, got {nx}"
    assert nx != int(round(360.0 / 1.3))


def test_lat_max_boundary_inclusive():
    """A lat center exactly at +-lat_max is kept (|lat| <= lat_max), which is
    why 2*lat_max/res underestimates ny at some lat_max values."""
    # step=20 (2 deg): centers at -90, -70, ..., 50, 70, ... so lat_max=45
    # keeps 46 centers (-45 is a center) while 2*45/2 = 45.
    nx, ny = global_grid_dims(2.0, lat_max=45.0)
    assert ny == 46, f"lat_max=45 at 2 deg should keep 46 rows, got {ny}"


def test_nx_formula_matches_naive_when_step_divides():
    """Sanity: when step divides the source the naive formula is correct."""
    for step in [1, 2, 4, 5, 10, 20]:
        res = round(step * 0.1, 10)
        nx, _ = global_grid_dims(res, lat_max=85.0)
        assert nx == 3600 // step
        assert nx == int(round(360.0 / res))


# ── 2. Known-good resolutions produce the expected grids ─────────────

@pytest.mark.parametrize("res,lat_max,exp_nx,exp_ny", [
    (2.0, 60.0, 180, 60),
    (1.0, 60.0, 360, 120),      # the production grid
    (0.5, 60.0, 720, 240),
    (0.2, 60.0, 1800, 600),
    (1.0, 85.0, 360, 170),      # the GlobalGridConfig default
])
def test_expected_grid_sizes(res, lat_max, exp_nx, exp_ny):
    assert global_grid_dims(res, lat_max) == (exp_nx, exp_ny)


# ── 3. Rejected inputs ───────────────────────────────────────────────

def test_below_source_grid_rejected():
    with pytest.raises(ValueError):
        global_grid_dims(0.05, lat_max=60.0)


def test_sub_source_resolution_rejected_by_runner_cli():
    """--resolution 0.25 is not a multiple of the 0.1 deg source -> argparse
    exits before any grid work."""
    import subprocess
    r = subprocess.run(
        [sys.executable, "run_long_integration_global.py",
         "--resolution", "0.25", "--days", "0.001"],
        capture_output=True, text=True, cwd=REPOSITORY_ROOT / 'src', timeout=180)
    assert r.returncode != 0
    assert "multiple of the 0.1" in (r.stdout + r.stderr)


@pytest.mark.parametrize("res,lat_max", [(2.0, 1.0), (5.0, 5.0)])
def test_too_coarse_rejected(res, lat_max):
    nx, ny = global_grid_dims(res, lat_max)
    assert nx < 4 or ny < 4, f"expected a too-coarse grid, got {nx}x{ny}"


# ── 4. Full grid build (includes the make_global_grid assert) ────────

@pytest.mark.parametrize("res,lat_max", [
    (2.0, 45.0),   # previously rejected by the round() formula
    (2.0, 85.0),
    (1.3, 60.0),   # step=13, non-dividing
    (1.0, 66.5),
    (0.5, 60.0),
])
@requires_bathy
def test_grid_builds_at_various_resolutions(res, lat_max):
    from dataclasses import replace
    nx, ny = global_grid_dims(res, lat_max)
    gc = replace(GlobalGridConfig(), lat_max=lat_max, ny=ny, nx=nx,
                 resolution=res)
    g = make_global_grid(gc, BATHY, smooth_passes=30, min_depth=100.0)
    assert (g.nx, g.ny) == (nx, ny)
    # Physical/metric consistency: spacing must equal the requested resolution.
    assert np.isclose(g.lat[1] - g.lat[0], res, atol=1e-6)
    assert np.isclose(g.lon[1] - g.lon[0], res, atol=1e-6)
    # Spherical metric must use the ACTUAL spacing (dy is derived from dlat).
    assert np.isclose(g.dy, np.radians(res) * 6371000.0, rtol=1e-3)


@requires_bathy
def test_coarser_grid_has_fewer_points_than_finer():
    """Monotonicity: doubling resolution roughly quadruples the grid."""
    n1 = global_grid_dims(1.0, 60.0)
    n2 = global_grid_dims(0.5, 60.0)
    assert n2[0] == 2 * n1[0]
    assert n2[1] == 2 * n1[1]


# ── 5. Backward compatibility: the legacy --ny path is unchanged ─────

@requires_bathy
def test_legacy_default_grid_unchanged():
    """With no resolution override the grid must be the historical 1 deg
    360x120 production grid."""
    from dataclasses import replace
    gc = replace(GlobalGridConfig(), lat_max=60.0, ny=120)
    assert gc.resolution == 1.0
    g = make_global_grid(gc, BATHY, smooth_passes=30, min_depth=100.0)
    assert (g.nx, g.ny) == (360, 120)


@requires_bathy
def test_legacy_ny_path_still_rejects_inconsistent_ny():
    """The pre-existing ETOPO-dimension assert still fires (not a regression)."""
    from dataclasses import replace
    gc = replace(GlobalGridConfig(), lat_max=60.0, ny=140)
    with pytest.raises(AssertionError):
        make_global_grid(gc, BATHY, smooth_passes=30, min_depth=100.0)


# ── 6. CFL guidance: dt must shrink with resolution ──────────────────
#
# Three 1-deg-calibrated params violate their CFL on finer grids; the runner
# auto-scales them by dx^1/2/4. See docs/resolution_cfl_limits.md for the
# measured divergence table.

def test_physics_autoscale_matches_dx_powers():
    """scaled_physics_for_resolution maps 1 deg -> (dt_bt, nu_h, nu_bi)
    with exponents 1, 2, 4 and is a no-op at 1.0 deg."""
    from run_long_integration_global import (
        DT_BT_DEFAULT,
        NU_BI_REF_1DEG,
        NU_H_REF_1DEG,
        scaled_physics_for_resolution,
    )

    # None = not overridden -> scaled from the 1 deg reference
    dt_bt, nu_h, nu_bi = scaled_physics_for_resolution(2.0, None, None, None)
    assert dt_bt == pytest.approx(DT_BT_DEFAULT * 2.0)
    assert nu_h == pytest.approx(NU_H_REF_1DEG * 4.0)
    assert nu_bi == pytest.approx(NU_BI_REF_1DEG * 16.0)

    dt_bt, nu_h, nu_bi = scaled_physics_for_resolution(0.5, None, None, None)
    assert dt_bt == pytest.approx(DT_BT_DEFAULT * 0.5)
    assert nu_h == pytest.approx(NU_H_REF_1DEG * 0.25)
    assert nu_bi == pytest.approx(NU_BI_REF_1DEG * (0.5 ** 4))

    # 1.0 deg is a no-op -> legacy defaults preserved exactly
    dt_bt, nu_h, nu_bi = scaled_physics_for_resolution(1.0, None, None, None)
    assert dt_bt == DT_BT_DEFAULT
    assert nu_h == NU_H_REF_1DEG
    assert nu_bi == NU_BI_REF_1DEG


def test_physics_autoscale_respects_explicit_override():
    """An explicitly-passed value is never overwritten by the scaling."""
    from run_long_integration_global import scaled_physics_for_resolution

    dt_bt, nu_h, nu_bi = scaled_physics_for_resolution(
        0.25, 999.0, 123.0, 456.0)
    assert dt_bt == 999.0
    assert nu_h == 123.0
    assert nu_bi == 456.0

    # partial override: only the given one is kept
    dt_bt, nu_h, nu_bi = scaled_physics_for_resolution(0.5, 111.0, None, None)
    assert dt_bt == 111.0
    assert nu_h != 111.0


@pytest.mark.parametrize("res", [2.0, 1.0, 0.5, 0.2])
@requires_bathy
def test_external_wave_cfl_scales_with_resolution(res):
    """dt_bt must shrink linearly with resolution (barotropic mode).
    External gravity-wave CFL for the FD free surface is
    0.5*min(dx,dy)/sqrt(g*H). The limiting dx is the poleward-most column
    (cos(lat) smallest), so dt_cfl is NOT the equatorial value -- but it must
    still scale linearly with resolution, since every dx scales with it.
    This bounds --dt-bt only; the full-step limit is separate and steeper."""
    ref = _dt_cfl_at(1.0)           # 1 deg reference (same lat_max)
    got = _dt_cfl_at(res)
    assert np.isclose(got, ref * res, rtol=0.02), (
        f"res={res}: dt_cfl={got:.0f}s should be {ref * res:.0f}s "
        f"from the 1 deg reference {ref:.0f}s")

    # Barotropic dt_bt=150 s (production) stays safe down to 0.5 deg
    # (dt_cfl 70.7 s is below 150 -- so 0.5 deg needs dt_bt <= 60 s); the
    # 0.4-deg value (56.5 s) makes that crossover explicit.
    if res >= 0.5:
        assert got > 60.0, f"res={res}: expected dt_cfl > 60 (dt=60 safe)"
    else:
        assert got < 60.0, "0.4 deg and finer must NOT use dt=60"


@requires_bathy
def test_dt60_crossover_between_0p5_and_0p4():
    """Pin the barotropic crossover: the external-wave dt_cfl crosses 60 s
    between 0.5 and 0.4 deg. (The full baroclinic step has a *different*,
    steeper limit -- see docs/resolution_cfl_limits.md.)"""
    assert _dt_cfl_at(0.5) > 60.0
    assert _dt_cfl_at(0.4) < 60.0


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
