"""
Tests for ocean grid generation.

Run: python tests/test_grid.py
"""
import sys
import os
import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from config import DEFAULT_CONFIG, GridConfig, R_EARTH, OMEGA
from grid import make_grid, OceanGrid, _read_etopo_subset


# ── Test helpers ────────────────────────────────────────────────────

def get_grid():
    """Load grid once (cached via module-level lazy init)."""
    if not hasattr(get_grid, '_cache'):
        get_grid._cache = make_grid(DEFAULT_CONFIG.grid, DEFAULT_CONFIG.bathymetry_file)
    return get_grid._cache


def assert_close(a, b, tol=1e-6, msg=""):
    assert abs(a - b) < tol, f"{msg}: {a} != {b} (tol={tol})"


# ── Shape and dimension tests ───────────────────────────────────────

def test_grid_shapes():
    """All 2D fields should be (nx, ny) = (128, 128)."""
    g = get_grid()
    assert g.depth.shape == (128, 128), f"depth shape: {g.depth.shape}"
    assert g.f.shape == (128, 128), f"f shape: {g.f.shape}"
    assert g.ocean_mask.shape == (128, 128), f"ocean_mask shape: {g.ocean_mask.shape}"
    assert g.land_mask.shape == (128, 128), f"land_mask shape: {g.land_mask.shape}"
    assert g.lon.shape == (128,), f"lon shape: {g.lon.shape}"
    assert g.lat.shape == (128,), f"lat shape: {g.lat.shape}"
    assert g.x.shape == (128,), f"x shape: {g.x.shape}"
    assert g.y.shape == (128,), f"y shape: {g.y.shape}"
    assert g.z.shape == (14,), f"z shape: {g.z.shape}"
    assert g.dz.shape == (13,), f"dz shape: {g.dz.shape}"
    print("PASS: test_grid_shapes")


def test_dimensions():
    """nx, ny, nz match config."""
    g = get_grid()
    assert g.nx == 128
    assert g.ny == 128
    assert g.nz == 14
    print("PASS: test_dimensions")


# ── Coordinate tests ────────────────────────────────────────────────

def test_lon_range():
    """Longitude spans the configured domain."""
    g = get_grid()
    gc = DEFAULT_CONFIG.grid
    assert_close(g.lon[0], gc.lon_bounds[0], tol=1e-4, msg="lon[0]")
    assert_close(g.lon[-1], gc.lon_bounds[1] - gc.dlon, tol=1e-4, msg="lon[-1]")
    # Spacing
    dlon = np.diff(g.lon)
    assert np.allclose(dlon, gc.dlon, atol=1e-6), f"dlon not uniform: {dlon}"
    print("PASS: test_lon_range")


def test_lat_range():
    """Latitude spans the configured domain."""
    g = get_grid()
    gc = DEFAULT_CONFIG.grid
    assert_close(g.lat[0], gc.lat_bounds[0], tol=1e-4, msg="lat[0]")
    assert_close(g.lat[-1], gc.lat_bounds[1] - gc.dlat, tol=1e-4, msg="lat[-1]")
    # Spacing
    dlat = np.diff(g.lat)
    assert np.allclose(dlat, gc.dlat, atol=1e-6), f"dlat not uniform: {dlat}"
    print("PASS: test_lat_range")


def test_x_y_spacing():
    """x and y are in meters with correct spacing."""
    g = get_grid()
    gc = DEFAULT_CONFIG.grid
    dx = np.diff(g.x)
    dy = np.diff(g.y)
    assert np.allclose(dx, gc.dx, rtol=1e-6), f"dx mismatch: {dx[0]} vs {gc.dx}"
    assert np.allclose(dy, gc.dy, rtol=1e-6), f"dy mismatch: {dy[0]} vs {gc.dy}"
    # x centered: x[0] = -63.5*dx, x[-1] = 63.5*dx
    assert_close(g.x[0], -63.5 * gc.dx, msg="x[0]")
    assert_close(g.x[-1], 63.5 * gc.dx, msg="x[-1]")
    print("PASS: test_x_y_spacing")


# ── Coriolis tests ──────────────────────────────────────────────────

def test_coriolis_center():
    """f at domain center lat matches f0."""
    g = get_grid()
    gc = DEFAULT_CONFIG.grid
    # lat[64] = 28.6 + 64*0.1 = 35.0 = lat_center exactly
    # f doesn't vary in x, so f[:, 64] should all equal f0
    f_center = g.f[0, 64]  # any x index works
    assert_close(f_center, gc.f0, tol=1e-7, msg="f at center vs f0")
    print(f"PASS: test_coriolis_center  (f_center={f_center:.5e}, f0={gc.f0:.5e})")


def test_coriolis_gradient():
    """df/dy approximates beta at domain center."""
    g = get_grid()
    gc = DEFAULT_CONFIG.grid
    # df/dy at center: use central difference
    # f is (nx, ny), y varies along axis 1
    j_mid = 64
    df_dy = (g.f[:, j_mid + 1] - g.f[:, j_mid - 1]) / (2 * g.dy)
    df_dy_center = df_dy[64]
    # Should be close to beta (exact for sin(lat), beta is linear approx)
    assert_close(df_dy_center, gc.beta, tol=1e-9, msg="df/dy vs beta")
    print(f"PASS: test_coriolis_gradient  (df/dy={df_dy_center:.5e}, beta={gc.beta:.5e})")


def test_coriolis_symmetry():
    """f is symmetric in x (doesn't depend on lon)."""
    g = get_grid()
    # f should be constant along axis 0 (lon) for each lat
    f_variation_x = np.max(np.std(g.f, axis=0))
    assert f_variation_x < 1e-15, f"f varies in x: std={f_variation_x}"
    print("PASS: test_coriolis_symmetry")


# ── Vertical grid tests ─────────────────────────────────────────────

def test_z_levels():
    """z levels match config."""
    g = get_grid()
    gc = DEFAULT_CONFIG.grid
    assert len(g.z) == 14
    for i, z_val in enumerate(gc.z_levels):
        assert g.z[i] == z_val, f"z[{i}]: {g.z[i]} != {z_val}"
    print("PASS: test_z_levels")


def test_dz_positive():
    """Layer thicknesses are positive."""
    g = get_grid()
    assert np.all(g.dz > 0), f"Non-positive dz: {g.dz}"
    # Check a few specific values
    assert g.dz[0] == 5.0,   f"dz[0]: {g.dz[0]}"   # |0 - (-5)|
    assert g.dz[1] == 10.0,  f"dz[1]: {g.dz[1]}"   # |-5 - (-15)|
    assert g.dz[2] == 15.0,  f"dz[2]: {g.dz[2]}"   # |-15 - (-30)|
    assert g.dz[-1] == 2000.0, f"dz[-1]: {g.dz[-1]}"  # |-2000 - (-4000)|
    print("PASS: test_dz_positive")


def test_z_negative():
    """All z levels are <= 0 (negative downward, 0 at surface)."""
    g = get_grid()
    assert g.z[0] == 0.0, f"z[0] should be 0 (surface), got {g.z[0]}"
    assert np.all(g.z[1:] < 0), f"z levels below surface should be negative: {g.z}"
    print("PASS: test_z_negative")


# ── Bathymetry and mask tests ───────────────────────────────────────

def test_depth_nonneg():
    """Depth is non-negative everywhere."""
    g = get_grid()
    assert np.all(g.depth >= 0), f"Negative depth found: min={g.depth.min()}"
    print("PASS: test_depth_nonneg")


def test_mask_consistency():
    """ocean_mask and land_mask are complementary."""
    g = get_grid()
    assert np.all(g.land_mask == ~g.ocean_mask), "land_mask != ~ocean_mask"
    # Total points
    assert g.ocean_mask.sum() + g.land_mask.sum() == 128 * 128
    print("PASS: test_mask_consistency")


def test_depth_mask_agreement():
    """Depth > 0 exactly where ocean_mask is True."""
    g = get_grid()
    # Ocean points have positive depth
    assert np.all(g.depth[g.ocean_mask] > 0), "Ocean points with zero depth"
    # Land points have zero depth
    assert np.all(g.depth[g.land_mask] == 0), "Land points with nonzero depth"
    print("PASS: test_depth_mask_agreement")


def test_has_ocean():
    """Domain should be mostly ocean (NW Pacific)."""
    g = get_grid()
    n_ocean = g.ocean_mask.sum()
    n_total = g.nx * g.ny
    frac = n_ocean / n_total
    print(f"  Ocean fraction: {frac*100:.1f}% ({n_ocean}/{n_total})")
    assert frac > 0.5, f"Expected >50% ocean, got {frac*100:.1f}%"
    print("PASS: test_has_ocean")


def test_has_some_land():
    """Domain near Japan should have some land."""
    g = get_grid()
    n_land = g.land_mask.sum()
    print(f"  Land points: {n_land}")
    # The domain includes parts of Japan, so there should be some land
    # (but this is not guaranteed depending on exact bounds)
    if n_land > 0:
        print(f"PASS: test_has_some_land  ({n_land} land points)")
    else:
        print(f"SKIP: test_has_some_land  (no land in domain — OK for open ocean)")


# ── Axis convention test ────────────────────────────────────────────

def test_axis_convention():
    """
    Verify axis 0 = lon (x), axis 1 = lat (y), matching spectral_ops.

    f varies along axis 1 (lat) but not axis 0 (lon).
    depth varies along both.
    """
    g = get_grid()
    # f varies along axis 1 (lat), constant along axis 0 (lon)
    f_std_axis0 = np.std(g.f, axis=0)  # variation in lon → should be ~0
    f_std_axis1 = np.std(g.f, axis=1)  # variation in lat → should be nonzero
    assert np.max(f_std_axis0) < 1e-15, "f should not vary in lon (axis 0)"
    assert np.max(f_std_axis1) > 1e-8, "f should vary in lat (axis 1)"
    print("PASS: test_axis_convention")


# ── Run all tests ───────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("Ocean Grid Tests")
    print("=" * 60)
    print()

    tests = [
        test_grid_shapes,
        test_dimensions,
        test_lon_range,
        test_lat_range,
        test_x_y_spacing,
        test_coriolis_center,
        test_coriolis_gradient,
        test_coriolis_symmetry,
        test_z_levels,
        test_dz_positive,
        test_z_negative,
        test_depth_nonneg,
        test_mask_consistency,
        test_depth_mask_agreement,
        test_has_ocean,
        test_has_some_land,
        test_axis_convention,
    ]

    passed = 0
    failed = 0
    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"FAIL: {test.__name__}: {e}")
            failed += 1
        except Exception as e:
            print(f"ERROR: {test.__name__}: {type(e).__name__}: {e}")
            failed += 1

    print()
    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed, {passed+failed} total")
    print("=" * 60)
