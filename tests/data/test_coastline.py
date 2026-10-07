"""Planted geometry controls independent of original GSHHG geography quality."""

import struct

import numpy as np
import pytest

from zhenmode.model.inputs.coastline import rasterize_gshhg
from zhenmode.provenance.sources import sha256_file


def _polygon(identifier, level, points, *, crossing=0, bounds=None):
    points = np.rint(np.asarray(points) * 1e6).astype(">i4")
    region = bounds or [
        points[:, 0].min(),
        points[:, 0].max(),
        points[:, 1].min(),
        points[:, 1].max(),
    ]
    header = struct.pack(
        ">11i", identifier, len(points), level | (crossing << 16), *map(int, region), 1, 1, -1, -1
    )
    return header + points.tobytes()


def test_known_land_lake_hierarchy_keeps_freshwater_out_of_marine(tmp_path):
    path = tmp_path / "fixture.b"
    path.write_bytes(
        _polygon(1, 1, [[0, 0], [3, 0], [3, 3], [0, 3]])
        + _polygon(2, 2, [[1, 1], [2, 1], [2, 2], [1, 2]])
    )
    levels = rasterize_gshhg(
        path, [0.5, 1.5, 2.5, 3.5], [0.5, 1.5, 2.5], expected_sha256=sha256_file(path)
    )
    np.testing.assert_array_equal(levels, [[1, 1, 1], [1, 2, 1], [1, 1, 1], [0, 0, 0]])
    assert levels[1, 1] != 0  # Negative lake relief must not define marine support.


def test_greenwich_and_dateline_polygons_do_not_create_world_spanning_stripe(tmp_path):
    path = tmp_path / "fixture.b"
    path.write_bytes(
        _polygon(0, 1, [[359, 0], [1, 0], [1, 2], [359, 2]], crossing=1)
        + _polygon(2, 1, [[179, -2], [181, -2], [181, 0], [179, 0]], crossing=2)
        # A vertex a few microdegrees outside an approximate header must stay
        # at 32 degrees, not become -328 and fill an unrelated longitude row.
        + _polygon(
            3,
            1,
            [[31, 3], [32.000003, 3], [32.000003, 5], [31, 5]],
            bounds=[31000000, 32000000, 3000000, 5000000],
        )
    )
    levels = rasterize_gshhg(
        path, [0.5, 31.5, 100.5, 180.0, 359.5], [-1.0, 1.0, 4.0], expected_sha256=sha256_file(path)
    )
    np.testing.assert_array_equal(levels, [[0, 1, 0], [0, 0, 1], [0, 0, 0], [1, 0, 0], [0, 1, 0]])


def test_antarctic_ice_front_closes_through_pole_and_ground_line_is_skipped(tmp_path):
    path = tmp_path / "fixture.b"
    path.write_bytes(
        _polygon(
            4,
            5,
            [[180, -70], [0, -70], [-180, -70]],
            crossing=3,
            bounds=[-180000000, 180000000, -90000000, -70000000],
        )
        + _polygon(5, 6, [[0, 0], [3, 0], [3, 3], [0, 3]])
    )
    levels = rasterize_gshhg(
        path, [0.5, 90.0, 270.0], [-89.5, -60.0, 1.0], expected_sha256=sha256_file(path)
    )
    np.testing.assert_array_equal(levels, [[1, 0, 0], [1, 0, 0], [1, 0, 0]])


def test_identity_truncation_and_nonreal_coordinates_are_rejected(tmp_path):
    path = tmp_path / "fixture.b"
    path.write_bytes(_polygon(1, 1, [[0, 0], [2, 0], [2, 2], [0, 2]]))
    identity = sha256_file(path)
    with pytest.raises(ValueError, match="identity"):
        rasterize_gshhg(path, [0.5], [0.5], expected_sha256="0" * 64)
    for values in ([0.5, 0.5], [np.nan], [1 + 2j]):
        with pytest.raises(ValueError, match="coordinates"):
            rasterize_gshhg(path, values, [0.5], expected_sha256=identity)
    path.write_bytes(path.read_bytes()[:-1])
    with pytest.raises(ValueError, match="truncated"):
        rasterize_gshhg(path, [0.5], [0.5], expected_sha256=sha256_file(path))
