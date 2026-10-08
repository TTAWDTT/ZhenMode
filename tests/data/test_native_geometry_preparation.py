"""Manufactured shoreline/depth/routing controls; no geography certificate."""

import json

import netCDF4
import numpy as np
import pytest

from zhenmode.model.inputs.coastline import (
    connect_binary_channels,
    marine_native_edges,
    marine_regions,
    wet_components,
)
from zhenmode.preparation.coast import prepare_native_geometry
from zhenmode.provenance.sources import sha256_file


def _inputs(tmp_path, elevation=-10.0):
    parent = tmp_path / "parent"
    parent.mkdir()
    lon, lat = np.arange(360) + 0.5, np.arange(180) - 89.5
    area = np.broadcast_to(
        6371000.0**2 * np.deg2rad(1.0) * np.diff(np.sin(np.deg2rad(np.arange(181) - 90)))[None, :],
        (360, 180),
    ).copy()
    np.savez(
        parent / "grid.npz",
        lon=lon,
        lat=lat,
        area=area,
        wet_mask=np.ones((360, 180)),
        lon_bounds=np.column_stack((np.arange(360), np.arange(1, 361))),
        lat_bounds=np.column_stack((np.arange(180) - 90, np.arange(1, 181) - 90)),
    )
    np.savez(parent / "bathymetry.npz", lon=lon, lat=lat, depth=np.full((360, 180), 10.0))
    relief = tmp_path / "relief.nc"
    with netCDF4.Dataset(relief, "w") as data:
        data.data_kind = "manufactured"
        for name, values in (
            ("lon", np.arange(40) * 9.0),
            ("lat", (np.arange(20) + 0.5) * 9.0 - 90),
        ):
            data.createDimension(name, len(values))
            data.createVariable(name, "f8", (name,))[:] = values
        variable = data.createVariable("z", "f8", ("lat", "lon"))
        variable.units = "meters"
        variable[:] = elevation
    (parent / "geometry.json").write_text(
        json.dumps(
            {
                "source": str(relief),
                "source_sha256": sha256_file(relief),
                "grid_sha256": sha256_file(parent / "grid.npz"),
                "bathymetry_sha256": sha256_file(parent / "bathymetry.npz"),
                "original_NOAA_bytes_reverified": False,
            }
        )
    )
    shoreline = tmp_path / "gshhs_h.b"
    shoreline.write_bytes(b"")
    acquisition = tmp_path / "acquisition.json"
    acquisition.write_text(
        json.dumps(
            {
                "product": "GSHHG",
                "version": "2.3.7",
                "data_kind": "manufactured",
                "files": [{"path": shoreline.name, "bytes": 0, "sha256": sha256_file(shoreline)}],
            }
        )
    )
    policy = tmp_path / "policy.json"
    policy.write_text(
        json.dumps({"policy": "binary_coast_channels_v1", "regional_refinements": []})
    )
    return parent, acquisition, policy, relief, area


def test_manufactured_all_sea_has_independent_depth_area_and_volume(tmp_path):
    parent, acquisition, policy, _, area = _inputs(tmp_path)
    out = tmp_path / "prepared"
    report = prepare_native_geometry(parent, acquisition, policy, out)
    assert report["status"] == "native_geometry_prepared" and report["data_kind"] == "manufactured"
    assert report["native_components"] == 1 and report["wet_cells"] == 64800
    assert not report["execution_ready"] and not report["source_geography_certified"]
    assert report["binary_volume_m3"] == pytest.approx(4 * np.pi * 6371000.0**2 * 10.0, rel=1e-12)
    assert report["binary_marine_area_m2"] == pytest.approx(4 * np.pi * 6371000.0**2, rel=1e-12)
    with np.load(out / "bathymetry.npz") as data:
        np.testing.assert_allclose(data["depth"], 10.0, rtol=1e-12, atol=1e-12)
    with np.load(out / "grid.npz") as data:
        np.testing.assert_array_equal(data["area"], area)
        np.testing.assert_allclose(data["land_fraction"], 0.0, atol=1e-12)
    for name, row in report["outputs"].items():
        assert sha256_file(out / name) == row["sha256"]
    with pytest.raises(FileExistsError):
        prepare_native_geometry(parent, acquisition, policy, out)


def test_marine_without_negative_relief_refuses_instead_of_deleting_sea(tmp_path):
    parent, acquisition, policy, _, _ = _inputs(tmp_path, elevation=10.0)
    with pytest.raises(ValueError, match="lack negative relief"):
        prepare_native_geometry(parent, acquisition, policy, tmp_path / "bad")
    report = json.loads((tmp_path / "bad/geometry.json").read_text())
    assert report["status"] == "failed" and not report["execution_ready"]


def test_changed_source_and_identity_bound_parent_area_are_rejected(tmp_path):
    parent, acquisition, policy, relief, _ = _inputs(tmp_path)
    with relief.open("ab") as handle:
        handle.write(b"changed original")
    with pytest.raises(ValueError, match="identity"):
        prepare_native_geometry(parent, acquisition, policy, tmp_path / "changed")
    receipt = json.loads((parent / "geometry.json").read_text())
    receipt["source_sha256"] = sha256_file(relief)
    with np.load(parent / "grid.npz") as data:
        values = {k: data[k].copy() for k in data.files}
    values["area"] *= 2
    np.savez(parent / "grid.npz", **values)
    receipt["grid_sha256"] = sha256_file(parent / "grid.npz")
    (parent / "geometry.json").write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="area"):
        prepare_native_geometry(parent, acquisition, policy, tmp_path / "area")


def test_channel_search_uses_source_edge_witness_and_minimum_new_cells():
    wet = np.zeros((5, 3), dtype=bool)
    wet[1, 1] = wet[3, 1] = True
    support = np.ones((5, 3))
    possible = np.ones((5, 3), dtype=bool)
    x = np.zeros_like(wet)
    x[:, 1] = True
    y = np.zeros_like(wet)
    final, records = connect_binary_channels(
        wet, support, possible, x, y, np.arange(5) + 0.5, np.arange(3) - 1.0
    )
    assert len(wet_components(wet)) == 2 and len(wet_components(final)) == 1
    assert records[0]["added_cells"] == [7] and final[2, 1]
    assert records[0]["route_native_cells"] == [10, 7, 4]
    # Planted unsupported shortcut / dry-depth route must never be used.
    x[1, 1] = x[0, 1] = False
    blocked, records = connect_binary_channels(
        wet, support, possible, x, y, np.arange(5) + 0.5, np.arange(3) - 1.0
    )
    np.testing.assert_array_equal(blocked, wet)
    assert records[0]["status"] == "unresolved_no_source_graph_path"


def test_fine_graph_seam_and_half_cell_alignment_have_known_answers():
    marine = np.zeros((4, 4), dtype=bool)
    marine[0, 1] = marine[-1, 1] = True
    regions, _ = marine_regions(marine, periodic_longitude=True)
    assert regions[0, 1] == regions[-1, 1]
    regions, _ = marine_regions(marine, periodic_longitude=False)
    assert regions[0, 1] != regions[-1, 1]
    fine = np.ones((6, 4), dtype=bool)
    cells, x, y = marine_native_edges(fine, 2, periodic_longitude=True)
    np.testing.assert_array_equal(cells, np.ones((3, 2), dtype=bool))
    assert x[-1].all() and not y[:, -1].any()
    _, x, _ = marine_native_edges(fine, 2, periodic_longitude=False)
    assert not x[-1].any()


def _one_reference_dry_conflict(tmp_path):
    parent, acquisition, policy, relief, area = _inputs(tmp_path)
    relief.unlink()
    with netCDF4.Dataset(relief, "w") as data:
        data.data_kind = "manufactured"
        for name, values in (("lon", np.arange(360)), ("lat", np.arange(180) - 89.5)):
            data.createDimension(name, len(values))
            data.createVariable(name, "f8", (name,))[:] = values
        v = data.createVariable("z", "f8", ("lat", "lon"))
        v.units = "meters"
        v[:] = -10
        # A source cell centred on integer longitude has half its support
        # on each native side. These two positive samples remove all depth
        # evidence only from native centre (0.5, 0.5).
        v[90, :2] = 10
    for name, key in (("grid.npz", "wet_mask"), ("bathymetry.npz", "depth")):
        with np.load(parent / name) as data:
            values = {k: data[k].copy() for k in data.files}
        values[key][0, 90] = 0
        np.savez(parent / name, **values)
    receipt = json.loads((parent / "geometry.json").read_text())
    receipt.update(
        source_sha256=sha256_file(relief),
        grid_sha256=sha256_file(parent / "grid.npz"),
        bathymetry_sha256=sha256_file(parent / "bathymetry.npz"),
    )
    (parent / "geometry.json").write_text(json.dumps(receipt))
    return parent, acquisition, policy, area


def test_named_reference_dry_conflict_preserves_uncertainty_and_only_one_cell(tmp_path):
    parent, acquisition, policy, area = _one_reference_dry_conflict(tmp_path)
    with pytest.raises(ValueError, match="lack negative relief"):
        prepare_native_geometry(parent, acquisition, policy, tmp_path / "strict")
    row = {"lon_lat": [0.5, 0.5], "basis": "retain_parent_dry_no_negative_marine_relief"}
    policy.write_text(
        json.dumps(
            {
                "policy": "binary_coast_channels_v1",
                "regional_refinements": [],
                "reference_dry_exclusions": [row],
            }
        )
    )
    report = prepare_native_geometry(parent, acquisition, policy, tmp_path / "prepared")
    assert report["wet_cells"] == 64799 and report["native_components"] == 1
    assert report["all_selected_depth_conflicts_explicitly_accounted"]
    assert not report["native_geographic_qualification"]
    assert not report["execution_ready"]
    (conflict,) = report["reference_dry_source_conflicts"]
    assert conflict["native_flat_index"] == 90
    assert not conflict["geography_resolved"]
    assert conflict["sampled_conflicting_marine_area_m2"] == pytest.approx(area[0, 90])
    with np.load(tmp_path / "prepared/bathymetry.npz") as data:
        assert data["depth"][0, 90] == 0
        assert np.count_nonzero(data["depth"] == 0) == 1


@pytest.mark.parametrize(
    "exclusions, message",
    [
        (
            [{"lon_lat": [1.5, 0.5], "basis": "retain_parent_dry_no_negative_marine_relief"}],
            "cannot delete known water",
        ),
        (
            [{"lon_lat": [0.0, 0.5], "basis": "retain_parent_dry_no_negative_marine_relief"}],
            "native centre",
        ),
        ([{"lon_lat": [0.5, 0.5], "basis": "invented"}], "invalid reference"),
        ([None], "invalid reference"),
        ("all", "explicit refinements"),
        (
            [{"lon_lat": [0.5, 0.5], "basis": "retain_parent_dry_no_negative_marine_relief"}] * 2,
            "cannot delete known water",
        ),
    ],
)
def test_exclusion_cannot_delete_water_or_accept_an_ambiguous_policy(tmp_path, exclusions, message):
    parent, acquisition, policy, _ = _one_reference_dry_conflict(tmp_path)
    policy.write_text(
        json.dumps(
            {
                "policy": "binary_coast_channels_v1",
                "regional_refinements": [],
                "reference_dry_exclusions": exclusions,
            }
        )
    )
    with pytest.raises(ValueError, match=message):
        prepare_native_geometry(parent, acquisition, policy, tmp_path / "bad")
