"""Native file/axis/capacity/source controls; manufactured values, not climate evidence."""

import json

import netCDF4
import numpy as np
import pytest

from tests.data.test_woa_initialization import fixture
from zhenmode.execution.benchmark import main
from zhenmode.execution.initialization import prepare_woa_thermodynamics
from zhenmode.execution.native_initialization import prepare_native_initialization
from zhenmode.provenance.sources import sha256_file


def _inputs(tmp_path, *, unanchored=False):
    acquisition, pressure = fixture(tmp_path)
    source = tmp_path / "source"
    prepare_woa_thermodynamics(acquisition, pressure, source)
    geometry = tmp_path / "geometry"
    geometry.mkdir()
    lon = np.array([90.0, 270.0])
    lat = np.array([-45.0, 45.0])
    bed = np.array([[0.5 if unanchored else 750.0, 6500.0], [750.0, 0.0]])
    area = np.full((2, 2), np.pi * 6371000.0**2)
    np.savez(
        geometry / "grid.npz",
        lon=lon,
        lat=lat,
        wet_mask=(bed > 0).astype(np.uint8),
        area=area,
        lon_bounds=[[0.0, 180.0], [180.0, 360.0]],
        lat_bounds=[[-90.0, 0.0], [0.0, 90.0]],
    )
    np.savez(geometry / "bathymetry.npz", lon=lon, lat=lat, depth=bed)
    (geometry / "geometry.json").write_text(
        json.dumps(
            {
                "grid_sha256": sha256_file(geometry / "grid.npz"),
                "bathymetry_sha256": sha256_file(geometry / "bathymetry.npz"),
            }
        )
    )
    nodes = tmp_path / "nodes.json"
    nodes.write_text(json.dumps({"z_nodes_m": [0.0, -500.0, -4000.0, -6000.0, -7000.0]}))
    return source, geometry, nodes, bed, area


def test_native_points_axis_support_deep_extension_and_equivalent_states(tmp_path):
    source, geometry, nodes, bed, area = _inputs(tmp_path)
    out = tmp_path / "native"
    report = prepare_native_initialization(source, geometry, nodes, out)
    assert report["status"] == "native_points_prepared" and report["complete_wet_support"]
    assert not report["execution_ready"] and not report["mom_layer_initialization_ready"]
    assert report["data_kind"] == "manufactured"
    assert report["levels"][1]["horizontal_infill_nodes"] == 1
    assert report["levels"][3]["deep_extension_nodes"] == 1
    assert report["water_inventory"]["geometry_volume_m3"] == float(np.sum(bed * area))
    assert report["water_inventory"]["resolved_initial_volume_m3"] == pytest.approx(
        float(np.sum(bed * area)), rel=2e-16
    )
    assert report["water_inventory"]["deep_extension_volume_m3"] == float(area[0, 1] * 2500.0)
    with (
        netCDF4.Dataset(out / report["output"]["path"]) as ds,
        netCDF4.Dataset(source / "woa13v2_thermodynamics.nc") as original,
    ):
        np.testing.assert_array_equal(ds["lon"][:], [90.0, 270.0])
        assert ds["ptemp"].dimensions == ("time", "depth", "lat", "lon")
        assert ds["salt"].units == "1" and ds["ct"].units == "degrees_celsius"
        assert ds["ptemp"][0, 0, 0, 0] == original["ptemp"][0, 0, 0, 1]
        assert ds["ptemp"][0, 3, 1, 0] == original["ptemp"][0, 2, 1, 1]
        np.testing.assert_allclose(ds["salt"][0].compressed(), 35.0, rtol=0, atol=1e-14)
        np.testing.assert_array_equal(ds["thickness_m"][:].sum(axis=0), bed.T)
        assert ds["deep_extension_thickness_m"][:].sum() == 2500.0
        assert np.ma.getmaskarray(ds["ct"][:]).sum() == 12
    assert sha256_file(out / report["output"]["path"]) == report["output"]["sha256"]
    with pytest.raises(FileExistsError):
        prepare_native_initialization(source, geometry, nodes, out)


def test_unanchored_support_is_retained_and_cli_returns_nonzero(tmp_path, capsys):
    source, geometry, nodes, bed, area = _inputs(tmp_path, unanchored=True)
    out = tmp_path / "native"
    code = main(
        [
            "prepare-native-initial",
            "--source-prepared",
            str(source),
            "--geometry",
            str(geometry),
            "--nodes-file",
            str(nodes),
            "--output",
            str(out),
        ]
    )
    assert code == 3
    report = json.loads((out / "native_initialization.json").read_text())
    assert report["status"] == "blocked_unanchored_native_support"
    assert not report["complete_wet_support"] and not report["native_initialization_ready"]
    assert report["levels"][1]["unresolved_nodes"] == 1
    assert (
        report["water_inventory"]["resolved_initial_volume_m3"]
        < report["water_inventory"]["geometry_volume_m3"]
    )
    with netCDF4.Dataset(out / report["output"]["path"]) as ds:
        assert ds["ptemp"][0, 1, 0, 1] is np.ma.masked
        assert ds["unresolved_support"][1, 0, 1] == 1


def test_source_units_and_changed_geometry_identity_are_rejected(tmp_path):
    source, geometry, nodes, _, _ = _inputs(tmp_path)
    with netCDF4.Dataset(source / "woa13v2_thermodynamics.nc", "a") as ds:
        ds["ptemp"].units = "K"
    p = source / "initialization.json"
    receipt = json.loads(p.read_text())
    receipt["output"]["sha256"] = sha256_file(source / "woa13v2_thermodynamics.nc")
    receipt["output"]["bytes"] = (source / "woa13v2_thermodynamics.nc").stat().st_size
    p.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="source field dimensions/units"):
        prepare_native_initialization(source, geometry, nodes, tmp_path / "bad-units")
    assert (
        json.loads((tmp_path / "bad-units/native_initialization.json").read_text())["status"]
        == "failed"
    )
    with (geometry / "grid.npz").open("ab") as f:
        f.write(b"identity defect")
    with pytest.raises(ValueError, match="identity"):
        prepare_native_initialization(source, geometry, nodes, tmp_path / "bad-identity")


def test_late_receipt_change_never_publishes_completed_native_status(tmp_path, monkeypatch):
    import zhenmode.execution.native_initialization as implementation

    source, geometry, nodes, _, _ = _inputs(tmp_path)
    original = implementation.smooth_native_paired_holes
    changed = False

    def mutate(*args, **kwargs):
        nonlocal changed
        result = original(*args, **kwargs)
        if not changed:
            nodes.write_text(nodes.read_text() + "\n")
            changed = True
        return result

    monkeypatch.setattr(implementation, "smooth_native_paired_holes", mutate)
    with pytest.raises(ValueError, match="receipt changed"):
        prepare_native_initialization(source, geometry, nodes, tmp_path / "late-change")
    assert (
        json.loads((tmp_path / "late-change/native_initialization.json").read_text())["status"]
        == "failed"
    )


def test_identity_bound_wrong_area_is_not_cancelled_by_inventory_reuse(tmp_path):
    source, geometry, nodes, _, _ = _inputs(tmp_path)
    with np.load(geometry / "grid.npz") as d:
        values = {k: d[k].copy() for k in d.files}
    values["area"] *= 2.0
    np.savez(geometry / "grid.npz", **values)
    path = geometry / "geometry.json"
    receipt = json.loads(path.read_text())
    receipt["grid_sha256"] = sha256_file(geometry / "grid.npz")
    path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="spherical bounds"):
        prepare_native_initialization(source, geometry, nodes, tmp_path / "wrong-area")


@pytest.mark.parametrize("include_source_bottom", [False, True])
def test_deep_extension_reuses_prepared_source_plane_across_shallower_sill(tmp_path, include_source_bottom):
    source, geometry, nodes, bed, _ = _inputs(tmp_path)
    bed[0, 0] = 4500.0  # source-depth anchor, below the deep target's 6000 m plane.
    np.savez(geometry / "bathymetry.npz", lon=[90.0, 270.0], lat=[-45.0, 45.0], depth=bed)
    geo_path = geometry / "geometry.json"
    geo = json.loads(geo_path.read_text())
    geo["bathymetry_sha256"] = sha256_file(geometry / "bathymetry.npz")
    geo_path.write_text(json.dumps(geo))
    prepared = source / "woa13v2_thermodynamics.nc"
    with netCDF4.Dataset(prepared, "a") as data:
        data["ptemp"][0, 2, 0, 1] = 1.0
        data["ptemp"][0, 2, 1, 1] = np.ma.masked
        data["sr"][0, 2, 1, 1] = np.ma.masked
        data["paired_source_support"][0, 2, 1, 1] = 0
    origin_path = source / "initialization.json"
    origin = json.loads(origin_path.read_text())
    origin["output"].update(bytes=prepared.stat().st_size, sha256=sha256_file(prepared))
    origin_path.write_text(json.dumps(origin))
    if not include_source_bottom:
        nodes.write_text(json.dumps({"z_nodes_m": [0.0, -500.0, -6000.0, -7000.0]}))
    output = tmp_path / "native-deep"
    report = prepare_native_initialization(source, geometry, nodes, output)
    assert report["complete_wet_support"]
    assert report["deep_extension_reference"]["source_depth_m"] == 4000.0
    with netCDF4.Dataset(output / "native_point_fields.nc") as data:
        deep = int(np.flatnonzero(data["depth"][:] == 6000.0)[0])
        assert data["ptemp"][0, deep, 1, 0] == 1.0
        assert data["horizontal_infill"][deep, 1, 0] == 1
        assert data["unresolved_support"][deep, 1, 0] == 0
        assert data["nearest_original_anchor_flat_index"][deep, 1, 0] == 0
        if include_source_bottom:
            assert data["ptemp"][0, deep, 1, 0] == data["ptemp"][0, 2, 1, 0]
    reference = report["deep_extension_reference"]
    assert sha256_file(output / reference["path"]) == reference["sha256"]
    assert report["levels"][deep]["infill_system_reference_depth_m"] == 4000.0
    assert report["levels"][deep]["infill_inherited_from_deepest_prepared_plane"]
