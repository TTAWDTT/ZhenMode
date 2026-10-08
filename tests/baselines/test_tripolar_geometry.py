"""Known geometry, polar/dateline controls, bounded extension and publication."""

import json

import netCDF4
import numpy as np
import pytest

from zhenmode.baselines.mom6 import tripolar
from zhenmode.execution.benchmark import main
from zhenmode.provenance.sources import sha256_file


@pytest.mark.parametrize("offset", [0, 359, -301])
def test_rectangular_areas_and_meridians_include_dateline(offset):
    lon, lat = np.meshgrid(np.array([0.0, 0.5, 1.0]) + offset, [-45.0, 0.0, 45.0])
    dx, dy, area = tripolar.supergrid_metrics(lon, lat)
    expected = tripolar.RADIUS_M**2 * np.deg2rad(0.5) * np.diff(np.sin(np.deg2rad(lat[:, 0])))
    np.testing.assert_allclose(area, np.broadcast_to(expected[:, None], area.shape), rtol=1e-13)
    np.testing.assert_allclose(dy, tripolar.RADIUS_M * np.pi / 4, rtol=1e-14)
    np.testing.assert_allclose(dx[1], tripolar.RADIUS_M * np.deg2rad(0.5), rtol=1e-13)


def test_polar_cap_and_degenerate_pole_cells_have_correct_area():
    lon, lat = np.meshgrid([0.0, 1.0, 2.0], [88.0, 89.0, 90.0])
    _, _, area = tripolar.supergrid_metrics(lon, lat)
    expected = tripolar.RADIUS_M**2 * np.deg2rad(1) * (1 - np.sin(np.deg2rad(89)))
    np.testing.assert_allclose(area[-1], expected, rtol=2e-12)
    assert tripolar._pole_area([0, 90, 180, 270], [90, 90, 90, 90]) == 0
    # Isolated pole: incoming and outgoing meridians bound one triangle.
    assert tripolar._pole_area([0, 1, 12], [89, 89, 90]) == pytest.approx(expected, rel=2e-12)


def _mercator():
    u = np.arcsinh(np.tan(np.deg2rad(-78))) + 0.01 * np.arange(41)
    x, y = np.meshgrid(np.arange(9) * 45.0, np.rad2deg(np.arctan(np.sinh(u))))
    return x, y


def test_continuation_keeps_original_bits_and_uses_whole_native_rows():
    x, y = _mercator()
    xx, yy, n = tripolar.extend_southern_mercator(x, y, -78.2)
    assert n > 0 and n % 2 == 0 and yy[0, 0] <= -78.2
    np.testing.assert_array_equal(xx[n:].view(np.uint64), x.view(np.uint64))
    np.testing.assert_array_equal(yy[n:].view(np.uint64), y.view(np.uint64))
    np.testing.assert_allclose(np.diff(np.arcsinh(np.tan(np.deg2rad(yy[:, 0])))), 0.01, rtol=1e-11)
    assert tripolar.extend_southern_mercator(x, y, -77.0)[2] == 0


@pytest.mark.parametrize("south", [np.nan, np.inf, -90.0, 0.0, 3.0, -89.5])
def test_invalid_boundary_is_refused(south):
    with pytest.raises(ValueError):
        tripolar.extend_southern_mercator(*_mercator(), south)


@pytest.mark.parametrize("defect", ["nan", "nonrectangular", "nonmercator", "wrongshape"])
def test_invalid_continuation_source_is_refused(defect):
    x, y = _mercator()
    if defect == "nan":
        y[0, 0] = np.nan
    elif defect == "nonrectangular":
        y[3, 2] += 0.01
    elif defect == "nonmercator":
        y[3, :] += 0.01
    else:
        y = y[:-1]
    with pytest.raises(ValueError):
        tripolar.extend_southern_mercator(x, y, -78.2)


@pytest.mark.parametrize("defect", ["latitude", "nan", "shape"])
def test_invalid_metric_coordinates_are_refused(defect):
    x, y = _mercator()
    if defect == "latitude":
        y[0, 0] = -91
    elif defect == "nan":
        x[0, 0] = np.nan
    else:
        x = x[:-1]
    with pytest.raises(ValueError):
        tripolar.supergrid_metrics(x, y)


def _mock_acquisition(tmp_path, monkeypatch):
    # A manufactured full-shape pole-cap grid stands in for pinned bytes ONLY
    # in this test via monkeypatch. No production manufactured-data bypass.
    path = tmp_path / "files/OM_1deg/INPUT/ocean_hgrid.nc"
    path.parent.mkdir(parents=True)
    u = np.arcsinh(np.tan(np.deg2rad(-78))) + 0.008726646259971648 * np.arange(641)
    latitude = np.rad2deg(np.arctan(np.sinh(u)))
    latitude[-1] = 90
    x, y = np.meshgrid(np.linspace(-300, 60, 721), latitude)
    with netCDF4.Dataset(path, "w") as ds:
        ds.createDimension("nyp", 641)
        ds.createDimension("nxp", 721)
        for k, v in [("x", x), ("y", y)]:
            ds.createVariable(k, "f8", ("nyp", "nxp"))[:] = v
    archive = tmp_path / "fixture.bin"
    archive.write_bytes(b"manufactured fixture")
    monkeypatch.setattr(tripolar, "SOURCE_GRID_SHA256", sha256_file(path))
    receipt = tmp_path / "acquisition.json"
    receipt.write_text(
        json.dumps(
            {
                "host": "ftp.gfdl.noaa.gov",
                "remote_path": "/perm/Alistair.Adcroft/MOM6-testing/OM_1deg.tgz",
                "status": "official_archive_retrieved_and_regular_members_extracted",
                "archive_path": archive.name,
                "archive_sha256": sha256_file(archive),
                "bytes": archive.stat().st_size,
                "files": [
                    {
                        "path": "files/OM_1deg/INPUT/ocean_hgrid.nc",
                        "sha256": sha256_file(path),
                        "bytes": path.stat().st_size,
                    }
                ],
            }
        )
    )
    return receipt, path, archive


def test_cli_writes_candidate_without_claiming_case_readiness(tmp_path, monkeypatch):
    receipt, _, _ = _mock_acquisition(tmp_path, monkeypatch)
    out = tmp_path / "prepared"
    assert (
        main(
            [
                "prepare-tripolar-grid",
                "--acquisition",
                str(receipt),
                "--south-boundary",
                "-78.6",
                "--output",
                str(out),
            ]
        )
        == 0
    )
    report = json.loads((out / "tripolar_grid.json").read_text())
    assert report["added_supergrid_rows"] % 2 == 0
    for flag in (
        "execution_ready",
        "MOM6_read_verified",
        "ocean_mask_prepared",
        "geography_certified",
    ):
        assert report[flag] is False
    assert report["completed_timesteps"] == 0
    assert report["output"]["sha256"] == sha256_file(out / "ocean_hgrid.nc")
    with netCDF4.Dataset(out / "ocean_hgrid.nc") as ds:
        assert ds["area"].units == "m2"
        assert ds["area"][:].sum() == pytest.approx(report["area_m2"], rel=1e-15)
    with pytest.raises(FileExistsError):
        tripolar.prepare_tripolar_grid(receipt, -78.6, out)


@pytest.mark.parametrize("defect", ["grid", "archive", "identity"])
def test_corrupt_acquisition_fails_before_output_directory(tmp_path, monkeypatch, defect):
    receipt, grid, archive = _mock_acquisition(tmp_path, monkeypatch)
    if defect == "grid":
        grid.write_bytes(b"broken")
    elif defect == "archive":
        archive.write_bytes(b"broken")
    else:
        value = json.loads(receipt.read_text())
        value["files"][0]["sha256"] = "0" * 64
        receipt.write_text(json.dumps(value))
    out = tmp_path / "refused"
    with pytest.raises(ValueError):
        tripolar.prepare_tripolar_grid(receipt, -78.6, out)
    assert not out.exists()


@pytest.mark.parametrize("changed", ["acquisition", "module"])
def test_precalculation_identity_snapshot_rejects_changed_inputs(tmp_path, monkeypatch, changed):
    receipt, _, _ = _mock_acquisition(tmp_path, monkeypatch)
    module = tmp_path / "fixture-module.py"
    module.write_text("original fixture source")
    monkeypatch.setattr(tripolar, "source_paths", lambda *args: {"fixture_module": module})
    original = tripolar.supergrid_metrics

    def mutate_after_calculation(*args):
        result = original(*args)
        if changed == "acquisition":
            receipt.write_text(receipt.read_text() + "\n")
        else:
            module.write_text("changed fixture source")
        return result

    monkeypatch.setattr(tripolar, "supergrid_metrics", mutate_after_calculation)
    out = tmp_path / "refused"
    with pytest.raises(ValueError, match="changed during preparation"):
        tripolar.prepare_tripolar_grid(receipt, -78.6, out)
    assert not out.exists()


def test_failed_netcdf_publication_leaves_destination_retryable(tmp_path, monkeypatch):
    receipt, _, _ = _mock_acquisition(tmp_path, monkeypatch)
    original = tripolar.netCDF4.Dataset

    def fail_output(path, mode="r", *args, **kwargs):
        if mode == "w":
            Path(path).write_bytes(b"partial NetCDF")
            raise OSError("publication interrupted")
        return original(path, mode, *args, **kwargs)

    from pathlib import Path

    monkeypatch.setattr(tripolar.netCDF4, "Dataset", fail_output)
    out = tmp_path / "prepared"
    with pytest.raises(OSError, match="publication interrupted"):
        tripolar.prepare_tripolar_grid(receipt, -78.6, out)
    assert not out.exists()
    assert not list(tmp_path.glob(".prepared.staging-*"))
    monkeypatch.setattr(tripolar.netCDF4, "Dataset", original)
    assert tripolar.prepare_tripolar_grid(receipt, -78.6, out)["output"]["path"] == "ocean_hgrid.nc"
