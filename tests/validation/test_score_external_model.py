"""Model-neutral external benchmark scorer tests."""
import netCDF4
import numpy as np
import pytest

from ocean_solver.evaluation.external import (
    _relative_drift,
    _time_days,
    score_external_field,
)


def _write_reference(path):
    lat = np.array([30.0, 45.0, 55.0])
    lon = np.array([300.0, 320.0, 0.0])
    wet = np.array([[True, True, True], [True, False, True], [True, True, False]])
    reference = np.stack([20.0 * np.ones((3, 3)), 22.0 * np.ones((3, 3))], axis=-1)
    np.savez(path, T_init=reference, wet_mask=wet, lat=lat, lon=lon)


def _write_external(path, values, lat, lon, wet_mask=None):
    with netCDF4.Dataset(path, "w") as ds:
        ds.createDimension("time", values.shape[0])
        ds.createDimension("lat", values.shape[1])
        ds.createDimension("lon", values.shape[2])
        time = ds.createVariable("time", "f8", ("time",))
        time.units = "days since 2023-01-01 00:00:00"
        latv = ds.createVariable("lat", "f8", ("lat",))
        lonv = ds.createVariable("lon", "f8", ("lon",))
        wet = ds.createVariable("wet_mask", "i1", ("lat", "lon"))
        temp = ds.createVariable("sst", "f8", ("time", "lat", "lon"))
        time[0] = 25.0
        if values.shape[0] > 1:
            time[1] = 30.0
        latv[:] = lat
        lonv[:] = lon
        wet[:] = (np.array([[1, 1, 1], [1, 0, 1], [1, 1, 0]], dtype=np.int8)
                  if wet_mask is None else wet_mask)
        temp[:] = values


def test_score_external_field_uses_shared_grid_and_wet_mask(tmp_path):
    ref_path = tmp_path / "reference.npz"
    model_path = tmp_path / "model.nc"
    _write_reference(ref_path)
    # Arrays are y,x here and are transposed to the solver's x,y convention.
    model = np.array([[[18.0, 20.0, 22.0], [21.0, 999.0, 23.0], [23.0, 25.0, 999.0]],
                      [[19.0, 21.0, 23.0], [22.0, 999.0, 24.0], [24.0, 26.0, 999.0]]])
    _write_external(model_path, model, np.array([30.0, 45.0, 55.0]), np.array([300.0, 320.0, 0.0]))
    result = score_external_field(model_path, variable="sst", reference_path=ref_path,
                                  wet_var="wet_mask", steady_days=5.0)
    assert result["verdict"] == "PASS"
    assert result["days_end"] == 30.0
    assert result["time_records"] == 2
    assert result["steady_window_days"] == [25.0, 30.0]
    assert result["global"]["n"] == 7
    assert result["n_scored"] == 7


def test_score_external_field_rejects_coordinate_mismatch(tmp_path):
    ref_path = tmp_path / "reference.npz"
    model_path = tmp_path / "model.nc"
    _write_reference(ref_path)
    model = np.full((1, 3, 3), 20.0)
    _write_external(model_path, model, np.array([30.0, 45.0, 56.0]), np.array([300.0, 320.0, 0.0]))
    try:
        score_external_field(model_path, variable="sst", reference_path=ref_path)
    except RuntimeError as exc:
        assert "differ" in str(exc)
    else:
        raise AssertionError("coordinate mismatch was accepted")


def test_score_external_field_converts_hours_to_days(tmp_path):
    reference_path = tmp_path / "reference.npz"
    model_path = tmp_path / "model.nc"
    _write_reference(reference_path)
    values = np.broadcast_to(np.arange(61)[:, None, None], (61, 3, 3)).copy()
    _write_external(model_path, values, np.array([30., 45., 55.]), np.array([300., 320., 0.]))
    with netCDF4.Dataset(model_path, "a") as dataset:
        dataset["time"].units = "hours since 2023-01-01 00:00:00"
        dataset["time"][:] = np.arange(61) * 24.
    result = score_external_field(model_path, variable="sst", reference_path=reference_path,
                                  wet_var="wet_mask", steady_days=30.)
    assert result["days_end"] == 60.
    assert result["steady_window_days"] == [30., 60.]
    assert result["global"]["raw_bias"] == pytest.approx(25.)
    model_path.unlink()


def test_score_external_field_rejects_missing_time_units(tmp_path):
    reference_path = tmp_path / "reference.npz"
    model_path = tmp_path / "model.nc"
    _write_reference(reference_path)
    _write_external(model_path, np.full((2, 3, 3), 20.),
                    np.array([30., 45., 55.]), np.array([300., 320., 0.]))
    with netCDF4.Dataset(model_path, "a") as dataset:
        dataset["time"].delncattr("units")
    with pytest.raises(ValueError, match="units"):
        score_external_field(model_path, variable="sst", reference_path=reference_path)


def test_score_external_field_flags_incomplete_coverage(tmp_path):
    reference_path = tmp_path / "reference.npz"
    model_path = tmp_path / "model.nc"
    _write_reference(reference_path)
    _write_external(model_path, np.full((1, 3, 3), 20.),
                    np.array([30., 45., 55.]), np.array([300., 320., 0.]))
    with netCDF4.Dataset(model_path, "a") as dataset:
        dataset["wet_mask"][:] = 0
        dataset["wet_mask"][0, 0] = 1
    result = score_external_field(model_path, variable="sst", reference_path=reference_path,
                                  wet_var="wet_mask")
    assert result["n_scored"] == 1
    assert result["verdict"] == "FAIL"
    assert result["coverage_fraction"] == pytest.approx(1 / 7)


@pytest.mark.parametrize("units,calendar,scale", [
    ("seconds since 2000-01-01", "standard", 86400.),
    ("hours since 2000-01-01", "360_day", 24.),
    ("days since 2000-01-01", "noleap", 1.),
])
def test_cf_time_conversion_respects_units_and_calendar(tmp_path, units, calendar, scale):
    path = tmp_path / "time.nc"
    with netCDF4.Dataset(path, "w") as dataset:
        dataset.createDimension("time", 3)
        variable = dataset.createVariable("time", "f8", ("time",))
        variable.units, variable.calendar = units, calendar
        variable[:] = np.array([0., 30., 60.]) * scale
        np.testing.assert_allclose(_time_days(variable), [0., 30., 60.])


@pytest.mark.parametrize("values", [[0., 0.], [1., 0.], [0., np.nan], [0., np.inf]])
def test_cf_time_rejects_invalid_records(tmp_path, values):
    path = tmp_path / "time.nc"
    with netCDF4.Dataset(path, "w") as dataset:
        dataset.createDimension("time", 2)
        variable = dataset.createVariable("time", "f8", ("time",))
        variable.units = "days since 2000-01-01"
        variable[:] = values
        with pytest.raises(ValueError, match="finite|increasing"):
            _time_days(variable)


@pytest.mark.parametrize("values", [[], [1.], [0., 1.], [1., np.inf], [1., np.nan, 2.]])
def test_undefined_relative_drift_is_not_reported_as_zero(values):
    assert np.isnan(_relative_drift(np.asarray(values)))


def test_relative_drift_uses_absolute_initial_content():
    assert _relative_drift(np.array([-100., -90.])) == 10.


def test_no_land_fill_still_transposes_non_square_grid(tmp_path):
    reference_path = tmp_path / "reference.npz"
    model_path = tmp_path / "model.nc"
    lat, lon = np.array([30., 45., 55.]), np.array([300., 320., 340., 0.])
    model = np.arange(12.).reshape(1, 3, 4)
    np.savez(reference_path, T_init=model[0].T[:, :, None], wet_mask=np.ones((4, 3)), lat=lat, lon=lon)
    _write_external(model_path, model, lat, lon, wet_mask=np.ones((3, 4)))
    result = score_external_field(model_path, variable="sst", reference_path=reference_path,
                                  replace_land_with_reference=False)
    assert result["global"]["raw_rmse"] == 0.


def test_mom6_cli_uses_normalized_shared_time_window(tmp_path, monkeypatch, capsys):
    from ocean_solver.evaluation.external import main

    reference_path = tmp_path / "reference.npz"
    model_path = tmp_path / "model.nc"
    output = tmp_path / "score.json"
    _write_reference(reference_path)
    _write_external(model_path, np.full((2, 3, 3), 20.),
                    np.array([30., 45., 55.]), np.array([300., 320., 0.]))
    with netCDF4.Dataset(model_path, "a") as dataset:
        dataset.renameVariable("sst", "temp")
        dataset.renameVariable("lat", "lath")
        dataset.renameVariable("lon", "lonh")
        dataset.renameVariable("wet_mask", "wet")
        dataset["time"].units = "hours since 2023-01-01"
        dataset["time"][:] = [600., 720.]
    monkeypatch.setattr("sys.argv", ["score_external_model", "--input", str(model_path), "--variable", "temp",
                                    "--lat-var", "lath", "--lon-var", "lonh", "--wet-var", "wet",
                                    "--geometry", str(model_path), "--reference-npz", str(reference_path),
                                    "--steady-days", "5", "--out", str(output)])
    main()
    import json

    result = json.loads(output.read_text())
    assert result["days_end"] == 30.
    assert result["steady_window_days"] == [25., 30.]
    assert result["n_model_wet"] == 7
    assert capsys.readouterr().out


def test_external_score_uses_provided_shared_area_without_independence_claim(tmp_path):
    reference_path = tmp_path / "reference.npz"
    model_path = tmp_path / "model.nc"
    _write_reference(reference_path)
    with np.load(reference_path) as saved:
        payload = {name: saved[name] for name in saved.files}
    area = np.arange(1., 10.).reshape(3, 3)
    np.savez(reference_path, **payload, cell_area_m2=area)
    values = np.arange(18.).reshape(2, 3, 3) + 20.
    _write_external(model_path, values, payload["lat"], payload["lon"],
                    wet_mask=payload["wet_mask"].T)
    result = score_external_field(model_path, variable="sst", reference_path=reference_path,
                                  wet_var="wet_mask")
    error = values.mean(axis=0).T - 20.
    ocean = payload["wet_mask"]
    assert result["global"]["raw_bias"] == pytest.approx(np.average(error[ocean], weights=area[ocean]))
    assert result["global"]["raw_rmse"] == pytest.approx(np.sqrt(np.average(error[ocean] ** 2, weights=area[ocean])))
    assert result["metric_definition"] == "area_weighted_angular_box_v2"
    assert result["area_source"] == "provided_cell_area"
    assert result["reference_role"] == "shared_initialization_field_not_independent_validation"

