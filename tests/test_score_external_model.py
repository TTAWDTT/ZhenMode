"""Model-neutral external benchmark scorer tests."""
import netCDF4
import numpy as np

from score_external_model import score_external_field


def _write_reference(path):
    lat = np.array([30.0, 45.0, 55.0])
    lon = np.array([300.0, 320.0, 0.0])
    wet = np.array([[True, True, True], [True, False, True], [True, True, False]])
    reference = np.stack([20.0 * np.ones((3, 3)), 22.0 * np.ones((3, 3))], axis=-1)
    np.savez(path, T_init=reference, wet_mask=wet, lat=lat, lon=lon)


def _write_external(path, values, lat, lon):
    with netCDF4.Dataset(path, "w") as ds:
        ds.createDimension("time", values.shape[0])
        ds.createDimension("lat", values.shape[1])
        ds.createDimension("lon", values.shape[2])
        time = ds.createVariable("time", "f8", ("time",))
        latv = ds.createVariable("lat", "f8", ("lat",))
        lonv = ds.createVariable("lon", "f8", ("lon",))
        wet = ds.createVariable("wet_mask", "i1", ("lat", "lon"))
        temp = ds.createVariable("sst", "f8", ("time", "lat", "lon"))
        time[0] = 25.0
        if values.shape[0] > 1:
            time[1] = 30.0
        latv[:] = lat
        lonv[:] = lon
        wet[:] = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 0]], dtype=np.int8)
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
    # Only the final 5d record is averaged before scoring.
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







def test_score_external_field_accepts_mom6_2d_centers(tmp_path):
    ref_path = tmp_path / "reference.npz"
    model_path = tmp_path / "mom6.nc"
    _write_reference(ref_path)
    # MOM6 stores a structured tile with lat varying down rows and lon across columns.
    with netCDF4.Dataset(model_path, "w") as ds:
        ds.createDimension("time", 1)
        ds.createDimension("zl", 1)
        ds.createDimension("yh", 3)
        ds.createDimension("xh", 3)
        ds.createVariable("time", "f8", ("time",))
        ds.createVariable("temp", "f8", ("time", "yh", "xh"))
        ds.createVariable("geolat", "f8", ("yh", "xh"))
        ds.createVariable("geolon", "f8", ("yh", "xh"))
        ds.createVariable("wet", "i1", ("yh", "xh"))
        ds["time"][:] = 30.0
        ds["temp"][:] = np.array([[[18.0, 20.0, 22.0],
                                    [21.0, 999.0, 23.0],
                                    [23.0, 25.0, 999.0]]], dtype=float)
        ds["geolat"][:] = np.array([[30.0] * 3, [45.0] * 3, [55.0] * 3], dtype=float)
        ds["geolon"][:] = np.array([[300.0, 320.0, 0.0]] * 3, dtype=float)
        ds["wet"][:] = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 0]], dtype=np.int8)
    result = score_external_field(model_path, variable="temp", reference_path=ref_path,
                                  geometry=model_path, lat_var="geolat", lon_var="geolon",
                                  wet_var="wet", level=0)
    assert result["verdict"] == "PASS"
    assert result["n_scored"] == 7


def test_score_external_field_rejects_empty_output(tmp_path):
    ref_path = tmp_path / "reference.npz"
    model_path = tmp_path / "empty.nc"
    _write_reference(ref_path)
    with netCDF4.Dataset(model_path, "w") as ds:
        ds.createDimension("time", 0)
        ds.createDimension("lat", 3)
        ds.createDimension("lon", 3)
        ds.createVariable("sst", "f8", ("time", "lat", "lon"))
        ds.createVariable("lat", "f8", ("lat",))
        ds.createVariable("lon", "f8", ("lon",))
        ds["lat"][:] = np.array([30.0, 45.0, 55.0], dtype=float)
        ds["lon"][:] = np.array([300.0, 320.0, 0.0], dtype=float)
    try:
        score_external_field(model_path, variable="sst", reference_path=ref_path)
    except RuntimeError as exc:
        assert "no time records" in str(exc)
    else:
        raise AssertionError("empty output was accepted")
