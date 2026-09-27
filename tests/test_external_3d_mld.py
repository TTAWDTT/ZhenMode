import numpy as np
from netCDF4 import Dataset

from score_external_3d import score_external_3d


def test_external_3d_scores_mld_when_salt_available(tmp_path):
    z = np.array([0.0, -100.0])
    reference = np.stack(
        [np.full((2, 2), 20.0), np.full((2, 2), 10.0)], axis=-1)
    reference_salt = np.full_like(reference, 35.0)
    npz_path = tmp_path / "ref.npz"
    np.savez(
        npz_path,
        T_init=reference,
        S_init=reference_salt,
        wet_mask=np.ones((2, 2), dtype=bool),
        lat=np.array([0.0, 1.0]),
        lon=np.array([0.0, 1.0]),
        z=z,
    )
    nc_path = tmp_path / "prog.nc"
    temp = reference.transpose(2, 1, 0)[None, ...]
    with Dataset(nc_path, "w") as ds:
        for name, size in (("time", 1), ("zl", 2), ("yh", 2), ("xh", 2)):
            ds.createDimension(name, size)
        for name, values in (("temp", temp),
                             ("salt", reference_salt.transpose(2, 1, 0)[None, ...])):
            var = ds.createVariable(
                name, "f8", ("time", "zl", "yh", "xh"))
            var[:] = values
        wet = ds.createVariable("wet", "i1", ("yh", "xh"))
        wet[:] = np.ones((2, 2), dtype=np.int8)
        lath = ds.createVariable("lath", "f8", ("yh",))
        lath[:] = np.array([0.0, 1.0])
        lonh = ds.createVariable("lonh", "f8", ("xh",))
        lonh[:] = np.array([0.0, 1.0])
        depth = ds.createVariable("D", "f8", ("yh", "xh"))
        depth[:] = np.full((2, 2), 500.0)
    result = score_external_3d(
        str(nc_path), variable="temp", reference_path=str(npz_path),
        lat_var="lath", lon_var="lonh", wet_var="wet", depth_var="D",
        steady_days=10.0, salt_var="salt")
    assert "mld" in result
    assert "mld_latitude_bands" in result
    assert result["mld"]["n"] == 4
    assert result["mld"]["raw_bias_m"] == 0.0


def test_external_3d_uses_actual_time_coordinates(tmp_path):
    z = np.array([0.0, -100.0])
    reference = np.stack(
        [np.full((2, 2), 20.0), np.full((2, 2), 10.0)], axis=-1)
    reference_salt = np.full_like(reference, 35.0)
    npz_path = tmp_path / "ref.npz"
    np.savez(
        npz_path,
        T_init=reference,
        S_init=reference_salt,
        wet_mask=np.ones((2, 2), dtype=bool),
        lat=np.array([0.0, 1.0]),
        lon=np.array([0.0, 1.0]),
        z=z,
    )
    nc_path = tmp_path / "prog.nc"
    # A distant early record and a near-final late record. Index-based scoring
    # would select the wrong pair; the time coordinate must drive the window.
    temp = np.stack([
        np.full((2, 2, 2), 100.0),
        reference.transpose(2, 1, 0),
    ], axis=0)
    salt = np.stack([reference_salt.transpose(2, 1, 0)] * 2, axis=0)
    with Dataset(nc_path, "w") as ds:
        for name, size in (("time", 2), ("zl", 2), ("yh", 2), ("xh", 2)):
            ds.createDimension(name, size)
        time = ds.createVariable("time", "f8", ("time",))
        time.units = "days since 2023-01-01"
        time[:] = np.array([0.0, 360.0])
        temp = ds.createVariable("temp", "f8", ("time", "zl", "yh", "xh"))
        temp[:] = temp_values = np.stack([
            reference.transpose(2, 1, 0) + 80.0,
            reference.transpose(2, 1, 0),
        ], axis=0)
        salt = ds.createVariable("salt", "f8", ("time", "zl", "yh", "xh"))
        salt[:] = salt_values = np.stack([
            reference_salt.transpose(2, 1, 0),
            reference_salt.transpose(2, 1, 0),
        ], axis=0)
        wet = ds.createVariable("wet", "i1", ("yh", "xh"))
        wet[:] = np.ones((2, 2), dtype=np.int8)
        lath = ds.createVariable("lath", "f8", ("yh",))
        lath[:] = np.array([0.0, 1.0])
        lonh = ds.createVariable("lonh", "f8", ("xh",))
        lonh[:] = np.array([0.0, 1.0])
        depth = ds.createVariable("D", "f8", ("yh", "xh"))
        depth[:] = np.full((2, 2), 500.0)
    result = score_external_3d(
        str(nc_path), variable="temp", reference_path=str(npz_path),
        lat_var="lath", lon_var="lonh", wet_var="wet", depth_var="D",
        steady_days=10.0, salt_var="salt")
    assert result["global_3d"]["raw_rmse"] == 0.0
