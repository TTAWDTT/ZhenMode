"""Native dimension/order controls; manufactured arrays are never run evidence."""

import json

import netCDF4
import numpy as np
import pytest

from tests.support.standing_wave import cgrid, fixture
from zhenmode.baselines.oceananigans.adapter import convert
from zhenmode.benchmarks.standing_wave import contract
from zhenmode.evaluation.standing_wave import validate


def test_native_time_first_netcdf_axes_and_layer_order(tmp_path):
    c = contract()
    _, a = fixture(sampling="cell_mean")
    a = cgrid(a, c)
    nt, nx, ny, nz = 33, c["nx"], c["ny"], c["nz"]
    h = np.broadcast_to(c["initial_h_m"], (nt, nx, ny, nz)).copy()
    eta = a["eta"].reshape(nt, nx, ny)
    h *= 1 + eta[..., None] / 100
    path = tmp_path / "native.nc"
    with netCDF4.Dataset(path, "w") as d:
        for name, size in (("time", nt), ("z", nz), ("x", nx), ("y", ny), ("yf", ny + 1)):
            d.createDimension(name, size)

        def field(name, values, dims):
            d.createVariable(name, "f8", dims)[:] = values

        field("time", a["time"], ("time",))
        for key, shape in (
            ("x_eta", (nx, ny)),
            ("y_eta", (nx, ny)),
            ("x_u", (nx, ny)),
            ("y_u", (nx, ny)),
            ("x_v", (nx, ny + 1)),
            ("y_v", (nx, ny + 1)),
        ):
            values = a[key].reshape(shape)
            axis = "x" if key.startswith("x") else "yf" if key == "y_v" else "y"
            field(key, values[:, 0] if axis == "x" else values[0, :], (axis,))
        field("eta", eta.transpose(0, 2, 1), ("time", "y", "x"))
        fields = {
            "h": h,
            "T": np.full_like(h, 15),
            "S": np.full_like(h, 35),
            "u": a["u"].reshape(nt, nx, ny, nz),
            "v": a["v"].reshape(nt, nx, ny + 1, nz),
        }
        for name, values in fields.items():
            field(
                name,
                values[..., ::-1].transpose(0, 3, 2, 1),
                ("time", "z", "yf" if name == "v" else "y", "x"),
            )
        d.initialization_s = 1.0
        d.integration_s = 2.0
        d.accepted_steps = 320
        d.vertical_order = "bottom_to_top"
        d.model_description = "manufactured axes control"
        d.substepping = "manufactured"
    info = {
        "environment": {"julia_executable_sha256": "b" * 64, "project_sha256": {}},
        "inputs": {"configuration_sha256": "c" * 64, "driver_sha256": "d" * 64},
    }
    (tmp_path / "native-run.json").write_text(json.dumps(info))
    arrays, m = convert(c, tmp_path, {"elapsed_wall_seconds": 4, "peak_host_rss_bytes": 1000000})
    arrays["metadata"] = m
    validate(c, arrays)
    np.testing.assert_array_equal(arrays["h"].reshape(nt, nx, ny, nz), h)
    np.testing.assert_array_equal(arrays["u"], a["u"])
    np.testing.assert_array_equal(arrays["v"], a["v"])


def test_unverified_julia_launcher_is_rejected_before_model_run(tmp_path, monkeypatch):
    from zhenmode.baselines.oceananigans import adapter
    from zhenmode.provenance.sources import sha256_file

    recorded = tmp_path / "verified-julia"
    selected = tmp_path / "other-julia"
    recorded.write_bytes(b"verified interpreter")
    selected.write_bytes(b"different interpreter")
    info = {
        "runtime": {"julia_executable": str(recorded)},
        "julia_executable_sha256": sha256_file(recorded),
    }
    monkeypatch.setattr(adapter, "verify", lambda cache: info)
    monkeypatch.setattr(adapter.subprocess, "check_output", lambda *args, **kwargs: str(selected))
    configuration = tmp_path / "config.json"
    configuration.write_text(
        json.dumps({"oceananigans_cache": str(tmp_path), "julia": "changed-launcher"})
    )
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError, match="unverified executable"):
        adapter.integrate(configuration)
    assert not (tmp_path / "native-run.json").exists()
