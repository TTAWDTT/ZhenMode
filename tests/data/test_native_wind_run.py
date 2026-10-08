"""Actual small core evolution and strict state/budget/time continuation."""

import json

import netCDF4
import numpy as np
import pytest

from tests.data.test_jra55 import WEATHER
from tests.data.test_native_fd_inputs import _prepared
from zhenmode.execution.native_run import integrate_fd_wind
from zhenmode.provenance.sources import sha256_file


def _inputs(tmp_path):
    native, policy = _prepared(tmp_path)
    from zhenmode.execution.native_fd import prepare_fd_native_inputs

    inputs = tmp_path / "fd-inputs"
    prepare_fd_native_inputs(native, policy, inputs)
    entries = []
    for field, variable, units, value, kind, cadence, height in WEATHER:
        path = tmp_path / (field + ".nc")
        with netCDF4.Dataset(path, "w") as data:
            data.source_id = "MRI-JRA55-do-1-4-0"
            data.data_kind = "manufactured"
            for dim, size in (("time", 2), ("lon", 2), ("lat", 2), ("bnds", 2)):
                data.createDimension(dim, size)
            for axis, values, unit in (
                ("lon", [90, 270], "degrees_east"),
                ("lat", [-45, 45], "degrees_north"),
            ):
                c = data.createVariable(axis, "f8", (axis,))
                c.units = unit
                c[:] = values
            time = data.createVariable("time", "f8", ("time",))
            time.units = "seconds since 1970-01-01 00:00:00"
            time.calendar = "proleptic_gregorian"
            if kind == "mean":
                time.bounds = "time_bounds"
                data.createVariable("time_bounds", "f8", ("time", "bnds"))[:] = [
                    [0, cadence],
                    [cadence, 2 * cadence],
                ]
                time[:] = [cadence / 2, 1.5 * cadence]
            else:
                time[:] = [0, cadence]
            if height:
                c = data.createVariable("height", "f8")
                c.units = "m"
                c[...] = height
            v = data.createVariable(variable, "f8", ("time", "lat", "lon"))
            v.units = units
            v.cell_methods = "time: " + kind
            v[:] = 0.001 if field in {"snow", "calving"} else value
        entries.append(
            {
                "field": field,
                "path": path.name,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "source_url": "urn:manufactured:native-wind-test",
                "license": "test fixture",
            }
        )
    manifest = tmp_path / "forcing.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "product": "JRA55-do",
                "version": "1.4.0",
                "data_kind": "manufactured",
                "files": entries,
            }
        )
    )
    return {
        "native_prepared": str(inputs),
        "forcing_manifest": str(manifest),
        "start": "1970-01-01T00:00:00",
        "dt_seconds": 0.1,
        "steps": 2,
        "resume": None,
        "polar_cap_rows": 0,
        "polar_cap_taper": 0,
        "match_transport": False,
    }


def _run(config, output):
    output.mkdir()
    return integrate_fd_wind(config | {"output": str(output)}, required_backend="cpu")


def _archive(path):
    with np.load(path, allow_pickle=False) as data:
        return {
            k: data[k].copy() for k in data.files if k != "metadata_json" and k != "metadata_sha256"
        }


def test_actual_wind_steps_save_all_state_and_restart_the_identical_trajectory(tmp_path):
    config = _inputs(tmp_path)
    whole = tmp_path / "whole"
    report = _run(config, whole)
    first = tmp_path / "first"
    _run(config | {"steps": 1}, first)
    rest = tmp_path / "rest"
    resumed = _run(config | {"steps": 1, "resume": str(first / "checkpoint.npz")}, rest)
    assert report["status"] == resumed["status"] == "completed_component"
    assert resumed["initial_step"] == 1 and resumed["completed_timesteps"] == 2
    assert resumed["elapsed_simulation_seconds"] == 0.2
    a, b = _archive(whole / "checkpoint.npz"), _archive(rest / "checkpoint.npz")
    assert a.keys() == b.keys()
    for name in a:
        np.testing.assert_array_equal(a[name], b[name])
    assert set(k[7:] for k in a if k.startswith("state__")) == {"u", "v", "T", "S", "eta", "ice"}
    assert np.any(a["state__u"] != 0) and np.isfinite(a["state__T"]).all()
    assert report["full_surface_blockers"]["nonzero_snow"] > 0
    assert report["full_surface_blockers"]["nonzero_calving"] > 0
    assert not report["full_case_qualification"] and not report["freshwater_routing_applied"]
    assert not np.any(a["cumulative__source_inputs"])


def test_changed_time_step_is_refused_without_changing_the_parent_checkpoint(tmp_path):
    config = _inputs(tmp_path)
    first = tmp_path / "first"
    _run(config | {"steps": 1}, first)
    checkpoint = first / "checkpoint.npz"
    original = sha256_file(checkpoint)
    out = tmp_path / "refused"
    with pytest.raises(ValueError, match="restart contract mismatch"):
        _run(config | {"dt_seconds": 0.2, "resume": str(checkpoint)}, out)
    assert sha256_file(checkpoint) == original
    report = json.loads((out / "run.json").read_text())
    assert report["status"] == "failed" and report["completed_timesteps"] == 0
