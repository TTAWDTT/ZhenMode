"""Manufactured source-reader controls; no observations masquerade as fixtures."""

import json
from pathlib import Path

import netCDF4
import numpy as np
import pytest

from zhenmode.model.inputs.forcing.jra55 import FIELDS
from zhenmode.preparation.wind_sample import prepare
from zhenmode.provenance.sources import sha256_file


def originals(root):
    row = json.loads((Path(__file__).parents[1] / "support/ncar_reference.json").read_text())[
        "cases"
    ][0]
    q = row["humidity_air"]
    ta = row["theta_air_k"] - 9.81 * 10 / (1004.6 * (1 + 0.8735 * q))
    files = {
        v[0]: dict(filename="unused-manufactured", bytes=0, sha256="0" * 64)
        for v in FIELDS.values()
    }
    values = {
        "uas": [5, -5, 5],
        "vas": [0, 0, 0],
        "tas": [ta] * 3,
        "huss": [q] * 3,
        "psl": [101325] * 3,
    }
    for field, (var, units, _, _, height) in FIELDS.items():
        if var not in values:
            continue
        path = root / (var + ".nc")
        with netCDF4.Dataset(path, "w") as d:
            for name, size in [("time", 3), ("lat", 3), ("lon", 3), ("height", 1)]:
                d.createDimension(name, size)
            for name, value, unit in [
                ("time", [0, 3, 6], "hours since 1958-01-01"),
                ("lat", [30, 45, 60], "degrees_north"),
                ("lon", [90, 180, 270], "degrees_east"),
            ]:
                a = d.createVariable(name, "f8", (name,))
                a[:] = value
                a.units = unit
            d["time"].calendar = "gregorian"
            a = d.createVariable("height", "f8", ("height",))
            a[:] = 10
            a.units = "m"
            a = d.createVariable(var, "f8", ("time", "lat", "lon"))
            a[:] = np.broadcast_to(np.array(values[var])[:, None, None], (3, 3, 3))
            a.units = units[0]
            a.cell_methods = "area: mean time: point"
            d.source_id = "MRI-JRA55-do-1-4-0"
            d.data_kind = "manufactured"
            d.license = "manufactured control, not observations"
        files[var] = dict(filename=path.name, bytes=path.stat().st_size, sha256=sha256_file(path))
    receipt = dict(
        product="JRA55-do",
        version="1.4.0",
        year=1958,
        data_kind="manufactured",
        execution_status="completed",
        files=files,
        verified={k: v["sha256"] for k, v in files.items()},
    )
    path = root / "acquisition.json"
    path.write_text(json.dumps(receipt))
    density = 101325 / (287.04 * ta * (1 + (28.966 / 18.016 - 1) * q))
    return path, density * row["cd"] * 25


def test_known_fortran_coefficient_point_and_zero_mean_roundtrip(tmp_path):
    acquisition, expected = originals(tmp_path)
    r = prepare(acquisition, tmp_path / "stress.json", sst_c=25)
    assert r["data_kind"] == "manufactured_weather_derived_stress"
    np.testing.assert_allclose(
        r["stress_samples_N_m2"]["tau_x"], [expected, -expected, expected], rtol=0, atol=3e-13
    )
    assert r["tau_x_N_m2"] == 0 and r["tau_y_N_m2"] == 0
    assert r["actual_point"] == {"latitude": 45.0, "longitude": 180.0}
    assert all(x["source_kind"] == "manufactured" for x in r["sources"].values())


def test_rejects_changed_source_and_bad_request(tmp_path):
    acquisition, _ = originals(tmp_path)
    with pytest.raises(ValueError, match="point request"):
        prepare(acquisition, tmp_path / "bad.json", latitude=100)
    with netCDF4.Dataset(tmp_path / "uas.nc", "a") as d:
        d["uas"][0, 1, 1] += 1
    with pytest.raises(ValueError, match="identity changed"):
        prepare(acquisition, tmp_path / "corrupted.json")
