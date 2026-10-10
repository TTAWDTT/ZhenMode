"""Manufactured source-reader controls; no observations masquerade as fixtures."""

import json

import netCDF4
import numpy as np
import pytest

from tests.support.weather_source import originals
from zhenmode.preparation.wind_sample import prepare, validate_sample


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
    validate_sample(r)


def test_rejects_changed_source_and_bad_request(tmp_path):
    acquisition, _ = originals(tmp_path)
    with pytest.raises(ValueError, match="point request"):
        prepare(acquisition, tmp_path / "bad.json", latitude=100)
    with netCDF4.Dataset(tmp_path / "uas.nc", "a") as d:
        d["uas"][0, 1, 1] += 1
    with pytest.raises(ValueError, match="identity changed"):
        prepare(acquisition, tmp_path / "corrupted.json")


@pytest.mark.parametrize("field", [
    "data_kind", "weights", "atmospheric_samples", "stress_samples_N_m2", "tau_x_N_m2",
    "actual_point", "sources", "acquisition_sha256", "package_source_sha256",
])
def test_complete_receipt_replay_rejects_modified_fields(tmp_path, field):
    acquisition, _ = originals(tmp_path)
    r = prepare(acquisition, tmp_path / "stress.json")
    if field == "data_kind":
        r[field] = "observed_weather_derived_stress"
    elif field == "weights":
        r[field] = [1, 0, 0]
    elif field == "atmospheric_samples":
        r[field]["wind_u"][0] += 1
    elif field == "stress_samples_N_m2":
        r[field]["tau_x"][0] += 0.01
    elif field == "tau_x_N_m2":
        r[field] += 0.01
    elif field == "actual_point":
        r[field]["latitude"] += 1
    elif field == "sources":
        r[field]["wind_u"]["original"][0]["sha256"] = "0" * 64
    elif field == "acquisition_sha256":
        r[field] = "0" * 64
    else:
        r[field]["zhenmode/preparation/wind_sample"] = "0" * 64
    with pytest.raises(ValueError):
        validate_sample(r)


def test_manufactured_acquisition_cannot_claim_publisher_identity(tmp_path):
    acquisition, _ = originals(tmp_path)
    r = json.loads(acquisition.read_text())
    r["data_kind"] = "observed"
    acquisition.write_text(json.dumps(r))
    with pytest.raises(ValueError, match="publisher identities"):
        prepare(acquisition, tmp_path / "forged.json")
