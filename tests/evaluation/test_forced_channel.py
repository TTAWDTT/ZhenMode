"""Known mechanical-work assembly through the actual native reader and validator."""

import json

import numpy as np
import pytest

from tests.support.channel_dynamics import planted
from zhenmode.benchmarks.forced_channel import contract, validate_contract
from zhenmode.evaluation.forced_channel import score
from zhenmode.evaluation.native_channel import load
from zhenmode.execution.forced_channel import run


def stress_source(tmp_path, tx):
    path = tmp_path / "manufactured-stress.json"
    path.write_text(
        json.dumps(
            dict(
                schema="jra-point-mean-stress-v1",
                data_kind="manufactured_weather_derived_stress",
                source_dates=["1958-01-01 00:00:00", "1958-01-01 03:00:00", "1958-01-01 06:00:00"],
                actual_point={"latitude": 0.0, "longitude": 180.0},
                tau_x_N_m2=tx,
                tau_y_N_m2=0.0,
            )
        )
    )
    return path


@pytest.mark.parametrize("stress", [0.0, 0.01])
def test_full_reader_recovers_planted_surface_work_and_water_salt_heat(tmp_path, stress):
    source = stress_source(tmp_path, stress)
    c = contract(source)
    a = planted(c, "ocean-solver")
    # Mechanical collector plant: fixed SSH, top-layer acceleration, unchanged tracers.
    # This does not claim a complete fluid-equation solution.
    a["eta"][:] = a["eta"][0]
    a["h"][:] = a["h"][0]
    for q in ("u", "v"):
        a[q][:] = 0.0
        a["volume_" + q] = a["h"] * a["area"][None, :, None]
    a["u"][:, :, 0] = stress * a["time"][:, None] / (1025 * a["h"][0, :, 0])
    path = tmp_path / "plant.npz"
    np.savez_compressed(path, **(a | {"metadata": np.array(json.dumps(a["metadata"]))}))
    report = score(c, load(path))
    assert report["data_kind"] == "manufactured_weather_derived_stress"
    assert not report["accuracy_ranking"]
    assert report["engineering_screen_pass"]
    assert report["metrics"]["work_residual_relative"] < 5e-13
    if stress == 0:
        assert report["trajectories"]["wind_work_J"] == [0.0] * len(a["time"])
    else:
        # Wrong observed surface velocities or an altered energy assembly must fire.
        a["u"][1:, :, 1] = 0.1
        assert score(c, a)["metrics"]["work_residual_relative"] > 1e-4


def test_manufactured_source_cannot_run_as_real_and_source_changes_are_rejected(tmp_path):
    source = stress_source(tmp_path, 0.01)
    c = contract(source)
    with pytest.raises(ValueError, match="observed"):
        run(source, tmp_path / "forbidden")
    data = json.loads(source.read_text())
    data["tau_x_N_m2"] += 0.01
    source.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="changed"):
        validate_contract(c)
