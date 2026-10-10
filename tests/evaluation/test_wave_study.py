"""Frozen plants for the space/time study, authored before native study runs."""

import json

import numpy as np
import pytest

from tests.support.standing_wave import cgrid, fixture
from zhenmode.benchmarks.standing_wave import (
    MOM6,
    OCEANANIGANS,
    STUDY_PAIRS,
    digest,
    study_contract,
    validate_contract,
)
from zhenmode.evaluation.standing_wave import score
from zhenmode.evaluation.wave_study import analyze, modal_trends
from zhenmode.provenance.sources import sha256_file


def study_fixture(nx, dt):
    c = study_contract(nx, dt)
    _, a = fixture(c["case"])
    a["metadata"].update(
        schema=c["schema"], contract_sha256=digest(c), dt_s=c["dt"], steps=int(32000 / dt)
    )
    return c, a


@pytest.mark.parametrize("nx,dt", STUDY_PAIRS)
def test_independently_constructed_wave_passes_every_study_pair(nx, dt):
    c, a = study_fixture(nx, dt)
    validate_contract(c)
    assert score(c, a)["engineering_screen_pass"]
    r = modal_trends(c, a)
    assert abs(r["relative_frequency_bias"]) < 1e-14
    assert abs(r["log_amplitude_change_per_period"]) < 1e-14


@pytest.mark.parametrize("bias,decay", [(0.01, 0.02), (-0.01, 0.05)])
def test_recovers_frequency_and_decay_planted_by_separate_formula(bias, decay):
    c, a = study_fixture(64, 25)
    time = a["time"] / 32000
    angles = 2 * np.pi * (1 + bias) * time
    decay_values = np.exp(-decay * time)
    spatial = 2 * np.pi * a["x_eta"] / (32000 * np.sqrt(981))
    a["eta"] = 0.01 * np.cos(spatial)[None, :] * (decay_values * np.cos(angles))[:, None]
    u = (
        0.01
        * np.sqrt(9.81 / 100)
        * np.sin(spatial)[None, :]
        * (decay_values * np.sin(angles))[:, None]
    )
    a["u"] = np.broadcast_to(u[..., None], a["u"].shape).copy()
    a["h"][:] = c["initial_h_m"]
    a["h"][:, :, 0] += a["eta"]
    a["volume_u"] = a["h"] * a["area"][None, :, None]
    a["volume_v"] = a["volume_u"].copy()
    r = modal_trends(c, a)
    assert r["relative_frequency_bias"] == pytest.approx(bias, abs=2e-14)
    assert r["log_amplitude_change_per_period"] == pytest.approx(-decay, abs=2e-14)
    assert r["phase_fit_max_residual_rad"] < 2e-14
    assert r["log_amplitude_fit_max_residual"] < 2e-14


@pytest.mark.parametrize("pair", [(64, 100), (128, 50), (256, 75), (True, 25), (256, True)])
def test_rejects_unfrozen_or_ambiguous_pair(pair):
    with pytest.raises(ValueError):
        study_contract(*pair)


def test_corrupted_declared_timestep_is_rejected():
    c, a = study_fixture(64, 25)
    a["metadata"]["dt_s"] = 100
    with pytest.raises(ValueError, match="timestep"):
        modal_trends(c, a)


def planted_corpus(tmp_path):
    directories = []
    for nx, dt in STUDY_PAIRS:
        c = study_contract(nx, dt)
        root = tmp_path / f"nx{nx}-dt{dt}"
        root.mkdir()
        (root / "contract.json").write_text(json.dumps(c))
        receipt = {
            "status": "completed",
            "models": {},
            "package_source_sha256": {"synthetic": "f" * 64},
        }
        for model in ("zhenmode", "mom6", "oceananigans"):
            _, a = fixture(c["case"], sampling="node" if model == "zhenmode" else "cell_mean")
            options = c["ocean_options"]
            if model != "zhenmode":
                if model == "oceananigans":
                    a["h"] = np.asarray(c["initial_h_m"]) * (1 + a["eta"][..., None] / 100)
                a = cgrid(a, c)
                a["metadata"].update(
                    model="MOM6" if model == "mom6" else "Oceananigans",
                    source_sha=MOM6 if model == "mom6" else OCEANANIGANS,
                )
                options = c["mom_time_options" if model == "mom6" else "oceananigans_options"]
            a["metadata"].update(
                schema=c["schema"], contract_sha256=digest(c), dt_s=c["dt"], steps=int(32000 / dt)
            )
            a["metadata"]["recorded_numerics"]["resolved_options"] = options
            folder = root / model
            folder.mkdir()
            path = folder / "output.npz"
            np.savez_compressed(path, **(a | {"metadata": np.array(json.dumps(a["metadata"]))}))
            saved = score(c, a)
            saved["output_sha256"] = sha256_file(path)
            receipt["models"][model] = saved
        (root / "comparison.json").write_text(json.dumps(receipt))
        directories.append(root)
    return directories


def test_full_reader_and_report_path_recovers_exact_fifteen_run_plant(tmp_path):
    directories = planted_corpus(tmp_path)
    result = analyze(directories, tmp_path / "report.json")
    assert result["native_runs"] == len(result["rows"]) == 15
    assert max(abs(row["trends"]["relative_frequency_bias"]) for row in result["rows"]) < 1e-14
    assert (
        max(abs(row["trends"]["log_amplitude_change_per_period"]) for row in result["rows"]) < 1e-14
    )
    with pytest.raises(ValueError, match="duplicate"):
        analyze([*directories[:-1], directories[0]], tmp_path / "duplicate.json")
    with pytest.raises(ValueError, match="five frozen"):
        analyze(directories[:-1], tmp_path / "missing.json")
    path = directories[0] / "comparison.json"
    value = json.loads(path.read_text())
    value["models"]["zhenmode"]["metrics"]["phase_rad"] += 0.01
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="output/score changed"):
        analyze(directories, tmp_path / "corrupted.json")
