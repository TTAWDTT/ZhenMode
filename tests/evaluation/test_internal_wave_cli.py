"""Offline CLI receipts and routing, using synthetic arrays without model runs."""

import hashlib
import json
import struct
import zipfile

import numpy as np
import pytest

from tests.support.internal_wave import planted
from zhenmode.benchmarks.internal_wave import contract, development_contract
from zhenmode.benchmarks.standing_wave import digest
from zhenmode.cli import main
from zhenmode.evaluation import channel_dynamics, internal_wave, native_channel
from zhenmode.evaluation.native_channel import load


def inputs(tmp_path, c, a):
    cp, output, report = (tmp_path / name for name in ("contract.json", "output.npz", "report.json"))
    cp.write_text(json.dumps(c), encoding="utf-8")
    np.savez_compressed(output, **(a | {"metadata": np.array(json.dumps(a["metadata"]))}))
    return cp, output, report


def command(paths, route="internal-wave"):
    return ["evaluate", route, *(item for name, path in zip(
        ("contract", "output", "report"), paths, strict=True
    ) for item in ("--" + name, str(path)))]


@pytest.mark.parametrize("model", ("zhenmode", "mom6", "oceananigans"))
@pytest.mark.parametrize("control", ("main", "null"))
def test_offline_score_matches_api_and_preserves_input_bytes(tmp_path, monkeypatch, model, control):
    c = contract(model, control=control)
    paths = inputs(tmp_path, c, planted(c))
    before = [path.read_bytes() for path in paths[:2]]
    expected = internal_wave.score(c, load(paths[1]))
    expected["output_sha256"] = hashlib.sha256(before[1]).hexdigest()
    calls = []
    scorer = internal_wave.score

    def observed_score(c, a):
        calls.append(digest(c))
        return scorer(c, a)

    def forbidden(*args, **kwargs):
        pytest.fail("offline scoring must not start a benchmark or initialize a model")

    monkeypatch.setattr(internal_wave, "score", observed_score)
    monkeypatch.setattr("zhenmode.benchmarks.internal_wave.initial_native", forbidden)
    monkeypatch.setattr("zhenmode.execution.benchmark.main", forbidden)
    assert main(command(paths)) == 0
    assert calls == [digest(c)]
    assert json.loads(paths[2].read_text()) == expected
    assert [path.read_bytes() for path in paths[:2]] == before


def test_cli_reports_failed_screen_without_changing_score(tmp_path):
    c = contract("zhenmode", control="null")
    a = planted(c)
    a["T"][1:] += 1e-8
    paths = inputs(tmp_path, c, a)
    expected = internal_wave.score(c, load(paths[1]))
    assert "null_T" in expected["failed_metrics"]
    assert main(command(paths)) == 0
    actual = json.loads(paths[2].read_text())
    assert {key: actual[key] for key in expected} == expected
    assert not actual["engineering_screen_pass"]


def test_existing_report_is_not_overwritten(tmp_path, capsys):
    c = contract("zhenmode", control="null")
    paths = inputs(tmp_path, c, planted(c))
    paths[2].write_bytes(b"existing evidence\n")
    with pytest.raises(SystemExit) as error:
        main(command(paths))
    assert error.value.code == 2
    assert "File exists" in capsys.readouterr().err
    assert paths[2].read_bytes() == b"existing evidence\n"


@pytest.mark.parametrize("damage", (
    "contract", "contract-json", "contract-type", "development-type", "metadata-type",
    "missing", "nonfinite", "identity", "not-npz"
))
def test_invalid_input_has_clear_cli_error_and_no_report(tmp_path, capsys, damage):
    c = contract("zhenmode", control="null")
    a = planted(c)
    if damage == "contract":
        c["thresholds"]["error"] *= 2
    elif damage == "missing":
        a.pop("h")
    elif damage == "nonfinite":
        a["v"][1, 0, 0] = np.nan
    elif damage == "identity":
        a["metadata"]["contract_sha256"] = "0" * 64
    elif damage == "development-type":
        c = development_contract(control="null")
        c["development"] = False
    elif damage == "metadata-type":
        a["metadata"] = 0
    paths = inputs(tmp_path, c, a)
    if damage == "contract-json":
        paths[0].write_text("{", encoding="utf-8")
    elif damage == "contract-type":
        paths[0].write_text("[]", encoding="utf-8")
    elif damage == "not-npz":
        paths[1].write_bytes(b"not an NPZ")
    with pytest.raises(SystemExit) as error:
        main(command(paths))
    assert error.value.code == 2
    stderr = capsys.readouterr().err
    assert "zhenmode evaluate internal-wave: error:" in stderr
    assert "Traceback" not in stderr
    assert not paths[2].exists()


@pytest.mark.parametrize("c", (contract("zhenmode"), development_contract()))
def test_channel_rejects_internal_wave_before_using_thermal_reference(tmp_path, capsys, c):
    with pytest.raises(ValueError, match="zhenmode evaluate internal-wave"):
        channel_dynamics.reference_surface(c, [0.0], [0.0])
    with pytest.raises(ValueError, match="zhenmode.evaluation.internal_wave.score"):
        channel_dynamics.score(c, {})
    paths = (tmp_path / "contract.json", tmp_path / "missing.npz", tmp_path / "report.json")
    paths[0].write_text(json.dumps(c), encoding="utf-8")
    with pytest.raises(SystemExit) as error:
        main(command(paths, "channel"))
    assert error.value.code == 2
    assert "zhenmode evaluate internal-wave" in capsys.readouterr().err
    assert not paths[2].exists()


def test_thermal_development_keeps_channel_reference_routing():
    c = development_contract(profile="coarse", problem="thermal-wind")
    from zhenmode.benchmarks.channel_dynamics import validate_contract
    from zhenmode.benchmarks.configuration_scan import combination_contract

    validate_contract(c)
    y = np.array([0.0, c["Ly_m"] / 2, c["Ly_m"]])
    expected = channel_dynamics.reference_surface(combination_contract(9), [0.0, 12.5], y)
    np.testing.assert_array_equal(channel_dynamics.reference_surface(c, [0.0, 12.5], y),
                                  expected)


def test_development_internal_wave_uses_dedicated_cli(tmp_path):
    c = development_contract(control="null")
    paths = inputs(tmp_path, c, planted(c))
    expected = internal_wave.score(c, load(paths[1]))
    assert main(command(paths)) == 0
    report = json.loads(paths[2].read_text())
    assert {key: report[key] for key in expected} == expected


@pytest.mark.parametrize("arguments", (["evaluate", "--help"], ["evaluate", "internal-wave", "--help"]))
def test_help_discovers_offline_route(capsys, arguments):
    with pytest.raises(SystemExit) as error:
        main(arguments)
    assert error.value.code == 0
    help_text = capsys.readouterr().out
    assert "internal-wave" in help_text
    if "internal-wave" in arguments:
        for flag in ("--contract", "--output", "--report"):
            assert flag in help_text


@pytest.mark.parametrize("phase", ("before-decode", "during-score"))
def test_receipt_hash_and_scores_bind_one_snapshot_when_file_changes(tmp_path, monkeypatch, phase):
    c = contract("zhenmode", control="null")
    a = planted(c)
    paths = inputs(tmp_path, c, a)
    original = paths[1].read_bytes()
    expected = internal_wave.score(c, a)
    expected["output_sha256"] = hashlib.sha256(original).hexdigest()
    if phase == "before-decode":
        reader = native_channel.np.load

        def replace_before_decode(source, **kwargs):
            paths[1].write_bytes(b"replacement after the snapshot was read")
            return reader(source, **kwargs)

        monkeypatch.setattr(native_channel.np, "load", replace_before_decode)
    else:
        scorer = internal_wave.score

        def replace_during_score(c, a):
            paths[1].write_bytes(b"replacement during scoring")
            return scorer(c, a)

        monkeypatch.setattr(internal_wave, "score", replace_during_score)
    assert main(command(paths)) == 0
    assert paths[1].read_bytes() != original
    assert json.loads(paths[2].read_text()) == expected


@pytest.mark.parametrize("damage", ("encrypted", "unsupported-compression", "eof"))
def test_archive_reader_errors_are_cli_errors_without_report(tmp_path, capsys, monkeypatch, damage):
    c = contract("zhenmode", control="null")
    paths = inputs(tmp_path, c, planted(c))
    if damage == "eof":
        def incomplete_archive(*args, **kwargs):
            raise EOFError("incomplete archive member")

        monkeypatch.setattr(native_channel.np, "load", incomplete_archive)
    else:
        raw = bytearray(paths[1].read_bytes())
        with zipfile.ZipFile(paths[1]) as archive:
            offsets = [info.header_offset for info in archive.infolist()]
        for offset in offsets:
            if damage == "encrypted":
                flags = struct.unpack_from("<H", raw, offset + 6)[0]
                struct.pack_into("<H", raw, offset + 6, flags | 1)
            else:
                struct.pack_into("<H", raw, offset + 8, 99)
        offset = raw.find(b"PK\x01\x02")
        while offset != -1:
            if damage == "encrypted":
                flags = struct.unpack_from("<H", raw, offset + 8)[0]
                struct.pack_into("<H", raw, offset + 8, flags | 1)
            else:
                struct.pack_into("<H", raw, offset + 10, 99)
            offset = raw.find(b"PK\x01\x02", offset + 4)
        paths[1].write_bytes(raw)
    with pytest.raises(SystemExit) as error:
        main(command(paths))
    assert error.value.code == 2
    assert "zhenmode evaluate internal-wave: error:" in capsys.readouterr().err
    assert not paths[2].exists()


@pytest.mark.parametrize("c", ([], None, True, 1, "contract"))
def test_channel_non_object_contract_is_rejected_before_output_io(tmp_path, capsys, c):
    with pytest.raises(ValueError, match="JSON object"):
        channel_dynamics.score(c, {})
    with pytest.raises(ValueError, match="JSON object"):
        channel_dynamics.reference_surface(c, [0.0], [0.0])
    paths = (tmp_path / "contract.json", tmp_path / "missing.npz", tmp_path / "report.json")
    paths[0].write_text(json.dumps(c), encoding="utf-8")
    with pytest.raises(SystemExit) as error:
        main(command(paths, "channel"))
    assert error.value.code == 2
    assert "JSON object" in capsys.readouterr().err
    assert not paths[2].exists()


@pytest.mark.parametrize("field", ("T", "u"))
def test_finite_trajectory_overflow_emits_complete_failed_json_report(tmp_path, field):
    c = contract("zhenmode", control="null")
    a = planted(c)
    a[field][1:] = 1e200
    assert np.isfinite(a[field]).all()
    paths = inputs(tmp_path, c, a)
    with np.errstate(over="ignore", invalid="ignore"):
        expected = internal_wave.score(c, a)
        assert main(command(paths)) == 0
    metric = "T_error_C" if field == "T" else "u_error"
    assert np.isinf(expected["metrics"][metric])

    def reject_nonstandard(value):
        pytest.fail("nonstandard JSON numeric constant: " + value)

    report = json.loads(paths[2].read_text(), parse_constant=reject_nonstandard)
    assert report["metrics"][metric] is None
    assert report["non_finite_derived_values"]["/metrics/" + metric] == "Infinity"
    assert set(expected["failed_metrics"]) <= set(report["failed_metrics"])
    assert "non_finite_derived_values" in report["failed_metrics"]
    assert not report["engineering_screen_pass"]
    assert report["output_sha256"] == hashlib.sha256(paths[1].read_bytes()).hexdigest()


def test_json_serialization_failure_does_not_create_partial_report(tmp_path, capsys, monkeypatch):
    c = contract("zhenmode", control="null")
    paths = inputs(tmp_path, c, planted(c))
    scorer = internal_wave.score

    def unsupported_result(c, a):
        return scorer(c, a) | {"unsupported": object()}

    monkeypatch.setattr(internal_wave, "score", unsupported_result)
    with pytest.raises(SystemExit) as error:
        main(command(paths))
    assert error.value.code == 2
    assert "not JSON serializable" in capsys.readouterr().err
    assert not paths[2].exists()


def test_finite_signed_overflow_inventory_is_explicitly_failed(tmp_path):
    c = contract("zhenmode", control="null")
    a = planted(c)
    a["T"][1:] = 1e308
    a["T"][1:, :, c["nz"] // 2:] *= -1
    assert np.isfinite(a["T"]).all()
    paths = inputs(tmp_path, c, a)
    with np.errstate(over="ignore", invalid="ignore"):
        assert main(command(paths)) == 0
    report = json.loads(paths[2].read_text())
    assert report["metrics"]["T_inventory"] is None
    assert report["non_finite_derived_values"]["/metrics/T_inventory"] == "NaN"
    assert not report["engineering_screen_pass"]
    assert "non_finite_derived_values" in report["failed_metrics"]
