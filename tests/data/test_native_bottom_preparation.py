"""Scientific support, preservation, regional precedence and lineage controls."""

import json

import netCDF4
import numpy as np
import pytest

from scripts.check_native_bottom_preparation import verify
from tests.support.data.native_initialization import native_inputs as _inputs
from zhenmode.execution.benchmark import main
from zhenmode.preparation.bottom import complete_native_bottom
from zhenmode.preparation.native_initial import prepare_native_initialization
from zhenmode.provenance.sources import sha256_file


def _prepared(tmp_path, regional=False):
    source, geometry, nodes, _, area = _inputs(tmp_path, unanchored=True)
    prepared = tmp_path / "native"
    report = prepare_native_initialization(source, geometry, nodes, prepared)
    assert report["levels"][1]["unresolved_nodes"] == 1
    evidence = tmp_path / "regional-audit.json"
    evidence.write_text(
        json.dumps(
            {
                "data_kind": "manufactured",
                "reason": "known fixture profile or deliberately no extra profile",
            }
        )
    )
    review = tmp_path / "review.json"
    column = {
        "native_flat_index": 2,
        "decision": "no_qualified_regional_profile",
        "reason": "manufactured absent regional support",
        "evidence_index": 0,
        "regional_points": [],
    }
    if regional:
        column.update(
            decision="regional_profile_then_fallback",
            regional_points=[
                {
                    "node_index": 1,
                    "ptemp": 8.0,
                    "sr": 36.0,
                    "source_trace": "manufactured known regional point",
                }
            ],
        )
    review.write_text(
        json.dumps(
            {
                "policy": "regional_profiles_then_same_column_zero_gradient_v1",
                "native_receipt_sha256": sha256_file(prepared / "native_initialization.json"),
                "evidence": [
                    {
                        "path": evidence.name,
                        "bytes": evidence.stat().st_size,
                        "sha256": sha256_file(evidence),
                    }
                ],
                "columns": [column],
            }
        )
    )
    return prepared, review, area


@pytest.mark.parametrize("regional", [False, True])
def test_reviewed_bottom_completes_only_missing_point_and_preserves_original_bits(
    tmp_path, regional
):
    prepared, review, area = _prepared(tmp_path, regional)
    out = tmp_path / "completed"
    report = complete_native_bottom(prepared, review, out)
    assert report["status"] == "native_points_completed_with_bottom_assumptions"
    assert report["complete_wet_support"] and report["bottom_assigned_nodes"] == 1
    assert report["regional_profile_nodes"] == int(regional)
    assert report["same_column_extension_nodes"] == int(not regional)
    assert not report["execution_ready"] and not report["mom_layer_initialization_ready"]
    assert not report["sensitivity"]["qualifies_dynamics"]
    assert report["sensitivity"]["no_second_native_anchor_nodes"] == 1
    assert report["water_inventory"]["bottom_assigned_volume_m3"] == area[1, 0] * 500
    with (
        netCDF4.Dataset(prepared / "native_point_fields.nc") as before,
        netCDF4.Dataset(out / "native_point_fields.nc") as after,
    ):
        for name in ("ptemp", "ct", "sr", "salt"):
            original, completed = before[name][:], after[name][:]
            known = ~np.ma.getmaskarray(original)
            np.testing.assert_array_equal(original.data[known], completed.data[known])
            if regional and name in ("ptemp", "sr"):
                assert completed[0, 1, 0, 1] == (8.0 if name == "ptemp" else 36.0)
            assert not np.ma.is_masked(completed[0, 1, 0, 1])
        if not regional:
            assert after["ptemp"][0, 1, 0, 1] == before["ptemp"][0, 0, 0, 1]
            assert after["sr"][0, 1, 0, 1] == before["sr"][0, 0, 0, 1]
            assert after["bottom_donor_native_level_index"][1, 0, 1] == 0
            assert after["bottom_extension_distance_m"][1, 0, 1] == 500
            assert after["bottom_donor_nearest_original_anchor_flat_index"][1, 0, 1] == 2
        assert after["pre_bottom_unresolved_support"][1, 0, 1] == 1
        assert after["unresolved_support"][:].sum() == 0
        np.testing.assert_array_equal(before["wet_node_mask"][:], after["wet_node_mask"][:])
    for name, row in report["outputs"].items():
        assert sha256_file(out / name) == row["sha256"]
    with pytest.raises(FileExistsError):
        complete_native_bottom(prepared, review, out)


@pytest.mark.parametrize(
    "defect", ["missing", "duplicate", "known", "bad_domain", "wrong_parent", "missing_evidence"]
)
def test_review_cannot_omit_a_column_replace_known_water_or_hide_source_identity(tmp_path, defect):
    prepared, review, _ = _prepared(tmp_path, regional=True)
    policy = json.loads(review.read_text())
    if defect == "missing":
        policy["columns"] = []
    if defect == "duplicate":
        policy["columns"] *= 2
    if defect == "known":
        policy["columns"][0]["regional_points"][0]["node_index"] = 0
    if defect == "bad_domain":
        policy["columns"][0]["regional_points"][0]["sr"] = -1
    if defect == "wrong_parent":
        policy["native_receipt_sha256"] = "0" * 64
    if defect == "missing_evidence":
        policy["evidence"] = []
    review.write_text(json.dumps(policy))
    with pytest.raises(ValueError):
        complete_native_bottom(prepared, review, tmp_path / "bad")


def test_late_review_evidence_mutation_never_publishes_success(tmp_path, monkeypatch):
    import zhenmode.preparation.bottom as implementation

    prepared, review, _ = _prepared(tmp_path)
    original = implementation.shutil.copyfile

    def changed(*args, **kwargs):
        result = original(*args, **kwargs)
        (tmp_path / "regional-audit.json").write_text("late evidence mutation")
        return result

    monkeypatch.setattr(implementation.shutil, "copyfile", changed)
    out = tmp_path / "bad"
    with pytest.raises(ValueError, match="changed during"):
        complete_native_bottom(prepared, review, out)
    report = json.loads((out / "native_initialization.json").read_text())
    assert report["status"] == "failed" and not report["execution_ready"]


def test_unanchored_surface_stays_blocked_in_public_cli(tmp_path, capsys):
    prepared, review, _ = _prepared(tmp_path)
    receipt = prepared / "native_initialization.json"
    metadata = json.loads(receipt.read_text())
    source = prepared / "native_point_fields.nc"
    with netCDF4.Dataset(source, "a") as data:
        for name in ("ptemp", "ct", "sr", "salt"):
            data[name][0, 0, 0, 1] = np.ma.masked
        data["unresolved_support"][0, 0, 1] = 1
    metadata["output"].update(bytes=source.stat().st_size, sha256=sha256_file(source))
    receipt.write_text(json.dumps(metadata))
    policy = json.loads(review.read_text())
    policy["native_receipt_sha256"] = sha256_file(receipt)
    policy["columns"] = []
    review.write_text(json.dumps(policy))
    code = main(
        [
            "complete-native-bottom",
            "--native-prepared",
            str(prepared),
            "--review-file",
            str(review),
            "--output",
            str(tmp_path / "blocked"),
        ]
    )
    assert code == 3
    result = json.loads((tmp_path / "blocked/native_initialization.json").read_text())
    assert result["status"] == "blocked_non_bottom_native_support"
    assert result["unresolved_nodes"] == 2 and not result["complete_wet_support"]


@pytest.mark.parametrize("anchor_pt, expected_supported", [(9.0, 1), (-2.0, 0)])
def test_arithmetic_sensitivity_has_two_anchors_and_refuses_out_of_domain_without_clipping(
    tmp_path, anchor_pt, expected_supported
):
    from zhenmode.model.solver.physics.teos10 import conservative_from_potential

    # Build a declared manufactured column with PT=10 at 0m and PT=9 or -2
    # at 250m. The 500m linear continuation is independently 8 or -14 degC.
    _, review, _ = _prepared(tmp_path)
    # Rebuild in a separate fixture directory; preserve the first fixture's
    # review evidence while rebinding its explicit native receipt.
    second = tmp_path / "second"
    second.mkdir()
    source, geometry, nodes, _, _ = _inputs(second, unanchored=True)
    nodes.write_text(json.dumps({"z_nodes_m": [0.0, -250.0, -500.0, -4000.0, -6000.0, -7000.0]}))
    new_prepared = second / "native"
    metadata = prepare_native_initialization(source, geometry, nodes, new_prepared)
    field = new_prepared / "native_point_fields.nc"
    with netCDF4.Dataset(field, "a") as data:
        sr = float(data["sr"][0, 0, 0, 1])
        values = {
            "ptemp": anchor_pt,
            "sr": sr,
            "salt": sr * 35 / 35.16504,
            "ct": float(np.asarray(conservative_from_potential(sr, anchor_pt))),
        }
        for name, value in values.items():
            data[name][0, 1, 0, 1] = value
        data["unresolved_support"][1, 0, 1] = 0
        data["nearest_original_anchor_flat_index"][1, 0, 1] = 2
        data["nearest_anchor_path_distance_m"][1, 0, 1] = 0
    metadata["output"].update(bytes=field.stat().st_size, sha256=sha256_file(field))
    (new_prepared / "native_initialization.json").write_text(json.dumps(metadata))
    policy = json.loads(review.read_text())
    policy["native_receipt_sha256"] = sha256_file(new_prepared / "native_initialization.json")
    review.write_text(json.dumps(policy))
    report = complete_native_bottom(new_prepared, review, tmp_path / "complete")
    assert report["sensitivity"]["candidate_supported_nodes"] == expected_supported
    assert report["sensitivity"]["candidate_outside_numeric_range_nodes"] == 1 - expected_supported
    with np.load(tmp_path / "complete/bottom-sensitivity.npz") as data:
        assert data["donor_native_level_index"][0] == 1
        assert data["second_donor_native_level_index"][0] == 0
        assert data["selected_pt"][0] == anchor_pt
        if expected_supported:
            assert data["linear_pt"][0] == pytest.approx(8.0, abs=1e-12)
        else:
            assert np.isnan(data["linear_pt"][0])


@pytest.mark.parametrize("tracer", ["ptemp", "sr"])
@pytest.mark.parametrize("literal", ["NaN", "Infinity", "-Infinity", "1e309"])
def test_regional_nonfinite_review_is_refused_before_output_publication(tmp_path, tracer, literal):
    prepared, review, _ = _prepared(tmp_path, regional=True)
    policy = json.loads(review.read_text())
    policy["columns"][0]["regional_points"][0][tracer] = json.loads(literal)
    review.write_text(json.dumps(policy))
    out = tmp_path / "bad"
    with pytest.raises(ValueError, match="finite real"):
        complete_native_bottom(prepared, review, out)
    assert not out.exists()
    # Correcting the input should permit a retry at the same unused target.
    policy["columns"][0]["regional_points"][0][tracer] = 8.0 if tracer == "ptemp" else 36.0
    review.write_text(json.dumps(policy))
    assert complete_native_bottom(prepared, review, out)["complete_wet_support"]


def test_independent_checker_accepts_complete_parent_with_no_bottom_assignments(tmp_path):
    source, geometry, nodes, _, _ = _inputs(tmp_path)
    prepared = tmp_path / "native"
    parent = prepare_native_initialization(source, geometry, nodes, prepared)
    assert parent["complete_wet_support"]
    evidence = tmp_path / "regional-audit.json"
    evidence.write_text(json.dumps({"data_kind": "manufactured", "bottom_gaps": 0}))
    review = tmp_path / "review.json"
    review.write_text(
        json.dumps(
            {
                "policy": "regional_profiles_then_same_column_zero_gradient_v1",
                "native_receipt_sha256": sha256_file(prepared / "native_initialization.json"),
                "evidence": [
                    {
                        "path": evidence.name,
                        "bytes": evidence.stat().st_size,
                        "sha256": sha256_file(evidence),
                    }
                ],
                "columns": [],
            }
        )
    )
    out = tmp_path / "complete"
    completed = complete_native_bottom(prepared, review, out)
    assert completed["bottom_assigned_nodes"] == 0 and completed["complete_wet_support"]
    checked = tmp_path / "checked.json"
    verify(prepared, out, checked)
    result = json.loads(checked.read_text())
    assert result["assigned_nodes"] == 0
    assert result["negative_controls"] == ["changed_resolved_value_refused"]


def test_independent_checker_refuses_changed_parent_receipt_with_unchanged_field_bytes(tmp_path):
    prepared, review, _ = _prepared(tmp_path)
    out = tmp_path / "complete"
    complete_native_bottom(prepared, review, out)
    source_sha = sha256_file(prepared / "native_point_fields.nc")
    receipt = prepared / "native_initialization.json"
    changed = json.loads(receipt.read_text())
    changed["initial_datetime"] = "1959-01-01T00:00:00"
    receipt.write_text(json.dumps(changed))
    assert sha256_file(prepared / "native_point_fields.nc") == source_sha
    with pytest.raises(AssertionError, match="parent receipt"):
        verify(prepared, out, tmp_path / "false-success.json")
    assert not (tmp_path / "false-success.json").exists()
