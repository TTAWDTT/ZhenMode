"""Actual factory consumer, bound inputs, inactive values and no clipping."""

import json

import netCDF4
import numpy as np
import pytest

from tests.support.data.fd_native import prepared_native_inputs as _prepared
from zhenmode.execution.benchmark import main
from zhenmode.model.config import G_EARTH, RHO_0, PhysicsConfig
from zhenmode.model.solver.factory import make_solver_global
from zhenmode.preparation.fd import load_fd_native_inputs, prepare_fd_native_inputs
from zhenmode.provenance.sources import sha256_file


def test_prepared_data_roundtrips_to_actual_factory_with_exact_wet_values_and_capacity(tmp_path):
    native, policy = _prepared(tmp_path)
    out = tmp_path / "fd"
    code = main(
        [
            "prepare-fd-initial",
            "--native-prepared",
            str(native),
            "--policy",
            str(policy),
            "--output",
            str(out),
        ]
    )
    assert code == 0
    report = json.loads((out / "fd_initialization.json").read_text())
    assert report["status"] == "fd_native_inputs_prepared" and report["data_kind"] == "manufactured"
    assert report["completed_timesteps"] == 0 and not report["execution_ready"]
    assert not report["initialization_entry_verified"]
    grid, ct, sr, pressure = load_fd_native_inputs(out)
    wet = grid.wet_mask_3d > 0
    with netCDF4.Dataset(native / "native_point_fields.nc") as original:
        for name, actual in (("ct", ct), ("sr", sr)):
            expected = np.ma.asarray(original[name][0]).filled(np.nan).transpose(2, 1, 0)
            np.testing.assert_array_equal(
                actual[wet].view(np.uint64), expected[wet].view(np.uint64)
            )
    np.testing.assert_array_equal(ct[~wet], 15.0)
    np.testing.assert_array_equal(sr[~wet], 35.0)
    np.testing.assert_array_equal(pressure[0, 0], -grid.z * RHO_0 * G_EARTH * 1e-4)
    assert report["parent_native_receipt_sha256"] == sha256_file(
        native / "native_initialization.json"
    )
    physics = PhysicsConfig(thermodynamics="teos10_reference")
    _, initialize, _, params, _ = make_solver_global(
        grid,
        physics,
        dt=1.0,
        dt_bt=1.0,
        mode_split=True,
        column_geometry="fixed_partial_v1",
        conservative_kv=True,
        localize_conv=True,
        return_params=True,
        eos_pressure_dbar=pressure,
        polar_cap_rows=0,
        polar_cap_taper=0,
    )
    state = initialize(T_init=ct, S_init=sr)
    np.testing.assert_array_equal(np.asarray(state.T)[wet], ct[wet])
    np.testing.assert_array_equal(np.asarray(state.S)[wet], sr[wet])
    with np.load(native / "native_geometry.npz") as original:
        np.testing.assert_allclose(grid.dx_2d * grid.dy, original["area"], rtol=1e-15)
        np.testing.assert_array_equal(np.asarray(params.dz_node) * wet, original["thickness_m"])
        assert float(
            np.sum(grid.dx_2d[..., None] * grid.dy * np.asarray(params.dz_node) * wet)
        ) == pytest.approx(float(np.sum(original["area"] * original["depth"])), rel=1e-14)
    with pytest.raises(FileExistsError):
        prepare_fd_native_inputs(native, policy, out)


@pytest.mark.parametrize(
    "defect", ["incomplete", "deep_pressure", "wet_nan", "units", "changed_geometry"]
)
def test_unqualified_or_corrupt_input_is_refused_before_publishing(tmp_path, defect):
    native, policy = _prepared(
        tmp_path,
        complete=defect != "incomplete",
        deepest=9000.0 if defect == "deep_pressure" else 7000.0,
    )
    receipt = native / "native_initialization.json"
    parent = json.loads(receipt.read_text())
    if defect in ("wet_nan", "units"):
        source = native / "native_point_fields.nc"
        with netCDF4.Dataset(source, "a") as data:
            if defect == "wet_nan":
                data["ct"][0, 0, 0, 0] = np.ma.masked
            else:
                data["sr"].units = "1"
        parent["output"].update(sha256=sha256_file(source), bytes=source.stat().st_size)
        receipt.write_text(json.dumps(parent))
    if defect == "changed_geometry":
        with (native / "native_geometry.npz").open("ab") as stream:
            stream.write(b"changed")
    with pytest.raises(ValueError):
        prepare_fd_native_inputs(native, policy, tmp_path / "bad")
    assert not (tmp_path / "bad").exists()


@pytest.mark.parametrize(
    "defect", ["wrong_pressure", "broadcast_tracer", "wrong_capacity", "wrong_inactive_values"]
)
def test_loader_checks_rebound_bundle_definition_not_only_its_digest(tmp_path, defect):
    native, policy = _prepared(tmp_path)
    out = tmp_path / "fd"
    prepare_fd_native_inputs(native, policy, out)
    path = out / "fd-native-inputs.npz"
    with np.load(path) as data:
        arrays = {k: data[k].copy() for k in data.files}
    if defect == "wrong_pressure":
        arrays["eos_pressure_dbar"][..., 1] += 1
    if defect == "broadcast_tracer":
        arrays["ct"] = np.array([10.0])
    if defect == "wrong_capacity":
        arrays["thickness_m"][0, 0, 0] += 1
    if defect == "wrong_inactive_values":
        arrays["sr"][~arrays["wet_node_mask"].astype(bool)] = 34.0
    np.savez_compressed(path, **arrays)
    receipt = out / "fd_initialization.json"
    metadata = json.loads(receipt.read_text())
    metadata["output"].update(sha256=sha256_file(path), bytes=path.stat().st_size)
    receipt.write_text(json.dumps(metadata))
    with pytest.raises(ValueError):
        load_fd_native_inputs(out)


def test_late_policy_change_never_reports_completed_inputs(tmp_path, monkeypatch):
    import zhenmode.preparation.fd as implementation

    native, policy = _prepared(tmp_path)
    original = implementation.np.savez_compressed

    def changed(*args, **kwargs):
        result = original(*args, **kwargs)
        policy.write_text("late change")
        return result

    monkeypatch.setattr(implementation.np, "savez_compressed", changed)
    out = tmp_path / "bad"
    with pytest.raises(ValueError, match="changed during"):
        prepare_fd_native_inputs(native, policy, out)
    report = json.loads((out / "fd_initialization.json").read_text())
    assert report["status"] == "failed" and not report["execution_ready"]


@pytest.mark.parametrize("perturbation", ["one_ulp", "substantive", "shape", "nonzero_at_zero"])
def test_rebuilt_float_metrics_allow_rounding_but_refuse_definition_changes(tmp_path, perturbation):
    native, policy = _prepared(tmp_path)
    out = tmp_path / "fd"
    prepare_fd_native_inputs(native, policy, out)
    path = out / "fd-native-inputs.npz"
    with np.load(path) as data:
        arrays = {k: data[k].copy() for k in data.files}
    if perturbation == "one_ulp":
        for name in ("cos_lat", "f", "dx_2d", "dy", "eos_pressure_dbar"):
            old = arrays[name]
            arrays[name] = np.where(old != 0, np.nextafter(old, np.inf), old)
    if perturbation == "substantive":
        arrays["f"] *= 1.000001
    if perturbation == "shape":
        arrays["f"] = arrays["f"][0]
    if perturbation == "nonzero_at_zero":
        arrays["eos_pressure_dbar"][..., 0] = 1e-20
    np.savez_compressed(path, **arrays)
    receipt = out / "fd_initialization.json"
    metadata = json.loads(receipt.read_text())
    metadata["output"].update(sha256=sha256_file(path), bytes=path.stat().st_size)
    receipt.write_text(json.dumps(metadata))
    if perturbation == "one_ulp":
        grid, _, _, pressure = load_fd_native_inputs(out)
        np.testing.assert_array_equal(grid.f, arrays["f"])
        np.testing.assert_array_equal(pressure, arrays["eos_pressure_dbar"])
    else:
        with pytest.raises(ValueError, match="metrics/reference pressure"):
            load_fd_native_inputs(out)
