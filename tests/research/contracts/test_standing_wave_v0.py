"""Synthetic oracle/negative controls test the scorer, never model performance."""

from tests.support.paths import REPOSITORY_ROOT

import copy
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location(
    "wave_score", REPOSITORY_ROOT / "research/experiments/standing_wave_v0/score.py"
)
w = importlib.util.module_from_spec(spec)
spec.loader.exec_module(w)


def fixture(case="coarse", half=False, sampling="node"):
    c = w.contract(case, half)
    nx, ny, nz = c["nx"], c["ny"], c["nz"]
    dx, dy = c["Lx_m"] / nx, c["Ly_m"] / ny
    x, y = np.meshgrid((np.arange(nx) + 0.5) * dx, (np.arange(ny) + 0.5) * dy, indexing="ij")
    x, y = x.ravel(), y.ravel()
    t = np.arange(0, 32001, 1000, dtype=float)
    eta, ubar = w.exact(c, t, x, sampling, dx)
    shape = (len(t), nx * ny, nz)
    h = np.broadcast_to(c["initial_h_m"], shape).copy()
    h[:, :, 0] += eta
    vol = h * dx * dy
    m = dict(
        schema=w.SCHEMA,
        contract_sha256=w.digest(c),
        model="ocean-solver",
        source_sha="a" * 40,
        executable_sha256="b" * 64,
        input_sha256="c" * 64,
        config_sha256="d" * 64,
        coordinate_system="cartesian_m",
        units=dict(
            time="s", x="m", eta="m", h="m", u="m/s", volume="m3", area="m2", T="degC", S="psu"
        ),
        snapshot_kind="instantaneous",
        sampling_eta=sampling,
        sampling_u=sampling,
        velocity_layout="collocated",
        dt_s=c["dt"],
        steps=int(32000 / c["dt"]),
        full_dynamics=True,
        zero_processes=c["explicitly_zero"].copy(),
        recorded_numerics={
            k: "synthetic oracle only"
            for k in ["time_scheme", "transport", "filters", "vertical_coordinate", "substeps"]
        },
        initialization_s=1.0,
        integration_s=2.0,
        total_wall_s=3.0,
        aggregate_peak_bytes=1024,
        cpu=1,
        ranks=1,
        trajectory=dict(kind="continuous", processes=1),
        g=9.81,
        rho0=1025.0,
        H=100.0,
        Lx=c["Lx_m"],
        Ly=c["Ly_m"],
        f=0.0,
        eos=c["eos"],
        boundary=c["boundary"],
    )
    m["recorded_numerics"]["resolved_options"] = c["ocean_options"].copy()
    a = dict(
        metadata=m,
        time=t,
        x_eta=x,
        y_eta=y,
        area=np.full(nx * ny, dx * dy),
        eta=eta,
        h=h,
        T=np.full(shape, 15.0),
        S=np.full(shape, 35.0),
        x_u=x.copy(),
        y_u=y.copy(),
        width_u=np.full(x.size, 0.0 if sampling == "node" else dx),
        u=np.broadcast_to(ubar[..., None], shape).copy(),
        volume_u=vol.copy(),
        x_v=x.copy(),
        y_v=y.copy(),
        v=np.zeros(shape),
        volume_v=vol.copy(),
    )
    return c, a


@pytest.mark.parametrize("case", ["coarse", "medium", "fine"])
@pytest.mark.parametrize("sampling", ["node", "cell_mean"])
def test_exact_oracle_passes_without_qualification(case, sampling):
    c, a = fixture(case, sampling=sampling)
    if sampling == "cell_mean":
        a = cgrid(a, c)
        a["metadata"]["model"] = "MOM6"
        a["metadata"]["source_sha"] = w.MOM6
        a["metadata"]["recorded_numerics"]["resolved_options"] = c["mom_time_options"]
    r = w.score(c, a)
    assert r["engineering_screen_pass"]
    assert not r["industrial_qualified"] and not r["equal_error_speedup_qualified"]
    assert (
        max(
            r["metrics"][key]
            for key in [
                "eta_error",
                "u_error",
                "phase_rad",
                "amplitude",
                "volume",
                "T",
                "S",
                "v_over_U",
            ]
        )
        < 2e-12
    )
    assert r["energy_units"] == "J"
    factor = 1.0 if sampling == "node" else np.sinc(1 / c["nx"])
    expected = (
        c["rho0"] * c["gravity"] * c["amplitude_m"] ** 2 * c["Lx_m"] * c["Ly_m"] / 4 * factor**2
    )
    assert r["initial_energy_J"] == pytest.approx(expected, rel=1e-14)


def test_cell_mean_is_integral_not_point():
    c = w.contract()
    node, _ = w.exact(c, [0.0], [0.0])
    mean, _ = w.exact(c, [0.0], [0.0], "cell_mean", c["Lx_m"] / 64)
    assert mean[0, 0] / node[0, 0] == pytest.approx(np.sinc(1 / 64))
    assert mean[0, 0] != node[0, 0]


@pytest.mark.parametrize(
    "field,value",
    [
        ("dt_s", 99.0),
        ("steps", 319),
        ("coordinate_system", "spherical"),
        ("rho0", 1035.0),
        ("g", 9.8),
        ("f", 0.0001),
        ("full_dynamics", False),
        ("cpu", 2),
        ("ranks", 2),
        ("source_sha", "unknown"),
        ("contract_sha256", "0" * 64),
        ("sampling_eta", "unknown"),
        ("sampling_u", "unknown"),
        ("aggregate_peak_bytes", 0),
        ("aggregate_peak_bytes", 4 * 1024**3 + 1),
        ("total_wall_s", 601.0),
        ("integration_s", 0.0),
        ("initialization_s", float("nan")),
        ("trajectory", dict(kind="restart", processes=1)),
        ("recorded_numerics", {}),
    ],
)
def test_wrong_metadata_rejected(field, value):
    c, a = fixture()
    a["metadata"][field] = value
    with pytest.raises(ValueError):
        w.score(c, a)


@pytest.mark.parametrize(
    "damage",
    [
        "missing",
        "extra",
        "time",
        "incomplete",
        "nan",
        "float32",
        "area",
        "grid",
        "h",
        "shape",
        "volume",
        "width",
        "initial",
    ],
)
def test_wrong_outputs_rejected(damage):
    c, a = fixture()
    if damage == "missing":
        del a["v"]
    elif damage == "extra":
        a["unknown"] = np.array([0.0])
    elif damage == "time":
        a["time"][1] += 1
    elif damage == "incomplete":
        a["eta"] = a["eta"][:-1]
    elif damage == "nan":
        a["S"][2, 0, 0] = np.nan
    elif damage == "float32":
        a["u"] = a["u"].astype(np.float32)
    elif damage == "area":
        a["area"][0] *= 2
    elif damage == "grid":
        a["x_eta"][0] += 100
    elif damage == "h":
        a["h"][1, 0, 0] = -1
    elif damage == "shape":
        a["T"] = a["T"][:, :, :1]
    elif damage == "volume":
        a["volume_u"] *= 2
    elif damage == "width":
        a["width_u"][0] = 1
    elif damage == "initial":
        a["u"][0, 0, 0] = 0.01
    with pytest.raises(ValueError):
        w.score(c, a)


@pytest.mark.parametrize(
    "damage,metric",
    [("tracer", "T"), ("velocity", "v_over_U"), ("phase", "phase_rad"), ("amplitude", "amplitude")],
)
def test_physical_failure_is_reported(damage, metric):
    c, a = fixture()
    if damage == "tracer":
        a["T"][1, 0, 0] += 1e-8
    elif damage == "velocity":
        a["v"][1, 0, 0] = 1e-5
    else:
        times = a["time"].copy()
        if damage == "phase":
            times[1:] += 1000
        eta, u = w.exact(c, times, a["x_eta"])
        if damage == "amplitude":
            eta[1:] *= 1.1
            u[1:] *= 1.1
        a["eta"] = eta
        a["h"][:, :, 0] = c["initial_h_m"][0] + eta
        a["volume_u"] = a["h"] * a["area"][None, :, None]
        a["volume_v"] = a["volume_u"].copy()
        a["u"][:] = u[..., None]
    r = w.score(c, a)
    assert not r["engineering_screen_pass"] and metric in r["failed_metrics"]


def restart(a):
    out = copy.deepcopy(a)
    out["metadata"]["trajectory"] = dict(
        kind="restart",
        processes=2,
        split_s=16000.0,
        checkpoint_sha256="e" * 64,
        full_native_state=True,
    )
    return out


def test_restart_and_full_state_difference():
    c, a = fixture("medium")
    b = restart(a)
    assert w.compare(c, a, b)["engineering_screen_pass"]
    b["S"][20, 0, 0] += 1e-8
    assert not w.compare(c, a, b)["engineering_screen_pass"]


def test_identical_continuous_output_is_not_restart():
    c, a = fixture("medium")
    with pytest.raises(ValueError, match="receipt"):
        w.compare(c, a, copy.deepcopy(a))


def test_half_amplitude_control():
    c, a = fixture("medium")
    _, b = fixture("medium", half=True)
    assert w.compare(c, a, b, half=True)["engineering_screen_pass"]
    b["u"][1:] *= 1.02
    assert not w.compare(c, a, b, half=True)["engineering_screen_pass"]


def test_npz_roundtrip(tmp_path):
    c, a = fixture()
    encoded = dict(a, metadata=np.array(json.dumps(a["metadata"])))
    path = tmp_path / "wave.npz"
    np.savez_compressed(path, **encoded)
    assert w.score(c, w.load(path))["engineering_screen_pass"]
    np.savez(path, metadata=np.array({"bad": 1}, dtype=object))
    with pytest.raises(ValueError):
        w.load(path)


def test_contract_tampering_rejected():
    c, a = fixture()
    c["thresholds"]["error"] = 100
    a["metadata"]["contract_sha256"] = w.digest(c)
    with pytest.raises(ValueError, match="frozen"):
        w.score(c, a)


def cgrid(a, c):
    nx, ny, nz = c["nx"], c["ny"], c["nz"]
    nt = len(a["time"])
    dx, dy = c["Lx_m"] / nx, c["Ly_m"] / ny
    h = a["h"].reshape(nt, nx, ny, nz)
    for q, xs, ys in [
        ("u", np.arange(nx) * dx, (np.arange(ny) + 0.5) * dy),
        ("v", (np.arange(nx) + 0.5) * dx, np.arange(ny + 1) * dy),
    ]:
        x, y = np.meshgrid(xs, ys, indexing="ij")
        a["x_" + q], a["y_" + q] = x.ravel(), y.ravel()
    a["width_u"] = np.zeros(nx * ny)
    _, u = w.exact(c, a["time"], a["x_u"])
    a["u"] = np.broadcast_to(u[..., None], (nt, nx * ny, nz)).copy()
    a["v"] = np.zeros((nt, nx * (ny + 1), nz))
    a["volume_u"] = (0.5 * (h + np.roll(h, 1, axis=1)) * dx * dy).reshape(nt, nx * ny, nz)
    a["volume_v"] = (
        np.concatenate(
            (0.5 * h[:, :, :1], 0.5 * (h[:, :, :-1] + h[:, :, 1:]), 0.5 * h[:, :, -1:]), axis=2
        )
        * dx
        * dy
    ).reshape(nt, nx * (ny + 1), nz)
    a["metadata"]["velocity_layout"] = "cgrid"
    a["metadata"]["sampling_u"] = "node"
    return a


def test_pinned_mom_native_cgrid_oracle():
    c, a = fixture(sampling="cell_mean")
    a = cgrid(a, c)
    a["metadata"]["model"] = "MOM6"
    a["metadata"]["source_sha"] = w.MOM6
    a["metadata"]["recorded_numerics"]["resolved_options"] = c["mom_time_options"]
    assert w.score(c, a)["engineering_screen_pass"]
    a["metadata"]["source_sha"] = "f" * 40
    with pytest.raises(ValueError, match="pinned"):
        w.score(c, a)


def test_redistributed_native_mass_cannot_forge_energy():
    c, a = fixture()
    a["volume_u"][:, 0] += 1
    a["volume_u"][:, 1] -= 1
    with pytest.raises(ValueError, match="quadrature"):
        w.score(c, a)


def test_wrong_velocity_positions_rejected():
    c, a = fixture()
    a["x_u"][:] = c["Lx_m"] / 4
    with pytest.raises(ValueError, match="positions"):
        w.score(c, a)


def test_changed_time_scheme_rejected():
    c, a = fixture()
    a["metadata"]["recorded_numerics"]["resolved_options"]["mode_split"] = True
    with pytest.raises(ValueError, match="frozen"):
        w.score(c, a)


def test_full_solver_preserves_tangential_wall_velocity():
    # One bounded full native step, no Python reference integrator.
    import sys
    from dataclasses import replace

    from config import PhysicsConfig
    from grid import GlobalOceanGrid
    from jax_solver_global import _step_impl, make_solver_global

    c = w.contract()
    nx, ny, nz = 4, 4, 4
    ones = np.ones((nx, ny))
    z = np.linspace(0, -100, nz)
    g = GlobalOceanGrid(
        lon=np.arange(nx, dtype=float),
        lat=np.arange(ny, dtype=float),
        dx_2d=ones * c["Lx_m"] / nx,
        dy=c["Ly_m"] / ny,
        cos_lat=np.ones(ny),
        f=ones * 0,
        z=z,
        dz=-np.diff(z),
        nz=nz,
        depth=ones * 100,
        wet_mask=ones,
        ocean_mask=ones.astype(bool),
        land_mask=ones.astype(bool) * False,
        wet_mask_3d=np.ones((nx, ny, nz)),
        nx=nx,
        ny=ny,
    )
    physics = replace(
        PhysicsConfig(),
        **dict.fromkeys(
            [
                "nu_h",
                "nu_v",
                "kappa_h",
                "kappa_v",
                "kappa_conv",
                "nu_bi",
                "kappa_bi",
                "kappa_gm",
                "kappa_redi",
                "r_bot",
                "cd",
            ],
            0.0,
        ),
    )
    _, init, _, params, _ = make_solver_global(
        g, physics, 100.0, return_params=True, **c["ocean_options"]
    )
    old = init()
    old = old._replace(u=old.u + 0.01)
    new = _step_impl(old, params)
    np.testing.assert_allclose(new.u[:, [0, -1], :], 0.01, rtol=0, atol=1e-14)
    np.testing.assert_array_equal(new.v, 0.0)


def test_cli_freeze_score_receipt(tmp_path, monkeypatch):
    import hashlib
    import sys

    cp, output, report = [
        tmp_path / name for name in ["contract.json", "output.npz", "report.json"]
    ]
    monkeypatch.setattr(sys, "argv", ["score.py", "freeze", "--out", str(cp)])
    w.main()
    c, a = fixture()
    assert json.loads(cp.read_text()) == c
    np.savez_compressed(output, **dict(a, metadata=np.array(json.dumps(a["metadata"]))))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "score.py",
            "score",
            "--contract",
            str(cp),
            "--output",
            str(output),
            "--report",
            str(report),
        ],
    )
    w.main()
    r = json.loads(report.read_text())
    assert r["output_sha256"] == hashlib.sha256(output.read_bytes()).hexdigest()
    assert r["engineering_screen_pass"] and not r["industrial_qualified"]


def test_averaged_fields_rejected():
    c, a = fixture()
    a["metadata"]["snapshot_kind"] = "time_mean"
    with pytest.raises(ValueError, match="snapshot"):
        w.score(c, a)


def test_mom_point_initial_cannot_be_relabelled_as_cell_mean():
    c, a = fixture()
    a = cgrid(a, c)
    a["metadata"]["model"] = "MOM6"
    a["metadata"]["source_sha"] = w.MOM6
    a["metadata"]["recorded_numerics"]["resolved_options"] = c["mom_time_options"]
    a["metadata"]["sampling_eta"] = "cell_mean"
    with pytest.raises(ValueError, match="initial wave"):
        w.score(c, a)
