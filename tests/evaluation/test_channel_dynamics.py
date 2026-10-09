"""Channel controls frozen before native runs; expectations never call the scorer."""

import json

import numpy as np
import pytest

from tests.support.standing_wave import cgrid, fixture
from zhenmode.benchmarks.channel_dynamics import contract, initial_native, initial_points, surface
from zhenmode.benchmarks.standing_wave import MOM6, OCEANANIGANS, digest
from zhenmode.evaluation import channel_dynamics as evaluate
from zhenmode.evaluation.native_channel import load


def planted(c, model):
    nx, ny, nz = c["nx"], c["ny"], c["nz"]
    time = np.arange(0, c["duration_s"] + 1, c["output_s"], dtype=float)
    dx, dy = c["Lx_m"] / nx, c["Ly_m"] / ny
    x, y = np.meshgrid((np.arange(nx) + 0.5) * dx, (np.arange(ny) + 0.5) * dy, indexing="ij")
    _, old = fixture()
    m = old["metadata"]
    m.update(
        schema=c["schema"],
        contract_sha256=digest(c),
        dt_s=c["dt"],
        steps=int(c["duration_s"] / c["dt"]),
        f=c["f"],
        Lx=c["Lx_m"],
        Ly=c["Ly_m"],
        model=model,
    )
    if model != "ocean-solver":
        m.update(
            sampling_eta="cell_mean",
            sampling_u="node",
            velocity_layout="cgrid",
            source_sha=MOM6 if model == "MOM6" else OCEANANIGANS,
        )
    m["recorded_numerics"]["resolved_options"] = c[
        {
            "ocean-solver": "ocean_options",
            "MOM6": "mom_time_options",
            "Oceananigans": "oceananigans_options",
        }[model]
    ]
    shape = (len(time), nx * ny, nz)
    k = np.pi / c["Ly_m"]
    omega = 2 * np.pi / 32000
    factor = 1 if model == "ocean-solver" else np.sinc(1 / (2 * ny))
    eta = (
        0.01
        * np.cos(k * y.ravel())[None, :]
        * (0.25 + 0.75 * np.cos(omega * time))[:, None]
        * factor
    )
    h = np.broadcast_to(c["initial_h_m"], shape).copy()
    if model == "Oceananigans":
        h *= 1 + eta[..., None] / 100
    else:
        h[:, :, 0] += eta
    a = dict(
        metadata=m,
        time=time,
        x_eta=x.ravel(),
        y_eta=y.ravel(),
        area=np.full(nx * ny, dx * dy),
        eta=eta,
        h=h,
        T=np.full(shape, 15.0),
        S=np.full(shape, 35.0),
        x_u=x.ravel(),
        y_u=y.ravel(),
        x_v=x.ravel(),
        y_v=y.ravel(),
        width_u=np.zeros(nx * ny),
        u=np.zeros(shape),
        v=np.zeros(shape),
        volume_u=h * dx * dy,
        volume_v=h * dx * dy,
    )
    if model != "ocean-solver":
        a = cgrid(a, c)
    velocity_scale = 0.01 * np.sqrt(9.81 / 100) * np.sqrt(0.75)
    for q in ("u", "v"):
        signal = 0.5 * (1 - np.cos(omega * time)) if q == "u" else np.sin(omega * time)
        a[q] = np.broadcast_to(
            (velocity_scale * np.sin(k * a["y_" + q])[None, :] * signal[:, None])[..., None],
            a[q].shape,
        ).copy()
    return a


@pytest.mark.parametrize("model", ["ocean-solver", "MOM6", "Oceananigans"])
def test_independent_rotating_adjustment_plant_roundtrips_and_scores(tmp_path, model):
    c = contract("geostrophic-adjustment")
    a = planted(c, model)
    path = tmp_path / "plant.npz"
    np.savez_compressed(path, **(a | {"metadata": np.array(json.dumps(a["metadata"]))}))
    r = evaluate.score(c, load(path))
    assert r["engineering_screen_pass"]
    assert max(r["metrics"].values()) < 2e-12
    # Deliberately swap a physical sign; the actual scorer must detect it.
    a["u"] *= -1
    assert "u_error" in evaluate.score(c, a)["failed_metrics"]


def test_catches_clock_and_actual_physics_metadata_corruption():
    c = contract("geostrophic-adjustment")
    a = planted(c, "ocean-solver")
    a["metadata"]["f"] *= 2
    with pytest.raises(ValueError, match="physical metadata"):
        evaluate.score(c, a)
    a["metadata"]["f"] = c["f"]
    a["time"][-1] -= 1
    with pytest.raises(ValueError, match="wrong time"):
        evaluate.score(c, a)


def test_initializer_has_known_quarter_period_and_front_anchors():
    c = contract("geostrophic-adjustment")
    np.testing.assert_allclose(
        surface(c, np.array([0.0, c["Ly_m"] / 2, c["Ly_m"]])), [0.01, 0, -0.01], atol=1e-17
    )
    eta = evaluate.reference_surface(c, np.array([8000.0]), np.array([0.0]))
    assert eta[0, 0] == pytest.approx(0.0025, abs=1e-17)
    c = contract("thermal-wind")
    p = initial_points(c, np.zeros(3), np.array([0.0, 50000.0, 100000.0]), np.zeros(3))
    np.testing.assert_allclose(p["T"], [14.6, 15.1, 15.6], atol=1e-14)
    np.testing.assert_allclose(
        p["u"], [0.0, -9.81 * 0.0002 * 100 * 0.5 * np.pi / (50000 * 0.0001), 0.0], atol=1e-14
    )


@pytest.mark.parametrize("model", ["ocean-solver", "MOM6", "Oceananigans"])
def test_thermal_initialization_conserves_density_neutral_spice_and_bottom_pressure(model):
    c = contract("thermal-wind")
    a = initial_native(c, model)
    # At fixed y/z, density-neutral perturbations cancel in the linear EOS.
    p1 = initial_points(c, 0.0, 50000.0, -50.0)
    p2 = initial_points(c, 50000.0, 50000.0, -50.0)
    density_difference = -0.205 * (p1["T"] - p2["T"]) + 0.779 * (p1["S"] - p2["S"])
    assert abs(density_difference) < 1e-14
    # Check the hydrostatic column equation directly rather than sharing the root routine.
    y = np.array([0.0, 12500.0, 50000.0, 87500.0, 100000.0])
    eta = surface(c, y)
    delta = np.array([-0.5, -0.5, 0, 0.5, 0.5])
    column = eta - 0.0002 * (0.02 * eta**2 / 2 + delta * (100 + eta))
    np.testing.assert_allclose(column, 0.0, atol=1e-16)
    assert np.isfinite(a["T"]).all() and np.isfinite(a["S"]).all()


def test_reference_thermal_quadrature_agrees_under_refinement():
    c = contract("thermal-wind")
    nx, ny, nz = c["nx"], c["ny"], c["nz"]
    initial = initial_native(c, "MOM6")
    x, y = np.meshgrid(
        (np.arange(nx) + 0.5) * c["Lx_m"] / nx,
        (np.arange(ny) + 0.5) * c["Ly_m"] / ny,
        indexing="ij",
    )
    time = np.array([0.0, 86400.0])
    shape = (2, nx * ny, nz)
    a = dict(
        metadata={"model": "MOM6"},
        time=time,
        x_eta=x.ravel(),
        y_eta=y.ravel(),
        eta=np.broadcast_to(initial["eta"].ravel(), (2, nx * ny)),
        h=np.broadcast_to(initial["h"].reshape(nx * ny, nz), shape),
        T=np.zeros(shape),
        S=np.zeros(shape),
    )
    t16, s16 = evaluate.reference_tracers(c, a, 16)
    t32, s32 = evaluate.reference_tracers(c, a, 32)
    np.testing.assert_allclose(t16, t32, rtol=0, atol=5e-13)
    np.testing.assert_allclose(s16, s32, rtol=0, atol=5e-13)
    np.testing.assert_allclose(t32[0], initial["T"].reshape(nx * ny, nz), rtol=0, atol=5e-13)
    np.testing.assert_allclose(s32[0], initial["S"].reshape(nx * ny, nz), rtol=0, atol=5e-13)
