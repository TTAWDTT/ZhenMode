"""Channel controls frozen before native runs; expectations never call the scorer."""

import json

import numpy as np
import pytest

from tests.support.channel_dynamics import planted
from zhenmode.benchmarks.channel_dynamics import contract, initial_native, initial_points, surface
from zhenmode.evaluation import channel_dynamics as evaluate
from zhenmode.evaluation.native_channel import load


@pytest.mark.parametrize("model", ["ocean-solver", "MOM6", "Oceananigans"])
@pytest.mark.parametrize("method", ["baseline", "symmetric-external-mode", "symmetric-closed-faces", "fourth-order-channel"])
def test_independent_rotating_adjustment_plant_roundtrips_and_scores(tmp_path, model, method):
    c = contract("geostrophic-adjustment", method=method)
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
