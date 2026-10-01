import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "research/experiments/local_top_bridge"))
spec = importlib.util.spec_from_file_location(
    "localtop", ROOT / "research/experiments/local_top_bridge/component.py"
)
b = importlib.util.module_from_spec(spec)
spec.loader.exec_module(b)


def fixture(eta=-2.49, profile="linear"):
    d = np.array([0.0, 5, 15, 30, 100, 300, 500, 1000])
    w = np.array([2.5, 7.5, 12.5, 42.5, 135, 200, 350, 750])
    mask = np.array([1, 1, 1, 1, 1, 1, 1, 0])
    x = -d
    temp = 20 + 0.005 * x
    if profile == "uniform":
        temp[:] = 15
    if profile == "curved":
        temp = 20 + 0.005 * x + 1e-6 * x * x
    v = np.array([temp, np.full(8, 35), 0.001 * x, -0.0005 * x]).T
    return dict(
        depth=d,
        reference_weights=w,
        wet_mask=mask,
        values=v,
        eta=eta,
        Tref=15.0,
        Sref=35.0,
        alpha=2e-4,
        beta=7.6e-4,
        rho0=1025.0,
        gravity=9.81,
        terrain_depth=750.0,
        discrete_bottom=750.0,
        control_interfaces=np.array([]),
        source_sha=np.array("2" * 40),
        terrain_sha=np.array("a" * 64),
    )


@pytest.mark.parametrize("profile", ["uniform", "linear", "curved"])
def test_local_preservation_and_pressure_cost(profile):
    p = fixture(profile=profile)
    snap = p["values"].copy()
    out, ok, r = b.column(p)
    assert ok
    assert abs(r["band_water_m"] - 20.01) < 1e-12
    assert np.all(abs(np.array(r["inventory_residual"])) <= r["inventory_roundoff_bound"])
    assert out["deep"]["values"].tobytes() == snap[3:].tobytes()
    assert p["values"].tobytes() == snap.tobytes()
    assert np.isfinite(b.pressure(out, -400))
    assert b.pressure(out, -400) - np.interp(
        400, out["original_depth"], out["original_pressure"]
    ) == pytest.approx(r["band_bottom_delta_pressure_Pa"], abs=1e-10)


def test_different_eta_own_domains_and_common_pair():
    a, ok, _ = b.column(fixture(-0.1))
    c, ok2, _ = b.column(fixture(-2.49))
    assert ok and ok2
    assert a["z"][0] == -0.1 and c["z"][0] == -2.49
    r = b.pair(a, c, 1000)
    assert r["common_domain_m"] == [-400.0, -2.49]
    assert r["bottom_delta_difference_Pa"] == pytest.approx(
        (a["new_base"] - a["original_base"]) - (c["new_base"] - c["original_base"])
    )
    with pytest.raises(ValueError):
        b.pressure(c, -0.1)


@pytest.mark.parametrize("eta", [-2.5, -2.5 + 1e-15, 0.1])
def test_invalid_geometry_rollback(eta):
    p = fixture(eta)
    snap = p["values"].copy()
    out, ok, r = b.column(p)
    assert not ok and r["rejection_reason"]
    assert out["values"].tobytes() == snap.tobytes() == p["values"].tobytes()


def test_positive_near_exhaustion_velocity_semantics():
    out, ok, r = b.column(fixture(-2.5 + 1e-8))
    assert ok
    assert r["original_reference_to_actual_velocity_factor_max"] > 1e8
    assert np.isfinite(r["candidate_actual_mass_K_J"])


def test_dry_column_not_fabricated():
    p = fixture()
    p["wet_mask"][:] = 0
    _, ok, _ = b.column(p)
    assert not ok


def test_deep_inventory_and_pressure_increments_byte_exact():
    p = fixture()
    out, ok, _ = b.column(p)
    assert ok
    h = p["reference_weights"] * p["wet_mask"]
    old = h[:, None] * p["values"]
    old[:, 2:] *= p["rho0"]
    assert out["deep"]["inventory"].tobytes() == old[3:].tobytes()
    increment = out["original_pressure"][3:] - out["original_base"]
    assert out["deep"]["pressure_increment"].tobytes() == increment.tobytes()
    assert out["deep"]["depth"].tobytes() == p["depth"][3:].tobytes()


def test_canonical_cli_static_pair(tmp_path):
    import replay

    shared = ["Tref", "Sref", "alpha", "beta", "rho0", "gravity", "source_sha", "terrain_sha"]
    p = fixture(-0.1)
    q = fixture(-2.4914792546513693)
    packet = {k: p[k] for k in shared}
    for k in [
        "depth",
        "reference_weights",
        "wet_mask",
        "values",
        "eta",
        "terrain_depth",
        "discrete_bottom",
    ]:
        packet[k] = np.array([p[k], q[k]])
    packet["distances_m"] = np.array([[0.0, 1000.0], [1000.0, 0.0]])
    path = tmp_path / "input.npz"
    np.savez(path, **packet)
    r = replay.run(path)
    assert all(c["accepted"] for c in r["columns"]) and len(r["pairs"]) == 1
    assert abs(r["columns"][1]["band_water_m"] - 20.00852074534863) < 1e-12
    assert r["qualification_passed"] is False
