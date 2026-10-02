"""Offline CLI contract probes: stubbed states, no ocean integration or downloads."""
import sys
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from config import PhysicsConfig
from jax_solver_global import make_solver_global
from tests.support.driver import driver, run_controlled_driver
from tests.support.grid import all_wet_grid


@pytest.mark.parametrize("scenario", ["persistent_v", "transient_u", "transient_nonfinite", "persistent_u"])
def test_production_cli_rejects_first_invalid_step(tmp_path, monkeypatch, scenario):
    original_factory = driver.make_solver_global

    def factory(*arguments, **keywords):
        result = list(original_factory(*arguments, **keywords))
        attempts = 0

        def stub(state, *forcing, **controls):
            nonlocal attempts
            attempts += 1
            zero = jnp.zeros_like(state.u)
            invalid_x = scenario == "persistent_u" or scenario == "transient_u" and attempts == 1
            velocity_x = jnp.full_like(zero, 11.) if invalid_x else zero
            velocity_y = jnp.full_like(zero, 11.) if scenario == "persistent_v" else zero
            temperature = jnp.full_like(state.T, jnp.nan if scenario == "transient_nonfinite" and attempts == 1 else 17.)
            return state._replace(u=velocity_x, v=velocity_y, T=temperature)

        result[-1] = stub
        return tuple(result)

    monkeypatch.setattr(driver, "make_solver_global", factory)
    if scenario == "persistent_u":
        with pytest.raises(SystemExit) as raised:
            run_controlled_driver(monkeypatch, tmp_path)
        assert raised.value.code != 0
        return
    run_controlled_driver(monkeypatch, tmp_path)
    with np.load(tmp_path / "global_controlled.npz", allow_pickle=False) as saved:
        observed = {"verdict": str(saved["verdict"]), "max_u_peak": float(saved["max_u_peak"]),
                    "days_end": float(saved["days"][-1])}
        print("OBSERVED", scenario, observed)
        assert str(saved["verdict"]) == "FAIL_BLOWUP"


@pytest.mark.parametrize("duration", [0., -60.])
def test_factory_rejects_nonpositive_timestep(duration):
    grid = all_wet_grid(nx=8, ny=8, nz=4)
    with pytest.raises(ValueError, match="dt|time|positive"):
        make_solver_global(grid, PhysicsConfig(), duration)


def test_production_cli_does_not_report_negative_duration_as_success(tmp_path, monkeypatch):
    original_main = driver.main

    def negative_duration_main():
        sys.argv[sys.argv.index("--days") + 1] = "-1"
        return original_main()

    monkeypatch.setattr(driver, "main", negative_duration_main)
    run_controlled_driver(monkeypatch, tmp_path)
    with np.load(tmp_path / "global_controlled.npz", allow_pickle=False) as saved:
        print("OBSERVED negative_days", str(saved["verdict"]), saved["days"].tolist())
        assert str(saved["verdict"]) != "PASS"
