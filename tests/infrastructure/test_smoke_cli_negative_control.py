"""Toy negative controls validate CLI files, not physical solver acceptance."""

import json
import sys

import jax.numpy as jnp
import numpy as np
import pytest

from tests.support.paths import REPOSITORY_ROOT

sys.path.insert(0, str(REPOSITORY_ROOT / "scripts"))

import verify_debug_integration as smoke

from config import PhysicsConfig
from stage_budgets import empty_budget


@pytest.mark.parametrize("nonfinite", [False, True])
def test_cli_saves_exact_rejected_and_last_accepted_states_without_booking_failed_step(tmp_path, monkeypatch, nonfinite):
    from tests.support.material.process_time import _candidate

    grid, solver = _candidate()
    initial = solver[1]()
    atmosphere = np.full((8, 8), 15.)
    forcing = tuple(np.zeros((8, 8)) for _ in range(3))
    monkeypatch.setattr(smoke, "make_smoke_fixture", lambda *arguments: (
        grid, PhysicsConfig(), np.asarray(initial.T), np.asarray(initial.S), atmosphere, forcing))
    monkeypatch.setattr(smoke, "make_solver_global", lambda *arguments, **options: solver)

    def controlled_step(current):
        ledger = empty_budget()
        ledger["budget_residual"] = jnp.array([1., 2., 3.])
        velocity = jnp.full_like(current.u, jnp.nan) if nonfinite else current.u + 4.
        return current._replace(u=velocity), ledger

    monkeypatch.setattr(smoke, "make_budget_step", lambda params: controlled_step)
    bathymetry = tmp_path / "toy_input.txt"
    bathymetry.write_text("toy fixture supplied by test, not ETOPO", encoding="utf-8")
    output = tmp_path / "negative_control.json"
    monkeypatch.setattr(sys, "argv", ["controlled_cli", "--days", "0.001", "--dt", "10",
                                      "--cases", "baseline", "--dtype", "float64", "--audit-budget",
                                      "--bathy", str(bathymetry), "--out", str(output)])
    with pytest.raises(SystemExit) as stopped:
        smoke.main()
    assert stopped.value.code == 1
    report = json.loads(output.read_text(encoding="utf-8"), parse_constant=lambda token: pytest.fail(f"nonstandard JSON {token}"))
    case = report["cases"][0]
    assert report["status"] == case["status"] == "failed"
    accepted = 0 if nonfinite else 2
    attempted = accepted + 1
    assert case["steps"] == case["first_failure"]["last_accepted_step"] == accepted
    assert case["first_failure"]["step"] == attempted
    assert case["records"][0]["accepted_steps_in_batch"] == accepted
    assert case["records"][0]["attempted_steps_in_batch"] == attempted
    np.testing.assert_array_equal(case["stage_budget"]["values"]["budget_residual"],
                                  np.array([1., 2., 3.]) * accepted)
    with np.load(case["final_state_path"]) as saved:
        np.testing.assert_array_equal(saved["u"], 0. if nonfinite else 8.)
    with np.load(case["first_failure"]["rejected_state_path"]) as saved:
        if nonfinite:
            assert np.isnan(saved["u"]).all()
            assert case["records"][0]["batch_peak_velocity"] is None
        else:
            np.testing.assert_array_equal(saved["u"], 12.)
    with np.load(case["first_failure"]["rejected_ledger_path"]) as saved:
        np.testing.assert_array_equal(saved["budget_residual"], [1., 2., 3.])
