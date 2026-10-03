"""Production first-rejection, duration, input and exit contracts; no downloads."""
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

import ocean_solver.runtime.entry as driver
from ocean_solver.audit.monitor import classify_state
from ocean_solver.config.definitions import PhysicsConfig
from ocean_solver.io.restart import load_restart
from ocean_solver.model.factory import make_solver_global
from ocean_solver.state.types import JaxStateG
from tests.support.driver import run_controlled_driver
from tests.support.grid import all_wet_grid


@pytest.mark.parametrize("field,value", [("u", 11.), ("v", 11.), ("eta", 16.)]
                       + [(name, value) for name in JaxStateG._fields for value in (np.nan, np.inf)])
@pytest.mark.parametrize("failure_step", [1, 3])
def test_every_attempt_is_checked_and_rejected_state_is_not_accepted(tmp_path, monkeypatch, field, value, failure_step):
    calls = []

    def step(state, count):
        calls.append(count)
        if count == failure_step:
            return state._replace(**{field: jnp.full_like(getattr(state, field), value)})
        return state

    _, contract = run_controlled_driver(monkeypatch, tmp_path, step_override=step, expected_code=1)
    assert calls == list(range(1, failure_step + 1))
    with np.load(tmp_path / "global_controlled.npz", allow_pickle=False) as saved:
        assert str(saved["verdict"]) == "FAIL_BLOWUP"
        assert int(saved["attempted_steps"]) == failure_step
        assert int(saved["accepted_steps"]) == failure_step - 1
        assert int(saved["first_rejected_step"]) == failure_step
        assert float(saved["days"][-1]) == (failure_step - 1) * 10. / 86400.
        assert float(saved["diverged_at"]) == failure_step * 10. / 86400.
        assert not bool(saved["duration_complete"])
        if field in {"u", "v"} and np.isfinite(value):
            assert float(saved["max_velocity_peak"]) == value
            assert float(saved["max_u_peak"]) == (value if field == "u" else 0.)
    with np.load(tmp_path / "rejected_controlled.npz", allow_pickle=False) as saved:
        assert not bool(saved["resumable"])
        np.testing.assert_array_equal(saved[field], value)
    checkpoint = tmp_path / "ckpt_controlled.npz"
    if failure_step == 1:
        assert not checkpoint.exists()
    else:
        valid = load_restart(checkpoint, contract)
        assert valid.step == 2
        assert all(np.isfinite(array).all() for array in valid.state.values())


@pytest.mark.parametrize("duration", [0., -60., np.nan, np.inf])
def test_factory_rejects_invalid_timestep_before_constructing_arrays(duration):
    with pytest.raises(ValueError, match="dt"):
        make_solver_global(None, PhysicsConfig(), duration)


@pytest.mark.parametrize("options", [
    ("--days", "-1"), ("--days", "0"), ("--days", "nan"), ("--days", "1e-20"),
    ("--dt", "0"), ("--dt", "inf"), ("--dt-bt", "-1"), ("--resolution", "nan"),
    ("--nu-h", "-1"), ("--kappa-conv", "nan"), ("--nu-nsub", "0"),
    ("--max-steps", "-1"), ("--polar-cap-rows", "-1"), ("--snap-days", "0"),
    ("--z-levels", "0,-5,nan"), ("--checkpoint-days", "nan"),
    ("--days", "1e308"), ("--dt", "1e-308"),
    ("--mixed-layer-lat-band", "nan", "20"), ("--mixed-layer-lat-band", "30", "20"),
    ("--eta-relax-box", "-6", "inf", "30", "46.5"),
])
def test_invalid_cli_inputs_fail_before_loading_data(tmp_path, monkeypatch, options):
    def forbidden(*args, **kwargs):
        pytest.fail("invalid CLI input reached grid loading")

    monkeypatch.setattr(driver, "make_global_grid", forbidden)
    monkeypatch.setattr(sys, "argv", ["ocean-solver", "--out-dir", str(tmp_path / "new"), *options])
    with pytest.raises(SystemExit) as raised:
        driver.main()
    assert raised.value.code == 2
    assert not (tmp_path / "new").exists()


@pytest.mark.parametrize("field", ["u", "v", "eta"])
def test_production_threshold_equality_is_not_silently_tightened(field):
    zero = jnp.zeros((2, 2, 2))
    state = JaxStateG(zero, zero, zero + 17., zero + 35., zero[..., 0], zero[..., 0])
    state = state._replace(**{field: jnp.full_like(getattr(state, field), 15. if field == "eta" else 10.)})
    assert int(classify_state(state).failure) == 0
    assert int(classify_state(state, inclusive=True).failure) != 0


def test_peak_between_snapshots_survives_and_partial_duration_is_not_pass(tmp_path, monkeypatch):
    def step(state, count):
        return state._replace(u=jnp.full_like(state.u, 9. if count % 2 else 0.),
                              v=jnp.full_like(state.v, 8. if count % 2 else 0.))

    run_controlled_driver(monkeypatch, tmp_path, step_override=step, options=("--max-steps", "5"), expected_code=3)
    with np.load(tmp_path / "global_controlled.npz", allow_pickle=False) as saved:
        assert str(saved["verdict"]) == "INCOMPLETE"
        assert int(saved["accepted_steps"]) == int(saved["attempted_steps"]) == 5
        assert int(saved["requested_steps"]) == 8
        assert float(saved["max_u_peak"]) == float(saved["max_velocity_peak"]) == 9.
        assert not bool(saved["duration_complete"])


@pytest.mark.parametrize("name,value", [("nu_h", -1.), ("kappa_conv", np.nan), ("r_bot", np.inf)])
def test_factory_validates_physical_coefficients(name, value):
    with pytest.raises(ValueError, match=name):
        make_solver_global(all_wet_grid(), replace(PhysicsConfig(), **{name: value}), 60.)


def test_module_cli_has_nonzero_parameter_error_exit():
    path = Path(driver.__file__).resolve()
    result = subprocess.run([sys.executable, str(path), "--dt", "0"], capture_output=True, text=True)
    assert result.returncode == 2
    assert "dt must be positive" in result.stderr


@pytest.mark.parametrize("field,value", [("v", 11.), ("eta", 16.), ("S", np.nan)])
def test_invalid_incoming_state_is_never_advanced(tmp_path, monkeypatch, field, value):
    def forbidden(state, count):
        pytest.fail("invalid incoming state was advanced")

    run_controlled_driver(monkeypatch, tmp_path, step_override=forbidden,
                          init_override=lambda state: state._replace(**{
                              field: jnp.full_like(getattr(state, field), value)}), expected_code=1)
    with np.load(tmp_path / "global_controlled.npz", allow_pickle=False) as saved:
        assert int(saved["accepted_steps"]) == int(saved["attempted_steps"]) == 0
        assert int(saved["first_rejected_step"]) == 0


@pytest.mark.parametrize("snap_steps", [1, 2, 4])
def test_snapshot_cadence_cannot_hide_first_rejection(tmp_path, monkeypatch, snap_steps):
    def step(state, count):
        return state._replace(v=jnp.full_like(state.v, 11. if count == 3 else 0.))

    run_controlled_driver(monkeypatch, tmp_path, step_override=step,
                          options=("--snap-days", str(snap_steps * 10. / 86400.),
                                   "--checkpoint-days", "0"), expected_code=1)
    with np.load(tmp_path / "global_controlled.npz", allow_pickle=False) as saved:
        assert int(saved["first_rejected_step"]) == 3
        assert int(saved["accepted_steps"]) == 2


@pytest.mark.parametrize("count", [0, -1, 1.5, True])
def test_factory_does_not_silently_clamp_substep_count(count):
    with pytest.raises(ValueError, match="nu_nsub"):
        make_solver_global(all_wet_grid(), PhysicsConfig(), 60., nu_nsub=count)


def test_state_monitor_does_not_change_an_actual_legal_step():
    grid = all_wet_grid(nx=8, ny=8, nz=4)
    step, initialize, *_ = make_solver_global(grid, PhysicsConfig(), 10., polar_cap_rows=0)
    state = initialize()
    expected = step(state)
    assert int(classify_state(expected).failure) == 0
    actual = step(state)
    for before, after in zip(expected, actual, strict=True):
        np.testing.assert_array_equal(before, after)


def test_factory_rejects_overlapping_polar_bands_before_constructing_arrays(monkeypatch):
    import ocean_solver.model.factory as factory
    monkeypatch.setattr(factory, 'make_fd_params', lambda *args, **kwargs:
                        pytest.fail('overlapping caps reached array construction'))
    with pytest.raises(ValueError, match='polar cap bands'):
        make_solver_global(all_wet_grid(nx=8, ny=8, nz=4), PhysicsConfig(), 10.)


@pytest.mark.parametrize('name,value', [('dy', 0.), ('dy', np.nan), ('dy', np.inf),
                                      ('dx_2d', np.zeros((32, 32))),
                                      ('dx_2d', np.full((32, 32), np.nan)),
                                      ('dz', np.ones(7)), ('z', np.arange(8.))])
def test_factory_rejects_invalid_grid_spacing(name, value):
    with pytest.raises(ValueError, match='dy|dx_2d|dz|z'):
        make_solver_global(replace(all_wet_grid(), **{name: value}), PhysicsConfig(), 10.)
