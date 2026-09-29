"""Actual FD, seasonal forcing and stage-ledger continuity across two strict restarts."""
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from test_legacy_reference_geometry import _fixture

from jax_solver_global import JaxStateG
from restart_contract import load_restart, make_restart_contract, save_restart
from run_long_integration_global import interp_monthly_field_jit, interp_seasonal_wind_jit
from stage_budgets import accumulate_budget, empty_budget, make_budget_step


@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("scheme", ["legacy", "subcycled_rk2_v2", "symmetric_fast_v3"])
@pytest.mark.parametrize("ice", [False, True])
def test_two_actual_restarts_preserve_state_seasonal_phase_and_every_budget_field(tmp_path, dtype, scheme, ice):
    grid, (_, initialize, _, params, _) = _fixture(
        dtype=dtype, column_geometry="legacy" if scheme == "legacy" else "nodal_dual_v1",
        match_barotropic_transport=scheme != "legacy", process_time_scheme=scheme,
        use_scan=True, dynamic_ice=ice, mixed_layer_depth_m=20., lambda_bulk=80.,
        T_atm=np.full((8, 8), -3.))
    initial = initialize()._replace(T=jnp.full((8, 8, 4), -1.8, dtype=dtype),
                                    ice=jnp.full((8, 8), 0.25 if ice else 0., dtype=dtype))
    wind = jnp.asarray(np.broadcast_to(np.arange(1., 13.)[:, None, None, None] * 0.001,
                                      (12, 2, 8, 8)), dtype=dtype)
    atmosphere = jnp.asarray(np.broadcast_to((-3. + np.arange(12.) * 0.1)[:, None, None],
                                            (12, 8, 8)), dtype=dtype)
    heat = jnp.zeros((8, 8), dtype=dtype)
    start = int(30. * 86400. / params.dt) - 3
    contract = make_restart_contract(
        grid, params, dtype=dtype, forcing={"wind": wind, "air": atmosphere, "heat": heat},
        controls={"calendar": "360_day", "blend_days": 5., "ledger": "actual_stage_budget"},
        code_paths={name: Path(__file__).resolve().parents[1] / "src" / f"{name}.py"
                    for name in ("jax_solver_global", "stage_budgets", "restart_contract")},
        execution={"backend": jax.default_backend(), "jax": jax.__version__})
    advance = make_budget_step(params)
    accumulate = jax.jit(accumulate_budget)

    def run(restart):
        state = initial
        totals = empty_budget()
        history = []
        for index in range(6):
            step = start + index
            day = step * params.dt / 86400.
            tau_x, tau_y = interp_seasonal_wind_jit(wind, day, blend_days=5.)
            air = interp_monthly_field_jit(atmosphere, day, blend_days=5.)
            state, ledger = advance(state, (tau_x, tau_y, heat), air[..., None])
            totals = accumulate(totals, ledger)
            history.append(float(np.max(np.asarray(state.ice))))
            if restart and index in {1, 3}:
                path = tmp_path / f"restart_{index}.npz"
                save_restart(path, state, contract, step=step + 1,
                             counters={"completed_here": index + 1}, cumulative=totals,
                             history={"ice_peak_m": history})
                saved = load_restart(path, contract)
                assert saved.step == step + 1
                assert saved.elapsed_seconds == (step + 1) * params.dt
                assert saved.counters == {"completed_here": index + 1}
                state = JaxStateG(**{name: jnp.asarray(value) for name, value in saved.state.items()})
                totals = {name: jnp.asarray(value) for name, value in saved.cumulative.items()}
                history = list(saved.history["ice_peak_m"])
        return state, totals, np.asarray(history)

    continuous = run(False)
    resumed = run(True)
    for name in initial._fields:
        np.testing.assert_array_equal(getattr(continuous[0], name), getattr(resumed[0], name))
        assert getattr(resumed[0], name).dtype == jnp.dtype(dtype)
    for name, values in continuous[1].items():
        np.testing.assert_array_equal(values, resumed[1][name], err_msg=name)
        assert np.isfinite(np.asarray(values)).all()
    np.testing.assert_array_equal(continuous[2], resumed[2])
    if ice:
        assert continuous[2][-1] > 0.25
    before = interp_seasonal_wind_jit(wind, (start + 1) * params.dt / 86400.)[0]
    after = interp_seasonal_wind_jit(wind, (start + 4) * params.dt / 86400.)[0]
    assert not np.array_equal(before, after)
