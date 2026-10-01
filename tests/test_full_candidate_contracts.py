"""Safety gates only; no physical or historical failure qualification."""
import numpy as np
import pytest
from test_full_candidate_geometry import configured, k


def assert_rollback(actual, expected):
    assert actual.identity == expected.identity and actual.step == expected.step
    for name in ('h', 'n', 'eta', 'ice', 'band_bottom'):
        assert getattr(actual, name).tobytes() == getattr(expected, name).tobytes()


@pytest.mark.parametrize('name', ['kappa_h', 'kappa_v', 'kappa_conv', 'nu_h', 'nu_v', 'r_bot'])
@pytest.mark.parametrize('value', [-.001, np.nan, np.inf, -np.inf])
def test_invalid_dissipation_rejected_with_complete_rollback(name, value, monkeypatch):
    grid, params, state = configured()
    setattr(params, name, value)
    state.identity['parameter_sha'] = k.parameter_digest(params)
    before = state.copy()
    def unexpected_update(*args, **kwargs):
        pytest.fail('invalid coefficient reached the numerical update')
    monkeypatch.setattr(k, 'linear', unexpected_update)
    out, accepted, report = k.advance(state, params, grid, forcing_sha='b' * 64)
    assert not accepted, (name, value, report)
    assert_rollback(out, before)
    assert_rollback(state, before)
    assert not np.shares_memory(out.n, state.n)


@pytest.mark.parametrize('name', ['dx_2d', 'dy', 'cos_lat'])
def test_consumed_metric_mismatch_rejected_on_advance_and_restart(name, tmp_path):
    grid, params, state = configured()
    path = tmp_path / 'restart.npz'
    k.save(state, path)
    setattr(grid, name, np.asarray(getattr(grid, name)) * 2)
    out, accepted, report = k.advance(state, params, grid, forcing_sha='b' * 64)
    assert not accepted, report
    assert_rollback(out, state)
    with pytest.raises(ValueError, match='metric'):
        k.load(path, expected_identity=state.identity, p=params, grid=grid)


def test_restart_binds_parameter_snapshot_even_when_grid_matches(tmp_path):
    grid, params, state = configured()
    path = tmp_path / 'restart.npz'
    k.save(state, path)
    grid.dx_2d = grid.dx_2d * 2
    params.dx_2d = np.asarray(params.dx_2d) * 2
    with pytest.raises(ValueError, match='snapshot'):
        k.load(path, expected_identity=state.identity, p=params, grid=grid)


@pytest.mark.parametrize('mixing', [False, True])
def test_zero_and_positive_controls_remain_accepted(mixing):
    grid, params, state = configured(mixing=mixing)
    _, accepted, report = k.advance(state, params, grid, forcing_sha='b' * 64)
    assert accepted, report


def test_positive_bottom_drag_dissipates():
    grid, params, state = configured(mixing=True)
    params.kappa_h = params.kappa_v = params.kappa_conv = params.nu_h = params.nu_v = 0.
    params.r_bot = 0.
    control = k.linear(state, params, grid, 15.)
    params.r_bot = .001
    before = state.copy()
    damped = k.linear(state, params, grid, 15.)
    area = np.asarray(params.dx_2d) * params.dy

    def energy(value):
        active = value.h > 0
        return np.sum((area[..., None, None] * value.n[..., 2:] ** 2)[active]
                      / (2 * k.RHO * value.h[active, None]))

    assert energy(damped) < energy(control)
    params.r_bot = -.001
    anti_damped = k.linear(state, params, grid, 15.)
    assert energy(anti_damped) > energy(control)  # Witness for why advance rejects this.
    assert_rollback(state, before)


@pytest.mark.parametrize('name', ['dx_2d', 'dy', 'cos_lat'])
@pytest.mark.parametrize('value', [0., -1., np.nan, np.inf])
def test_invalid_matching_metrics_rejected(name, value):
    grid, params, state = configured()
    metric = np.full(np.asarray(getattr(grid, name)).shape, value)
    setattr(grid, name, metric)
    setattr(params, name, metric.copy())
    state.identity['parameter_sha'] = k.parameter_digest(params)
    out, accepted, report = k.advance(state, params, grid, forcing_sha='b' * 64)
    assert not accepted, report
    assert 'metric' in report['rejection_reason']
    assert_rollback(out, state)


@pytest.mark.parametrize('name', ['kappa_h', 'kappa_v', 'kappa_conv', 'nu_h', 'nu_v', 'r_bot'])
def test_masked_control_rejected(name):
    grid, params, state = configured()
    setattr(params, name, np.ma.array(0., mask=True))
    state.identity['parameter_sha'] = k.parameter_digest(params)
    out, accepted, report = k.advance(state, params, grid, forcing_sha='b' * 64)
    assert not accepted and 'unknown' in report['rejection_reason']
    assert_rollback(out, state)


@pytest.mark.parametrize('name', ['dx_2d', 'dy', 'cos_lat'])
def test_matching_masked_metrics_rejected(name):
    grid, params, state = configured()
    metric = np.ma.array(getattr(grid, name), mask=True)
    setattr(grid, name, metric)
    setattr(params, name, metric.copy())
    state.identity['parameter_sha'] = k.parameter_digest(params)
    out, accepted, report = k.advance(state, params, grid, forcing_sha='b' * 64)
    assert not accepted and 'unknown' in report['rejection_reason']
    assert_rollback(out, state)
