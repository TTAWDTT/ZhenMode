"""S2 production contracts on a bounded synthetic CPU grid, never an industrial gate."""
import argparse
import json
from dataclasses import replace

import jax.numpy as jnp
import numpy as np
import pytest

import zhenmode.model.audit.schema as owner_schema
import zhenmode.model.audit.stages as owner_stages
import zhenmode.model.config.definitions as owner_definitions
import zhenmode.model.forcing.air as owner_air
import zhenmode.model.forcing.fields as owner_fields
import zhenmode.model.io.recovery as owner_recovery
import zhenmode.model.io.restart as owner_restart
import zhenmode.model.runtime.entry as driver
import zhenmode.model.runtime.identity as owner_identity
import zhenmode.model.runtime.inputs as owner_inputs
from tests.support.driver import run_controlled_driver
from zhenmode.model.config.definitions import C_P, RHO_0
from zhenmode.model.io.restart import load_restart


def read_result(directory):
    with np.load(directory / 'global_controlled.npz', allow_pickle=False) as saved:
        return {name: saved[name] for name in saved.files}


@pytest.mark.parametrize('wind_jit', [False, True])
def test_real_steps_two_restarts_preserve_accepted_ledger_bytes(tmp_path, monkeypatch, wind_jit):
    options = ('--budget-audit',) + (('--wind-jit',) if wind_jit else ())
    continuous, resumed = tmp_path / 'continuous', tmp_path / 'resumed'
    with monkeypatch.context() as scoped:
        run_controlled_driver(scoped, continuous, options=options)
    checkpoint = resumed / 'ckpt_controlled.npz'
    for restart in (None, checkpoint):
        with monkeypatch.context() as scoped:
            grid, contract = run_controlled_driver(scoped, resumed, restart=restart,
                                                  crash_after=3, options=options)
    record = load_restart(checkpoint, contract)
    assert record.step == 4
    altered = replace(record, cumulative={**record.cumulative,
                      'ledger_source_inputs': record.cumulative['ledger_source_inputs'] + 1})
    with pytest.raises(ValueError, match='ledger differs'):
        owner_recovery._validate_restart_history(altered, grid, 2, 10., False, False, {}, True)
    with monkeypatch.context() as scoped:
        run_controlled_driver(scoped, resumed, restart=checkpoint, options=options)
    first, second = read_result(continuous), read_result(resumed)
    for name in first:
        assert first[name].dtype == second[name].dtype
        assert first[name].tobytes() == second[name].tobytes(), name
    assert first['accepted_steps'] == 8
    assert first['ledger_source_inputs'].shape == (5, 9, 3)
    assert not first['physical_budget_closed']


def test_prescribed_heat_is_independent_source_not_residual(tmp_path, monkeypatch):
    # Nonuniform positive prescribed heat; no bulk, sponge, restoration or ice.
    heat = np.arange(64, dtype=float).reshape(8, 8) + 50.
    monkeypatch.setattr(owner_fields, 'heat_flux_meridional', lambda *args, **kwargs: heat)
    # Helper disables heat via CLI. Remove that one argument at parser boundary.
    parse = argparse.ArgumentParser.parse_args
    def parse_args(parser, *args, **kwargs):
        result = parse(parser, *args, **kwargs)
        result.no_meridional_heat_flux = False
        return result
    monkeypatch.setattr(argparse.ArgumentParser, 'parse_args', parse_args)
    grid, _ = run_controlled_driver(monkeypatch, tmp_path, options=('--budget-audit',))
    result = read_result(tmp_path)
    expected = np.sum(heat * grid.dx_2d * grid.dy) * 80.
    source = result['ledger_source_inputs'][-1, 0, 0]
    np.testing.assert_allclose(source, expected, rtol=2e-15)
    assert np.count_nonzero(result['ledger_source_inputs'][-1]) == 1
    observed = result['ledger_observed_change'][-1, 0]
    # Sum of per-step residuals and subtraction of two cumulative O(1e18)
    # totals have different rounding; bound from their operands, never from
    # the small residual. No physical closure tolerance is changed here.
    rounding = 8 * np.finfo(float).eps * (abs(observed) + abs(source))
    assert abs(result['ledger_budget_residual'][-1, 0] - (observed - source)) <= rounding
    # Independent endpoint inventory, distinct from stage telescoping.
    # Full state at step 6 is in existing checkpoint; avoid inventing a second format.
    with np.load(tmp_path / 'ckpt_controlled.npz') as z:
        temperature = z['state__T']
    spacing = np.abs(np.diff(grid.z))
    widths = np.r_[spacing[0], (spacing[:-1] + spacing[1:]) / 2, spacing[-1]]
    volume = grid.dx_2d[..., None] * grid.dy * widths
    expected_inventory = RHO_0 * C_P * np.sum((temperature - 17.) * volume)
    np.testing.assert_allclose(result['ledger_observed_change'][-2, 0], expected_inventory, rtol=1e-12)


@pytest.mark.parametrize('kind', ['identity', 'nonfinite', 'negative_infinity', 'velocity'])
def test_rejected_attempt_never_enters_ledger(tmp_path, monkeypatch, kind):
    original = owner_stages.make_budget_step
    def make(params):
        advance = original(params)
        calls = 0
        def audit(*args, **kwargs):
            nonlocal calls
            calls += 1
            state, ledger = advance(*args, **kwargs)
            if calls == 3:
                if kind == 'identity':
                    state = state._replace(T=state.T.at[0, 0, 0].add(1e-10))
                elif kind == 'negative_infinity':
                    ledger = {**ledger, 'projection_relative_residual_max': jnp.asarray(-jnp.inf)}
                elif kind == 'nonfinite':
                    ledger = {**ledger, 'source_inputs': ledger['source_inputs'] + jnp.nan}
            return state, ledger
        return audit
    monkeypatch.setattr(owner_stages, 'make_budget_step', make)
    override = (lambda state, count: state._replace(v=jnp.full_like(state.v, 11.))
                if count == 3 else state) if kind == 'velocity' else None
    # Velocity stub differs from real ordinary step; use identical audit for first 2 attempts.
    if override:
        def make_stub(params):
            count = 0
            def audit(state, *args):
                nonlocal count
                count += 1
                return override(state, count), owner_schema.empty_budget()
            return audit
        monkeypatch.setattr(owner_stages, 'make_budget_step', make_stub)
    _, contract = run_controlled_driver(monkeypatch, tmp_path, step_override=override,
                                        options=('--budget-audit',), expected_code=1)
    result = read_result(tmp_path)
    assert result['accepted_steps'] == 2 and result['attempted_steps'] == 3
    assert result['failure_code'] == {'identity': 7, 'nonfinite': 2, 'negative_infinity': 2, 'velocity': 3}[kind]
    saved = load_restart(tmp_path / 'ckpt_controlled.npz', contract)
    for name in owner_schema.empty_budget():
        assert saved.cumulative['ledger_' + name].tobytes() == result['ledger_' + name][-1].tobytes()
    with np.load(tmp_path / 'rejected_controlled.npz') as z:
        assert not z['resumable']
        assert 'ledger_source_inputs' in z


def test_requested_air_failure_requires_explicit_fallback(tmp_path, monkeypatch):
    def unavailable(*args, **kwargs):
        raise OSError('controlled unavailable air')
    monkeypatch.setattr(owner_air, 'load_monthly_mean_air_temp', unavailable)
    # Enable bulk despite fixture's default --no-bulk-flux.
    parse = argparse.ArgumentParser.parse_args
    def parse_args(parser, *args, **kwargs):
        result = parse(parser, *args, **kwargs)
        result.no_bulk_flux = False
        return result
    monkeypatch.setattr(argparse.ArgumentParser, 'parse_args', parse_args)
    with monkeypatch.context() as scoped:
        with pytest.raises(ValueError, match='fallback disabled'):
            run_controlled_driver(scoped, tmp_path / 'strict', options=('--real-air-temp-monthly',))
    with monkeypatch.context() as scoped:
        run_controlled_driver(scoped, tmp_path / 'explore', options=(
            '--real-air-temp-monthly', '--allow-forcing-fallback'))
    report = json.loads(str(read_result(tmp_path / 'explore')['forcing_provenance_json']))
    assert report['requested']['monthly_air']
    assert report['applied']['air'] == 'zonal WOA SST (explicit fallback)'
    assert len(report['fallback_events']) == 1
    assert report['effective_arrays']['T_atm']['sha256']
    assert 'air' not in report['selected_files']
    assert report['physical_forcing_qualification'] == 'not_assessed'


def test_strict_preflight_fails_before_loading(tmp_path, monkeypatch):
    monkeypatch.setattr(owner_inputs, '_input_files', lambda *args, **kwargs: {'missing': tmp_path / 'missing'})
    with pytest.raises(ValueError, match='strict forcing preflight'):
        run_controlled_driver(monkeypatch, tmp_path, options=('--strict-forcing',))


def test_effective_nonfinite_forcing_fails_before_solver(tmp_path, monkeypatch):
    monkeypatch.setattr(owner_fields, 'air_temp_profile', lambda *args: np.full((8, 8), np.inf))
    parse = argparse.ArgumentParser.parse_args
    def parse_args(parser, *args, **kwargs):
        result = parse(parser, *args, **kwargs)
        result.no_bulk_flux = False
        return result
    monkeypatch.setattr(argparse.ArgumentParser, 'parse_args', parse_args)
    with pytest.raises(ValueError, match='nonfinite effective forcing'):
        run_controlled_driver(monkeypatch, tmp_path)


def test_eta_peak_between_snapshots_is_persisted(tmp_path, monkeypatch):
    def step(state, count):
        return state._replace(eta=jnp.full_like(state.eta, 4. if count % 2 else 0.))
    _, contract = run_controlled_driver(monkeypatch, tmp_path, step_override=step)
    assert read_result(tmp_path)['max_eta_peak'] == 4.
    assert np.max(read_result(tmp_path)['max_eta']) == 0.
    assert load_restart(tmp_path / 'ckpt_controlled.npz', contract).cumulative['max_eta_peak'] == 4.


def test_audit_identity_checks_bytes_including_signed_zero():
    assert not owner_identity._same_state_bytes((np.array([0.]),), (np.array([-0.]),))
    assert not owner_identity._same_state_bytes((np.array([0.], dtype='float32'),), (np.array([0.]),))


def test_source_hashes_survive_flat_wheel_layout(tmp_path, monkeypatch):
    import shutil

    from zhenmode.provenance.sources import source_paths, source_root
    source = source_root(driver.__file__)
    paths = source_paths(source, owner_identity.SOURCE_MODULES)
    installed = tmp_path / 'site-packages'
    installed.mkdir()
    for name in owner_identity.SOURCE_MODULES:
        target = installed / f'{name}.py'
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(paths[name], target)
    monkeypatch.setattr(owner_identity, '__file__', str(installed / 'zhenmode/model/runtime/identity.py'))
    identity = owner_identity._source_identity()
    assert len(identity['source_sha256']) == len(owner_identity.SOURCE_MODULES)
    assert identity['source_sha256']['zhenmode/model/runtime/entry.py'] == owner_restart.file_sha256(paths['zhenmode/model/runtime/entry'])


def test_bathymetry_provenance_follows_offline_loader_precedence(tmp_path, monkeypatch):
    from types import SimpleNamespace

    import zhenmode.model.io.grid as grid
    bathy = tmp_path / 'bathy.nc'
    bathy.write_bytes(b'nc sentinel')
    twin = tmp_path / 'bathy.nc.npz'
    twin.write_bytes(b'npz sentinel')
    monkeypatch.setattr(owner_definitions, 'DEFAULT_CONFIG', SimpleNamespace(bathymetry_file=str(bathy)))
    args = SimpleNamespace(init_from='init.npz', seasonal_wind=False, month='2023-01',
                           wind_year=2023, no_bulk_flux=True)
    assert owner_inputs._input_files(args, default_config=owner_definitions.DEFAULT_CONFIG)['bathymetry'] == bathy
    monkeypatch.setattr(grid, 'Dataset', None)
    assert owner_inputs._input_files(args, default_config=owner_definitions.DEFAULT_CONFIG)['bathymetry'] == twin
