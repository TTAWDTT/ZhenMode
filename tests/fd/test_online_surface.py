"""External surface inventory oracle and an actual small FD coupling smoke."""
from dataclasses import replace

import numpy as np
import pytest

from tests.support.grid import all_wet_grid
from zhenmode.model.config import C_P, RHO_0, PhysicsConfig
from zhenmode.model.diagnostics.surface_budget import surface_budget
from zhenmode.model.inputs.forcing.online import bind_open_water_step
from zhenmode.model.solver.factory import make_solver_global
from zhenmode.model.solver.numerics.backend import jnp
from zhenmode.model.solver.physics.air_sea import (
    AirState,
    SurfaceFluxes,
    apply_open_water_exchange,
)
from zhenmode.model.solver.state import JaxStateG


def test_external_heat_salt_inventory_and_corrupted_source_control():
    shape = (2, 2, 2)
    before = JaxStateG(jnp.zeros(shape), jnp.zeros(shape), jnp.full(shape, 20.),
                       jnp.full(shape, 35.), jnp.zeros((2, 2)), jnp.zeros((2, 2)))
    area = np.array([[2., 3.], [5., 7.]])
    thickness = np.array([2., 8.])
    volume = area[:, :, None] * thickness
    weights = jnp.broadcast_to(jnp.array([.5, 0.]), shape)
    def field(value):
        return jnp.full((2, 2), value)
    air = AirState(*[field(value) for value in (290., .005, 101325., 0., 0., 0., 0., .001, 0., 0., 0.)])
    flux = SurfaceFluxes(*[field(value) for value in (0., 0., 10., -20., .0002, 100., 10., .001, .001, .001)])
    after, heat, salt, _, _ = apply_open_water_exchange(before, flux, air, field(36.), weights, 60.)
    # Separate constant-flux formula, without asking the tendency/ledger for its source.
    expected_heat = 100 * area.sum() * 60
    expected_salt = (-(35 * .0008 / RHO_0) + 50 / (365 * 86400)) * area.sum() * 60 * RHO_0 / 1000
    assert RHO_0 * C_P * np.sum(volume * (np.asarray(after.T) - 20)) == pytest.approx(expected_heat, rel=1e-10)
    assert RHO_0 / 1000 * np.sum(volume * (np.asarray(after.S) - 35)) == pytest.approx(expected_salt, rel=1e-8)
    report = surface_budget(before, after, wet_volume=volume, wet_area=area,
                            heat_flux=heat, salt_flux=salt, dt_seconds=60.)
    np.testing.assert_allclose(report['residual'], [0., 0.], atol=2e-6)
    bad = surface_budget(before, after, wet_volume=volume, wet_area=area,
                         heat_flux=heat * 2, salt_flux=salt, dt_seconds=60.)
    assert bad['residual'][0] == pytest.approx(-expected_heat, rel=1e-10)
    assert bad['climate_qualification'] is False


def test_fp32_state_is_not_silently_promoted_by_fp64_weather():
    shape = (2, 2, 2)
    field = jnp.ones((2, 2), dtype=jnp.float64)
    before = JaxStateG(jnp.zeros(shape, dtype=jnp.float32), jnp.zeros(shape, dtype=jnp.float32),
                       jnp.full(shape, 20., dtype=jnp.float32), jnp.full(shape, 35., dtype=jnp.float32),
                       jnp.zeros((2, 2), dtype=jnp.float32), jnp.zeros((2, 2), dtype=jnp.float32))
    air = AirState(*[field * value for value in (290., .005, 101325., 0., 0., 0., 0., .001, 0., 0., 0.)])
    flux = SurfaceFluxes(*[field * value for value in (0., 0., 10., -20., .0002, 100., 10., .001, .001, .001)])
    weights = jnp.broadcast_to(jnp.array([.5, 0.]), shape)
    after, *_ = apply_open_water_exchange(before, flux, air, field * 36, weights, 60.)
    assert after.T.dtype == after.S.dtype == jnp.float32
    expected_T = 20 + 100 * 60 / (RHO_0 * C_P * 2)
    expected_S = 35 + 60 / 2 * (-35 * .0008 / RHO_0 + 50 / (365 * 86400))
    assert abs(float(after.T[0, 0, 0]) - expected_T) <= float(np.spacing(np.float32(20))) / 2
    assert abs(float(after.S[0, 0, 0]) - expected_S) <= float(np.spacing(np.float32(35))) / 2


class ManufacturedWeather:
    data_kind = 'manufactured'

    def __init__(self, shape):
        self.shape = shape
        self.snow = 0.

    def sample(self, start, *, interval_end_seconds):
        assert interval_end_seconds - start == 60
        values = (290., .005, 101325., 6., 2., 200., 320., 1e-5, self.snow, 0., 0.)
        return AirState(*[jnp.full(self.shape, value) for value in values])


def test_actual_fd_route_and_explicit_scope_rejection():
    grid = all_wet_grid(nx=8, ny=8, nz=4)
    grid = replace(grid, z=np.array([0., -5., -20., -50.]), dz=np.array([5., 15., 30.]))
    physics = replace(PhysicsConfig(), nu_h=0., nu_bi=0., nu_v=0., kappa_h=0., kappa_v=0.,
                      kappa_conv=0., kappa_gm=0., kappa_redi=0.)
    _, _, _, params, _, step_dyn = make_solver_global(
        grid, physics, 60., dynamic_forcing=True, return_params=True,
        polar_cap_rows=0, polar_cap_taper=0)
    shape = (8, 8, 4)
    state = JaxStateG(jnp.zeros(shape), jnp.zeros(shape), jnp.full(shape, 20.),
                      jnp.full(shape, 35.), jnp.zeros((8, 8)), jnp.zeros((8, 8)))
    weather = ManufacturedWeather((8, 8))
    step = bind_open_water_step(weather, step_dyn, params, sss_reference=np.full((8, 8), 35.))
    result, report = step(state, 0.)
    assert np.isfinite(np.array(result.T)).all()
    assert np.max(abs(np.asarray(result.T) - np.asarray(state.T))) > 0
    assert np.max(abs(np.asarray(result.u))) > 0  # online stress reaches the actual FD step
    # Roundoff is judged against independently measured surface input, not refilled.
    np.testing.assert_allclose(report['observed_change'], report['independent_surface_input'], rtol=1e-8)
    assert report['data_kind'] == 'manufactured' and not report['execution_ready_for_full_protocol']
    assert report['surface_integrals']['tau_x'] > 0 and report['integral_units']['latent'] == 'J'
    weather.snow = 1e-5
    with pytest.raises(ValueError, match='snow/calving'):
        step(state, 0.)
    with pytest.raises(ValueError, match='ice-free'):
        step(state._replace(ice=jnp.ones((8, 8))), 0.)
    with pytest.raises(ValueError, match='conflicts'):
        bind_open_water_step(weather, step_dyn, params._replace(lambda_bulk=80.),
                             sss_reference=np.full((8, 8), 35.))
