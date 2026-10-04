"""Preserve the legacy hard-gate boundary; these are not smooth-AD gates."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tests.support.fd.convection import _material_fixture, _state
from zhenmode.model.config import ALPHA_T, BETA_S, RHO_0
from zhenmode.model.solver.physics.vertical import _conv_flux_tendency, _convective_mask


@pytest.fixture
def neutral_column():
    _, (_, initialize, _, params, _) = _material_fixture()
    params = params._replace(kappa_conv=.01)
    state = _state(initialize)
    return state._replace(T=state.T.at[..., 0].set(16.),
                          S=state.S.at[..., 0].set(35. + ALPHA_T / BETA_S)), params


@pytest.mark.parametrize("salinity_offset", [-1e-4, -1e-6, -1e-8, -1e-10,
                                           1e-10, 1e-8, 1e-6, 1e-4])
def test_hard_gate_has_finite_flux_jump_despite_column_conservation(neutral_column, salinity_offset):
    state, params = neutral_column
    state = state._replace(S=state.S.at[..., 0].add(salinity_offset))
    column, gate = _convective_mask(state, params)
    contrast = RHO_0 * (-ALPHA_T * float(state.T[0, 0, 0] - state.T[0, 0, 1])
                       + BETA_S * float(state.S[0, 0, 0] - state.S[0, 0, 1]))
    assert bool(gate[0, 0, 0]) == (contrast > 0.) == (salinity_offset > 0.)
    expected = -params.kappa_conv / (params.dz_surface * np.asarray(params.dz_iface).ravel()[0])
    for tracer in (state.T, state.S):
        tendency = np.asarray(_conv_flux_tendency(tracer, column, params.kappa_conv, params, iface_gate=gate))
        assert np.max(np.abs(np.sum(tendency * np.asarray(params.dz_node), axis=-1))) < 1e-12
        if tracer is state.T:
            assert tendency[0, 0, 0] == pytest.approx(expected if salinity_offset > 0. else 0., abs=1e-13)


def test_branch_local_ad_does_not_capture_a_neutral_gate_jump(neutral_column):
    state, params = neutral_column

    def response(offset):
        selected = state._replace(S=state.S.at[..., 0].add(offset))
        column, gate = _convective_mask(selected, params)
        return _conv_flux_tendency(selected.T, column, params.kappa_conv, params, iface_gate=gate)[0, 0, 0]

    offset = jnp.asarray(0., dtype=jnp.float64)
    tangent = jax.jvp(response, (offset,), (jnp.asarray(1., dtype=jnp.float64),))[1]
    assert float(tangent) == 0.
    scale = 1e-8
    central_difference = (response(offset + scale) - response(offset - scale)) / (2. * scale)
    expected = -params.kappa_conv / (params.dz_surface * np.asarray(params.dz_iface).ravel()[0] * 2. * scale)
    assert float(central_difference) == pytest.approx(expected, rel=1e-13)
    assert abs(float(central_difference - tangent)) > 1e4
