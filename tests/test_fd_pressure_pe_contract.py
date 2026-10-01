"""Instantaneous actual-operator counterexamples, never complete-step energy gates."""
import importlib.util
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from config import G_EARTH, RHO_0
from jax_solver_global import (
    _advection_scalar,
    _compute_hydrostatic_pressure,
    _gradient_conservative_3d,
    _layer_face_transports,
    _vertical_transport_iface,
)

spec = importlib.util.spec_from_file_location(
    'fd_pressure_pe_contract', Path(__file__).parents[1]
    / 'scripts/diagnostics/fd_pressure_pe_contract.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_stationary_uniform_stable_profile_has_no_conversion():
    r = module.cases()['rest_stable']
    assert r['pressure_work_W'] == r['donor_PE_rate_W'] == 0.


@pytest.mark.parametrize('case', ['circulation_uniform_stable', 'layered_horizontal_exchange',
                                 'moving_top_exchange'])
def test_centered_oracle_pairs_but_actual_donor_has_explained_defect(case):
    r = module.cases()[case]
    assert abs(r['centered_pairing_residual_W']) <= r['arithmetic_bound_W']
    assert abs(r['defect_reconstruction_residual_W']) <= r['arithmetic_bound_W']
    assert r['donor_pairing_residual_W'] > 1000*r['arithmetic_bound_W']
    assert r['north_exterior_max_m2_s'] == r['bed_exterior_max_m_s'] == 0.
    assert abs(r['density_stock_rate_kg_s']) < 1e-10


def test_reverse_circulation_reverses_exchange_but_not_stable_donor_mixing():
    p, area, depth = module.fixture()
    phase = 2*np.pi*np.arange(8)[:, None, None]/8
    u = np.broadcast_to(.01*np.sin(phase)*[1., -.25, 0., 0.], (8, 3, 4))
    rho = np.broadcast_to([0., .2, .8, 2.], u.shape) + .1*np.cos(phase)*[1., 2., 0., 0.]
    f = module.evaluate(rho, u, np.zeros_like(u), p, area, depth)
    b = module.evaluate(rho, -u, np.zeros_like(u), p, area, depth)
    assert b['pressure_work_W'] == pytest.approx(-f['pressure_work_W'], abs=1e-9)
    assert b['donor_flux_defect_W'] == pytest.approx(f['donor_flux_defect_W'], abs=1e-9)


def test_unstable_profile_changes_defect_sign_not_a_universal_PE_source():
    p, area, depth = module.fixture()
    phase = 2*np.pi*np.arange(8)[:, None, None]/8
    u = np.broadcast_to(.01*np.sin(phase)*[1., -.25, 0., 0.], (8, 3, 4))
    rho = np.broadcast_to([0., -.2, -.8, -2.], u.shape)
    r = module.evaluate(rho, u, np.zeros_like(u), p, area, depth)
    assert r['pressure_work_W'] == 0.
    assert r['donor_flux_defect_W'] < -1000.


def test_moving_capacity_requires_EOS_reference_terms():
    p, _, _ = module.fixture()
    phase = 2*np.pi*np.arange(8)[:, None, None]/8
    u = jnp.asarray(np.broadcast_to(.01*np.sin(phase), (8, 3, 4)))
    v = jnp.zeros_like(u)
    faces = _layer_face_transports(u, v, p)
    vertical = _vertical_transport_iface(u, v, p, face_transport=faces)
    hdot = np.zeros(u.shape)
    hdot[..., 0] = -np.asarray(vertical)[..., 0]

    def stock_rate(q):
        q = jnp.full_like(u, q)
        tendency, top = _advection_scalar(q, u, v, vertical, p,
                                         return_boundary=True, face_transport=faces)
        rate = np.asarray(p.dz_node*tendency).copy()
        rate[..., 0] -= np.asarray(top)
        return rate

    rt, rs = stock_rate(p.T_ref), stock_rate(p.S_ref)
    corrected = module.density_stock_rate(rt, rs, hdot, p)
    naive = module.density_stock_rate(rt, rs, np.zeros_like(hdot), p)
    np.testing.assert_allclose(corrected, 0., rtol=0, atol=1e-15)
    assert np.max(np.abs(naive)) > .001  # zero rho' cannot acquire this source


def test_constant_pressure_reference_does_not_change_force():
    from types import SimpleNamespace
    p, _, _ = module.fixture()
    state = SimpleNamespace(T=jnp.full((8, 3, 4), 14.), S=jnp.full((8, 3, 4), 35.),
                            eta=jnp.zeros((8, 3)))
    pressure = _compute_hydrostatic_pressure(state, p)
    before = _gradient_conservative_3d(pressure, p)
    after = _gradient_conservative_3d(pressure + RHO_0*G_EARTH, p)
    for a, b in zip(before, after, strict=True):
        np.testing.assert_array_equal(a, b)


def test_partial_bed_is_explicitly_outside_oracle_scope():
    p, area, depth = module.fixture()
    p.wet_mask_z = p.wet_mask_z.at[1, 1, -1].set(0.)
    zero = np.zeros((8, 3, 4))
    with pytest.raises(ValueError, match='flat-bed'):
        module.evaluate(zero, zero, zero, p, area, depth)


def test_top_node_moment_is_not_moving_control_geometric_moment():
    # Uniform anomaly in a top control extending z=-h0 to z=eta.
    area, anomaly, h0, eta, eta_rate = 100., 2., 2.5, .4, .01
    node_z = 0.
    nodal_rate = G_EARTH * area * node_z * anomaly * eta_rate
    geometric_rate = G_EARTH * area * anomaly * eta * eta_rate
    geometric_stock = .5 * G_EARTH * area * anomaly * (eta**2-h0**2)
    assert nodal_rate == 0.
    assert geometric_rate == pytest.approx(7.848)
    assert geometric_stock < 0.


def test_spherical_meridional_closed_wall_pairing():
    p, area, depth = module.fixture()
    v = np.broadcast_to(np.array([0., .01, 0.])[None, :, None]
                        * np.array([1., -.25, 0., 0.]), (8, 3, 4))
    rho = (np.broadcast_to([0., .2, .8, 2.], v.shape)
           + np.array([0., .1, .3])[None, :, None]*np.array([1., 2., 0., 0.]))
    r = module.evaluate(rho, np.zeros_like(v), v, p, area, depth)
    assert abs(r['centered_pairing_residual_W']) <= r['arithmetic_bound_W']
    assert abs(r['defect_reconstruction_residual_W']) <= r['arithmetic_bound_W']
    assert r['north_exterior_max_m2_s'] == 0.
