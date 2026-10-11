"""Physical affine truth, face bounds, wet masks and conservative flux controls."""

from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest

from tests.fd.test_tracer_product_flux import channel
from zhenmode.model.solver.dynamics.transport import _advection_scalar, _vertical_transport_iface
from zhenmode.model.solver.numerics.vertical import _vertical_tracer_face_values


def test_physical_affine_profile_has_exact_midpoint_values_on_nonuniform_nodes():
    depths = np.array([0., 2., 13., 46., 100.])
    tracer = jnp.array(15 - .02 * depths)[None, None]
    p = SimpleNamespace(dz_iface=jnp.array(np.diff(depths))[None, None], wet_mask_z=jnp.ones_like(tracer))
    left, right = _vertical_tracer_face_values(tracer, p)
    expected = 15 - .02 * (depths[:-1] + depths[1:]) / 2
    np.testing.assert_allclose(np.asarray(left)[0, 0], expected, atol=3e-15)
    np.testing.assert_allclose(np.asarray(right)[0, 0], expected, atol=3e-15)


@pytest.mark.parametrize('values', ([0., 0., 1., 1., 1.], [0., 1., 0., -1., 0.], [5., 4., 1., 2., 6.]))
def test_face_values_stay_within_adjacent_bounds(values):
    field = jnp.array(values)[None, None]
    p = SimpleNamespace(dz_iface=jnp.array([1., 3., 2., 7.])[None, None], wet_mask_z=jnp.ones_like(field))
    lo, hi = np.minimum(values[:-1], values[1:]), np.maximum(values[:-1], values[1:])
    for face in _vertical_tracer_face_values(field, p):
        assert np.all(np.asarray(face)[0, 0] >= lo)
        assert np.all(np.asarray(face)[0, 0] <= hi)


def test_dry_sentinel_does_not_change_last_wet_face():
    a = jnp.array([15., 14., 13., -10000.])[None, None]
    p = SimpleNamespace(dz_iface=jnp.ones((1, 1, 3)), wet_mask_z=jnp.array([1., 1., 1., 0.])[None, None])
    first = _vertical_tracer_face_values(a, p)
    second = _vertical_tracer_face_values(a.at[..., -1].set(10000), p)
    for left, right in zip(first, second, strict=True):
        np.testing.assert_array_equal(np.asarray(left)[..., :2], np.asarray(right)[..., :2])


def test_corrected_tracer_stays_constant_and_inventory_telescopes():
    _, (_, _, _, p, _) = channel(8, vertical_tracer_scheme='muscl_nodal_v1')
    rng = np.random.default_rng(121026)
    u, v = [jnp.array(rng.normal(size=(8, 8, 4)) * .01) for _ in range(2)]
    Fz = _vertical_transport_iface(u, v, p)
    constant = jnp.full_like(u, 15.)
    np.testing.assert_allclose(_advection_scalar(constant, u, v, Fz, p), 0, atol=2e-18)
    field = jnp.array(15 + rng.normal(size=u.shape))
    tendency, boundary = _advection_scalar(field, u, v, Fz, p, return_boundary=True)
    assert abs(float(jnp.sum(tendency * p.dz_node)) - float(jnp.sum(boundary))) < 1e-15


def test_factory_rejects_unsupported_vertical_geometry():
    with pytest.raises(ValueError, match='requires nodal_dual'):
        channel(vertical_tracer_scheme='muscl_nodal_v1', column_geometry='legacy', external_mode_scheme='forward_backward')
    with pytest.raises(ValueError, match='unknown vertical'):
        channel(vertical_tracer_scheme='unknown')
