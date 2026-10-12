"""Constrained endpoint flux, inventory, envelopes and unforced controls."""

from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest

from zhenmode.model.solver.numerics.vertical import (
    _conservative_endpoint_flux,
    _vertical_tracer_face_values,
)


def column(values, mass, dt=.01):
    q = jnp.array(values)[None, None]
    n = q.shape[-1]
    widths = np.ones(n)
    widths[[0, -1]] = .5
    p = SimpleNamespace(dz_node=jnp.array(widths)[None, None], dz_iface=jnp.ones((1, 1, n-1)),
                        wet_mask_z=jnp.ones_like(q), dt=dt, adv_nsub=1)
    F = jnp.array(mass)[None, None]
    up, down = _vertical_tracer_face_values(q, p)
    internal = F[..., 1:-1] * jnp.where(F[..., 1:-1] > 0., up, down)
    flux = jnp.concatenate((F[..., :1] * q[..., :1], internal, jnp.zeros_like(F[..., :1])), axis=-1)
    horizontal = q * (F[..., :-1] - F[..., 1:]) / p.dz_node
    return q, F, flux, horizontal, p


@pytest.mark.parametrize('sign', (-1., 1.))
def test_closed_affine_background_has_zero_endpoint_material_tendency(sign):
    mass = np.r_[0., sign * np.sin(np.pi * (np.arange(5) + .5) / 5), 0.]
    q, F, low_flux, horizontal, p = column(15 - .02 * np.arange(6), mass)
    corrected = _conservative_endpoint_flux(q, F, low_flux, horizontal, p)
    rate = -horizontal - (corrected[..., 1:] - corrected[..., :-1]) / p.dz_node
    np.testing.assert_allclose(np.asarray(rate)[..., [0, -1]], 0, atol=1e-14)
    np.testing.assert_allclose(jnp.sum((corrected[..., 1:] - corrected[..., :-1])
                                     - (low_flux[..., 1:] - low_flux[..., :-1])), 0, atol=1e-14)


def test_zero_velocity_leaves_flux_unchanged():
    q, F, flux, horizontal, p = column([0, 2, 1, 4, 0], np.zeros(6))
    np.testing.assert_array_equal(_conservative_endpoint_flux(q, F, flux, horizontal, p), flux)


def test_surface_characteristic_allowance_does_not_cancel_affine_inflow():
    q, F, flux, horizontal, p = column([5., 4., 3., 2., 1., 0.], [.1, .09, .06, .04, .02, .01, 0.])
    new = _conservative_endpoint_flux(q, F, flux, horizontal, p)
    rate = -horizontal - (new[..., 1:] - new[..., :-1]) / p.dz_node
    np.testing.assert_allclose(rate[..., 0], .1, atol=1e-13)


def test_nonnegative_zero_surface_does_not_admit_negative_extrapolated_inflow():
    q, F, flux, horizontal, p = column([0., 1., 2., 3., 4.], [.1, .09, .06, .04, .02, 0.])
    new = _conservative_endpoint_flux(q, F, flux, horizontal, p, minimum=0.)
    updated = q - p.dt * (horizontal + (new[..., 1:] - new[..., :-1]) / p.dz_node)
    assert float(updated[..., 0][0, 0]) >= -1e-14


def test_temperature_transport_is_invariant_under_a_change_of_zero_point():
    q, F, flux, horizontal, p = column([0., 1., 2., 3., 4.], [.1, .09, .06, .04, .02, 0.])
    first = _conservative_endpoint_flux(q, F, flux, horizontal, p)
    shifted, Fs, fluxs, hs, ps = column([273.15, 274.15, 275.15, 276.15, 277.15], [.1, .09, .06, .04, .02, 0.])
    second = _conservative_endpoint_flux(shifted, Fs, fluxs, hs, ps)
    rate1 = -horizontal - (first[..., 1:] - first[..., :-1]) / p.dz_node
    rate2 = -hs - (second[..., 1:] - second[..., :-1]) / ps.dz_node
    np.testing.assert_allclose(rate1, rate2, atol=1e-12)


def test_stagnant_internal_faces_do_not_transmit_correction_or_change_disconnected_cells():
    q, F, flux, horizontal, p = column([5., 4., 3., 2., 1., 0.], [0., -.1, 0., 0., 0., 0., 0.])
    new = _conservative_endpoint_flux(q, F, flux, horizontal, p)
    np.testing.assert_array_equal(np.asarray(new)[..., 2:], np.asarray(flux)[..., 2:])
    np.testing.assert_array_equal(np.asarray(new)[..., 0], np.asarray(flux)[..., 0])


def test_constant_tracer_stays_constant_with_surface_mass_transport():
    q, F, flux, horizontal, p = column([35] * 5, [.01, .02, -.03, .01, .02, 0.])
    new = _conservative_endpoint_flux(q, F, flux, horizontal, p)
    np.testing.assert_allclose(-horizontal - (new[..., 1:] - new[..., :-1]) / p.dz_node, 0, atol=1e-14)


def test_column_blend_conserves_and_cannot_worsen_uncorrected_euler_envelope():
    rng = np.random.default_rng(131026)
    for _ in range(12):
        values = rng.uniform(0, 2, 9)
        mass = np.r_[0., rng.uniform(-.5, .5, 8), 0.]
        q, F, flux, horizontal, p = column(values, mass, dt=.2)
        new = _conservative_endpoint_flux(q, F, flux, horizontal, p)
        low = q - p.dt * (horizontal + (flux[..., 1:] - flux[..., :-1]) / p.dz_node)
        updated = q - p.dt * (horizontal + (new[..., 1:] - new[..., :-1]) / p.dz_node)
        lower, upper = jnp.minimum(q.min(), low), jnp.maximum(q.max(), low)
        assert np.all(np.asarray(updated) >= np.asarray(lower) - 1e-14)
        assert np.all(np.asarray(updated) <= np.asarray(upper) + 1e-14)
        np.testing.assert_allclose(jnp.sum((updated - low) * p.dz_node), 0, atol=1e-14)
