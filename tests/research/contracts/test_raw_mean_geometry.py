"""Independent volume integration of the new manufactured mean authority."""
from dataclasses import replace

import numpy as np
import pytest

from research.experiments.material_top_band.moving_raw_cases import manufactured_mean_case
from research.experiments.material_top_band.raw_mean_geometry import RawMeanGeometry


def direct_matrices(profile, distance=2., length=1.5):
    state, rho0 = profile.state, profile.eos.rho0
    nodes, weights = np.polynomial.legendre.leggauss(7)
    Q, W = np.zeros((28, 28)), np.zeros((28, 28))
    cuts = np.unique(state.interfaces)
    for lower, upper in zip(cuts[:-1], cuts[1:]):
        z = .5 * (lower + upper)
        owners = [int(np.flatnonzero((state.interfaces[s, 1:] < z) & (z < state.interfaces[s, :-1]))[0]) for s in range(2)]
        for side in range(2):
            for node, weight in zip(nodes, weights):
                x = distance * (side / 2. + (node + 1.) / 4.)
                basis = np.zeros(28)
                basis[owners[0]], basis[14 + owners[1]] = 1. - x / distance, x / distance
                volume = length * (upper - lower) * distance * weight / 4.
                Q[14 * side + owners[side]] += basis * volume
                W += rho0 * volume * np.outer(basis, basis)
    Q /= (distance * length / 2. * state.h.ravel())[:, None]
    inverse = np.linalg.inv(Q)
    return Q, inverse.T @ W @ inverse


def moved(profile, increment):
    z, h = profile.state.interfaces.copy(), profile.state.h.copy()
    z[:, 0] += increment
    h[:, 0] += increment
    return replace(profile, state=replace(profile.state, eta=z[:, 0].copy(), interfaces=z, h=h))


def test_actual_mean_map_and_metric_against_volume_quadrature():
    profile, _ = manufactured_mean_case()
    geometry = RawMeanGeometry(profile, authority='manufactured_half_prism_raw_means')
    Q, M = direct_matrices(profile)
    np.testing.assert_allclose(geometry.Q, Q, rtol=0., atol=3e-14)
    np.testing.assert_allclose(geometry.M, M, rtol=4e-14, atol=4e-12)
    np.testing.assert_allclose(geometry.Q @ np.ones(28), 1., rtol=0., atol=3e-14)
    np.testing.assert_allclose(geometry.M @ np.ones(28), geometry.D.diagonal(), rtol=0., atol=4e-12)
    root = np.diag(1. / np.sqrt(geometry.D.diagonal()))
    eigenvalues = np.linalg.eigvalsh(root @ geometry.M @ root)
    assert eigenvalues.min() >= 1. - 2e-13
    assert eigenvalues.max() <= 4. / 3. + 2e-13


def test_actual_affine_covariance_is_present():
    profile, spec = manufactured_mean_case()
    geometry = RawMeanGeometry(profile, authority='manufactured_half_prism_raw_means')
    a = geometry.means
    covariance = .5 * np.sum(a * ((geometry.M - geometry.D) @ a))
    expected = profile.eos.rho0 * spec.length * (spec.eta - spec.bottom) * spec.distance**3 * spec.alpha**2 / 96.
    np.testing.assert_allclose(covariance, expected, rtol=2e-12, atol=0.)
    endpoints = geometry.inverse @ a
    np.testing.assert_allclose(endpoints[:14, 0], spec.U, rtol=0., atol=1e-15)
    np.testing.assert_allclose(endpoints[14:, 0], spec.U + spec.alpha * spec.distance, rtol=0., atol=1e-15)
    assert abs(.5 * np.sum(a * (geometry.endpoint_metric @ a)) - .5 * np.sum(a * (geometry.M @ a))) > .1


@pytest.mark.parametrize('epsilon', [1e-6, 5e-7])
def test_frame_derivatives_with_predetermined_fd_envelopes(epsilon):
    profile, _ = manufactured_mean_case()
    geometry = RawMeanGeometry(profile, authority='manufactured_half_prism_raw_means')
    direction = geometry.direction(1.)
    matrices = []
    for increment in (epsilon, -epsilon):
        changed = moved(profile, increment)
        Q, M = direct_matrices(changed)
        D = profile.eos.rho0 * geometry.area * changed.state.h.ravel()
        matrices.append(dict(Q=Q, M=M, R=M / D[None, :]))
    for name in ('Q', 'M', 'R'):
        derivative = (matrices[0][name] - matrices[1][name]) / (2. * epsilon)
        envelope = geometry.fd_envelope(name, epsilon)
        assert np.linalg.norm(derivative - direction[name], 2) <= envelope < 1.
    omitted = geometry.inverse.T @ direction['endpoint_metric'] @ geometry.inverse
    assert np.linalg.norm(direction['M'] - omitted,2) > 100. * geometry.fd_envelope('M', epsilon)
    for term in (direction['inverse'].T @ geometry.endpoint_metric @ geometry.inverse,
                 geometry.inverse.T @ geometry.endpoint_metric @ direction['inverse'],
                 geometry.inverse.T @ direction['endpoint_metric'] @ geometry.inverse):
        assert np.linalg.norm(term,2) > 100. * geometry.fd_envelope('M',epsilon)
    assert np.linalg.norm(direction['R'],2) > 100. * geometry.fd_envelope('R', epsilon)


@pytest.mark.parametrize('authority', [None, 'original', 'endpoint'])
def test_explicit_new_authority_required(authority):
    profile, _ = manufactured_mean_case()
    with pytest.raises(ValueError, match='authority'):
        RawMeanGeometry(profile, authority=authority)
