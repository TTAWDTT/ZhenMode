"""Shared analytic-profile controls separating FD, source P1 and remap P1."""

from tests.support.paths import REPOSITORY_ROOT

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = REPOSITORY_ROOT
spec = importlib.util.spec_from_file_location(
    "null_top", ROOT / "research/experiments/local_top_bridge/component.py"
)
c = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = c
spec.loader.exec_module(c)
RHO, G, ALPHA, BETA, TREF, SREF = 1025.0, 9.81, 2e-4, 7.6e-4, 15.0, 35.0


def stocks(z, coef):
    primitive = np.polynomial.polynomial.polyint(coef)
    temp = np.polynomial.polynomial.polyval(z[:-1], primitive) - np.polynomial.polynomial.polyval(
        z[1:], primitive
    )
    h = -np.diff(z)
    return np.array([temp, h * SREF, np.zeros_like(h), np.zeros_like(h)]).T


def remap(z, n, target):
    return np.array([c.integrate(z, n, target[k + 1], target[k]) for k in range(len(target) - 1)])


def pressure(z, n, depth):
    amount = c.integrate(z, n, depth, z[0])
    length = z[0] - depth
    return RHO * G * z[0] + RHO * G * (
        -ALPHA * (amount[0] - TREF * length) + BETA * (amount[1] - SREF * length)
    )


def exact(eta, coef, depth):
    adjusted = np.array(coef, dtype=float)
    adjusted[0] -= TREF
    p = np.polynomial.polynomial.polyint(adjusted)
    return RHO * G * eta - RHO * G * ALPHA * (
        np.polynomial.polynomial.polyval(eta, p) - np.polynomial.polynomial.polyval(depth, p)
    )


def fd_pressure(nodes, temp, salt, eta, depth, wet=None):
    """Original node EOS -> masked trapezoid -> physical-depth interpolation.

    This independent oracle has NO analytic antiderivative shortcut.
    """
    d = -np.asarray(nodes)
    if wet is None:
        wet = np.ones_like(d)
    anomaly = RHO * (-ALPHA * (np.asarray(temp) - TREF) + BETA * (np.asarray(salt) - SREF)) * wet
    nodal = np.zeros_like(d)
    for k in range(1, len(d)):
        nodal[k] = nodal[k - 1] + G * 0.5 * (anomaly[k - 1] + anomaly[k]) * (d[k] - d[k - 1])
    nodal += RHO * G * eta
    if depth > nodes[0] or depth < nodes[-1]:
        raise ValueError("FD extrapolation unsupported")
    return float(np.interp(-depth, d, nodal))


def budget(z, n):
    u = np.finfo(float).eps / 2
    gamma = 2048 * u / (1 - 2048 * u)
    h = -np.diff(z)
    scale = abs(RHO * G * z[0]) + RHO * G * (
        ALPHA * (abs(n[:, 0]).sum() + abs(TREF) * h.sum())
        + BETA * (abs(n[:, 1]).sum() + abs(SREF) * h.sum())
    )
    return gamma * scale


@pytest.mark.parametrize("coef", [[15.2], [20.0, 0.1]])
@pytest.mark.parametrize("eta", [0.0, -2.0, -2.4914792546513693])
def test_constant_linear_shared_profile(coef, eta):
    z = np.array([eta, -2.5, -10.0, -22.5])
    n = stocks(z, coef)
    target = np.array([eta, eta - 0.2 * (eta + 22.5), eta - 0.7 * (eta + 22.5), -22.5])
    new = remap(z, n, target)
    bound = budget(z, n) + budget(target, new)
    for d in np.linspace(-22.5, eta, 81):
        assert abs(pressure(z, n, d) - exact(eta, coef, d)) <= bound
        assert abs(pressure(target, new, d) - exact(eta, coef, d)) <= bound
    assert abs(pressure(target, new, -22.5) - pressure(z, n, -22.5)) <= bound


@pytest.mark.parametrize("coef", [[15.2], [20.0, 0.1]])
def test_different_eta_compare_physical_oracle_difference(coef):
    values = []
    for eta in [-0.1, -2.49]:
        z = np.array([eta, -2.5, -10.0, -22.5])
        n = stocks(z, coef)
        values.append((z, n))
    for depth in np.linspace(-22.5, -2.49, 41):
        measured = pressure(*values[0], depth) - pressure(*values[1], depth)
        oracle = exact(-0.1, coef, depth) - exact(-2.49, coef, depth)
        assert abs(measured - oracle) <= sum(budget(z, n) for z, n in values)
    assert abs(measured) > 1  # distinct integration origins are physically distinct


def curvature_scan():
    coef = [20.0, 0.1, 0.001]
    eta = -2.0
    rows = []
    for count in [8, 16, 32, 64]:
        z = np.linspace(eta, -22.5, count + 1)
        n = stocks(z, coef)
        target = z.copy()
        target[1:-1:2] += 0.2 * (eta + 22.5) / count
        new = remap(z, n, target)
        depths = np.linspace(-22.5, eta, 201)
        errors = [
            max(abs(pressure(grid, stock, d) - exact(eta, coef, d)) for d in depths)
            for grid, stock in [(z, n), (target, new)]
        ]
        base = pressure(target, new, -22.5) - pressure(z, n, -22.5)
        assert abs(base) <= budget(z, n) + budget(target, new)
        rows.append(
            dict(
                layers=count,
                max_source_remap_error_Pa=errors,
                remap_minus_source_base_Pa=float(base),
            )
        )
    return rows


def test_curvature_refinement_and_conservative_base():
    rows = curvature_scan()
    for before, after in zip(rows[-3:-1], rows[-2:]):
        assert (
            min(
                a / b
                for a, b in zip(
                    before["max_source_remap_error_Pa"], after["max_source_remap_error_Pa"]
                )
            )
            >= 3
        )


def test_node_as_mean_explanation_separate_from_remap():
    eta = -2.0
    z = np.array([eta, -2.5, -10.0, -22.5])
    coef = [20.0, 0.1]
    nodes = np.array([0.0, -5.0, -15.0])
    h = -np.diff(z)
    sampled = stocks(z, coef)
    node_mean = sampled.copy()
    node_mean[:, 0] = h * np.polynomial.polynomial.polyval(nodes, coef)
    target = np.array([eta, -4.0, -12.0, -22.5])
    new = remap(z, node_mean, target)
    delta = pressure(z, node_mean, -22.5) - exact(eta, coef, -22.5)
    predicted = -RHO * G * ALPHA * (node_mean[:, 0].sum() - sampled[:, 0].sum())
    assert abs(delta - predicted) <= budget(z, node_mean) + budget(z, sampled)
    assert abs(delta) > 1
    assert abs(pressure(target, new, -22.5) - pressure(z, node_mean, -22.5)) <= budget(
        z, node_mean
    ) + budget(target, new)
    full_nodes = np.array([0.0, -5.0, -15.0, -30.0])
    fd = fd_pressure(
        full_nodes, np.polynomial.polynomial.polyval(full_nodes, coef), np.full(4, SREF), eta, -22.5
    )
    analytic_origin_zero = RHO * G * eta + exact(0, coef, -22.5)
    expected_interpolation = RHO * G * ALPHA * 0.1 * 15**2 / 8
    assert fd - analytic_origin_zero == pytest.approx(expected_interpolation, abs=1e-10)
    assert expected_interpolation == pytest.approx(5.656078125)
    assert abs(fd - exact(eta, coef, -22.5)) > 1


@pytest.mark.parametrize("coef", [[15.2], [20.0, 0.1], [20.0, 0.1, 0.001]])
@pytest.mark.parametrize("partial", [False, True])
def test_fd_oracle_matches_original_node_operator(coef, partial):
    import os
    from types import SimpleNamespace

    os.environ["JAX_PLATFORMS"] = "cpu"
    import jax
    import jax.numpy as jnp

    from ocean_solver.dynamics.pressure import _compute_hydrostatic_pressure

    jax.config.update("jax_enable_x64", True)
    nodes = np.array([0.0, -5.0, -15.0, -30.0])
    temp = np.polynomial.polynomial.polyval(nodes, coef)
    salt = np.full(4, SREF)
    wet = np.array([1.0, 1.0, 1.0, 0.0]) if partial else np.ones(4)
    params = SimpleNamespace(
        T_ref=TREF,
        S_ref=SREF,
        dz_3d=jnp.asarray(-np.diff(nodes)),
        wet_mask_z=jnp.asarray(wet[None, None, :]),
    )
    state = SimpleNamespace(
        T=jnp.asarray(temp[None, None, :]),
        S=jnp.asarray(salt[None, None, :]),
        eta=jnp.asarray([[-2.0]]),
    )
    actual = np.asarray(_compute_hydrostatic_pressure(state, params))[0, 0]
    for depth in [-5.0, -15.0, -22.5]:
        assert fd_pressure(nodes, temp, salt, -2.0, depth, wet) == pytest.approx(
            np.interp(-depth, -nodes, actual), abs=1e-10
        )
