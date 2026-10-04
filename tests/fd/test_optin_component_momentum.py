"""Existing modal negative control for the FD component half-step; not a climate gate."""
import jax.numpy as jnp
import numpy as np
import pytest

from tests.support.fd.momentum_metric import _controlled_factory, _numpy_operators
from tests.support.fd.reference_geometry import WIDTHS
from zhenmode.model.config.definitions import RHO_0
from zhenmode.model.dynamics.processes import _linear_half_step


def test_component_half_step_retains_declared_modal_negative_control():
    _, params, state = _controlled_factory()
    assert params.nu_nsub == 2 and params.n_subcyc == 24
    _, vertical, _ = _numpy_operators(params)
    column = vertical[:4, :4]
    square_root = np.sqrt(WIDTHS)
    eigenvalues, eigenvectors = np.linalg.eigh(column * square_root[:, None] / square_root[None, :])
    vertical_mode = eigenvectors[:, 0] / square_root
    vertical_mode /= np.max(np.abs(vertical_mode))
    mode = (-1.) ** np.arange(8)[:, None, None] * np.cos(7. * np.pi * (np.arange(8) + .5) / 8.)[None, :, None] * vertical_mode[None, None, :]
    weights = np.asarray(params.dx_2d)[..., None] * params.dy * WIDTHS
    for amplitude in (1e-8, 1e-10, 1e-12):
        initial = state._replace(u=jnp.asarray(mode * amplitude))
        old_half = _linear_half_step(initial, params, params.dt / 2.)
        before = .5 * RHO_0 * np.sum(weights * np.asarray(initial.u) ** 2)
        old_energy = .5 * RHO_0 * np.sum(weights * np.asarray(old_half.u) ** 2)
        assert old_energy / before == pytest.approx(3.522760274093153, rel=1e-12)
