"""Independent inertial ownership and oscillator-energy gates for one opt-in module."""

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tests.support.grid import all_wet_grid
from zhenmode.model.config import PhysicsConfig
from zhenmode.model.solver.factory import make_solver_global


def core(scheme="forward_backward", f=1e-4, dt=100.0):
    grid = all_wet_grid(nx=8, ny=32, nz=4)
    grid = replace(
        grid,
        z=np.array([0.0, -5.0, -20.0, -50.0]),
        dz=np.array([5.0, 15.0, 30.0]),
        depth=np.full((8, 32), 50.0),
        dx_2d=np.full((8, 32), 1e5),
        dy=1e5,
        cos_lat=np.ones(32),
        f=np.full((8, 32), f),
    )
    physics = PhysicsConfig(
        **{
            name: 0.0
            for name in (
                "nu_h",
                "nu_v",
                "nu_bi",
                "kappa_h",
                "kappa_v",
                "kappa_bi",
                "kappa_conv",
                "kappa_gm",
                "kappa_redi",
                "r_bot",
                "cd",
            )
        }
    )
    return make_solver_global(
        grid,
        physics,
        dt=dt,
        return_params=True,
        column_geometry="nodal_dual_v1",
        conservative_kv=True,
        localize_conv=True,
        polar_cap_rows=0,
        polar_cap_taper=0,
        external_mode_scheme=scheme,
    )


@pytest.mark.parametrize("scheme", ["forward_backward", "symmetric"])
def test_complete_step_mean_rotation_has_declared_single_or_legacy_double_owner(scheme):
    step, initialize, _, p, _ = core(scheme)
    shear = np.array([1.0, 0.5, -0.5, -1.0])
    shear -= np.sum(shear * np.asarray(p.dz_node)) / 50
    state = initialize()._replace(
        u=jnp.broadcast_to(jnp.asarray(0.001 + 0.0003 * shear), (8, 32, 4))
    )
    result = step(state)
    jax.block_until_ready(result)
    q = np.asarray(result.u)[0, 16] + 1j * np.asarray(result.v)[0, 16]
    mean = np.sum(q * np.asarray(p.dz_node)) / 50
    angle = float(p.dt) * 1e-4
    if scheme == "forward_backward":
        # Source-level ownership defect is retained as an explicit legacy control.
        expected = 0.001 * np.exp(-1j * angle) / (1 + 1j * angle / 2) ** 2
    else:
        expected = 0.001 * np.exp(-4j * np.arctan(angle / 4))
        assert abs(np.angle(mean) + angle) < angle**3 / 40
        assert abs(abs(mean) - 0.001) < 2e-15
    np.testing.assert_allclose(mean, expected, rtol=0, atol=2e-15)
    # Vertical shear has exactly one full-step rotation in both paths.
    np.testing.assert_allclose(q - mean, 0.0003 * shear * np.exp(-1j * angle), rtol=0, atol=2e-15)


def test_candidate_factory_rejects_unsupported_combinations():
    with pytest.raises(ValueError, match="unknown external"):
        core("unknown")


def test_external_pressure_block_has_quadratic_energy_ripple():
    from tests.support.fd.reference_geometry import _fixture
    from zhenmode.model.solver.dynamics.barotropic import _free_surface_step_fd

    _, (_, initialize, _, base, _) = _fixture(mode_split=False, external_mode_scheme="symmetric")
    base = base._replace(
        dx_2d=jnp.full_like(base.dx_2d, 1000.0),
        dy=1000.0,
        inv_dx=jnp.full_like(base.inv_dx, 0.001),
        inv_dy=0.001,
        cos_lat=jnp.ones_like(base.cos_lat),
    )
    eta = jnp.broadcast_to(
        jnp.asarray(1e-7 * np.cos(2 * np.pi * np.arange(8) / 8))[:, None], (8, 8)
    )
    state = initialize()._replace(eta=eta)

    def energy(eta, u):
        return jnp.mean(eta**2 + 50.0 / 9.81 * u[..., 0] ** 2)

    reference = float(energy(state.eta, state.u))
    errors = []
    for dt in (4.0, 2.0, 1.0):

        def evolve(values):
            eta, u, v = values
            return _free_surface_step_fd(eta, u, v, base, dt_half=dt)

        step = jax.jit(evolve)
        current = (state.eta, state.u, state.v)
        maximum = 0.0
        for _ in range(int(64 / dt)):
            current = step(current)
            maximum = max(maximum, abs(float(energy(current[0], current[1])) / reference - 1))
        errors.append(maximum)
    assert all(3.5 < a / b < 4.5 for a, b in zip(errors[:-1], errors[1:], strict=True)), errors
