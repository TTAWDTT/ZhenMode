"""Memory-only scan arithmetic and complete stock checks, not smooth physics."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import zhenmode_research.candidates.material.solver as material_top
from tests.support.material.top import _material_fixture, _state
from zhenmode_research.candidates.material.solver import (
    _checkpointed_material_scan,
    make_material_top_step,
)


@pytest.mark.parametrize("maximum,active", [(1, 1), (7, 3), (8, 8), (13, 11),
                                          (128, 52), (13, 0)])
def test_blocks_preserve_flat_order_forward_tangent_and_adjoint(maximum, active):
    initial = jnp.linspace(.1, .9, 11, dtype=jnp.float64)

    def response(control, blocked):
        def advance(carry, index):
            updated = .97 * carry + control * jnp.sin(.1 * index)
            return jnp.where(index < active, updated, carry), None

        if blocked:
            return _checkpointed_material_scan(advance, initial, maximum)
        return jax.lax.scan(advance, initial, jnp.arange(maximum))[0]

    def flat(control):
        return response(control, False)

    def blocked(control):
        return response(control, True)
    control = jnp.asarray(.4, dtype=jnp.float64)
    direction = jnp.asarray(.7, dtype=jnp.float64)
    assert np.asarray(flat(control)).tobytes() == np.asarray(blocked(control)).tobytes()
    tangent_flat = jax.jvp(flat, (control,), (direction,))[1]
    tangent_blocked = jax.jvp(blocked, (control,), (direction,))[1]
    np.testing.assert_allclose(tangent_flat, tangent_blocked, rtol=1e-13, atol=1e-13)
    adjoint_flat = jax.grad(lambda selected: jnp.sum(flat(selected)))(control)
    adjoint_blocked = jax.grad(lambda selected: jnp.sum(blocked(selected)))(control)
    np.testing.assert_allclose(adjoint_flat, adjoint_blocked, rtol=1e-13, atol=1e-13)


def test_padded_slots_never_apply_an_update():
    maximum = 13

    def advance(carry, index):
        return carry + jnp.where(index < maximum, 1., jnp.nan), None

    actual = _checkpointed_material_scan(advance, jnp.asarray(0., dtype=jnp.float64), maximum)
    assert float(actual) == maximum


@pytest.mark.parametrize("source", ["convection", "bulk"])
def test_complete_thin_step_fields_checks_and_budget_match_flat(monkeypatch, source):
    _, (_, initialize, _, params, _) = _material_fixture(T_atm=np.full((8, 8), 5.))
    initial = _state(initialize, eta=-2.499, temperature=10.)
    if source == "convection":
        params = params._replace(kappa_conv=.01)
        initial = initial._replace(T=initial.T.at[..., 0].set(1.))
    else:
        params = params._replace(lambda_bulk=8000., T_atm_3d=jnp.full_like(params.T_atm_3d, 5.))
    blocked = make_material_top_step(params, subcycle_scheme="actual_geometry_v2")(initial)

    def flat_scan(advance, carry, maximum):
        return jax.lax.scan(advance, carry, jnp.arange(maximum))[0]

    monkeypatch.setattr(material_top, "_checkpointed_material_scan", flat_scan)
    flat = make_material_top_step(params, subcycle_scheme="actual_geometry_v2")(initial)
    assert bool(blocked.valid) and bool(flat.valid)
    assert int(blocked.checks["nonlinear_active_subcycles"]) > 8
    for left, right in zip(jax.tree.leaves(flat), jax.tree.leaves(blocked), strict=True):
        assert np.asarray(left).tobytes() == np.asarray(right).tobytes()
