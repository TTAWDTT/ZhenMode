"""Independent polynomial primitives and smooth-mode quadrature convergence."""

from dataclasses import replace

import jax.numpy as jnp
import numpy as np
import pytest

from tests.support.grid import all_wet_grid
from zhenmode.model.config import PhysicsConfig
from zhenmode.model.solver.dynamics.pressure import (
    _baroclinic_pressure,
    _cubic_uniform_density_integral,
)
from zhenmode.model.solver.factory import make_solver_global


@pytest.mark.parametrize('n', (4, 7, 16, 32))
@pytest.mark.parametrize('degree', (0, 1, 2, 3))
def test_every_interval_is_exact_for_independent_polynomial_primitive(n, degree):
    x = np.linspace(0, 1, n)
    density = jnp.asarray(x**degree)[None, None]
    actual = _cubic_uniform_density_integral(density, jnp.asarray(np.diff(x))[None, None])
    np.testing.assert_allclose(actual[0, 0], x**(degree+1)/(degree+1), rtol=0, atol=5e-15)


def test_resolved_sine_pressure_error_has_fourth_order_convergence():
    errors = []
    for n in (16, 31, 61):
        x = np.linspace(0, 1, n)
        actual = np.asarray(_cubic_uniform_density_integral(jnp.asarray(np.sin(np.pi*x))[None, None], jnp.asarray(np.diff(x))[None, None]))[0, 0]
        exact = (1-np.cos(np.pi*x))/np.pi
        errors.append(np.max(abs(actual-exact)))
    assert all(12 < a/b < 20 for a, b in zip(errors[:-1], errors[1:], strict=True)), errors


@pytest.mark.parametrize('invalid', ('spacing', 'surface', 'bottom', 'wet', 'geometry', 'eos'))
def test_cubic_factory_rejects_unsupported_columns(invalid):
    grid = all_wet_grid(nx=8, ny=8, nz=4)
    z = np.linspace(0., -100., 4)
    grid = replace(grid, z=z, dz=-np.diff(z), depth=np.full((8, 8), 100.))
    physics = PhysicsConfig(thermodynamics='linear')
    options = dict(column_geometry='nodal_dual_v1', hydrostatic_integral_scheme='cubic_uniform_v1')
    if invalid == 'spacing':
        z = np.array([0., -10., -60., -100.])
        grid = replace(grid, z=z, dz=-np.diff(z))
    elif invalid == 'surface':
        grid = replace(grid, z=z-1.)
    elif invalid == 'bottom':
        grid = replace(grid, depth=np.full((8, 8), 99.))
    elif invalid == 'wet':
        mask = grid.wet_mask_3d.copy()
        mask[0, 0, -1] = 0.
        grid = replace(grid, wet_mask_3d=mask)
    elif invalid == 'geometry':
        options['column_geometry'] = 'legacy'
    else:
        physics = replace(physics, thermodynamics='teos10')
    with pytest.raises(ValueError):
        make_solver_global(grid, physics, 1., **options)


def test_selected_pressure_uses_eos_density_and_preserves_tracer_capacities():
    grid = all_wet_grid(nx=8, ny=8, nz=7)
    z = np.linspace(0., -100., 7)
    grid = replace(grid, z=z, dz=-np.diff(z), depth=np.full((8, 8), 100.))
    _, initialize, _, p, _ = make_solver_global(
        grid, PhysicsConfig(), 1., return_params=True, column_geometry='nodal_dual_v1',
        conservative_kv=True, localize_conv=True, polar_cap_rows=0, polar_cap_taper=0,
        hydrostatic_integral_scheme='cubic_uniform_v1')
    x = -z/100
    state = initialize()._replace(T=jnp.broadcast_to(15-x**3/(1025*.0002), (8, 8, 7)),
                                  S=jnp.full((8, 8, 7), 35.))
    exact = 9.81*100*x**4/4
    np.testing.assert_allclose(_baroclinic_pressure(state, p),
                               np.broadcast_to(exact, (8, 8, 7)), rtol=0, atol=1e-12)
    trap = p._replace(hydrostatic_integral_scheme='trapezoidal')
    assert np.max(abs(np.asarray(_baroclinic_pressure(state, trap))-exact)) > 1.
    np.testing.assert_array_equal(p.dz_norm, trap.dz_norm)
