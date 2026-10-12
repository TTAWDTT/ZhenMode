"""Declared affine moment, conservation, gauge and barrier controls for v4."""
from dataclasses import replace
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tests.support.grid import all_wet_grid
from tests.support.uniform_tracer_channel import column_case
from zhenmode.model.config import PhysicsConfig
from zhenmode.model.solver.dynamics.transport import (
    _advection_scalar,
    _horizontal_tracer_fluxes,
    _layer_face_transports,
    _material_point_scalar_rate,
    _vertical_transport_iface,
)
from zhenmode.model.solver.factory import make_solver_global
from zhenmode.model.solver.numerics.column import cubic_reverse_primitive
from zhenmode.model.solver.numerics.vertical import joint_scalar_blend, point_vertical_antiflux


def case():
    return column_case(column_mean_scheme='nodal_capacity',
                       vertical_tracer_scheme='material_point_v4',
                       tracer_flux_arithmetic='integrated_reference_v1')


@pytest.mark.parametrize('n', (16, 32, 64))
def test_interior_cubic_primitive_and_original_surface(n):
    d = np.linspace(0., 100., n)
    primitive = cubic_reverse_primitive(n, d[1])
    h = np.full(n, d[1])
    h[[0, -1]] *= .5
    np.testing.assert_array_equal(primitive[0], h)
    for degree in range(4):
        exact = (100.**(degree+1)-d**(degree+1))/(degree+1)
        np.testing.assert_allclose((primitive@d**degree)[1:], exact[1:],
                                   rtol=0., atol=1e-14*max(1., np.max(abs(exact))))
    # Do not describe the deliberately retained top trapezoid row as cubic.
    assert abs((primitive@d**2)[0]-100.**3/3) > 1.


def fields(grid):
    x = np.arange(grid.nx)[:, None, None]*2*np.pi/grid.nx
    y = (np.arange(grid.ny)[None, :, None]+.5)*np.pi/grid.ny
    d = -grid.z[None, None, :]
    u = jnp.asarray(.002*np.sin(x)*np.cos(d*np.pi/100)+np.zeros((1, grid.ny, 1)))
    v = jnp.asarray(.003*np.sin(y)*np.cos(d*np.pi/100)+np.zeros((grid.nx, 1, 1)))
    T = jnp.asarray(15.-.02*d+.001*np.cos(x)*np.cos(y)*np.sin(d*np.pi/100))
    return u, v, T


def test_actual_owner_constant_bad_mass_and_gauge_inventory():
    grid, (_, _, _, p, _) = case()
    u, v, T = fields(grid)
    F = _vertical_transport_iface(u, v, p)
    rate = jax.jit(lambda q: _advection_scalar(q, u, v, F, p))(jnp.full_like(T, 35.))
    np.testing.assert_allclose(rate, 0., rtol=0., atol=1e-17)
    wrong = _advection_scalar(jnp.full_like(T, 35.), u, v, -F, p)
    assert np.max(abs(np.asarray(wrong))) > 1e-12
    rate, boundary = _advection_scalar(T, u, v, F, p, return_boundary=True)
    shifted = _advection_scalar(T+273.15, u, v, F, p)
    np.testing.assert_allclose(shifted, rate, rtol=0., atol=1e-13)
    assert abs(float(jnp.sum(rate*p.dz_node)-jnp.sum(boundary))) < 1e-13


def test_stagnant_faces_pin_projection_without_hidden_scalar_transfer():
    _, (_, _, _, p, _) = case()
    shape = (2, 2, 16)
    mass = jnp.ones((2, 2, 17)).at[..., 0].set(.2).at[..., -1].set(0.).at[..., 8].set(0.)
    anti = point_vertical_antiflux(jnp.ones(shape)*1e-5, jnp.ones(shape)*1e-7, mass, p)
    np.testing.assert_array_equal(np.asarray(anti)[..., [0, 8, -1]], 0.)
    rate = -(anti[..., 1:]-anti[..., :-1])/p.dz_node
    for a, b in ((0, 8), (8, 16)):
        assert abs(float(jnp.sum(rate[..., a:b]*p.dz_node[..., a:b]))) < 1e-13


def test_joint_floor_feasible_interval_and_impossible_baseline():
    p = SimpleNamespace(dt=1., adv_nsub=1, dz_iface=jnp.ones((1, 1, 3)))
    tracer = jnp.array([[[0., 1., 2., 3.]]])
    mass = jnp.zeros((1, 1, 5))
    low = jnp.array([[[0., -.1, 0., 0.]]])
    delta = jnp.array([[[-1., 0., 0., 0.]]])
    alpha, feasible = joint_scalar_blend(tracer, low, delta, mass, p, minimum=0.)
    assert bool(feasible)
    assert np.min(np.asarray(tracer+low+alpha*delta)) >= -1e-14
    alpha, feasible = joint_scalar_blend(tracer, low.at[..., 0].set(-1.),
                                         jnp.zeros_like(delta), mass, p, minimum=0.)
    assert not bool(feasible)
    assert float(alpha) == 0.  # Explicit infeasibility, not a positivity claim.
    alpha, feasible = joint_scalar_blend(-jnp.ones_like(tracer), jnp.zeros_like(low),
                                         jnp.ones_like(delta), mass, p, minimum=0.)
    assert not bool(feasible)  # Floor and original envelope do not intersect.


def test_shared_domain_blend_active_and_positive_lower_interval():
    p = SimpleNamespace(dt=1., adv_nsub=1, dz_iface=jnp.ones((1, 1, 3)))
    tracer = jnp.broadcast_to(jnp.array([0.,1.,2.,3.]), (2,1,4))
    delta = jnp.array([[[0.,4.,-4.,0.]], [[0.,-4.,4.,0.]]])
    alpha, feasible = joint_scalar_blend(tracer, jnp.zeros_like(tracer), delta,
                                         jnp.zeros((2,1,5)), p, minimum=0.)
    assert bool(feasible)
    assert float(alpha) == .25
    assert float(jnp.sum(alpha*delta)) == 0.
    assert float(jnp.min(tracer+alpha*delta)) >= 0.
    alpha, feasible = joint_scalar_blend(jnp.ones_like(tracer), -2*jnp.ones_like(tracer),
                                         3*jnp.ones_like(tracer), jnp.zeros((2,1,5)), p, minimum=0.)
    assert bool(feasible)
    assert float(alpha) == pytest.approx(2/3)


def test_affine_quarter_mode_rhs_and_joint_blend():
    nx, ny, n = 8, 32, 16
    grid = all_wet_grid(nx=nx, ny=ny, nz=n)
    d = np.linspace(0., 100., n)
    k = np.pi/20000.
    m = 3.1417199723837514/100.
    grid = replace(grid, z=-d, dz=np.diff(d), depth=np.full((nx, ny), 100.),
                   dx_2d=np.full((nx, ny), 20000./nx), dy=20000./ny,
                   cos_lat=np.ones(ny), f=np.full((nx, ny), 1e-4))
    physics = PhysicsConfig(**dict.fromkeys(('nu_h','nu_v','nu_bi','kappa_h','kappa_v','kappa_bi',
                                             'kappa_conv','kappa_gm','kappa_redi','r_bot','cd'), 0.))
    _, _, _, p, _ = make_solver_global(grid, physics, 12.5, return_params=True,
        column_geometry='nodal_dual_v1', conservative_kv=True, localize_conv=True,
        polar_cap_rows=0, polar_cap_taper=0, external_mode_scheme='symmetric',
        meridional_boundary_scheme='closed_faces', pressure_continuity_scheme='centered_fourth',
        density_rotation_scheme='joint_shear', hydrostatic_surface_scheme='linear_density',
        hydrostatic_integral_scheme='cubic_uniform_v1', vertical_tracer_scheme='material_point_v4',
        tracer_flux_arithmetic='integrated_reference_v1')
    omega = np.sqrt(1e-8+9.81*.0002*.02*(k/m)**2)
    y = (np.arange(ny)+.5)*20000./ny
    shape = (nx, ny, n)
    v = jnp.asarray(np.broadcast_to(.01*omega*m/k*np.sin(k*y)[None,:,None]*np.cos(m*(100.-d))[None,None,:], shape))
    u = jnp.zeros(shape)
    T = jnp.asarray(np.broadcast_to(15.-.02*d, shape))
    faces = _layer_face_transports(u, v, p)
    F = _vertical_transport_iface(u, v, p)
    fx, fy = _horizontal_tracer_fluxes(T-T.reshape(-1)[0], u, v, p, faces)
    rate, _, diagnostics = _material_point_scalar_rate(T, F, faces, (fx*p.dz_node, fy*p.dz_node), p,
                                                        return_diagnostics=True)
    truth = .02*.01*omega*np.cos(k*y)[None,:,None]*np.sin(m*(100.-d))[None,None,:]
    error = np.asarray(rate)-truth
    relative = np.sqrt(np.sum(error**2*np.asarray(p.dz_node))/np.sum(np.broadcast_to(truth,shape)**2*np.asarray(p.dz_node)))
    print('affine measured', dict(relative=float(relative), blend=float(diagnostics['blend']),
                                  feasible=bool(diagnostics['feasible']),
                                  low_boundary=np.asarray(diagnostics['low_rate'])[0, :, [0,-1]].tolist(),
                                  delta_boundary=np.asarray(diagnostics['delta'])[0, :, [0,-1]].tolist()))
    assert relative <= .0001
    assert bool(diagnostics['feasible'])
    assert float(diagnostics['blend']) > .999


def test_factory_rejects_unqualified_mean_and_arithmetic():
    with pytest.raises(ValueError, match='material_point_v4'):
        column_case(vertical_tracer_scheme='material_point_v4')
