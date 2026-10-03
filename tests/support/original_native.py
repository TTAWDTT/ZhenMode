"""Frozen original native case and independent signed-face/work controls."""
import math

import jax.numpy as jnp
import numpy as np

from ocean_solver.configuration import PhysicsConfig
from ocean_solver.fd.factory import make_solver_global
from ocean_solver.geometry.types import GlobalOceanGrid


def original_factory(grid):
    physics = PhysicsConfig(nu_h=2., nu_v=.0003, kappa_h=.4, kappa_v=.0002,
                           kappa_conv=.01, nu_bi=100., kappa_bi=20., r_bot=.0001)
    forcing = tuple(np.full((8, 4), value) for value in (.02, .003, 35.))
    return make_solver_global(
        grid, physics, .12, forcing=forcing, T_atm=np.full((8, 4), 16.), lambda_bulk=50.,
        dtype="float64", mode_split=True, dt_bt=.01, nu_nsub=2, use_scan=False,
        conservative_kv=True, localize_conv=True, monotone_adv=True, fct_adv=True,
        column_geometry="nodal_dual_v1", process_time_scheme="symmetric_fast_v3",
        match_barotropic_transport=True, polar_cap_rows=0, polar_cap_taper=0,
        projection_niter=150, projection_rtol=1e-12, projection_preconditioner="none",
        projection_max_refinements=2, return_params=True)


def make_original_native_case():
    nx, ny, nz = 8, 4, 6
    latitude = np.array([-30., -10., 10., 30.])
    cosine = np.cos(np.radians(latitude))
    z = np.array([0., -5., -15., -30., -50., -80.])
    grid = GlobalOceanGrid(
        lon=(np.arange(nx)+.5)*360./nx, lat=latitude,
        dx_2d=np.broadcast_to(6.371e6*cosine*np.radians(360./nx), (nx, ny)).copy(),
        dy=float(6.371e6*np.radians(20.)), cos_lat=cosine,
        f=np.broadcast_to(2*7.2921e-5*np.sin(np.radians(latitude)), (nx, ny)).copy(),
        z=z, dz=-np.diff(z), nz=nz, depth=np.full((nx, ny), 80.),
        wet_mask=np.ones((nx, ny)), ocean_mask=np.ones((nx, ny), bool),
        land_mask=np.zeros((nx, ny)), wet_mask_3d=np.ones((nx, ny, nz)), nx=nx, ny=ny)
    _, initialize, _, params, _ = original_factory(grid)
    x, y, level = np.indices((nx, ny, nz))
    shear = np.array([1., 1.2, 1.4, .6, .3, .1])
    normal = .003*np.sin(np.pi*y/3.)*(1.+level/20.)
    normal[:, (0, -1), :] = 0.
    state = initialize()._replace(
        u=jnp.asarray((.02+.01*np.sin(2*np.pi*x/8.))*shear),
        v=jnp.asarray(normal),
        T=jnp.asarray(15.+.2*level+.03*np.cos(2*np.pi*x/8.)),
        S=jnp.asarray(35.+.005*level+.004*np.sin(np.pi*y/3.)),
        eta=jnp.asarray(-.4+.02*np.cos(2*np.pi*np.arange(nx)[:, None]/8.)*np.ones((1, ny))),
        ice=jnp.zeros((nx, ny), dtype=jnp.float64))
    return state, params, grid


def independent_outflow(grid):
    """Assemble native volume outflow directly from signed face owners."""
    shape = (grid.nx, grid.ny, grid.nz)
    n = math.prod(shape)
    B = np.zeros((n, 2*n))
    edges = np.r_[0., -.5*(grid.z[:-1]+grid.z[1:]), -grid.z[-1]]
    widths = np.diff(edges)
    for i, j, k in np.ndindex(shape):
        owner = np.ravel_multi_index((i, j, k), shape)
        east = np.ravel_multi_index(((i+1) % grid.nx, j, k), shape)
        coefficient = .5*grid.dy*widths[k]
        for column in (owner, east):
            B[owner, column] += coefficient
            B[east, column] -= coefficient
        if j+1 < grid.ny:
            north = np.ravel_multi_index((i, j+1, k), shape)
            face_cosine = .5*(grid.cos_lat[j]+grid.cos_lat[j+1])
            coefficient = .5*grid.dx_2d[i, j]/grid.cos_lat[j]*face_cosine*widths[k]
            for column in (owner+n, north+n):
                B[owner, column] += coefficient
                B[north, column] -= coefficient
    for i, j, k in np.ndindex(shape):
        if j in (0, grid.ny-1):
            B[:, n+np.ravel_multi_index((i, j, k), shape)] = 0.
    return B


def bound(*operands):
    return 512.*np.finfo(float).eps*sum(np.abs(value) for value in operands)

def assert_identity(actual, expected, *operands):
    """Frozen dimensional roundoff gate, including exact zero-bound rejection."""
    arrays = []
    for value in (actual, expected, *operands):
        if np.ma.getmaskarray(value).any():
            raise ValueError("masked identity input")
        array = np.asarray(value)
        if array.dtype.kind not in "biuf" or not np.isfinite(array).all():
            raise ValueError("nonfinite identity input")
        arrays.append(array)
    actual, expected, *operands = arrays
    difference = abs(actual-expected)
    envelope = bound(*operands)
    if not np.isfinite(difference).all() or not np.isfinite(envelope).all():
        raise ValueError("nonfinite identity difference/bound")
    if np.any(difference > envelope):
        raise ValueError("original identity exceeds frozen primitive bound")
    ratios = np.divide(difference,envelope,out=np.zeros_like(difference,dtype=float),where=envelope>0.)
    return float(np.max(ratios))


def audit_native_receipt(view, kick, receipt, grid):
    """Independent signed-face, actual-stock and reference-work witness gates."""
    B = independent_outflow(grid)
    p = view.pressure.ravel()
    native_velocity = np.stack((view.state.u,view.state.v)).ravel()
    mass = np.tile(view.mass.ravel(),2)
    before, after = kick.before_velocity.ravel(),kick.after_velocity.ravel()
    before_M, after_M = kick.before_momentum.ravel(),kick.after_momentum.ravel()
    force, independent_force = kick.force.ravel(),B.T@p
    ratios = {}
    ratios["native_velocity"] = assert_identity(before,native_velocity,before,native_velocity)
    ratios["native_momentum"] = assert_identity(before_M,mass*native_velocity,before_M,mass*native_velocity)
    ratios["before_stock_quotient"] = assert_identity(before_M,mass*before,before_M,mass*before)
    ratios["after_stock_quotient"] = assert_identity(after_M,mass*after,after_M,mass*after)
    ratios["pressure_transpose"] = assert_identity(force,independent_force,abs(B).T@abs(p),force)
    ratios["actual_stock_kick"] = assert_identity(after_M-before_M,.01*independent_force,before_M,after_M,.01*independent_force)
    midpoint = .5*(before+after)
    independent_work = .01*math.fsum(midpoint*independent_force)
    kinetic_change = math.fsum(.5*mass*(after**2-before**2))
    energy_scale = math.fsum(.5*mass*(before**2+after**2))
    impulse_scale = math.fsum(abs(midpoint)*(abs(before_M)+abs(after_M)+.01*abs(independent_force)))
    flux_scale = .01*float(abs(p)@(abs(B)@abs(midpoint)))
    ratios["kinetic_value"] = assert_identity(kick.kinetic_change,kinetic_change,energy_scale)
    ratios["fixed_mass_work"] = assert_identity(kinetic_change,independent_work,energy_scale,independent_work)
    ratios["reported_fixed_mass_work"] = assert_identity(kinetic_change,kick.pressure_work,energy_scale,kick.pressure_work)
    ratios["impulse_work"] = assert_identity(kick.impulse_work,independent_work,impulse_scale,independent_work)
    ratios["dt_force_work"] = assert_identity(kick.pressure_work,independent_work,flux_scale,kick.pressure_work)
    ratios["native_transport_work"] = assert_identity(kick.transport_work,independent_work,flux_scale,kick.transport_work)
    fast_residuals, fast_ratios = [], []
    for row in receipt.fast_calls:
        a,b = row["before"][0],row["after"][0]
        fx,fy = row["mean_faces"]
        incoming = np.roll(fy,1,axis=1).copy()
        incoming[:,0] = 0.
        outflow = grid.dy*(fx-np.roll(fx,1,axis=0))
        outflow += grid.dx_2d/grid.cos_lat[None,:]*(fy-incoming)
        transported = .01*outflow/(grid.dx_2d*grid.dy)
        residual = b-a+transported-row["filter_change"]
        fast_ratios.append(assert_identity(residual,0.,a,b,transported,row["filter_change"]))
        fast_residuals.append(float(np.max(abs(residual))))
    if (not receipt.original_valid or len(receipt.fast_calls)!=12
            or np.max(abs(after_M-before_M))<=0.):
        raise ValueError("native original result/clock/impulse gate")
    return {"identity_roundoff_ratios":ratios,
            "per_fast_continuity_roundoff_ratio_max":max(fast_ratios),
            "per_fast_continuity_residual_max_m":max(fast_residuals),
            "independent_pressure_work_J":independent_work,
            "fixed_mass_work_roundoff_bound_J":float(bound(energy_scale,kick.pressure_work))}
