"""Original native small-grid stock and generic-clock handoff controls."""
import copy
import math

import jax.numpy as jnp
import numpy as np
import pytest

from ocean_solver.candidates.material import solver as material
from ocean_solver.configuration import G_EARTH, RHO_0, PhysicsConfig
from ocean_solver.fd.factory import make_solver_global
from ocean_solver.geometry.types import GlobalOceanGrid
from research.experiments.material_top_band.original_native import (
    OriginalNativeAudit,
    capture_native,
    reference_pressure_kick,
)


@pytest.fixture(scope="module")
def case():
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
    physics = PhysicsConfig(nu_h=2., nu_v=.0003, kappa_h=.4, kappa_v=.0002,
                           kappa_conv=.01, nu_bi=100., kappa_bi=20., r_bot=.0001)
    forcing = tuple(np.full((nx, ny), value) for value in (.02, .003, 35.))
    _, initialize, _, params, _ = make_solver_global(
        grid, physics, .12, forcing=forcing, T_atm=np.full((nx, ny), 16.), lambda_bulk=50.,
        dtype="float64", mode_split=True, dt_bt=.01, nu_nsub=2, use_scan=False,
        conservative_kv=True, localize_conv=True, monotone_adv=True, fct_adv=True,
        column_geometry="nodal_dual_v1", process_time_scheme="symmetric_fast_v3",
        match_barotropic_transport=True, polar_cap_rows=0, polar_cap_taper=0,
        projection_niter=150, projection_rtol=1e-12, projection_preconditioner="none",
        projection_max_refinements=2, return_params=True)
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


def unchanged(first, second):
    for a, b in zip(first, second, strict=True):
        assert np.asarray(a).dtype == np.asarray(b).dtype
        assert np.asarray(a).shape == np.asarray(b).shape
        assert np.asarray(a).tobytes() == np.asarray(b).tobytes()


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


def test_true_native_fields_stocks_metric_and_pressure(case):
    state, params, grid = case
    view = capture_native(state, params, grid)
    unchanged(view.state, state)
    expected_h = np.broadcast_to([2.5, 7.5, 12.5, 17.5, 25., 15.], state.u.shape)
    np.testing.assert_array_equal(view.h_ref, expected_h)
    expected_actual = expected_h.copy()
    expected_actual[..., 0] += state.eta
    np.testing.assert_array_equal(view.h_tracer, expected_actual)
    np.testing.assert_array_equal(view.tracer_stocks[..., 0], expected_actual*state.T)
    np.testing.assert_array_equal(view.reference_momentum[..., 0], RHO_0*expected_h*state.u)
    np.testing.assert_array_equal(view.mass, RHO_0*(grid.dx_2d*grid.dy)[..., None]*expected_h)
    rho = RHO_0*(-2e-4*(np.asarray(state.T)-15.)+7.6e-4*(np.asarray(state.S)-35.))
    pressure = np.zeros_like(rho)+RHO_0*G_EARTH*np.asarray(state.eta)[..., None]
    pressure[..., 1:] += G_EARTH*np.cumsum(.5*(rho[..., :-1]+rho[..., 1:])*(-np.diff(grid.z)), axis=-1)
    np.testing.assert_allclose(view.rho_prime, rho, rtol=0., atol=1e-15)
    np.testing.assert_allclose(view.pressure, pressure, rtol=0., atol=2e-12)
    assert not view.h_ref.flags.writeable
    assert view.authority == "FD_point_samples_material_top_mass_lumped_linear_momentum_v1"


def test_reference_pressure_kick_uses_actual_midpoint_and_transpose(case):
    state, params, grid = case
    view = capture_native(state, params, grid)
    kick = reference_pressure_kick(view, params, duration=.01)
    B = independent_outflow(grid)
    p = view.pressure.ravel()
    independently_forced = B.T@p
    scale = np.abs(B).T@np.abs(p)
    assert np.all(np.abs(kick.force.ravel()-independently_forced) <= bound(kick.force.ravel(), scale))
    native_velocity = np.stack((state.u, state.v))
    repeated_mass = np.broadcast_to(view.mass, native_velocity.shape)
    assert np.all(abs(kick.before_velocity-native_velocity) <= bound(kick.before_velocity, native_velocity))
    assert np.all(abs(kick.before_momentum-repeated_mass*native_velocity) <= bound(kick.before_momentum, repeated_mass*native_velocity))
    assert np.all(abs(kick.before_momentum-repeated_mass*kick.before_velocity) <= bound(kick.before_momentum, repeated_mass*kick.before_velocity))
    assert np.all(abs(kick.after_momentum-repeated_mass*kick.after_velocity) <= bound(kick.after_momentum, repeated_mass*kick.after_velocity))
    assert np.all(abs(kick.after_momentum-kick.before_momentum-.01*kick.force) <= bound(kick.before_momentum, kick.after_momentum, .01*kick.force))
    np.testing.assert_array_equal(kick.force, repeated_mass*view.acceleration)
    assert np.max(np.abs(kick.after_momentum-kick.before_momentum)) > 0.
    before, after = kick.before_velocity.ravel(), kick.after_velocity.ravel()
    mass = np.tile(view.mass.ravel(), 2)
    change = math.fsum(.5*mass*(after*after-before*before))
    direct_work = math.fsum(.01*.5*(before+after)*independently_forced)
    scale = math.fsum(.5*mass*(before*before+after*after))
    assert abs(change-direct_work) <= bound(scale, direct_work)
    transport_work = .01*math.fsum(p*(B@(.5*(before+after))))
    assert abs(direct_work-transport_work) <= bound(np.abs(p)@(np.abs(B)@np.abs(.01*.5*(before+after))))
    unchanged(view.state, state)
    assert not kick.physical_moving_pressure_qualified


def test_moving_metric_top_band_and_half_mean_counterexamples(case):
    state, params, grid = case
    view = capture_native(state, params, grid)
    decoded = view.reference_momentum[..., 0]/(RHO_0*view.h_tracer)
    expected = np.asarray(state.u)*view.h_ref/view.h_tracer
    np.testing.assert_allclose(decoded, expected, rtol=4e-16, atol=0.)
    assert abs(2.5/(2.5-.4)-25./21.) < 3e-16
    rates = np.array([1., 0., 0.])-np.array([1./9., 1./3., 5./9.])
    np.testing.assert_allclose(rates, [8./9., -1./3., -5./9.], rtol=0., atol=2e-16)
    assert abs(rates.sum()) < 1e-16 and np.max(abs(rates)) > .8
    native_mean = np.sum(view.h_ref*state.u, axis=-1)/80.
    actual_mean = np.sum(view.h_tracer*state.u, axis=-1)/(80.+state.eta)
    defect = state.eta*(state.u[..., 0]-native_mean)/(80.+state.eta)
    assert np.all(abs(actual_mean-native_mean-defect) <= bound(actual_mean, native_mean))
    np.testing.assert_array_equal(np.linalg.solve([[.75, .25], [.25, .75]], [0., 1.]), [-.5, 1.5])


@pytest.mark.parametrize("fault", ["eta", "inverse_metric", "clock", "grid_metric", "masked", "derived_count"])
def test_invalid_native_binding_rejects_without_changes(case, fault):
    state, params, grid = case
    saved = tuple(np.asarray(value).copy() for value in state)
    copied = copy.deepcopy(grid)
    bad_state, bad_params = state, params
    if fault == "eta":
        bad_state = state._replace(eta=jnp.full_like(state.eta, np.nan))
    elif fault == "inverse_metric":
        bad_params = params._replace(inv_dx=params.inv_dx*1.01)
    elif fault == "clock":
        bad_params = params._replace(dt_bt=.02)
    elif fault == "grid_metric":
        copied.dx_2d[0, 0] *= 1.01
    elif fault == "masked":
        bad_state = state._replace(T=np.ma.array(state.T, mask=np.ones(state.T.shape, bool)))
    else:
        bad_params = params._replace(adv_nsub=2)
    with pytest.raises(ValueError):
        capture_native(bad_state, bad_params, copied)
    unchanged(state, saved)


def test_joint_handoff_refuses_all_six_fields(case):
    state, params, grid = case
    saved = tuple(np.asarray(value).copy() for value in state)
    audit = OriginalNativeAudit(state, params, grid)
    for target in ("joint_moving_stocks", "half_prism_raw_means"):
        with pytest.raises(ValueError, match="incompatible"):
            audit.require_handoff(target)
        unchanged(audit.state, state)
    unchanged(state, saved)
    assert audit.accepted_joint_steps == 0 and audit.last_receipt is None


def test_original_ten_stages_twelve_generic_fast_calls_and_default_identity(case, monkeypatch):
    state, params, grid = case
    default = material._material_step(state, params, "actual_geometry_v2", 16, "joint_heun_v1")
    from ocean_solver.fd import barotropic
    actual_calls = []
    original_fast = barotropic._symmetric_free_surface_step
    def observed_original(*args, **kwargs):
        actual_calls.append(float(args[4]))
        return original_fast(*args, **kwargs)
    monkeypatch.setattr(barotropic, '_symmetric_free_surface_step', observed_original)
    audit = OriginalNativeAudit(state, params, grid)
    receipt = audit.diagnose()
    assert actual_calls == [.01]*12
    assert receipt.original_valid and bool(default.valid)
    unchanged(receipt.original_state, default.state)
    assert len(receipt.fast_calls) == 12
    assert [row["index"] for row in receipt.fast_calls] == list(range(12))
    assert len(receipt.stages) == 10
    assert receipt.stages[2]["name"] == "nonlinear_predictor"
    assert receipt.stages[6]["name"] == "accepted_tracer_replay"
    B = independent_outflow(grid)
    area = grid.dx_2d*grid.dy
    for row in receipt.fast_calls:
        eta_before, eta_after = row["before"][0], row["after"][0]
        fx, fy = row["mean_faces"]
        volume = grid.dy*(fx-np.roll(fx, 1, axis=0))
        incoming = np.roll(fy, 1, axis=1).copy()
        incoming[:, 0] = 0.
        volume += grid.dx_2d/grid.cos_lat[None, :]*(fy-incoming)
        predicted = eta_before-.01*volume/area+row["filter_change"]
        assert np.all(abs(eta_after-predicted) <= bound(eta_before, eta_after, .01*volume/area))
    assert np.max(abs(receipt.original_state.eta-state.eta)) > 0.
    assert np.max(abs(receipt.original_state.T-state.T)) > 0.
    assert np.max(abs(receipt.original_state.u-state.u)) > 0.
    assert receipt.convection_activity > 0.
    assert receipt.pressure_impulse_abs_max > 0.
    assert B.shape == (192, 384)
    assert audit.accepted_joint_steps == 0
    unchanged(audit.state, state)


@pytest.mark.parametrize("late", [False, True])
def test_observer_failure_full_rollback(case, late):
    state, params, grid = case
    saved = tuple(np.asarray(value).copy() for value in state)
    audit = OriginalNativeAudit(state, params, grid)
    def fail(row):
        if row["index"] == (11 if late else 0):
            raise RuntimeError("injected observer failure")
    with pytest.raises(RuntimeError, match="observer failure"):
        audit.diagnose(fast_observer=fail)
    unchanged(audit.state, state)
    unchanged(state, saved)
    assert audit.accepted_joint_steps == 0 and audit.last_receipt is None


def test_scan_observation_rejects_before_stages(case):
    state, params, grid = case
    saved = tuple(np.asarray(value).copy() for value in state)
    seen = []
    with pytest.raises(ValueError, match="use_scan=False"):
        material._material_step(state, params._replace(use_scan=True), "actual_geometry_v2",
                                16, "joint_heun_v1", stage_observer=lambda *args: seen.append(args),
                                fast_observer=lambda *args: seen.append(args))
    assert not seen
    unchanged(state, saved)

def test_current_mass_only_pressure_reinterpretation_is_detected(case):
    state, params, grid = case
    view = capture_native(state, params, grid)
    B = independent_outflow(grid)
    force = B.T@view.pressure.ravel()
    acceleration = view.acceleration.reshape(2, -1).ravel()
    actual_mass = np.tile((RHO_0*view.area[..., None]*view.h_tracer).ravel(), 2)
    wrongly_forced = actual_mass*acceleration
    scale = np.abs(B).T@abs(view.pressure.ravel())+abs(wrongly_forced)
    blocked = scale == 0.
    assert np.all(force[blocked] == 0.) and np.all(wrongly_forced[blocked] == 0.)
    envelope = bound(scale)
    ratio = np.divide(abs(wrongly_forced-force), envelope, out=np.zeros_like(force), where=envelope > 0.)
    assert np.max(ratio) > 100.


def test_parameter_change_after_capture_refuses_before_observers(case):
    state, params, grid = case
    saved = tuple(np.asarray(value).copy() for value in state)
    audit = OriginalNativeAudit(state, params, grid)
    audit.params = params._replace(tau_x_2d=params.tau_x_2d+.001)
    seen = []
    with pytest.raises(ValueError, match="binding"):
        audit.diagnose(fast_observer=seen.append)
    assert not seen
    unchanged(state, saved)
    unchanged(audit.state, state)
    assert audit.accepted_joint_steps == 0 and audit.last_receipt is None
