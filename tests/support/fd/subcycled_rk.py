"""Independent nonlinear time oracles, not a whole-model second-order claim."""

import jax
import jax.numpy as jnp
import numpy as np

from tests.support.fd.process_time import _numpy_vertical_matrix
from tests.support.fd.reference_geometry import WIDTHS, _fixture
from zhenmode.model.config.definitions import C_P, RHO_0
from zhenmode.model.timestepping.integration import _step_impl, _tracer_step_with_transport


def _candidate(**options):
    return _fixture(match_barotropic_transport=True, process_time_scheme="subcycled_rk2_v2", **options)

def _trajectory(initial, params, duration):
    count = int(round(duration / params.dt))
    assert count * params.dt == duration
    return jax.jit(lambda values: jax.lax.fori_loop(
        0, count, lambda index, current: _step_impl(current, params), values))(initial)

def convection_order(scheme):
    _, (_, initialize, _, base, _) = _candidate(use_scan=True)
    base = base._replace(process_time_scheme=scheme)
    operator = 0.1 * _numpy_vertical_matrix()
    square_root_width = np.sqrt(WIDTHS)
    eigenvalues, eigenvectors = np.linalg.eigh(square_root_width[:, None] * operator / square_root_width[None, :])
    profile = np.arange(4., dtype=float)
    reference = (eigenvectors @ (np.exp(eigenvalues * 160.)
                 * (eigenvectors.T @ (square_root_width * profile)))) / square_root_width
    rows = []
    for duration in (40., 20., 10.):
        params = base._replace(dt=duration, dt_bt=duration / 2., n_subcyc=2,
                               kappa_conv=0.1, conv_nsub=2, adv_nsub=1)
        initial = initialize()._replace(T=jnp.broadcast_to(jnp.asarray(profile), (8, 8, 4)))
        actual = np.asarray(_trajectory(initial, params, 160.).T)[0, 0]
        rows.append({"dt_s": duration, "rms_K": float(np.sqrt(np.sum(WIDTHS * (actual - reference) ** 2) / 50.)),
                     "profile_K": actual.tolist(), "reference_K": reference.tolist(),
                     "weighted_change_K_m": float(np.sum(WIDTHS * (actual - profile)))})
    return rows

def quadratic_drag_order(scheme):
    _, (_, initialize, _, base, _) = _candidate(use_scan=True)
    base = base._replace(process_time_scheme=scheme, bottom_friction="quadratic", cd=0.01)
    initial = initialize()._replace(u=jnp.full((8, 8, 4), 0.8))
    duration = 40.
    reference = 0.8 / (1. + 0.01 * 0.8 * duration)
    rows = []
    for outer_dt in (5., 2.5, 1.25):
        params = base._replace(dt=outer_dt, dt_bt=outer_dt / 2., n_subcyc=2)
        actual = np.asarray(_trajectory(initial, params, duration).u)[0, 3]
        rows.append({"dt_s": outer_dt, "error_m_per_s": abs(float(actual[-1]) - reference),
                     "bottom_velocity_m_per_s": float(actual[-1]), "reference_m_per_s": reference})
        np.testing.assert_allclose(actual[:-1], 0.8, atol=2e-14, rtol=0.)
    return rows

def _transport_case(duration, subcycles=3):
    _, (_, initialize, _, params, _) = _candidate(use_scan=True, monotone_adv=True)
    params = params._replace(dt=duration, adv_nsub=subcycles, conv_nsub=1,
                             dx_2d=jnp.full_like(params.dx_2d, 1000.), dy=1000.,
                             inv_dx=jnp.full_like(params.inv_dx, 0.001), inv_dy=0.001,
                             inv_dx2=jnp.full_like(params.inv_dx2, 1e-6), inv_dy2=1e-6,
                             cos_lat=jnp.ones_like(params.cos_lat), lambda_bulk=400.,
                             T_atm_3d=jnp.full_like(params.T_atm_3d, 15.))
    phase = 2. * np.pi * np.arange(8) / 8.
    temperature = np.broadcast_to((16. + np.cos(phase))[:, None, None], (8, 8, 4))
    initial = initialize()._replace(T=jnp.asarray(temperature), u=jnp.full((8, 8, 4), 0.2))
    column_transport = (jnp.full((8, 8), 10.), jnp.zeros((8, 8)))
    return initial, params, column_transport

def advection_bulk_order():
    phase = 2. * np.pi * np.arange(8) / 8.
    decay = np.zeros(4)
    decay[0] = 400. / (RHO_0 * C_P * WIDTHS[0])
    eigenvalue = 0.2 / 1000. * (np.exp(-1j * 2. * np.pi / 8.) - 1.)
    reference = (15. + np.exp(-decay * 1200.))[None, :] + np.real(
        np.exp(1j * phase[:, None] + (eigenvalue - decay[None, :]) * 1200.))
    rows = []
    for duration in (400., 200., 100.):
        initial, params, faces = _transport_case(duration)
        count = int(1200. / duration)
        actual = jax.jit(lambda values: jax.lax.fori_loop(0, count, lambda index, current:
            _tracer_step_with_transport(current, params, faces), values))(initial)
        error = np.asarray(actual.T)[:, 3] - reference
        rows.append({"dt_s": duration, "rms_K": float(np.sqrt(np.mean(error ** 2))),
                     "profile_K": np.asarray(actual.T)[:, 3].tolist(), "reference_K": reference.tolist()})
    return rows
