"""Independent M3 process oracles; isolated order is not whole-model order."""


import jax.numpy as jnp
import numpy as np

from ocean_solver.dynamics.barotropic import _free_surface_step_fd
from ocean_solver.dynamics.processes import _linear_half_step
from ocean_solver.dynamics.transport import _barotropic_velocity
from tests.support.fd.reference_geometry import WIDTHS, _fixture


def _candidate(**options):
    return _fixture(match_barotropic_transport=True, process_time_scheme="consistent_split_v1", **options)

def _numpy_vertical_matrix():
    distances = np.array([5., 15., 30.])
    matrix = np.zeros((4, 4))
    for interface, distance in enumerate(distances):
        matrix[interface, interface] -= 1. / (WIDTHS[interface] * distance)
        matrix[interface, interface + 1] += 1. / (WIDTHS[interface] * distance)
        matrix[interface + 1, interface] += 1. / (WIDTHS[interface + 1] * distance)
        matrix[interface + 1, interface + 1] -= 1. / (WIDTHS[interface + 1] * distance)
    return matrix

def _inertial_source_path(params, state, duration, subcycles):
    """Isolate source allocation by supplying zero divergence to the fast step."""
    params = params._replace(f=jnp.full_like(params.f, 1e-3))
    state = _linear_half_step(state, params, duration / 2.)
    mean_x, mean_y = _barotropic_velocity(state.u, state.v, params)
    initial_x, initial_y = mean_x, mean_y
    eta = state.eta
    for _ in range(subcycles):
        eta, mean_x, mean_y = _free_surface_step_fd(
            eta, mean_x, mean_y, params, dt_half=duration / subcycles,
            column_face_transport=(jnp.zeros_like(eta), jnp.zeros_like(eta)))
    state = state._replace(u=state.u + (mean_x - initial_x)[..., None],
                          v=state.v + (mean_y - initial_y)[..., None], eta=eta)
    return _linear_half_step(state, params, duration / 2.)
