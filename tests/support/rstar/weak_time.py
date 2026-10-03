"""Measured time order and joint moving-consistent-mass/source stage contracts."""


from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from ocean_solver.config.definitions import C_P, RHO_0
from research.experiments.material_rstar_coordinates.kernel import rstar_geometry
from research.experiments.material_rstar_coordinates.nodal_mass import (
    apply_nodal_mass,
    make_nodal_mass,
    solve_nodal_mass,
)
from research.experiments.material_rstar_coordinates.weak_bounded import bounded_weak_euler
from research.experiments.material_rstar_coordinates.weak_sparse import make_weak_graph
from research.experiments.material_rstar_coordinates.weak_time import bounded_weak_heun
from research.experiments.material_rstar_coordinates.weak_transport import WeakParameters
from tests.support.rstar.weak_bounded import _bounded_case

_compiled_heun = jax.jit(bounded_weak_heun, static_argnames=("fields",))


_compiled_euler = jax.jit(bounded_weak_euler)


def _flat_time_case():
    nodes = jnp.array([0., 5., 20., 50.])
    wet = jnp.ones((12, 3, 4))
    gaps = jnp.diff(nodes)
    widths = .5 * (jnp.concatenate((jnp.zeros(1), gaps)) + jnp.concatenate((gaps, jnp.zeros(1))))
    params = WeakParameters(wet, wet[..., 0], jnp.broadcast_to(widths, wet.shape),
                            jnp.full(wet.shape[:2], 100000.), 100000., jnp.ones(3))
    surface = jnp.zeros(wet.shape[:2])
    geometry = rstar_geometry(surface, nodes, params.dz_node, wet)
    mass = make_nodal_mass(nodes, params)
    graph = make_weak_graph(params)
    angles = 2. * jnp.pi * jnp.arange(12)[:, None, None] / 12.
    concentration = jnp.broadcast_to(15. + .6 * jnp.cos(angles) + .3 * nodes / 50., wet.shape)
    content = apply_nodal_mass(concentration, geometry, params.dx_2d * params.dy, mass)
    return params, nodes, mass, graph, surface, concentration, content


def _flat_fields(time, content, surface, geometry):
    return (jnp.ones_like(content), jnp.zeros_like(content)), jnp.zeros_like(content)


def _time_diagnostic(with_arrays=False):
    params, nodes, mass, graph, surface, concentration, content = _flat_time_case()
    phase = 12000. * 1e-5 * np.sin(2. * np.pi / 12.)
    angles = 2. * np.pi * np.arange(12)[:, None, None] / 12.
    exact = np.broadcast_to(15. + .6 * np.cos(angles - phase) + .3 * np.asarray(nodes) / 50., content.shape)
    weights = np.asarray(params.dx_2d * params.dy)[..., None] * np.asarray(params.dz_node)
    errors = {"heun": [], "euler": []}
    arrays = {"exact": exact, "row_mass": weights, "original_content": np.asarray(content),
              "original_concentration": np.asarray(concentration)}
    minima = []
    for count in (8, 16, 32):
        duration = 12000. / count
        for method in ("heun", "euler"):
            current, eta, minimum = content, surface, 1.
            for step in range(count):
                if method == "heun":
                    result = _compiled_heun(current, eta, step * duration, duration, params, nodes, mass, graph, _flat_fields)
                else:
                    geometry = rstar_geometry(eta, nodes, params.dz_node, params.wet_mask_z)
                    velocity, source = _flat_fields(step * duration, current, eta, geometry)
                    result = _compiled_euler(current, eta, velocity, source, duration, params, nodes, mass, graph)
                assert bool(result.valid)
                current, eta = result.content, result.surface
                minimum = min(minimum, float(result.minimum_limiter))
            geometry = rstar_geometry(eta, nodes, params.dz_node, params.wet_mask_z)
            decoded = np.asarray(solve_nodal_mass(current, geometry, params.dx_2d * params.dy, mass))
            errors[method].append(float(np.sqrt(np.sum(weights * (decoded - exact) ** 2) / np.sum(weights))))
            arrays[method + "_" + str(count)] = decoded
            if method == "heun":
                minima.append(minimum)
    record = {"errors": errors, "orders": {name: np.log2(np.asarray(values[:-1]) / values[1:]).tolist()
                                           for name, values in errors.items()},
              "minimum_limiter": minima, "scope": "full-node independent semidiscrete Fourier control, not active moving-mass branch order"}
    return (record, arrays) if with_arrays else record


def _state_fields(velocities, source_rate, time, content, surface, geometry):
    factor = 1. + time / 1000. + surface[..., None] / 1000.
    return tuple(value * factor for value in velocities), source_rate * (1. + time / 2000.)


def _moving_time_diagnostic(kind):
    _, params, nodes, mass, graph, _, _, surface, concentration, content, velocities, source = _bounded_case(kind)
    fields = partial(_state_fields, velocities, source)
    arrays = {"wet": np.asarray(mass.wet), "nodes": np.asarray(nodes), "reference_widths": np.asarray(params.dz_node),
              "area": np.asarray(params.dx_2d * params.dy), "original_content": np.asarray(content),
              "original_concentration": np.asarray(concentration), "original_surface": np.asarray(surface),
              "velocity_x": np.asarray(velocities[0]), "velocity_y": np.asarray(velocities[1])}
    minima, fractions = [], []
    wet = np.asarray(mass.wet)
    for count in (8, 16, 32, 256, 512):
        duration = 4000. / count
        current, eta, minimum, fraction = content, surface, 1., 0.
        for step in range(count):
            result = _compiled_heun(current, eta, step * duration, duration, params, nodes, mass, graph, fields)
            assert bool(result.valid), (kind, count, step)
            current, eta = result.content, result.surface
            minimum = min(minimum, float(result.minimum_limiter))
            fraction = max(fraction, float(result.fraction))
        geometry = rstar_geometry(eta, nodes, params.dz_node, params.wet_mask_z)
        decoded = np.asarray(solve_nodal_mass(current, geometry, params.dx_2d * params.dy, mass))
        arrays["decoded_" + str(count)] = decoded
        arrays["content_" + str(count)] = np.asarray(current)
        arrays["surface_" + str(count)] = np.asarray(eta)
        minima.append(minimum)
        fractions.append(fraction)
    weights = np.asarray(params.dx_2d * params.dy)[..., None] * np.asarray(geometry.thickness)
    arrays["endpoint_reference_row_mass"] = weights
    def norm(difference):
        return float(np.sqrt(np.sum(weights[wet] * difference[wet] ** 2) / np.sum(weights[wet])))
    errors = [norm(arrays["decoded_" + str(count)] - arrays["decoded_512"]) for count in (8, 16, 32)]
    gap = norm(arrays["decoded_256"] - arrays["decoded_512"])
    record = {"scope": "fixed spatial grid moving active-limiter prescribed-field self convergence, not independently solved full PDE",
              "errors": errors, "orders": np.log2(np.asarray(errors[:-1]) / errors[1:]).tolist(),
              "reference_gap": gap, "reference_resolved": bool(gap <= .1 * errors[-1]),
              "minimum_limiter": minima, "maximum_fraction": fractions,
              "counts": [8, 16, 32, 256, 512], "total_seconds": 4000.}
    return record, arrays


def _heat_fields(params, time, content, surface, geometry):
    source = jnp.zeros_like(content).at[..., 0].set((100. + 5. * time) * params.dx_2d * params.dy / (RHO_0 * C_P))
    return (jnp.zeros_like(content), jnp.zeros_like(content)), source


def _refusal_fields(velocities, source, failure, time, content, surface, geometry):
    if failure == "second_nan_source":
        source = source.at[3, 2, 2].set(jnp.where(time > 0., jnp.nan, 0.))
    elif failure == "second_dry_source":
        source = source.at[0, 2, 0].set(jnp.where(time > 0., 1., 0.))
    elif failure == "second_cfl":
        velocities = tuple(value * jnp.where(time > 0., 1e7, 1.) for value in velocities)
    return velocities, source
