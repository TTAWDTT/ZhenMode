"""Same-reconstruction physical weak rates and separate pressure/rest gates."""
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from test_rstar_metric_controls import _fixture, _geometry
from test_rstar_pressure_work import _work
from test_rstar_representation import _representation_case

from config import RHO_0
from research.experiments.material_rstar_coordinates.bed_completion import (
    REPRESENTATION_TAG,
    complete_bed_reference,
)
from research.experiments.material_rstar_coordinates.kernel import hydrostatic_pressure
from research.experiments.material_rstar_coordinates.nodal_mass import (
    apply_nodal_mass,
    make_nodal_mass,
)
from research.experiments.material_rstar_coordinates.pressure_work import potential_conjugates
from research.experiments.material_rstar_coordinates.weak_oracle import physical_weak_rates
from research.experiments.material_rstar_coordinates.weak_transport import (
    WeakParameters,
    consistent_potential_basis,
    weak_content_rate,
    weak_pressure_force,
)

_compiled_rates = jax.jit(weak_content_rate)
_compiled_force = jax.jit(weak_pressure_force)


def _weak_case(truncate=False, stairs=True, flat=True, kind="affine", complete=False):
    if stairs:
        values = _representation_case(truncate, True, flat)
        params, depths, geometry, _, surface, density, _, velocities = values[:8]
    else:
        _, params, depths = _fixture(stairs=False)
        generator = np.random.default_rng(930109)
        surface = jnp.asarray(np.full(params.wet_mask.shape, -2.6) if flat else generator.uniform(-3., 2., params.wet_mask.shape))
        geometry = _geometry(params, depths, surface)
        density = 1.5 + .002 * geometry.node_depth
        velocities = tuple(jnp.asarray(generator.normal(0., .03, density.shape)) for _ in range(2))
    if complete:
        if truncate:
            raise ValueError("bed completion is registered on the original physical domain")
        bed_reference = complete_bed_reference(depths, params)
        params = SimpleNamespace(**{**vars(params), "dz_node": bed_reference.widths,
                                    "wet_mask_z": bed_reference.wet, "column_geometry": REPRESENTATION_TAG})
        depths = bed_reference.nodes
        geometry = _geometry(params, depths, surface)
        density = (1.5 + .002 * geometry.node_depth) * params.wet_mask_z
        generator = np.random.default_rng(930306)
        velocities = tuple(jnp.asarray(generator.normal(0., .03, density.shape)) * params.wet_mask_z for _ in range(2))
    if kind == "zero":
        density = jnp.zeros_like(density)
    elif kind == "constant":
        density = jnp.full_like(density, 1.5) * params.wet_mask_z
    elif kind == "random":
        density = jnp.asarray(np.random.default_rng(930304).normal(size=density.shape)) * params.wet_mask_z
    mass = make_nodal_mass(depths, params)
    reference = _geometry(params, depths, jnp.zeros(params.wet_mask.shape))
    basis = consistent_potential_basis(depths, params, reference, mass)
    content = apply_nodal_mass(density, geometry, params.dx_2d * params.dy, mass)
    weak_params = WeakParameters(*(getattr(params, name) for name in WeakParameters._fields))
    return params, weak_params, geometry, basis, surface, density, content, velocities


@pytest.mark.parametrize("flat", [False, True])
@pytest.mark.parametrize("truncate", [False, True])
def test_physical_weak_rhs_matches_independent_five_point_gauss_and_global_inventory(flat, truncate):
    params, weak_params, geometry, _, surface, density, _, velocities = _weak_case(truncate=truncate, flat=flat, kind="random")
    expected, eta_rate, scale, eta_scale = physical_weak_rates(density, velocities, surface, geometry, params)
    actual = _compiled_rates(density, velocities, surface, geometry, weak_params)
    assert np.all(np.abs(np.asarray(actual[0]) - expected) <= 64. * np.finfo(float).eps * (1. + scale))
    assert np.all(np.abs(np.asarray(actual[1]) - eta_rate) <= 64. * np.finfo(float).eps * (1e-20 + eta_scale))
    assert abs(float(jnp.sum(actual[0]))) <= 64. * np.finfo(float).eps * np.sum(scale)


@pytest.mark.parametrize("flat", [False, True])
def test_weak_constant_content_rate_is_exact_actual_hat_mass_geometry_rate(flat):
    params, weak_params, geometry, _, surface, density, _, velocities = _weak_case(flat=flat, kind="constant")
    actual, eta_rate = _compiled_rates(density, velocities, surface, geometry, weak_params)
    _, _, scale, _ = physical_weak_rates(density, velocities, surface, geometry, params)
    expected = 1.5 * (params.dx_2d * params.dy)[..., None] * geometry.weights * eta_rate[..., None]
    assert np.all(np.abs(np.asarray(actual - expected)) <= 64. * np.finfo(float).eps * (1. + scale + np.abs(np.asarray(expected))))


@pytest.mark.parametrize("truncate", [False, True])
def test_weak_pressure_power_pairs_with_independent_rates_without_rest_promotion(truncate):
    params, weak_params, geometry, basis, surface, density, content, velocities = _weak_case(truncate=truncate)
    expected, eta_rate, _, _ = physical_weak_rates(density, velocities, surface, geometry, params)
    forces = _compiled_force(density, content, surface, geometry, basis, weak_params)
    conjugates = potential_conjugates(content, surface, geometry, basis, params)
    residual, floor = _work(forces, velocities, expected, eta_rate, conjugates, geometry, params)
    assert abs(residual) <= floor


def test_weak_traces_do_not_read_dry_density_or_velocity_sentinels():
    params, weak_params, geometry, basis, surface, density, content, velocities = _weak_case(kind="random")
    wet = np.asarray(params.wet_mask_z) > 0.
    original = _compiled_rates(density, velocities, surface, geometry, weak_params)
    force = _compiled_force(density, content, surface, geometry, basis, weak_params)
    for sentinel in (123., -1e6, np.nan):
        changed = jnp.asarray(np.where(wet, density, sentinel))
        shifted = tuple(jnp.asarray(np.where(wet, value, sentinel)) for value in velocities)
        actual = _compiled_rates(changed, shifted, surface, geometry, weak_params)
        for result, reference in zip(actual, original, strict=True):
            np.testing.assert_array_equal(result, reference)
        for result, reference in zip(_compiled_force(changed, content, surface, geometry, basis, weak_params), force, strict=True):
            np.testing.assert_array_equal(result, reference)


def _weak_diagnostic(truncate, stairs, flat, kind, complete=False):
    params, weak_params, geometry, basis, surface, density, content, velocities = _weak_case(truncate, stairs, flat, kind, complete)
    actual, actual_eta = _compiled_rates(density, velocities, surface, geometry, weak_params)
    expected, eta_rate, scale, eta_scale = physical_weak_rates(density, velocities, surface, geometry, params)
    forces = _compiled_force(density, content, surface, geometry, basis, weak_params)
    conjugates = potential_conjugates(content, surface, geometry, basis, params)
    residual, floor = _work(forces, velocities, expected, eta_rate, conjugates, geometry, params)
    pressure = hydrostatic_pressure(density, surface, geometry, params)
    floor_reference_pressure = pressure
    force_floor = 64. * np.finfo(float).eps * float(jnp.max(jnp.abs(pressure))) / RHO_0 * max(float(jnp.max(params.inv_dx)), float(params.inv_dy))
    if complete:
        old = _weak_case(False, stairs, flat, kind)
        old_pressure = hydrostatic_pressure(old[5], old[4], old[2], old[0])
        floor_reference_pressure = old_pressure
        old_floor = 64. * np.finfo(float).eps * float(jnp.max(jnp.abs(old_pressure))) / RHO_0 * max(float(jnp.max(old[0].inv_dx)), float(old[0].inv_dy))
        force_floor = min(force_floor, old_floor)
    maximum = max(float(jnp.max(jnp.abs(force))) for force in forces)
    rest_applicable = flat and kind in ("zero", "constant", "affine")
    constant_passed = None
    if kind == "constant":
        expected_constant = 1.5 * (params.dx_2d * params.dy)[..., None] * geometry.weights * actual_eta[..., None]
        constant_floor = 64. * np.finfo(float).eps * (1. + scale + np.abs(np.asarray(expected_constant)))
        constant_passed = bool(np.all(np.abs(np.asarray(actual - expected_constant)) <= constant_floor))
    volume_rate = np.asarray(actual_eta * params.dx_2d * params.dy)
    diagnostic = {"truncate": truncate, "stairs": stairs, "flat": flat, "density": kind, "complete_bed": complete,
            "representation_tag": REPRESENTATION_TAG if complete else "constant_tail_nodal_control",
            "rhs_numpy_passed": bool(np.all(np.abs(np.asarray(actual) - expected) <= 64. * np.finfo(float).eps * (1. + scale))),
            "eta_numpy_passed": bool(np.all(np.abs(np.asarray(actual_eta) - eta_rate) <= 64. * np.finfo(float).eps * (1e-20 + eta_scale))),
            "global_inventory_residual_kg_per_s": float(jnp.sum(actual)),
            "global_inventory_floor_kg_per_s": 64. * np.finfo(float).eps * float(np.sum(scale)),
            "constant_tracer_geometry_passed": constant_passed,
            "global_volume_residual_m3_per_s": float(np.sum(volume_rate)),
            "global_volume_floor_m3_per_s": 64. * np.finfo(float).eps * float(np.sum(np.abs(volume_rate))),
            "pressure_work_residual_watts": residual, "pressure_work_64eps_floor_watts": floor,
            "pressure_work_passed": bool(abs(residual) <= floor), "maximum_force_m_per_s2": maximum,
            "rest_force_64eps_floor_m_per_s2": force_floor, "rest_applicable": rest_applicable,
            "physical_affine_rest_passed": bool(maximum <= force_floor) if rest_applicable else None,
            "production_promotion": False}
    arrays = {"content_rate_kg_per_s": expected, "surface_rate_m_per_s": eta_rate,
              "content_conjugate_m2_per_s2": conjugates[0], "surface_conjugate_J_per_m": conjugates[1],
              "area_m2": params.dx_2d * params.dy, "thickness_m": geometry.thickness,
              "velocity_x_m_per_s": velocities[0], "velocity_y_m_per_s": velocities[1],
              "force_x_m_per_s2": forces[0], "force_y_m_per_s2": forces[1],
              "pressure_Pa": pressure, "dx_m": params.dx_2d, "dy_m": params.dy,
              "reference_density_kg_per_m3": RHO_0,
              "density_kg_per_m3": density, "surface_m": surface, "content_kg": content,
              "rhs_absolute_scale_kg_per_s": scale, "area_surface_rate_m3_per_s": actual_eta * params.dx_2d * params.dy,
              "actual_content_rate_kg_per_s": actual, "actual_surface_rate_m_per_s": actual_eta,
              "eta_absolute_scale_m_per_s": eta_scale, "geometry_mass_fraction": geometry.weights,
              "registered_force_floor_m_per_s2": force_floor, "pressure_floor_reference_Pa": floor_reference_pressure}
    return diagnostic, {name: np.asarray(value) for name, value in arrays.items()}
