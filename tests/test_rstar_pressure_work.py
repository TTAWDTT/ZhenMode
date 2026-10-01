"""Independent pressure-work controls; physical candidate gates remain separate."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from test_rstar_metric_controls import _fixture, _geometry

from config import G_EARTH, RHO_0
from jax_solver_global import _advection_scalar, _face_transport_divergence
from research.experiments.material_rstar_coordinates.kernel import (
    coordinate_pressure_gradient,
    hydrostatic_pressure,
    relative_vertical_transport,
)
from research.experiments.material_rstar_coordinates.pressure_work import (
    centered_content_rate,
    coordinate_layer_faces,
    energy_adjoint_force,
    make_potential_basis,
    paired_chain_rule_force,
    potential_conjugates,
    potential_energy,
    transpose_force,
)


def _case(stairs=True, flat=False, kind="random"):
    _, params, depths = _fixture(stairs=stairs)
    generator = np.random.default_rng(930109)
    surface = (np.full(params.wet_mask.shape, -2.6) if flat else generator.uniform(-3., 2., params.wet_mask.shape)) * np.asarray(params.wet_mask)
    geometry = _geometry(params, depths, surface)
    basis = make_potential_basis(depths, params)
    density = generator.normal(size=params.wet_mask_z.shape)
    if kind == "zero":
        density[:] = 0.
    elif kind == "constant":
        density[:] = 1.5
    elif kind == "affine":
        density = 1.5 + .002 * np.asarray(geometry.node_depth)
    density *= np.asarray(params.wet_mask_z)
    content = jnp.asarray(density) * geometry.thickness * (params.dx_2d * params.dy)[..., None]
    velocities = tuple(jnp.asarray(generator.normal(0., .03, density.shape)) * params.wet_mask_z for _ in range(2))
    return params, depths, geometry, basis, jnp.asarray(surface), jnp.asarray(density), content, velocities


def _numpy_basis(depths, params):
    wet = np.asarray(params.wet_mask_z) > 0.
    widths = np.asarray(params.dz_node) * wet
    mass, moment = np.zeros(wet.shape), np.zeros(wet.shape)
    quadrature, weights = np.polynomial.legendre.leggauss(3)
    for column in np.ndindex(wet.shape[:2]):
        count = int(wet[column].sum())
        for level in range(count - 1):
            low, high = depths[level], depths[level + 1]
            positions = .5 * ((high - low) * quadrature + high + low)
            measures = .5 * (high - low) * weights
            for node, values in ((level, (high - positions) / (high - low)),
                                 (level + 1, (positions - low) / (high - low))):
                mass[column + (node,)] += np.sum(values * measures)
                moment[column + (node,)] += np.sum(positions * values * measures)
        if count:
            low, high = depths[count - 1], widths[column].sum()
            positions = .5 * ((high - low) * quadrature + high + low)
            measures = .5 * (high - low) * weights
            mass[column + (count - 1,)] += measures.sum()
            moment[column + (count - 1,)] += np.sum(positions * measures)
    return mass, moment / np.where(mass > 0., mass, 1.)


def _numpy_rates(density, velocities, geometry, params):
    wet = np.asarray(params.wet_mask_z) > 0.
    thickness = np.asarray(geometry.thickness)
    area = np.asarray(params.dx_2d) * float(params.dy)
    volume, content = np.zeros(wet.shape), np.zeros(wet.shape)
    for index in np.ndindex(wet.shape):
        for axis in (0, 1):
            neighbor = list(index)
            neighbor[axis] += 1
            if axis == 0:
                neighbor[axis] %= wet.shape[axis]
            elif neighbor[axis] == wet.shape[axis]:
                continue
            neighbor = tuple(neighbor)
            if not (wet[index] and wet[neighbor]):
                continue
            measure = float(params.dy)
            if axis == 1:
                cosine = np.asarray(params.cos_lat)
                measure = float(np.asarray(params.dx_2d)[index[:2]] / cosine[index[1]]) * .5 * (cosine[index[1]] + cosine[neighbor[1]])
            flux = (.5 * (np.asarray(velocities[axis])[index] + np.asarray(velocities[axis])[neighbor])
                    * .5 * (thickness[index] + thickness[neighbor]) * measure)
            centered = .5 * (density[index] + density[neighbor])
            volume[index] -= flux
            volume[neighbor] += flux
            content[index] -= flux * centered
            content[neighbor] += flux * centered
    surface_rate = volume.sum(axis=-1) / area
    widths = np.asarray(params.dz_node) * wet
    fractions = widths / np.where(widths.sum(axis=-1, keepdims=True) > 0., widths.sum(axis=-1, keepdims=True), 1.)
    relative_volume = np.cumsum(volume - area[..., None] * fractions * surface_rate[..., None], axis=-1)
    for index in np.ndindex(wet.shape):
        neighbor = index[:2] + (index[2] + 1,)
        if neighbor[-1] == wet.shape[-1] or not (wet[index] and wet[neighbor]):
            continue
        amount = relative_volume[index] * .5 * (density[index] + density[neighbor])
        content[index] -= amount
        content[neighbor] += amount
    return content, surface_rate


def _work(forces, velocities, content_rate, surface_rate, conjugates, geometry, params):
    mass = (params.dx_2d * params.dy)[..., None] * geometry.thickness
    kinetic = tuple(RHO_0 * mass * velocity * force for velocity, force in zip(velocities, forces, strict=True))
    potential = content_rate * conjugates[0], surface_rate * conjugates[1]
    residual = sum(float(jnp.sum(value)) for value in (*kinetic, *potential))
    floor = 64. * np.finfo(float).eps * sum(float(jnp.sum(jnp.abs(value))) for value in (*kinetic, *potential))
    return residual, floor


@pytest.mark.parametrize("stairs", [False, True])
def test_pressure_potential_basis_matches_independent_gauss_moments_mass_and_derivatives(stairs):
    params, depths, geometry, basis, surface, _, content, _ = _case(stairs)
    mass, mean = _numpy_basis(depths, params)
    widths = np.asarray(params.dz_node) * np.asarray(params.wet_mask_z)
    assert np.max(np.abs(mass - widths)) <= 64. * np.finfo(float).eps * np.max(widths)
    assert np.max(np.abs(mean - np.asarray(basis.mean_depth))) <= 64. * np.finfo(float).eps * np.max(np.abs(mean))
    area = np.asarray(params.dx_2d) * float(params.dy)
    actual_mean = -np.asarray(surface)[..., None] + np.asarray(geometry.scale)[..., None] * mean
    expected = .5 * RHO_0 * G_EARTH * np.sum(area * np.asarray(surface) ** 2) - G_EARTH * np.sum(np.asarray(content) * actual_mean)
    actual = float(potential_energy(content, surface, geometry, basis, params))
    assert abs(actual - expected) <= 64. * np.finfo(float).eps * abs(expected)

    def energy(values, eta):
        return potential_energy(values, eta, _geometry(params, depths, eta), basis, params)

    automatic = jax.grad(energy, argnums=(0, 1))(content, surface)
    explicit = potential_conjugates(content, surface, geometry, basis, params)
    for actual, expected in zip(automatic, explicit, strict=True):
        assert float(jnp.max(jnp.abs(actual - expected))) <= 64. * np.finfo(float).eps * float(jnp.max(jnp.abs(expected)))


@pytest.mark.parametrize("stairs", [False, True])
@pytest.mark.parametrize("flat", [False, True])
def test_pressure_energy_adjoint_matches_explicit_transpose_and_independent_transport(stairs, flat, record_property):
    params, _, geometry, basis, surface, density, content, velocities = _case(stairs, flat)
    faces = coordinate_layer_faces(*velocities, geometry, params)
    rates = centered_content_rate(density, faces, geometry, params)
    independent = _numpy_rates(np.asarray(density), velocities, geometry, params)
    for actual, expected in zip(rates, independent, strict=True):
        assert float(jnp.max(jnp.abs(actual - expected))) <= 64. * np.finfo(float).eps * max(1e-30, float(np.max(np.abs(expected))))
    conjugates = potential_conjugates(content, surface, geometry, basis, params)
    explicit = jax.jit(lambda field: energy_adjoint_force(field, content, surface, geometry, basis, params))(density)
    automatic = transpose_force(density, conjugates, geometry, params)
    for actual, expected in zip(explicit, automatic, strict=True):
        assert float(jnp.max(jnp.abs(actual - expected))) <= 64. * np.finfo(float).eps * float(jnp.max(jnp.abs(expected)))
    residual, floor = _work(explicit, velocities, *independent, conjugates, geometry, params)
    record_property("reversible_work_residual_watts", residual)
    record_property("reversible_work_64eps_floor_watts", floor)
    assert abs(residual) <= floor


@pytest.mark.parametrize("kind", ["zero", "constant"])
def test_actual_width_paired_chain_rule_surface_work_closes(kind, record_property):
    params, _, geometry, basis, surface, density, content, velocities = _case(kind=kind)
    pressure = hydrostatic_pressure(density, surface, geometry, params)
    rates = _numpy_rates(np.asarray(density), velocities, geometry, params)
    forces = paired_chain_rule_force(pressure, density, geometry, params)
    conjugates = potential_conjugates(content, surface, geometry, basis, params)
    residual, floor = _work(forces, velocities, *rates, conjugates, geometry, params)
    record_property("paired_chain_surface_work_residual_watts", residual)
    record_property("paired_chain_surface_work_64eps_floor_watts", floor)
    assert abs(residual) <= floor


def test_pressure_density_force_and_centered_rates_do_not_read_finite_dry_sentinels():
    params, _, geometry, basis, surface, density, content, velocities = _case()
    evaluate = jax.jit(lambda field: energy_adjoint_force(field, content, surface, geometry, basis, params))
    reference = evaluate(density)
    for sentinel in (123., -1e6):
        actual = evaluate(jnp.where(params.wet_mask_z > 0., density, sentinel))
        for changed, expected in zip(actual, reference, strict=True):
            np.testing.assert_array_equal(changed, expected)


@pytest.mark.parametrize("sentinel", [123., -1e6, np.nan])
def test_potential_conjugate_ignores_dry_surface_values(sentinel):
    params, depths, geometry, basis, surface, _, content, _ = _case()
    changed = jnp.where(params.wet_mask > 0., surface, sentinel)

    def energy(eta):
        return potential_energy(content, eta, _geometry(params, depths, eta), basis, params)

    derivative = jax.grad(energy)(changed)
    assert bool(jnp.isfinite(energy(changed)))
    explicit = potential_conjugates(content, changed, geometry, basis, params)[1]
    np.testing.assert_array_equal(np.asarray(explicit)[np.asarray(params.wet_mask) == 0.], np.asarray(derivative)[np.asarray(params.wet_mask) == 0.])


def test_pressure_surface_weight_mismatch_is_detected_without_relaxing_work_gate():
    result = _candidate_diagnostics(True, False, "zero")["candidate_gates"]
    assert not result["existing_chain_rule"]["pressure_work_passed"]
    assert result["actual_width_chain_rule"]["pressure_work_passed"]
    assert result["energy_adjoint"]["pressure_work_passed"]


def test_energy_pairing_alone_does_not_qualify_affine_physical_rest_on_stairs():
    result = _candidate_diagnostics(True, True, "affine")["candidate_gates"]
    assert result["actual_width_chain_rule"]["rest_gate_passed"]
    assert not result["actual_width_chain_rule"]["pressure_work_passed"]
    assert result["energy_adjoint"]["pressure_work_passed"]
    assert not result["energy_adjoint"]["rest_gate_passed"]


def test_pressure_full_wet_vertical_refinement_does_not_pass_registered_second_order_gate():
    result = _vertical_refinement()
    assert not result["second_order_diagnostic_passed"]
    assert not result["changes_interface_gate"]


def _candidate_diagnostics(stairs, flat, kind):
    params, _, geometry, basis, surface, density, content, velocities = _case(stairs, flat, kind)
    faces = coordinate_layer_faces(*velocities, geometry, params)
    rates = _numpy_rates(np.asarray(density), velocities, geometry, params)
    conjugates = potential_conjugates(content, surface, geometry, basis, params)
    pressure = hydrostatic_pressure(density, surface, geometry, params)
    candidate_forces = {"existing_chain_rule": coordinate_pressure_gradient(pressure, density, geometry, params),
                        "actual_width_chain_rule": paired_chain_rule_force(pressure, density, geometry, params),
                        "energy_adjoint": energy_adjoint_force(density, content, surface, geometry, basis, params)}
    force_floor = (64. * np.finfo(float).eps * float(jnp.max(jnp.abs(pressure))) / RHO_0
                   * max(float(jnp.max(params.inv_dx)), float(params.inv_dy)))
    results = {}
    for name, forces in candidate_forces.items():
        residual, floor = _work(forces, velocities, *rates, conjugates, geometry, params)
        maximum = max(float(jnp.max(jnp.abs(force))) for force in forces)
        results[name] = {"work_residual_watts": residual, "work_64eps_floor_watts": floor,
                         "pressure_work_passed": bool(abs(residual) <= floor), "maximum_force_m_per_s2": maximum,
                         "rest_force_64eps_floor_m_per_s2": force_floor,
                         "rest_gate_applicable": flat and kind in ("zero", "constant", "affine"),
                         "rest_gate_passed": bool(maximum <= force_floor) if flat and kind in ("zero", "constant", "affine") else None}
    divergence = _face_transport_divergence(*faces, params)
    relative = relative_vertical_transport(divergence, geometry)
    zeros = jnp.zeros_like(density)
    donor = (params.dx_2d * params.dy)[..., None] * params.dz_node * _advection_scalar(density, zeros, zeros, relative, params, face_transport=faces)
    mixing_work = float(jnp.sum((donor - rates[0]) * conjugates[0]))
    result = {"stairs": stairs, "flat_eta": flat, "density": kind, "candidate_gates": results,
              "original_donor_minus_reversible_potential_transfer_watts": mixing_work, "production_promotion": False}
    if kind == "affine":
        area = np.asarray(params.dx_2d) * float(params.dy)
        depth = np.asarray(basis.column_depth)
        eta = np.asarray(surface)
        column = np.asarray(params.wet_mask) > 0.
        exact_mass = area * (1.5 * (depth + eta) + .001 * (depth ** 2 - eta ** 2))
        represented_mass = np.asarray(content).sum(axis=-1)
        exact_potential = .5 * RHO_0 * G_EARTH * np.sum(area[column] * eta[column] ** 2)
        exact_potential -= G_EARTH * np.sum(area[column] * (.75 * (depth[column] ** 2 - eta[column] ** 2)
                                                          + .002 / 3. * (depth[column] ** 3 + eta[column] ** 3)))
        represented_potential = float(potential_energy(content, surface, geometry, basis, params))
        result["affine_point_vs_constant_tail_representation"] = {
            "nodal_anomaly_mass_minus_analytic_affine_kg": float(np.sum((represented_mass - exact_mass)[column])),
            "max_column_anomaly_mass_difference_kg_per_m2": float(np.max(np.abs((represented_mass - exact_mass)[column] / area[column]))),
            "constant_tail_potential_minus_analytic_affine_joules": represented_potential - exact_potential,
            "interpretation": "saved nodal point values lie on the affine profile, but constant extension below last wet node is not the same continuous profile; this gap is not a numerical source or proof of production instability"}
    return result


def _vertical_refinement():
    records = []
    for count in (9, 17, 33):
        depths = np.linspace(0., 2000., count)
        _, params, _ = _fixture(stairs=False, nx=16, ny=8, depths=depths)
        longitude = 2. * jnp.pi * jnp.arange(16)[:, None] / 16.
        latitude = jnp.linspace(-jnp.pi / 6., jnp.pi / 6., 8)[None, :]
        surface = jnp.cos(longitude) * jnp.cos(latitude)
        geometry = _geometry(params, depths, surface)
        basis = make_potential_basis(depths, params)
        density = 1.5 + .002 * geometry.node_depth
        content = density * (params.dx_2d * params.dy)[..., None] * geometry.thickness
        velocities = (jnp.broadcast_to((.03 * jnp.sin(longitude) * jnp.cos(latitude))[..., None], density.shape),
                      jnp.broadcast_to((.02 * jnp.cos(longitude) * jnp.sin(latitude))[..., None], density.shape))
        pressure = hydrostatic_pressure(density, surface, geometry, params)
        chain = paired_chain_rule_force(pressure, density, geometry, params)
        paired = energy_adjoint_force(density, content, surface, geometry, basis, params)
        mass = (params.dx_2d * params.dy)[..., None] * geometry.thickness
        error = float(jnp.sqrt(sum(jnp.sum(mass * (actual - expected) ** 2) for actual, expected in zip(chain, paired, strict=True)) / jnp.sum(mass)))
        rates = _numpy_rates(np.asarray(density), velocities, geometry, params)
        conjugates = potential_conjugates(content, surface, geometry, basis, params)
        residual, floor = _work(chain, velocities, *rates, conjugates, geometry, params)
        records.append({"node_count": count, "force_difference_l2_m_per_s2": error,
                        "chain_work_residual_watts": residual, "work_64eps_floor_watts": floor,
                        "work_residual_watts_per_m2": residual / float(jnp.sum(params.dx_2d * params.dy))})
    ratios = [before["force_difference_l2_m_per_s2"] / after["force_difference_l2_m_per_s2"]
              for before, after in zip(records[:-1], records[1:], strict=True)]
    return {"scope": "full wet fixed-domain vertical refinement, not stairs or complete PDE MMS", "records": records,
            "ratios": ratios, "second_order_diagnostic_passed": all(3.3 <= ratio <= 4.5 for ratio in ratios), "changes_interface_gate": False}
