"""Independent capacity/adjoint/budget controls for fixed reference partial cells."""

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tests.support.fd.fixed_partial import fixed_partial_case as _case
from zhenmode.model.config import C_P, G_EARTH, RHO_0
from zhenmode.model.diagnostics.snapshot import compute_budget_diagnostics
from zhenmode.model.solver.dynamics.projection import (
    _column_projection_diagonal,
    _project_column_divergence,
)
from zhenmode.model.solver.dynamics.transport import (
    _advection_flux_form,
    _advection_scalar,
    _column_divergence,
    _layer_face_transports,
    _match_layer_face_transports,
    _sum_layer_transports,
    _vertical_transport_iface,
)
from zhenmode.model.solver.geometry.fd_metrics import make_fd_params
from zhenmode.model.solver.numerics.horizontal import (
    _apply_polar_cap,
    _divergence_h,
    _gradient_conservative_3d,
    _gradient_face_gated_3d,
    _horizontal_tracer_diffusion,
)
from zhenmode.model.solver.numerics.vertical import _d2_dz2_flux
from zhenmode.model.solver.physics.surface import _surface_heat_weights


def test_capacity_and_snapshot_follow_bed_including_shallow_columns():
    grid, h, (_, init, _, p, _) = _case()
    np.testing.assert_array_equal(p.dz_node * p.wet_mask_z, h)
    np.testing.assert_array_equal(p.H_sw * p.wet_mask, grid.depth)
    np.testing.assert_allclose(np.sum(p.dz_norm, axis=-1), grid.wet_mask, rtol=0, atol=2e-16)
    state = init()
    budget = compute_budget_diagnostics(state, grid, column_geometry="fixed_partial_v1")
    volume = float(np.sum(grid.depth * grid.dx_2d * grid.dy))
    assert budget.total_volume_m3 == pytest.approx(volume, rel=2e-16)
    assert budget.heat_content_J == pytest.approx(RHO_0 * C_P * 15.0 * volume, rel=3e-16)


def test_face_overlap_uses_shared_water_with_independent_expected_flux():
    _, h, (_, _, _, p, _) = _case()
    u = jnp.ones(h.shape)
    x, y = _layer_face_transports(u, jnp.zeros_like(u), p)
    expected = np.zeros(h.shape)
    expected[0, :, 0] = 0.5
    expected[1, :, 0:2] = [2.5, 4.5]
    np.testing.assert_array_equal(x[0], expected)
    np.testing.assert_array_equal(x[1:], 0.0)
    np.testing.assert_array_equal(y, 0.0)
    np.testing.assert_array_equal(
        np.sum(_divergence_h(u, jnp.zeros_like(u), p) * h, axis=-1),
        _column_divergence(u, jnp.zeros_like(u), p),
    )


def test_divergence_and_gradient_are_volume_weighted_negative_adjoints():
    grid, h, (_, _, _, p, _) = _case()
    rng = np.random.default_rng(142507)
    u, v, t = (jnp.asarray(a) for a in rng.normal(size=(3, *h.shape)))
    volume = jnp.asarray(grid.dx_2d * grid.dy)[..., None] * h
    div = _divergence_h(u, v, p)
    gx, gy = _gradient_conservative_3d(t, p)
    a = float(jnp.sum(t * div * volume))
    b = float(jnp.sum((u * gx + v * gy) * volume))
    assert abs(a + b) < 1e-13 * max(abs(a), abs(b))
    assert abs(float(jnp.sum(div * volume))) < 1e-13 * float(jnp.sum(abs(div * volume)))


@pytest.mark.parametrize("operator", ["horizontal", "vertical"])
def test_diffusion_conserves_inventory_and_dissipates_variance(operator):
    grid, h, (_, _, _, p, _) = _case()
    t = jnp.asarray(np.random.default_rng(2347).normal(size=h.shape))
    if operator == "horizontal":
        tendency = _horizontal_tracer_diffusion(t, p._replace(kappa_h=3.0))
    else:
        tendency = _d2_dz2_flux(t, 3.0, p)
    volume = jnp.asarray(grid.dx_2d * grid.dy)[..., None] * h
    assert abs(float(jnp.sum(tendency * volume))) < 1e-13 * float(jnp.sum(abs(tendency * volume)))
    assert float(jnp.sum(t * tendency * volume)) < 0.0


def test_constant_tracer_has_pointwise_continuity_and_variable_tracer_boundary_budget():
    grid, h, (_, _, _, p, _) = _case()
    rng = np.random.default_rng(9512)
    u, v, t = (jnp.asarray(a) for a in rng.normal(size=(3, *h.shape)))
    faces = _layer_face_transports(u, v, p)
    fz = _vertical_transport_iface(u, v, p, face_transport=faces)
    constant = _advection_scalar(jnp.full(h.shape, 4.0), u, v, fz, p)
    np.testing.assert_allclose(constant, 0.0, rtol=0, atol=1e-18)
    tendency, top = _advection_scalar(t, u, v, fz, p, return_boundary=True)
    area = jnp.asarray(grid.dx_2d * grid.dy)
    a = float(jnp.sum(tendency * area[..., None] * h))
    b = float(jnp.sum(top * area))
    assert abs(a - b) < 1e-13 * max(abs(a), abs(b))
    targets = tuple(_sum_layer_transports(f) * 1.3 for f in faces)
    corrected = _match_layer_face_transports(u, v, targets, p)
    for value, target in zip(corrected, targets, strict=True):
        np.testing.assert_allclose(_sum_layer_transports(value), target, rtol=1e-14, atol=1e-15)


def test_projection_diagonal_matches_explicit_matrix_and_reduces_transport():
    grid, h, (_, _, _, p, _) = _case(projection_preconditioner="jacobi", projection_niter=200)

    def matrix(psi):
        gx, gy = _gradient_conservative_3d(psi[..., None], p)
        return -_column_divergence(gx, gy, p) * p.dx_2d * p.dy

    explicit = np.asarray(jax.jacfwd(matrix)(jnp.zeros((4, 4)))).reshape(16, 16)
    np.testing.assert_allclose(
        np.asarray(_column_projection_diagonal(p)).ravel(),
        np.diag(explicit),
        rtol=1e-13,
        atol=1e-14,
    )
    np.testing.assert_allclose(explicit, explicit.T, rtol=1e-13, atol=1e-14)
    assert np.linalg.eigvalsh(explicit).min() > -1e-12
    rng = np.random.default_rng(12027)
    u, v = (jnp.asarray(a) for a in rng.normal(size=(2, *h.shape)))
    before = float(jnp.linalg.norm(_column_divergence(u, v, p)))
    corrected = _project_column_divergence(u, v, p, 1.0)
    after = float(jnp.linalg.norm(_column_divergence(*corrected, p)))
    assert after < before * 1e-10


def test_polar_filter_preserves_partial_cell_inventory():
    grid, h, (_, _, _, p, _) = _case()
    p = p._replace(polar_cap_rows=1, polar_cap_taper=1)
    t = jnp.asarray(np.random.default_rng(8701).normal(size=h.shape)) * p.wet_mask_z
    filtered = _apply_polar_cap(t, p.wet_mask_z, p)
    np.testing.assert_allclose(
        np.sum(filtered * h, axis=0), np.sum(t * h, axis=0), rtol=1e-13, atol=1e-14
    )


@pytest.mark.parametrize("depth", [None, 10.0])
def test_heat_deposition_uses_true_shallow_capacity_and_full_column_limit(depth):
    grid, h, (_, _, _, p, _) = _case(mixed_layer_depth_m=depth)
    weights = _surface_heat_weights(p)
    np.testing.assert_allclose(np.sum(weights * h, axis=-1), grid.wet_mask, rtol=0, atol=2e-16)
    assert weights[0, 0, 0] == 2.0  # .5 m, not the global 2.5 m surface cell.


@pytest.mark.parametrize("matched", [False, True])
def test_actual_compiled_step_retains_rest_and_closes_prescribed_heat_inventory(matched):
    grid, h, (step, init, _, p, _) = _case(
        forcing=(np.zeros((4, 4)), np.zeros((4, 4)), np.full((4, 4), 80.0)),
        match_barotropic_transport=matched,
        process_time_scheme="symmetric_fast_v3" if matched else "legacy",
    )
    start = init()
    end = step(start)
    jax.block_until_ready(end)
    assert np.isfinite(np.asarray(end.T)).all()
    volume = grid.dx_2d[..., None] * grid.dy * h
    change = float(np.sum((np.asarray(end.T) - np.asarray(start.T)) * volume) * RHO_0 * C_P)
    expected = float(np.sum(grid.dx_2d * grid.dy * grid.wet_mask) * 80.0 * p.dt)
    assert change == pytest.approx(expected, rel=2e-9)


def test_grid_mask_conflict_and_deep_truncation_are_rejected():
    grid, h, _ = _case()
    with pytest.raises(ValueError, match="wet nodes"):
        make_fd_params(
            replace(grid, wet_mask_3d=np.ones(h.shape)), column_geometry="fixed_partial_v1"
        )
    with pytest.raises(ValueError, match="covering"):
        make_fd_params(
            replace(grid, depth=np.where(grid.depth > 0, 31.0, 0.0)),
            column_geometry="fixed_partial_v1",
        )


def test_partial_geometry_rejects_unqualified_old_ice_and_diffusion():
    with pytest.raises(ValueError, match="sea-ice operators"):
        _case(dynamic_ice=True)
    with pytest.raises(ValueError, match="conservative_kv"):
        _case(conservative_kv=False)


def test_all_reference_contacts_cover_actual_common_water_in_both_directions():
    grid, h, (_, _, _, p, _) = _case(cross_nodes=True)
    for axis in (0, 1):
        expected = np.minimum(grid.depth, np.roll(grid.depth, -1, axis=axis))
        if axis == 1:
            expected[:, -1] = 0.0
        np.testing.assert_allclose(
            _sum_layer_transports(p.face_contacts[axis]), expected, rtol=0, atol=0
        )
    assert p.face_contacts[0][1, 0, 0, 0] == 0.5  # [2.5,3] joins nodes 0 and 5.
    assert p.face_contacts[0][1, 1, 0, 1] == 4.0  # [10,14] joins nodes 5 and 15.
    assert p.face_contacts[0][1, 2, 0, 2] == 4.5  # [22.5,27] joins nodes 15 and 30.
    destroyed = np.sum(np.asarray(p.face_contacts[0][0]), axis=-1)
    assert np.any(destroyed != np.minimum(grid.depth, np.roll(grid.depth, -1, axis=0)))


def test_cross_node_balance_adjoint_projection_and_scalar_inventory():
    grid, h, (_, _, _, p, _) = _case(cross_nodes=True, projection_preconditioner="jacobi")
    rng = np.random.default_rng(504219)
    u, v, t = (jnp.asarray(a) for a in rng.normal(size=(3, *h.shape)))
    volume = jnp.asarray(grid.dx_2d * grid.dy)[..., None] * h
    divergence = _divergence_h(u, v, p)
    gx, gy = _gradient_conservative_3d(t, p)
    a = float(jnp.sum(t * divergence * volume))
    b = float(jnp.sum((u * gx + v * gy) * volume))
    assert abs(a + b) < 1e-13 * max(abs(a), abs(b))
    faces = _layer_face_transports(u, v, p)
    fz = _vertical_transport_iface(u, v, p, face_transport=faces)
    np.testing.assert_allclose(
        _advection_scalar(jnp.full(h.shape, 4.0), u, v, fz, p), 0.0, rtol=0, atol=1e-18
    )
    tendency, top = _advection_scalar(t, u, v, fz, p, return_boundary=True)
    a = float(jnp.sum(tendency * volume))
    b = float(jnp.sum(top * grid.dx_2d * grid.dy))
    assert abs(a - b) < 1e-13 * max(abs(a), abs(b))
    corrected = _project_column_divergence(u, v, p, 1.0)
    assert float(jnp.linalg.norm(_column_divergence(*corrected, p))) < 1e-10 * float(
        jnp.linalg.norm(_column_divergence(u, v, p))
    )

    def matrix(psi):
        return (
            -_column_divergence(*_gradient_conservative_3d(psi[..., None], p), p) * p.dx_2d * p.dy
        )

    explicit = np.asarray(jax.jacfwd(matrix)(jnp.zeros((4, 4)))).reshape(16, 16)
    np.testing.assert_allclose(
        np.diag(explicit),
        np.asarray(_column_projection_diagonal(p)).ravel(),
        rtol=1e-13,
        atol=1e-14,
    )
    np.testing.assert_allclose(explicit, explicit.T, rtol=1e-13, atol=1e-14)


def test_cross_node_pressure_uses_common_depth_and_rejects_naive_nodal_difference():
    from zhenmode.model.solver.dynamics.pressure import (
        _compute_hydrostatic_pressure,
        _compute_pressure_gradient,
    )

    grid, h, (step, init, _, p, _) = _case(
        cross_nodes=True, match_barotropic_transport=True, process_time_scheme="symmetric_fast_v3"
    )
    state = init(np.full(h.shape, 16.0), np.full(h.shape, 35.0))
    # Constant nonzero density anomaly creates a pressure increasing with depth.
    # The actual common-depth difference is zero; the naive cross-node one is not.
    correct = _compute_pressure_gradient(state, p)
    for value in correct:
        np.testing.assert_allclose(value, 0.0, rtol=0, atol=1e-20)
    naive = _gradient_conservative_3d(_compute_hydrostatic_pressure(state, p), p)
    assert max(float(jnp.max(jnp.abs(a))) for a in naive) > 1e-8
    end = step(state)
    jax.block_until_ready(end)
    for name in ("T", "S", "eta", "u", "v"):
        np.testing.assert_allclose(
            np.asarray(getattr(end, name)), np.asarray(getattr(state, name)), rtol=0, atol=1e-14
        )


def test_cross_node_teos_reference_rest_uses_the_same_pressure_profile():
    from zhenmode.model.solver.dynamics.pressure import _compute_pressure_gradient

    grid, h, (step, init, _, p, _) = _case(
        cross_nodes=True,
        thermodynamics="teos10_reference",
        match_barotropic_transport=True,
        process_time_scheme="symmetric_fast_v3",
    )
    state = init(np.full(h.shape, 16.0), np.full(h.shape, 35.16504))
    for force in _compute_pressure_gradient(state, p):
        np.testing.assert_allclose(force, 0.0, rtol=0, atol=1e-20)
    end = step(state)
    jax.block_until_ready(end)
    for name in ("T", "S", "eta", "u", "v"):
        np.testing.assert_allclose(
            np.asarray(getattr(end, name)), np.asarray(getattr(state, name)), rtol=0, atol=1e-14
        )


@pytest.mark.parametrize("fct,donor", [(False, False), (False, True), (True, True)])
def test_cross_node_tracer_choices_and_diffusion_have_real_boundary_budgets(fct, donor):
    grid, h, (_, _, _, p, _) = _case(cross_nodes=True, fct_adv=fct, monotone_adv=donor)
    rng = np.random.default_rng(94061)
    u, v, t = (jnp.asarray(a) for a in rng.normal(size=(3, *h.shape)))
    volume = jnp.asarray(grid.dx_2d * grid.dy)[..., None] * h
    diffusion = _horizontal_tracer_diffusion(t, p._replace(kappa_h=3.0))
    assert abs(float(jnp.sum(diffusion * volume))) < 1e-13 * float(jnp.sum(abs(diffusion * volume)))
    assert float(jnp.sum(t * diffusion * volume)) < 0.0
    fz = _vertical_transport_iface(u, v, p)
    rate, top = _advection_scalar(t, u, v, fz, p, return_boundary=True)
    a = float(jnp.sum(rate * volume))
    b = float(jnp.sum(top * grid.dx_2d * grid.dy))
    assert abs(a - b) < 1e-13 * max(abs(a), abs(b))
    np.testing.assert_allclose(
        _advection_scalar(jnp.full(h.shape, 4.0), u, v, fz, p), 0.0, rtol=0, atol=1e-18
    )


def test_cross_node_actual_step_momentum_gradient_matches_finite_difference():
    grid, h, (step, init, _, p, _) = _case(
        cross_nodes=True, match_barotropic_transport=True, process_time_scheme="symmetric_fast_v3"
    )
    rng = np.random.default_rng(496210)
    temperature = jnp.asarray(16.0 + rng.normal(size=h.shape) * 0.02)
    salinity = np.full(h.shape, 35.0)
    probe = jnp.asarray(rng.normal(size=h.shape) * h)

    def objective(t):
        return jnp.sum(step(init(t, salinity)).u * probe)

    derivative = jax.grad(objective)(temperature)
    index = (1, 1, 1)
    epsilon = 1e-4
    plus = temperature.at[index].add(epsilon)
    minus = temperature.at[index].add(-epsilon)
    reference = float((objective(plus) - objective(minus)) / (2.0 * epsilon))
    assert abs(reference) > 1e-10
    assert float(derivative[index]) == pytest.approx(reference, rel=1e-7, abs=1e-15)


@pytest.mark.parametrize("thermodynamics", ["linear", "teos10_reference"])
def test_partial_contact_arrays_and_actual_step_preserve_explicit_float32(thermodynamics):
    grid, h, (step, init, _, p, _) = _case(
        cross_nodes=True,
        thermodynamics=thermodynamics,
        dtype="float32",
        match_barotropic_transport=True,
        process_time_scheme="symmetric_fast_v3",
    )
    for values in (*p.face_contacts, *p.contact_depths_m, p.node_depth_m, p.dz_node):
        assert values.dtype == jnp.float32
    salt = 35.16504 if thermodynamics == "teos10_reference" else 35.0
    state = init(np.full(h.shape, 16.0), np.full(h.shape, salt))
    for _ in range(3):
        state = step(state)
    jax.block_until_ready(state)
    for values in state:
        assert values.dtype == jnp.float32
        assert np.isfinite(np.asarray(values)).all()
    # Uniform CT/T and reference salinity at common pressure remain at rest.
    for values in (state.u, state.v, state.eta):
        np.testing.assert_array_equal(values, np.zeros_like(values))


def test_public_partial_metrics_dispatch_to_contacts_and_reject_incompatible_metrics():
    grid, h, (_, _, _, physics, _) = _case(cross_nodes=True)
    metrics = make_fd_params(grid, column_geometry="fixed_partial_v1")
    assert metrics.column_geometry == "fixed_partial_v1"
    u = jnp.zeros_like(physics.dz_node).at[1, 0, 1].set(1.0)
    v = jnp.zeros_like(u)
    # Independently: the [2.5,3] contact carries .5*(1+0)*.5=.25,
    # arriving at shallow node 0 of width 3; divergence is .25/(dx*3).
    expected = 0.25 / (grid.dx_2d[0, 0] * 3.0)
    assert float(_divergence_h(u, v, metrics)[0, 0, 0]) == pytest.approx(expected, rel=1e-15)
    for actual, reference in zip(
        _layer_face_transports(u, v, metrics), _layer_face_transports(u, v, physics), strict=True
    ):
        np.testing.assert_array_equal(actual, reference)
    invalid = grid.dx_2d.copy()
    invalid[:, 1] *= 1.1
    with pytest.raises(ValueError, match="compatible dx"):
        make_fd_params(replace(grid, dx_2d=invalid), column_geometry="fixed_partial_v1")


@pytest.mark.parametrize("split", [False, True])
def test_partial_eta_force_is_cancelled_at_each_node_before_external_step(split):
    from zhenmode.model.solver.dynamics.pressure import _reference_depth_gradient
    from zhenmode.model.solver.dynamics.tendencies import _compute_momentum_residual

    _, h, (_, init, _, p, _) = _case(cross_nodes=True, mode_split=split)
    eta = jnp.asarray(np.random.default_rng(208401).normal(size=(4, 4))) * p.wet_mask
    state = init()._replace(eta=eta)
    for residual in _compute_momentum_residual(state, p):
        np.testing.assert_allclose(residual, 0.0, rtol=0, atol=1e-18)
    full = _gradient_conservative_3d(eta[..., None], p)
    averaged = _reference_depth_gradient(eta, p)
    # Planted old cancellation leaves node-dependent eta shear at bottom steps.
    wrong = [
        G_EARTH * (mean[..., None] - node) * p.wet_mask_z
        for mean, node in zip(averaged, full, strict=True)
    ]
    assert max(float(jnp.max(jnp.abs(a))) for a in wrong) > 1e-9


@pytest.mark.parametrize("scheme", ["legacy", "symmetric_fast_v3"])
@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_actual_barotropic_cap_uses_depth_capacity_while_eta_uses_area(scheme, dtype):
    from zhenmode.model.solver.dynamics.barotropic import _free_surface_step_fd

    matched = scheme != "legacy"
    grid, _, (_, _, _, p, _) = _case(
        cross_nodes=True,
        dtype=dtype,
        match_barotropic_transport=matched,
        process_time_scheme=scheme,
    )
    p = p._replace(polar_cap_rows=1, polar_cap_taper=1)
    values = (
        np.broadcast_to(np.array([-2.0, -1.0, 1.0, 2.0])[:, None], (4, 4)).copy() * grid.wet_mask
    )
    u = jnp.asarray(values, dtype=dtype)
    v = u * p.interior_mask_z[..., 0]
    # Zero-duration probe isolates the actual stepper's eta/velocity filters.
    eta, capped_u, capped_v = _free_surface_step_fd(u, u, v, p, dt_half=0.0)
    fractions = np.array([1.0, 0.5, 0.5, 1.0])[None, :]
    depth = grid.depth
    expected = []
    for field, weights in ((values, grid.wet_mask), (values, depth), (np.asarray(v), depth)):
        total = weights.sum(axis=0, keepdims=True)
        mean = (field * weights).sum(axis=0, keepdims=True) / np.where(total > 0, total, 1)
        expected.append((field + fractions * (mean - field)) * grid.wet_mask)
    tolerance = 32 * np.finfo(dtype).eps
    for actual, reference in zip((eta, capped_u, capped_v), expected, strict=True):
        assert actual.dtype == jnp.dtype(dtype)
        np.testing.assert_allclose(actual, reference, rtol=tolerance, atol=tolerance)
    for before, after in ((u, capped_u), (v, capped_v)):
        np.testing.assert_allclose(
            np.sum(np.asarray(after) * depth, axis=0),
            np.sum(np.asarray(before) * depth, axis=0),
            rtol=tolerance,
            atol=tolerance,
        )
    np.testing.assert_allclose(
        np.sum(np.asarray(eta) * grid.wet_mask, axis=0),
        np.sum(values, axis=0),
        rtol=tolerance,
        atol=tolerance,
    )
    wrong = _apply_polar_cap(u, p.wet_mask, p)
    assert np.max(np.abs(np.sum(np.asarray(wrong - u) * depth, axis=0))) > 1.0


def test_contact_momentum_advection_matches_independent_all_pair_material_balance():
    from zhenmode.model.solver.numerics.contacts import contact_material_derivative

    grid, h, (_, _, _, p, _) = _case(cross_nodes=True)
    rng = np.random.default_rng(208402)
    u, v = rng.normal(size=(2, *h.shape)) * grid.wet_mask_3d
    tops, bottoms = np.cumsum(h, axis=-1) - h, np.cumsum(h, axis=-1)
    reference = [np.zeros_like(h), np.zeros_like(h)]
    for axis in (0, 1):
        for i in range(4):
            for j in range(4):
                if axis == 1 and j == 3:
                    continue
                ii, jj = ((i + 1) % 4, j) if axis == 0 else (i, j + 1)
                for k in range(4):
                    for level in range(4):
                        if h[i, j, k] == 0 or h[ii, jj, level] == 0:
                            continue
                        weight = max(
                            0.0,
                            min(bottoms[i, j, k], bottoms[ii, jj, level])
                            - max(tops[i, j, k], tops[ii, jj, level]),
                        )
                        velocity = u if axis == 0 else v
                        flux = weight * 0.5 * (velocity[i, j, k] + velocity[ii, jj, level])
                        if axis == 1:
                            flux *= 0.5 * (grid.cos_lat[j] + grid.cos_lat[jj])
                        for field, result in zip((u, v), reference, strict=True):
                            term = 0.5 * flux * (field[ii, jj, level] - field[i, j, k])
                            left = grid.dx_2d[i, j] if axis == 0 else grid.dy * grid.cos_lat[j]
                            right = grid.dx_2d[ii, jj] if axis == 0 else grid.dy * grid.cos_lat[jj]
                            result[i, j, k] += term / (left * h[i, j, k])
                            result[ii, jj, level] += term / (right * h[ii, jj, level])
    faces = _layer_face_transports(jnp.asarray(u), jnp.asarray(v), p)
    for field, expected in zip((u, v), reference, strict=True):
        np.testing.assert_allclose(
            contact_material_derivative(jnp.asarray(field), faces, p),
            expected,
            rtol=2e-13,
            atol=1e-18,
        )
    np.testing.assert_array_equal(
        contact_material_derivative(jnp.full(h.shape, 17.0), faces, p), 0.0
    )

    def independent_filter(field):
        transformed = np.fft.fft(field, axis=0) * np.asarray(p.dealias_lon_mask)
        padded = np.pad(
            np.fft.ifft(transformed, axis=0).real, ((0, 0), (2, 2), (0, 0)), mode="edge"
        )
        return (
            sum(w * padded[:, offset : offset + 4] for offset, w in enumerate((1, 4, 6, 4, 1)))
            / 16
            * grid.wet_mask_3d
        )

    actual = _advection_flux_form(jnp.asarray(u), jnp.asarray(v), jnp.zeros_like(p.dz_node), p)
    for value, expected in zip(actual, reference, strict=True):
        np.testing.assert_allclose(value, independent_filter(-expected), rtol=2e-13, atol=1e-18)
    # Same-index stencil misses the actual shallow-to-next-node contribution.
    probe = jnp.zeros_like(p.dz_node).at[1, :, 1].set(1.0)
    fluxes = _layer_face_transports(probe, jnp.zeros_like(probe), p)
    material = contact_material_derivative(probe, fluxes, p)
    assert float(material[0, 0, 0]) == pytest.approx(0.125 / (3 * grid.dx_2d[0, 0]), rel=1e-15)
    legacy = probe * _gradient_face_gated_3d(probe, p)[0]
    assert legacy[0, 0, 0] == 0.0
