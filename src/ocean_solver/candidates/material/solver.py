"""Opt-in moving nodal tracer inventory on the original linear FD dynamics.

This is a mass-lumped point-sample approximation, not the C-grid migration or
fully nonlinear momentum. Only float64, unfiltered material surfaces and the
explicitly supported physics are accepted. Production defaults are untouched.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from ocean_solver._compat import preserve_legacy_names
from ocean_solver.configuration import C_P, RHO_0
from ocean_solver.fd.barotropic import _barotropic_subcycle_transport
from ocean_solver.fd.horizontal import (
    _biharmonic_h,
    _horizontal_biharmonic_tracer,
    _horizontal_tracer_diffusion,
    _laplacian_h,
)
from ocean_solver.fd.integration import _explicit_full_step
from ocean_solver.fd.processes import (
    _compute_tracer_tendency,
    _linear_bottom_drag_step,
    _linear_half_step,
)
from ocean_solver.fd.sources import _surface_heat_weights
from ocean_solver.fd.transport import (
    _face_transport_divergence,
    _match_layer_face_transports,
    _vertical_transport_iface,
)
from ocean_solver.fd.vertical import (
    _convective_mask,
    _d2_dz2_flux,
    _effective_kappa_v,
    _vertical_momentum_diffusion,
)
from ocean_solver.provenance.locations import source_root
from ocean_solver.provenance.restart import make_restart_contract
from ocean_solver.provenance.sources import solver_source_modules, source_paths

INVENTORY_SCHEME = "material_top_v1"
SOURCE_NAMES = ("prescribed_heat", "bulk_heat", "coastal_bulk_heat",
                "temperature_restore", "salinity_restore", "prescribed_brine")
METRIC_NAMES = ("moving_node_water_sensible_heat_J", "moving_node_water_salt_kg")
CONTINUITY_TOLERANCE_M = 1e-10
FACE_TOLERANCE_M2_PER_S = 1e-9
CFL_LIMIT = 0.5
SUBCYCLE_SCHEMES = ("reference_static_v1", "actual_geometry_v2")
MOMENTUM_DIFFUSION_SCHEMES = ("legacy_component_v1", "joint_heun_v1")


class SubcyclePlan(NamedTuple):
    required: jnp.ndarray
    count: jnp.ndarray
    supported: jnp.ndarray


class MaterialTopResult(NamedTuple):
    state: object
    attempted_state: object
    valid: jnp.ndarray
    checks: dict
    budget: dict


def _validate_params(params):
    requirements = {"column_geometry": "nodal_dual_v1", "mode_split": True,
                    "match_barotropic_transport": True, "process_time_scheme": "symmetric_fast_v3",
                    "conservative_kv": True, "localize_conv": True,
                    "polar_cap_rows": 0,
                    "dynamic_ice": False, "ice_salt_flux": 0., "kappa_gm": 0.,
                    "kappa_redi": 0.}
    for name, expected in requirements.items():
        if getattr(params, name) != expected:
            raise ValueError(f"{INVENTORY_SCHEME} requires {name}={expected!r}")
    for name in ("sponge_rate", "eta_relax_rate"):
        if np.any(np.asarray(getattr(params, name)) != 0.):
            raise ValueError(f"{INVENTORY_SCHEME} requires {name}=0")
    if not params.monotone_adv and not params.fct_adv:
        raise ValueError(f"{INVENTORY_SCHEME} requires donor-cell or the existing minmod flux")
    if np.asarray(params.dz_node).dtype != np.dtype("float64"):
        raise ValueError(f"{INVENTORY_SCHEME} requires float64 parameters")
    for name, value in params._asdict().items():
        if isinstance(value, (np.ndarray, jax.Array)):
            values = np.asarray(value)
            if not np.isfinite(values).all():
                raise ValueError(f"material parameters require finite {name}")
            if values.dtype.kind == "f" and values.dtype != np.dtype("float64"):
                raise ValueError(f"material parameters require float64 {name}")
    for name in ("nu_h", "nu_v", "nu_bi", "kappa_h", "kappa_v", "kappa_conv", "kappa_bi",
                 "r_bot", "lambda_bulk", "restore_coef_S", "coastal_kappa_h_2d", "coastal_kappa_v_2d",
                 "coastal_bulk_lambda_2d", "coastal_restore_coef_2d"):
        values = np.asarray(getattr(params, name))
        if not np.isfinite(values).all() or np.any(values < 0.):
            raise ValueError(f"material parameters require finite nonnegative {name}")
    if not np.isfinite(params.dt) or params.dt <= 0.:
        raise ValueError("material timestep must be finite and positive")


def _validate_subcycle_policy(scheme, maximum):
    if scheme not in SUBCYCLE_SCHEMES:
        raise ValueError("unknown material subcycle scheme")
    if isinstance(maximum, bool) or not isinstance(maximum, int) or maximum < 1:
        raise ValueError("material max_subcycles must be a positive integer")


def _validate_momentum_policy(scheme, subcycle_scheme):
    if scheme not in MOMENTUM_DIFFUSION_SCHEMES:
        raise ValueError("unknown material momentum diffusion scheme")
    if scheme == "joint_heun_v1" and subcycle_scheme != "actual_geometry_v2":
        raise ValueError("joint momentum diffusion requires actual_geometry_v2")


def material_thickness(eta, params):
    """Actual top mass weight; negative thickness is retained for rejection."""
    return jnp.where(params.wet_mask_z > 0.,
                     params.dz_node + eta[..., None] * params.surface_mask, 0.)


def _contents(state, params):
    concentration = jnp.stack((state.T, state.S), axis=-1)
    return material_thickness(state.eta, params)[..., None] * jnp.where(
        params.wet_mask_z[..., None] > 0., concentration, 0.)


def _concentrations(content, eta, original, params):
    thickness = material_thickness(eta, params)
    denominator = jnp.where(thickness > 0., thickness, 1.)
    concentration = content / denominator[..., None]
    return original._replace(
        T=jnp.where(params.wet_mask_z > 0., concentration[..., 0], original.T),
        S=jnp.where(params.wet_mask_z > 0., concentration[..., 1], original.S), eta=eta)


def _sum_content(content, params):
    area = params.dx_2d * params.dy
    units = jnp.asarray([RHO_0 * C_P, RHO_0 * 1e-3], dtype=jnp.float64)
    return jnp.sum(content * area[..., None, None], axis=(0, 1, 2)) * units


def material_inventory(state, params):
    """Declared moving nodal J/kg stocks, not a full ice/momentum energy budget."""
    return _sum_content(_contents(state, params), params)


def _safe_tracers(state, params):
    return state._replace(T=jnp.where(params.wet_mask_z > 0., state.T, 0.),
                          S=jnp.where(params.wet_mask_z > 0., state.S, 0.))


def _linear_rhs(state, params):
    safe = _safe_tracers(state, params)
    tendencies = tuple(_horizontal_tracer_diffusion(tracer, params)
                       + _d2_dz2_flux(tracer, _effective_kappa_v(params), params)
                       - (params.kappa_bi * _horizontal_biharmonic_tracer(tracer, params)
                          if params.kappa_bi > 0. else 0.) for tracer in (safe.T, safe.S))
    return params.dz_node[..., None] * jnp.stack(tendencies, axis=-1)


def _vertical_row_rate(coefficient, gate, params):
    conductance = coefficient * gate / params.dz_iface
    zero = jnp.zeros_like(conductance[..., :1])
    return jnp.concatenate((zero, conductance), axis=-1) + jnp.concatenate((conductance, zero), axis=-1)


def _horizontal_row_rate(coefficient, params):
    wet = params.wet_mask_z
    east = .5 * (coefficient + jnp.roll(coefficient, -1, axis=0)) * wet * jnp.roll(wet, -1, axis=0)
    north = .5 * (coefficient + jnp.roll(coefficient, -1, axis=1)) * wet * jnp.roll(wet, -1, axis=1)
    cosine_face = .5 * (params.cos_lat + jnp.roll(params.cos_lat, -1))
    north = (north * cosine_face[None, :, None]).at[:, -1].set(0.)
    south = jnp.roll(north, 1, axis=1).at[:, 0].set(0.)
    return ((east + jnp.roll(east, 1, axis=0)) * params.inv_dx ** 2
            + (north + south) * params.inv_dy ** 2 / params.cos_lat[None, :, None])


def _linear_row_rate(params):
    wet = params.wet_mask_z
    horizontal = params.dz_node * _horizontal_row_rate(params.kappa_h + params.coastal_kappa_h_2d[..., None], params)
    if params.kappa_bi > 0.:
        unit_row = _horizontal_row_rate(jnp.ones_like(params.coastal_kappa_h_2d)[..., None], params)
        horizontal = horizontal + params.dz_node * params.kappa_bi * 4. * jnp.max(unit_row) * unit_row
    vertical = _vertical_row_rate(_effective_kappa_v(params), wet[..., :-1] * wet[..., 1:], params)
    return horizontal + vertical


def _fraction(rate, eta, duration, params):
    thickness = material_thickness(eta, params)
    return jnp.max(jnp.where(params.wet_mask_z > 0.,
                             duration * rate / jnp.where(thickness > 0., thickness, 1.), 0.))


def _linear_material_step(state, params, duration, *, momentum_diffusion=None):
    content = _contents(state, params)
    first = _linear_rhs(state, params)
    predicted = _concentrations(content + duration * first, state.eta, state, params)
    second = _linear_rhs(predicted, params)
    change = .5 * duration * (first + second)
    absolute_change = .5 * duration * (jnp.abs(first) + jnp.abs(second))
    updated = _concentrations(content + change, state.eta, state, params)
    momentum = _linear_half_step(state, params, duration, momentum_diffusion=momentum_diffusion)
    updated = updated._replace(u=momentum.u, v=momentum.v)
    return updated, change, absolute_change, _fraction(_linear_row_rate(params), state.eta, duration, params)


def _subcycle_plan(rate, lower_thickness, duration, reference_count, params, maximum):
    """Uncut required count; unsupported plans activate no numerical substeps."""
    wet = params.wet_mask_z > 0.
    fractions = jnp.where(wet, duration * rate / jnp.where(lower_thickness > 0., lower_thickness, 1.), 0.)
    required = jnp.maximum(float(reference_count), jnp.ceil(jnp.max(fractions) / CFL_LIMIT))
    supported = (jnp.all(jnp.isfinite(fractions)) & jnp.all(jnp.isfinite(lower_thickness))
                 & jnp.all(jnp.where(wet, lower_thickness > 0., True)) & jnp.isfinite(required)
                 & (required <= maximum))
    count = jnp.where(supported, required, 0.).astype(jnp.int32)
    return SubcyclePlan(required, count, supported)


def _checkpointed_material_scan(advance, initial, maximum):
    """Preserve ordered updates while bounding reverse-mode carry history."""
    stride = 8

    @jax.checkpoint
    def block(carry, block_index):
        def inner(current, offset):
            index = block_index * stride + offset
            updated = jax.lax.cond(index < maximum, lambda values: advance(values, index)[0],
                                   lambda values: values, current)
            return updated, None

        updated, _ = jax.lax.scan(inner, carry, jnp.arange(stride))
        return updated, None

    final, _ = jax.lax.scan(block, initial, jnp.arange((maximum + stride - 1) // stride))
    return final


def _momentum_diffusion_norm_bound(params):
    """Absolute row-sum bound of the retained reference-momentum operators."""
    wet = params.wet_mask_z
    unit_diffusivity = jnp.ones_like(params.coastal_kappa_h_2d)[..., None]
    horizontal_norm = 2. * jnp.max(_horizontal_row_rate(unit_diffusivity, params))
    vertical = _vertical_row_rate(params.nu_v, wet[..., :-1] * wet[..., 1:], params)
    vertical_norm = 2. * jnp.max(vertical / params.dz_node)
    return params.nu_h * horizontal_norm + vertical_norm + params.nu_bi * horizontal_norm ** 2


def _momentum_diffusion_plan(params, duration, maximum):
    rate = jnp.full_like(params.wet_mask_z, .5 * _momentum_diffusion_norm_bound(params))
    plan = _subcycle_plan(rate, jnp.ones_like(rate), duration, 1, params, maximum)
    fraction = jnp.where(plan.supported, duration * jnp.max(rate) / jnp.maximum(plan.count, 1), 0.)
    return plan, fraction


def _joint_momentum_diffusion(state, params, duration, maximum):
    """Joint Heun stages; unsupported schedules execute no viscosity updates."""
    plan, _ = _momentum_diffusion_plan(params, duration, maximum)
    subduration = duration / jnp.maximum(plan.count, 1)
    wet = params.wet_mask_z > 0.

    def tendency(velocity):
        return (params.nu_h * _laplacian_h(velocity, params)
                + _vertical_momentum_diffusion(velocity, params)
                - params.nu_bi * _biharmonic_h(velocity, params)) * params.wet_mask_z

    @jax.checkpoint
    def integrate(velocities):
        first = tuple(tendency(velocity) for velocity in velocities)
        predicted = tuple(jnp.where(wet, velocity + subduration * rhs, velocity)
                          for velocity, rhs in zip(velocities, first, strict=True))
        second = tuple(tendency(velocity) for velocity in predicted)
        return tuple(jnp.where(wet, velocity + .5 * subduration * (rhs_1 + rhs_2), velocity)
                     for velocity, rhs_1, rhs_2 in zip(velocities, first, second, strict=True))

    def advance(velocities, index):
        return jax.lax.cond(plan.supported & (index < plan.count), integrate,
                            lambda current: current, velocities), None

    return _checkpointed_material_scan(advance, (state.u, state.v), maximum)


def _linear_material_subcycle(state, params, duration, maximum, *, momentum_diffusion=None):
    """Bounded differentiable scan; only tracer mixing is subcycled here."""
    rate = _linear_row_rate(params)
    plan = _subcycle_plan(rate, material_thickness(state.eta, params), duration, 1, params, maximum)
    subduration = duration / jnp.maximum(plan.count, 1)
    content = _contents(state, params)
    zero = jnp.zeros_like(content)

    @jax.checkpoint
    def integrate(carry):
        current, rhs, absolute = carry
        stage = _concentrations(current, state.eta, state, params)
        first = _linear_rhs(stage, params)
        predicted = _concentrations(current + subduration * first, state.eta, state, params)
        second = _linear_rhs(predicted, params)
        change = .5 * subduration * (first + second)
        absolute_change = .5 * subduration * (jnp.abs(first) + jnp.abs(second))
        return current + change, rhs + change, absolute + absolute_change

    def advance(carry, index):
        return jax.lax.cond(plan.supported & (index < plan.count), integrate, lambda current: current, carry), None

    final = _checkpointed_material_scan(advance, (content, zero, zero), maximum)
    updated = _concentrations(final[0], state.eta, state, params)
    momentum = _linear_half_step(state, params, duration, momentum_diffusion=momentum_diffusion)
    updated = updated._replace(u=momentum.u, v=momentum.v)
    fraction = jnp.where(plan.supported, _fraction(rate, state.eta, subduration, params), 0.)
    return updated, final[1], final[2], fraction, plan


def _transport_outflow(faces, vertical, params):
    east, north = faces
    west = jnp.roll(east, 1, axis=0)
    south = jnp.roll(north, 1, axis=1).at[:, 0].set(0.)
    horizontal = ((jnp.maximum(east, 0.) + jnp.maximum(-west, 0.)) * params.inv_dx
                  + (jnp.maximum(north, 0.) + jnp.maximum(-south, 0.))
                  * params.inv_dy / params.cos_lat[None, :, None])
    relative_vertical = vertical.at[..., 0].set(0.)
    return horizontal + jnp.maximum(relative_vertical[..., 1:], 0.) + jnp.maximum(-relative_vertical[..., :-1], 0.)


def _surface_feedback_row_rate(params):
    heat = (params.dz_node * _surface_heat_weights(params)
            * (params.lambda_bulk + params.coastal_bulk_lambda_2d[..., None]) / (RHO_0 * C_P))
    restore_temperature = params.dz_node * params.surface_mask * params.coastal_restore_coef_2d[..., None]
    restore_salinity = params.dz_node * params.surface_mask * params.restore_coef_S
    return jnp.maximum(heat + restore_temperature, restore_salinity) * params.wet_mask_z


def _nonlinear_subcycle_plan(state, params, faces, maximum, endpoint_eta=None):
    vertical = _vertical_transport_iface(state.u, state.v, params, face_transport=faces)
    eta_rate = -_face_transport_divergence(jnp.sum(faces[0], axis=-1), jnp.sum(faces[1], axis=-1), params)
    endpoint_eta = state.eta + params.dt * eta_rate if endpoint_eta is None else endpoint_eta
    lower_thickness = jnp.minimum(material_thickness(state.eta, params), material_thickness(endpoint_eta, params))
    wet = params.wet_mask_z
    convective = _vertical_row_rate(params.kappa_conv, wet[..., :-1] * wet[..., 1:], params)
    rate = _transport_outflow(faces, vertical, params) + convective + _surface_feedback_row_rate(params)
    return _subcycle_plan(rate, lower_thickness, params.dt, max(params.adv_nsub, params.conv_nsub), params, maximum)


def _nonlinear_rhs(state, params, faces, vertical):
    safe = _safe_tracers(state, params)
    terms = _compute_tracer_tendency(safe, params, face_transport=faces, return_terms=True)[2]
    sources, processes, top_temperature, top_salinity = terms[:4]
    advection = params.dz_node[..., None] * jnp.stack(processes[0], axis=-1)
    surface_reference_exchange = jnp.stack((top_temperature, top_salinity), axis=-1)
    advection = advection.at[..., 0, :].add(-surface_reference_exchange)
    convection = params.dz_node[..., None] * jnp.stack(processes[1], axis=-1)
    zero = jnp.zeros_like(sources[0])
    source_pairs = tuple(jnp.stack((source, zero) if index < 4 else (zero, source), axis=-1)
                         for index, source in enumerate(sources))
    source_content = jnp.stack(source_pairs) * params.dz_node[None, ..., None]
    rhs = advection + convection + jnp.sum(source_content, axis=0)
    _, unstable = _convective_mask(safe, params)
    convective_rate = _vertical_row_rate(params.kappa_conv, unstable, params)
    return rhs, source_content, advection, convection, convective_rate


def _material_tracer_step(state, params, faces, *, subcycle_plan=None, max_subcycles=128, endpoint_eta=None):
    scheduled = subcycle_plan is not None
    subcycles = jnp.maximum(subcycle_plan.count, 1) if scheduled else max(int(params.adv_nsub), int(params.conv_nsub))
    subparams = params._replace(dt=params.dt / subcycles, adv_nsub=1, conv_nsub=1)
    vertical = _vertical_transport_iface(state.u, state.v, params, face_transport=faces)
    eta_rate = -_face_transport_divergence(jnp.sum(faces[0], axis=-1), jnp.sum(faces[1], axis=-1), params)
    outflow = _transport_outflow(faces, vertical, params)
    initial_content = _contents(state, params)
    zero = jnp.zeros_like(initial_content)
    source_zero = jnp.zeros((len(SOURCE_NAMES), 2) if scheduled else (len(SOURCE_NAMES),) + zero.shape, dtype=zero.dtype)
    exchange_zero = jnp.zeros(2, dtype=zero.dtype) if scheduled else zero
    wet = params.wet_mask_z > 0.

    def advance(carry, index):
        content, eta, total_rhs, absolute_rhs, total_sources, total_advection, total_convection, maxima, minimum = carry
        stage = _concentrations(content, eta, state, params)
        first = _nonlinear_rhs(stage, subparams, faces, vertical)
        predicted_eta = eta + subparams.dt * eta_rate
        if scheduled:
            predicted_eta = state.eta + (index + 1) * subparams.dt * eta_rate
            if endpoint_eta is not None:
                predicted_eta = jnp.where(index + 1 == subcycle_plan.count, endpoint_eta, predicted_eta)
        predicted = _concentrations(content + subparams.dt * first[0], predicted_eta, state, params)
        second = _nonlinear_rhs(predicted, subparams, faces, vertical)
        weight = .5 * subparams.dt
        change = weight * (first[0] + second[0])
        sources = weight * (first[1] + second[1])
        advection = weight * (first[2] + second[2])
        convection = weight * (first[3] + second[3])
        fractions = jnp.asarray([jnp.maximum(_fraction(outflow, eta, subparams.dt, params),
                                             _fraction(outflow, predicted_eta, subparams.dt, params)),
                                  jnp.maximum(_fraction(first[4], eta, subparams.dt, params),
                                              _fraction(second[4], predicted_eta, subparams.dt, params))])
        if scheduled:
            feedback = _surface_feedback_row_rate(params)
            combined = jnp.maximum(_fraction(outflow + first[4] + feedback, eta, subparams.dt, params),
                                    _fraction(outflow + second[4] + feedback, predicted_eta, subparams.dt, params))
            fractions = jnp.concatenate((fractions, combined[None]))
            sources = jax.vmap(lambda values: _sum_content(values, params))(sources)
            advection, convection = _sum_content(advection, params), _sum_content(convection, params)
        minimum = jnp.minimum(minimum, jnp.minimum(
            jnp.min(jnp.where(wet, material_thickness(eta, params), jnp.inf)),
            jnp.min(jnp.where(wet, material_thickness(predicted_eta, params), jnp.inf))))
        return (content + change, predicted_eta, total_rhs + change,
                absolute_rhs + weight * (jnp.abs(first[0]) + jnp.abs(second[0])),
                total_sources + sources, total_advection + advection, total_convection + convection,
                jnp.maximum(maxima, fractions), minimum), None

    initial = (initial_content, state.eta, zero, zero, source_zero, exchange_zero, exchange_zero,
               jnp.zeros(3 if scheduled else 2, dtype=zero.dtype), jnp.asarray(jnp.inf, dtype=zero.dtype))
    if scheduled:
        integrate = jax.checkpoint(lambda operands: advance(operands[0], operands[1])[0])

        def bounded_advance(carry, index):
            return jax.lax.cond(subcycle_plan.supported & (index < subcycle_plan.count),
                                integrate, lambda operands: operands[0], (carry, index)), None

        final = _checkpointed_material_scan(bounded_advance, initial, max_subcycles)
    else:
        final, _ = jax.lax.scan(advance, initial, xs=None, length=subcycles)
    content, eta, rhs, absolute_rhs, sources, advection, convection, maxima, minimum = final
    return _concentrations(content, eta, state, params), rhs, absolute_rhs, sources, advection, convection, maxima, minimum


def _material_step(state, params, subcycle_scheme="reference_static_v1", max_subcycles=128,
                   momentum_diffusion_scheme="legacy_component_v1", *, stage_observer=None, fast_observer=None):
    # Optional eager research observer; default/JIT production paths are unchanged.
    if fast_observer is not None and params.use_scan:
        raise ValueError('eager fast observation requires use_scan=False')
    def observe(name, current, diagnostics=None):
        if stage_observer is not None:
            stage_observer(name, current, diagnostics)

    for name, value in state._asdict().items():
        if value.dtype != jnp.float64:
            raise ValueError(f"{INVENTORY_SCHEME} requires float64 {name}")
        expected_shape = params.wet_mask.shape if name in {"eta", "ice"} else params.wet_mask_z.shape
        if value.shape != expected_shape:
            raise ValueError("material state shapes must match the frozen FD grid")
    duration = params.dt / 2.
    scheduled = subcycle_scheme == "actual_geometry_v2"
    momentum_diffusion = None
    if momentum_diffusion_scheme == "joint_heun_v1":
        def momentum_diffusion(current, values, interval):
            return _joint_momentum_diffusion(current, values, interval, max_subcycles)
    start = _linear_bottom_drag_step(state, params, duration)
    observe("bottom_drag_first", start)
    if scheduled:
        first, linear_rhs_1, absolute_1, fraction_1, linear_plan_1 = _linear_material_subcycle(
            start, params, duration, max_subcycles, momentum_diffusion=momentum_diffusion)
    else:
        first, linear_rhs_1, absolute_1, fraction_1 = _linear_material_step(start, params, duration)
    observe("linear_first", first)
    nonlinear_predictor = _explicit_full_step(first, params, params.dt)
    observe("nonlinear_predictor", nonlinear_predictor)
    predictor = _linear_half_step(nonlinear_predictor, params, duration, momentum_diffusion=momentum_diffusion)
    observe("predictor_linear_second", predictor)
    if fast_observer is None:
        dynamical, column_faces, filter_change = _barotropic_subcycle_transport(predictor, params)
    else:
        dynamical, column_faces, filter_change = _barotropic_subcycle_transport(
            predictor, params, fast_observer=fast_observer)
    observe("barotropic", dynamical)
    velocity_x = .5 * (first.u + nonlinear_predictor.u)
    velocity_y = .5 * (first.v + nonlinear_predictor.v)
    faces = _match_layer_face_transports(velocity_x, velocity_y, column_faces, params)
    observe("transport_match", dynamical, {"faces": faces, "column_faces": column_faces,
                                           "filter_change": filter_change})
    nonlinear_plan = _nonlinear_subcycle_plan(first, params, faces, max_subcycles, endpoint_eta=dynamical.eta) if scheduled else None
    middle, nonlinear_rhs, absolute_nonlinear, sources, advection, convection, fractions, minimum = _material_tracer_step(
        first, params, faces, subcycle_plan=nonlinear_plan, max_subcycles=max_subcycles,
        endpoint_eta=dynamical.eta if scheduled else None)
    observe("accepted_tracer_replay", middle)
    if scheduled:
        end, linear_rhs_2, absolute_2, fraction_2, linear_plan_2 = _linear_material_subcycle(
            middle, params, duration, max_subcycles, momentum_diffusion=momentum_diffusion)
    else:
        end, linear_rhs_2, absolute_2, fraction_2 = _linear_material_step(middle, params, duration)
    observe("linear_second", end)
    attempted = dynamical._replace(T=end.T, S=end.S)
    attempted = _linear_bottom_drag_step(attempted, params, duration)
    observe("bottom_drag_second", attempted)
    attempted = attempted._replace(v=attempted.v * params.interior_mask_z)
    observe("closed_wall", attempted)
    before, after = _contents(state, params), _contents(attempted, params)
    expected = linear_rhs_1 + nonlinear_rhs + linear_rhs_2
    absolute_rhs = absolute_1 + absolute_nonlinear + absolute_2
    local_residual = after - before - expected
    floor = 64. * jnp.finfo(jnp.float64).eps * (1. + jnp.abs(before) + jnp.abs(after) + absolute_rhs)
    matched_error = jnp.max(jnp.stack([jnp.max(jnp.abs(jnp.sum(layer, axis=-1) - column))
                                      for layer, column in zip(faces, column_faces, strict=True)]))
    continuity = attempted.eta - state.eta + params.dt * _face_transport_divergence(*column_faces, params)
    minimum = jnp.minimum(minimum, jnp.min(jnp.where(params.wet_mask_z > 0., material_thickness(state.eta, params), jnp.inf)))
    minimum = jnp.minimum(minimum, jnp.min(jnp.where(params.wet_mask_z > 0., material_thickness(attempted.eta, params), jnp.inf)))
    checks = {"minimum_wet_thickness_m": minimum,
              "transport_outflow_fraction_max": fractions[0], "convective_fraction_max": fractions[1],
              "linear_diffusion_fraction_max": jnp.maximum(fraction_1, fraction_2),
              "local_continuity_residual_max_m": jnp.max(jnp.abs(continuity)),
              "tracer_face_mismatch_max_m2_per_s": matched_error,
              "unpaired_eta_filter_max_m": jnp.max(jnp.abs(filter_change)),
              "local_inventory_roundoff_ratio_max": jnp.max(jnp.abs(local_residual) / floor),
              "ice_abs_max_m": jnp.max(jnp.abs(state.ice)),
              "velocity_abs_max_m_per_s": jnp.maximum(jnp.max(jnp.abs(attempted.u)), jnp.max(jnp.abs(attempted.v))),
              "eta_abs_max_m": jnp.max(jnp.abs(attempted.eta))}
    source_totals = sources if scheduled else jax.vmap(lambda content: _sum_content(content, params))(sources)
    observed = _sum_content(after - before, params)
    budget = {"observed_change": observed, "source_inputs": source_totals,
              "source_budget_residual": observed - jnp.sum(source_totals, axis=0),
              "linear_exchange": _sum_content(linear_rhs_1 + linear_rhs_2, params),
              "advection_exchange": advection if scheduled else _sum_content(advection, params),
              "convection_exchange": convection if scheduled else _sum_content(convection, params),
              "local_implementation_residual": _sum_content(local_residual, params),
              "absolute_local_implementation_residual": _sum_content(jnp.abs(local_residual), params)}
    if scheduled:
        checks["nonlinear_combined_fraction_max"] = fractions[2]
        for name, plan in (("linear_first", linear_plan_1), ("nonlinear", nonlinear_plan), ("linear_second", linear_plan_2)):
            checks[f"{name}_required_subcycles"] = plan.required
            checks[f"{name}_active_subcycles"] = plan.count
            checks[f"{name}_schedule_supported"] = plan.supported
    if momentum_diffusion is not None:
        momentum_plan, momentum_fraction = _momentum_diffusion_plan(params, duration, max_subcycles)
        checks["momentum_diffusion_fraction_max"] = momentum_fraction
        checks["momentum_required_subcycles"] = momentum_plan.required
        checks["momentum_active_subcycles"] = momentum_plan.count
        checks["momentum_schedule_supported"] = momentum_plan.supported
    finite = jnp.all(jnp.stack([jnp.all(jnp.isfinite(value)) for value in jax.tree.leaves((attempted, checks, budget))]))
    valid = (finite & (minimum > 0.) & (fractions[0] <= CFL_LIMIT) & (fractions[1] <= CFL_LIMIT)
             & (checks["linear_diffusion_fraction_max"] <= CFL_LIMIT)
             & (checks["local_continuity_residual_max_m"] <= CONTINUITY_TOLERANCE_M)
             & (matched_error <= FACE_TOLERANCE_M2_PER_S)
             & (checks["unpaired_eta_filter_max_m"] <= CONTINUITY_TOLERANCE_M)
             & (checks["local_inventory_roundoff_ratio_max"] <= 1.) & (checks["ice_abs_max_m"] == 0.)
             & (checks["velocity_abs_max_m_per_s"] <= 10.) & (checks["eta_abs_max_m"] <= 15.))
    if scheduled:
        valid = (valid & linear_plan_1.supported & nonlinear_plan.supported & linear_plan_2.supported
                 & (checks["nonlinear_combined_fraction_max"] <= CFL_LIMIT))
    if momentum_diffusion is not None:
        valid = valid & momentum_plan.supported & (momentum_fraction <= CFL_LIMIT)
    checks["finite"] = finite
    selected = jax.tree.map(lambda new, old: jnp.where(valid, new, old), attempted, state)
    return MaterialTopResult(selected, attempted, valid, checks, budget)


def make_material_top_step(params, *, subcycle_scheme="reference_static_v1", max_subcycles=128,
                           momentum_diffusion_scheme="legacy_component_v1"):
    """Build an explicit checked candidate; callers must stop at first rejection."""
    _validate_params(params)
    _validate_subcycle_policy(subcycle_scheme, max_subcycles)
    _validate_momentum_policy(momentum_diffusion_scheme, subcycle_scheme)

    @jax.jit
    def step(state, forcing=None, atmosphere=None):
        current = params
        if forcing is not None:
            for value in forcing:
                if value.dtype != jnp.float64 or value.shape != params.wet_mask.shape:
                    raise ValueError("material forcing requires float64 horizontal fields")
            current = current._replace(tau_x_2d=forcing[0], tau_y_2d=forcing[1], Q_heat_2d=forcing[2])
        if atmosphere is not None:
            if atmosphere.dtype != jnp.float64 or atmosphere.shape != params.T_atm_3d.shape:
                raise ValueError("material atmosphere requires the frozen float64 field shape")
            current = current._replace(T_atm_3d=atmosphere)
        return _material_step(state, current, subcycle_scheme, max_subcycles, momentum_diffusion_scheme)

    return step


def make_material_top_restart_contract(grid, params, *, forcing, controls, execution,
                                       subcycle_scheme="reference_static_v1", max_subcycles=128,
                                       momentum_diffusion_scheme="legacy_component_v1"):
    """Separate stock semantics and source hashes; no old-contract continuation."""
    _validate_params(params)
    _validate_subcycle_policy(subcycle_scheme, max_subcycles)
    _validate_momentum_policy(momentum_diffusion_scheme, subcycle_scheme)
    controls = dict(controls)
    if any(name in controls for name in ("tracer_inventory_scheme", "material_subcycle_scheme", "material_max_subcycles",
                                         "material_momentum_diffusion_scheme")):
        raise ValueError("inventory scheme is frozen by the material restart factory")
    controls["tracer_inventory_scheme"] = INVENTORY_SCHEME
    controls["material_subcycle_scheme"] = subcycle_scheme
    controls["material_max_subcycles"] = max_subcycles
    controls["material_momentum_diffusion_scheme"] = momentum_diffusion_scheme
    source_directory = source_root(__file__)
    contract = make_restart_contract(
        grid, params, dtype="float64", forcing=forcing, controls=controls,
        code_paths=source_paths(source_directory, solver_source_modules()), execution=execution)
    contract["state_family"] = "FD_point_samples_material_top_mass_lumped_linear_momentum_v1"
    return contract

preserve_legacy_names(globals(), 'material_top')
