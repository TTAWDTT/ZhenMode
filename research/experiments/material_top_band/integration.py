"""Full original-stage seam with a deliberately restricted top-band contract.

Only reference-coincident, all-wet, horizontally identical resting columns are
implemented. Moving/inhomogeneous momentum, pressure work and transport require
new adapters and are refused before numerical stages. This is NOT their repair.
"""
import copy
import hashlib
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

import zhenmode_research.candidates.material.solver as material
from ocean_solver.config.definitions import C_P, RHO_0
from ocean_solver.geometry.types import nodal_control_thickness
from ocean_solver.dynamics.pressure import _compute_bt_rho_pgf
from ocean_solver.dynamics.pressure import _compute_pressure_gradient
from ocean_solver.dynamics.transport import _vertical_transport_iface
from ocean_solver.geometry.fd import make_fd_params
from research.experiments.fd_static_bridge.bridge import integrate
from zhenmode_research.provenance import solver_source_modules

STAGES = ('bottom_drag_first', 'linear_first', 'nonlinear_predictor',
          'predictor_linear_second', 'barotropic', 'transport_match', 'accepted_tracer_replay',
          'linear_second', 'bottom_drag_second', 'closed_wall')
CONTRACT = 'reference_coincident_resting_columns_v1'


def parameter_digest(params):
    digest = hashlib.sha256()
    for name, value in sorted(params._asdict().items()):
        array = np.asarray(value) if value is not None else None
        digest.update(name.encode() + b'\0')
        if array is None:
            digest.update(b'None')
        else:
            if array.dtype.kind not in 'biufUS':
                raise ValueError('unsupported parameter identity: ' + name)
            digest.update(str(array.shape).encode() + array.dtype.str.encode() + array.tobytes())
    return digest.hexdigest()


def source_digest():
    from ocean_solver.provenance.archives import current_source_files

    root = Path(__file__).resolve().parents[3]
    selected = ["research/experiments/fd_static_bridge/bridge.py",
                "research/experiments/material_top_band/integration.py"]
    return {name: hashlib.sha256(path.read_bytes()).hexdigest()
            for name, path in current_source_files(root, selected).items()}


def state_digest(state):
    return {name: hashlib.sha256(np.asarray(value).tobytes()).hexdigest()
            for name, value in state._asdict().items()}


def _uniform(array):
    values = np.asarray(array)
    return (values.ndim >= 2 and values.shape[0] > 0 and values.shape[1] > 0
            and np.array_equal(values, np.broadcast_to(values[0, 0], values.shape)))


def _state_contract(state):
    missing = []
    if any(np.asarray(value).dtype != np.float64 or not np.isfinite(value).all()
           for value in state):
        missing.append('finite_float64_state')
    if np.any(np.asarray(state.eta) != 0):
        missing.append('moving_geometry_momentum_pressure_diffusion_FCT_source_adapters')
    if np.any(np.asarray(state.u) != 0) or np.any(np.asarray(state.v) != 0):
        missing.append('actual_mass_momentum_and_cross_band_transport')
    if not _uniform(state.T) or not _uniform(state.S):
        missing.append('common_physical_depth_pressure_force_and_work')
    if np.any(np.asarray(state.ice) != 0):
        missing.append('nonzero_ice_stock')
    if np.any(np.asarray(state.T) < -5) or np.any(np.asarray(state.T) > 45):
        missing.append('temperature_bounds')
    if np.any(np.asarray(state.S) < 0) or np.any(np.asarray(state.S) > 50):
        missing.append('salinity_bounds')
    return missing


def coverage(state, params, grid, *, max_subcycles=256):
    """Executable capability checks; neither disabling nor editing any parameter."""
    missing = _state_contract(state)
    for name, value in list(params._asdict().items()) + list(state._asdict().items()):
        if np.ma.getmaskarray(value).any():
            missing.append('masked_unknown:' + name)
        array = np.asarray(value) if value is not None else None
        if array is not None and array.dtype.kind in 'biuf' and not np.isfinite(array).all():
            missing.append('nonfinite_input:' + name)
    for name in ('dx_2d', 'dy', 'cos_lat', 'f', 'z', 'depth', 'wet_mask', 'wet_mask_3d'):
        if np.ma.getmaskarray(getattr(grid, name)).any():
            missing.append('masked_grid_unknown:' + name)
        values = np.asarray(getattr(grid, name))
        if values.dtype.kind not in 'biuf' or not np.isfinite(values).all():
            missing.append('invalid_grid_values:' + name)
    shape = (grid.nx, grid.ny, grid.nz)
    if (not 4 <= grid.nz <= 64 or grid.nx < 2 or grid.ny < 2 or grid.nx * grid.ny > 1024
            or any(np.asarray(getattr(state, key)).shape != shape for key in ('u', 'v', 'T', 'S'))
            or any(np.asarray(getattr(state, key)).shape != shape[:2] for key in ('eta', 'ice'))):
        missing.append('bounded_small_grid_state_shape')
    if (np.asarray(grid.wet_mask_3d).shape != shape
            or not np.all(np.asarray(grid.wet_mask_3d) == 1)
            or not np.all(np.asarray(params.wet_mask_z) == 1)
            or not np.all(np.asarray(params.wet_mask) == 1)
            or not np.all(np.asarray(grid.wet_mask) == 1)):
        missing.append('coast_shallow_band_and_bottom_adapters')
    if not _uniform(grid.depth):
        missing.append('variable_bathymetry_pressure_and_bottom_flux')
    for name, expected_shape in (('dx_2d', shape[:2]), ('dy', ()), ('cos_lat', (grid.ny,))):
        first, second = np.asarray(getattr(params, name)), np.asarray(getattr(grid, name))
        if (first.shape != expected_shape or second.shape != expected_shape
                or not np.isfinite(first).all() or not np.isfinite(second).all()
                or np.any(first <= 0) or np.any(second <= 0)
                or not np.array_equal(first, second)):
            missing.append('horizontal_metric_binding:' + name)
    widths = nodal_control_thickness(grid.z)
    if (widths.shape != (grid.nz,) or np.asarray(grid.z)[0] != 0
            or not np.array_equal(grid.dz, -np.diff(grid.z))
            or np.any(np.asarray(grid.depth) < -np.asarray(grid.z)[-1])):
        missing.append('grid_node_and_bottom_contract')
    if not np.array_equal(np.asarray(params.dz_node).ravel(), widths):
        missing.append('vertical_metric_binding')
    if np.any(np.asarray(params.H_sw) != widths.sum()):
        missing.append('column_depth_binding')
    expected_fields = (make_fd_params(grid, column_geometry='nodal_dual_v1')._asdict()
                       if not missing else {})
    if not missing:
        expected_fields['dz_norm'] = np.broadcast_to(widths / widths.sum(), shape)
    for name, expected in expected_fields.items():
        if not np.array_equal(np.asarray(getattr(params, name)), expected):
            missing.append('derived_metric_binding:' + name)
    array_shapes = {name: shape[:2] for name in (
        'tau_x_2d', 'tau_y_2d', 'Q_heat_2d', 'H_sw', 'S_ref_2d',
        'coastal_restore_coef_2d', 'coastal_restore_T_2d', 'coastal_bulk_lambda_2d',
        'coastal_kappa_h_2d', 'coastal_kappa_v_2d', 'sponge_rate_2d',
        'eta_relax_mask', 'mixed_layer_mask_2d')}
    if params.projection_inv_diagonal is not None:
        array_shapes['projection_inv_diagonal'] = shape[:2]
    array_shapes.update(T_atm_3d=shape[:2] + (1,), sponge_rate=shape[:2] + (1,),
                        T_clim_3d=shape, S_clim_3d=shape, dealias_lon_mask=(grid.nx, 1, 1))
    for name, expected_shape in array_shapes.items():
        if np.asarray(getattr(params, name)).shape != expected_shape:
            missing.append('parameter_array_shape:' + name)
    if np.any(np.asarray(params.tau_x_2d) != 0) or np.any(np.asarray(params.tau_y_2d) != 0):
        missing.append('wind_driven_actual_mass_momentum_and_pressure_work')
    for name in ('Q_heat_2d', 'T_atm_3d'):
        if not _uniform(getattr(params, name)):
            missing.append('inhomogeneous_source_pressure_feedback:' + name)
    for name in ('coastal_restore_coef_2d', 'coastal_bulk_lambda_2d',
                 'coastal_kappa_h_2d', 'coastal_kappa_v_2d'):
        if np.any(np.asarray(getattr(params, name)) != 0):
            missing.append('coastal_adapter:' + name)
    if params.restore_coef_S != 0:
        missing.append('salinity_restoring_adapter')
    if params.mixed_layer_depth_2d is not None or params.mixed_layer_depth_m != 0:
        missing.append('mixed_layer_deposition_adapter')
    if not params.fct_adv or params.monotone_adv:
        missing.append('original_FCT_configuration')
    if params.bottom_friction != 'linear':
        missing.append('original_linear_drag_configuration')
    if any(type(getattr(params, name)) is not int or getattr(params, name) != 2
           for name in ('adv_nsub', 'conv_nsub', 'nu_nsub')):
        missing.append('original_process_subcycle_clocks')
    if (type(max_subcycles) is not int or not 1 <= max_subcycles <= 256
            or type(params.n_subcyc) is not int or params.n_subcyc != 12
            or not np.isfinite(params.dt_bt)
            or abs(params.dt_bt * 12 - params.dt) > 8 * np.finfo(float).eps * params.dt):
        missing.append('bounded_original_12_fast_clock')
    try:
        material._validate_params(params)
    except ValueError as error:
        missing.append('original_material_policy:' + str(error))
    plans = {}
    if not missing:
        duration = params.dt / 2
        linear = material._subcycle_plan(
            material._linear_row_rate(params), material.material_thickness(state.eta, params),
            duration, 1, params, max_subcycles)
        momentum, _ = material._momentum_diffusion_plan(params, duration, max_subcycles)
        zero = jnp.zeros_like(state.T)
        nonlinear = material._nonlinear_subcycle_plan(state, params, (zero, zero), max_subcycles)
        for name, plan in (('linear', linear), ('momentum', momentum), ('nonlinear', nonlinear)):
            jax.block_until_ready(plan)
            required, count, supported = map(np.asarray, plan)
            if (required.shape != () or count.shape != () or supported.shape != ()
                    or required.dtype.kind not in 'iuf' or count.dtype.kind not in 'iu'
                    or supported.dtype.kind != 'b'):
                plans[name] = None
                missing.append('invalid_original_' + name + '_plan')
                continue
            # Finite coefficients can overflow derived row bounds. Never convert
            # a nonfinite requirement to int or emit NaN/Inf in the capability ledger.
            if not np.isfinite(required) or not np.isfinite(count):
                plans[name] = None
                missing.append('nonfinite_original_' + name + '_plan')
                continue
            if not bool(supported):
                plans[name] = float(required)  # Preserve the finite, uncut requirement.
                missing.append('original_' + name + '_capacity')
                continue
            if (required < 1 or required > max_subcycles or required != np.floor(required)
                    or count != required):
                plans[name] = None
                missing.append('invalid_original_' + name + '_plan')
                continue
            plans[name] = int(required)
    return {'contract': CONTRACT, 'missing_contracts': sorted(set(missing)),
            'stages': list(STAGES), 'moving_geometry_supported': False,
            'declared_time_scheme': params.process_time_scheme,
            'inventory_authority': 'original_material_stock_on_coincident_geometry',
            'preflight_required_subcycles': plans,
            'stage_contracts': {name: {'scope': CONTRACT,
                                      'status': 'blocked' if missing else 'supported_restricted'}
                                for name in STAGES},
            'qualification_passed': False}


def export_reference_band(state, params):
    """Reuse the verified P1 bridge on identical physical control intervals.

    This export is a diagnostic view, not a second independent restart authority.
    Actual and reference masses coincide within the executable contract.
    """
    widths = np.asarray(params.dz_node).ravel()[:3]
    edges = np.r_[0., -np.cumsum(widths)]
    values = np.stack([np.asarray(field)[..., :3] for field in
                       (state.T, state.S, state.u, state.v)], axis=-1)
    stock = widths[:, None] * values
    stock[..., 2:] *= RHO_0
    result = np.empty_like(stock)
    for cell in np.ndindex(state.eta.shape):
        result[cell] = np.array([integrate(edges, stock[cell], edges[k + 1], edges[k])
                                 for k in range(3)])
    bound = 2048 * np.finfo(float).eps * (abs(stock).sum(axis=-2) + abs(result).sum(axis=-2))
    if np.any(abs(result.sum(axis=-2) - stock.sum(axis=-2)) > bound):
        raise ValueError('reference-band bridge stock closure')
    return {'edges_m': edges, 'inventory': result,
            'velocity_actual_mass': result[..., 2:] / (RHO_0 * widths[:, None])}


def advance(state, params, grid, *, max_subcycles=256):
    """One complete original-stage call or complete rollback, never a partial pass."""
    start_time = time.perf_counter()
    before = copy.deepcopy(state)
    report = {'executed_stages': [], 'validated_stages': [],
              'missing_contracts': [], 'qualification_passed': False,
              'moving_geometry_supported': False, 'all_declared_stages_executed': False}
    try:
        report.update(coverage(state, params, grid, max_subcycles=max_subcycles))
        if report['missing_contracts']:
            raise ValueError('unimplemented stage contracts: ' + ','.join(report['missing_contracts']))
        digest = parameter_digest(params)
        input_digest = state_digest(before)
        sources = source_digest()
        frozen = copy.deepcopy(params)
        export_reference_band(before, frozen)

        report['pressure_work_J'] = 0.
        def observe(name, current, diagnostics):
            jax.block_until_ready(current)
            report['executed_stages'].append(name)
            violations = _state_contract(current)
            if violations:
                raise ValueError('stage left declared contract: ' + name + ':' + ','.join(violations))
            force = _compute_bt_rho_pgf(current, frozen)
            force_3d = _compute_pressure_gradient(current, frozen)
            if any(np.any(np.asarray(value) != 0) for value in (*force, *force_3d)):
                raise ValueError('nonzero pressure force outside coincident resting contract')
            power = np.sum(np.asarray(frozen.dx_2d)[..., None] * frozen.dy
                           * np.asarray(frozen.dz_node) * RHO_0
                           * (np.asarray(force_3d[0]) * np.asarray(current.u)
                              + np.asarray(force_3d[1]) * np.asarray(current.v)))
            if power != 0:
                raise ValueError('nonzero pressure work outside resting contract')
            if diagnostics is not None:
                vertical = _vertical_transport_iface(current.u, current.v, frozen,
                                                      face_transport=diagnostics['faces'])
                cross = float(np.max(abs(np.asarray(vertical)[..., 3])))
                mismatch = max(float(np.max(abs(np.asarray(layer).sum(axis=-1)
                                               - np.asarray(column))))
                               for layer, column in zip(diagnostics['faces'],
                                                        diagnostics['column_faces'], strict=True))
                report.update(cross_band_transport_max_m_s=cross,
                              column_layer_match_max_m2_s=mismatch)
                if cross != 0 or mismatch != 0 or any(
                        np.any(np.asarray(value) != 0) for value in diagnostics['faces']):
                    raise ValueError('nonzero transport outside resting contract')
            report['validated_stages'].append(name)

        result = material._material_step(
            before, frozen, 'actual_geometry_v2', max_subcycles,
            'joint_heun_v1', stage_observer=observe)
        jax.block_until_ready(result)
        if parameter_digest(params) != digest:
            raise ValueError('parameter snapshot changed during full step')
        if state_digest(state) != input_digest:
            raise ValueError('input state changed during full step')
        if source_digest() != sources:
            raise ValueError('source snapshot changed during full step')
        if not bool(result.valid):
            raise ValueError('original complete material-step gates rejected')
        band = export_reference_band(result.state, params)
        source_inputs = np.asarray(result.budget['source_inputs'])
        if (source_inputs.shape != (len(material.SOURCE_NAMES), 2)
                or source_inputs.dtype.kind not in 'iuf'
                or not np.isfinite(source_inputs).all()):
            raise ValueError('invalid complete-step source budget')
        source_heat = float(source_inputs[:, 0].sum())
        scale = np.sum(np.asarray(params.dx_2d)[..., None] * params.dy
                       * np.asarray(params.dz_node)
                       * (abs(np.asarray(before.T)) + abs(np.asarray(result.state.T)))) * RHO_0 * C_P
        bound = 8192 * np.finfo(float).eps * scale
        capacity = np.asarray(frozen.dx_2d)[..., None] * frozen.dy * np.asarray(frozen.dz_node)
        change = float(np.sum(capacity * (np.asarray(result.state.T) - np.asarray(before.T)))
                       * RHO_0 * C_P)
        if not np.isfinite([change, source_heat, bound]).all() or bound < 0:
            raise ValueError('nonfinite independent heat/source budget')
        if abs(change - source_heat) > bound:
            raise ValueError('independent full-step heat/source closure')
        salt_change = float(np.sum(capacity * (np.asarray(result.state.S) - np.asarray(before.S)))
                            * RHO_0 * 1e-3)
        salt_source = float(source_inputs[:, 1].sum())
        salt_bound = float(8192 * np.finfo(float).eps * RHO_0 * 1e-3 * np.sum(
            capacity * (abs(np.asarray(before.S)) + abs(np.asarray(result.state.S)))))
        if not np.isfinite([salt_change, salt_source, salt_bound]).all() or salt_bound < 0:
            raise ValueError('nonfinite independent salt/source budget')
        if abs(salt_change - salt_source) > salt_bound:
            raise ValueError('independent full-step salt/source closure')
        report.update(parameter_sha256=digest, source_sha256=sources, fast_subcycles=params.n_subcyc,
                      input_field_sha256=input_digest,
                      source_heat_J=source_heat, heat_roundoff_bound_J=float(bound),
                      independent_heat_change_J=change, independent_salt_change_kg=salt_change,
                      source_salt_kg=salt_source, salt_roundoff_bound_kg=salt_bound,
                      original_checks={key: np.asarray(value).item() for key, value in result.checks.items()},
                      all_declared_stages_executed=report['executed_stages'] == list(STAGES))
        if not report['all_declared_stages_executed']:
            raise ValueError('incomplete stage dispatch')
        if report['validated_stages'] != list(STAGES):
            raise ValueError('incomplete stage contract validation')
        report['wall_seconds'] = time.perf_counter() - start_time
        return {'state': result.state, 'band': band, 'accepted': True, 'report': report}
    except (ValueError, TypeError, FloatingPointError, IndexError, AttributeError) as error:
        report.update(rejection_reason=str(error), wall_seconds=time.perf_counter() - start_time,
                      all_declared_stages_executed=report['executed_stages'] == list(STAGES))
        return {'state': before, 'band': None, 'accepted': False, 'report': report}
