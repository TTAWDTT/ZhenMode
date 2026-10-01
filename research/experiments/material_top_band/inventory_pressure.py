"""Static inventory-authority P1 hydrostatics and an independent force gate.

The profile fills defined control volumes, not missing observations. The
algebraic C-transpose work trial never returns an advanced/accepted state.
Solid reactions are pressure integrals over supplied physical wall faces;
they are never defined by negating the trial's total fluid impulse.
"""

import math
from dataclasses import dataclass, fields

import numpy as np

from .pressure_footprint import RectangularFootprint as RectangularFootprint
from .pressure_footprint import bind_footprint_area, validate_footprint
from .real_geometry import ColumnStocks, _array, _column, _readonly, _scalar

CONTRACT = 'inventory_mean_density_p1_hydrostatic_v1'
EPSILON = np.finfo(float).eps


def _bound(scale):
    return 256. * EPSILON * np.maximum(1., scale)


@dataclass(frozen=True)
class EOS:
    Tref: float = 15.
    Sref: float = 35.
    alpha: float = 2e-4
    beta: float = 7.6e-4
    rho0: float = 1025.
    gravity: float = 9.81


@dataclass(frozen=True)
class Profile:
    state: ColumnStocks
    eos: EOS
    temperature_mean: np.ndarray
    salinity_mean: np.ndarray
    temperature_slope: np.ndarray
    salinity_slope: np.ndarray
    density_mean: np.ndarray
    density_slope: np.ndarray
    external_pressure_Pa: np.ndarray
    slope_limited_count: int
    representation: str
    observational_profile_recovered: bool = False
    density_slope_limited_count: int = 0
    density_authority: str = 'direct_EOS_inventory_mean_P1'
    pointwise_density_equals_auxiliary_TS_reconstruction: bool = False


@dataclass(frozen=True)
class FaceRequest:
    left: tuple
    right: tuple
    length: float
    normal: tuple
    periodic_shift_m: tuple = (0., 0.)


@dataclass(frozen=True)
class OuterWall:
    column: tuple
    length: float
    normal: tuple


@dataclass(frozen=True)
class PressureFace:
    left: tuple
    right: tuple
    left_layer: int
    right_layer: int
    upper_z_m: float
    lower_z_m: float
    area_m2: float
    normal: tuple
    pressure_jump_Pa: float
    volume_flux_m3_s: float


class PressureForceIncompatibility(ValueError):
    pass


def _copy_inventory(state):
    if not isinstance(state, ColumnStocks):
        raise ValueError('ColumnStocks inventory required')
    count = np.asarray(state.active_layers)
    if np.ma.isMaskedArray(state.active_layers) or count.dtype.kind not in 'iu':
        raise ValueError('integer active layer counts required')
    h, stocks, z, eta, bottom, band = [_array(value, name) for value, name in
        [(state.h, 'h'), (state.stocks, 'stocks'), (state.interfaces, 'interfaces'),
         (state.eta, 'eta'), (state.bottom, 'bottom'), (state.band_bottom, 'band_bottom')]]
    mask = _array(state.wet_mask, 'mask')
    if h.ndim < 2 or stocks.shape != h.shape + (4,) or z.shape != h.shape[:-1] + (h.shape[-1] + 1,) or any(a.shape != h.shape[:-1] for a in [eta, bottom, band, count]) or mask.shape != h.shape:
        raise ValueError('finite inventory shape mismatch')
    if np.any(count < 0) or np.any(count > h.shape[-1]):
        raise ValueError('invalid active layer count')
    wet = np.arange(h.shape[-1]) < count[..., None]
    if not np.array_equal(mask, wet):
        raise ValueError('binary prefix inventory required')
    if np.any(h[wet] <= 0) or np.any(h[~wet] != 0) or np.any(stocks[~wet] != 0):
        raise ValueError('positive wet inventory and zero dry inventory required')
    if np.any(abs(-np.diff(z, axis=-1) - h) > _bound(abs(h) + abs(z[..., :-1]) + abs(z[..., 1:]))):
        raise ValueError('physical interfaces must match inventory thickness')
    if not np.array_equal(z[..., 0], eta):
        raise ValueError('surface interface mismatch')
    for index in np.ndindex(eta.shape):
        n = int(count[index])
        if z[index][n] != bottom[index] or np.any(z[index][n:] != bottom[index]) or np.any(np.diff(z[index][:n + 1]) >= 0):
            raise ValueError('ordered active interfaces and repeated dry bottom required')
        if not n and eta[index] != 0:
            raise ValueError('dry eta must be zero')
    return ColumnStocks(*[_readonly(a) for a in [count.copy(), wet, eta, bottom, band, z, h, stocks]])


def reconstruct(state, *, eos=EOS(), representation='p1', external_pressure_Pa=0.):
    if not isinstance(eos, EOS) or any(_scalar(getattr(eos, item.name), 'EOS') != getattr(EOS(), item.name) for item in fields(EOS)):
        raise ValueError('EOS must match the original inventory contract exactly')
    if representation not in ['p0', 'p1']:
        raise ValueError('explicit p0 or p1 representation required')
    state = _copy_inventory(state)
    external = _array(external_pressure_Pa, 'external pressure')
    if external.shape not in [(), state.eta.shape]:
        raise ValueError('external pressure scalar/column shape required')
    external = np.broadcast_to(external, state.eta.shape).copy()
    means = np.zeros(state.h.shape + (2,))
    np.divide(state.stocks[..., :2], state.h[..., None], out=means, where=state.wet_mask[..., None])
    limits_low, limits_high = np.array([-5., 0.]), np.array([45., 50.])
    tolerance = _bound(abs(means) + limits_high)
    if np.any((means < limits_low - tolerance)[state.wet_mask]) or np.any((means > limits_high + tolerance)[state.wet_mask]):
        raise ValueError('stock means outside physical tracer bounds; no mean clipping allowed')
    slopes = np.zeros_like(means)
    center = .5 * (state.interfaces[..., :-1] + state.interfaces[..., 1:])
    limited = 0
    for index in np.ndindex(state.eta.shape):
        n = int(state.active_layers[index])
        if representation == 'p1' and n >= 2:
            secant = np.diff(means[index][:n], axis=0) / np.diff(center[index][:n])[:, None]
            slopes[index][0], slopes[index][n - 1] = secant[0], secant[-1]
            left, right = secant[:-1], secant[1:]
            slopes[index][1:n - 1] = np.where(left * right > 0, np.sign(left) * np.minimum(abs(left), abs(right)), 0.)
            excursion = .5 * state.h[index][:n, None] * abs(slopes[index][:n])
            room = np.maximum(0., np.minimum(means[index][:n] - limits_low, limits_high - means[index][:n]))
            theta = np.ones_like(room)
            np.divide(np.minimum(room, excursion), excursion, out=theta, where=excursion > 0)
            limited += int(np.count_nonzero(theta < 1.))
            slopes[index][:n] *= theta
    rho = eos.rho0 * (-eos.alpha * (means[..., 0] - eos.Tref) + eos.beta * (means[..., 1] - eos.Sref))
    rho *= state.wet_mask
    # Pressure/PE share an independent density-mean representation. Separate
    # nonlinear T/S limiter branches need not preserve an affine EOS density.
    slope = np.zeros_like(rho)
    density_limited = 0
    rho_low = eos.rho0 * (-eos.alpha * (45. - eos.Tref) + eos.beta * (0. - eos.Sref))
    rho_high = eos.rho0 * (-eos.alpha * (-5. - eos.Tref) + eos.beta * (50. - eos.Sref))
    for index in np.ndindex(state.eta.shape):
        n = int(state.active_layers[index])
        if representation == 'p1' and n >= 2:
            secant = np.diff(rho[index][:n]) / np.diff(center[index][:n])
            slope[index][0], slope[index][n - 1] = secant[0], secant[-1]
            left, right = secant[:-1], secant[1:]
            slope[index][1:n - 1] = np.where(left * right > 0, np.sign(left) * np.minimum(abs(left), abs(right)), 0.)
            excursion = .5 * state.h[index][:n] * abs(slope[index][:n])
            room = np.maximum(0., np.minimum(rho[index][:n] - rho_low, rho_high - rho[index][:n]))
            theta = np.ones_like(room)
            np.divide(np.minimum(room, excursion), excursion, out=theta, where=excursion > 0)
            density_limited += int(np.count_nonzero(theta < 1.))
            slope[index][:n] *= theta
    if not all(np.isfinite(a).all() for a in [means, slopes, rho, slope]) or np.any((eos.rho0 + rho - .5 * state.h * abs(slope))[state.wet_mask] <= 0):
        raise ValueError('finite positive reconstructed density required')
    return Profile(state, eos, *[_readonly(a) for a in
                   [means[..., 0].copy(), means[..., 1].copy(), slopes[..., 0].copy(), slopes[..., 1].copy(), rho, slope, external]],
                   limited, representation, density_slope_limited_count=density_limited)


def _interval(profile, column, lower, upper):
    column = _column(profile.state, column)
    lower, upper = _scalar(lower, 'lower'), _scalar(upper, 'upper')
    if not profile.state.active_layers[column] or lower > upper or lower < profile.state.bottom[column] or upper > profile.state.eta[column]:
        raise ValueError('interval must lie in an active physical control-volume domain')
    return column, lower, upper


def density_integral(profile, column, lower, upper, *, first_moment=False):
    column, lower, upper = _interval(profile, column, lower, upper)
    z = profile.state.interfaces[column]
    terms = []
    for k in range(int(profile.state.active_layers[column])):
        lo, hi = max(lower, z[k + 1]), min(upper, z[k])
        if hi <= lo:
            continue
        center = .5 * (z[k] + z[k + 1])
        mean, slope = profile.density_mean[column][k], profile.density_slope[column][k]
        if first_moment:
            # Local coordinates avoid cancellation of differences of deep z^3.
            width, middle = hi - lo, .5 * (hi + lo)
            terms.append(width * ((mean + slope * (middle - center)) * middle + slope * width**2 / 12.))
        else:
            terms.append((hi - lo) * (mean + slope * (.5 * (hi + lo) - center)))
    result = math.fsum(terms)
    if not math.isfinite(result):
        raise ValueError('finite hydrostatic density integral required')
    return result


def pressure(profile, column, z, *, reduced=False):
    column, z, _ = _interval(profile, column, z, z)
    eta = float(profile.state.eta[column])
    baseline = profile.eos.rho0 * (eta if reduced else eta - z)
    value = float(profile.external_pressure_Pa[column]) + profile.eos.gravity * (baseline + density_integral(profile, column, z, eta))
    if not math.isfinite(value):
        raise ValueError('finite hydrostatic pressure required')
    return value


def pressure_integral(profile, column, lower, upper, *, reduced=False):
    column, lower, upper = _interval(profile, column, lower, upper)
    cuts = profile.state.interfaces[column][:int(profile.state.active_layers[column]) + 1]
    cuts = np.unique(np.r_[lower, upper, cuts[(cuts > lower) & (cuts < upper)]])
    terms = []
    for lo, hi in zip(cuts[:-1], cuts[1:]):
        middle, offset = .5 * (lo + hi), (hi - lo) / (2 * math.sqrt(3.))
        terms.append(.5 * (hi - lo) * (pressure(profile, column, middle - offset, reduced=reduced)
                                       + pressure(profile, column, middle + offset, reduced=reduced)))
    result = math.fsum(terms)
    if not math.isfinite(result):
        raise ValueError('finite pressure face integral required')
    return result


def potential_energy(profile, area, *, footprint=None):
    area = _array(area, 'area')
    if footprint is not None:
        area = bind_footprint_area(profile.state, footprint, area)
    if area.shape != profile.state.eta.shape or np.any(area <= 0):
        raise ValueError('positive area for each original column required')
    terms = []
    for index in np.ndindex(area.shape):
        if not profile.state.active_layers[index]:
            continue
        anomaly = density_integral(profile, index, profile.state.bottom[index], profile.state.eta[index], first_moment=True)
        terms.append(float(area[index]) * profile.eos.gravity * (.5 * profile.eos.rho0 * profile.state.eta[index]**2 + anomaly))
    result = math.fsum(terms)
    if not math.isfinite(result):
        raise ValueError('finite inventory representation PE required')
    return result


def potential_energy_difference_ledger(before, after, area):
    """Integrate representation differences directly on the same physical domain."""
    area = _array(area, 'area')
    if area.shape != before.state.eta.shape or np.any(area <= 0):
        raise ValueError('positive area for representation difference required')
    if (before.eos != after.eos or before.state.eta.shape != after.state.eta.shape
            or not np.array_equal(before.state.eta, after.state.eta)
            or not np.array_equal(before.state.bottom, after.state.bottom)
            or not np.array_equal(before.state.active_layers > 0, after.state.active_layers > 0)):
        raise ValueError('representations must have the same physical domain and EOS')
    terms, scales = [], []
    for column in np.ndindex(area.shape):
        nb, na = int(before.state.active_layers[column]), int(after.state.active_layers[column])
        if not nb:
            continue
        zb, za = before.state.interfaces[column][:nb + 1], after.state.interfaces[column][:na + 1]
        cuts = np.unique(np.r_[zb, za])
        for lo, hi in zip(cuts[:-1], cuts[1:]):
            width, middle = hi - lo, .5 * (hi + lo)
            kb, ka = (int(np.searchsorted(-z, -middle, side='right') - 1) for z in [zb, za])
            cb, ca = .5 * (zb[kb] + zb[kb + 1]), .5 * (za[ka] + za[ka + 1])
            rb, ra = before.density_mean[column][kb], after.density_mean[column][ka]
            sb, sa = before.density_slope[column][kb], after.density_slope[column][ka]
            delta = (ra - rb) + sa * (middle - ca) - sb * (middle - cb)
            moment = width * (delta * middle + (sa - sb) * width**2 / 12.)
            terms.append(float(area[column]) * before.eos.gravity * moment)
            scales.append(float(area[column]) * before.eos.gravity * width
                          * (abs(delta * middle) + abs(sa - sb) * width**2 / 12.))
    result = math.fsum(terms)
    scale = math.fsum(scales)
    if not math.isfinite(result) or not math.isfinite(scale):
        raise ValueError('finite direct representation PE difference required')
    return dict(change_J=result, absolute_contribution_scale_J=scale)


def potential_energy_difference(before, after, area):
    return potential_energy_difference_ledger(before, after, area)['change_J']


def _face_geometry(profile, request):
    left, right = _column(profile.state, request.left), _column(profile.state, request.right)
    if left == right:
        raise ValueError('distinct face columns required')
    length = _scalar(request.length, 'face length', positive=True)
    normal = _array(request.normal, 'normal')
    if normal.shape != (2,) or abs(float(np.dot(normal, normal)) - 1.) > 8 * EPSILON:
        raise ValueError('unit two-component face normal required')
    return left, right, length, normal


def pressure_faces(profile, requests):
    state, output = profile.state, []
    for request in requests:
        left, right, length, normal = _face_geometry(profile, request)
        nl, nr = int(state.active_layers[left]), int(state.active_layers[right])
        if not nl or not nr:
            continue
        lower, upper = max(state.bottom[left], state.bottom[right]), min(state.eta[left], state.eta[right])
        if lower >= upper:
            raise ValueError('disjoint wet domains require a separately declared boundary contract')
        zl, zr = state.interfaces[left][:nl + 1], state.interfaces[right][:nr + 1]
        cuts = np.unique(np.r_[lower, upper, zl[(zl > lower) & (zl < upper)], zr[(zr > lower) & (zr < upper)]])
        for lo, hi in zip(cuts[:-1], cuts[1:]):
            middle = .5 * (lo + hi)
            il, ir = (int(np.searchsorted(-z, -middle, side='right') - 1) for z in [zl, zr])
            offset = (hi - lo) / (2 * math.sqrt(3.))
            jumps = [pressure(profile, right, middle + shift, reduced=True) - pressure(profile, left, middle + shift, reduced=True)
                     for shift in [-offset, offset]]
            S = length * (hi - lo)
            vl = state.stocks[left][il, 2:] / (profile.eos.rho0 * state.h[left][il])
            vr = state.stocks[right][ir, 2:] / (profile.eos.rho0 * state.h[right][ir])
            Q = .5 * S * float(np.dot(vl + vr, normal))
            if not math.isfinite(Q) or not all(math.isfinite(value) for value in jumps):
                raise ValueError('finite shared Q and pressure jump required')
            output.append(PressureFace(left, right, il, ir, float(hi), float(lo), float(S), tuple(normal),
                                       .5 * math.fsum(jumps), Q))
    return tuple(output)


def boundary_force_budget(profile, requests, outer_walls):
    """Integrate independent physical tractions; keep free caps separate.

    Closed footprint walls must be supplied explicitly. An unshared upper
    interval is a free-surface geometric cap, not a solid wall. Its physical
    external traction uses p_ext; its p-p_ext integral diagnoses the mismatch
    of the common-min-height C operator with the physical boundary force.
    """
    parts = {name: np.zeros(2) for name in ['outer_solid', 'dry_wall', 'staircase_solid', 'external_surface', 'surface_geometry_difference']}
    absolute_force = 0.
    area_closure = np.zeros(profile.state.eta.shape + (2,))
    footprint_scale = np.zeros_like(area_closure)
    def traction(category, column, length, normal, lo, hi, external_only=False, geometric_difference=False):
        nonlocal absolute_force
        if hi <= lo:
            return
        integral = float(profile.external_pressure_Pa[column]) * (hi - lo) if external_only else pressure_integral(profile, column, lo, hi)
        if geometric_difference:
            integral -= float(profile.external_pressure_Pa[column]) * (hi - lo)
        force = -length * integral * normal
        parts[category] += force
        absolute_force += float(np.abs(force).sum())

    for wall in outer_walls:
        column = _column(profile.state, wall.column)
        length = _scalar(wall.length, 'wall length', positive=True)
        normal = _array(wall.normal, 'normal')
        if normal.shape != (2,) or abs(float(np.dot(normal, normal)) - 1.) > 8 * EPSILON:
            raise ValueError('unit wall normal required')
        area_closure[column] += length * normal
        footprint_scale[column] += length * abs(normal)
        if profile.state.active_layers[column]:
            traction('outer_solid', column, length, normal, profile.state.bottom[column], profile.state.eta[column])
    for request in requests:
        left, right, length, normal = _face_geometry(profile, request)
        area_closure[left] += length * normal
        area_closure[right] -= length * normal
        footprint_scale[left] += length * abs(normal)
        footprint_scale[right] += length * abs(normal)
        nl, nr = profile.state.active_layers[left], profile.state.active_layers[right]
        if not nl and not nr:
            continue
        if not nl or not nr:
            column, outward = (left, normal) if nl else (right, -normal)
            traction('dry_wall', column, length, outward, profile.state.bottom[column], profile.state.eta[column])
            continue
        lower, upper = max(profile.state.bottom[left], profile.state.bottom[right]), min(profile.state.eta[left], profile.state.eta[right])
        if lower >= upper:
            raise ValueError('disjoint wet physical face domains')
        for column, outward in [(left, normal), (right, -normal)]:
            traction('staircase_solid', column, length, outward, profile.state.bottom[column], lower)
            traction('external_surface', column, length, outward, upper, profile.state.eta[column], external_only=True)
            traction('surface_geometry_difference', column, length, outward, upper, profile.state.eta[column], geometric_difference=True)
    solid = parts['outer_solid'] + parts['dry_wall'] + parts['staircase_solid']
    physical = solid + parts['external_surface']
    area_closure[profile.state.active_layers == 0] = 0.
    if not all(np.isfinite(value).all() for value in [*parts.values(), physical, area_closure, footprint_scale]) or not math.isfinite(absolute_force):
        raise ValueError('finite independent physical boundary force ledger required')
    return {**{name + '_force_N': value.tolist() for name, value in parts.items() if name != 'surface_geometry_difference'},
            'surface_geometry_difference_N': parts['surface_geometry_difference'].tolist(),
            'physical_boundary_force_N': physical.tolist(), 'solid_reaction_N': (-solid).tolist(),
            'wall_work_J': 0., 'absolute_traction_scale_N': absolute_force,
            'footprint_vector_closure_m': area_closure.tolist(),
            'absolute_footprint_contribution_m': footprint_scale.tolist()}


def certify_force_consumption(profile, requests=(), outer_walls=(), *, footprint=None):
    """Recompute qualification from inventory; diagnostic receipts are not authority."""
    if not isinstance(profile, Profile):
        raise ValueError('Profile inventory required; diagnostic receipts cannot authorize force')
    profile = reconstruct(profile.state, eos=profile.eos, representation=profile.representation,
                          external_pressure_Pa=profile.external_pressure_Pa)
    requests, outer_walls = tuple(requests), tuple(outer_walls)
    wet = bool(np.any(profile.state.active_layers))
    if not wet and footprint is None and not requests and not outer_walls:
        return dict(accepted=False, reason='empty all-dry domain has no wet scientific qualification',
                    footprint_closed=False, footprint_complete=False)
    geometry = validate_footprint(profile.state, footprint, requests, outer_walls)
    faces = pressure_faces(profile, requests)
    boundary = boundary_force_budget(profile, requests, outer_walls)
    forces = [-face.area_m2 * face.pressure_jump_Pa * np.asarray(face.normal) for face in faces]
    total = np.array([math.fsum(float(force[axis]) for force in forces) for axis in range(2)])
    physical = np.asarray(boundary['physical_boundary_force_N'], dtype=float)
    scale = float(boundary['absolute_traction_scale_N']) + math.fsum(float(np.abs(force).sum()) for force in forces)
    closure = np.asarray(boundary['footprint_vector_closure_m'], dtype=float)
    if physical.shape != (2,) or not all(np.isfinite(a).all() for a in [total, physical, closure]) or not math.isfinite(scale):
        raise ValueError('finite independent force certification required')
    residual, bound = total - physical, float(_bound(scale))
    footprint_closed = bool(np.all(abs(closure) <= _bound(np.asarray(boundary['absolute_footprint_contribution_m']))))
    accepted = wet and footprint_closed and bool(np.all(abs(residual) <= bound))
    return dict(accepted=accepted, Ctranspose_force_N=total.tolist(), physical_boundary_force_N=physical.tolist(),
                incompatibility_N=residual.tolist(), bound_N=bound, footprint_closed=footprint_closed,
                footprint_complete=geometry['complete'], footprint_geometry=geometry,
                reason='compatible static physical boundary force' if accepted else ('empty all-dry domain has no wet scientific qualification' if not wet else 'Ctranspose pressure force is incompatible with the independently integrated physical boundary/geometry ledger'))


def require_force_consumption(profile, requests=(), outer_walls=(), *, footprint=None):
    certificate = certify_force_consumption(profile, requests, outer_walls, footprint=footprint)
    if certificate.get('accepted') is not True:
        raise PressureForceIncompatibility(certificate.get('reason', 'pressure force not certified'))


def algebraic_midpoint_work(profile, faces, area, *, footprint=None):
    """One-second unit impulse as algebra only; no time step or state returned."""
    area = _array(area, 'area')
    state = profile.state
    if footprint is not None:
        area = bind_footprint_area(state, footprint, area)
    if area.shape != state.eta.shape or np.any(area <= 0):
        raise ValueError('positive bound column area required')
    impulse = np.zeros(state.h.shape + (2,))
    for face in faces:
        force = -.5 * face.area_m2 * face.pressure_jump_Pa * np.asarray(face.normal)
        impulse[face.left][face.left_layer] += force / area[face.left]
        impulse[face.right][face.right_layer] += force / area[face.right]
    before, after = state.stocks[..., 2:], state.stocks[..., 2:] + impulse
    midpoint = np.zeros_like(before)
    np.divide(before + after, 2. * profile.eos.rho0 * state.h[..., None], out=midpoint, where=state.wet_mask[..., None])
    change = np.zeros_like(state.h)
    np.divide(np.sum((after - before) * (after + before), axis=-1), 2. * profile.eos.rho0 * state.h, out=change, where=state.wet_mask)
    change = math.fsum((area[..., None] * change).ravel())
    work_terms = [-face.pressure_jump_Pa * .5 * face.area_m2 * float(np.dot(midpoint[face.left][face.left_layer] + midpoint[face.right][face.right_layer], face.normal)) for face in faces]
    work = math.fsum(work_terms)
    kinetic_scale = np.zeros_like(state.h)
    np.divide(np.sum(before**2 + after**2, axis=-1), 2. * profile.eos.rho0 * state.h, out=kinetic_scale, where=state.wet_mask)
    scale = math.fsum((area[..., None] * kinetic_scale).ravel()) + math.fsum(abs(value) for value in work_terms)
    bound = float(_bound(scale))
    if not all(math.isfinite(value) for value in [change, work, scale, bound]):
        raise ValueError('finite algebraic fixed-mass work required')
    return dict(algebraic_impulse_only=True, impulse_duration_s=1., accepted_state_returned=False,
                time_step_executed=False, kinetic_change_J=change, midpoint_work_J=work,
                work_bound_J=bound, work_roundoff_ratio=abs(change - work) / bound)
