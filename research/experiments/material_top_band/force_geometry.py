"""Static sigma momentum-dual geometry and instantaneous full-stock identities.

The Cavg operator is new: it pairs velocities on overlapping normalized-depth
intervals, not old common-physical-depth Cmin faces. Equal-half force allocation
is its declared weak allocation. No advanced state or finite step is returned.
"""
import math
from dataclasses import dataclass

import numpy as np

from . import inventory_pressure as p
from .pressure_footprint import bind_footprint_area, validate_footprint
from .real_geometry import _array, _readonly

CONTRACT = 'flat_bottom_sigma_dual_shared_stock_force_v1'


class UnsupportedGeometry(ValueError):
    """A missing physical/discrete contract, never a zero-force fallback."""


@dataclass(frozen=True)
class DualSegment:
    left: tuple
    right: tuple
    left_layer: int
    right_layer: int
    sigma_lower: float
    sigma_upper: float
    length_m: float
    normal: tuple
    distance_m: float
    area_m2: float


@dataclass(frozen=True)
class StockRates:
    hdot: np.ndarray
    etadot: np.ndarray
    divergence: np.ndarray
    relative_flux: np.ndarray
    stock_transport: np.ndarray
    pressure_momentum: np.ndarray
    shared_Q: np.ndarray
    transport_budget: dict
    energy: dict
    kinetic_transport: dict


def _sum(values):
    if not np.isfinite(values).all():
        raise ValueError('finite local contributions required')
    try:
        result = math.fsum(float(x) for x in np.asarray(values).ravel())
    except OverflowError as exc:
        raise ValueError('finite aggregate required') from exc
    if not math.isfinite(result):
        raise ValueError('finite aggregate required')
    return result


def _bind(profile, footprint, requests, walls):
    if not isinstance(profile, p.Profile):
        raise ValueError('inventory Profile required; force receipts are not authority')
    profile = p.reconstruct(profile.state, eos=profile.eos, representation=profile.representation,
                            external_pressure_Pa=profile.external_pressure_Pa)
    if not isinstance(footprint, p.RectangularFootprint):
        raise ValueError('bound rectangular footprint required')
    footprint = p.RectangularFootprint(_readonly(_array(footprint.bounds_m, 'footprint bounds')),
                                      _readonly(_array(footprint.area_m2, 'footprint area')),
                                      tuple(footprint.periodic_extent_m), tuple(footprint.periodic_origin_m))
    requests = tuple(p.FaceRequest(tuple(r.left), tuple(r.right), r.length, tuple(r.normal), tuple(r.periodic_shift_m)) for r in requests)
    walls = tuple(p.OuterWall(tuple(w.column), w.length, tuple(w.normal)) for w in walls)
    validate_footprint(profile.state, footprint, requests, walls)
    bind_footprint_area(profile.state, footprint, footprint.area_m2)
    wet = profile.state.active_layers > 0
    if not np.any(wet):
        raise UnsupportedGeometry('empty_wet_domain: no scientific qualification')
    with np.errstate(over='ignore', invalid='ignore'):
        heights = profile.state.eta[wet] - profile.state.bottom[wet]
    if not np.isfinite(heights).all() or np.any(heights <= 0.):
        raise ValueError('finite positive total wet column height required')
    bottoms = profile.state.bottom[wet]
    if np.any(bottoms != bottoms[0]):
        raise UnsupportedGeometry('step_bottom: cut dual polygon and vertical solid contour are not implemented')
    for col in np.ndindex(profile.state.eta.shape):
        n = int(profile.state.active_layers[col])
        if not n:
            continue
        m = min(3, n)
        if profile.state.band_bottom[col] != profile.state.interfaces[col][m]:
            raise ValueError('band_bottom must equal the active fixed top-band interface')
    return profile, footprint, requests, walls


def _segments(profile, footprint, requests):
    state = profile.state
    result = []
    for request in requests:
        left, right = request.left, request.right
        if not state.active_layers[left] or not state.active_layers[right]:
            continue  # Bound wet/dry adjacency is impermeable, never a water face.
        heights = [state.eta[c] - state.bottom[c] for c in [left, right]]
        edges = [(state.interfaces[c][:int(state.active_layers[c]) + 1] - state.bottom[c]) / H
                 for c, H in zip([left, right], heights)]
        cuts = np.unique(np.r_[edges[0], edges[1]])
        centers = [np.array([.5 * (footprint.bounds_m[c][0] + footprint.bounds_m[c][1]),
                             .5 * (footprint.bounds_m[c][2] + footprint.bounds_m[c][3])]) for c in [left, right]]
        distance = float(np.dot(centers[1] + request.periodic_shift_m - centers[0], request.normal))
        if not math.isfinite(distance) or distance <= 0.:
            raise ValueError('resolved positive dual center distance required')
        for low, high in zip(cuts[:-1], cuts[1:]):
            middle = .5 * (low + high)
            layers = [int(np.searchsorted(-edge, -middle, side='right') - 1) for edge in edges]
            area = request.length * (.5 * heights[0] + .5 * heights[1]) * (high - low)
            if not math.isfinite(area) or area <= 0. or any(k < 0 or k >= state.active_layers[c] for c, k in zip([left, right], layers)):
                raise ValueError('resolved positive sigma segment required')
            result.append(DualSegment(left, right, *layers, float(low), float(high), request.length,
                                      request.normal, distance, float(area)))
    return tuple(result)


def _z(profile, column, sigma):
    state = profile.state
    if sigma == 0.:
        return float(state.bottom[column])
    if sigma == 1.:
        return float(state.eta[column])
    # Keep algebraic normalized coordinates inside the already bound domain.
    return min(float(state.eta[column]), max(float(state.bottom[column]),
               float(state.bottom[column] + sigma * (state.eta[column] - state.bottom[column]))))


def _lift_pressure(profile, segment, x, sigma):
    left, right = segment.left, segment.right
    H = [(profile.state.eta[c] - profile.state.bottom[c]) for c in [left, right]]
    anomalies = [p.density_integral(profile, c, _z(profile, c, sigma), profile.state.eta[c]) / height
                 for c, height in zip([left, right], H)]
    height = (1. - x) * H[0] + x * H[1]
    external = (1. - x) * profile.external_pressure_Pa[left] + x * profile.external_pressure_Pa[right]
    return external + profile.eos.gravity * height * (profile.eos.rho0 * (1. - sigma)
                                                    + (1. - x) * anomalies[0] + x * anomalies[1])


def _contour(profile, segment):
    lo, hi, L = segment.sigma_lower, segment.sigma_upper, segment.length_m
    left, right = segment.left, segment.right
    jump_H = profile.state.eta[right] - profile.state.eta[left]
    offset = .5 / math.sqrt(3.)
    components = dict(left=L * p.pressure_integral(profile, left, _z(profile, left, lo), _z(profile, left, hi)),
                      right=-L * p.pressure_integral(profile, right, _z(profile, right, lo), _z(profile, right, hi)),
                      top=L * jump_H * hi * .5 * sum(_lift_pressure(profile, segment, x, hi) for x in [.5 - offset, .5 + offset]),
                      bottom=-L * jump_H * lo * .5 * sum(_lift_pressure(profile, segment, x, lo) for x in [.5 - offset, .5 + offset]))
    if not all(math.isfinite(v) for v in components.values()):
        raise ValueError('finite individual physical contour contributions required')
    total = _sum(list(components.values()))
    scale = _sum([abs(v) for v in components.values()])
    if not all(math.isfinite(v) for v in [total, scale]):
        raise ValueError('finite physical contour required')
    return components, total, float(p._bound(scale))


def static_contours(profile, footprint, requests, walls):
    """General P1 flat-bottom contour diagnostic with an explicit horizontal lift.

    CV P1 data alone does not specify that lift or a stock-PE gradient. This
    diagnostic never permits P1 force consumption or an ALE pressure kick.
    """
    profile, footprint, requests, walls = _bind(profile, footprint, requests, walls)
    result = []
    for segment in _segments(profile, footprint, requests):
        components, force, bound = _contour(profile, segment)
        result.append(dict(segment=segment, components_N=components, contour_force_N=force, bound_N=bound,
                           force_consumption_qualified=False,
                           reason='general_P1: geometry-bound Ctranspose/stock-PE/ALE work coupling unproved',
                           horizontal_density_lift='linear_in_x_of_inventory_P1_at_equal_sigma'))
    return tuple(result)


@dataclass(frozen=True, init=False)
class BarotropicOperator:
    profile: p.Profile
    footprint: p.RectangularFootprint
    requests: tuple
    walls: tuple
    segments: tuple
    density_kg_m3: float

    def __init__(self, profile, footprint, requests, walls):
        profile, footprint, requests, walls = _bind(profile, footprint, requests, walls)
        wet = profile.state.wet_mask
        means = profile.density_mean[wet]
        anomaly = float(means[0])
        # A local roundoff envelope must not grow with the domain size.
        tolerance = float(p._bound(profile.eos.rho0 + np.max(abs(means))))
        if np.any(abs(means - anomaly) > tolerance) or np.any(abs(profile.density_slope[wet] * profile.state.h[wet]) > tolerance):
            raise UnsupportedGeometry('general_P1: nonuniform density needs a proved stock-PE/contour/ALE coupling')
        for name, value in dict(profile=profile, footprint=footprint, requests=requests, walls=walls,
                                segments=_segments(profile, footprint, requests), density_kg_m3=profile.eos.rho0 + anomaly).items():
            object.__setattr__(self, name, value)
        if not all(item['accepted'] for item in self.contour_certificate()):
            raise UnsupportedGeometry('physical_contour: derived shared transpose failed independent local traction closure')

    def velocity(self):
        result = np.zeros(self.profile.state.h.shape + (2,))
        with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
            mass = self.profile.eos.rho0 * self.profile.state.h
            np.divide(self.profile.state.stocks[..., 2:], mass[..., None], out=result,
                      where=self.profile.state.wet_mask[..., None])
            squared = np.sum(result**2, axis=-1)
        if (not all(np.isfinite(a).all() for a in [mass, result, squared])
                or np.any(mass[self.profile.state.wet_mask] <= 0.)):
            raise ValueError('finite current mass, velocity and squared speed required')
        return result

    def shared_flux(self, velocity=None):
        velocity = self.velocity() if velocity is None else _array(velocity, 'velocity')
        if velocity.shape != self.profile.state.h.shape + (2,):
            raise ValueError('column-layer vector velocity shape required')
        result = np.array([.5 * s.area_m2 * np.dot(velocity[s.left][s.left_layer] + velocity[s.right][s.right_layer], s.normal)
                           for s in self.segments])
        if not np.isfinite(result).all():
            raise ValueError('finite shared volume flux required')
        return _readonly(result)

    def potential(self):
        profile = self.profile
        with np.errstate(over='ignore', invalid='ignore'):
            values = np.array([profile.external_pressure_Pa[s.right] - profile.external_pressure_Pa[s.left]
                               + self.density_kg_m3 * profile.eos.gravity * (profile.state.eta[s.right] - profile.state.eta[s.left])
                               for s in self.segments])
        if not np.isfinite(values).all():
            raise ValueError('finite pressure potential required')
        return _readonly(values)

    def column_flux_diagnostics(self, velocity=None):
        """Sum accepted segment Q and independently compare depth-mean formula.

        A mismatch refuses the diagnostic; no layer Q is modified or repaired.
        """
        velocity = self.velocity() if velocity is None else _array(velocity, 'velocity')
        flux = self.shared_flux(velocity)
        state = self.profile.state
        rows = []
        for request in self.requests:
            if not state.active_layers[request.left] or not state.active_layers[request.right]:
                rows.append(dict(column_Q_m3_s=0., depth_mean_Q_m3_s=0., roundoff_ratio=0., dry_face=True))
                continue
            terms = [q for s, q in zip(self.segments, flux)
                     if s.left == request.left and s.right == request.right and s.normal == request.normal]
            total = math.fsum(terms)
            means, heights = [], []
            for col in [request.left, request.right]:
                height = float(state.eta[col] - state.bottom[col])
                means.append(_sum(state.h[col] * (velocity[col] @ request.normal)) / height)
                heights.append(height)
            expected = request.length * (.5 * heights[0] + .5 * heights[1]) * (.5 * means[0] + .5 * means[1])
            bound = float(p._bound(math.fsum(abs(q) for q in terms) + abs(expected)))
            residual = abs(total - expected)
            if residual > bound:
                raise ValueError('column Q does not match the bound segment sum; no flux repair permitted')
            rows.append(dict(column_Q_m3_s=total, depth_mean_Q_m3_s=expected, roundoff_ratio=residual / bound, dry_face=False))
        return tuple(rows)

    def transpose(self, potential):
        potential = _array(potential, 'face potential')
        if potential.shape != (len(self.segments),):
            raise ValueError('one pressure potential per bound segment required')
        result = np.zeros(self.profile.state.h.shape + (2,))
        for s, value in zip(self.segments, potential):
            force = -.5 * s.area_m2 * value * np.asarray(s.normal)
            result[s.left][s.left_layer] += force / self.footprint.area_m2[s.left]
            result[s.right][s.right_layer] += force / self.footprint.area_m2[s.right]
        if not np.isfinite(result).all():
            raise ValueError('finite pressure stock rate required')
        return _readonly(result)

    def force_total(self):
        force = self.transpose(self.potential()) * self.footprint.area_m2[..., None, None]
        return np.array([_sum(force[..., axis]) for axis in range(2)])

    def contour_certificate(self):
        from .force_contour_oracle import contour
        result = []
        for s, value in zip(self.segments, self.potential()):
            components, candidate, bound = _contour(self.profile, s)
            independent = contour(self.profile, s)
            algebraic = -s.area_m2 * value
            bound += independent['bound_N'] + float(p._bound(abs(algebraic)))
            residual = max(abs(algebraic - independent['force_N']), abs(candidate - independent['force_N']))
            if not all(math.isfinite(v) for v in [algebraic, candidate, bound, residual, residual / bound]):
                raise ValueError('finite physical force certificate required')
            result.append(dict(accepted=residual <= bound, components_N=components, contour_force_N=candidate,
                               independent_force_N=independent['force_N'], Ctranspose_force_N=algebraic,
                               bound_N=bound, roundoff_ratio=residual / bound))
        return tuple(result)

    def midpoint_identity(self):
        """One-second unit impulse algebra using C on actual momentum midpoints."""
        self.velocity()  # Bind finite current mass/squared speed even with no faces.
        state, area = self.profile.state, self.footprint.area_m2
        before = state.stocks[..., 2:]
        impulse = self.transpose(self.potential())
        after = before + impulse
        midpoint, kinetic_change, kinetic_scale = np.zeros_like(before), np.zeros_like(state.h), np.zeros_like(state.h)
        np.divide(before + after, 2. * self.profile.eos.rho0 * state.h[..., None],
                  out=midpoint, where=state.wet_mask[..., None])
        np.divide(np.sum((after - before) * (after + before), axis=-1), 2. * self.profile.eos.rho0 * state.h,
                  out=kinetic_change, where=state.wet_mask)
        np.divide(np.sum(before**2 + after**2, axis=-1), 2. * self.profile.eos.rho0 * state.h,
                  out=kinetic_scale, where=state.wet_mask)
        change = _sum(area[..., None] * kinetic_change)
        work_terms = -self.potential() * self.shared_flux(midpoint)
        work = _sum(work_terms)
        bound = float(p._bound(_sum(area[..., None] * kinetic_scale) + _sum(abs(work_terms))))
        if not all(math.isfinite(value) for value in [change, work, bound]):
            raise ValueError('finite fixed-mass midpoint identity required')
        return dict(algebraic_impulse_only=True, impulse_duration_s=1., accepted_state_returned=False,
                    time_step_executed=False, kinetic_change_J=change, midpoint_work_J=work,
                    work_bound_J=bound, work_roundoff_ratio=abs(change - work) / bound)

    def rates(self):
        """All-column instantaneous transport/pressure rates; no accepted commit."""
        state, area, rho0 = self.profile.state, self.footprint.area_m2, self.profile.eos.rho0
        Q, potential, velocity = self.shared_flux(), self.potential(), self.velocity()
        self.column_flux_diagnostics(velocity)
        D = np.zeros_like(state.h)
        edges = []
        for s, flux in zip(self.segments, Q):
            D[s.left][s.left_layer] += flux / area[s.left]
            D[s.right][s.right_layer] -= flux / area[s.right]
            edges.append((s.left, s.left_layer, s.right, s.right_layer, flux))
        etadot = -D.sum(axis=-1)
        hdot, R = np.zeros_like(state.h), np.zeros(state.interfaces.shape)
        for col in np.ndindex(state.eta.shape):
            n = int(state.active_layers[col])
            if not n:
                continue
            m = min(3, n)
            fractions = state.h[col][:m] / (state.eta[col] - state.band_bottom[col])
            if abs(_sum(fractions) - 1.) > float(p._bound(_sum(abs(fractions)))):
                raise ValueError('fixed top fractions must sum to one')
            hdot[col][:m] = fractions * etadot[col]
            for k in range(n - 1, m - 1, -1):
                R[col][k] = R[col][k + 1] + D[col][k]
            bottom_value = R[col][m]
            for k in range(m):
                value = R[col][k] - D[col][k] - hdot[col][k]
                if k + 1 < m:
                    R[col][k + 1] = value
                elif abs(value - bottom_value) > float(p._bound(_sum(abs(D[col])) + _sum(abs(hdot[col])))):
                    raise ValueError('top/deep ALE band flux mismatch')
            for k in range(1, n):
                edges.append((col, k - 1, col, k, area[col] * R[col][k]))
        specific = np.zeros_like(state.stocks)
        np.divide(state.stocks, state.h[..., None], out=specific, where=state.wet_mask[..., None])
        if not np.isfinite(specific).all():
            raise ValueError('finite specific stock required')
        transport = np.zeros_like(state.stocks)
        loss_terms = []
        for left, kl, right, kr, flux in edges:
            donor = specific[left][kl] if flux >= 0. else specific[right][kr]
            stock_flux = flux * donor
            transport[left][kl] -= stock_flux / area[left]
            transport[right][kr] += stock_flux / area[right]
            loss_terms.append(.5 * rho0 * abs(flux) * float(np.sum((velocity[right][kr] - velocity[left][kl])**2)))
        pressure_rate = self.transpose(potential)
        pressure_power = -_sum(potential * Q)
        gravity_rate = self.density_kg_m3 * self.profile.eos.gravity * _sum(area * state.eta * etadot)
        external_power = -_sum(area * self.profile.external_pressure_Pa * etadot)
        bound = float(p._bound(abs(pressure_power) + abs(gravity_rate) + abs(external_power)))
        from .force_contour_oracle import stock_pe_direction_ledger
        stock_PE = stock_pe_direction_ledger(self.profile, area, hdot, transport, etadot)
        PE_bound = stock_PE['bound_W'] + float(p._bound(abs(gravity_rate)))
        if abs(stock_PE['direction_W'] - gravity_rate) > PE_bound:
            raise UnsupportedGeometry('stock_PE_tangent: full inventory direction failed the barotropic pressure pairing')
        pairing_bound = bound + stock_PE['bound_W']
        pairing_residual = abs(pressure_power + stock_PE['direction_W'] - external_power)
        if pairing_residual > pairing_bound:
            raise UnsupportedGeometry('stock_PE_tangent: pressure work and full inventory PE direction do not close')
        energy = dict(pressure_power_W=pressure_power, gravity_PE_rate_W=stock_PE['direction_W'],
                      collapsed_barotropic_PE_rate_W=gravity_rate, stock_PE_direction=stock_PE,
                      stock_PE_binding_bound_W=PE_bound,
                      external_pressure_power_W=external_power, bound_W=bound,
                      full_stock_pairing_bound_W=pairing_bound,
                      pairing_residual_ratio=pairing_residual / pairing_bound)
        kinetic_terms = area[..., None] * (np.sum(velocity * transport[..., 2:], axis=-1)
                                           - .5 * rho0 * np.sum(velocity**2, axis=-1) * hdot)
        kinetic_rate, loss = _sum(kinetic_terms), math.fsum(loss_terms)
        kinetic_bound = float(p._bound(_sum(abs(kinetic_terms)) + loss))
        kinetic = dict(transport_kinetic_rate_W=kinetic_rate, upwind_loss_W=loss, bound_W=kinetic_bound,
                       residual_ratio=abs(kinetic_rate + loss) / kinetic_bound)
        budget = dict(water_residual_m3_s=_sum(area[..., None] * hdot),
                      stock_residual_per_s=[_sum(area[..., None] * transport[..., j]) for j in range(4)])
        arrays = [hdot, etadot, D, R, transport, pressure_rate, Q]
        if not all(np.isfinite(a).all() for a in arrays):
            raise ValueError('finite full-stock rate required')
        scalars = list(stock_PE.values()) + [value for value in energy.values() if isinstance(value, float)]
        scalars += list(kinetic.values()) + [budget['water_residual_m3_s']] + budget['stock_residual_per_s']
        if not all(math.isfinite(value) for value in scalars):
            raise ValueError('every full-stock ledger scalar must be finite')
        return StockRates(*[_readonly(a) for a in arrays], budget, energy, kinetic)
