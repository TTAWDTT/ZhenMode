"""Restricted physical pressure loads and affine instantaneous stock directions.

No accepted state, P0 upwind adapter or general-P1 pressure gradient is returned.
"""
import math
from dataclasses import dataclass

import numpy as np
from numpy.polynomial import Polynomial

from . import inventory_pressure as inventory

EPS = np.finfo(float).eps


class UnsupportedAffineContract(ValueError):
    """The explicit instantaneous affine contract does not cover this input."""


def total(values):
    array = np.asarray(values, dtype=float)
    if not np.isfinite(array).all():
        raise ValueError('finite local contributions required')
    try:
        result = math.fsum(float(x) for x in array.ravel())
    except OverflowError as exc:
        raise ValueError('finite aggregate required') from exc
    if not math.isfinite(result):
        raise ValueError('finite aggregate required')
    return result


def bound(scale):
    return 512. * EPS * max(1., float(scale))


def integral(polynomial, distance):
    primitive = polynomial.integ()
    return float(primitive(distance) - primitive(0.))


def polynomial_scale(polynomial, distance):
    return total([abs(c) * distance**(k + 1) / (k + 1) for k, c in enumerate(polynomial.coef)])


@dataclass(frozen=True)
class PhysicalStrip:
    lower_left: float
    upper_left: float
    lower_right: float
    upper_right: float
    left_layer: int
    right_layer: int
    actual_top: bool


class AffinePhysicalDual:
    """Copied two-column auxiliary momentum metric and full physical dual."""

    def __init__(self, profile, *, distance_m, length_m, node_area_m2=None):
        if not isinstance(profile, inventory.Profile):
            raise ValueError('inventory Profile required')
        self.profile = inventory.reconstruct(profile.state, eos=profile.eos,
                                             external_pressure_Pa=profile.external_pressure_Pa)
        self.state, self.eos = self.profile.state, self.profile.eos
        self.distance, self.length = float(distance_m), float(length_m)
        if not all(math.isfinite(v) and v > 0. for v in (self.distance, self.length)):
            raise ValueError('positive resolved physical dual dimensions required')
        if self.state.h.shape != (2, 14) or np.any(self.state.active_layers != 14):
            raise UnsupportedAffineContract('two fully wet endpoint columns required; dry/partial patch unsupported')
        if self.state.bottom[0] != self.state.bottom[1]:
            raise UnsupportedAffineContract('step bottom unsupported')
        self.bottom = float(self.state.bottom[0])
        area = self.distance * self.length / 2.
        if not math.isfinite(area) or area <= 0.:
            raise ValueError('finite positive resolved auxiliary area required')
        self.node_area_m2 = np.full(2, area) if node_area_m2 is None else np.asarray(node_area_m2, dtype=float).copy()
        if self.node_area_m2.shape != (2,) or not np.all(self.node_area_m2 == area):
            raise UnsupportedAffineContract('auxiliary area must equal dL/2 at both endpoints')
        self.node_area_m2.setflags(write=False)
        self.centers = .5 * (self.state.interfaces[:, :-1] + self.state.interfaces[:, 1:])
        self.density_mean = self.profile.density_mean
        self.eos_scale = self.eos.rho0 * (self.eos.alpha * (abs(self.profile.temperature_mean) + self.eos.Tref)
                                        + self.eos.beta * (abs(self.profile.salinity_mean) + self.eos.Sref))
        coefficients = []
        for side in range(2):
            slope = (self.density_mean[side, -1] - self.density_mean[side, 0]) / (self.centers[side, -1] - self.centers[side, 0])
            intercept = self.density_mean[side, 0] - slope * self.centers[side, 0]
            predicted = intercept + slope * self.centers[side]
            if np.any(abs(predicted - self.density_mean[side]) > 512. * EPS * np.maximum(1., self.eos_scale[side])):
                raise UnsupportedAffineContract('non-affine density means or limiter branch unsupported')
            coefficients.append((intercept, slope))
        if abs(coefficients[0][1] - coefficients[1][1]) > bound(float(np.max(self.eos_scale))):
            raise UnsupportedAffineContract('common stable affine density slope required')
        self.slope = .5 * (coefficients[0][1] + coefficients[1][1])
        if self.slope > 0.:
            # A constant field may acquire a positive secant from EOS roundoff.
            # Project only if the original per-cell fit bounds also admit zero.
            self.slope = 0.
        for side in range(2):
            predicted = coefficients[side][0] + self.slope * self.centers[side]
            if np.any(abs(predicted - self.density_mean[side]) > 512. * EPS * np.maximum(1., self.eos_scale[side])):
                raise UnsupportedAffineContract('common stable affine density must bind every actual mean')
        self.anomaly = Polynomial([coefficients[0][0], (coefficients[1][0] - coefficients[0][0]) / self.distance])
        self.surface = Polynomial([self.state.eta[0], np.diff(self.state.eta)[0] / self.distance])
        external = self.profile.external_pressure_Pa
        self.external = Polynomial([external[0], np.diff(external)[0] / self.distance])
        low = self.eos.rho0 * (-self.eos.alpha * (45. - self.eos.Tref) + self.eos.beta * (0. - self.eos.Sref))
        high = self.eos.rho0 * (-self.eos.alpha * (-5. - self.eos.Tref) + self.eos.beta * (50. - self.eos.Sref))
        endpoint_density = np.array([[self.anomaly(x) + self.slope * z for z in (self.bottom, self.surface(x))] for x in (0., self.distance)])
        if np.any(endpoint_density <= low + bound(float(np.max(self.eos_scale)))) or np.any(endpoint_density >= high - bound(float(np.max(self.eos_scale)))):
            raise UnsupportedAffineContract('density endpoint clipping unsupported')
        with np.errstate(over='ignore', invalid='ignore'):
            velocities = self.state.stocks[:, :, 2:] / (self.eos.rho0 * self.state.h[:, :, None])
            squared = velocities**2
        if not np.isfinite(squared).all():
            raise ValueError('finite actual mass, velocity and squared speed required')
        self.segments = self._physical_strips()
        self._rows = tuple(self._load(segment) for segment in self.segments)
        self.force_N = np.zeros_like(self.state.h)
        self.force_scale_N = np.zeros_like(self.state.h)
        for row in self._rows:
            segment = row['segment']
            for side, layer in enumerate((segment.left_layer, segment.right_layer)):
                self.force_N[side, layer] += row['basis'][side]
                self.force_scale_N[side, layer] += row['basis_scale'][side]
        total(self.force_N)
        total(self.force_scale_N)
        self.force_N.setflags(write=False)
        self.force_scale_N.setflags(write=False)

    def _physical_strips(self):
        heights = self.state.eta - self.bottom
        normalized = (self.state.interfaces - self.bottom) / heights[:, None]
        cuts = np.unique(normalized)
        strips = []
        for lower, upper in zip(cuts[:-1], cuts[1:]):
            middle = .5 * (lower + upper)
            layers = [int(np.searchsorted(-normalized[side], -middle, side='right') - 1) for side in range(2)]
            top = self.state.eta if upper == 1. else self.bottom + upper * heights
            strips.append(PhysicalStrip(float(self.bottom + lower * heights[0]), float(top[0]),
                                        float(self.bottom + lower * heights[1]), float(top[1]),
                                        *layers, bool(upper == 1.)))
        return tuple(strips)

    def pressure(self, x, z):
        eta, anomaly = self.surface(x), self.anomaly(x)
        return float(self.external(x) + self.eos.gravity * ((self.eos.rho0 + anomaly) * (eta - z)
                                                           + .5 * self.slope * (eta**2 - z**2)))

    @staticmethod
    def _gauss(function, lower, upper):
        midpoint, half = .5 * (lower + upper), .5 * (upper - lower)
        offset = half / math.sqrt(3.)
        return half * total([function(midpoint - offset), function(midpoint + offset)])

    def _load(self, segment):
        d, L = self.distance, self.length
        def upper(x):
            return segment.upper_left + x / d * (segment.upper_right - segment.upper_left)

        def lower(x):
            return segment.lower_left + x / d * (segment.lower_right - segment.lower_left)
        top_slope = (segment.upper_right - segment.upper_left) / d
        bottom_slope = (segment.lower_right - segment.lower_left) / d
        sides = [L * self._gauss(lambda z: self.pressure(0., z), segment.lower_left, segment.upper_left),
                 -L * self._gauss(lambda z: self.pressure(d, z), segment.lower_right, segment.upper_right)]
        edges = dict(left=sides[0], right=sides[1],
                     top=L * top_slope * self._gauss(lambda x: self.pressure(x, upper(x)), 0., d),
                     bottom=-L * bottom_slope * self._gauss(lambda x: self.pressure(x, lower(x)), 0., d))
        volume = L / d * self._gauss(lambda x: self._gauss(lambda z: self.pressure(x, z), lower(x), upper(x)), 0., d)
        forces, scales = [], []
        for side in range(2):
            phi = (lambda x: 1. - x / d) if side == 0 else (lambda x: x / d)
            terms = [sides[side], L * top_slope * self._gauss(lambda x: phi(x) * self.pressure(x, upper(x)), 0., d),
                     -L * bottom_slope * self._gauss(lambda x: phi(x) * self.pressure(x, lower(x)), 0., d),
                     (-1. if side == 0 else 1.) * volume]
            forces.append(total(terms))
            scales.append(total(np.abs(terms)))
        return dict(segment=segment, edges=edges, force_N=total(list(edges.values())),
                    scale=total(np.abs(list(edges.values()))), basis=tuple(forces), basis_scale=tuple(scales))

    def contours(self):
        return tuple(dict(row, edges=dict(row['edges'])) for row in self._rows)

    def pressure_power(self, velocity):
        velocity = np.asarray(velocity, dtype=float)
        if velocity.shape != self.state.h.shape:
            raise ValueError('actual endpoint-layer velocity shape required')
        return total(velocity * self.force_N)

    def _energy_terms(self):
        eta, a, s, b = self.surface, self.anomaly, self.slope, self.bottom
        return [self.eos.rho0 * eta**2 / 2., a * (eta**2 - b**2) / 2., s * (eta**3 - b**3) / 3.]

    def physical_PE(self):
        return self.length * self.eos.gravity * total([integral(term, self.distance) for term in self._energy_terms()])

    def PE_scale(self):
        return self.length * self.eos.gravity * total([polynomial_scale(term, self.distance) for term in self._energy_terms()]) + total(self.node_area_m2[:, None] * abs(self.density_mean * self.state.h * self.centers)) * self.eos.gravity

    def node_PE(self):
        values = self.density_mean * self.state.h * self.centers + self.slope * self.state.h**3 / 12.
        return self.eos.gravity * total(self.node_area_m2 * (self.eos.rho0 * self.state.eta**2 / 2. + values.sum(axis=-1)))

    def stock_PE_direction(self, hdot, stockdot, etadot):
        hdot, stockdot, etadot = (np.asarray(value, dtype=float) for value in (hdot, stockdot, etadot))
        if hdot.shape != self.state.h.shape or stockdot.shape != self.state.stocks.shape or etadot.shape != (2,) or not all(np.isfinite(v).all() for v in (hdot, stockdot, etadot)):
            raise ValueError('finite shape-matched stock direction required')
        if np.any(hdot[:, 1:] != 0.) or not np.array_equal(hdot[:, 0], etadot):
            raise UnsupportedAffineContract('fixed physical interfaces required')
        center_dot = np.zeros_like(hdot)
        center_dot[:, 0] = .5 * etadot
        Jdot = self.eos.rho0 * (-self.eos.alpha * (stockdot[:, :, 0] - self.eos.Tref * hdot)
                                + self.eos.beta * (stockdot[:, :, 1] - self.eos.Sref * hdot))
        mean_dot = (Jdot - self.density_mean * hdot) / self.state.h
        field_dot = mean_dot - self.slope * center_dot
        coefficients, direction_scales = [], []
        for side in range(2):
            ds = (field_dot[side, -1] - field_dot[side, 0]) / (self.centers[side, -1] - self.centers[side, 0])
            da = field_dot[side, 0] - ds * self.centers[side, 0]
            scale = self.eos.rho0 * (self.eos.alpha * (abs(stockdot[side, :, 0]) + self.eos.Tref * abs(hdot[side]))
                                     + self.eos.beta * (abs(stockdot[side, :, 1]) + self.eos.Sref * abs(hdot[side]))) / self.state.h[side]
            if np.any(abs(field_dot[side] - da - ds * self.centers[side]) > 512. * EPS * np.maximum(1., scale)):
                raise UnsupportedAffineContract('non-affine stock direction unsupported')
            coefficients.append((da, ds))
            direction_scales.append(scale)
        if abs(coefficients[0][1] - coefficients[1][1]) > bound(total(abs(field_dot))):
            raise UnsupportedAffineContract('common affine slope direction required')
        ds = .5 * (coefficients[0][1] + coefficients[1][1])
        for side in range(2):
            if np.any(abs(field_dot[side] - coefficients[side][0] - ds * self.centers[side]) > 512. * EPS * np.maximum(1., direction_scales[side])):
                raise UnsupportedAffineContract('common affine slope direction must bind every actual stock rate')
        slope_terms = ds * self.state.h**3 / 12. + self.slope * self.state.h**2 * hdot / 4.
        terms = [self.node_area_m2 * self.eos.rho0 * self.state.eta * etadot,
                 self.node_area_m2[:, None] * Jdot * self.centers,
                 self.node_area_m2[:, None] * self.density_mean * self.state.h * center_dot,
                 self.node_area_m2[:, None] * slope_terms]
        return dict(value=self.eos.gravity * total([total(t) for t in terms]),
                    scale=self.eos.gravity * total([total(abs(t)) for t in terms]),
                    slope_value=self.eos.gravity * total(self.node_area_m2[:, None] * slope_terms),
                    da=Polynomial([coefficients[0][0], (coefficients[1][0] - coefficients[0][0]) / self.distance]), ds=ds)

    def affine_rates(self, U, alpha):
        U, alpha = float(U), float(alpha)
        if not math.isfinite(U) or not math.isfinite(alpha):
            raise ValueError('finite manufactured velocity required')
        T = self.profile.temperature_mean
        if np.any(abs(T - self.eos.Tref) > 512. * EPS * np.maximum(1., abs(T) + abs(self.eos.Tref))):
            raise UnsupportedAffineContract('constant T affine S transport family required')
        eta, a, s, b = self.surface, self.anomaly, self.slope, self.bottom
        velocity = Polynomial([U, alpha])
        actual = self.state.stocks[:, :, 2:] / (self.eos.rho0 * self.state.h[:, :, None])
        expected_u = np.array([U, U + alpha * self.distance])[:, None]
        V = float(actual[0, 0, 1])
        if np.any(abs(actual[:, :, 0] - expected_u) > 512. * EPS * np.maximum(1., abs(actual[:, :, 0]) + abs(expected_u))) or np.any(abs(actual[:, :, 1] - V) > 512. * EPS * np.maximum(1., abs(actual[:, :, 1]) + abs(V))):
            raise UnsupportedAffineContract('actual momentum must match affine u and constant v family')
        eta_dot = -alpha * (eta - b) - velocity * eta.deriv()
        da = -velocity * a.deriv() - alpha * s * b
        ds = alpha * s
        etadot = np.array([eta_dot(x) for x in (0., self.distance)])
        hdot = np.zeros_like(self.state.h)
        hdot[:, 0] = etadot
        stockdot = np.zeros_like(self.state.stocks)
        for side, x in enumerate((0., self.distance)):
            center_dot = np.zeros(self.state.h.shape[1])
            center_dot[0] = .5 * etadot[side]
            Jdot = self.state.h[side] * (da(x) + ds * self.centers[side]) + (a(x) + s * self.centers[side]) * hdot[side] + s * self.state.h[side] * center_dot
            stockdot[side, :, 0] = self.eos.Tref * hdot[side]
            stockdot[side, :, 1] = self.eos.Sref * hdot[side] + Jdot / (self.eos.rho0 * self.eos.beta)
            stockdot[side, :, 2] = self.eos.rho0 * (velocity(x) * hdot[side] - alpha * velocity(x) * self.state.h[side])
            stockdot[side, :, 3] = V * self.eos.rho0 * hdot[side]
        node = self.stock_PE_direction(hdot, stockdot, etadot)
        physical_terms = [(self.eos.rho0 + a + s * eta) * eta * eta_dot,
                          node['da'] * (eta**2 - b**2) / 2., node['ds'] * (eta**3 - b**3) / 3.]
        physical = self.length * self.eos.gravity * total([integral(term, self.distance) for term in physical_terms])
        physical_scale = self.length * self.eos.gravity * total([polynomial_scale(term, self.distance) for term in physical_terms])
        velocities = np.broadcast_to([U, U + alpha * self.distance], self.state.h.T.shape).T
        R = alpha * (self.state.interfaces - b)
        R[:, [0, -1]] = 0.
        rate = dict(hdot=hdot, stockdot=stockdot, etadot=etadot, relative_downward=R,
                    node_PE_dot_W=node['value'], node_PE_scale=node['scale'], slope_PE_dot_W=node['slope_value'],
                    physical_PE_dot_W=physical, PE_scale=physical_scale, pressure_power_W=self.pressure_power(velocities),
                    pressure_power_scale=total(abs(velocities * self.force_N)),
                    endpoint_pairing_passed=(self.state.eta[0] == self.state.eta[1] and etadot[0] == etadot[1]
                                             and abs(self.node_PE() - self.physical_PE()) <= bound(self.PE_scale())
                                             and abs(node['value'] - physical) <= bound(node['scale'] + physical_scale)),
                    production_force_consumption_qualified=False, da=node['da'], ds=node['ds'], eta_dot_poly=eta_dot,
                    velocity_poly=velocity, velocity_dot_poly=-alpha * velocity, velocity_y=V)
        for name, value in rate.items():
            if isinstance(value, (float, np.ndarray)) and not np.isfinite(value).all():
                raise ValueError('finite rate and ledger required: ' + name)
        return rate

    def require_endpoint_pairing(self, U, alpha):
        rate = self.affine_rates(U, alpha)
        if not rate['endpoint_pairing_passed']:
            raise UnsupportedAffineContract('endpoint inventory PE direction differs from physical B/Bdot')
        return rate

    def physical_stock_direction(self, rate):
        eta, a, b, s = self.surface, self.anomaly, self.bottom, self.slope
        edot, da, ds = rate['eta_dot_poly'], rate['da'], rate['ds']
        water = self.length * integral(edot, self.distance)
        anomaly = self.length * integral(da * (eta - b) + ds * (eta**2 - b**2) / 2. + (a + s * eta) * edot, self.distance)
        momentum = self.eos.rho0 * self.length * integral(rate['velocity_dot_poly'] * (eta - b) + rate['velocity_poly'] * edot, self.distance)
        return np.array([self.eos.Tref * water, self.eos.Sref * water + anomaly / (self.eos.rho0 * self.eos.beta), momentum, rate['velocity_y'] * self.eos.rho0 * water])

    def validate_perturbation(self, rate, epsilon):
        hdot, etadot = np.asarray(rate['hdot']), np.asarray(rate['etadot'])
        if hdot.shape != self.state.h.shape or etadot.shape != (2,) or not all(np.isfinite(v).all() for v in (hdot, etadot)):
            raise ValueError('finite shape-matched geometry direction required')
        if not math.isfinite(epsilon) or epsilon <= 0.:
            raise ValueError('finite positive perturbation required')
        with np.errstate(over='ignore', invalid='ignore'):
            future_h = self.state.h + epsilon * hdot
            future_top = self.state.eta + epsilon * etadot
        if not all(np.isfinite(v).all() for v in (future_h, future_top)):
            raise ValueError('finite future physical geometry required')
        if np.any(future_h <= 0.):
            raise UnsupportedAffineContract('physical interface crossing refused')
        for strip in self.segments:
            if strip.actual_top:
                if future_top[0] <= strip.lower_left or future_top[1] <= strip.lower_right:
                    raise UnsupportedAffineContract('frozen virtual cut crossing refused')

    def fixed_mass_impulse(self):
        before = self.state.stocks[:, :, 2]
        impulse = self.force_N / self.node_area_m2[:, None]
        after = before + impulse
        mass = self.eos.rho0 * self.state.h
        midpoint = (before + after) / (2. * mass)
        change = total(self.node_area_m2[:, None] * (after - before) * (after + before) / (2. * mass))
        scale = total(self.node_area_m2[:, None] * (before**2 + after**2) / (2. * mass))
        return dict(kinetic_change_J=change, scale=scale, midpoint_u=midpoint,
                    nodal_lumped_mass_metric=True, accepted_state_returned=False)
