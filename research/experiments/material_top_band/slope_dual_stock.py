"""Auxiliary weighted physical inventory projection on a frozen affine chart.

Raw ColumnStocks and their endpoint-prism PE are never replaced or advanced.
"""
import math
from dataclasses import dataclass, replace

import numpy as np

from .affine_physical import EPS, AffinePhysicalDual, bound, total

POINTS = (-math.sqrt(3. / 5.), 0., math.sqrt(3. / 5.))
WEIGHTS = (5. / 9., 8. / 9., 5. / 9.)


def readonly(values):
    values = np.array(values, dtype=float, copy=True)
    total(values)
    values.setflags(write=False)
    return values


def gauss(function, lower, upper):
    half, center = .5 * (upper - lower), .5 * (upper + lower)
    terms = [half * weight * np.asarray(function(center + half * point), dtype=float)
             for point, weight in zip(POINTS, WEIGHTS)]
    values = np.array([total([term.flat[j] for term in terms]) for j in range(terms[0].size)])
    scale = np.array([total([abs(term.flat[j]) for term in terms]) for j in range(terms[0].size)])
    return values, scale


def heights(strip, x, distance):
    fraction = x / distance
    return (strip.lower_left + fraction * (strip.lower_right - strip.lower_left),
            strip.upper_left + fraction * (strip.upper_right - strip.upper_left))


def volume(operator, strip, side, function):
    def horizontal(x):
        lower, upper = heights(strip, x, operator.distance)
        weight = 1. - x / operator.distance if side == 0 else x / operator.distance
        values, scale = gauss(lambda z: function(x, z), lower, upper)
        return np.r_[weight * values, abs(weight) * scale]
    combined, rounding = gauss(horizontal, 0., operator.distance)
    count = combined.size // 2
    return operator.length * combined[:count], operator.length * (combined[count:] + rounding[:count])


def gram(operator, strips, *, top_direction=None):
    mass = np.zeros((28, 28))
    for strip in strips:
        if top_direction is None:
            left = strip.upper_left - strip.lower_left
            right = strip.upper_right - strip.lower_right
        elif strip.actual_top:
            left, right = top_direction
        else:
            continue
        block = operator.eos.rho0 * operator.length * operator.distance / 12. * np.array(
            [[3. * left + right, left + right], [left + right, left + 3. * right]])
        indices = [strip.left_layer, 14 + strip.right_layer]
        mass[np.ix_(indices, indices)] += block
    return readonly(mass)


@dataclass(frozen=True)
class PhysicalProjection:
    values: np.ndarray
    scale: np.ndarray
    mass: np.ndarray
    velocity: np.ndarray
    free_PE: float
    PE: float
    PE_scale: float
    raw_inventory: np.ndarray
    raw_free_PE: float
    raw_PE: float
    raw_conservative_remap: bool = False
    production_qualified: bool = False


@dataclass(frozen=True)
class PhysicalDirection:
    values: np.ndarray
    scale: np.ndarray
    mass_dot: np.ndarray
    velocity_dot: np.ndarray
    PE_dot: float
    PE_scale: float
    raw_inventory_dot: np.ndarray
    raw_PE_dot: float
    rows: tuple


class FrozenPhysicalProjection:
    """A diagnostic physical projection; its chart and raw inventory are copied."""

    def __init__(self, profile, *, distance_m, length_m):
        self.base = AffinePhysicalDual(profile, distance_m=distance_m, length_m=length_m)
        self.profile = self.base.profile
        self.chart = self.base.segments
        self._bind_temperature(self.base)

    @staticmethod
    def _bind_temperature(operator):
        T, reference = operator.profile.temperature_mean, operator.eos.Tref
        if np.any(abs(T - reference) > 512. * EPS * np.maximum(1., abs(T) + abs(reference))):
            raise ValueError('physical projection requires every actual T to equal Tref')

    def _current(self, profile):
        if profile is None:
            return self.base, self.chart
        state, original = profile.state, self.profile.state
        if state.interfaces.shape != original.interfaces.shape or not np.array_equal(state.interfaces[:, 1:], original.interfaces[:, 1:]) or not np.array_equal(state.bottom, original.bottom):
            raise ValueError('frozen raw interior interfaces and bottom required')
        operator = AffinePhysicalDual(profile, distance_m=self.base.distance, length_m=self.base.length)
        self._bind_temperature(operator)
        strips = []
        for strip in self.chart:
            if strip.actual_top:
                if operator.state.eta[0] <= strip.lower_left or operator.state.eta[1] <= strip.lower_right:
                    raise ValueError('frozen virtual cut crossing refused')
                strip = replace(strip, upper_left=float(operator.state.eta[0]), upper_right=float(operator.state.eta[1]))
            strips.append(strip)
        return operator, tuple(strips)

    @staticmethod
    def _field(operator, strip, x, z):
        fraction = x / operator.distance
        density = operator.anomaly(x) + operator.slope * z
        q = operator.state.stocks[:, :, 2:] / operator.state.h[:, :, None]
        momentum = (1. - fraction) * q[0, strip.left_layer] + fraction * q[1, strip.right_layer]
        return np.r_[1., operator.eos.Tref, operator.eos.Sref + density / (operator.eos.rho0 * operator.eos.beta), momentum, z * density]

    @staticmethod
    def _density_operation_scale(operator, x, z):
        density = operator.anomaly(x) + operator.slope * z
        S = operator.eos.Sref + density / (operator.eos.rho0 * operator.eos.beta)
        return operator.eos.rho0 * (operator.eos.alpha * (2. * abs(operator.eos.Tref))
                                    + operator.eos.beta * (abs(S) + abs(operator.eos.Sref)))

    def evaluate(self, profile=None):
        operator, strips = self._current(profile)
        values, scale = np.zeros((2, 14, 6)), np.zeros((2, 14, 6))
        for strip in strips:
            for side, layer in enumerate((strip.left_layer, strip.right_layer)):
                field, field_scale = volume(operator, strip, side, lambda x, z: self._field(operator, strip, x, z))
                density_scale, _ = volume(operator, strip, side, lambda x, z: [abs(z) * self._density_operation_scale(operator, x, z)])
                field_scale[5] += density_scale[0]
                values[side, layer] += field
                scale[side, layer] += field_scale
        free, free_scale = gauss(lambda x: [operator.surface(x)**2 / 2.], 0., operator.distance)
        free_PE = operator.eos.rho0 * operator.eos.gravity * operator.length * free[0]
        PE = free_PE + operator.eos.gravity * total(values[:, :, 5])
        PE_scale = operator.eos.rho0 * operator.eos.gravity * operator.length * free_scale[0] + operator.eos.gravity * total(scale[:, :, 5])
        raw_fields = np.concatenate([operator.state.h[:, :, None], operator.state.stocks], axis=-1)
        raw = np.array([total(operator.node_area_m2[:, None] * raw_fields[:, :, slot]) for slot in range(5)])
        raw_free = operator.eos.rho0 * operator.eos.gravity * total(operator.node_area_m2 * operator.state.eta**2 / 2.)
        velocity = operator.state.stocks[:, :, 2:].reshape(28, 2) / (operator.eos.rho0 * operator.state.h.reshape(28, 1))
        total([PE, PE_scale, raw_free])
        return PhysicalProjection(readonly(values), readonly(scale), gram(operator, strips), readonly(velocity),
                                  free_PE, PE, PE_scale, readonly(raw), raw_free, operator.node_PE())

    def direction(self, U, strain):
        operator = self.base
        rate = operator.affine_rates(U, strain)  # recompute from actual authority
        q = operator.state.stocks[:, :, 2:] / operator.state.h[:, :, None]
        qdot = (rate['stockdot'][:, :, 2:] - q * rate['hdot'][:, :, None]) / operator.state.h[:, :, None]
        qdot_scale = (abs(rate['stockdot'][:, :, 2:]) + abs(q * rate['hdot'][:, :, None])) / operator.state.h[:, :, None]
        hdot, stockdot, eos = rate['hdot'], rate['stockdot'], operator.eos
        Jdot_scale = eos.rho0 * (eos.alpha * (abs(stockdot[:, :, 0]) + abs(eos.Tref * hdot))
                                + eos.beta * (abs(stockdot[:, :, 1]) + abs(eos.Sref * hdot)))
        field_dot_scale = (Jdot_scale + abs(operator.density_mean * hdot)) / operator.state.h
        field_dot_scale[:, 0] += abs(.5 * operator.slope * rate['etadot'])
        fit_slope_scale = (field_dot_scale[:, -1] + field_dot_scale[:, 0]) / abs(operator.centers[:, -1] - operator.centers[:, 0])
        fit_intercept_scale = field_dot_scale[:, 0] + fit_slope_scale * abs(operator.centers[:, 0])
        common_slope_scale = .5 * total(fit_slope_scale)

        def density_dot_scale(x, z):
            # Include the actual intercept subtraction and polynomial evaluation.
            return (fit_intercept_scale[0] + x / operator.distance * total(fit_intercept_scale)
                    + common_slope_scale * abs(z))

        def eta_dot_scale(x):
            return total([abs(coefficient) * abs(x)**degree
                          for degree, coefficient in enumerate(rate['eta_dot_poly'].coef)])
        values, scale, rows = np.zeros((2, 14, 6)), np.zeros((2, 14, 6)), []
        for strip in self.chart:
            def field_dot(x, z):
                fraction = x / operator.distance
                density_dot = rate['da'](x) + rate['ds'] * z
                momentum_dot = (1. - fraction) * qdot[0, strip.left_layer] + fraction * qdot[1, strip.right_layer]
                return np.r_[0., 0., density_dot / (operator.eos.rho0 * operator.eos.beta), momentum_dot, z * density_dot]
            for side, layer in enumerate((strip.left_layer, strip.right_layer)):
                interior, interior_scale = volume(operator, strip, side, field_dot)

                def field_operations(x, z):
                    fraction = x / operator.distance
                    density = density_dot_scale(x, z)
                    momentum = (1. - fraction) * qdot_scale[0, strip.left_layer] + fraction * qdot_scale[1, strip.right_layer]
                    return np.r_[0., 0., density / (eos.rho0 * eos.beta), momentum, abs(z) * density]

                input_scale, _ = volume(operator, strip, side, field_operations)
                interior_scale += input_scale
                shape, shape_scale = np.zeros(6), np.zeros(6)
                if strip.actual_top:
                    def top(x):
                        weight = 1. - x / operator.distance if side == 0 else x / operator.distance
                        z = heights(strip, x, operator.distance)[1]
                        return weight * self._field(operator, strip, x, z) * rate['eta_dot_poly'](x)
                    shape, shape_scale = gauss(top, 0., operator.distance)
                    shape *= operator.length
                    shape_scale *= operator.length

                    def top_operations(x):
                        weight = 1. - x / operator.distance if side == 0 else x / operator.distance
                        z = heights(strip, x, operator.distance)[1]
                        field = abs(self._field(operator, strip, x, z))
                        density = self._density_operation_scale(operator, x, z)
                        field[2] += density / (eos.rho0 * eos.beta)
                        field[5] += abs(z) * density
                        return weight * field * eta_dot_scale(x)

                    operations, _ = gauss(top_operations, 0., operator.distance)
                    shape_scale += operator.length * operations
                row = interior + shape
                row_scale = interior_scale + shape_scale
                values[side, layer] += row
                scale[side, layer] += row_scale
                rows.append(dict(strip=strip, side=side, layer=layer, value=readonly(row),
                                 volume=readonly(interior), shape=readonly(shape), scale=readonly(row_scale)))
        free, free_scale = gauss(lambda x: [operator.surface(x) * rate['eta_dot_poly'](x)], 0., operator.distance)
        PE_dot = operator.eos.gravity * (operator.eos.rho0 * operator.length * free[0] + total(values[:, :, 5]))
        PE_scale = operator.eos.gravity * (operator.eos.rho0 * operator.length * free_scale[0] + total(scale[:, :, 5]))
        raw_fields = np.concatenate([rate['hdot'][:, :, None], rate['stockdot']], axis=-1)
        raw = np.array([total(operator.node_area_m2[:, None] * raw_fields[:, :, slot]) for slot in range(5)])
        total([PE_dot, PE_scale])
        return PhysicalDirection(readonly(values), readonly(scale), gram(operator, self.chart, top_direction=rate['etadot']),
                                 readonly(qdot.reshape(28, 2) / operator.eos.rho0), PE_dot, PE_scale,
                                 readonly(raw), rate['node_PE_dot_W'], tuple(rows))

    def fixed_mass_impulse(self, duration):
        if isinstance(duration, (bool, np.bool_)) or not math.isfinite(duration) or duration <= 0.:
            raise ValueError('finite positive auxiliary impulse duration required')
        mapped = self.evaluate()
        mass, before = mapped.mass, mapped.velocity[:, 0]
        eigenvalues = np.linalg.eigvalsh(mass)
        if not np.isfinite(eigenvalues).all() or eigenvalues[0] <= 0.:
            raise ValueError('resolved positive physical Gram matrix required')
        momentum_before = mapped.values[:, :, 3].ravel()
        binding_scale = abs(mass) @ abs(before) + abs(momentum_before)
        if np.any(abs(mass @ before - momentum_before) > 512. * EPS * np.maximum(1., binding_scale)):
            raise ValueError('actual projected momentum and physical mass disagree')
        force = self.base.force_N.ravel()
        momentum_after = momentum_before + duration * force
        after = np.linalg.solve(mass, momentum_after)
        solve_scale = abs(mass) @ abs(after) + abs(momentum_after)
        if not np.isfinite(after).all() or np.any(abs(mass @ after - momentum_after) > 512. * EPS * np.maximum(1., solve_scale)):
            raise ValueError('physical momentum solve backward error refused')
        midpoint = .5 * (before + after)
        change = .5 * total((after - before) * (mass @ (after + before)))
        work = duration * total(force * midpoint)
        scale = (.5 * total(abs(before) * (abs(mass) @ abs(before)))
                 + .5 * total(abs(after) * (abs(mass) @ abs(after)))
                 + duration * total(self.base.force_scale_N.ravel() * abs(midpoint)))
        if abs(change - work) > bound(scale):
            raise ValueError('fixed physical mass midpoint work failed')
        return dict(kinetic_change=change, pressure_work=work, scale=scale,
                    velocity_after=readonly(after), momentum_after=readonly(momentum_after),
                    solve_residual=float(np.max(abs(mass @ after - momentum_after))),
                    minimum_mass_eigenvalue=float(eigenvalues[0]), duration=duration,
                    accepted_state_returned=False, production_qualified=False)
