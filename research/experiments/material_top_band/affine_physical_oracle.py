"""Independent scalar physical quadrature and raw-stock finite differences.

No candidate pressure, PE, contour, C/transpose or reconstruction is called.
Geometry is declared input. This is not a general stock-PE gradient oracle.
"""
import math
from dataclasses import dataclass

import numpy as np

NODES, WEIGHTS = np.polynomial.legendre.leggauss(7)
RHO0, GRAVITY, ALPHA, BETA, TREF, SREF = 1025., 9.81, 2e-4, 7.6e-4, 15., 35.
EPS = np.finfo(float).eps


@dataclass(frozen=True)
class Specification:
    eta: tuple = (-.2, -.2)
    external: tuple = (80., 92.)
    density_gradient_x: float = .15
    density_intercept: float = 1.
    density_slope: float = -.4
    bottom: float = -3.
    distance: float = 2.
    length: float = 1.5
    velocity_y: float = -.02


def gauss(function, lower, upper):
    half, middle = .5 * (upper - lower), .5 * (upper + lower)
    return half * math.fsum(float(w) * function(middle + half * float(t)) for t, w in zip(NODES, WEIGHTS))


def eta(spec, x):
    return spec.eta[0] + x * (spec.eta[1] - spec.eta[0]) / spec.distance


def anomaly(spec, x, z):
    return spec.density_intercept + spec.density_gradient_x * x + spec.density_slope * z


def external(spec, x):
    return spec.external[0] + x * (spec.external[1] - spec.external[0]) / spec.distance


def pressure(spec, x, z):
    # Integrate the physical density rather than reuse the candidate primitive.
    return external(spec, x) + GRAVITY * gauss(lambda depth: RHO0 + anomaly(spec, x, depth), z, eta(spec, x))


def pressure_gradient(spec, x, z):
    surface_slope = (spec.eta[1] - spec.eta[0]) / spec.distance
    return ((spec.external[1] - spec.external[0]) / spec.distance
            + GRAVITY * ((RHO0 + anomaly(spec, x, eta(spec, x))) * surface_slope
                         + spec.density_gradient_x * (eta(spec, x) - z)))


def strip_edges(spec, strip, x):
    fraction = x / spec.distance
    return (strip.lower_left + fraction * (strip.lower_right - strip.lower_left),
            strip.upper_left + fraction * (strip.upper_right - strip.upper_left))


def validate_strip_ownership(spec, state, strips):
    """Resolve owners directly from raw interfaces and close each raw width."""
    coverage = np.zeros_like(state.h)
    for strip in strips:
        for side in range(2):
            lower = (strip.lower_left, strip.lower_right)[side]
            upper = (strip.upper_left, strip.upper_right)[side]
            middle = .5 * (lower + upper)
            z = state.interfaces[side]
            owners = np.flatnonzero((z[1:] < middle) & (middle < z[:-1]))
            declared = (strip.left_layer, strip.right_layer)[side]
            if len(owners) != 1 or int(owners[0]) != declared or upper <= lower:
                raise ValueError('raw physical interface ownership mismatch')
            scale = max(1., abs(lower), abs(upper), abs(z[declared]), abs(z[declared + 1]))
            tolerance = 512. * EPS * scale
            if lower < z[declared + 1] - tolerance or upper > z[declared] + tolerance:
                raise ValueError('strip extends beyond raw physical ownership')
            coverage[side, declared] += upper - lower
        expected_top = strip.upper_left == state.eta[0] and strip.upper_right == state.eta[1]
        if strip.actual_top != expected_top:
            raise ValueError('raw actual-top ownership mismatch')
    if np.any(abs(coverage - state.h) > 512. * EPS * np.maximum(1., abs(state.h))):
        raise ValueError('raw physical ownership coverage does not close')
    if np.any(state.bottom != spec.bottom) or not np.array_equal(state.eta, spec.eta):
        raise ValueError('raw ownership domain differs from specification')
    return True


def volume(spec, strip, function):
    return spec.length * gauss(lambda x: gauss(lambda z: function(x, z), *strip_edges(spec, strip, x)), 0., spec.distance)


def segment_contour(spec, strip):
    d, L = spec.distance, spec.length
    components = [L * gauss(lambda z: pressure(spec, 0., z), strip.lower_left, strip.upper_left),
                  -L * gauss(lambda z: pressure(spec, d, z), strip.lower_right, strip.upper_right),
                  L * (strip.upper_right - strip.upper_left) / d * gauss(lambda x: pressure(spec, x, strip_edges(spec, strip, x)[1]), 0., d),
                  -L * (strip.lower_right - strip.lower_left) / d * gauss(lambda x: pressure(spec, x, strip_edges(spec, strip, x)[0]), 0., d)]
    direct = -volume(spec, strip, lambda x, z: pressure_gradient(spec, x, z))
    contour = math.fsum(components)
    scale = math.fsum(abs(v) for v in components) + abs(direct)
    if abs(contour - direct) > 512. * EPS * max(1., scale):
        raise ValueError('independent physical contour and volume disagree')
    return dict(value=direct, scale=scale, contour=contour)


def basis_force(spec, strips, side, layer):
    terms = []
    for strip in strips:
        if (strip.left_layer, strip.right_layer)[side] == layer:
            terms.append(-volume(spec, strip, lambda x, z: (1. - x / spec.distance if side == 0 else x / spec.distance) * pressure_gradient(spec, x, z)))
    return dict(value=math.fsum(terms), scale=math.fsum(abs(v) for v in terms))


def pressure_power(spec, strips, velocity):
    terms = []
    for strip in strips:
        left, right = velocity[0, strip.left_layer], velocity[1, strip.right_layer]
        terms.append(-volume(spec, strip, lambda x, z: (left + x / spec.distance * (right - left)) * pressure_gradient(spec, x, z)))
    return dict(value=math.fsum(terms), scale=math.fsum(abs(v) for v in terms))


def geometric_load_variation(spec, strips, side, layer, epsilon):
    if not math.isfinite(epsilon) or epsilon <= 0.:
        raise ValueError('finite positive geometric perturbation required')
    positive, negative, compression, remainder = [], [], [], []
    derivative = (-1. if side == 0 else 1.) / spec.distance
    k = (spec.eta[1] - spec.eta[0]) / spec.distance
    second_pressure = GRAVITY * (2. * spec.density_gradient_x * k + spec.density_slope * k**2)
    for strip in strips:
        if (strip.left_layer, strip.right_layer)[side] != layer:
            continue
        phi = (lambda x: 1. - x / spec.distance) if side == 0 else (lambda x: x / spec.distance)
        positive.append(volume(spec, strip, lambda x, z: pressure(spec, x + epsilon * phi(x), z) * (1. + epsilon * derivative)))
        negative.append(volume(spec, strip, lambda x, z: pressure(spec, x - epsilon * phi(x), z) * (1. - epsilon * derivative)))
        compression.append(volume(spec, strip, lambda x, z: pressure(spec, x, z) * derivative))
        remainder.append(.5 * volume(spec, strip, lambda x, z: abs(second_pressure * phi(x)**2 * derivative)))
    value = -(math.fsum(positive) - math.fsum(negative)) / (2. * epsilon) + math.fsum(compression)
    rounding = 512. * EPS * max(1., math.fsum(abs(v) for v in positive + negative)) / (2. * epsilon)
    rounding += 512. * EPS * max(1., math.fsum(abs(v) for v in compression))
    return dict(value=value, bound=epsilon**2 * math.fsum(remainder) + rounding)


def flow(spec, x, z, U, strain):
    return U + strain * x, -strain * (z - spec.bottom)


def surface_direction(spec, x, U, strain):
    u, w = flow(spec, x, eta(spec, x), U, strain)
    return w - u * (spec.eta[1] - spec.eta[0]) / spec.distance


def density_direction(spec, x, z, U, strain):
    u, w = flow(spec, x, z, U, strain)
    return -u * spec.density_gradient_x - w * spec.density_slope


def specific(spec, x, z, U, strain):
    return np.array([TREF, SREF + anomaly(spec, x, z) / (RHO0 * BETA),
                     RHO0 * (U + strain * x), spec.velocity_y * RHO0])


def layer_stock_direction(spec, state, side, layer, U, strain):
    x = side * spec.distance
    lower, upper = float(state.interfaces[side, layer + 1]), float(state.interfaces[side, layer])
    edot = surface_direction(spec, x, U, strain) if layer == 0 else 0.
    anomaly_rate = gauss(lambda z: density_direction(spec, x, z, U, strain), lower, upper) + anomaly(spec, x, upper) * edot
    actual = state.stocks[:, :, 2:] / (RHO0 * state.h[:, :, None])
    momentum_gradient = RHO0 * (actual[1, 0, 0] - actual[0, 0, 0]) / spec.distance
    return np.array([TREF * edot, SREF * edot + anomaly_rate / (RHO0 * BETA),
                     -(U + strain * x) * momentum_gradient * (upper - lower) + RHO0 * actual[side, layer, 0] * edot,
                     RHO0 * actual[side, layer, 1] * edot])


def ale_fluxes(spec, strips, U, strain):
    rows, local = [], []
    for strip in strips:
        edges = np.zeros((4, 4))
        for slot in range(4):
            edges[0, slot] = -spec.length * gauss(lambda z: specific(spec, 0., z, U, strain)[slot] * U, strip.lower_left, strip.upper_left)
            edges[1, slot] = spec.length * gauss(lambda z: specific(spec, spec.distance, z, U, strain)[slot] * (U + strain * spec.distance), strip.lower_right, strip.upper_right)
            for index, upper in ((2, True), (3, False)):
                slope = ((strip.upper_right - strip.upper_left) if upper else (strip.lower_right - strip.lower_left)) / spec.distance
                def cut_flux(x):
                    z = strip_edges(spec, strip, x)[1 if upper else 0]
                    u, w = flow(spec, x, z, U, strain)
                    grid = surface_direction(spec, x, U, strain) if upper and strip.actual_top else 0.
                    normal = w - u * slope - grid
                    return specific(spec, x, z, U, strain)[slot] * normal * (1. if upper else -1.)
                edges[index, slot] = spec.length * gauss(cut_flux, 0., spec.distance)
        rows.append(edges)
        content_dot, content_scale = [], []
        for slot in range(4):
            def specific_dot(x, z):
                return (0., density_direction(spec, x, z, U, strain) / (RHO0 * BETA),
                        -RHO0 * strain * (U + strain * x), 0.)[slot]
            volume_dot = volume(spec, strip, specific_dot)
            volume_scale = volume(spec, strip, lambda x, z: abs(specific_dot(x, z)))
            def top_dot(x):
                z = strip_edges(spec, strip, x)[1]
                return specific(spec, x, z, U, strain)[slot] * surface_direction(spec, x, U, strain)
            shape_dot = spec.length * gauss(top_dot, 0., spec.distance) if strip.actual_top else 0.
            shape_scale = spec.length * gauss(lambda x: abs(top_dot(x)), 0., spec.distance) if strip.actual_top else 0.
            content_dot.append(volume_dot + shape_dot)
            content_scale.append(volume_scale + shape_scale + np.abs(edges[:, slot]).sum())
        local.append(dict(content_dot=np.array(content_dot), flux=edges.sum(axis=0), scale=np.array(content_scale)))
    rows = np.asarray(rows)
    pairs = rows[:-1, 2] + rows[1:, 3]
    outer = rows[:, :2].sum(axis=(0, 1)) + rows[0, 3] + rows[-1, 2]
    # Remove an actual active internal top flux from each independently closed row.
    omitted = np.array([row['content_dot'] + row['flux'] - rows[i, 2] for i, row in enumerate(local[:-1])])
    index, slot = np.unravel_index(np.argmax(abs(omitted)), omitted.shape)
    return dict(outer=outer, scale=np.abs(rows).sum(axis=(0, 1)), internal=rows[:-1, 2],
                internal_pair_residual=pairs, local=tuple(local),
                omitted_local_residual=float(abs(omitted[index, slot])),
                omitted_local_bound=512. * EPS * max(1., local[index]['scale'][slot]))


def energy_direction(spec, U, strain):
    volume_term = spec.length * GRAVITY * gauss(lambda x: gauss(lambda z: z * density_direction(spec, x, z, U, strain), spec.bottom, eta(spec, x)), 0., spec.distance)
    shape = spec.length * GRAVITY * gauss(lambda x: eta(spec, x) * (RHO0 + anomaly(spec, x, eta(spec, x))) * surface_direction(spec, x, U, strain), 0., spec.distance)
    boundary = []
    for x, normal in ((0., -1.), (spec.distance, 1.)):
        boundary.append(-normal * spec.length * gauss(lambda z: (pressure(spec, x, z) + GRAVITY * z * (RHO0 + anomaly(spec, x, z))) * (U + strain * x), spec.bottom, eta(spec, x)))
    boundary.append(-spec.length * gauss(lambda x: external(spec, x) * surface_direction(spec, x, U, strain), 0., spec.distance))
    return dict(PE_dot=volume_term + shape, boundary_power=math.fsum(boundary),
                scale=abs(volume_term) + abs(shape) + math.fsum(abs(v) for v in boundary))


def endpoint_PE_discrepancy(spec, U, strain):
    e, k = spec.eta[0], (spec.eta[1] - spec.eta[0]) / spec.distance
    R, a, s, d = RHO0 + spec.density_intercept, spec.density_gradient_x, spec.density_slope, spec.distance
    de = -strain * (e - spec.bottom) - U * k
    dk, dR, da, ds = -2. * strain * k, -U * a - strain * s * spec.bottom, -strain * a, strain * s
    energy = d**3 / 6. * ((R / 2. + s * e) * k**2 + a * e * k) + d**4 / 4. * (a * k**2 / 2. + s * k**3 / 3.)
    direction = d**3 / 6. * ((dR / 2. + ds * e + s * de) * k**2 + 2. * (R / 2. + s * e) * k * dk + da * e * k + a * de * k + a * e * dk)
    direction += d**4 / 4. * (da * k**2 / 2. + a * k * dk + ds * k**3 / 3. + s * k**2 * dk)
    return dict(energy=GRAVITY * spec.length * energy, direction=GRAVITY * spec.length * direction)


def nodal_minus_volume_KE(spec, U, strain):
    if spec.eta[0] != spec.eta[1]:
        raise ValueError('flat-surface KE representation control required')
    return RHO0 * spec.length * spec.distance * (spec.eta[0] - spec.bottom) * (strain * spec.distance)**2 / 12.


def _density_stock(stocks, h):
    return RHO0 * (-ALPHA * (stocks[:, :, 0] - TREF * h) + BETA * (stocks[:, :, 1] - SREF * h))


def _scalar_profiles(stocks, h, z):
    means = _density_stock(stocks, h) / h
    centers = .5 * (z[:, :-1] + z[:, 1:])
    slopes = np.zeros_like(means)
    low, high = RHO0 * (-ALPHA * 30. - BETA * 35.), RHO0 * (ALPHA * 20. + BETA * 15.)
    for side in range(2):
        secants = [(float(means[side, k + 1]) - float(means[side, k])) / float(centers[side, k + 1] - centers[side, k]) for k in range(h.shape[1] - 1)]
        raw = [secants[0]] + [math.copysign(min(abs(a), abs(b)), a) if a * b > 0. else 0. for a, b in zip(secants[:-1], secants[1:])] + [secants[-1]]
        for k, value in enumerate(raw):
            room = min(float(means[side, k]) - low, high - float(means[side, k]))
            if room <= .5 * float(h[side, k]) * abs(value):
                raise ValueError('finite-difference endpoint limiter inactive required')
            slopes[side, k] = value
    return means, centers, slopes


def _stock_energy(stocks, h, z, area):
    means, centers, slopes = _scalar_profiles(stocks, h, z)
    terms = [GRAVITY * area[side] * RHO0 * z[side, 0]**2 / 2. for side in range(2)]
    for side in range(2):
        for k in range(h.shape[1]):
            terms.append(GRAVITY * area[side] * gauss(lambda depth: depth * (means[side, k] + slopes[side, k] * (depth - centers[side, k])), z[side, k + 1], z[side, k]))
    return math.fsum(terms), math.fsum(abs(v) for v in terms)


def _curvature_majorant(state, rate, epsilon, area):
    h, hd = state.h, rate['hdot']
    J = _density_stock(state.stocks, h)
    Jd = _density_stock(rate['stockdot'], hd)
    means, centers, slopes = _scalar_profiles(state.stocks, h, state.interfaces)
    cd = np.zeros_like(h)
    cd[:, 0] = .5 * rate['etadot']
    minimum = h - epsilon * abs(hd)
    if np.any(minimum <= 0.):
        raise ValueError('finite-difference crossing refused')
    constant = abs(Jd * h - J * hd)
    qd = constant / minimum**2
    qdd = 2. * abs(hd) * constant / minimum**3
    result = RHO0 * math.fsum(float(area[i]) * rate['etadot'][i]**2 for i in range(2))
    for side in range(2):
        secant_bounds = []
        for k in range(h.shape[1] - 1):
            distance = abs(centers[side, k + 1] - centers[side, k])
            dd = abs(cd[side, k + 1] - cd[side, k])
            minimum_distance = distance - epsilon * dd
            if minimum_distance <= 0.:
                raise ValueError('finite-difference center crossing refused')
            nd = qd[side, k] + qd[side, k + 1]
            ndd = qdd[side, k] + qdd[side, k + 1]
            sm = (abs(means[side, k + 1] - means[side, k]) + epsilon * nd) / minimum_distance
            sd = (nd + sm * dd) / minimum_distance
            sdd = (ndd + 2. * sd * dd) / minimum_distance
            secant_bounds.append((sm, sd, sdd))
        for k in range(h.shape[1]):
            adjacent = secant_bounds[max(0, k - 1):min(len(secant_bounds), k + 1)]
            sm, sd, sdd = [max(v[j] for v in adjacent) for j in range(3)]
            hm = h[side, k] + epsilon * abs(hd[side, k])
            second = sdd * hm**3 / 12. + sd * hm**2 * abs(hd[side, k]) / 2. + sm * hm * hd[side, k]**2 / 2.
            result += area[side] * (2. * abs(Jd[side, k] * cd[side, k]) + second)
    return GRAVITY * result


def stock_PE_forward(state, rate, epsilon, area):
    if not math.isfinite(epsilon) or epsilon <= 0.:
        raise ValueError('finite positive one-sided perturbation required')
    z = state.interfaces.copy()
    z[:, 0] += epsilon * rate['etadot']
    h = state.h + epsilon * rate['hdot']
    before, before_scale = _stock_energy(state.stocks, state.h, state.interfaces, area)
    after, after_scale = _stock_energy(state.stocks + epsilon * rate['stockdot'], h, z, area)
    curvature = _curvature_majorant(state, rate, epsilon, area)
    return dict(value=(after - before) / epsilon,
                bound=.5 * epsilon * curvature + 512. * EPS * max(1., before_scale + after_scale) / epsilon,
                truncation_bound=.5 * epsilon * curvature)


def p0_partition_pressure_defect(state, depth):
    means = _density_stock(state.stocks, state.h) / state.h
    values = []
    for side in range(2):
        terms = []
        for k in range(state.h.shape[1]):
            lower = max(depth, float(state.interfaces[side, k + 1]))
            upper = float(state.interfaces[side, k])
            if upper > lower:
                terms.append(float(means[side, k]) * (upper - lower))
        values.append(GRAVITY * math.fsum(terms))
    return values[1] - values[0]
