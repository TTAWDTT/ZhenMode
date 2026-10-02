"""Independent seven-point physical integrals and explicit raw-stock curves.

No projection candidate, candidate PE or candidate quadrature is imported.
"""
import math
from dataclasses import replace
from fractions import Fraction

import numpy as np

from . import affine_physical_oracle as physical

EPS = np.finfo(float).eps
RHO0, G, BETA, TREF, SREF = 1025., 9.81, 7.6e-4, 15., 35.


def phi(spec, side, x):
    return 1. - x / spec.distance if side == 0 else x / spec.distance


def integrate(spec, strip, side, function):
    return physical.volume(spec, strip, lambda x, z: phi(spec, side, x) * function(x, z))


def local_specific(spec, state, strip, x, z):
    q = state.stocks[:, :, 2:] / state.h[:, :, None]
    left, right = q[0, strip.left_layer], q[1, strip.right_layer]
    m = left + x / spec.distance * (right - left)
    density = physical.anomaly(spec, x, z)
    return np.r_[1., TREF, SREF + density / (RHO0 * BETA), m, z * density]


def validate_state(spec, state, material=None):
    """Bind declared fields independently to every actual endpoint stock."""
    if not np.isfinite(state.stocks).all() or not np.isfinite(state.h).all() or np.any(state.h <= 0.):
        raise ValueError('finite wet raw stock state required')
    with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
        specific = state.stocks / state.h[:, :, None]
    if not np.isfinite(specific).all():
        raise ValueError('finite derived raw specific stocks required')
    T, S = specific[:, :, 0], specific[:, :, 1]
    if np.any(abs(T - TREF) > 512. * EPS * np.maximum(1., abs(T) + TREF)):
        raise ValueError('actual material temperature must equal Tref')
    actual_density = RHO0 * (-2e-4 * (T - TREF) + BETA * (S - SREF))
    center = .5 * (state.interfaces[:, :-1] + state.interfaces[:, 1:])
    declared = np.array([[physical.anomaly(spec, side * spec.distance, z) for z in center[side]] for side in range(2)])
    density_scale = RHO0 * (2e-4 * (abs(T) + TREF) + BETA * (abs(S) + SREF)) + abs(declared)
    if not all(np.isfinite(field).all() for field in (actual_density, declared, density_scale)):
        raise ValueError('finite actual and declared density binding required')
    if np.any(abs(actual_density - declared) > 512. * EPS * np.maximum(1., density_scale)):
        raise ValueError('actual material density differs from affine specification')
    if material is not None:
        U, strain = material
        if not all(math.isfinite(value) for value in (U, strain)):
            raise ValueError('finite material velocity required')
        actual = specific[:, :, 2:] / RHO0
        expected = np.broadcast_to(np.array([[U, spec.velocity_y], [U + strain * spec.distance, spec.velocity_y]])[:, None, :], actual.shape)
        if not np.isfinite(expected).all():
            raise ValueError('finite declared material velocity required')
        if np.any(abs(actual - expected) > 512. * EPS * np.maximum(1., abs(actual) + abs(expected))):
            raise ValueError('actual momentum differs from declared material affine velocity')


def validate_chart(spec, state, strips):
    physical.validate_strip_ownership(spec, state, strips)
    ordered = sorted(strips, key=lambda strip: strip.lower_left)
    previous = np.asarray(state.bottom)
    for strip in ordered:
        lower = np.array([strip.lower_left, strip.lower_right])
        upper = np.array([strip.upper_left, strip.upper_right])
        tolerance = 512. * EPS * np.maximum(1., abs(previous) + abs(lower) + abs(upper))
        if np.any(abs(lower - previous) > tolerance):
            raise ValueError('physical chart pairing does not form a full-domain partition')
        previous = upper
    if (not ordered or sum(strip.actual_top for strip in ordered) != 1 or not ordered[-1].actual_top
            or np.any(abs(previous - state.eta) > 512. * EPS * np.maximum(1., abs(previous) + abs(state.eta)))):
        raise ValueError('physical chart partition endpoints or unique actual top disagree')
    # A width sum can conceal equal overlap and gap. Resolve each interval
    # adjacency independently, in addition to the inherited raw-owner check.
    for side in range(2):
        for layer in range(14):
            intervals = sorted(((strip.lower_left, strip.upper_left) if side == 0 else (strip.lower_right, strip.upper_right))
                               for strip in strips if (strip.left_layer, strip.right_layer)[side] == layer)
            current = state.interfaces[side, layer + 1]
            for lower, upper in intervals:
                tolerance = 512. * EPS * max(1., abs(current), abs(lower), abs(upper))
                if abs(lower - current) > tolerance:
                    raise ValueError('raw physical coverage has an overlap or gap')
                current = upper
            if abs(current - state.interfaces[side, layer]) > 512. * EPS * max(1., abs(current), abs(state.interfaces[side, layer])):
                raise ValueError('raw physical coverage endpoint mismatch')


def mapped(spec, strips, state):
    validate_state(spec, state)
    validate_chart(spec, state, strips)
    values, scale, mass = np.zeros((2, 14, 6)), np.zeros((2, 14, 6)), np.zeros((28, 28))
    for strip in strips:
        owners = (strip.left_layer, strip.right_layer)
        for side, layer in enumerate(owners):
            for slot in range(6):
                def field(x, z):
                    return local_specific(spec, state, strip, x, z)[slot]
                values[side, layer, slot] += integrate(spec, strip, side, field)
                scale[side, layer, slot] += integrate(spec, strip, side, lambda x, z: abs(field(x, z)))
            def eos_operations(x, z):
                S = SREF + physical.anomaly(spec, x, z) / (RHO0 * BETA)
                return abs(z) * RHO0 * (2e-4 * 30. + BETA * (abs(S) + SREF))
            scale[side, layer, 5] += integrate(spec, strip, side, eos_operations)
            for other, other_layer in enumerate(owners):
                mass[14 * side + layer, 14 * other + other_layer] += RHO0 * integrate(spec, strip, side, lambda x, z: phi(spec, other, x))
    free = G * RHO0 * spec.length * physical.gauss(lambda x: physical.eta(spec, x)**2 / 2., 0., spec.distance)
    free_scale = G * RHO0 * spec.length * physical.gauss(lambda x: abs(physical.eta(spec, x)**2 / 2.), 0., spec.distance)
    return dict(values=values, scale=scale, mass=mass, PE=free + G * math.fsum(values[:, :, 5].ravel()),
                PE_scale=free_scale + G * math.fsum(scale[:, :, 5].ravel()), free_PE=free)


def raw_inventory(state, distance=2., length=1.5):
    area = distance * length / 2.
    arrays = [state.h, *(state.stocks[:, :, j] for j in range(4))]
    return np.array([area * math.fsum(float(v) for v in field.ravel()) for field in arrays])


def material_specific(spec, x, z, U, strain):
    return np.r_[1., physical.specific(spec, x, z, U, strain), z * physical.anomaly(spec, x, z)]


def material_specific_dot(spec, x, z, U, strain):
    density_dot = physical.density_direction(spec, x, z, U, strain)
    return np.array([0., 0., density_dot / (RHO0 * BETA),
                     -RHO0 * strain * (U + strain * x), 0., z * density_dot])


def direction(spec, strips, state, U, strain):
    validate_state(spec, state, (U, strain))
    validate_chart(spec, state, strips)
    values, scale = np.zeros((2, 14, 6)), np.zeros((2, 14, 6))
    for strip in strips:
        for side, layer in enumerate((strip.left_layer, strip.right_layer)):
            for slot in range(6):
                interior = integrate(spec, strip, side, lambda x, z: material_specific_dot(spec, x, z, U, strain)[slot])
                shape = 0.
                if strip.actual_top:
                    shape = spec.length * physical.gauss(lambda x: phi(spec, side, x) * material_specific(spec, x, physical.eta(spec, x), U, strain)[slot] * physical.surface_direction(spec, x, U, strain), 0., spec.distance)
                values[side, layer, slot] += interior + shape
                interior_scale = integrate(spec, strip, side, lambda x, z: abs(material_specific_dot(spec, x, z, U, strain)[slot]))
                shape_scale = spec.length * physical.gauss(lambda x: abs(phi(spec, side, x) * material_specific(spec, x, physical.eta(spec, x), U, strain)[slot] * physical.surface_direction(spec, x, U, strain)), 0., spec.distance) if strip.actual_top else 0.
                scale[side, layer, slot] += interior_scale + shape_scale
    return dict(values=values, scale=scale)


def weighted_balances(spec, strips, state, U, strain):
    validate_state(spec, state, (U, strain))
    validate_chart(spec, state, strips)
    rows = []
    for strip in strips:
        for side, layer in enumerate((strip.left_layer, strip.right_layer)):
            content, shape_terms, fluxes, source, basis_source, scales = [], [], [], [], [], []
            edges = np.zeros((4, 6))
            derivative = (-1. if side == 0 else 1.) / spec.distance
            for slot in range(6):
                def q(x, z):
                    return material_specific(spec, x, z, U, strain)[slot]
                volume_dot = integrate(spec, strip, side, lambda x, z: material_specific_dot(spec, x, z, U, strain)[slot])
                volume_scale = integrate(spec, strip, side, lambda x, z: abs(material_specific_dot(spec, x, z, U, strain)[slot]))
                shape = spec.length * physical.gauss(lambda x: phi(spec, side, x) * q(x, physical.eta(spec, x)) * physical.surface_direction(spec, x, U, strain), 0., spec.distance) if strip.actual_top else 0.
                shape_scale = spec.length * physical.gauss(lambda x: abs(phi(spec, side, x) * q(x, physical.eta(spec, x)) * physical.surface_direction(spec, x, U, strain)), 0., spec.distance) if strip.actual_top else 0.
                edge_scales = []
                for edge, x, normal, lower, upper in ((0, 0., -1., strip.lower_left, strip.upper_left),
                                                     (1, spec.distance, 1., strip.lower_right, strip.upper_right)):
                    edges[edge, slot] = normal * spec.length * physical.gauss(lambda z: phi(spec, side, x) * q(x, z) * (U + strain * x), lower, upper)
                    edge_scales.append(spec.length * physical.gauss(lambda z: abs(phi(spec, side, x) * q(x, z) * (U + strain * x)), lower, upper))
                for edge, top in ((2, True), (3, False)):
                    slope = ((strip.upper_right - strip.upper_left) if top else (strip.lower_right - strip.lower_left)) / spec.distance
                    def edge_flux(x):
                        z = physical.strip_edges(spec, strip, x)[1 if top else 0]
                        u, w = physical.flow(spec, x, z, U, strain)
                        grid = physical.surface_direction(spec, x, U, strain) if top and strip.actual_top else 0.
                        return (1. if top else -1.) * phi(spec, side, x) * q(x, z) * (w - u * slope - grid)
                    edges[edge, slot] = spec.length * physical.gauss(edge_flux, 0., spec.distance)
                    edge_scales.append(spec.length * physical.gauss(lambda x: abs(edge_flux(x)), 0., spec.distance))
                weighted = physical.volume(spec, strip, lambda x, z: q(x, z) * (U + strain * x) * derivative)
                weighted_scale = physical.volume(spec, strip, lambda x, z: abs(q(x, z) * (U + strain * x) * derivative))
                vertical = integrate(spec, strip, side, lambda x, z: physical.anomaly(spec, x, z) * physical.flow(spec, x, z, U, strain)[1]) if slot == 5 else 0.
                vertical_scale = integrate(spec, strip, side, lambda x, z: abs(physical.anomaly(spec, x, z) * physical.flow(spec, x, z, U, strain)[1])) if slot == 5 else 0.
                content.append(volume_dot + shape)
                shape_terms.append(shape)
                fluxes.append(math.fsum(edges[:, slot]))
                basis_source.append(weighted)
                source.append(weighted + vertical)
                scales.append(volume_scale + shape_scale + math.fsum(edge_scales) + weighted_scale + vertical_scale)
            rows.append(dict(side=side, layer=layer, content_dot=np.array(content), shape=np.array(shape_terms),
                             flux=np.array(fluxes), source=np.array(source), basis_source=np.array(basis_source),
                             scale=np.array(scales), top_flux=edges[2].copy()))
    return tuple(rows)


def curve(spec, state, U, strain, epsilon):
    if not math.isfinite(epsilon) or epsilon <= 0.:
        raise ValueError('finite positive curve parameter required')
    e = [physical.surface_direction(spec, x, U, strain) for x in (0., spec.distance)]
    changed = replace(spec, eta=tuple(spec.eta[j] + epsilon * e[j] for j in range(2)),
                      density_intercept=spec.density_intercept + epsilon * (-U * spec.density_gradient_x - strain * spec.density_slope * spec.bottom),
                      density_gradient_x=spec.density_gradient_x * (1. - epsilon * strain),
                      density_slope=spec.density_slope * (1. + epsilon * strain))
    z = state.interfaces.copy()
    z[:, 0] = changed.eta
    h = -np.diff(z, axis=-1)
    stocks = np.zeros_like(state.stocks)
    actual = state.stocks[:, :, 2:] / (RHO0 * state.h[:, :, None])
    for side, x in enumerate((0., spec.distance)):
        for layer in range(14):
            center = .5 * (z[side, layer] + z[side, layer + 1])
            S = SREF + physical.anomaly(changed, x, center) / (RHO0 * BETA)
            stocks[side, layer] = h[side, layer] * np.array([TREF, S, RHO0 * actual[side, layer, 0] * (1. - epsilon * strain), RHO0 * actual[side, layer, 1]])
    return replace(state, eta=np.array(changed.eta), interfaces=z, h=h, stocks=stocks), changed


def fixed_chart(strips, eta):
    return tuple(replace(strip, upper_left=eta[0], upper_right=eta[1]) if strip.actual_top else strip for strip in strips)


def polynomial_add(*polynomials):
    result = [Fraction(0)] * max(map(len, polynomials))
    for polynomial in polynomials:
        for degree, coefficient in enumerate(polynomial):
            result[degree] += coefficient
    while len(result) > 1 and result[-1] == 0:
        result.pop()
    return tuple(result)


def polynomial_scale(polynomial, factor):
    return tuple(factor * coefficient for coefficient in polynomial)


def polynomial_product(left, right):
    result = [Fraction(0)] * (len(left) + len(right) - 1)
    for i, coefficient in enumerate(left):
        for j, other in enumerate(right):
            result[i + j] += coefficient * other
    return polynomial_add(result)


def polynomial_integral(polynomial, distance):
    return sum((coefficient * distance**(degree + 1) / (degree + 1)
                for degree, coefficient in enumerate(polynomial)), Fraction(0))


def certified_absolute_integral(polynomial, side, distance):
    """Exact binary-input Bernstein certificate, then exact weighted integral.

    A mixed Bernstein hull refuses; no Gauss sampling or sign tolerance is
    used to turn a sign-changing absolute value into a polynomial integral.
    """
    polynomial = polynomial_add(polynomial)
    degree = len(polynomial) - 1
    power = [coefficient * distance**k for k, coefficient in enumerate(polynomial)]
    bernstein = [sum((power[k] * Fraction(math.comb(i, k), math.comb(degree, k))
                     for k in range(i + 1)), Fraction(0)) for i in range(degree + 1)]
    if not (all(value >= 0 for value in bernstein) or all(value <= 0 for value in bernstein)):
        raise ValueError('absolute remainder polynomial sign is not certified')
    basis = (Fraction(1), -1 / distance) if side == 0 else (Fraction(0), 1 / distance)
    return abs(polynomial_integral(polynomial_product(basis, polynomial), distance))


def remainder_polynomials(spec, U, strain):
    """Construct the frozen C2/C3/C4 expressions with exact rational algebra."""
    d, bottom, alpha, u0 = map(Fraction, (spec.distance, spec.bottom, strain, U))
    slope, gradient = map(Fraction, (spec.density_slope, spec.density_gradient_x))
    z = (Fraction(spec.eta[0]), (Fraction(spec.eta[1]) - Fraction(spec.eta[0])) / d)
    velocity = (u0, alpha)
    e = polynomial_add(polynomial_scale(polynomial_add(z, (-bottom,)), -alpha),
                       polynomial_scale(velocity, -z[1]))
    a = (Fraction(spec.density_intercept), gradient)
    da = polynomial_add(polynomial_scale(velocity, -gradient), (-alpha * slope * bottom,))
    ds = alpha * slope
    z2, e2 = polynomial_product(z, z), polynomial_product(e, e)
    e3 = polynomial_product(e2, e)
    zero = (Fraction(0),)
    rows = [(zero, zero, zero) for _ in range(6)]
    factor = Fraction(RHO0) * Fraction(BETA)
    q1 = polynomial_scale(polynomial_add(da, polynomial_scale(z, ds)), 1 / factor)
    rows[2] = (polynomial_add(polynomial_product(q1, e), polynomial_scale(e2, slope / (2 * factor))),
               polynomial_scale(e2, ds / (2 * factor)), zero)
    rows[3] = (polynomial_product(polynomial_scale(velocity, -Fraction(RHO0) * alpha), e), zero, zero)
    rows[5] = (
        polynomial_add(polynomial_product(polynomial_add(polynomial_product(da, z), polynomial_scale(z2, ds)), e),
                       polynomial_scale(polynomial_product(polynomial_add(a, polynomial_scale(z, 2 * slope)), e2), Fraction(1, 2))),
        polynomial_add(polynomial_scale(polynomial_product(polynomial_add(da, polynomial_scale(z, 2 * ds)), e2), Fraction(1, 2)),
                       polynomial_scale(e3, slope / 3)), polynomial_scale(e3, ds / 3))
    return rows, e2


def upward(value):
    return math.nextafter(float(value), math.inf) if value else 0.


def forward_envelope(spec, strips, state, U, strain, epsilon):
    validate_state(spec, state, (U, strain))
    changed_state, changed = curve(spec, state, U, strain, epsilon)
    before, after = mapped(spec, strips, state), mapped(changed, fixed_chart(strips, changed.eta), changed_state)
    coefficients, e2 = remainder_polynomials(spec, U, strain)
    d, length, eps = map(Fraction, (spec.distance, spec.length, epsilon))
    exact = np.full((2, 14, 6), Fraction(0), dtype=object)
    for strip in strips:
        if strip.actual_top:
            for side, layer in enumerate((strip.left_layer, strip.right_layer)):
                for slot, polynomials in enumerate(coefficients):
                    exact[side, layer, slot] += length * sum(
                        (eps**(j + 1) * certified_absolute_integral(polynomial, side, d)
                         for j, polynomial in enumerate(polynomials)), Fraction(0))
    truncation = np.array([upward(value) for value in exact.ravel()]).reshape(exact.shape)
    rounding = 512. * EPS * np.maximum(1., before['scale'] + after['scale']) / epsilon
    free_exact = eps * Fraction(G) * Fraction(RHO0) * length * polynomial_integral(e2, d) / 2
    PE_exact = Fraction(G) * sum(exact[:, :, 5].ravel(), Fraction(0)) + free_exact
    PE_rounding = 512. * EPS * max(1., before['PE_scale'] + after['PE_scale']) / epsilon
    return dict(values_bound=np.nextafter(truncation + rounding, np.inf),
                PE_bound=math.nextafter(upward(PE_exact) + PE_rounding, math.inf),
                PE_truncation_bound=upward(PE_exact), free_PE_truncation_bound=upward(free_exact))


def static_momentum_defect(spec, strips, state):
    q = state.stocks[:, :, 2:] / state.h[:, :, None]
    terms = []
    for strip in strips:
        left_h, right_h = strip.upper_left - strip.lower_left, strip.upper_right - strip.lower_right
        terms.append((right_h - left_h) * (q[1, strip.right_layer] - q[0, strip.left_layer]))
    return spec.length * spec.distance / 6. * np.array([math.fsum(float(v[j]) for v in terms) for j in range(2)])


def analytic_defects(spec, U, strain):
    k = (spec.eta[1] - spec.eta[0]) / spec.distance
    salinity = spec.length * spec.distance**3 * (spec.density_gradient_x * k + .5 * spec.density_slope * k**2) / (6. * RHO0 * BETA)
    momentum = RHO0 * spec.length * spec.distance**3 * strain * k / 6.
    return np.array([0., 0., salinity, momentum, 0.])


def kinetic(spec, strips, velocity):
    velocity = np.asarray(velocity)
    if velocity.shape == (28,):
        velocity = velocity[:, None]
    terms = []
    for strip in strips:
        left, right = velocity[strip.left_layer], velocity[14 + strip.right_layer]
        terms.append(.5 * RHO0 * physical.volume(spec, strip, lambda x, z: float(np.sum((left + x / spec.distance * (right - left))**2))))
    return dict(value=math.fsum(terms), scale=math.fsum(abs(v) for v in terms))
