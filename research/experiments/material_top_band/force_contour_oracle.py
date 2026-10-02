"""Scalar seven-point physical contour and full-stock PE directional oracles.

No candidate pressure/PE/C/transpose routine is called. Sigma geometry is
declared input; contour measures are reconstructed from endpoints, not S.
"""
import math

import numpy as np

GAUSS = np.polynomial.legendre.leggauss(7)


def _integral_density(profile, col, sigma):
    state = profile.state
    H = float(state.eta[col] - state.bottom[col])
    lower = float(state.bottom[col] + sigma * H)
    terms = []
    for k in range(int(state.active_layers[col])):
        lo = max(lower, float(state.interfaces[col][k + 1]))
        hi = float(state.interfaces[col][k])
        if hi > lo:
            midpoint = .5 * (float(state.interfaces[col][k]) + float(state.interfaces[col][k + 1]))
            terms.append((hi - lo) * (float(profile.density_mean[col][k])
                                      + float(profile.density_slope[col][k]) * (.5 * (hi + lo) - midpoint)))
    return math.fsum(terms) / H


def _pressure(profile, segment, x, sigma):
    left, right = segment.left, segment.right
    HL = float(profile.state.eta[left] - profile.state.bottom[left])
    HR = float(profile.state.eta[right] - profile.state.bottom[right])
    H = HL + x * (HR - HL)
    external = float(profile.external_pressure_Pa[left]) + x * float(profile.external_pressure_Pa[right] - profile.external_pressure_Pa[left])
    anomaly = (1. - x) * _integral_density(profile, left, sigma) + x * _integral_density(profile, right, sigma)
    return external + profile.eos.gravity * H * (profile.eos.rho0 * (1. - sigma) + anomaly)


def _gauss(function, lower, upper):
    middle, half = .5 * (lower + upper), .5 * (upper - lower)
    return half * math.fsum(float(weight) * function(middle + half * float(node)) for node, weight in zip(*GAUSS))


def contour(profile, segment):
    """Integrate -p*n_x over the four physical trapezoid boundaries."""
    L, lo, hi = float(segment.length_m), segment.sigma_lower, segment.sigma_upper
    HL, HR = [float(profile.state.eta[c] - profile.state.bottom[c]) for c in [segment.left, segment.right]]
    components = [L * HL * _gauss(lambda s: _pressure(profile, segment, 0., s), lo, hi),
                  -L * HR * _gauss(lambda s: _pressure(profile, segment, 1., s), lo, hi),
                  L * (HR - HL) * hi * _gauss(lambda x: _pressure(profile, segment, x, hi), 0., 1.),
                  -L * (HR - HL) * lo * _gauss(lambda x: _pressure(profile, segment, x, lo), 0., 1.)]
    force = math.fsum(components)
    scale = math.fsum(abs(v) for v in components)
    bound = 512. * np.finfo(float).eps * max(1., scale)
    if not all(math.isfinite(v) for v in [force, scale, bound]):
        raise ValueError('finite independent physical contour required')
    return dict(force_N=force, bound_N=bound, absolute_traction_N=scale)


def _minmod_direction(left, right, dleft, dright):
    """Right directional derivative, including zero and equal-magnitude kinks."""
    if left * right > 0.:
        if abs(left) < abs(right):
            return dleft
        if abs(right) < abs(left):
            return dright
        return min(dleft, dright) if left > 0. else max(dleft, dright)
    if left == 0. and right == 0.:
        return math.copysign(min(abs(dleft), abs(dright)), dleft) if dleft * dright > 0. else 0.
    if left == 0.:
        return dleft if dleft * right > 0. else 0.
    if right == 0.:
        return dright if dright * left > 0. else 0.
    return 0.


def _slope_direction(profile, col, densities, density_rate, center_rate, hdot):
    state, eos = profile.state, profile.eos
    n = int(state.active_layers[col])
    if profile.representation == 'p0' or n < 2:
        return [0.] * n, [0.] * n
    centers = [.5 * float(state.interfaces[col][k] + state.interfaces[col][k + 1]) for k in range(n)]
    secants, directions = [], []
    for k in range(n - 1):
        distance = centers[k + 1] - centers[k]
        secant = (densities[k + 1] - densities[k]) / distance
        secants.append(secant)
        directions.append((density_rate[k + 1] - density_rate[k] - secant * (center_rate[k + 1] - center_rate[k])) / distance)
    raw = [secants[0]]
    raw_rate = [directions[0]]
    for k in range(1, n - 1):
        a, b = secants[k - 1:k + 1]
        raw.append(math.copysign(min(abs(a), abs(b)), a) if a * b > 0. else 0.)
        raw_rate.append(_minmod_direction(a, b, directions[k - 1], directions[k]))
    raw.append(secants[-1])
    raw_rate.append(directions[-1])
    lower = eos.rho0 * (-eos.alpha * (45. - eos.Tref) + eos.beta * (0. - eos.Sref))
    upper = eos.rho0 * (-eos.alpha * (-5. - eos.Tref) + eos.beta * (50. - eos.Sref))
    base, result = [], []
    for k, (slope, derivative) in enumerate(zip(raw, raw_rate)):
        mean, dmean, width = densities[k], density_rate[k], float(state.h[col][k])
        low_room, high_room = mean - lower, upper - mean
        raw_room = min(low_room, high_room)
        room = max(0., raw_room)
        droom = dmean if low_room < high_room else -dmean if high_room < low_room else -abs(dmean)
        if raw_room < 0.:
            droom = 0.
        elif raw_room == 0.:
            droom = max(0., droom)
        excursion = .5 * width * abs(slope)
        base.append(slope * min(1., room / excursion) if excursion else slope)
        dexcursion = .5 * (hdot[k] * abs(slope) + width * (math.copysign(1., slope) * derivative if slope else abs(derivative)))
        if excursion < room or (excursion == room and dexcursion <= droom):
            result.append(derivative)
        elif excursion > 0.:
            result.append(math.copysign(1., slope) * 2. * (droom - room * (hdot[k] / width)) / width)
        else:
            # At zero slope and zero room, the directional bound clips ds.
            result.append(math.copysign(min(abs(derivative), 2. * droom / width), derivative))
    return base, result


def stock_pe_direction_ledger(profile, area, hdot, stockdot, etadot):
    """Right directional derivative of rho0 free PE + full density-P1 moment.

    The limiter chain is differentiated independently, including kinks. This
    oracle does not establish a consumable general-P1 momentum/ALE geometry.
    """
    state, eos = profile.state, profile.eos
    if (np.shape(area) != state.eta.shape or np.shape(hdot) != state.h.shape
            or np.shape(stockdot) != state.stocks.shape or np.shape(etadot) != state.eta.shape
            or not all(np.isfinite(a).all() for a in [area, hdot, stockdot, etadot]) or np.any(np.asarray(area) <= 0)):
        raise ValueError('finite shape-matched positive-area stock direction required')
    terms = []
    for col in np.ndindex(state.eta.shape):
        n = int(state.active_layers[col])
        if not n:
            continue
        zrate = float(etadot[col])
        densities, density_rate, center_rate = [], [], []
        terms.append(float(area[col]) * eos.gravity * eos.rho0 * float(state.eta[col]) * zrate)
        for k in range(n):
            width = float(state.h[col][k])
            width_rate = float(hdot[col][k])
            mean_T = float(state.stocks[col][k, 0]) / width
            mean_S = float(state.stocks[col][k, 1]) / width
            dT = (float(stockdot[col][k, 0]) - mean_T * width_rate) / width
            dS = (float(stockdot[col][k, 1]) - mean_S * width_rate) / width
            drho = eos.rho0 * (-eos.alpha * dT + eos.beta * dS)
            anomaly = eos.rho0 * (-eos.alpha * (mean_T - eos.Tref) + eos.beta * (mean_S - eos.Sref))
            densities.append(anomaly)
            center = .5 * float(state.interfaces[col][k] + state.interfaces[col][k + 1])
            dcenter = zrate - .5 * width_rate
            density_rate.append(drho)
            center_rate.append(dcenter)
            factor = float(area[col]) * eos.gravity
            terms.extend([factor * drho * width * center,
                          factor * anomaly * width_rate * center,
                          factor * anomaly * width * dcenter])
            zrate -= width_rate
        slopes, dslope = _slope_direction(profile, col, densities, density_rate, center_rate, hdot[col][:n])
        for k, (slope, slope_rate) in enumerate(zip(slopes, dslope)):
            width, dwidth = float(state.h[col][k]), float(hdot[col][k])
            factor = float(area[col]) * eos.gravity
            terms.extend([factor * slope_rate * width * width * width / 12.,
                          factor * slope * width * width * dwidth / 4.])
    result = math.fsum(terms)
    scale = math.fsum(abs(v) for v in terms)
    bound = 256. * np.finfo(float).eps * max(1., scale)
    if not all(math.isfinite(v) for v in [result, scale, bound]):
        raise ValueError('finite independent stock PE direction required')
    return dict(direction_W=result, absolute_local_contribution_W=scale, bound_W=bound)


def stock_pe_direction(profile, area, hdot, stockdot, etadot):
    return stock_pe_direction_ledger(profile, area, hdot, stockdot, etadot)['direction_W']
