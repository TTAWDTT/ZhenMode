"""Independent scalar polynomial/Gauss oracle; no production pressure calls."""

import math

import numpy as np

NODES, WEIGHTS = np.polynomial.legendre.leggauss(7)


def pressure(profile, column, depth, *, reduced=False):
    eta = float(profile.state.eta[column])
    terms = []
    n = int(profile.state.active_layers[column])
    for k in range(n):
        top = float(profile.state.interfaces[column][k])
        base = max(float(depth), float(profile.state.interfaces[column][k + 1]))
        if top <= base:
            continue
        center = .5 * float(profile.state.interfaces[column][k] + profile.state.interfaces[column][k + 1])
        intercept = float(profile.density_mean[column][k]) - float(profile.density_slope[column][k]) * center
        slope = float(profile.density_slope[column][k])
        terms.append(intercept * (top - base) + .5 * slope * (top**2 - base**2))
    rho_base = profile.eos.rho0 * (eta if reduced else eta - depth)
    return float(profile.external_pressure_Pa[column]) + profile.eos.gravity * (rho_base + math.fsum(terms))


def pressure_integral(profile, column, lower, upper, *, reduced=False):
    n = int(profile.state.active_layers[column])
    edges = profile.state.interfaces[column][:n + 1]
    cuts = sorted(set([float(lower), float(upper), *[float(z) for z in edges if lower < z < upper]]))
    terms = []
    for lo, hi in zip(cuts[:-1], cuts[1:]):
        mid, half = .5 * (hi + lo), .5 * (hi - lo)
        terms.append(half * math.fsum(float(w) * pressure(profile, column, mid + half * float(x), reduced=reduced)
                                     for x, w in zip(NODES, WEIGHTS)))
    return math.fsum(terms)


def potential_energy(profile, area):
    terms = []
    for column in np.ndindex(profile.state.eta.shape):
        n = int(profile.state.active_layers[column])
        if not n:
            continue
        gravity_moments = []
        for k in range(n):
            top, base = (float(profile.state.interfaces[column][i]) for i in [k, k + 1])
            mid, half = .5 * (top + base), .5 * (top - base)
            mean, slope = float(profile.density_mean[column][k]), float(profile.density_slope[column][k])
            gravity_moments.append(half * math.fsum(float(w) * (mean + slope * (half * float(x))) * (mid + half * float(x))
                                                    for x, w in zip(NODES, WEIGHTS)))
        eta = float(profile.state.eta[column])
        terms.append(float(area[column]) * profile.eos.gravity * (.5 * profile.eos.rho0 * eta**2 + math.fsum(gravity_moments)))
    return math.fsum(terms)


def potential_energy_difference(before, after, area):
    """Independent seven-point integration of rho_after(z)-rho_before(z)."""
    terms = []
    for column in np.ndindex(before.state.eta.shape):
        if not before.state.active_layers[column]:
            continue
        edges = [profile.state.interfaces[column][:int(profile.state.active_layers[column]) + 1]
                 for profile in [before, after]]
        cuts = sorted(set(float(z) for column_edges in edges for z in column_edges))
        for lo, hi in zip(cuts[:-1], cuts[1:]):
            mid, half = .5 * (hi + lo), .5 * (hi - lo)
            samples = []
            for node, weight in zip(NODES, WEIGHTS):
                depth = mid + half * float(node)
                densities = []
                for profile in [before, after]:
                    for k in range(int(profile.state.active_layers[column])):
                        top, base = profile.state.interfaces[column][k:k + 2]
                        if base <= depth <= top:
                            center = .5 * (top + base)
                            densities.append(float(profile.density_mean[column][k])
                                             + float(profile.density_slope[column][k]) * (depth - center))
                            break
                samples.append(float(weight) * (densities[1] - densities[0]) * depth)
            terms.append(float(area[column]) * before.eos.gravity * half * math.fsum(samples))
    return math.fsum(terms)


def boundary_forces(profile, requests, outer_walls):
    """Separate assembly from profile intervals, not negative C impulse."""
    terms = {name: [[], []] for name in ['solid', 'external', 'cap']}

    def add(kind, column, normal, length, lo, hi):
        if lo >= hi:
            return
        external = float(profile.external_pressure_Pa[column]) * (hi - lo)
        integral = external if kind == 'external' else pressure_integral(profile, column, lo, hi)
        if kind == 'cap':
            integral -= external
        for axis in range(2):
            terms[kind][axis].append(-float(normal[axis]) * float(length) * integral)

    for wall in outer_walls:
        if profile.state.active_layers[wall.column]:
            add('solid', wall.column, wall.normal, wall.length, profile.state.bottom[wall.column], profile.state.eta[wall.column])
    for request in requests:
        left, right = request.left, request.right
        nl, nr = profile.state.active_layers[left], profile.state.active_layers[right]
        if not nl and not nr:
            continue
        if not nl or not nr:
            column, normal = (left, request.normal) if nl else (right, tuple(-v for v in request.normal))
            add('solid', column, normal, request.length, profile.state.bottom[column], profile.state.eta[column])
            continue
        shared_bottom = max(profile.state.bottom[left], profile.state.bottom[right])
        shared_surface = min(profile.state.eta[left], profile.state.eta[right])
        for column, normal in [(left, request.normal), (right, tuple(-v for v in request.normal))]:
            add('solid', column, normal, request.length, profile.state.bottom[column], shared_bottom)
            add('external', column, normal, request.length, shared_surface, profile.state.eta[column])
            add('cap', column, normal, request.length, shared_surface, profile.state.eta[column])
    result = {name: [math.fsum(axis) for axis in values] for name, values in terms.items()}
    result['physical'] = [result['solid'][axis] + result['external'][axis] for axis in range(2)]
    return result
