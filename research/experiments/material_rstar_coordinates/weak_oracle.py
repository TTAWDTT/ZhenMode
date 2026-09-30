"""NumPy-only face-by-face Gauss weak-form oracle for small controls."""
import numpy as np


def _hat(nodes, count, position):
    weights, derivative = np.zeros(nodes.size), np.zeros(nodes.size)
    if count == 1 or position >= nodes[count - 1]:
        weights[count - 1] = 1.
    else:
        level = min(max(int(np.searchsorted(nodes[:count], position, side="right")) - 1, 0), count - 2)
        spacing = nodes[level + 1] - nodes[level]
        fraction = (position - nodes[level]) / spacing
        weights[level:level + 2] = 1. - fraction, fraction
        derivative[level:level + 2] = -1. / spacing, 1. / spacing
    return weights, derivative


def physical_weak_rates(density, velocities, surface, geometry, params):
    wet = np.asarray(params.wet_mask_z) > 0.
    field = np.where(wet, np.asarray(density), 0.)
    velocity = tuple(np.where(wet, np.asarray(value), 0.) for value in velocities)
    nodes = np.asarray(geometry.node_depth)
    bed = np.asarray(geometry.interface_depth)[..., -1]
    eta = np.where(wet.any(axis=-1), np.asarray(surface), 0.)
    area = np.asarray(params.dx_2d) * float(params.dy)
    content, scale = np.zeros(wet.shape), np.zeros(wet.shape)
    volume, volume_scale = np.zeros(wet.shape[:2]), np.zeros(wet.shape[:2])
    gauss, weights = np.polynomial.legendre.leggauss(5)
    for column in np.ndindex(wet.shape[:2]):
        for axis in (0, 1):
            neighbor = list(column)
            neighbor[axis] += 1
            if axis == 0:
                neighbor[axis] %= wet.shape[axis]
            elif neighbor[axis] == wet.shape[axis]:
                continue
            neighbor = tuple(neighbor)
            count, other_count = int(wet[column].sum()), int(wet[neighbor].sum())
            if not (count and other_count):
                continue
            top, other_top = -eta[column], -eta[neighbor]
            face_top, face_bed = max(top, other_top), min(bed[column], bed[neighbor])
            measure = float(params.dy)
            if axis == 1:
                cosine = np.asarray(params.cos_lat)
                measure = np.asarray(params.dx_2d)[column] / cosine[column[1]] * .5 * (cosine[column[1]] + cosine[neighbor[1]])
            knots = np.unique(np.concatenate((nodes[column][:count], nodes[neighbor][:other_count],
                                              [top, other_top, bed[column], bed[neighbor]])))

            def face_flux(position):
                left_hat = _hat(nodes[column], count, position)[0]
                right_hat = _hat(nodes[neighbor], other_count, position)[0]
                return .5 * measure * (left_hat @ velocity[axis][column] + right_hat @ velocity[axis][neighbor])

            intervals, total = [], 0.
            for low, high in zip(knots[:-1], knots[1:], strict=True):
                aperture = face_top <= .5 * (low + high) <= face_bed
                low_flux, high_flux = (face_flux(low), face_flux(high)) if aperture else (0., 0.)
                amount = .5 * (high - low) * (low_flux + high_flux)
                intervals.append((low, high, low_flux, high_flux, total))
                total += amount
            volume[column] -= total
            volume[neighbor] += total
            volume_scale[column] += abs(total)
            volume_scale[neighbor] += abs(total)
            for low, high, low_flux, high_flux, prefix in intervals:
                positions = .5 * ((high - low) * gauss + low + high)
                measures = .5 * (high - low) * weights
                slope = (high_flux - low_flux) / (high - low)
                for position, quadrature in zip(positions, measures, strict=True):
                    left_hat, left_derivative = _hat(nodes[column], count, position)
                    right_hat, right_derivative = _hat(nodes[neighbor], other_count, position)
                    left_density, right_density = left_hat @ field[column], right_hat @ field[neighbor]
                    displacement = position - low
                    flux = low_flux + slope * displacement
                    partial = prefix + low_flux * displacement + .5 * slope * displacement ** 2
                    horizontal = quadrature * flux * .5 * (left_density + right_density)
                    left_terms, right_terms = -horizontal * left_hat, horizontal * right_hat
                    left_absolute, right_absolute = abs(horizontal) * abs(left_hat), abs(horizontal) * abs(right_hat)
                    if top <= position <= bed[column]:
                        primitive = partial - total * (position - top) / (bed[column] - top)
                        vertical = -quadrature * left_density * primitive * left_derivative
                        left_terms += vertical
                        left_absolute += abs(vertical)
                    if other_top <= position <= bed[neighbor]:
                        primitive = partial - total * (position - other_top) / (bed[neighbor] - other_top)
                        vertical = quadrature * right_density * primitive * right_derivative
                        right_terms += vertical
                        right_absolute += abs(vertical)
                    content[column] += left_terms
                    content[neighbor] += right_terms
                    scale[column] += left_absolute
                    scale[neighbor] += right_absolute
    return content, volume / area, scale, volume_scale / area
