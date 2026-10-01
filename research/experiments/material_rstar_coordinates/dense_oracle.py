"""Independent NumPy/Lagrange assembly for small moving-coordinate controls."""
from typing import NamedTuple

import numpy as np


class DenseDiffusion(NamedTuple):
    wet: np.ndarray
    thickness: np.ndarray
    area: np.ndarray
    mass: np.ndarray
    stiffness: np.ndarray
    horizontal_stiffness: np.ndarray
    old_divergence_stiffness: np.ndarray


def assemble_diffusion(eta, reference_depth, reference_width, wet, dx, dy, cosine,
                       kappa_h, kappa_v):
    """Assemble open-face matrices without calling any JAX metric kernel."""
    active = np.asarray(wet) > 0.
    widths = np.broadcast_to(reference_width, active.shape) * active
    depth = widths.sum(axis=-1)
    scale = 1. + np.where(depth > 0., eta, 0.) / np.where(depth > 0., depth, 1.)
    thickness = widths * scale[..., None]
    nodes = -np.asarray(eta)[..., None] + scale[..., None] * reference_depth
    area = np.asarray(dx) * dy
    if np.any(thickness[active] <= 0.):
        raise ValueError("dense audit needs positive wet masses")
    indices = np.full(active.shape, -1, dtype=int)
    count = int(active.sum())
    indices[active] = np.arange(count)
    derivative = np.zeros(active.shape + (count,))
    for column in np.ndindex(active.shape[:2]):
        wet_count = int(active[column].sum())
        if wet_count == 1 or np.any(np.diff(active[column].astype(int)) > 0):
            raise ValueError("dense audit needs contiguous columns with at least two nodes")
        for level in range(wet_count):
            evaluation = nodes[column + (level,)]
            if wet_count == 2:
                selected = np.arange(2)
                coefficients = np.array([-1., 1.]) / np.diff(nodes[column][:2])[0]
            else:
                start = min(max(level - 1, 0), wet_count - 3)
                selected = np.arange(start, start + 3)
                points = nodes[column][selected]
                coefficients = np.empty(3)
                for position in range(3):
                    others = np.delete(points, position)
                    coefficients[position] = ((2. * evaluation - others.sum())
                                               / np.prod(points[position] - others))
            row = derivative[column + (level,)]
            row[indices[column][selected]] = coefficients
    coordinate_rows, physical_rows, horizontal_weights = [], [], []
    vertical_rows, vertical_weights = [], []
    horizontal = np.broadcast_to(kappa_h, active.shape)
    vertical = np.broadcast_to(kappa_v, active.shape)
    longitude_count, latitude_count, _ = active.shape
    for cell in zip(*np.nonzero(active), strict=True):
        longitude, latitude, level = cell
        for axis in (0, 1):
            if axis == 1 and latitude == latitude_count - 1:
                continue
            neighbor = ((longitude + 1) % longitude_count, latitude, level) if axis == 0 else (longitude, latitude + 1, level)
            if not active[neighbor]:
                continue
            spacing = dx[longitude, latitude] if axis == 0 else dy
            row = np.zeros(count)
            row[indices[neighbor]] = 1. / spacing
            row[indices[cell]] = -1. / spacing
            slope = (nodes[neighbor] - nodes[cell]) / spacing
            corrected = row - .5 * slope * (derivative[cell] + derivative[neighbor])
            measure = area[longitude, latitude]
            if axis == 1:
                measure *= .5 * (cosine[latitude] + cosine[latitude + 1]) / cosine[latitude]
            weight = (measure * .5 * (thickness[cell] + thickness[neighbor])
                      * .5 * (horizontal[cell] + horizontal[neighbor]))
            coordinate_rows.append(row)
            physical_rows.append(corrected)
            horizontal_weights.append(weight)
        lower = (longitude, latitude, level + 1)
        if level + 1 < active.shape[-1] and active[lower]:
            spacing = nodes[lower] - nodes[cell]
            row = np.zeros(count)
            row[indices[lower]] = 1. / spacing
            row[indices[cell]] = -1. / spacing
            vertical_rows.append(row)
            vertical_weights.append(area[longitude, latitude] * spacing
                                    * .5 * (vertical[cell] + vertical[lower]))
    coordinate = np.asarray(coordinate_rows).reshape(-1, count)
    physical = np.asarray(physical_rows).reshape(-1, count)
    vertical_matrix = np.asarray(vertical_rows).reshape(-1, count)
    horizontal_measure = np.asarray(horizontal_weights)[:, None]
    vertical_measure = np.asarray(vertical_weights)[:, None]
    horizontal_stiffness = -physical.T @ (horizontal_measure * physical)
    stiffness = horizontal_stiffness - vertical_matrix.T @ (vertical_measure * vertical_matrix)
    old_divergence = -coordinate.T @ (horizontal_measure * physical)
    wet_area = np.broadcast_to(area[..., None], active.shape)[active]
    return DenseDiffusion(active, thickness, wet_area, wet_area * thickness[active],
                          stiffness, horizontal_stiffness, old_divergence)
