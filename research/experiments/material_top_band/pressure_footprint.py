"""Complete rectangular diagnostic footprint authority, including periodic images.

Bounds are declared geometry, never inferred from a caller's force receipt.
Partial sides and periodic self-faces need a separate geometry contract.
"""
import math
from dataclasses import dataclass
from itertools import product

import numpy as np

from .real_geometry import _array, _column, _scalar


@dataclass(frozen=True)
class RectangularFootprint:
    bounds_m: np.ndarray  # xlo, xhi, ylo, yhi for every bound column
    area_m2: np.ndarray
    periodic_extent_m: tuple = (0., 0.)
    periodic_origin_m: tuple = (0., 0.)


def _close(a, b):
    a, b = float(a), float(b)
    epsilon = 256. * np.finfo(float).eps
    difference = a - b
    bound = max(epsilon, epsilon * abs(a) + epsilon * abs(b))
    return math.isfinite(difference) and math.isfinite(bound) and abs(difference) <= bound


def _side(normal):
    normal = _array(normal, 'footprint normal')
    if normal.shape != (2,):
        raise ValueError('footprint cardinal normal required')
    for axis, sign in product(range(2), [-1, 1]):
        expected = np.zeros(2)
        expected[axis] = sign
        if np.array_equal(normal, expected):
            return axis, sign
    raise ValueError('footprint cardinal normal required')


def _geometry(state, footprint):
    if not isinstance(footprint, RectangularFootprint):
        raise ValueError('explicit rectangular footprint geometry required for wet qualification')
    boxes = _array(footprint.bounds_m, 'footprint bounds')
    areas = _array(footprint.area_m2, 'footprint area')
    extent = _array(footprint.periodic_extent_m, 'periodic extent')
    origin = _array(footprint.periodic_origin_m, 'periodic origin')
    shape = state.eta.shape
    if boxes.shape != shape + (4,) or areas.shape != shape or extent.shape != (2,) or origin.shape != (2,) or np.any(extent < 0):
        raise ValueError('footprint column/periodic geometry shape mismatch')
    with np.errstate(over='ignore', invalid='ignore'):
        widths = boxes[..., [1, 3]] - boxes[..., [0, 2]]
        geometric_area = np.prod(widths, axis=-1)
        periodic_end = origin + extent
    epsilon = 256. * np.finfo(float).eps
    resolution = np.maximum(epsilon, epsilon * abs(boxes[..., [1, 3]]) + epsilon * abs(boxes[..., [0, 2]]))
    if np.any(widths <= resolution) or not all(np.isfinite(a).all() for a in [widths, geometric_area, periodic_end]) or np.any(areas <= 0):
        raise ValueError('resolved nondegenerate footprint required')
    if any(not _close(float(a), float(b)) for a, b in zip(areas.ravel(), geometric_area.ravel())):
        raise ValueError('footprint area must match physical bounds')
    return boxes, areas, extent, origin, widths, resolution


def bind_footprint_area(state, footprint, consumed_area):
    _, areas, _, _, _, _ = _geometry(state, footprint)
    consumed = _array(consumed_area, 'consumed footprint area')
    if consumed.shape != areas.shape or any(not _close(a, b) for a, b in zip(consumed.ravel(), areas.ravel())):
        raise ValueError('consumed area must match the bound footprint geometry')
    return consumed


def validate_footprint(state, footprint, requests, walls):
    boxes, _, extent, origin, widths, resolution = _geometry(state, footprint)
    shape = state.eta.shape
    columns = list(np.ndindex(shape))
    shifts = list(product(*[[-float(L), 0., float(L)] if L else [0.] for L in extent]))
    for column in columns:
        for axis in range(2):
            if extent[axis] and (boxes[column][2 * axis] < origin[axis] or boxes[column][2 * axis + 1] > origin[axis] + extent[axis]):
                raise ValueError('footprint lies outside its declared periodic domain')
    adjacency = {}
    for column in columns:
        own = boxes[column].reshape(2, 2)
        for other in columns:
            for shift in shifts:
                if column == other and shift == (0., 0.):
                    continue
                with np.errstate(over='ignore', invalid='ignore'):
                    neighbor = boxes[other].reshape(2, 2) + np.array(shift)[:, None]
                if not np.isfinite(neighbor).all():
                    raise ValueError('finite representable periodic footprint image required')
                overlap = np.minimum(own[:, 1], neighbor[:, 1]) - np.maximum(own[:, 0], neighbor[:, 0])
                if all(value > resolution[column][axis] for axis, value in enumerate(overlap)):
                    raise ValueError('overlapping footprint boxes/images')
        for axis, sign in product(range(2), [-1, 1]):
            side_coordinate = own[axis, int(sign > 0)]
            tangent = 1 - axis
            matches = []
            for other in columns:
                for shift in shifts:
                    if column == other and shift == (0., 0.):
                        continue
                    neighbor = boxes[other].reshape(2, 2) + np.array(shift)[:, None]
                    if not _close(side_coordinate, neighbor[axis, int(sign < 0)]):
                        continue
                    overlap = min(own[tangent, 1], neighbor[tangent, 1]) - max(own[tangent, 0], neighbor[tangent, 0])
                    if overlap <= resolution[column][tangent]:
                        continue
                    if not all(_close(a, b) for a, b in zip(own[tangent], neighbor[tangent])):
                        raise ValueError('partial footprint sides are unsupported')
                    if other == column:
                        raise ValueError('periodic self-faces are unsupported')
                    matches.append((other, shift))
            if len(matches) > 1:
                raise ValueError('ambiguous footprint neighbors')
            if not matches and extent[axis] and (_close(side_coordinate, origin[axis]) or _close(side_coordinate, origin[axis] + extent[axis])):
                raise ValueError('periodic footprint side lacks its exact image neighbor')
            adjacency[(column, axis, sign)] = matches[0] if matches else None

    occupied = set()
    def cover(column, axis, sign, length):
        token = (column, axis, sign)
        if token in occupied:
            raise ValueError('duplicate footprint side coverage')
        if not _close(length, float(widths[column][1 - axis])):
            raise ValueError('footprint face length/area mismatch')
        occupied.add(token)

    for request in requests:
        left, right = _column(state, request.left), _column(state, request.right)
        axis, sign = _side(request.normal)
        length = _scalar(request.length, 'footprint face length', positive=True)
        shift = _array(request.periodic_shift_m, 'footprint periodic image')
        expected = adjacency[(left, axis, sign)]
        reverse = adjacency[(right, axis, -sign)]
        if shift.shape != (2,) or expected is None or reverse is None or expected[0] != right or reverse[0] != left or not np.array_equal(shift, expected[1]) or not np.array_equal(-shift, reverse[1]):
            raise ValueError('footprint face column/normal/periodic image mismatch')
        cover(left, axis, sign, length)
        cover(right, axis, -sign, length)
    for wall in walls:
        column = _column(state, wall.column)
        axis, sign = _side(wall.normal)
        length = _scalar(wall.length, 'footprint wall length', positive=True)
        if adjacency[(column, axis, sign)] is not None:
            raise ValueError('known wet/dry adjacency cannot be reclassified as an exterior wall')
        cover(column, axis, sign, length)
    required = {(column, axis, sign) for column in columns if state.active_layers[column]
                for axis, sign in product(range(2), [-1, 1])}
    if not required <= occupied:
        raise ValueError('incomplete footprint: every wet column needs all four unique sides')
    return dict(authority='declared_rectangular_cardinal_footprint', complete=True,
                wet_columns=int(np.count_nonzero(state.active_layers)),
                covered_wet_sides=len(required), nondegenerate=True,
                column_areas_bound=True, periodic_extent_m=extent.tolist(),
                periodic_origin_m=origin.tolist())
